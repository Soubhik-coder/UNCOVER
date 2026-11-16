#!/usr/bin/env python3
"""
MethodAAdaptive.py — Technique 2: adaptive-depth Merkle hashing.

Motivation
----------
Plain Merkle at a fixed depth d has a fundamental dilemma when the
locked variants have different depths:

  * If d is too small, deep variants collide with shallow ones (their
    cones get truncated before reaching their KEY leaves).
  * If d is too large, shallow variants leak upstream into the original
    logic that drives their A/B inputs, and instances of the same
    variant get different hashes depending on what's plugged in.

Solution: pick d adaptively, per cell. For each cell c, walk upstream
BFS hop by hop until one of:

  * Stop condition A: a hop introduces a CELL whose fan-in does NOT
    reach a KEY leaf — this is the variant's data boundary. Use the
    PREVIOUS hop's depth.
  * Stop condition B: the cone is exhausted (every frontier reaches
    KEY/PI/CONST leaves with no cell continuations). Use the current
    hop's depth.
  * Hard cap: hop count reaches --max-depth.

At the chosen depth, compute the Merkle hash using the same
commutativity-invariant scheme as MethodAStruct.py:

    H(n, 0) = "TRUNC"  (cells)  or  leaf token  (PI / KEY / CONST)
    H(n, d) = sha1(cell_type || sorted([H(p, d-1) for p in predecessors]))

The cluster id is "d<DEPTH>:<HASH>" so cells at different depths can
never accidentally collide.

Caveats
-------
Y-cuts: when the Y output of one variant drives the data input of
another variant, the BFS sees a key-reaching successor cell and keeps
walking, which conflates the two variants. Same caveat as the
KEY-region method. At low locking percentages this is rare.

Output
------
    <out-dir>/<stem>.cluster.parquet
        node, cell_type, cone_size_d3, reaches_key,
        adaptive_depth   (int, informational; NOT a cluster id)
        adaptive_h       (str, the cluster id)
    <out-dir>/summary.parquet           (one row per graph)
    <out-dir>/summary.csv

Dependencies:
    pip install pandas networkx tqdm pyarrow
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------
# Primitives shared with the other Method-A scripts
# ---------------------------------------------------------------------

def _h(s: str) -> str:
    """Generates a short, stable hex digest for the given string."""
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _leaf_token(g: nx.DiGraph, n: str) -> str | None:
    """
    Returns a standardized token string for non-cell leaf nodes.

    Args:
        g (nx.DiGraph): The networkx directed graph representing the netlist.
        n (str): The node ID.

    Returns:
        str | None: A string token for 'PI', 'KEY', 'CONST', or 'PO', or None if it's a cell.
    """
    kind = g.nodes[n]["kind"]
    if kind == "PI":    return "PI"
    if kind == "KEY":   return "KEY"
    if kind == "CONST": return f"CONST:{g.nodes[n]['port']}"
    if kind == "PO":    return "PO"
    return None


def compute_reaches_key(g: nx.DiGraph) -> dict[str, bool]:
    """
    Computes whether each node has a KEY leaf anywhere in its upstream fan-in cone.

    Args:
        g (nx.DiGraph): The networkx directed graph.

    Returns:
        dict[str, bool]: A mapping of node ID to a boolean (True if it reaches a KEY).
    """
    rk = {}
    for n in nx.topological_sort(g):
        kind = g.nodes[n]["kind"]
        if kind == "KEY":
            rk[n] = True
        elif kind in ("PI", "CONST"):
            rk[n] = False
        else:
            rk[n] = any(rk[p] for p in g.predecessors(n))
    return rk


# ---------------------------------------------------------------------
# Adaptive depth picker
# ---------------------------------------------------------------------

def find_adaptive_depth(g: nx.DiGraph, c: str, rk: dict, max_depth: int) -> int:
    """
    Finds the optimal adaptive depth for a cell by walking its upstream cone.

    Returns the largest depth d in [0, max_depth] such that the d-hop fan-in BFS 
    from cell `c` has NOT (yet) introduced a non-KEY-reaching cell.

    Args:
        g (nx.DiGraph): The networkx directed graph.
        c (str): The target cell node ID.
        rk (dict): The reachability mapping from `compute_reaches_key`.
        max_depth (int): The hard maximum limit for BFS traversal.

    Returns:
        int: The computed depth (0 if the cell is unlocked/non-key-reaching).
    """
    if g.nodes[c]["kind"] != "cell" or not rk.get(c, False):
        return 0

    frontier = {c}
    for d in range(1, max_depth + 1):
        new_cells = set()
        for n in frontier:
            for p in g.predecessors(n):
                kind = g.nodes[p]["kind"]
                if kind in ("KEY", "PI", "CONST"):
                    continue                # typed leaves don't extend
                if not rk.get(p, False):
                    return d - 1            # data boundary detected
                new_cells.add(p)
        if not new_cells:
            # cone exhausted naturally at typed leaves — depth is enough
            return d
        frontier = new_cells
    return max_depth


# ---------------------------------------------------------------------
# Merkle hash at a single depth (memoised)
# ---------------------------------------------------------------------

def make_merkle(g: nx.DiGraph):
    """
    Creates a memoized recursive Merkle hash function for the given graph.

    Using a closure ensures the cache is tied to the graph instance and 
    gets garbage collected cleanly per graph.

    Args:
        g (nx.DiGraph): The graph to hash.

    Returns:
        Callable[[str, int], str]: A function taking (node_id, depth) and returning the hash.
    """
    memo = {}

    def rec(n, d):
        key = (n, d)
        if key in memo:
            return memo[key]
        leaf = _leaf_token(g, n)
        if leaf is not None:
            h = leaf
        elif d == 0:
            h = "TRUNC"
        else:
            preds = sorted(rec(p, d - 1) for p in g.predecessors(n))
            ct = g.nodes[n]["cell_type"] or "?"
            h = _h(ct + "|" + "|".join(preds))
        memo[key] = h
        return h

    return rec


# ---------------------------------------------------------------------
# Per-graph worker
# ---------------------------------------------------------------------

def process_one(args: tuple) -> tuple:
    """
    Worker function to process a single graph pickle via adaptive hashing.

    Args:
        args (tuple): (pkl_path, max_depth)

    Returns:
        tuple: (status, stem, per_cell_dataframe, summary_rows)
    """
    pkl_path, max_depth = args
    try:
        with open(pkl_path, "rb") as f:
            rec = pickle.load(f)
        g, meta = rec["graph"], rec["meta"]

        t0 = time.perf_counter()
        rk = compute_reaches_key(g)
        merkle = make_merkle(g)

        rows = []
        depth_hist = Counter()
        for n in g.nodes:
            if g.nodes[n]["kind"] != "cell":
                continue
            if not rk.get(n, False):
                rows.append({
                    "node":            n,
                    "cell_type":       g.nodes[n]["cell_type"],
                    "cone_size_d3":    0,
                    "reaches_key":     False,
                    "adaptive_depth":  0,
                    "adaptive_h":      "UNLOCKED",
                })
                depth_hist["unlocked"] += 1
                continue
            d = find_adaptive_depth(g, n, rk, max_depth)
            h = merkle(n, d)
            rows.append({
                "node":            n,
                "cell_type":       g.nodes[n]["cell_type"],
                "cone_size_d3":    0,
                "reaches_key":     True,
                "adaptive_depth":  d,
                "adaptive_h":      f"d{d}:{h}",
            })
            depth_hist[d] += 1
        dt = time.perf_counter() - t0

        per_cell = pd.DataFrame(rows)

        # Summary
        vc = per_cell["adaptive_h"].value_counts()
        n_cells = len(per_cell)
        n_clusters = vc.size
        n_singletons = int((vc == 1).sum())
        largest = int(vc.iloc[0]) if n_clusters else 0
        ge2 = int(vc[vc >= 2].sum())
        keyful = per_cell[per_cell["reaches_key"]]
        keyful_ge2 = 0
        if len(keyful):
            kvc = keyful["adaptive_h"].value_counts()
            keyful_ge2 = int(kvc[kvc >= 2].sum())
        mean_depth = (per_cell.loc[per_cell["reaches_key"],
                                   "adaptive_depth"]).mean() \
                     if keyful_ge2 else 0

        summary_row = {
            "stem":                       meta["stem"],
            "circuit":                    meta["circuit"],
            "pct":                        meta["pct"],
            "mode":                       "adaptive",
            "depth":                      max_depth,   # report the bound
            "n_cells":                    n_cells,
            "n_clusters":                 n_clusters,
            "n_singletons":               n_singletons,
            "largest_cluster":            largest,
            "cells_in_clusters_ge2":      ge2,
            "frac_cells_in_clusters_ge2": ge2 / n_cells if n_cells else 0,
            "keyful_in_clusters_ge2":     keyful_ge2,
            "n_cells_reaches_key":        int(per_cell["reaches_key"].sum()),
            "mean_adaptive_depth_keyful": round(float(mean_depth), 2)
                                          if keyful_ge2 else 0,
            "n_unlocked":                 int(depth_hist.get("unlocked", 0)),
            "depth_hist_keyful":          dict(
                                            (k, v) for k, v in depth_hist.items()
                                            if k != "unlocked"),
            "runtime_sec":                round(dt, 3),
        }
        return ("ok", meta["stem"], per_cell, [summary_row])
    except Exception as e:
        import traceback
        return ("err", str(pkl_path),
                f"{e}\n{traceback.format_exc(limit=2)}", [])


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Method A — adaptive-depth Merkle hashing (Technique 2).")
    ap.add_argument("--cache-dir", default="cache/graphs",
                    help="Directory containing the serialized graph pickles.")
    ap.add_argument("--out-dir", default="results/method_a_adaptive",
                    help="Target directory for output parquets and summaries.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of parallel processing workers.")
    ap.add_argument("--circuit", default=None,
                    help="Filter target to process a specific circuit only (e.g., c499).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the total number of processed graphs (useful for testing).")
    ap.add_argument("--max-depth", type=int, default=8,
                    help="Hard cap on adaptive depth BFS (default 8; deepest shipped variant is < 6).")
    ap.add_argument("--force", action="store_true",
                    help="Force overwriting of existing output files.")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable the TQDM progress bar output.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    out_dir   = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(cache_dir / "manifest.parquet")
    locked = manifest[manifest["kind"] == "locked"].copy()
    if args.circuit:
        locked = locked[locked["circuit"] == args.circuit]
    if args.limit is not None:
        locked = locked.sort_values(["circuit", "pct"]).head(args.limit)

    jobs = []
    for _, row in locked.iterrows():
        pkl = cache_dir / "locked" / row["circuit"] / f"{row['stem']}.pkl"
        if not pkl.exists():
            stored = Path(row["path"])
            pkl = stored if stored.is_absolute() else (cache_dir / stored)
        out = out_dir / f"{row['stem']}.cluster.parquet"
        if args.force or not out.exists():
            jobs.append((pkl, args.max_depth))

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"cache:   {cache_dir}")
    print(f"out:     {out_dir}")
    print(f"to do:   {len(jobs)} / {len(locked)} locked graphs")
    print(f"workers: {workers}")
    print(f"max-depth: {args.max_depth}\n")

    summary_rows = []
    errors = []
    if workers == 1:
        it = (process_one(j) for j in jobs)
        for r in tqdm(it, total=len(jobs), desc="adaptive",
                      disable=args.no_progress):
            kind, stem, *rest = r
            if kind == "err":
                errors.append(r);  continue
            per_cell, sum_rows = rest
            per_cell.to_parquet(out_dir / f"{stem}.cluster.parquet")
            summary_rows.extend(sum_rows)
    else:
        ex = cf.ProcessPoolExecutor(max_workers=workers)
        try:
            for r in tqdm(ex.map(process_one, jobs), total=len(jobs),
                          desc="adaptive", disable=args.no_progress):
                kind, stem, *rest = r
                if kind == "err":
                    errors.append(r);  continue
                per_cell, sum_rows = rest
                per_cell.to_parquet(out_dir / f"{stem}.cluster.parquet")
                summary_rows.extend(sum_rows)
        finally:
            ex.shutdown()

    if summary_rows:
        new_sum = pd.DataFrame(summary_rows)
        sp = out_dir / "summary.parquet"
        slim = new_sum.drop(columns=["depth_hist_keyful"])
        if sp.exists() and not args.force:
            old = pd.read_parquet(sp)
            combined = (pd.concat([old, slim])
                          .drop_duplicates(subset=["stem", "mode", "depth"],
                                           keep="last"))
        else:
            combined = slim
        combined.sort_values(["circuit", "pct"]).to_parquet(sp)
        new_sum["depth_hist_keyful"] = new_sum["depth_hist_keyful"].apply(str)
        new_sum.sort_values(["circuit", "pct"]).to_csv(
            out_dir / "summary.csv", index=False)
        print(f"\nsummary: {sp}  ({len(combined)} rows)")
        agg = (combined.groupby(["mode", "depth"])
                       .agg(n=("stem","count"),
                            mean_clusters=("n_clusters","mean"),
                            mean_largest=("largest_cluster","mean"),
                            mean_frac_ge2=("frac_cells_in_clusters_ge2","mean"),
                            mean_adaptive_depth=("mean_adaptive_depth_keyful",
                                                 "mean"),
                            mean_runtime=("runtime_sec","mean"))
                       .round(3))
        print(agg.to_string())

    if errors:
        print(f"\n{len(errors)} error(s):")
        for r in errors[:5]:
            print(f"  {r[1]}: {r[2].splitlines()[0]}")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()