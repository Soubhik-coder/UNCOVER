#!/usr/bin/env python3
"""
MethodAStruct.py — Unsupervised structural fingerprinting of cached
                   locked netlists.

Three hashing modes (all O(d * |V|), all deterministic, no learning,
no labels touched):

    merkle    fan-in Merkle hash up to depth d (commutativity-invariant
              over AND/OR/XOR via sorted children). Equivalent to
              1-WL color refinement on the reverse-DAG view.
    wl        classic Weisfeiler-Lehman with d rounds: each round mixes
              successor + predecessor hashes, then re-hashes.
    skeleton  Merkle hash, but any sub-tree that does NOT contain a KEY
              leaf is collapsed to the placeholder "DATA". This is the
              variant-shape hash from our earlier POC.

For each cached locked graph this writes a per-cell parquet:

    results/method_a/<stem>.cluster.parquet
      node, kind, cell_type, merkle_d2, merkle_d3, merkle_d4,
            wl_d2, wl_d3, skeleton_d4, skeleton_d6,
            cone_size_d3, reaches_key

and a global summary parquet across all graphs:

    results/method_a/summary.parquet
      stem, circuit, pct, mode, depth,
      n_cells, n_clusters, n_singletons,
      largest_cluster, frac_cells_in_clusters_ge2,
      ...

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
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------
# Hashing primitives
# --------------------------------------------------------------------

def _h(s: str) -> str:
    """Generates a stable 12-character SHA-1 hex digest."""
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _leaf_token(g: nx.DiGraph, n: str) -> str | None:
    """
    Returns a string token for non-cell nodes; None for logic cells.

    Args:
        g (nx.DiGraph): The networkx directed graph.
        n (str): Node ID.

    Returns:
        str | None: The leaf-type identifier mapping.
    """
    kind = g.nodes[n]["kind"]
    if kind == "PI":    return "PI"
    if kind == "KEY":   return "KEY"
    if kind == "CONST": return f"CONST:{g.nodes[n]['port']}"
    if kind == "PO":    return "PO"
    return None


def merkle_hashes(g: nx.DiGraph, depth: int) -> dict[str, str]:
    """
    Computes depth-limited Merkle tree hashes for every node's fan-in cone.
    
    Args:
        g (nx.DiGraph): The directed networkx graph.
        depth (int): The max traversal depth.
        
    Returns:
        dict[str, str]: Map of cell IDs to their computed hex hashes.
    """
    memo: dict = {}

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

    return {n: rec(n, depth) for n in g.nodes if g.nodes[n]["kind"] == "cell"}


def wl_hashes(g: nx.DiGraph, rounds: int) -> dict[str, str]:
    """
    Computes classic Weisfeiler-Lehman coloring hashes across the graph.

    Args:
        g (nx.DiGraph): The directed networkx graph.
        rounds (int): Number of message-passing/refinement iterations.
        
    Returns:
        dict[str, str]: Map of cell IDs to their refined string hashes.
    """
    color = {}
    for n, a in g.nodes(data=True):
        leaf = _leaf_token(g, n)
        color[n] = leaf if leaf is not None else (a["cell_type"] or "?")
    for _ in range(rounds):
        new_color = {}
        for n in g.nodes:
            pred = sorted(color[p] for p in g.predecessors(n))
            succ = sorted(color[s] for s in g.successors(n))
            new_color[n] = _h(color[n] + "|<|" + "|".join(pred)
                                  + "|>|" + "|".join(succ))
        color = new_color
    return {n: color[n] for n in g.nodes if g.nodes[n]["kind"] == "cell"}


def reaches_key(g: nx.DiGraph) -> dict[str, bool]:
    """
    Identifies which nodes have a KEY leaf anywhere upstream.
    
    Args:
        g (nx.DiGraph): The directed graph.
        
    Returns:
        dict[str, bool]: Dictionary mapping cell ID to True if reachable.
    """
    rk: dict = {}

    def rec(n):
        if n in rk:
            return rk[n]
        if g.nodes[n]["kind"] == "KEY":
            rk[n] = True;  return True
        if g.nodes[n]["kind"] in ("PI", "CONST"):
            rk[n] = False; return False
        v = any(rec(p) for p in g.predecessors(n))
        rk[n] = v
        return v

    for n in g.nodes:
        rec(n)
    return rk


def skeleton_hashes(g: nx.DiGraph, depth: int, rk: dict) -> dict[str, str]:
    """
    Computes structural hashes where data-only branches are prematurely collapsed.

    Args:
        g (nx.DiGraph): The networkx graph.
        depth (int): Max allowed depth to traverse.
        rk (dict): Key reachability dictionary.
        
    Returns:
        dict[str, str]: Map of node ID to string hash.
    """
    memo: dict = {}

    def rec(n, d):
        key = (n, d)
        if key in memo:
            return memo[key]
        kind = g.nodes[n]["kind"]
        if kind == "KEY":
            h = "KEY"
        elif not rk.get(n, False):
            h = "DATA"
        elif kind == "PI":
            h = "PI"
        elif kind == "CONST":
            h = f"CONST:{g.nodes[n]['port']}"
        elif d == 0:
            h = "TRUNC"
        else:
            preds = sorted(rec(p, d - 1) for p in g.predecessors(n))
            ct = g.nodes[n]["cell_type"] or "?"
            h = _h(ct + "|" + "|".join(preds))
        memo[key] = h
        return h

    return {n: rec(n, depth) for n in g.nodes if g.nodes[n]["kind"] == "cell"}


def cone_size(g: nx.DiGraph, depth: int) -> dict[str, int]:
    """
    Calculates the number of distinct ancestors within a depth limit.

    Args:
        g (nx.DiGraph): The networkx graph.
        depth (int): The max BFS radius.
        
    Returns:
        dict[str, int]: Map of node ID to the integer ancestor count.
    """
    out: dict = {}
    for n in g.nodes:
        if g.nodes[n]["kind"] != "cell":
            continue
        frontier = {n}
        seen = {n}
        for _ in range(depth):
            nxt = set()
            for v in frontier:
                for p in g.predecessors(v):
                    if p not in seen:
                        seen.add(p);  nxt.add(p)
            frontier = nxt
            if not frontier:
                break
        out[n] = len(seen) - 1
    return out


# --------------------------------------------------------------------
# Per-graph driver (one worker)
# --------------------------------------------------------------------

DEPTHS_MERKLE   = (2, 3, 4)
DEPTHS_WL       = (2, 3)
DEPTHS_SKELETON = (4, 6)


def process_one(args: str | Path) -> tuple:
    """
    Worker job to read a graph pickle and generate all associated structural features.

    Args:
        args (str | Path): Path to the target pickle file.

    Returns:
        tuple: Worker status message and processing data.
    """
    pkl_path = args
    try:
        with open(pkl_path, "rb") as f:
            rec = pickle.load(f)
        g, meta = rec["graph"], rec["meta"]
        stem, circuit, pct = meta["stem"], meta["circuit"], meta["pct"]

        t0 = time.perf_counter()
        rk = reaches_key(g)
        feats = {}
        for d in DEPTHS_MERKLE:
            feats[f"merkle_d{d}"] = merkle_hashes(g, d)
        for d in DEPTHS_WL:
            feats[f"wl_d{d}"] = wl_hashes(g, d)
        for d in DEPTHS_SKELETON:
            feats[f"skeleton_d{d}"] = skeleton_hashes(g, d, rk)
        cs3 = cone_size(g, 3)
        dt = time.perf_counter() - t0

        cell_nodes = [n for n in g.nodes if g.nodes[n]["kind"] == "cell"]
        rows = []
        for n in cell_nodes:
            row = {
                "node":          n,
                "cell_type":     g.nodes[n]["cell_type"],
                "cone_size_d3":  cs3.get(n, 0),
                "reaches_key":   bool(rk.get(n, False)),
            }
            for k, d in feats.items():
                row[k] = d[n]
            rows.append(row)
        per_cell = pd.DataFrame(rows)

        summary_rows = []
        for col in [c for c in per_cell.columns
                    if c.startswith(("merkle_", "wl_", "skeleton_"))]:
            vc = per_cell[col].value_counts()
            n_cells = len(per_cell)
            n_clusters = vc.size
            n_singletons = int((vc == 1).sum())
            largest = int(vc.iloc[0]) if n_clusters else 0
            cells_in_ge2 = int(vc[vc >= 2].sum())
            keyful = per_cell[per_cell["reaches_key"]]
            keyful_in_ge2 = 0
            if len(keyful):
                kvc = keyful[col].value_counts()
                keyful_in_ge2 = int(kvc[kvc >= 2].sum())
            mode, depth = col.split("_")
            summary_rows.append({
                "stem":               stem,
                "circuit":            circuit,
                "pct":                pct,
                "mode":               mode,
                "depth":              int(depth[1:]),
                "n_cells":            n_cells,
                "n_clusters":         n_clusters,
                "n_singletons":       n_singletons,
                "largest_cluster":    largest,
                "cells_in_clusters_ge2": cells_in_ge2,
                "frac_cells_in_clusters_ge2": cells_in_ge2 / n_cells if n_cells else 0,
                "keyful_in_clusters_ge2":    keyful_in_ge2,
                "n_cells_reaches_key":       int(per_cell["reaches_key"].sum()),
                "runtime_sec":               round(dt, 3),
            })
        return ("ok", stem, per_cell, summary_rows)
    except Exception as e:
        import traceback
        return ("err", str(pkl_path), f"{e}\n{traceback.format_exc(limit=2)}", [])


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Unsupervised structural fingerprinting of cached "
                    "locked netlists. Method A baseline.")
    ap.add_argument("--cache-dir", default="cache/graphs",
                    help="Directory containing the serialized graph pickles.")
    ap.add_argument("--out-dir", default="results/method_a",
                    help="Target destination directory for the structural fingerprinting results.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of concurrent workers for parallel processing.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N locked graphs (useful for smoke tests).")
    ap.add_argument("--circuit", default=None,
                    help="Target a specific circuit to process (e.g., 'c499').")
    ap.add_argument("--force", action="store_true",
                    help="Force rebuilding and overwriting of existing parquet files.")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable the TQDM progress bar output.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    out_dir   = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    man_path = cache_dir / "manifest.parquet"
    if not man_path.exists():
        raise SystemExit(f"manifest not found: {man_path}; "
                         "run BuildGraphs.py first")
    manifest = pd.read_parquet(man_path)
    locked = manifest[manifest["kind"] == "locked"].copy()
    if args.circuit:
        locked = locked[locked["circuit"] == args.circuit]
    if args.limit is not None:
        locked = locked.sort_values(["circuit", "pct"]).head(args.limit)

    to_do = []
    for _, row in locked.iterrows():
        out_pkl = out_dir / f"{row['stem']}.cluster.parquet"
        if args.force or not out_pkl.exists():
            to_do.append(Path(row["path"]))

    print(f"cache:    {cache_dir}")
    print(f"out:      {out_dir}")
    print(f"to do:    {len(to_do)} / {len(locked)} locked graphs")

    if not to_do:
        print("(everything already cached; pass --force to rebuild)")

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"workers:  {workers}\n")

    summary_rows = []
    errors = []

    def handle(result):
        nonlocal summary_rows
        kind = result[0]
        if kind == "err":
            errors.append(result)
            return
        stem, per_cell, sum_rows = result[1], result[2], result[3]
        per_cell.to_parquet(out_dir / f"{stem}.cluster.parquet")
        summary_rows.extend(sum_rows)

    if workers == 1:
        it = (process_one(j) for j in to_do)
    else:
        ex = cf.ProcessPoolExecutor(max_workers=workers)
        it = ex.map(process_one, to_do)
    pbar = tqdm(it, total=len(to_do), desc="hashing",
                disable=args.no_progress)
    for r in pbar:
        handle(r)
    if workers != 1:
        ex.shutdown()

    summary_path = out_dir / "summary.parquet"
    if summary_rows:
        new_sum = pd.DataFrame(summary_rows)
        if summary_path.exists() and not args.force:
            old = pd.read_parquet(summary_path)
            combined = (pd.concat([old, new_sum])
                          .drop_duplicates(subset=["stem", "mode", "depth"],
                                           keep="last"))
        else:
            combined = new_sum
        combined.sort_values(["circuit", "pct", "mode", "depth"]) \
                .to_parquet(summary_path)
        combined.to_csv(out_dir / "summary.csv", index=False)
        print(f"\nsummary:  {summary_path}  ({len(combined)} rows)")

        agg = (combined.groupby(["mode", "depth"])
                       .agg(n=("stem", "count"),
                            mean_clusters=("n_clusters", "mean"),
                            mean_largest=("largest_cluster", "mean"),
                            mean_frac_ge2=("frac_cells_in_clusters_ge2", "mean"),
                            mean_runtime=("runtime_sec", "mean"))
                       .round(3))
        print()
        print(agg.to_string())
    else:
        print("no summary rows produced")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for kind, path, msg, _ in errors[:5]:
            print(f"  {path}: {msg.splitlines()[0]}")

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()