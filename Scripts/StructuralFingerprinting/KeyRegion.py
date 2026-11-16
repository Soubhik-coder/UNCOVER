#!/usr/bin/env python3
"""
MethodAKeyRegion.py — KEY-region (variant-shape) hashing.

This is Technique 3: instead of hashing a fixed-depth fan-in cone, we
hash the maximal connected subgraph of KEY-reaching cells around each
cell. The boundary policy mirrors what GateLock variants actually look
like:

  * A node is IN the KEY-region of a cell c iff
       (i) it lies on the upstream side of c (a transitive predecessor
           of c, or c itself), AND
       (ii) its fan-in cone contains at least one KEY leaf.
  * On the data side (predecessors that do NOT reach a KEY leaf), we
    stop recursion and emit a generic 'DATA' leaf token. This matches
    Q1(a) we agreed on.
  * We do NOT cross another locked variant's Y output. Without labels
    we detect Y-cuts heuristically: a Y node is a cell whose KEY-region
    has at least one OUT-edge to a cell that is itself NOT KEY-reaching.
    This matches Q2(b).

For each cell c we then compute:

    keyregion_hash(c) = WL graph hash of the canonicalised KEY-region
                        subgraph, with KEY/DATA/PI/CONST leaves typed.

Implementation choices:
  * We use `networkx.weisfeiler_lehman_graph_hash` (iteration_count and
    digest_size are exposed via CLI), which is itself a sound and
    well-studied canonical hash for small graphs.
  * The recursion has a guard `--max-region-size` (default 64): regions
    larger than this get a placeholder "BIG:<n>" hash so a single
    pathological cell can't cost us seconds. All shipped Database.py
    variants are <50 cells.

Outputs:
    <out-dir>/<stem>.cluster.parquet  with columns:
        node, cell_type, cone_size_d3, reaches_key,
        region_size, region_depth, n_key_in_region, n_data_boundaries,
        keyregion_h        (string hash)
    <out-dir>/summary.parquet           (one row per graph)

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
import warnings
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm

# nx >= 3.4 deprecates the 'edge_attr=None' default; silence it
warnings.filterwarnings("ignore", category=DeprecationWarning, module="networkx")


# ---------------------------------------------------------------------
# Reachability: which nodes have a KEY leaf in their fan-in cone?
# ---------------------------------------------------------------------

def compute_reaches_key(g: nx.DiGraph) -> dict[str, bool]:
    """
    Computes upstream key reachability for all nodes.

    Args:
        g (nx.DiGraph): The networkx directed graph.

    Returns:
        dict[str, bool]: A mapping where `rk[n] = True` iff some KEY leaf is upstream of n.
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
# KEY-region extraction around a cell c
# ---------------------------------------------------------------------

def extract_keyregion(g: nx.DiGraph, c: str, rk: dict, max_size: int) -> tuple[set, int, bool]:
    """
    Isolates the connected structural component involving KEY lines bounded by data inputs or other logic locks.

    Args:
        g (nx.DiGraph): The source graph.
        c (str): The root cell being evaluated.
        rk (dict): Key reachability dictionary.
        max_size (int): Size cap to abort runaway extractions.

    Returns:
        tuple[set, int, bool]: (region_nodes, data_boundary_count, hit_max)
    """
    if not rk.get(c, False):
        return set(), 0, False
    if g.nodes[c]["kind"] != "cell":
        return set(), 0, False

    region = {c}
    data_boundaries = 0
    hit_max = False
    stack = [c]

    while stack:
        n = stack.pop()
        for p in g.predecessors(n):
            kind = g.nodes[p]["kind"]
            if kind == "KEY":
                continue
            if kind == "CONST":
                continue
            if kind == "PI":
                continue
            if not rk.get(p, False):
                data_boundaries += 1
                continue
            if p in region:
                continue

            is_y_cut = False
            for s in g.successors(p):
                if s == n:
                    continue
                if g.nodes[s]["kind"] == "cell" and rk.get(s, False):
                    if s not in region:
                        is_y_cut = True
                        break
            region.add(p)
            if is_y_cut:
                continue
            stack.append(p)
            if len(region) > max_size:
                hit_max = True
                return region, data_boundaries, True

    return region, data_boundaries, hit_max


# ---------------------------------------------------------------------
# KEY-region subgraph -> canonical hash
# ---------------------------------------------------------------------

def keyregion_hash(g: nx.DiGraph, c: str, region: set, rk: dict,
                   data_boundaries: int, wl_iter: int, digest_size: int) -> tuple[str, int, int]:
    """
    Builds a bounded subgraph representing the KEY-region and returns its WL hash.

    Args:
        g (nx.DiGraph): Master graph.
        c (str): Root cell ID.
        region (set): Set of nodes included in the region.
        rk (dict): Reachability dict.
        data_boundaries (int): Counter for unconnected data boundary nodes.
        wl_iter (int): WL refinement iterations.
        digest_size (int): Max digest size for the output string.

    Returns:
        tuple[str, int, int]: (hash_str, n_key_in_region, region_depth).
    """
    sub = nx.DiGraph()
    n_key = 0
    for n in region:
        sub.add_node(n, label=f"cell:{g.nodes[n]['cell_type']}")
    data_counter = 0
    for n in region:
        for p in g.predecessors(n):
            kind = g.nodes[p]["kind"]
            if p in region:
                dst_port = g[p][n].get("dst_port", "?")
                sub.add_edge(p, n, label=f"in:{dst_port}")
                continue
            if kind == "KEY":
                leaf = f"_K_{p}"
                if leaf not in sub:
                    sub.add_node(leaf, label="KEY")
                    n_key += 1
                sub.add_edge(leaf, n,
                             label=f"in:{g[p][n].get('dst_port','?')}")
            elif kind == "PI":
                leaf = f"_P_{p}"
                if leaf not in sub:
                    sub.add_node(leaf, label="PI")
                sub.add_edge(leaf, n,
                             label=f"in:{g[p][n].get('dst_port','?')}")
            elif kind == "CONST":
                leaf = f"_C_{p}"
                if leaf not in sub:
                    sub.add_node(leaf, label=f"CONST:{g.nodes[p]['port']}")
                sub.add_edge(leaf, n,
                             label=f"in:{g[p][n].get('dst_port','?')}")
            else:
                data_counter += 1
                leaf = f"_D_{data_counter}"
                sub.add_node(leaf, label="DATA")
                sub.add_edge(leaf, n,
                             label=f"in:{g[p][n].get('dst_port','?')}")

    if sub.number_of_nodes() == 0:
        return ("EMPTY", 0, 0)

    try:
        depth = nx.dag_longest_path_length(sub)
    except Exception:
        depth = -1

    sub.nodes[c]["label"] = "ROOT:" + sub.nodes[c]["label"]
    h = nx.weisfeiler_lehman_graph_hash(
        sub.to_undirected(),
        node_attr="label",
        edge_attr="label",
        iterations=wl_iter,
        digest_size=digest_size,
    )
    return (h, n_key, depth)


# ---------------------------------------------------------------------
# Per-graph worker
# ---------------------------------------------------------------------

def process_one(args: tuple) -> tuple:
    """
    Multiprocessing job to compute region hashes for a single graph pickle.

    Args:
        args (tuple): (pkl_path, max_region_size, wl_iter, digest_size)

    Returns:
        tuple: Worker status message and internal DataFrames.
    """
    pkl_path, max_region_size, wl_iter, digest_size = args
    try:
        with open(pkl_path, "rb") as f:
            rec = pickle.load(f)
        g, meta = rec["graph"], rec["meta"]

        t0 = time.perf_counter()
        rk = compute_reaches_key(g)

        rows = []
        for n in g.nodes:
            if g.nodes[n]["kind"] != "cell":
                continue
            if not rk[n]:
                rows.append({
                    "node":               n,
                    "cell_type":          g.nodes[n]["cell_type"],
                    "cone_size_d3":       0,
                    "reaches_key":        False,
                    "region_size":        0,
                    "region_depth":       0,
                    "n_key_in_region":    0,
                    "n_data_boundaries":  0,
                    "keyregion_h":        "DATA",
                })
                continue
            region, n_data, hit_max = extract_keyregion(g, n, rk,
                                                       max_region_size)
            if hit_max:
                h, n_key, dep = (f"BIG:{len(region)}", 0, -1)
            else:
                h, n_key, dep = keyregion_hash(g, n, region, rk, n_data,
                                               wl_iter, digest_size)
            rows.append({
                "node":               n,
                "cell_type":          g.nodes[n]["cell_type"],
                "cone_size_d3":       0,
                "reaches_key":        True,
                "region_size":        len(region),
                "region_depth":       int(dep),
                "n_key_in_region":    n_key,
                "n_data_boundaries":  n_data,
                "keyregion_h":        h,
            })
        dt = time.perf_counter() - t0

        per_cell = pd.DataFrame(rows)

        vc = per_cell["keyregion_h"].value_counts()
        n_cells = len(per_cell)
        n_clusters = vc.size
        n_singletons = int((vc == 1).sum())
        largest = int(vc.iloc[0]) if n_clusters else 0
        ge2 = int(vc[vc >= 2].sum())
        keyful = per_cell[per_cell["reaches_key"]]
        keyful_ge2 = 0
        if len(keyful):
            kvc = keyful["keyregion_h"].value_counts()
            keyful_ge2 = int(kvc[kvc >= 2].sum())

        summary_row = {
            "stem":                       meta["stem"],
            "circuit":                    meta["circuit"],
            "pct":                        meta["pct"],
            "mode":                       "keyregion",
            "depth":                      max_region_size,
            "n_cells":                    n_cells,
            "n_clusters":                 n_clusters,
            "n_singletons":               n_singletons,
            "largest_cluster":            largest,
            "cells_in_clusters_ge2":      ge2,
            "frac_cells_in_clusters_ge2": ge2 / n_cells if n_cells else 0,
            "keyful_in_clusters_ge2":     keyful_ge2,
            "n_cells_reaches_key":        int(per_cell["reaches_key"].sum()),
            "mean_region_size":           round(
                per_cell.loc[per_cell["reaches_key"], "region_size"]
                        .mean() if keyful_ge2 else 0, 2),
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
    ap = argparse.ArgumentParser(description="Method A — KEY-region hashing (Variant Shape)")
    ap.add_argument("--cache-dir", default="cache/graphs",
                    help="Directory containing the serialized graph pickles.")
    ap.add_argument("--out-dir", default="results/method_a_keyregion",
                    help="Target directory for output parquets and summaries.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of concurrent workers for parallel processing.")
    ap.add_argument("--circuit", default=None,
                    help="Target a specific circuit to process (e.g., 'c499').")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N locked graphs (useful for smoke tests).")
    ap.add_argument("--max-region-size", type=int, default=64,
                    help="Hard cap on region size to keep WL hashing cheap (default 64; shipped variants are <50).")
    ap.add_argument("--wl-iter", type=int, default=3,
                    help="WL iterations for the subgraph hashing mechanism (default 3).")
    ap.add_argument("--digest-size", type=int, default=16,
                    help="WL digest string size in bytes (default 16).")
    ap.add_argument("--force", action="store_true",
                    help="Force rebuilding and overwriting of existing parquet files.")
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
            jobs.append((pkl, args.max_region_size,
                         args.wl_iter, args.digest_size))

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"cache:   {cache_dir}")
    print(f"out:     {out_dir}")
    print(f"to do:   {len(jobs)} / {len(locked)} locked graphs")
    print(f"workers: {workers}")
    print(f"max-region-size={args.max_region_size}  wl-iter={args.wl_iter}  "
          f"digest-size={args.digest_size}\n")

    summary_rows = []
    errors = []
    if workers == 1:
        it = (process_one(j) for j in jobs)
        for r in tqdm(it, total=len(jobs), desc="keyregion",
                      disable=args.no_progress):
            kind, stem, *rest = r
            if kind == "err":
                errors.append(r); continue
            per_cell, sum_rows = rest
            per_cell.to_parquet(out_dir / f"{stem}.cluster.parquet")
            summary_rows.extend(sum_rows)
    else:
        ex = cf.ProcessPoolExecutor(max_workers=workers)
        try:
            for r in tqdm(ex.map(process_one, jobs), total=len(jobs),
                          desc="keyregion", disable=args.no_progress):
                kind, stem, *rest = r
                if kind == "err":
                    errors.append(r); continue
                per_cell, sum_rows = rest
                per_cell.to_parquet(out_dir / f"{stem}.cluster.parquet")
                summary_rows.extend(sum_rows)
        finally:
            ex.shutdown()

    if summary_rows:
        new_sum = pd.DataFrame(summary_rows)
        sp = out_dir / "summary.parquet"
        if sp.exists() and not args.force:
            old = pd.read_parquet(sp)
            combined = (pd.concat([old, new_sum])
                          .drop_duplicates(subset=["stem", "mode", "depth"],
                                           keep="last"))
        else:
            combined = new_sum
        combined.sort_values(["circuit", "pct"]).to_parquet(sp)
        combined.to_csv(out_dir / "summary.csv", index=False)
        print(f"\nsummary: {sp}  ({len(combined)} rows)")
        agg = (combined.groupby(["mode", "depth"])
                       .agg(n=("stem","count"),
                            mean_clusters=("n_clusters","mean"),
                            mean_largest=("largest_cluster","mean"),
                            mean_frac_ge2=("frac_cells_in_clusters_ge2","mean"),
                            mean_region_size=("mean_region_size","mean"),
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