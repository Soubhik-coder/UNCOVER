#!/usr/bin/env python3
"""
MethodACollapse.py — Technique 4: post-processing cluster collapse.

Given any per-cell parquet file produced by another method
(MethodAStruct.py or MethodAKeyRegion.py), we *merge* clusters
that look like the same Database.py variant attached to different
upstream contexts.

The collapse uses a context-INDEPENDENT fingerprint per cluster:

  fp(cluster) = (
      root_cell_type,
      #distinct KEY bits touched by the cluster's fan-in cone,
      cell-type histogram inside the KEY-region,
      KEY-region size (capped at 64),
      KEY-region max depth (capped at 16),
  )

Two clusters with the same fingerprint are merged into one super-cluster.

We support two collapse policies:

  conservative   merge only on (root_cell_type, n_key_in_region)
  aggressive     merge on the full fingerprint above

The script writes the SAME per-cell parquet files but with new columns
added:  collapsed_cons_<orig_col>,  collapsed_aggr_<orig_col>.

So downstream evaluation can score both the original and the collapsed 
versions side by side without any further changes.

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


METADATA_COLS = {"node", "cell_type", "cone_size_d3", "reaches_key",
                 "region_size", "region_depth",
                 "n_key_in_region", "n_data_boundaries"}


def _h(s: str) -> str:
    """Generates a stable 12-character SHA-1 hex digest."""
    return hashlib.sha1(s.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------
# Per-cluster fingerprint using cached graph context
# ---------------------------------------------------------------------

def compute_reaches_key(g: nx.DiGraph) -> dict[str, bool]:
    """
    Computes whether each node has a KEY leaf anywhere in its upstream fan-in cone.

    Args:
        g (nx.DiGraph): The networkx directed graph.

    Returns:
        dict[str, bool]: Mapping of node ID to boolean.
    """
    rk = {}
    for n in nx.topological_sort(g):
        kind = g.nodes[n]["kind"]
        if kind == "KEY":            rk[n] = True
        elif kind in ("PI","CONST"): rk[n] = False
        else:                        rk[n] = any(rk[p]
                                                  for p in g.predecessors(n))
    return rk


def cell_keyregion(g: nx.DiGraph, c: str, rk: dict, max_size: int = 64) -> tuple:
    """
    Walks upstream from a cell through nodes whose fan-in reaches a KEY.

    Args:
        g (nx.DiGraph): The networkx directed graph.
        c (str): The root cell node ID.
        rk (dict): Pre-computed key reachability dict.
        max_size (int, optional): Cut-off limit to prevent unbounded traversal. Defaults to 64.

    Returns:
        tuple: (region_size, n_key, depth, cell_type_hist) summarizing the region's properties.
    """
    if g.nodes[c]["kind"] != "cell" or not rk.get(c, False):
        return (0, 0, 0, Counter())
    region = {c}
    stack = [c]
    n_key = 0
    seen_key = set()
    edges = []
    while stack:
        n = stack.pop()
        for p in g.predecessors(n):
            kind = g.nodes[p]["kind"]
            if kind == "KEY":
                if p not in seen_key:
                    seen_key.add(p); n_key += 1
                continue
            if kind in ("PI", "CONST"):
                continue
            if not rk.get(p, False):
                continue
            edges.append((p, n))
            if p in region: continue
            region.add(p)
            stack.append(p)
            if len(region) > max_size:
                return (len(region), n_key, -1,
                        Counter(g.nodes[x]["cell_type"] for x in region))
    # depth via small subgraph
    if len(region) > 1:
        sub = nx.DiGraph()
        for x in region:
            sub.add_node(x)
        for p, n in edges:
            if p in region and n in region:
                sub.add_edge(p, n)
        try:
            depth = nx.dag_longest_path_length(sub)
        except Exception:
            depth = -1
    else:
        depth = 0
    return (len(region), n_key, depth,
            Counter(g.nodes[x]["cell_type"] for x in region))


def per_cell_fingerprint(g: nx.DiGraph, c: str, rk: dict, max_size: int) -> tuple[str, str]:
    """
    Generates conservative and aggressive heuristic fingerprints for a cell.

    Args:
        g (nx.DiGraph): The networkx directed graph.
        c (str): Target cell ID.
        rk (dict): Key reachability dictionary.
        max_size (int): Max region constraint.

    Returns:
        tuple[str, str]: (conservative_fp, aggressive_fp) hashed signatures.
    """
    cell_type = g.nodes[c]["cell_type"]
    if not rk.get(c, False):
        return ("UNLOCKED", "UNLOCKED")
    size, n_key, depth, hist = cell_keyregion(g, c, rk, max_size)
    cons = _h(f"{cell_type}|k={n_key}")
    hist_str = "|".join(f"{t}:{n}" for t, n in sorted(hist.items()))
    aggr = _h(f"{cell_type}|k={n_key}|sz={size}|d={depth}|{hist_str}")
    return (cons, aggr)


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------

def process_one(args: tuple) -> tuple:
    """
    Multiprocessing worker to merge structurally identical clusters within a parquet file.

    Args:
        args (tuple): (pkl_path, cluster_pq, out_pq, max_region)

    Returns:
        tuple: Worker status message and processing metadata.
    """
    pkl_path, cluster_pq, out_pq, max_region = args
    try:
        with open(pkl_path, "rb") as f:
            rec = pickle.load(f)
        g = rec["graph"]
        rk = compute_reaches_key(g)

        per_cell = pd.read_parquet(cluster_pq)
        original_cols = [c for c in per_cell.columns
                         if c not in METADATA_COLS and not c.startswith("_")
                         and not c.startswith("collapsed_")]

        cons_fp = {}
        aggr_fp = {}
        for n in per_cell["node"]:
            if n in g.nodes:
                c_fp, a_fp = per_cell_fingerprint(g, n, rk, max_region)
            else:
                c_fp, a_fp = ("UNK", "UNK")
            cons_fp[n] = c_fp
            aggr_fp[n] = a_fp

        for col in original_cols:
            for policy_name, fp_map in (("cons", cons_fp), ("aggr", aggr_fp)):
                cluster_fps = defaultdict(set)
                for n, cid in zip(per_cell["node"], per_cell[col]):
                    cluster_fps[cid].add(fp_map[n])

                cluster_to_super = {}
                super_for = {}
                next_super = 0
                for cid, fps in cluster_fps.items():
                    if len(fps) == 1:
                        key = next(iter(fps))
                    else:
                        key = f"MIX:{cid}"
                    if key not in super_for:
                        super_for[key] = f"S{next_super}"
                        next_super += 1
                    cluster_to_super[cid] = super_for[key]

                new_col = f"collapsed_{policy_name}_{col}"
                per_cell[new_col] = per_cell[col].map(cluster_to_super)

        per_cell.to_parquet(out_pq)
        return ("ok", str(cluster_pq.name),
                {"n_cells": len(per_cell),
                 "n_orig_cols": len(original_cols)})
    except Exception as e:
        import traceback
        return ("err", str(cluster_pq), f"{e}\n{traceback.format_exc(limit=2)}",
                {})


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Post-processing cluster collapse. Reads per-cell "
                    "parquets from --in-method-dir and writes augmented "
                    "parquets to --out-dir with new 'collapsed_*' columns.")
    ap.add_argument("--cache-dir", default="cache/graphs",
                    help="Directory containing the serialized graph pickles.")
    ap.add_argument("--in-method-dir", required=True,
                    help="Directory containing <stem>.cluster.parquet files produced by a primary hashing method.")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write the augmented collapsed per-cell parquets.")
    ap.add_argument("--max-region-size", type=int, default=64,
                    help="Hard cap on subgraph region sizes considered for fingerprints (default: 64).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of parallel processing workers.")
    ap.add_argument("--force", action="store_true",
                    help="Force rebuilding and overwriting of existing output parquets.")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable the TQDM progress bar output.")
    args = ap.parse_args()

    cache_dir  = Path(args.cache_dir).resolve()
    in_dir     = Path(args.in_method_dir).resolve()
    out_dir    = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(cache_dir / "manifest.parquet")
    locked = manifest[manifest["kind"] == "locked"]
    pkl_of_stem = {}
    for _, row in locked.iterrows():
        p = cache_dir / "locked" / row["circuit"] / f"{row['stem']}.pkl"
        if not p.exists():
            stored = Path(row["path"])
            p = stored if stored.is_absolute() else (cache_dir / stored)
        pkl_of_stem[row["stem"]] = p

    cluster_files = sorted(in_dir.glob("*.cluster.parquet"))
    jobs = []
    for cf_path in cluster_files:
        stem = cf_path.name[:-len(".cluster.parquet")]
        if stem not in pkl_of_stem:
            print(f"  WARNING: no cached graph for {stem}, skipping")
            continue
        out = out_dir / cf_path.name
        if args.force or not out.exists():
            jobs.append((pkl_of_stem[stem], cf_path, out,
                         args.max_region_size))

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"cache:   {cache_dir}")
    print(f"input:   {in_dir}  ({len(cluster_files)} files)")
    print(f"output:  {out_dir}")
    print(f"to do:   {len(jobs)}")
    print(f"workers: {workers}\n")

    errors = []
    if workers == 1:
        it = (process_one(j) for j in jobs)
        for r in tqdm(it, total=len(jobs), desc="collapsing",
                      disable=args.no_progress):
            if r[0] == "err":
                errors.append(r)
    else:
        ex = cf.ProcessPoolExecutor(max_workers=workers)
        try:
            for r in tqdm(ex.map(process_one, jobs), total=len(jobs),
                          desc="collapsing", disable=args.no_progress):
                if r[0] == "err":
                    errors.append(r)
        finally:
            ex.shutdown()

    if errors:
        print(f"\n{len(errors)} error(s):")
        for r in errors[:5]:
            print(f"  {r[1]}: {r[2].splitlines()[0]}")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()