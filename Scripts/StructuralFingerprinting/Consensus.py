#!/usr/bin/env python3
"""
MethodAConsensus.py — Multi-resolution consensus clustering (Memory & CPU Optimized).

Optimizations applied:
  1. Integer Node Mapping & 64-bit Bit-Packing: Replaces (str, str) tuple keys 
     with packed 64-bit integers ((u << 32) | v) to slash RAM usage.
  2. Incremental Union-Find: Evaluates k_values in descending order using a single
     persistent Union-Find array. Runs in O(|E|) total time regardless of how many
     --k values are passed.
  3. Safe Default Caps: Lowers default --max-cluster-size to 250 (all shipped 
     GateLock variants are <50 cells) and caps default workers to 8 to prevent 
     concurrent memory spikes on high-thread Xeon systems.
  4. Atomic File Writes: Prevents corrupted/half-written parquets if interrupted.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


METADATA_COLS = {
    "node", "cell_type", "cone_size_d3", "reaches_key",
    "region_size", "region_depth",
    "n_key_in_region", "n_data_boundaries"
}


# ---------------------------------------------------------------------
# Array-based Union-Find (0..N-1 integer indexed)
# ---------------------------------------------------------------------

class FastUF:
    """Array-backed Disjoint-Set structure for fast integer indexing."""
    __slots__ = ("parent",)

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        parent = self.parent
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            # Deterministic root selection (smaller index wins)
            if a < b:
                parent[b] = a
            else:
                parent[a] = b


# ---------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------

def process_one(args: tuple) -> tuple:
    """
    Multiprocessing worker to calculate consensus clusters across method files.
    """
    stem, file_list, k_values, max_cluster_size, out_pq = args
    tmp_pq = out_pq.with_suffix(".tmp.parquet")

    try:
        if not file_list:
            return ("err", stem, "no input files", {})

        # 1. Read first file to establish canonical node order and metadata
        df0 = pd.read_parquet(file_list[0]).set_index("node")
        nodes = list(df0.index)
        n_nodes = len(nodes)
        node_to_id = {node: idx for idx, node in enumerate(nodes)}

        per_cell = pd.DataFrame({"node": nodes})
        for c in ("cell_type", "cone_size_d3", "reaches_key"):
            if c in df0.columns:
                per_cell[c] = df0[c].values

        # 2. Extract valid cluster columns and count pairs using 64-bit packed ints
        pair_count: dict[int, int] = defaultdict(int)
        n_cols_in = 0
        min_k = min(k_values) if k_values else 2

        for src_idx, p in enumerate(file_list):
            df = df0 if src_idx == 0 else pd.read_parquet(p).set_index("node")

            for c in df.columns:
                if c in METADATA_COLS or c.startswith("_"):
                    continue
                n_cols_in += 1

                # Align series to canonical node order if indices differ
                s = df[c]
                if src_idx > 0 and not s.index.equals(df0.index):
                    s = s.reindex(df0.index)

                # Group using integer positional indices directly
                codes, uniques = pd.factorize(s.values, sort=False)
                if len(uniques) == 0:
                    continue

                # Count frequencies of each cluster code (ignoring NaNs where code == -1)
                valid_mask = codes >= 0
                valid_codes = codes[valid_mask]
                valid_indices = np.nonzero(valid_mask)[0]

                counts = np.bincount(valid_codes)
                # Keep only clusters within (1, max_cluster_size]
                keep_clusters = np.where((counts > 1) & (counts <= max_cluster_size))[0]
                if len(keep_clusters) == 0:
                    continue

                # Sort indices by cluster code to slice groups contiguously
                order = np.argsort(valid_codes, kind="mergesort")
                sorted_codes = valid_codes[order]
                sorted_nodes = valid_indices[order]

                split_pts = np.searchsorted(sorted_codes, keep_clusters)
                for cl_id, start_idx in zip(keep_clusters, split_pts):
                    end_idx = start_idx + counts[cl_id]
                    members = sorted_nodes[start_idx:end_idx]
                    members.sort()
                    # Pack (u, v) into a single 64-bit integer: (u << 32) | v
                    for a, b in combinations(members.tolist(), 2):
                        pair_count[(a << 32) | b] += 1

            if src_idx > 0:
                del df

        del df0

        # 3. Bucket packed pairs by their agreement count (filtering out < min_k)
        max_k = max(k_values)
        buckets: dict[int, list[int]] = defaultdict(list)
        for packed_pair, cnt in pair_count.items():
            if cnt >= min_k:
                # Cap at max_k so counts >= max(k_values) start in the top bucket
                b_key = cnt if cnt < max_k else max_k
                buckets[b_key].append(packed_pair)

        n_pairs = len(pair_count)
        pair_count.clear()
        del pair_count

        # 4. Incremental Union-Find (Descending k order)
        uf = FastUF(n_nodes)
        sorted_k_desc = sorted(k_values, reverse=True)
        max_possible_cnt = max(buckets.keys(), default=0)

        current_ceil = max_possible_cnt
        results_by_k = {}

        for k in sorted_k_desc:
            # Merge all buckets from current_ceil down to k
            for c_val in range(current_ceil, k - 1, -1):
                if c_val in buckets:
                    for packed in buckets.pop(c_val):
                        uf.union(packed >> 32, packed & 0xFFFFFFFF)
            current_ceil = k - 1
            results_by_k[k] = [nodes[uf.find(i)] for i in range(n_nodes)]

        # Add columns in ascending k order to match original schema
        for k in sorted(k_values):
            per_cell[f"consensus_k{k}"] = results_by_k[k]

        # 5. Atomic write to prevent corrupted files on interruption
        per_cell.to_parquet(tmp_pq)
        tmp_pq.replace(out_pq)

        gc.collect()
        return ("ok", stem, "", {
            "n_cells": n_nodes,
            "n_cols_in": n_cols_in,
            "n_pairs": n_pairs
        })

    except Exception as e:
        if tmp_pq.exists():
            try:
                tmp_pq.unlink()
            except OSError:
                pass
        import traceback
        return ("err", stem, f"{e}\n{traceback.format_exc(limit=2)}", {})


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Multi-resolution consensus clustering across one or "
                    "more method-output directories.")
    ap.add_argument("--in-method-dir", required=True, action="append",
                    help="Directory of per-cell parquets to consume. May be passed multiple times.")
    ap.add_argument("--out-dir", required=True,
                    help="Target destination directory for the consensus clustering output parquets.")
    ap.add_argument("--k", type=int, action="append",
                    help="Agreement threshold. May be passed multiple times; defaults to [2, 3].")
    ap.add_argument("--max-cluster-size", type=int, default=250,
                    help="Ignore voting on clusters larger than this threshold (default: 250; "
                         "all shipped GateLock variants are <50 cells).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of concurrent multiprocessing workers (defaults to min(8, cpu_count - 1)).")
    ap.add_argument("--force", action="store_true",
                    help="Forces complete recreation and overwriting of output parquets.")
    ap.add_argument("--no-progress", action="store_true",
                    help="Hides all TQDM activity progress indicators.")
    args = ap.parse_args()

    in_dirs = [Path(d).resolve() for d in args.in_method_dir]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = sorted(set(args.k if args.k else [2, 3]))

    stem_sets = []
    file_maps = []
    for d in in_dirs:
        files = list(d.glob("*.cluster.parquet"))
        m = {f.name[:-len(".cluster.parquet")]: f for f in files}
        stem_sets.append(set(m))
        file_maps.append(m)

    common = sorted(set.intersection(*stem_sets)) if stem_sets else []
    print(f"input dirs : {len(in_dirs)}")
    for d, m in zip(in_dirs, file_maps):
        print(f"  {d}  ({len(m)} stems)")
    print(f"common stems: {len(common)}")
    print(f"output      : {out_dir}")
    print(f"k values    : {k_values}")
    print(f"max cluster size considered: {args.max_cluster_size}\n")

    jobs = []
    for stem in common:
        out = out_dir / f"{stem}.cluster.parquet"
        if args.force or not out.exists():
            files = [fm[stem] for fm in file_maps]
            jobs.append((stem, files, k_values, args.max_cluster_size, out))

    print(f"to do: {len(jobs)}")
    # Cap automatic worker count to 8 so 24-thread Xeons don't saturate 2400MHz memory bandwidth
    default_workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    workers = args.workers or default_workers
    print(f"workers: {workers}\n")

    errors = []
    if workers == 1:
        it = (process_one(j) for j in jobs)
        for r in tqdm(it, total=len(jobs), desc="consensus", disable=args.no_progress):
            if r[0] == "err":
                errors.append(r)
    else:
        # max_tasks_per_child=50 (Python 3.11+) recycles worker processes to prevent OS heap fragmentation
        ex_kwargs = {"max_workers": workers}
        if sys.version_info >= (3, 11):
            ex_kwargs["max_tasks_per_child"] = 50

        with cf.ProcessPoolExecutor(**ex_kwargs) as ex:
            for r in tqdm(ex.map(process_one, jobs, chunksize=1), total=len(jobs),
                          desc="consensus", disable=args.no_progress):
                if r[0] == "err":
                    errors.append(r)

    if errors:
        print(f"\n{len(errors)} error(s):")
        for r in errors[:5]:
            print(f"  {r[1]}: {r[2].splitlines()[0]}")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()