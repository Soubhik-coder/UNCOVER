#!/usr/bin/env python3
"""
CoveringAttack.py — Functional set-cover attack on GateLock-locked variants.

Threat model
------------
Standard literature assumption (SAIL, GNNUnlock, OMLA, KOLA, ...):
the attacker has done the standard pre-step of identifying which nets
are key inputs, data inputs, and locked-sub-network outputs. This is
a one-time physical / structural reverse-engineering step. In our
evaluation harness this information is read from the sidecar's
`Y_bit`, `A_bit`, `B_bit`, `K_bits` fields.

The attacker does NOT use:
  - variant_partition       (which partition class)
  - valid_key_sequences     (designer's covering)
  - variant_id              (designer's labelling)

Attack
------
For each locked variant in the sidecar, extract the variant's internal
cells via BFS upstream from Y_bit, stopping at A_bit, B_bit, or any
K_bit. Build the local truth table by exhaustive simulation over
2^(n_data+k) (input, key) pairs. Then for each input pattern compute
valid_keys(input) = { K : locked(input, K) == original_gate(input) }.

Solve the per-cluster set-cover problem under one of four policies:
  - table              : return the raw valid_keys table
  - first              : greedy left-to-right covering
  - minimal-partitions : minimum #partitions; tie-break by smallest
                         total key Hamming weight
  - designer           : (evaluator-side only) check whether the
                         designer's covering is in the table

Memoization
-----------
Two variants with the same Boolean function (canonicalised via WL graph
hash of their typed extracted subgraph) share a truth table. We compute
the table once per signature; per-variant work is only the abstract->
concrete key-bit lookup. Speedup is typically 100x-1000x on real
datasets where only a few dozen distinct templates exist.

Dependencies:
    pip install pandas networkx tqdm pyarrow
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm


ORIGINAL_GATE = {
    "$_AND_":  lambda a, b: a & b,
    "$_NAND_": lambda a, b: 1 - (a & b),
    "$_OR_":   lambda a, b: a | b,
    "$_NOR_":  lambda a, b: 1 - (a | b),
    "$_XOR_":  lambda a, b: a ^ b,
    "$_XNOR_": lambda a, b: 1 - (a ^ b),
    "$_NOT_":  lambda a:    1 - a,
    "$_BUF_":  lambda a:    a,
}

CELL_FN = ORIGINAL_GATE   # same set; named separately for readability


# --------------------------------------------------------------------
# Cached-graph helpers: bit <-> cell/port mapping
# --------------------------------------------------------------------

def _build_bit_indices(g: nx.DiGraph) -> tuple[dict, dict, dict]:
    """
    Computes three structural lookup maps from a cached networkx graph.

    Args:
        g (nx.DiGraph): The networkx directed graph representing the netlist.

    Returns:
        tuple[dict, dict, dict]: 
            - bit_to_cell: driver bit id -> cell node name
            - port_bit_to_port_node: bit id of a port-bit -> the PI/KEY/PO node it corresponds to
            - bit_to_kind: bit id -> 'cell' | 'PI' | 'KEY' | 'PO' | 'CONST'
    """
    bit_to_cell = {}
    port_bit_to_port_node = {}
    bit_to_kind = {}
    for n, attrs in g.nodes(data=True):
        kind = attrs.get("kind")
        b = attrs.get("bit")
        if kind == "cell":
            # cell's output bit is the 'bit' attribute on every fan-out edge
            for s in g.successors(n):
                ed = g.get_edge_data(n, s)
                if ed and "bit" in ed:
                    bit_to_cell[ed["bit"]] = n
                    bit_to_kind[ed["bit"]] = "cell"
                    break
        elif kind in ("PI", "KEY", "PO") and isinstance(b, int):
            port_bit_to_port_node[b] = n
            bit_to_kind[b] = kind
    return bit_to_cell, port_bit_to_port_node, bit_to_kind


# --------------------------------------------------------------------
# Sidecar-driven cluster extraction
# --------------------------------------------------------------------

def extract_variant_cluster(g: nx.DiGraph, y_bit: int, data_bits: list, k_bits: list,
                            bit_to_cell: dict, port_bit_to_port_node: dict,
                            other_y_bits: set = None, max_size: int = 256) -> dict:
    """
    Walks upstream from y_bit's driver cell, stopping at any boundary net to isolate the variant.

    Args:
        g (nx.DiGraph): The networkx graph.
        y_bit (int): The target output bit of the locked sub-network.
        data_bits (list): The list of data input bits.
        k_bits (list): The list of key input bits.
        bit_to_cell (dict): Lookup table mapping bit ID to cell node.
        port_bit_to_port_node (dict): Lookup table mapping port bits to port nodes.
        other_y_bits (set, optional): Set of other known Y outputs to prevent crossing boundaries. Defaults to None.
        max_size (int, optional): Size threshold to prevent infinite runaway BFS. Defaults to 256.

    Returns:
        dict: A cluster extraction dictionary containing:
            - 'y_node': cell name of Y_bit's driver
            - 'cluster_cells': list of cell-node names inside the variant
            - 'data_edges': list of (src, dst, dst_port, bit) boundary edges
            - 'key_edges': list of (src, dst, dst_port, bit) boundary edges
            - 'reason': 'ok' or string explanation if extraction failed.
    """
    y_cell = bit_to_cell.get(y_bit)
    if y_cell is None:
        return {"reason": f"y_bit_{y_bit}_has_no_driver_cell"}

    stop_bits = set(data_bits) | set(k_bits)
    other_y_bits = other_y_bits or set()
    # Also stop at any wire that is another variant's Y output -- we
    # don't cross variant Y-cuts.
    stop_bits |= (other_y_bits - {y_bit})

    # BFS upstream over cells, stopping at any incoming bit in stop_bits.
    cluster = {y_cell}
    frontier = [y_cell]
    data_edges = []
    key_edges = []
    while frontier and len(cluster) < max_size:
        n = frontier.pop()
        # Inspect each incoming edge of n
        for p in g.predecessors(n):
            ed = g.get_edge_data(p, n) or {}
            edge_bit = ed.get("bit")
            dst_port = ed.get("dst_port", "?")
            if edge_bit in stop_bits:
                # This is a variant boundary: don't include p in cluster
                if edge_bit in set(data_bits):
                    data_edges.append((p, n, dst_port, edge_bit))
                else:
                    key_edges.append((p, n, dst_port, edge_bit))
                continue
            p_kind = g.nodes[p].get("kind")
            if p_kind in ("PI", "KEY", "CONST", "PO"):
                # Hit a non-stop primary boundary (shouldn't happen for
                # a well-formed locked sub-network but stays defensive).
                # If it's a KEY not in our k_bits list, the sidecar
                # appears inconsistent with the netlist -- record and
                # continue.
                continue
            if p in cluster:
                continue
            cluster.add(p)
            frontier.append(p)

    if len(cluster) >= max_size:
        return {"reason": f"cluster_exceeded_max_size_{max_size}"}

    # Sanity: every data_bit and k_bit must have contributed at least one
    # boundary edge. If not, the sidecar's claim about which bits feed
    # this variant disagrees with the netlist.
    seen_data = {b for (_, _, _, b) in data_edges}
    seen_key  = {b for (_, _, _, b) in key_edges}
    missing_data = set(data_bits) - seen_data
    missing_key  = set(k_bits)    - seen_key
    if missing_data:
        return {"reason": f"data_bits_not_used:{sorted(missing_data)[:3]}"}
    if missing_key:
        return {"reason": f"key_bits_not_used:{sorted(missing_key)[:3]}"}

    return {
        "y_node":         y_cell,
        "cluster_cells":  sorted(cluster),
        "data_edges":     data_edges,
        "key_edges":      key_edges,
        "reason":         "ok",
    }


# --------------------------------------------------------------------
# Local simulation of the variant sub-network
# --------------------------------------------------------------------

def simulate_variant(g: nx.DiGraph, cluster_cells: list, y_node: str, bit_value: dict) -> int | None:
    """
    Evaluates the variant's Y output given specific boundary bit states.

    Args:
        g (nx.DiGraph): The networkx graph.
        cluster_cells (list): List of cells belonging to this specific cluster.
        y_node (str): The root output node of the variant.
        bit_value (dict): Dictionary mapping boundary bit ids (data and key) to 0/1.

    Returns:
        int | None: The computed Boolean integer output (0/1), or None if simulation fails due to inconsistency.
    """
    cell_val = {}
    cluster_set = set(cluster_cells)
    topo = list(nx.topological_sort(g.subgraph(cluster_cells)))
    for n in topo:
        ct = g.nodes[n].get("cell_type", "")
        if ct not in CELL_FN:
            return None
        port_vals = {}
        for p in g.predecessors(n):
            ed = g.get_edge_data(p, n) or {}
            port = ed.get("dst_port", "?")
            edge_bit = ed.get("bit")
            p_kind = g.nodes[p].get("kind")
            if p in cell_val:
                port_vals[port] = cell_val[p]
            elif edge_bit in bit_value:
                port_vals[port] = bit_value[edge_bit]
            elif p_kind == "CONST":
                # Yosys-JSON constant: port attribute is the literal '0'/'1'
                lit = g.nodes[p].get("port", "x")
                if lit in ("0", "1"):
                    port_vals[port] = int(lit)
                else:
                    return None
            else:
                # Predecessor missing from value sources -- inconsistent
                return None
        if ct in ("$_NOT_", "$_BUF_"):
            a = port_vals.get("A")
            if a is None:
                return None
            cell_val[n] = CELL_FN[ct](a)
        else:
            a, b = port_vals.get("A"), port_vals.get("B")
            # Workaround for DiGraph edge-collapse: if A or B is missing
            # but the other is present, both ports must have shared the
            # same source bit (e.g. OR(x,x) buffer hack in GateLock.py).
            if a is None and b is not None:
                a = b
            elif b is None and a is not None:
                b = a
            if a is None:
                return None
            cell_val[n] = CELL_FN[ct](a, b)
    return cell_val.get(y_node)


# --------------------------------------------------------------------
# Truth table + set-cover policies
# --------------------------------------------------------------------

def compute_truth_table(g: nx.DiGraph, cluster_cells: list, y_node: str,
                        data_bits: list, k_bits: list, gate_type: str) -> dict:
    """
    Simulates all possible data and key combinations to build the localized truth table.

    Args:
        g (nx.DiGraph): The networkx graph.
        cluster_cells (list): Internal cells of the variant.
        y_node (str): The root output node.
        data_bits (list): Ordered data bits.
        k_bits (list): Ordered key bits.
        gate_type (str): The hypothesized original un-locked gate type.

    Returns:
        dict: A mapping of (input_tuple) -> set(valid_key_tuples) where outputs match the original gate.
    """
    if gate_type == "$_NOT_":
        gate_fn = lambda inputs: ORIGINAL_GATE[gate_type](inputs[0])
    else:
        gate_fn = lambda inputs: ORIGINAL_GATE[gate_type](inputs[0], inputs[1])

    n_data = len(data_bits)
    n_key  = len(k_bits)

    table = {}
    for input_assign in itertools.product([0, 1], repeat=n_data):
        target = gate_fn(input_assign)
        valid_for_input = set()
        for key_assign in itertools.product([0, 1], repeat=n_key):
            bit_value = dict(zip(data_bits, input_assign))
            bit_value.update(zip(k_bits, key_assign))
            out = simulate_variant(g, cluster_cells, y_node, bit_value)
            if out == target:
                valid_for_input.add(key_assign)
        table[input_assign] = valid_for_input
    return table


def policy_first(table: dict) -> list | None:
    """
    Computes a greedy left-to-right covering of the truth table.

    Args:
        table (dict): The truth table.

    Returns:
        list | None: A covering assignment sequence, or None if no covering exists.
    """
    covering = []
    for I in sorted(table):
        valid = table[I]
        if not valid:
            return None
        K = min(valid)
        merged = False
        for i, (Iset, Kc) in enumerate(covering):
            if Kc == K:
                covering[i] = (Iset + (I,), Kc)
                merged = True
                break
        if not merged:
            covering.append(((I,), K))
    return covering


def policy_minimal_partitions(table: dict) -> list | None:
    """
    Computes a set-cover minimizing total partitions, tie-breaking by lowest Hamming weight.

    Args:
        table (dict): The generated truth table.

    Returns:
        list | None: The minimal set-cover key assignment sequence, or None if un-coverable.
    """
    inputs = sorted(table)
    if any(not table[I] for I in inputs):
        return None

    all_keys = set()
    for I in inputs:
        all_keys |= table[I]

    key_cover = {K: frozenset(I for I in inputs if K in table[I])
                 for K in all_keys}
    inputs_set = frozenset(inputs)

    best_size = len(inputs) + 1
    best_solutions = []

    def search(remaining, chosen_keys):
        nonlocal best_size, best_solutions
        if not remaining:
            n = len(chosen_keys)
            if n < best_size:
                best_size = n
                best_solutions = [list(chosen_keys)]
            elif n == best_size:
                best_solutions.append(list(chosen_keys))
            return
        if len(chosen_keys) + 1 > best_size:
            return
        I = next(iter(sorted(remaining)))
        for K in sorted(table[I]):
            search(remaining - key_cover[K], chosen_keys + [(I, K)])

    search(inputs_set, [])
    if not best_solutions:
        return None

    coverings = []
    for sol in best_solutions:
        chosen = [K for (_, K) in sol]
        cov_map = {K: [] for K in chosen}
        for I in inputs:
            for K in chosen:
                if K in table[I]:
                    cov_map[K].append(I)
                    break
        cov_tuples = sorted([(tuple(sorted(Iset)), K)
                             for K, Iset in cov_map.items() if Iset],
                            key=lambda x: (x[1], x[0]))
        coverings.append(cov_tuples)

    coverings.sort(key=lambda cov: (sum(sum(K) for (_, K) in cov), cov))
    return coverings[0]


# --------------------------------------------------------------------
# Variant signature for memoisation
# --------------------------------------------------------------------

def variant_signature(g: nx.DiGraph, cluster_cells: list, y_node: str,
                      data_edges: list, key_edges: list, gate_type: str) -> str:
    """
    Computes a canonical WL graph hash over the variant's typed extracted subgraph.
    
    Used for extreme speedup via memoization: two variants with the exact same structural 
    signature are guaranteed to have identical Boolean truth tables.

    Args:
        g (nx.DiGraph): The main networkx graph.
        cluster_cells (list): Nodes within the variant.
        y_node (str): The root output node.
        data_edges (list): Boundary data input edges.
        key_edges (list): Boundary key input edges.
        gate_type (str): Assumed target un-locked gate.

    Returns:
        str: A unique signature string representing the topological function.
    """
    sub = nx.DiGraph()
    for n in cluster_cells:
        sub.add_node(n, label="cell:" + g.nodes[n].get("cell_type", "?"))
    sub.nodes[y_node]["label"] = "ROOT:" + sub.nodes[y_node]["label"]
    # internal edges
    for (u, v, ed) in g.subgraph(cluster_cells).edges(data=True):
        sub.add_edge(u, v, label="in:" + ed.get("dst_port", "?"))
    # boundary leaves: distinct names per (kind, bit) so that two edges
    # sharing the same data bit share the same leaf
    boundary_leaf_name = {}
    def leaf_for(kind, bit):
        key = (kind, bit)
        if key not in boundary_leaf_name:
            boundary_leaf_name[key] = f"_{kind}_{len(boundary_leaf_name)}"
            sub.add_node(boundary_leaf_name[key], label=kind)
        return boundary_leaf_name[key]
    for (p, n, port, bit) in data_edges:
        leaf = leaf_for("DATA", bit)
        sub.add_edge(leaf, n, label="in:" + port)
    for (p, n, port, bit) in key_edges:
        leaf = leaf_for("KEY", bit)
        sub.add_edge(leaf, n, label="in:" + port)

    h = nx.weisfeiler_lehman_graph_hash(
        sub.to_undirected(), node_attr="label", edge_attr="label",
        iterations=3, digest_size=12)
    return f"{gate_type}|n={len(set(b for _,_,_,b in data_edges))}|" \
           f"k={len(set(b for _,_,_,b in key_edges))}|{h}"


# --------------------------------------------------------------------
# Per-graph worker
# --------------------------------------------------------------------

def process_one(args: tuple) -> tuple:
    """
    Multiprocessing worker job to evaluate and cover extract sub-networks from a single graph pickle.

    Args:
        args (tuple): (pkl_path, sidecar_path, policy, max_size, out_pq_path)

    Returns:
        tuple: Status flag, stem identifier, and execution statistics.
    """
    (pkl_path, sidecar_path, policy, max_size, out_pq_path) = args
    try:
        with open(pkl_path, "rb") as f:
            rec = pickle.load(f)
        g = rec["graph"]
        sidecar = json.loads(Path(sidecar_path).read_text())
        bit_to_cell, port_bit_to_port_node, _ = _build_bit_indices(g)

        sig_cache = {}   # signature -> {table, covering_first, covering_minimal}

        rows = []
        all_y_bits = {e["Y_bit"] for e in sidecar.get("locked_gates", [])}
        for entry in sidecar.get("locked_gates", []):
            y_bit = entry["Y_bit"]
            a_bit = entry["A_bit"]
            b_bit = entry.get("B_bit")
            k_bits = list(entry.get("K_bits", []))
            data_bits = [a_bit] if b_bit is None else [a_bit, b_bit]

            y_cell = bit_to_cell.get(y_bit)
            base = {
                "y_bit":         y_bit,
                "y_cell":        y_cell,
                "gate_type":     None,    # to be inferred
                "n_data":        len(data_bits),
                "k":             len(k_bits),
                "data_bits":     json.dumps(data_bits),
                "k_bits":        json.dumps(k_bits),
                "ok":            False,
                "reason":        "init",
                "n_candidates":  0,
                "table":         None,
                "covering":      None,
                "signature":     None,
            }
            if y_cell is None:
                base["reason"] = f"y_bit_{y_bit}_no_driver"
                rows.append(base)
                continue

            extract = extract_variant_cluster(
                g, y_bit, data_bits, k_bits,
                bit_to_cell, port_bit_to_port_node,
                other_y_bits=all_y_bits,
                max_size=max_size)
            if extract.get("reason") != "ok":
                base["reason"] = extract["reason"]
                rows.append(base)
                continue

            cluster_cells = extract["cluster_cells"]
            data_edges = extract["data_edges"]
            key_edges = extract["key_edges"]

            # Try every plausible original gate type for this arity.
            # An attacker who knows n_data infers the candidate set from
            # the standard gate library, then picks the one whose truth
            # table is fully coverable.
            if len(data_bits) == 1:
                candidate_gate_types = ["$_NOT_", "$_BUF_"]
            else:
                candidate_gate_types = ["$_AND_", "$_NAND_", "$_OR_",
                                        "$_NOR_", "$_XOR_", "$_XNOR_"]

            # Compute the table once (independent of original_gate),
            # by recording for each (input, key) pair what the locked
            # function OUTPUTS, then for each candidate original gate
            # type derive valid_keys as the keys whose output matches.
            # This is more efficient than re-simulating per gate type.
            raw_outputs = {}
            for input_assign in itertools.product([0,1], repeat=len(data_bits)):
                row = {}
                for key_assign in itertools.product([0,1], repeat=len(k_bits)):
                    bit_value = dict(zip(data_bits, input_assign))
                    bit_value.update(zip(k_bits, key_assign))
                    out = simulate_variant(g, cluster_cells, y_cell, bit_value)
                    row[key_assign] = out
                raw_outputs[input_assign] = row

            best_table = None
            best_gate = None
            best_n_cand = 0
            for gt in candidate_gate_types:
                if gt == "$_NOT_":
                    target_fn = lambda inp: ORIGINAL_GATE[gt](inp[0])
                elif gt == "$_BUF_":
                    target_fn = lambda inp: ORIGINAL_GATE[gt](inp[0])
                else:
                    target_fn = lambda inp, _gt=gt: ORIGINAL_GATE[_gt](inp[0], inp[1])
                table_g = {}
                ok_table = True
                for input_assign in raw_outputs:
                    target = target_fn(input_assign)
                    valid = {k for k, out in raw_outputs[input_assign].items()
                             if out == target}
                    if not valid:
                        ok_table = False
                        break
                    table_g[input_assign] = valid
                if ok_table:
                    n_cand = 1
                    for ks in table_g.values():
                        n_cand *= len(ks)
                    if best_table is None or n_cand > best_n_cand:
                        # Prefer the gate type that gives the most valid
                        # coverings (looser fit suggests this is the
                        # actual original gate)
                        best_table = table_g
                        best_gate = gt
                        best_n_cand = n_cand

            if best_table is None:
                base["reason"] = "no_compatible_gate_type"
                rows.append(base)
                continue

            gate_type = best_gate
            base["gate_type"] = gate_type

            sig = variant_signature(g, cluster_cells, y_cell,
                                    data_edges, key_edges, gate_type)
            if sig not in sig_cache:
                sig_cache[sig] = {
                    "table": best_table,
                    "n_candidates": best_n_cand,
                    "covering_first": policy_first(best_table),
                    "covering_minimal": policy_minimal_partitions(best_table),
                }
            cached = sig_cache[sig]

            if policy == "table":
                cov = None
            elif policy == "first":
                cov = cached["covering_first"]
            elif policy == "minimal-partitions":
                cov = cached["covering_minimal"]
            else:
                cov = None

            table_serialized = {
                ",".join(str(b) for b in I): [list(K) for K in sorted(ks)]
                for I, ks in cached["table"].items()
            }
            covering_serialized = (
                [{"input_set": [list(I) for I in Iset], "key": list(K)}
                 for (Iset, K) in cov] if cov else None
            )

            base.update({
                "ok":            True,
                "reason":        "ok",
                "n_cluster_cells": len(cluster_cells),
                "n_candidates":  cached["n_candidates"],
                "table":         json.dumps(table_serialized),
                "covering":      json.dumps(covering_serialized),
                "signature":     sig,
            })
            rows.append(base)

        df = pd.DataFrame(rows)
        df.to_parquet(out_pq_path)
        return ("ok", rec["meta"]["stem"], {
            "n_variants":           len(rows),
            "n_ok":                 int(df["ok"].sum()) if "ok" in df.columns else 0,
            "n_distinct_signatures": len(sig_cache),
        })
    except Exception as e:
        return ("err", str(pkl_path),
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}", {})


# --------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------

def main():
    """Parses CLI arguments, orchestrates graph set-covering extractions, and generates analytical parquets."""
    ap = argparse.ArgumentParser(
        description="Functional set-cover attack on GateLock variants.")
    ap.add_argument("--cache-dir", required=True,
                    help="Directory containing the serialized graph pickles from BuildGraphs.py.")
    ap.add_argument("--sidecar-dir", required=True,
                    help="Directory containing the original GateLock *.key.json sidecar files.")
    ap.add_argument("--out-dir", required=True,
                    help="Target destination directory for generated covering .parquet and summary files.")
    ap.add_argument("--match-policy",
                    choices=["table", "first", "minimal-partitions"],
                    default="table",
                    help="Set-cover extraction logic. 'table' returns raw valid keys, 'first' covers left-to-right, "
                         "'minimal-partitions' minimizes covering sets.")
    ap.add_argument("--max-cluster-size", type=int, default=256,
                    help="Cap on extracted-cluster cell count to prevent runaway BFS (default 256).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of concurrent multiprocessing workers.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N graphs (useful for rapid smoke tests).")
    ap.add_argument("--force", action="store_true",
                    help="Forces complete recreation and overwriting of output parquets.")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable TQDM progress bar output.")
    args = ap.parse_args()

    cache_dir   = Path(args.cache_dir).resolve()
    sidecar_dir = Path(args.sidecar_dir).resolve()
    out_dir     = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(cache_dir / "manifest.parquet")
    locked = manifest[manifest["kind"] == "locked"]

    jobs = []
    for _, row in locked.iterrows():
        stem = row["stem"]
        pkl = cache_dir / "locked" / row["circuit"] / f"{stem}.pkl"
        if not pkl.exists():
            stored = Path(row["path"])
            pkl = stored if stored.is_absolute() else (cache_dir / stored)
        sd_path = sidecar_dir / f"{stem}.key.json"
        if not sd_path.exists():
            continue
        out_pq = out_dir / f"{stem}.covering.parquet"
        if args.force or not out_pq.exists():
            jobs.append((pkl, sd_path, args.match_policy,
                         args.max_cluster_size, out_pq))
    if args.limit:
        jobs = jobs[:args.limit]

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"cache:    {cache_dir}")
    print(f"sidecar:  {sidecar_dir}")
    print(f"out:      {out_dir}")
    print(f"policy:   {args.match_policy}")
    print(f"max size: {args.max_cluster_size}")
    print(f"to do:    {len(jobs)}")
    print(f"workers:  {workers}\n")

    rows = []
    errors = []
    if workers == 1:
        it = (process_one(j) for j in jobs)
    else:
        ex = cf.ProcessPoolExecutor(max_workers=workers)
        it = ex.map(process_one, jobs)
    for r in tqdm(it, total=len(jobs), desc="covering",
                  disable=args.no_progress):
        if r[0] == "err":
            errors.append(r)
        else:
            rows.append({"stem": r[1], **r[2]})
    if workers != 1:
        ex.shutdown()

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "summary.csv", index=False)
        print(f"\nsummary: {out_dir/'summary.csv'}  ({len(df)} rows)")
        print(df.describe(include="all").to_string())

    if errors:
        from collections import Counter
        print(f"\n{len(errors)} error(s).")
        bucket = Counter(r[2].splitlines()[0] for r in errors)
        for msg, cnt in bucket.most_common(5):
            print(f"  {cnt:>4}  {msg}")
        print("first traceback:")
        print(errors[0][2])
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()