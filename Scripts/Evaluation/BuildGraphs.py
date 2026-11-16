#!/usr/bin/env python3
"""
BuildGraphs.py — Convert GateLock-locked netlists (and their unlocked
                 originals) into cached networkx graphs ready for
                 downstream unsupervised analysis.

Inputs
------
    --locked-dir   : BatchLock.py output dir; expects subdirs
                       netlists/   *.json
                       keys/       *.key.json
    --original-dir : folder with the un-locked ISCAS-85 *.json files
                     (must contain one .json per circuit named <circuit>.json)

Outputs (under --cache-dir, default = "cache/graphs")
-------
    locked/<circuit>/<stem>.pkl        per locked netlist
    original/<circuit>.pkl             per unlocked circuit
    manifest.parquet                   pandas DataFrame of every cached graph

Each .pkl contains:
    {
      "graph":       networkx.DiGraph,
      "meta":        dict (circuit, pct, seed, key_port, variant_policy, ...),
      "labels":      dict   node_id -> int           # 0 = unlocked,
                                                     # 1..K = variant id
                                                     # (None for original graphs)
      "variant_id":  dict   variant_partition_str -> int  (only for locked)
    }

Graph schema
------------
Node id      : string, identical to the Yosys cell name OR port name
Node attrs   :
    kind        : "cell" | "PI" | "KEY" | "PO" | "CONST"
    cell_type   : "$_AND_" | "$_OR_" | "$_NOT_" | "$_XOR_" | "" (non-cells)
    bit         : int (Yosys bit id for ports/consts), -1 for cells
Edge attrs   :
    src_port    : output port of the source cell (for PI/KEY/PO/CONST nodes
                  the src_port is the empty string)
    dst_port    : input port name on the destination cell ("A", "B", ...)

The graph is a DAG of FAN-OUT edges:  src=driver  ->  dst=consumer

Determinism
-----------
JSON IO is deterministic, no RNG is used, and we sort children when
building a node's predecessor/successor lists implicitly through the
JSON file order. Re-running the script overwrites the cache with
byte-identical .pkl files (we test this in the smoke test).

Dependencies
------------
    pip install pandas networkx tqdm pyarrow
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pickle
import re
import sys
import traceback
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm


FILENAME_RE = re.compile(r"^(?P<circuit>.+?)_p(?P<pct>[0-9p]+)_s(?P<seed>\d+)$")
BOOKKEEPING = {"seeds", "runs", "verification_failures"}


def parse_stem(stem: str) -> tuple[str, float, int] | None:
    """
    Parses a locked netlist filename stem to extract execution parameters.

    Args:
        stem (str): Filename base string (e.g., "c432_p10_s0").

    Returns:
        tuple[str, float, int] | None: (circuit, percentage, seed) if match is successful, 
                                       otherwise None.
    """
    m = FILENAME_RE.match(stem)
    if not m:
        return None
    pct = float(m["pct"].replace("p", "."))
    return m["circuit"], pct, int(m["seed"])


def _strip_key_suffix(name: str) -> str:
    """
    Strips the standard sidecar suffix from a filename to recover the raw stem.

    Args:
        name (str): Full filename of the key map file.

    Returns:
        str: Filename base token without the sidecar extension.
    """
    suf = ".key.json"
    return name[:-len(suf)] if name.endswith(suf) else name


# --------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------

def _bit_to_port_name(ports: dict) -> dict[int, str]:
    """
    Maps structural individual bit IDs to their parent port identifier names.

    Args:
        ports (dict): Dictionary of ports found in the Yosys JSON module.

    Returns:
        dict[int, str]: Lookup dictionary of bit_id -> port_name.
    """
    out = {}
    for name, p in ports.items():
        for b in p["bits"]:
            out[b] = name
    return out


def build_graph(json_data: dict, key_port: str | None = None) -> tuple[nx.DiGraph, dict]:
    """
    Builds a NetworkX directed graph from a Yosys-style JSON module description.

    Translates hardware nets, primary I/O, constants, and logical cells into structured 
    nodes and connectivity edges, modeling the design as a data-flow DAG.

    Args:
        json_data (dict): Parsed Yosys netlist design JSON.
        key_port (str | None, optional): Name of the top-level key input port bus, if any.

    Returns:
        tuple[nx.DiGraph, dict]: A 2-tuple containing:
            - nx.DiGraph: The constructed gate-level graph.
            - dict: Collected structural metadata metrics (cell counts, ports counts).
    """
    mod_name, mod = next(iter(json_data["modules"].items()))
    cells = mod["cells"]
    ports = mod.get("ports", {})

    # ------ classify each port-bit as PI / KEY / PO --------------------
    pi_bits, key_bits, po_bits = set(), set(), set()
    for pname, p in ports.items():
        d = p["direction"]
        for b in p["bits"]:
            if d == "input":
                (key_bits if pname == key_port else pi_bits).add(b)
            elif d == "output":
                po_bits.add(b)

    # ------ map every bit to the cell that drives it -------------------
    driver_of_bit: dict = {}
    cell_out_ports: dict = {}   # cell_name -> list of (port_name, bit)
    for cname, c in cells.items():
        outs = []
        for p, dir_ in c["port_directions"].items():
            if dir_ == "output":
                for b in c["connections"][p]:
                    outs.append((p, b))
                    if isinstance(b, int):
                        driver_of_bit[b] = (cname, p)
        cell_out_ports[cname] = outs

    # ------ build the graph --------------------------------------------
    g = nx.DiGraph()

    # PI / KEY / PO nodes: name them "<port>[<bit>]" so multi-bit ports
    # become multiple nodes with predictable, deterministic names.
    pi_nodes, key_nodes, po_nodes = set(), set(), set()
    bit_to_port_node: dict = {}    # bit id -> graph node name (PI/KEY only)

    def make_port_node(pname, bit, kind, target):
        node_id = f"{pname}[{bit}]" if len(ports[pname]["bits"]) > 1 \
            else pname
        g.add_node(node_id, kind=kind, cell_type="", bit=bit, port=pname)
        target.add(node_id)
        if kind in ("PI", "KEY"):
            bit_to_port_node[bit] = node_id
        return node_id

    for pname, p in ports.items():
        d = p["direction"]
        for b in p["bits"]:
            if d == "input":
                kind = "KEY" if pname == key_port else "PI"
                make_port_node(pname, b, kind,
                               key_nodes if kind == "KEY" else pi_nodes)

    # Cell nodes
    for cname, c in cells.items():
        g.add_node(cname, kind="cell", cell_type=c["type"], bit=-1, port="")

    # PO sinks (separate nodes so we can tell where outputs leave the design)
    po_node_of_bit: dict = {}
    for pname, p in ports.items():
        if p["direction"] != "output":
            continue
        for b in p["bits"]:
            po_node = make_port_node(pname, b, "PO", po_nodes)
            po_node_of_bit[b] = po_node

    # CONST nodes (the Yosys-JSON strings "0" / "1")
    const_nodes_added = set()

    def get_const_node(bit_token):
        node_id = f"const_{bit_token}"
        if node_id not in const_nodes_added:
            g.add_node(node_id, kind="CONST", cell_type="",
                       bit=-1, port=bit_token)
            const_nodes_added.add(node_id)
        return node_id

    # ------ edges -------------------------------------------------------
    # For each cell input bit, draw an edge from its driver (cell / PI /
    # KEY / CONST) to this cell.
    for cname, c in cells.items():
        for p, dir_ in c["port_directions"].items():
            if dir_ != "input":
                continue
            for b in c["connections"][p]:
                if isinstance(b, int):
                    if b in driver_of_bit:
                        src_cell, src_port = driver_of_bit[b]
                        g.add_edge(src_cell, cname,
                                   src_port=src_port, dst_port=p, bit=b)
                    elif b in bit_to_port_node:
                        g.add_edge(bit_to_port_node[b], cname,
                                   src_port="", dst_port=p, bit=b)
                    else:
                        # undriven bit — treat as const-x
                        cn = get_const_node(f"x{b}")
                        g.add_edge(cn, cname, src_port="", dst_port=p, bit=b)
                else:
                    cn = get_const_node(str(b))
                    g.add_edge(cn, cname, src_port="", dst_port=p, bit=b)

    # PO-sink edges: connect the driver of each PO bit to its PO node
    for b, po_node in po_node_of_bit.items():
        if b in driver_of_bit:
            src_cell, src_port = driver_of_bit[b]
            g.add_edge(src_cell, po_node, src_port=src_port,
                       dst_port="", bit=b)
        elif b in bit_to_port_node:
            # a PO directly tied to a PI/KEY input — unusual but legal
            g.add_edge(bit_to_port_node[b], po_node, src_port="",
                       dst_port="", bit=b)

    meta = {
        "module":   mod_name,
        "n_cells":  len(cells),
        "n_pi":     len(pi_nodes),
        "n_key":    len(key_nodes),
        "n_po":     len(po_nodes),
        "n_const":  len(const_nodes_added),
        "n_nodes":  g.number_of_nodes(),
        "n_edges":  g.number_of_edges(),
    }
    return g, meta


# --------------------------------------------------------------------
# Sidecar -> per-node variant labels
# --------------------------------------------------------------------

def _output_bit_to_cell(json_data: dict) -> dict[int, str]:
    """Maps wire bit IDs back to the gate name driving them for quick structural tracebacks."""
    mod = next(iter(json_data["modules"].values()))
    out = {}
    for cname, c in mod["cells"].items():
        for p, d in c["port_directions"].items():
            if d == "output":
                for b in c["connections"][p]:
                    out[b] = cname
    return out


def _cell_input_bits(cell: dict) -> list[int]:
    """Returns a list of input bit IDs feeding into a cell."""
    return [cell["connections"][p][0]
            for p, d in cell["port_directions"].items() if d == "input"]


def build_labels(json_data: dict, sidecar: dict) -> tuple[dict[str, int], dict[str, int]]:
    """
    Generates training/evaluation ground-truth labels for all cells in the netlist.
    
    Traces backward from each locked gate's output wire to identify exactly which cells 
    were injected as part of a GateLock structural sub-network variant.

    Args:
        json_data (dict): The parsed locked Yosys netlist.
        sidecar (dict): The matching key-map tracking original gate data and variants.

    Returns:
        tuple[dict, dict]: A 2-tuple containing:
            - dict[str, int]: Map of cell_name -> label_id (0 = unlocked background logic, 
                              >=1 = integer index of specific obfuscating architecture).
            - dict[str, int]: Map tracking string configuration descriptions to their labeled IDs.
    """
    mod = next(iter(json_data["modules"].values()))
    cells = mod["cells"]
    bit_to_cell = _output_bit_to_cell(json_data)

    # Map each distinct variant_partition string to a small integer id.
    variant_id: dict = {}
    for entry in sidecar.get("locked_gates", []):
        v = entry["variant_partition"]
        if v not in variant_id:
            variant_id[v] = len(variant_id) + 1

    # Start with 0 for every node we know about (cells only — port/CONST
    # nodes are by definition "unlocked").
    labels = {cname: 0 for cname in cells}

    # For each locked variant, BFS through fan-in starting from Y_bit,
    # stopping at A_bit / B_bit / K_bits boundaries. Every cell visited
    # belongs to that variant.
    for entry in sidecar.get("locked_gates", []):
        vid = variant_id[entry["variant_partition"]]
        stops = {entry["A_bit"]}
        if entry.get("B_bit") is not None:
            stops.add(entry["B_bit"])
        stops |= set(entry.get("K_bits", []))
        seen = set()
        stack = [entry["Y_bit"]]
        while stack:
            b = stack.pop()
            if b in seen or b in stops:
                continue
            seen.add(b)
            cname = bit_to_cell.get(b)
            if cname is None:
                continue
            # Once a cell is claimed by one variant, keep the first.
            if labels.get(cname, 0) == 0:
                labels[cname] = vid
            for cin in _cell_input_bits(cells[cname]):
                if cin not in stops:
                    stack.append(cin)
    return labels, variant_id


# --------------------------------------------------------------------
# Worker functions
# --------------------------------------------------------------------

def process_locked(args: tuple) -> tuple[str, str, dict | str]:
    """
    Multiprocessing job worker to ingest and convert a locked file pairing into a graph.

    Args:
        args (tuple): (stem, netlist_path, key_map_path, output_path)

    Returns:
        tuple: (status_flag, stem_identifier, metadata_dict_or_error_trace)
    """
    stem, net_path, key_path, out_path = args
    try:
        with open(net_path) as f:
            jd = json.load(f)
        with open(key_path) as f:
            sd = json.load(f)
        key_port = sd.get("key_port", "lock_key")
        g, meta = build_graph(jd, key_port=key_port)
        labels, vmap = build_labels(jd, sd)

        circ, pct, seed = parse_stem(stem)
        meta.update({
            "stem":            stem,
            "circuit":         circ,
            "pct":             pct,
            "seed":            seed,
            "key_port":        key_port,
            "variant_policy":  sd.get("variant_policy", "?"),
            "key_width":       sd.get("key_width"),
            "n_locked_gates":  len(sd.get("locked_gates", [])),
            "n_variants":      len(vmap),
            "n_locked_nodes":  sum(1 for v in labels.values() if v != 0),
        })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump({"graph": g, "meta": meta,
                         "labels": labels, "variant_id": vmap},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        return ("ok", stem, meta)
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return ("err", stem, f"{e}\n{tb}")


def process_original(args: tuple) -> tuple[str, str, dict | str]:
    """
    Multiprocessing job worker to convert an original unlocked netlist into a graph.

    Args:
        args (tuple): (circuit_name, netlist_path, output_path)

    Returns:
        tuple: (status_flag, circuit_name, metadata_dict_or_error_trace)
    """
    circ, net_path, out_path = args
    try:
        with open(net_path) as f:
            jd = json.load(f)
        g, meta = build_graph(jd, key_port=None)
        meta.update({
            "circuit":         circ,
            "pct":             0.0,
            "seed":            -1,
            "key_port":        None,
            "variant_policy":  "none",
            "key_width":       0,
            "n_locked_gates":  0,
            "n_variants":      0,
            "n_locked_nodes":  0,
            "stem":            circ,
        })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump({"graph": g, "meta": meta,
                         "labels": None, "variant_id": {}},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        return ("ok", circ, meta)
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return ("err", circ, f"{e}\n{tb}")


# --------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------

def discover_pairs(locked_dir: Path) -> list[tuple[str, Path, Path]]:
    """
    Scans the BatchLock output directories to extract paired json/sidecar sets.

    Args:
        locked_dir (Path): Core BatchLock output directory containing netlists and keys.

    Returns:
        list[tuple]: Discovered file path mappings.
    """
    netlists_dir = locked_dir / "netlists"
    keys_dir     = locked_dir / "keys"
    nets = {p.stem: p for p in netlists_dir.iterdir()
            if p.suffix.lower() == ".json" and p.stem not in BOOKKEEPING}
    keys = {_strip_key_suffix(p.name): p for p in keys_dir.iterdir()
            if p.name.endswith(".key.json")}
    paired = sorted(set(nets) & set(keys))
    return [(stem, nets[stem], keys[stem]) for stem in paired]


def main():
    """Coordinates file ingestion parsing, concurrent graph generation pools, and summary compilation."""
    ap = argparse.ArgumentParser(
        description="Build cached networkx graphs from locked + original ISCAS-85 netlists."
    )
    ap.add_argument("--locked-dir", required=True,
                    help="Path to the primary output folder generated by BatchLock.py containing netlist subdirs.")
    ap.add_argument("--original-dir", required=True,
                    help="Path to the directory housing the un-locked baseline JSON netlists.")
    ap.add_argument("--cache-dir", default="cache/graphs",
                    help="Target destination directory where serialized graph pickles will be cached. Defaults to 'cache/graphs'.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Total concurrent parallel processes to spawn. Defaults to system CPU count minus 1.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Truncate processing after evaluation of N locked structures. Ideal for execution test checks.")
    ap.add_argument("--force", action="store_true",
                    help="Forces complete recreation and serialization execution regardless of existing pickle statuses.")
    ap.add_argument("--no-progress", action="store_true",
                    help="Hides all TQDM activity progress indicators from standard streams.")
    args = ap.parse_args()

    locked_dir   = Path(args.locked_dir).resolve()
    original_dir = Path(args.original_dir).resolve()
    cache_dir    = Path(args.cache_dir).resolve()
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    print(f"locked dir   : {locked_dir}")
    print(f"original dir : {original_dir}")
    print(f"cache dir    : {cache_dir}")
    print(f"workers      : {workers}")

    # ---- originals ----------------------------------------------------
    orig_files = sorted(original_dir.glob("*.json"))
    orig_jobs = []
    for p in orig_files:
        circ = p.stem
        out = cache_dir / "original" / f"{circ}.pkl"
        if args.force or not out.exists():
            orig_jobs.append((circ, p, out))
    print(f"\noriginals to build: {len(orig_jobs)} / {len(orig_files)}")

    # ---- locked -------------------------------------------------------
    pairs = discover_pairs(locked_dir)
    if args.limit is not None:
        pairs = pairs[:args.limit]
    locked_jobs = []
    for stem, np_, kp_ in pairs:
        circ, _, _ = parse_stem(stem)
        out = cache_dir / "locked" / circ / f"{stem}.pkl"
        if args.force or not out.exists():
            locked_jobs.append((stem, np_, kp_, out))
    print(f"locked to build:    {len(locked_jobs)} / {len(pairs)}\n")

    # ---- run ----------------------------------------------------------
    results_orig, results_locked = [], []
    errors = []

    def run(jobs, fn, desc):
        out = []
        if not jobs:
            return out
        if workers == 1:
            it = (fn(j) for j in jobs)
        else:
            ex = cf.ProcessPoolExecutor(max_workers=workers)
            it = ex.map(fn, jobs)
        pbar = tqdm(it, total=len(jobs), desc=desc,
                    disable=args.no_progress)
        for r in pbar:
            out.append(r)
            if r[0] == "err":
                errors.append(r)
        if workers != 1:
            ex.shutdown()
        return out

    results_orig   = run(orig_jobs,   process_original, "originals")
    results_locked = run(locked_jobs, process_locked,   "locked   ")

    # ---- manifest -----------------------------------------------------
    # Always rebuild the manifest from EVERYTHING currently in cache,
    # not just this run, so partial re-runs stay correct.
    manifest_rows = []

    def collect(folder, kind):
        for pkl in folder.rglob("*.pkl"):
            with open(pkl, "rb") as f:
                rec = pickle.load(f)
            m = rec["meta"]
            manifest_rows.append({
                "kind":           kind,
                "circuit":        m["circuit"],
                "pct":            m["pct"],
                "seed":           m["seed"],
                "stem":           m["stem"],
                "n_cells":        m["n_cells"],
                "n_pi":           m["n_pi"],
                "n_key":          m["n_key"],
                "n_po":           m["n_po"],
                "n_const":        m["n_const"],
                "n_nodes":        m["n_nodes"],
                "n_edges":        m["n_edges"],
                "n_locked_nodes": m.get("n_locked_nodes", 0),
                "n_locked_gates": m.get("n_locked_gates", 0),
                "n_variants":     m.get("n_variants", 0),
                "key_width":      m.get("key_width", 0),
                "variant_policy": m.get("variant_policy", "none"),
                "key_port":       m.get("key_port", None),
                "path":           str(pkl.resolve()),
            })

    if (cache_dir / "original").exists():
        collect(cache_dir / "original", "original")
    if (cache_dir / "locked").exists():
        collect(cache_dir / "locked", "locked")

    if manifest_rows:
        df = pd.DataFrame(manifest_rows).sort_values(
            ["kind", "circuit", "pct", "seed"]).reset_index(drop=True)
        man_path = cache_dir / "manifest.parquet"
        df.to_parquet(man_path)
        # also write a human-readable CSV for quick inspection
        df.to_csv(cache_dir / "manifest.csv", index=False)
        print(f"\nmanifest:    {man_path}  ({len(df)} entries)")
        # quick summary
        g = df.groupby("kind").agg(
            n=("stem", "count"),
            tot_nodes=("n_nodes", "sum"),
            tot_edges=("n_edges", "sum"),
        )
        print(g.to_string())
    else:
        print("\n  (manifest empty)")

    if errors:
        print(f"\n  {len(errors)} error(s):")
        for kind, stem, msg in errors[:5]:
            print(f"    [{kind}] {stem}: {msg.splitlines()[0]}")

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()