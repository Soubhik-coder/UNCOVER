#!/usr/bin/env python3
"""
DatasetScan.py — Inspect a directory of GateLock-locked netlists produced
                  by BatchLock.py.

Expected layout:

    <locked-dir>/
        netlists/  *.json
        keys/      *.key.json
        metadata/  (ignored)

Optionally, point --original-dir at the folder containing the un-locked
ISCAS-85 .json files to also report the original cell count per circuit.

Outputs:
    stdout                                    pretty table + histogram
    <out-prefix>.per_file.csv                 one row per locked netlist
    <out-prefix>.per_circuit.csv              one row per circuit
    <out-prefix>.gate_hist.csv                aggregate gate-type counts
    <out-prefix>.summary.json                 misc.: orphans, bad files, etc.

Dependencies:
    pip install pandas tqdm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# --------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------
FILENAME_RE = re.compile(r"^(?P<circuit>.+?)_p(?P<pct>[0-9p]+)_s(?P<seed>\d+)$")
BOOKKEEPING = {"seeds", "runs", "verification_failures"}


def parse_stem(stem: str) -> tuple[str, float, int] | None:
    """
    Parses a locked netlist filename stem to extract circuit metadata.

    Args:
        stem (str): The filename without extension (e.g., "c432_p10_s42").

    Returns:
        tuple | None: A tuple containing (circuit_name, locking_percentage, seed), 
                      or None if the filename doesn't match the expected pattern.
    """
    m = FILENAME_RE.match(stem)
    if not m:
        return None
    pct = float(m["pct"].replace("p", "."))
    return m["circuit"], pct, int(m["seed"])


def _strip_key_suffix(name: str) -> str:
    """
    Removes the '.key.json' suffix from a sidecar filename to match the netlist stem.

    Args:
        name (str): The full sidecar filename.

    Returns:
        str: The base stem of the filename.
    """
    suf = ".key.json"
    return name[:-len(suf)] if name.endswith(suf) else name


# --------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------
def discover(locked_dir: Path) -> tuple[dict, dict, list, list, list]:
    """
    Scans the locked directory and pairs generated netlists with their key sidecars.

    Args:
        locked_dir (Path): The root directory containing 'netlists' and 'keys' subdirectories.

    Returns:
        tuple: A 5-tuple containing:
            - net_files (dict): Map of stem -> Path for all netlists.
            - key_files (dict): Map of stem -> Path for all sidecars.
            - paired (list): Stems that have both a netlist and a sidecar.
            - orphan_nets (list): Stems with a netlist but missing a sidecar.
            - orphan_keys (list): Stems with a sidecar but missing a netlist.

    Raises:
        SystemExit: If the required 'netlists' or 'keys' subdirectories are missing.
    """
    netlists_dir = locked_dir / "netlists"
    keys_dir     = locked_dir / "keys"
    if not netlists_dir.is_dir():
        raise SystemExit(f"missing subdir: {netlists_dir}")
    if not keys_dir.is_dir():
        raise SystemExit(f"missing subdir: {keys_dir}")

    net_files = {p.stem: p for p in netlists_dir.iterdir()
                 if p.suffix.lower() == ".json" and p.stem not in BOOKKEEPING}
    key_files = {_strip_key_suffix(p.name): p
                 for p in keys_dir.iterdir()
                 if p.name.endswith(".key.json")}

    paired      = sorted(set(net_files) & set(key_files))
    orphan_nets = sorted(set(net_files) - set(key_files))
    orphan_keys = sorted(set(key_files) - set(net_files))
    return net_files, key_files, paired, orphan_nets, orphan_keys


# --------------------------------------------------------------------
# Per-file pass (sidecar only — fast even on 1 GB datasets)
# --------------------------------------------------------------------
def scan_sidecars(net_files: dict, key_files: dict, paired: list, progress: bool = True) -> tuple[pd.DataFrame, list]:
    """
    Reads all sidecar JSONs to rapidly extract circuit characteristics without parsing full netlists.

    Args:
        net_files (dict): Mapping of valid netlist files.
        key_files (dict): Mapping of valid sidecar files.
        paired (list): List of successfully paired stems.
        progress (bool, optional): Whether to display a TQDM progress bar. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame containing the metadata for all valid pairs.
            - list: A list of tuples for stems that failed parsing (stem, reason).
    """
    rows, bad = [], []
    it = tqdm(paired, desc="reading keys", disable=not progress)
    for stem in it:
        parsed = parse_stem(stem)
        if parsed is None:
            bad.append((stem, "filename"));  continue
        circ, pct, seed = parsed
        try:
            kd = json.loads(key_files[stem].read_text())
        except Exception as e:
            bad.append((stem, f"key: {e}"));  continue
        rows.append({
            "circuit":        circ,
            "pct":            pct,
            "seed":           seed,
            "key_width":      kd.get("key_width"),
            "n_locked_gates": len(kd.get("locked_gates", [])),
            "variant_policy": kd.get("variant_policy", "?"),
            "stem":           stem,
            "netlist_path":   str(net_files[stem]),
            "key_path":       str(key_files[stem]),
        })
    return pd.DataFrame(rows), bad


# --------------------------------------------------------------------
# Per-circuit pass (read ONE netlist per circuit for cell count / mix)
# --------------------------------------------------------------------
def sample_netlists(df: pd.DataFrame, progress: bool = True) -> tuple[pd.DataFrame, Counter]:
    """
    Opens the lowest-percentage netlist for each circuit to grab cell counts and gate distributions.

    Args:
        df (pd.DataFrame): The DataFrame generated by `scan_sidecars`.
        progress (bool, optional): Whether to show a TQDM progress bar. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Per-circuit sampling data (cell counts, gate types).
            - Counter: An aggregated histogram of all gate types encountered across samples.
    """
    sample_rows = []
    gate_hist_total = Counter()
    circuits = sorted(df["circuit"].unique())
    it = tqdm(circuits, desc="sampling netlists", disable=not progress)
    for circ in it:
        sub = df[df["circuit"] == circ].sort_values(["pct", "seed"])
        path = Path(sub.iloc[0]["netlist_path"])
        try:
            nd = json.loads(path.read_text())
            mod = next(iter(nd["modules"].values()))
            gh = Counter(c["type"] for c in mod["cells"].values())
            sample_rows.append({
                "circuit":                 circ,
                "sample_pct":              float(sub.iloc[0]["pct"]),
                "sample_cells_after_lock": len(mod["cells"]),
                "sample_gate_types":       dict(gh),
            })
            gate_hist_total.update(gh)
        except Exception as e:
            sample_rows.append({
                "circuit":                 circ,
                "sample_pct":              float(sub.iloc[0]["pct"]),
                "sample_cells_after_lock": None,
                "sample_gate_types":       {"_error": str(e)},
            })
    return pd.DataFrame(sample_rows), gate_hist_total


# --------------------------------------------------------------------
# Optional: pull original-cell counts
# --------------------------------------------------------------------
def attach_original_cells(per_circ: pd.DataFrame, original_dir: Path) -> tuple[pd.DataFrame, int]:
    """
    Compares locked netlists against their original, unlocked counterparts to calculate cell bloat.

    Args:
        per_circ (pd.DataFrame): The aggregated per-circuit DataFrame.
        original_dir (Path): The directory containing the original unlocked JSONs.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The updated DataFrame with an 'original_cells' column.
            - int: The number of circuits that were successfully matched and counted.
    """
    counts = {}
    for circ in per_circ["circuit"]:
        cand = original_dir / f"{circ}.json"
        if cand.exists():
            try:
                d = json.loads(cand.read_text())
                mod = next(iter(d["modules"].values()))
                counts[circ] = len(mod["cells"])
            except Exception:
                counts[circ] = None
        else:
            counts[circ] = None
    per_circ["original_cells"] = per_circ["circuit"].map(counts)
    return per_circ, sum(1 for v in counts.values() if v is not None)


# --------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------
def aggregate_per_circuit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the per-file DataFrame to generate circuit-level statistics (min/max/mean metrics).

    Args:
        df (pd.DataFrame): The parsed per-file data.

    Returns:
        pd.DataFrame: A summarized DataFrame where each row represents one distinct circuit.
    """
    g = df.groupby("circuit")
    agg = pd.DataFrame({
        "n_files":          g.size(),
        "pct_min":          g["pct"].min(),
        "pct_max":          g["pct"].max(),
        "n_distinct_pct":   g["pct"].nunique(),
        "n_distinct_seed":  g["seed"].nunique(),
        "key_width_min":    g["key_width"].min(),
        "key_width_max":    g["key_width"].max(),
        "key_width_mean":   g["key_width"].mean().round(2),
        "locked_gates_min": g["n_locked_gates"].min(),
        "locked_gates_max": g["n_locked_gates"].max(),
        "locked_gates_mean":g["n_locked_gates"].mean().round(2),
        "variant_policies": g["variant_policy"].agg(
                              lambda s: "|".join(sorted(set(s)))),
    }).reset_index()
    return agg


# --------------------------------------------------------------------
# Pretty-print
# --------------------------------------------------------------------
def print_report(per_circ: pd.DataFrame,
                 gate_hist_total: Counter,
                 paired: int, n_net: int, n_key: int,
                 orphan_nets: list, orphan_keys: list, bad: list, locked_dir: Path,
                 show_orig: bool):
    """Prints a formatted CLI summary of the scanned dataset."""
    print("=" * 96)
    print(f"  locked dir         : {locked_dir}")
    print(f"  netlists / keys    : {n_net} / {n_key}")
    print(f"  paired             : {paired}")
    if orphan_nets:
        print(f"  orphan netlists    : {len(orphan_nets)} "
              f"(e.g. {orphan_nets[:3]})")
    if orphan_keys:
        print(f"  orphan keys        : {len(orphan_keys)} "
              f"(e.g. {orphan_keys[:3]})")
    if bad:
        print(f"  bad / unreadable   : {len(bad)} (e.g. {bad[:3]})")
    print(f"  circuits           : {len(per_circ)}")
    print("=" * 96)

    cols = ["circuit", "n_files", "pct_min", "pct_max",
            "n_distinct_pct", "n_distinct_seed"]
    if show_orig:
        cols.append("original_cells")
    cols += ["sample_cells_after_lock",
             "key_width_min", "key_width_max", "key_width_mean",
             "locked_gates_min", "locked_gates_max", "locked_gates_mean"]
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 200):
        print()
        print(per_circ[cols].to_string(index=False))
        print()

    if show_orig:
        ratios = (per_circ["sample_cells_after_lock"]
                  / per_circ["original_cells"]).round(2)
        print("  bloat ratio at lowest sampled percentage "
              "(sample_cells_after_lock / original_cells):")
        for c, r in zip(per_circ["circuit"], ratios):
            print(f"     {c:<10}  ×{r}")
        print()

    print("  Aggregate gate-type histogram across per-circuit samples:")
    total = sum(gate_hist_total.values()) or 1
    for t, c in gate_hist_total.most_common():
        print(f"     {t:<14} {c:>10}  ({c/total*100:5.2f}%)")

    pols = sorted(set("|".join(per_circ["variant_policies"]).split("|")))
    print()
    print(f"  variant policies present: {pols}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    """Parses arguments, orchestrates the scan, and writes out final reports."""
    ap = argparse.ArgumentParser(
        description="Inspect a directory of GateLock-locked netlists produced by BatchLock.py."
    )
    
    ap.add_argument("--locked-dir", required=True,
                    help="Path to the directory containing the 'netlists' and 'keys' subdirectories.")
    
    ap.add_argument("--original-dir", default=None,
                    help="Optional path to the directory containing the un-locked ISCAS "
                         "(or similar) .json files. Used to calculate cell bloat ratios.")
    
    ap.add_argument("--out-prefix", default=None,
                    help="Prefix for the generated CSV and JSON report files (e.g., 'results/scan_report'). "
                         "If omitted, output is only printed to the console.")
    
    ap.add_argument("--no-progress", action="store_true",
                    help="Disables the TQDM progress bars during scanning (useful for CI/CD logs).")
    
    args = ap.parse_args()

    locked_dir = Path(args.locked_dir).resolve()
    net_files, key_files, paired, orphan_nets, orphan_keys = discover(locked_dir)

    df, bad = scan_sidecars(net_files, key_files, paired,
                            progress=not args.no_progress)
    sample_df, gate_hist_total = sample_netlists(df,
                                                 progress=not args.no_progress)

    per_circ = aggregate_per_circuit(df).merge(sample_df, on="circuit",
                                               how="left")

    show_orig = False
    if args.original_dir:
        odir = Path(args.original_dir).resolve()
        if not odir.is_dir():
            print(f"  WARNING: --original-dir not found: {odir}", file=sys.stderr)
        else:
            per_circ, found = attach_original_cells(per_circ, odir)
            show_orig = True
            print(f"  matched {found} original circuit(s) in {odir}",
                  file=sys.stderr)

    print_report(per_circ, gate_hist_total,
                 paired=len(paired), n_net=len(net_files), n_key=len(key_files),
                 orphan_nets=orphan_nets, orphan_keys=orphan_keys,
                 bad=bad, locked_dir=locked_dir, show_orig=show_orig)

    if args.out_prefix:
        op = Path(args.out_prefix)
        op.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["netlist_path", "key_path"]).to_csv(
            op.with_suffix(".per_file.csv"), index=False)
        per_circ_out = per_circ.drop(columns=["sample_gate_types"], errors="ignore")
        per_circ_out.to_csv(op.with_suffix(".per_circuit.csv"), index=False)
        pd.DataFrame(
            sorted(gate_hist_total.items(), key=lambda kv: -kv[1]),
            columns=["gate_type", "count"]
        ).to_csv(op.with_suffix(".gate_hist.csv"), index=False)
        with open(op.with_suffix(".summary.json"), "w") as f:
            json.dump({
                "locked_dir":       str(locked_dir),
                "n_netlists":       len(net_files),
                "n_keys":           len(key_files),
                "n_paired":         len(paired),
                "orphan_netlists":  orphan_nets,
                "orphan_keys":      orphan_keys,
                "bad":              bad,
            }, f, indent=2)
        print()
        print(f"  wrote: {op}.per_file.csv  ({len(df)} rows)")
        print(f"         {op}.per_circuit.csv  ({len(per_circ)} rows)")
        print(f"         {op}.gate_hist.csv")
        print(f"         {op}.summary.json")


if __name__ == "__main__":
    main()