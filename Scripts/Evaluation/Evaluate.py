#!/usr/bin/env python3
"""
Evaluate.py — Compare unsupervised clusterings produced by a method
              (MethodAStruct.py, later MethodB_*, MethodC_*) against
              the ground-truth variant labels from the GateLock sidecars.

This is the ONLY script in the pipeline that ever reads the per-node
`labels` dict from a cached graph. Methods themselves must never use
them; this script is a measurement instrument, not part of the loop.

Two evaluations per (graph, mode, depth):

  1. Clustering quality vs. the ground-truth partition of LOCKED cells.
       - homogeneity, completeness, V-measure
       - normalised mutual information (NMI)
       - adjusted Rand index (ARI)
     The ground truth has K classes (K = number of distinct variants
     used in that netlist); the predicted clustering is the hash column.
     "Unlocked" cells (label 0) can be either INCLUDED (and treated as
     class 0) or EXCLUDED. We report both; the latter is the right
     number for "how well does the method separate variants among the
     cells it correctly identifies as locked."

  2. Binary "is locked" classification:
       - predicted positive  =  cell sits in a cluster of size >= min_size
                                (default 2)
       - true positive       =  cell whose label != 0
     -> precision, recall, F1.

Outputs:
    --out-prefix .csv         tidy one row per (stem, mode, depth)
    --out-prefix .leaderboard.csv   ranked by mean V-measure (excl-unlocked)
    stdout summary

Dependencies:
    pip install pandas networkx tqdm pyarrow scikit-learn
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm


# Per-cell parquet metadata columns; everything ELSE is treated as a
# cluster id column. This lets new method scripts add their own hash
# columns without editing this file.


# --------------------------------------------------------------------
# Equivalence-class re-labelling
# --------------------------------------------------------------------
# The sidecar gives each locked cell a variant_partition string like
#   "k=3 | split-by-A | KS=(010,101)"
# but in IDKLL many KS choices unlock the same partition equivalently
# (Rathor confirms this in both GateLock and SubLock). So for the
# security-relevant metric we collapse classes by the (gate_type,
# partition_class) tuple, dropping the KS suffix.

def _partition_class(variant_partition_str: str) -> str:
    """
    Extracts the partition descriptor from a Database.py variant partition string.

    This function isolates the middle section (e.g., 'split-by-A') from strings 
    formatted like 'k=3 | split-by-A | KS=(010,101)'.

    Args:
        variant_partition_str (str): The raw variant descriptor string.

    Returns:
        str: The extracted partition class, or the original string if parsing fails.
    """
    parts = [p.strip() for p in variant_partition_str.split("|")]
    return parts[1] if len(parts) >= 2 else variant_partition_str


def _remap_labels(labels_dict: dict, graph, variant_id_table: dict, mode: str) -> dict[str, int]:
    """
    Translates literal ground-truth labels into evaluation equivalence classes.

    Args:
        labels_dict (dict): Map of cell_name -> original ground truth integer label.
        graph (nx.DiGraph): The corresponding networkx directed graph.
        variant_id_table (dict): The sidecar mapping of partition strings to label IDs.
        mode (str): The equivalence mode ('strict', 'partition', or 'gate_type').

    Returns:
        dict[str, int]: A new mapping of cell_name -> updated equivalence class integer ID.

    Raises:
        ValueError: If an unknown label-equivalence mode is provided.
    """
    if mode == "strict":
        return labels_dict

    # invert variant_id_table:  int -> variant_partition_str
    id_to_str = {v: k for k, v in variant_id_table.items()}
    new_id_of = {0: 0}   # 0 = unlocked, always
    next_id = 1
    out = {}

    for cell, old_id in labels_dict.items():
        if old_id == 0:
            out[cell] = 0
            continue

        variant_str = id_to_str.get(old_id, "")
        if mode == "partition":
            gate_type = graph.nodes[cell].get("cell_type", "")
            key = (gate_type, _partition_class(variant_str))
        elif mode == "gate_type":
            key = graph.nodes[cell].get("cell_type", "")
        else:
            raise ValueError(f"unknown --label-equivalence {mode!r}")

        if key not in new_id_of:
            new_id_of[key] = next_id
            next_id += 1
        out[cell] = new_id_of[key]
    return out

METADATA_COLS = {
    "node", "cell_type", "cone_size_d3", "reaches_key",
    "region_size", "region_depth", "n_key_in_region",
    "n_data_boundaries", "adaptive_depth",
}


def find_hash_cols(df: pd.DataFrame) -> list[str]:
    """
    Identifies which columns in the DataFrame contain clustering predictions.

    Args:
        df (pd.DataFrame): The DataFrame read from a method's output parquet.

    Returns:
        list[str]: A sorted list of column names that represent prediction hashes.
    """
    return sorted([c for c in df.columns
                   if c not in METADATA_COLS and not c.startswith("_")])


def parse_mode_depth(col: str) -> tuple[str, int]:
    """
    Parses cluster-column names robustly to extract their hashing mode and depth.

    Examples:
        'merkle_d3'              -> ('merkle', 3)
        'collapsed_cons_merkle_d3' -> ('collapsed_cons_merkle', 3)
        'consensus_k2'           -> ('consensus', 2)   (k treated as depth)
        'keyregion_h'            -> ('keyregion_h', 0)

    Args:
        col (str): The column name.

    Returns:
        tuple[str, int]: (Method/Mode Name, Depth/K-value).
    """
    parts = col.rsplit("_", 1)
    if len(parts) == 2:
        suffix = parts[1]
        if len(suffix) >= 2 and suffix[0] in ("d", "k") and suffix[1:].isdigit():
            return parts[0], int(suffix[1:])
    return col, 0


def evaluate_one(stem: str, per_cell: pd.DataFrame, labels: dict, min_cluster_size: int = 2) -> list[dict]:
    """
    Evaluates all clustering columns in a DataFrame against ground-truth labels.

    Computes comprehensive clustering metrics (Homogeneity, Completeness, V-measure,
    NMI, ARI) and binary classification metrics (Precision, Recall, F1).

    Args:
        stem (str): Identifier for the specific netlist/circuit being evaluated.
        per_cell (pd.DataFrame): The DataFrame containing predictions.
        labels (dict): The ground-truth node -> integer class mapping.
        min_cluster_size (int, optional): Minimum cluster size required for a cell 
                                          to be classified as "locked". Defaults to 2.

    Returns:
        list[dict]: A list of evaluation metric dictionaries (one per prediction column).
    """
    rows = []
    # Per-cell ground truth
    gt = per_cell["node"].map(labels).astype("Int64")
    if gt.isna().any():
        # Should not happen — labels dict covers every cell. Skip safely.
        missing = int(gt.isna().sum())
        gt = gt.fillna(0).astype(int)
    else:
        missing = 0
    gt = gt.to_numpy()
    is_locked_true = (gt != 0)

    hash_cols = find_hash_cols(per_cell)
    for col in hash_cols:
        # Encode the string hashes as small ints (sklearn wants ints)
        pred_str = per_cell[col].to_numpy()
        # pd.factorize is fast and stable
        pred, _ = pd.factorize(pred_str, sort=False)

        # --- clustering metrics: INCLUDE unlocked (label 0) as its own class
        h_in, c_in, v_in = homogeneity_completeness_v_measure(gt, pred)
        nmi_in = normalized_mutual_info_score(gt, pred)
        ari_in = adjusted_rand_score(gt, pred)

        # --- clustering metrics: EXCLUDE unlocked
        mask = is_locked_true
        if mask.sum() >= 2 and len(np.unique(gt[mask])) >= 2:
            h_ex, c_ex, v_ex = homogeneity_completeness_v_measure(
                gt[mask], pred[mask])
            nmi_ex = normalized_mutual_info_score(gt[mask], pred[mask])
            ari_ex = adjusted_rand_score(gt[mask], pred[mask])
        else:
            h_ex = c_ex = v_ex = nmi_ex = ari_ex = float("nan")

        # --- binary "is locked": predicted positive = cluster size >= min
        vc = per_cell[col].value_counts()
        big_hashes = set(vc[vc >= min_cluster_size].index)
        pred_locked = per_cell[col].isin(big_hashes).to_numpy()

        if pred_locked.any() or is_locked_true.any():
            p, r, f1, _ = precision_recall_fscore_support(
                is_locked_true, pred_locked, average="binary",
                zero_division=0)
        else:
            p = r = f1 = 0.0

        mode, depth = parse_mode_depth(col)
        rows.append({
            "stem":  stem,
            "mode":  mode,
            "depth": depth,
            "n_cells":            len(per_cell),
            "n_locked_truth":     int(is_locked_true.sum()),
            "n_locked_pred":      int(pred_locked.sum()),
            "n_clusters":         int(per_cell[col].nunique()),
            "n_singletons":       int((vc == 1).sum()),
            "largest_cluster":    int(vc.iloc[0]) if len(vc) else 0,
            # clustering quality (including unlocked as class 0)
            "homogeneity_inc":   round(h_in,  4),
            "completeness_inc": round(c_in,  4),
            "vmeasure_inc":     round(v_in,  4),
            "nmi_inc":          round(nmi_in, 4),
            "ari_inc":          round(ari_in, 4),
            # excluding unlocked: "how well do clusters resolve VARIANTS?"
            "homogeneity_exc":  round(h_ex,  4) if not np.isnan(h_ex) else None,
            "completeness_exc": round(c_ex,  4) if not np.isnan(c_ex) else None,
            "vmeasure_exc":     round(v_ex,  4) if not np.isnan(v_ex) else None,
            "nmi_exc":          round(nmi_ex, 4) if not np.isnan(nmi_ex) else None,
            "ari_exc":          round(ari_ex, 4) if not np.isnan(ari_ex) else None,
            # binary is-locked
            "precision":        round(p,  4),
            "recall":           round(r,  4),
            "f1":               round(f1, 4),
            "missing_labels":   missing,
        })
    return rows


def main():
    """Parses arguments, orchestrates graph evaluations, and writes performance leaderboards."""
    ap = argparse.ArgumentParser(
        description="Evaluate unsupervised clusterings against GateLock ground-truth labels."
    )
    ap.add_argument("--cache-dir", default="cache/graphs",
                    help="Directory containing the serialized graph pickles and the manifest.")
    ap.add_argument("--method-dir", default="results/method_a",
                    help="Directory containing the generated <stem>.cluster.parquet prediction files to evaluate.")
    ap.add_argument("--out-prefix", default="results/method_a/evaluation",
                    help="Prefix for the output metric files (generates .csv and .leaderboard.csv).")
    ap.add_argument("--label-equivalence", default="strict",
                    choices=["strict", "partition", "gate_type"],
                    help="How to bucket the ground-truth labels: "
                         "'strict' = full variant_partition string (default); "
                         "'partition' = (gate_type, partition_class), "
                         "merging variants that differ only in KS choice; "
                         "'gate_type' = collapse by underlying gate kind only.")
    ap.add_argument("--min-cluster-size", type=int, default=2,
                    help="Cells in any cluster >= this size are predicted as 'locked' (default 2).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Currently unused; evaluation is fast and the bottleneck is disk IO, not CPU.")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable the TQDM progress bar output.")
    args = ap.parse_args()

    cache_dir  = Path(args.cache_dir).resolve()
    method_dir = Path(args.method_dir).resolve()
    out_prefix = Path(args.out_prefix).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build {stem: cached_graph_pkl_path}
    # ------------------------------------------------------------------
    man_path = cache_dir / "manifest.parquet"
    if not man_path.exists():
        raise SystemExit(f"manifest not found: {man_path}")
    manifest = pd.read_parquet(man_path)
    locked = manifest[manifest["kind"] == "locked"]
    
    def _resolve_pkl(row):
        # Preferred: reconstruct relative to --cache-dir so the cache is
        # relocatable. This handles the common case where the cache was
        # built under one name (e.g. cache/graphs) and now lives under
        # another (e.g. cache/graphs_sd99).
        cand = cache_dir / "locked" / row["circuit"] / f"{row['stem']}.pkl"
        if cand.exists():
            return cand
        # Fallback: trust the path stored in the manifest, resolving
        # it relative to cache_dir if it is relative.
        stored = Path(row["path"])
        return stored if stored.is_absolute() else (cache_dir / stored)
    
    pkl_of_stem = {row["stem"]: _resolve_pkl(row)
                   for _, row in locked.iterrows()}

    # ------------------------------------------------------------------
    # For every per-cell parquet in method_dir, load + evaluate
    # ------------------------------------------------------------------
    cluster_files = sorted(method_dir.glob("*.cluster.parquet"))
    if not cluster_files:
        raise SystemExit(f"no *.cluster.parquet under {method_dir}; "
                         "run MethodAStruct.py first")

    print(f"cache dir : {cache_dir}")
    print(f"method dir: {method_dir}")
    print(f"label equivalence: {args.label_equivalence}")
    print(f"to do     : {len(cluster_files)} per-cell parquets\n")

    all_rows = []
    skipped = []
    pbar = tqdm(cluster_files, disable=args.no_progress, desc="evaluating")
    for cf in pbar:
        stem = cf.name[:-len(".cluster.parquet")]
        pkl = pkl_of_stem.get(stem)
        if pkl is None:
            skipped.append((stem, "no manifest entry", ""));  continue
        try:
            with open(pkl, "rb") as f:
                rec = pickle.load(f)
            labels = rec["labels"]
            if labels is None:
                skipped.append((stem, "no labels (is this an original?)", ""));
                continue
            per_cell = pd.read_parquet(cf)
            if args.label_equivalence != "strict":
                labels = _remap_labels(labels,
                                       rec["graph"],
                                       rec.get("variant_id", {}),
                                       args.label_equivalence)
            rows = evaluate_one(stem, per_cell, labels,
                                min_cluster_size=args.min_cluster_size)
            # enrich with manifest metadata
            meta = locked[locked["stem"] == stem].iloc[0]
            for r in rows:
                r["circuit"] = meta["circuit"]
                r["pct"]     = float(meta["pct"])
            all_rows.extend(rows)
        except Exception as e:
            import traceback
            skipped.append((stem, f"{type(e).__name__}: {e}",
                            traceback.format_exc()))

    if not all_rows:
        print(f"\n  NO EVALUATIONS PRODUCED — every file was skipped.")
        if skipped:
            print(f"  {len(skipped)} skipped files; reasons (top 10):")
            from collections import Counter
            reason_hist = Counter(s[1] for s in skipped)
            for reason, count in reason_hist.most_common(10):
                print(f"    {count:>4}  {reason}")
            print()
            print("  Full traceback for the first failing file:")
            for stem, reason, tb in skipped[:3]:
                print(f"  ----- {stem}  ({reason})")
                if tb:
                    for line in tb.strip().splitlines():
                        print(f"    {line}")
                print()
        raise SystemExit(1)

    df = pd.DataFrame(all_rows)
    cols = ["circuit", "pct", "stem", "mode", "depth",
            "n_cells", "n_locked_truth", "n_locked_pred",
            "n_clusters", "n_singletons", "largest_cluster",
            "homogeneity_inc", "completeness_inc", "vmeasure_inc",
            "nmi_inc", "ari_inc",
            "homogeneity_exc", "completeness_exc", "vmeasure_exc",
            "nmi_exc", "ari_exc",
            "precision", "recall", "f1"]
    df = df[cols].sort_values(["circuit", "pct", "mode", "depth"])

    out_csv = out_prefix.with_suffix(".csv")
    df.to_csv(out_csv, index=False)

    # ------------------------------------------------------------------
    # Leaderboard: aggregate by (mode, depth) across all graphs
    # ------------------------------------------------------------------
    agg = (df.groupby(["mode", "depth"])
             .agg(n=("stem", "count"),
                  vmeasure_exc_mean=("vmeasure_exc", "mean"),
                  vmeasure_exc_std=("vmeasure_exc", "std"),
                  nmi_exc_mean=("nmi_exc", "mean"),
                  ari_exc_mean=("ari_exc", "mean"),
                  vmeasure_inc_mean=("vmeasure_inc", "mean"),
                  precision_mean=("precision", "mean"),
                  recall_mean=("recall", "mean"),
                  f1_mean=("f1", "mean"),
                  n_clusters_mean=("n_clusters", "mean"))
             .round(4)
             .reset_index()
             .sort_values("vmeasure_exc_mean", ascending=False))
    lb_csv = out_prefix.with_suffix(".leaderboard.csv")
    agg.to_csv(lb_csv, index=False)

    # ------------------------------------------------------------------
    # Stdout report
    # ------------------------------------------------------------------
    print("\nLeaderboard (sorted by mean V-measure, EXCLUDING unlocked cells)")
    print("=" * 96)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(agg.to_string(index=False))
    print()
    print("Headline binary 'is-locked' metric (cluster size >= "
          f"{args.min_cluster_size}):")
    print(agg[["mode", "depth", "precision_mean", "recall_mean", "f1_mean"]]
          .sort_values("f1_mean", ascending=False)
          .to_string(index=False))

    # Per-circuit breakdown for the best (mode, depth)
    best_row = agg.iloc[0]
    best_mode, best_depth = best_row["mode"], int(best_row["depth"])
    print()
    print(f"Per-circuit breakdown for best (mode, depth) = "
          f"({best_mode}, d{best_depth}), by V-measure-exc:")
    sub = df[(df["mode"] == best_mode) & (df["depth"] == best_depth)]
    pc = (sub.groupby("circuit")
             .agg(n=("stem", "count"),
                  vme=("vmeasure_exc", "mean"),
                  nmi=("nmi_exc", "mean"),
                  pre=("precision", "mean"),
                  rec=("recall", "mean"),
                  f1=("f1", "mean"))
             .round(4))
    print(pc.to_string())

    print(f"\nwrote: {out_csv}  ({len(df)} rows)")
    print(f"       {lb_csv}  ({len(agg)} rows)")
    if skipped:
        print(f"\nskipped {len(skipped)} files:")
        for s, why, _tb in skipped[:5]:
            print(f"  {s}: {why}")


if __name__ == "__main__":
    main()