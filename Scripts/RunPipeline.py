#!/usr/bin/env python3
"""
RunPipeline.py — Master Orchestrator for the Logic Locking & Evaluation Pipeline

This version provides total parameter control:
- Complete Verification, Generation, and Hash configuration bounds.
- Mutually exclusive K-value selection (explicit list vs. min/max/step range).
- Selectable evaluation equivalences and covering policies.
- Global circuit filtering, dry-run, and skip-existing capabilities.
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

# --- Default Fallback Lists ---
ALL_EVAL_EQUIVALENCES = ["strict", "gate_type", "partition"]
ALL_COVERING_POLICIES = ["table", "first", "minimal-partitions"]
BASE_METHODS = ["structural", "adaptive", "keyregion"]

def run_cmd(cmd: list, desc: str, log_file: Path = None):
    """Executes a subprocess. Routes stdout to log_file if provided, leaving stderr to terminal."""
    print(f"\n[>>>] {desc}")
    cmd_str = [str(c) for c in cmd]
    
    try:
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as f:
                subprocess.run(cmd_str, stdout=f, check=True)
            print(f"      |_ Log saved to: {log_file}")
        else:
            subprocess.run(cmd_str, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] Command failed during: {desc}")
        print(f"Command: {' '.join(cmd_str)}")
        sys.exit(1)

def display_confirmation_manifest(args, k_list):
    """Displays an exhaustive list of all parsed parameters and requests user confirmation."""
    print("\n" + "=" * 90)
    print(" PIPELINE EXECUTION MANIFEST".center(90))
    print("=" * 90)
    
    print("\n[GLOBAL OVERRIDES & PATHS]")
    print(f"  Databases            : {', '.join(args.dbs)}")
    print(f"  Worker Threads       : {args.workers}")
    print(f"  Circuit Target       : {args.circuit if args.circuit else 'All Circuits'}")
    print(f"  Graph Limit          : {args.limit if args.limit else 'None (Processing All)'}")
    print(f"  Force Overwrite      : {'YES (--force)' if args.force else 'No'}")
    print(f"  Skip Existing        : {'YES (--skip-existing)' if args.skip_existing else 'No'}")
    print(f"  Disable TQDM Bars    : {'YES (--no-progress)' if args.no_progress else 'No'}")
    print(f"  Root Directory       : {args.root_dir} ")
    print(f"  Original Nelists Dir : {args.original_netlists}")
    print(f"  Output Base Direcotry: {args.out_base}")
    
    print("\n[BATCHLOCK: GENERATION PARAMETERS]")
    print(f"  Locking Pct Bounds   : Min {args.min_pct}% -> Max {args.max_pct}% (Step: {args.step_pct}%)")
    print(f"  Metaseed / Seed      : Metaseed={args.metaseed} / Seed={args.seed}")
    print(f"  Variant Policy       : {args.variant_policy}")
    print(f"  Key Port Name        : {args.key_port}")
    print(f"  No NAND/NOR          : {'YES (--no-nand-nor)' if args.no_nand_nor else 'No (Allowed)'}")
    
    print("\n[BATCHLOCK: VERIFICATION PARAMETERS]")
    print(f"  Perform Verify?      : {'YES (--verify)' if args.verify else 'NO (Skipping Verification)'}")
    if args.verify:
        print(f"  Keep Failed Locks    : {'YES (--keep-failed)' if args.keep_failed else 'No'}")
        print(f"  Verify Samples       : {args.verify_samples}")
        print(f"  Verify Timeout       : {args.verify_timeout} seconds")
        print(f"  Verify No IDKLL      : {'YES (--verify-no-idkll)' if args.verify_no_idkll else 'No'}")
    
    print("\n[STRUCTURAL FINGERPRINTING PARAMETERS]")
    print(f"  Adaptive Max Depth   : {args.max_depth} hops")
    print(f"  Max Region Size      : {args.max_region_size} cells (KeyRegion & Collapse)")
    print(f"  WL Iterations        : {args.wl_iter}")
    print(f"  WL Digest Size       : {args.digest_size} bytes")
    
    print("\n[CONSENSUS PARAMETERS]")
    print(f"  Computed K-Values    : {k_list}")
    print(f"  Max Cluster Size     : {args.max_cluster_size_consensus} cells")
    
    print("\n[UNSUPERVISED EVALUATION PARAMETERS]")
    print(f"  Label Equivalences   : {', '.join(args.eval_equivalences)}")
    print(f"  Min Cluster Size     : {args.min_cluster_size_eval} cells (for 'is-locked' prediction)")
    
    print("\n[FUNCTIONAL ATTACK (COVERING) PARAMETERS]")
    print(f"  Match Policies       : {', '.join(args.covering_policies)}")
    print(f"  Max Extr. Cluster    : {args.max_cluster_size_attack} cells")
    print(f"  Covering Limit       : {args.covering_limit}")
    print("=" * 90)
        
    while True:
        choice = input("\nDo you want to proceed with this exact configuration? [Y/n]: ").strip().lower()
        if choice in ['y', 'yes', '']:
            return True
        elif choice in ['n', 'no']:
            print("\nPipeline execution aborted by user.")
            sys.exit(0)
        else:
            print("Please enter 'Y' or 'n'.")

def append_global_flags(cmd: list, args):
    """Appends global boolean and filtering flags to downstream scripts. Currently, these flags include:
    `force, no_progress, limit, circuit`"""
    if args.force: cmd.append("--force")
    if args.no_progress: cmd.append("--no-progress")
    if args.limit: cmd.extend(["--limit", str(args.limit)])
    if args.circuit: cmd.extend(["--circuit", args.circuit])
    return cmd

def main():
    ap = argparse.ArgumentParser(description="Master Pipeline Orchestrator")
    
    # Global Parameters
    ap.add_argument("--dbs", required=True, action="append", help="List of databases (e.g. CWS_Bench)")
    ap.add_argument("--workers", type=int, default=12, help="Number of parallel workers")
    ap.add_argument("--circuit", type=str, default=None, help="Target a specific circuit (e.g. 'c432') [THIS IS A WORK IN PROGRESS FEATURE]")
    ap.add_argument("--limit", type=int, default=None, help="Limit graphs processed")
    ap.add_argument("--force", action="store_true", help="Force overwrite of existing caches and outputs")
    ap.add_argument("--skip-existing", action="store_true", help="Skip existing files/variants")
    ap.add_argument("--no-progress", action="store_true", help="Disable all underlying TQDM progress bars")
    
    # Path Parameters
    ap.add_argument("--root-dir", default=".", help="Root directory")
    ap.add_argument("--original-netlists", default="Original_ISCAS85/dot_json", help="Path to original netlists")
    ap.add_argument("--out-base", required=True, help="Base output directory")
    
    # BatchLock: Generation Parameters
    ap.add_argument("--min-pct", type=float, default=5.0, help="Minimum locking percentage")
    ap.add_argument("--max-pct", type=float, default=25.0, help="Maximum locking percentage")
    ap.add_argument("--step-pct", type=float, default=5.0, help="Step size for locking percentages")
    ap.add_argument("--metaseed", type=int, help="Metaseed for dataset generation")
    ap.add_argument("--seed", type=int, help="Base seed for reproducibility")
    ap.add_argument("--variant-policy", type=str, default="uniform", help="Distribution policy for locked variants")
    ap.add_argument("--key-port", type=str, default="key_input", help="Name of the key port injected into netlist")
    ap.add_argument("--no-nand-nor", action="store_true", help="Exclude NAND/NOR gates from locking targets")
    
    # BatchLock: Verification Parameters
    ap.add_argument("--verify", action="store_true",
                      help="Enables verification for locked netlists.")
    ap.add_argument("--verifier", default=None, help="Explicit path to VerifyLock.py.")
    ap.add_argument("--verify-samples", type=int, default=2000, help="Forwarded to VerifyLock.py as --samples.")
    ap.add_argument("--verify-key-samples", type=int, default=1024, help="Forwarded to VerifyLock.py as --key-samples.")
    ap.add_argument("--verify-no-idkll", action="store_true", help="Skip the IDKLL global-key check inside VerifyLock.py.")
    ap.add_argument("--verify-timeout", type=float, default=None, help="Per-circuit verification timeout.")
    ap.add_argument("--keep-failed", action="store_true", help="KEEP locked files even if verification fails.")
    
    # Hashing/Structural Parameters
    ap.add_argument("--max-depth", type=int, default=8, help="Hard cap on adaptive depth BFS")
    ap.add_argument("--max-region-size", type=int, default=64, help="Hard cap on subgraph region sizes")
    ap.add_argument("--wl-iter", type=int, default=3, help="Weisfeiler-Lehman iterations")
    ap.add_argument("--digest-size", type=int, default=16, help="WL digest string size in bytes")
    
    # Consensus Parameters (Mutually Exclusive Group)
    k_group = ap.add_mutually_exclusive_group()
    k_group.add_argument("--k-values", nargs="+", type=int, 
                         help="Explicit list of K values (e.g., --k-values 2 4 6)")
    k_group.add_argument("--k-range", nargs=3, type=int, metavar=('MIN', 'MAX', 'STEP'),
                         help="Range for K values: MIN MAX STEP (e.g., --k-range 2 10 2)")
    ap.add_argument("--max-cluster-size-consensus", type=int, default=500, help="Ignore voting on clusters larger than this threshold to prevent combinatorial explosion.")
    
    # Evaluation & Coverage Filtering Parameters
    ap.add_argument("--eval-equivalences", nargs="+", choices=ALL_EVAL_EQUIVALENCES, default=ALL_EVAL_EQUIVALENCES, 
                    help="Subset of evaluation equivalences to run")
    ap.add_argument("--min-cluster-size-eval", type=int, default=2, help="Binary classification min cluster size")
    ap.add_argument("--covering-policies", nargs="+", choices=ALL_COVERING_POLICIES, default=ALL_COVERING_POLICIES, 
                    help="Subset of covering match policies to execute")
    ap.add_argument("--max-cluster-size-attack", type=int, default=256, help="Cap on BFS extraction for attack")
    ap.add_argument("--covering-limit", type=int, help="Max candidates to evaluate during covering")

    args = ap.parse_args()

    # Determine K-values list
    if args.k_values:
        k_list = args.k_values
    elif args.k_range:
        k_list = list(range(args.k_range[0], args.k_range[1] + 1, args.k_range[2]))
    else:
        k_list = [2, 4, 6, 8, 10] # Default

    # Request explicit confirmation
    display_confirmation_manifest(args, k_list)

    root = Path(args.root_dir).resolve()
    orig_netlists = root / args.original_netlists
    out_base = root / args.out_base

    scripts_eval = root / "Scripts" / "Evaluation"
    scripts_struct = root / "Scripts" / "StructuralFingerprinting"
    scripts_func = root / "Scripts" / "FunctionalAnalysis"

    for db in args.dbs:
        print("\n" + "=" * 80)
        print(f" RUNNING DATABASE: {db}".center(80))
        print("=" * 80)

        db_path = out_base / db
        cache_dir = db_path / "cache" / "graphs"
        res_dir = db_path / "results"
        out_log_dir = db_path / "outputs"
        eval_dir = db_path / "evaluation"

        for d in [db_path/"netlists", db_path/"keys", db_path/"metadata", 
                  cache_dir, res_dir, out_log_dir, eval_dir]:
            d.mkdir(parents=True, exist_ok=True)

        script_locking_dir = root / "Scripts" / "Locking" / db
        
        # --- Step 1: BatchLock ---
        cmd = [
            "python", script_locking_dir / "BatchLock.py",
            "--in-dir", orig_netlists,
            "--out-dir", db_path,
            "--workers", args.workers,
            "--lower", args.min_pct,
            "--upper", args.max_pct,
            "--step", args.step_pct,
            "--variant-policy", args.variant_policy,
            "--key-port", args.key_port
        ]
        if args.seed is not None: 
            cmd.extend(["--seed", args.seed])
        else: 
            cmd.extend(["--meta-seed", args.metaseed if args.metaseed is not None else random.randrange(2**31)])

        if args.limit: cmd.extend(["--limit", str(args.limit)])
        if args.force: cmd.append("--force")
        if args.skip_existing: cmd.append("--skip-existing")
        if args.no_nand_nor: cmd.append("--no-nand-nor")
        
        
        if args.verify:
            cmd.append("--verify")
            cmd.extend(["--verify-samples", str(args.verify_samples)])
            cmd.extend(["--verify-key-samples", str(args.verify_key_samples)])
            if args.verify_timeout is not None: cmd.extend(["--verify-timeout", str(args.verify_timeout)])
            if args.verify_no_idkll: cmd.append("--verify-no-idkll")
            if args.keep_failed: cmd.append("--keep-failed")
            
        run_cmd(cmd, f"[{db}] Locking Netlists (BatchLock)", log_file=None)

        # --- Step 2: DatasetScan ---
        cmd = [
            "python", scripts_eval / "DatasetScan.py",
            "--locked-dir", db_path,
            "--original-dir", orig_netlists,
            "--out-prefix",  db_path/"metadata"/"scan_report"
        ]
        if args.no_progress: cmd.append("--no-progress")
        run_cmd(cmd, f"[{db}] Scanning Dataset", log_file=out_log_dir / "datasetscan.txt")

        # --- Step 3: BuildGraphs ---
        cmd = [
            "python", scripts_eval / "BuildGraphs.py",
            "--locked-dir", db_path,
            "--original-dir", orig_netlists,
            "--cache-dir", cache_dir,
            "--workers", args.workers
        ]
        cmd = append_global_flags(cmd, args)
        run_cmd(cmd, f"[{db}] Building Graph Cache", log_file=None)

        # --- Step 4: Structural Fingerprinting ---
        cmd_struct = [
            "python", scripts_struct / "StructuralHash.py",
            "--cache-dir", cache_dir,
            "--out-dir", res_dir / "structural",
            "--workers", args.workers
        ]
        cmd_struct = append_global_flags(cmd_struct, args)
        run_cmd(cmd_struct, f"[{db}] Hashing: STRUCTURAL", log_file=out_log_dir / "structural.txt")

        cmd_adapt = [
            "python", scripts_struct / "Adaptive.py",
            "--cache-dir", cache_dir,
            "--out-dir", res_dir / "adaptive",
            "--max-depth", args.max_depth,
            "--workers", args.workers
        ]
        cmd_adapt = append_global_flags(cmd_adapt, args)
        run_cmd(cmd_adapt, f"[{db}] Hashing: ADAPTIVE", log_file=out_log_dir / "adaptive.txt")

        cmd_keyreg = [
            "python", scripts_struct / "KeyRegion.py",
            "--cache-dir", cache_dir,
            "--out-dir", res_dir / "keyregion",
            "--max-region-size", args.max_region_size,
            "--wl-iter", args.wl_iter,
            "--digest-size", args.digest_size,
            "--workers", args.workers
        ]
        cmd_keyreg = append_global_flags(cmd_keyreg, args)
        run_cmd(cmd_keyreg, f"[{db}] Hashing: KEYREGION", log_file=out_log_dir / "keyregion.txt")

        # --- Step 5: Collapse ---
        for method in BASE_METHODS:
            cmd = [
                "python", scripts_struct / "Collapse.py",
                "--cache-dir", cache_dir,
                "--in-method-dir", res_dir / method,
                "--out-dir", res_dir / "collapse" / method,
                "--max-region-size", args.max_region_size,
                "--workers", args.workers
            ]
            if args.force: cmd.append("--force")
            if args.no_progress: cmd.append("--no-progress")
            run_cmd(cmd, f"[{db}] Collapsing: {method}", log_file=out_log_dir / f"collapse_{method}.txt")

        # --- Step 6: Consensus ---
        cmd = [
            "python", scripts_struct / "Consensus2.py",
            "--out-dir", res_dir / "consensus",
            "--max-cluster-size", args.max_cluster_size_consensus,
            "--workers", args.workers
        ]
        if args.force: cmd.append("--force")
        if args.no_progress: cmd.append("--no-progress")
        for method in BASE_METHODS:
            cmd.extend(["--in-method-dir", res_dir / method])
            cmd.extend(["--in-method-dir", res_dir / "collapse" / method])
        for k in k_list:
            cmd.extend(["--k", str(k)])
            
        run_cmd(cmd, f"[{db}] Consensus Clustering", log_file=out_log_dir / "consensus.txt")

        # --- Step 7: Unsupervised Evaluation ---
        all_eval_dirs = BASE_METHODS + [f"collapse/{m}" for m in BASE_METHODS] # + ["consensus"]
        
        for tech in all_eval_dirs:
            tech_safe_name = tech.replace("/", "_")
            for eq in args.eval_equivalences:
                cmd = [
                    "python", scripts_eval / "Evaluate.py",
                    "--cache-dir", cache_dir,
                    "--method-dir", res_dir / tech,
                    "--label-equivalence", eq,
                    "--min-cluster-size", args.min_cluster_size_eval,
                    "--out-prefix", eval_dir / f"{tech_safe_name}_{eq}"
                ]
                if args.no_progress: cmd.append("--no-progress")
                run_cmd(cmd, f"[{db}] Evaluating: {tech_safe_name} ({eq})", log_file=out_log_dir / f"evaluation_{tech_safe_name}_{eq}.txt")

        # --- Step 8: Functional Analysis (Covering Attack) ---
        for policy in args.covering_policies:
            cmd = [
                "python", scripts_func / "CoveringAttack.py",
                "--cache-dir", cache_dir,
                "--sidecar-dir", db_path / "keys",
                "--out-dir", res_dir / "covering" / policy,
                "--match-policy", policy,
                "--max-cluster-size", args.max_cluster_size_attack,
                "--workers", args.workers,
            ]
            if args.force: cmd.append("--force")
            if args.no_progress: cmd.append("--no-progress")
            if args.limit is not None: cmd.extend(["--limit", str(args.limit)])
            run_cmd(cmd, f"[{db}] Covering Attack: {policy}", log_file=out_log_dir / f"covering_{policy}.txt")

        # --- Step 9: Evaluate Covering Attack ---
        for policy in args.covering_policies:
            cmd = [
                "python", scripts_func / "EvaluateCovering.py",
                "--cache-dir", cache_dir,
                "--covering-dir", res_dir / "covering" / policy,
                "--sidecar-dir", db_path / "keys",
                "--out-prefix", eval_dir / f"covering_{policy}",
                "--workers", args.workers
            ]
            if args.no_progress: cmd.append("--no-progress")
            run_cmd(cmd, f"[{db}] Evaluating Covering: {policy}", log_file=out_log_dir / f"evaluation_covering_{policy}.txt")

    print("\n" + "=" * 90)
    print(" PIPELINE COMPLETED SUCCESSFULLY".center(90))
    print("=" * 90)

if __name__ == "__main__":
    main()