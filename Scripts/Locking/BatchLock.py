#!/usr/bin/env python3
"""
BatchLock.py — Multithreaded batch driver that runs `GateLock.py` over every Yosys JSON
               netlist in an input directory across a range of locking percentages, 
               optionally followed by `VerifyLock.py`.

Naming convention for outputs (placed under <output-dir>/):
    <circuit>_p<pct>_s<seed>.json           # locked netlist
    <circuit>_p<pct>_s<seed>.key.json       # GateLock key sidecar

Bookkeeping files written under <output-dir>/:
    seeds.json   — circuit -> seed used for that circuit
    seeds.csv    — same, as a 2-column CSV
    runs.csv     — one row per (circuit, pct) with status, key width,
                   #locked gates, runtime, output filename and (if
                   --verify is used) the verification verdict.
    verification_failures.txt
                 — present only when --verify is used.  Lists every
                   combo whose verification did not pass, with the tail
                   of VerifyLock.py's output for each.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def frange(lo: float, hi: float, step: float) -> list[float]:
    """Generates a list of floating-point numbers from a start to an end value."""
    if step <= 0:
        raise ValueError("--step must be > 0")
    if hi < lo:
        raise ValueError("--upper must be >= --lower")
    eps = step * 1e-6
    n = int(round((hi - lo) / step))
    out = []
    for i in range(n + 1):
        v = lo + i * step
        if v > hi + eps:
            break
        out.append(round(v, 6))
    if out and abs(out[-1] - hi) > eps:
        out.append(round(hi, 6))
    return out


def pct_tag(p: float) -> str:
    """Converts a float percentage into a filesystem-safe string tag."""
    if abs(p - round(p)) < 1e-9:
        return f"{int(round(p))}"
    return f"{p:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def find_sibling(name: str, override: str | None) -> Path:
    """Locates an auxiliary Python script in the local directory."""
    if override:
        p = Path(override).resolve()
        if not p.exists():
            raise SystemExit(f"--{name.replace('.py','')} path not found: {p}")
        return p
    here = Path(__file__).resolve().parent
    cand = here / name
    if cand.exists():
        return cand
    raise SystemExit(
        f"could not find {name} next to BatchLock.py; "
        f"pass --{name.replace('.py','')} /path/to/{name}")


def discover_inputs(in_dir: Path, circuit_filter: list[str] | None=None) -> list[Path]:
    """Returns a sorted list of all JSON files in the given directory."""
    files= sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    if circuit_filter:
        allowed = {c if c.lower().endswith(".json") else f"{c}.json" for c in circuit_filter}
        files = [p for p in files if p.name in allowed]
    return files


def is_yosys_json(path: Path) -> bool:
    """Performs a fast preliminary check to verify if a JSON file is a Yosys netlist."""
    try:
        with path.open() as f:
            head = f.read(4096)
        return '"modules"' in head
    except OSError:
        return False


# ----------------------------------------------------------------------
# Verification helper
# ----------------------------------------------------------------------

def run_verifier(verifier: Path, orig: Path, locked: Path, key_map: Path,
                 samples: int, key_samples: int, no_idkll: bool,
                 timeout: float | None) -> dict:
    """Invokes VerifyLock.py as a subprocess and parses the results."""
    cmd = [
        sys.executable, str(verifier),
        "--orig",    str(orig),
        "--locked",  str(locked),
        "--key-map", str(key_map),
        "--samples", str(samples),
        "--key-samples", str(key_samples),
    ]
    if no_idkll:
        cmd.append("--no-idkll")

    t0 = time.perf_counter()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        dt = time.perf_counter() - t0
        return {"ok": False, "reason": "timeout",
                "tail": [f"verify timed out after {dt:.1f}s"],
                "duration": dt}
    except Exception as exc:
        dt = time.perf_counter() - t0
        return {"ok": False, "reason": "exception",
                "tail": [f"verify raised: {exc}"],
                "duration": dt}
    dt = time.perf_counter() - t0

    out_lines = (res.stdout or "").splitlines()
    err_lines = (res.stderr or "").splitlines()
    tail = out_lines[-12:] + (["--- stderr ---"] + err_lines[-4:]
                              if err_lines else [])

    if res.returncode == 0:
        return {"ok": True, "reason": "pass",
                "tail": tail, "duration": dt}
    return {"ok": False,
            "reason": f"verify rc={res.returncode}",
            "tail": tail, "duration": dt}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Multithreaded batch-lock every Yosys JSON in a directory with "
                    "GateLock.py over a range of locking percentages, "
                    "optionally verifying each result with VerifyLock.py.")
    
    ap.add_argument("--workers", type=int, default=23,
                    help="Number of concurrent worker threads to distribute the load across. Defaults to 23.")
    ap.add_argument("--in-dir",  required=True, help="Input directory of Yosys JSON netlists.")
    ap.add_argument("--out-dir", required=True, help="Output directory for locked netlists and reports.")
    ap.add_argument("--circuit", action="append", default=None, help="Specific circuit file names or stems to evaluate (e.g., --circuit c17.json c499). If omitted, evaluates all in directory.")
    ap.add_argument("--lower", type=float, required=True, help="Starting locking percentage.")
    ap.add_argument("--upper", type=float, required=True, help="Ending locking percentage.")
    ap.add_argument("--step",  type=float, required=True, help="Percentage increment size.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Global seed. Omit for one random seed per circuit.")
    ap.add_argument("--variant-policy", default="random",
                    help="Forwarded to GateLock.py (paper|random|lightest|heaviest|fixed:<i>).")
    ap.add_argument("--key-port", default="lock_key", help="Name of the new key input port.")
    ap.add_argument("--module", default=None, help="Specific target module to lock.")
    ap.add_argument("--no-nand-nor", action="store_true", help="Exclude NAND/NOR gates.")
    ap.add_argument("--skip-existing", action="store_true", help="Bypass if output JSON already exists.")
    ap.add_argument("--dry-run", action="store_true", help="Simulate without writing or executing.")
    ap.add_argument("--gatelock", default=None, help="Explicit path to GateLock.py.")
    ap.add_argument("--meta-seed", type=int, default=None, help="Seed for the per-circuit-seed RNG.")

    # ---- verification flags ------------------------------------------
    vgrp = ap.add_argument_group("verification (off by default)")
    vgrp.add_argument("--verify", action="store_true",
                      help="After lock, run VerifyLock.py and log failures.")
    vgrp.add_argument("--verifier", default=None, help="Explicit path to VerifyLock.py.")
    vgrp.add_argument("--verify-samples", type=int, default=2000, help="Forwarded to VerifyLock.py as --samples.")
    vgrp.add_argument("--verify-key-samples", type=int, default=1024, help="Forwarded to VerifyLock.py as --key-samples.")
    vgrp.add_argument("--verify-no-idkll", action="store_true", help="Skip the IDKLL global-key check inside VerifyLock.py.")
    vgrp.add_argument("--verify-timeout", type=float, default=None, help="Per-circuit verification timeout.")
    vgrp.add_argument("--keep-failed", action="store_true", help="KEEP locked files even if verification fails.")
    
    args = ap.parse_args()

    in_dir  = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not in_dir.is_dir():
        raise SystemExit(f"--in-dir {in_dir} is not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    gatelock_script = find_sibling("GateLock.py", args.gatelock)
    verifier_script = find_sibling("VerifyLock.py", args.verifier) if args.verify else None

    percentages = frange(args.lower, args.upper, args.step)
    if not percentages:
        raise SystemExit("empty percentage schedule")

    files = [p for p in discover_inputs(in_dir, args.circuit) if is_yosys_json(p)]
    if not files:
        raise SystemExit(f"no Yosys JSON files found in {in_dir}")

    if args.seed is not None:
        seed_of: dict = {p.stem: args.seed for p in files}
        seed_mode = f"global seed = {args.seed}"
    else:
        meta_seed = args.meta_seed if args.meta_seed is not None else random.randrange(2**31)
        rng = random.Random(meta_seed)
        seed_of = {p.stem: rng.randrange(2**31) for p in files}
        seed_mode = (f"random per-circuit seeds (meta_seed={meta_seed}); recorded in seeds.json/csv")

    # Header report ------------------------------------------------------
    print("=" * 78)
    print(" BatchLock.py (Multithreaded)")
    print("=" * 78)
    print(f"  input dir       : {in_dir}")
    print(f"  output dir      : {out_dir}")
    print(f"  circuits found  : {len(files)}")
    print(f"  percentages     : {percentages}")
    print(f"  workers         : {args.workers}")
    print(f"  variant policy  : {args.variant_policy}")
    print(f"  seed strategy   : {seed_mode}")
    print(f"  GateLock script : {gatelock_script}")
    if args.verify:
        print(f"  verifier        : {verifier_script}")
    else:
        print("  verification    : OFF  (pass --verify to enable)")
    if args.dry_run:
        print("  (dry-run)")
    print()

    if not args.dry_run:
        with (out_dir / "metadata" / "seeds.json").open("w") as f:
            json.dump({"strategy": "global" if args.seed is not None else "per_circuit",
                       "global_seed": args.seed,
                       "meta_seed": (args.meta_seed if args.seed is None else None),
                       "seeds": seed_of}, f, indent=2)
        with (out_dir / "metadata" / "seeds.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["circuit", "seed"])
            for cname, s in seed_of.items():
                w.writerow([cname, s])

    # Threading states & Synchronization Primitives ----------------------
    tracker_locks = {fp.stem: threading.Lock() for fp in files}
    print_lock = threading.Lock()
    failures_log_lock = threading.Lock()
    
    circuit_candidates = {}
    circuit_locked_counts = collections.defaultdict(set)

    runs_path = out_dir / "runs.csv"
    failures_path = out_dir / "verification_failures.txt"
    runs_rows = []
    
    if args.verify and not args.dry_run:
        with failures_path.open("w") as f:
            f.write("# verification_failures.txt\n"
                    "# Generated by BatchLock.py --verify.\n\n")

    # --------------------------------------------------------------------
    # Worker Function
    # --------------------------------------------------------------------
    def process_combination(job_spec):
        fp, cname, pct, seed = job_spec
        label = f"{cname:<18}  p={pct:>6.2f}%  seed={seed}"
        tag_out_json = out_dir / "netlists" / f"{cname}_p{pct_tag(pct)}_s{seed}.json"
        tag_keymap   = out_dir / "keys" / f"{cname}_p{pct_tag(pct)}_s{seed}.key.json"

        # 1. Skip Existing check
        if args.skip_existing and tag_out_json.exists():
            with print_lock:
                print(f"  [skip ] {label}  (already exists: {tag_out_json.name})")
            
            # Attempt to populate the locked_gates history to help skip future duplicates
            try:
                with tag_keymap.open() as f:
                    nl = len(json.load(f).get("locked_gates", []))
                with tracker_locks[cname]:
                    circuit_locked_counts[cname].add(nl)
            except Exception:
                pass
            return [cname, pct, seed, "skipped", "", "", "", "", tag_out_json.name]

        # 2. Concurrency-Safe Duplicate Prevention
        tc = None
        expected_lock = None
        is_chosen_to_count = False

        tracker_locks[cname].acquire()
        tc = circuit_candidates.get(cname)

        if tc is not None:
            expected_lock = min(tc, max(1, round(tc * pct / 100.0)))
            if expected_lock in circuit_locked_counts[cname]:
                tracker_locks[cname].release()
                with print_lock:
                    print(f"  [skip ] {label}  (duplicate: gate count remains {expected_lock})")
                return [cname, pct, seed, "duplicate_skipped", "", expected_lock, "", "", ""]
            
            circuit_locked_counts[cname].add(expected_lock)
            tracker_locks[cname].release()
        else:
            # We are the first thread to run GateLock for this circuit.
            # Hold the lock so other threads for this circuit wait until we discover Total Candidates (TC).
            is_chosen_to_count = True

        try:
            cmd = [
                sys.executable, str(gatelock_script),
                "--in", str(fp), "--out", str(tag_out_json),
                "--lock-percent", str(pct), "--seed", str(seed),
                "--variant-policy", args.variant_policy,
                "--key-port", args.key_port, "--key-map", str(tag_keymap),
            ]
            if args.module: cmd += ["--module", args.module]
            if args.no_nand_nor: cmd += ["--no-nand-nor"]

            if args.dry_run:
                with print_lock:
                    print(f"  [dry  ] {label}  -> {tag_out_json.name}")
                return [cname, pct, seed, "dry-run", "", "", "", "", tag_out_json.name]

            # 3. Execute GateLock Subprocess
            t0 = time.perf_counter()
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except Exception as exc:
                dt = time.perf_counter() - t0
                with print_lock:
                    print(f"  [ERR  ] {label}  exception: {exc}")
                return [cname, pct, seed, "exception", "", "", f"{dt:.3f}", "", tag_out_json.name]
            dt_lock = time.perf_counter() - t0

            if res.returncode != 0:
                with print_lock:
                    print(f"  [FAIL ] {label}  rc={res.returncode}  (lock)")
                    for line in (res.stderr or "").strip().splitlines()[-3:]:
                        print(f"           ! {line}")
                return [cname, pct, seed, "lock_fail", "", "", f"{dt_lock:.3f}", "", tag_out_json.name]

            # Parse Keymap
            key_width = num_locked = ""
            try:
                with tag_keymap.open() as f:
                    sc = json.load(f)
                key_width = sc.get("key_width", "")
                num_locked = len(sc.get("locked_gates", []))
            except Exception:
                pass

            # Populate total candidates so blocked threads can be unblocked
            if is_chosen_to_count:
                m = re.search(r"(\d+) candidate gates", res.stderr)
                if m:
                    circuit_candidates[cname] = int(m.group(1))
                if num_locked != "":
                    circuit_locked_counts[cname].add(int(num_locked))

        finally:
            # Always ensure the lock is released, even if subprocess crashes
            if is_chosen_to_count:
                tracker_locks[cname].release()

        # 4. Verification (Safe to run concurrently; we no longer hold the circuit lock)
        verify_dt = 0.0
        if args.verify:
            vres = run_verifier(verifier_script, fp, tag_out_json, tag_keymap,
                                samples=args.verify_samples, key_samples=args.verify_key_samples,
                                no_idkll=args.verify_no_idkll, timeout=args.verify_timeout)
            verify_dt = vres["duration"]

            if vres["ok"]:
                with print_lock:
                    print(f"  [ ok  ] {label}  k={key_width:>4} #gates={num_locked:<5} lock={dt_lock:5.2f}s verify={verify_dt:5.2f}s  ✓  -> {tag_out_json.name}")
                return [cname, pct, seed, "ok", key_width, num_locked, f"{dt_lock:.3f}", f"{verify_dt:.3f}", tag_out_json.name]
            else:
                verify_status = vres["reason"]
                with print_lock:
                    print(f"  [VFAIL] {label}  k={key_width:>4} #gates={num_locked:<5} lock={dt_lock:5.2f}s verify={verify_dt:5.2f}s  ✗ ({verify_status})")
                    for line in vres["tail"][-4:]:
                        print(f"           ! {line.rstrip()}")

                with failures_log_lock:
                    with failures_path.open("a") as f:
                        f.write(f"==== {tag_out_json.name} ({verify_status}) ====\n")
                        f.write(f"  circuit       : {cname}\n  lock_percent  : {pct}\n  seed          : {seed}\n")
                        f.write(f"  key_width     : {key_width}\n  locked_gates  : {num_locked}\n  verify_runtime: {verify_dt:.3f}s\n")
                        f.write("  --- VerifyLock.py tail ---\n")
                        for line in vres["tail"]:
                            f.write(f"    {line.rstrip()}\n")
                        f.write("\n")

                if not args.keep_failed:
                    for badp in (tag_out_json, tag_keymap):
                        try: badp.unlink()
                        except OSError: pass

                return [cname, pct, seed, "verify_fail", key_width, num_locked, f"{dt_lock:.3f}", f"{verify_dt:.3f}", tag_out_json.name]
        else:
            with print_lock:
                print(f"  [ ok  ] {label}  k={key_width:>4} #gates={num_locked:<5} {dt_lock:5.2f}s  -> {tag_out_json.name}")
            return [cname, pct, seed, "ok", key_width, num_locked, f"{dt_lock:.3f}", "", tag_out_json.name]

    # Build the jobs list. Sorting ensures we run the lowest pct for each circuit first, 
    # making the double-checked locking highly efficient.
    jobs = [(fp, fp.stem, pct, seed_of[fp.stem]) for fp in files for pct in percentages]
    jobs.sort(key=lambda x: (x[1], x[2])) 
    
    # Process Load distribution via ThreadPool
    n_ok = n_fail_lock = n_fail_verify = n_skip = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_combination, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            runs_rows.append(row)
            
            status = row[3]
            if status in ("skipped", "duplicate_skipped"): n_skip += 1
            elif status == "ok": n_ok += 1
            elif status in ("exception", "lock_fail"): n_fail_lock += 1
            elif status == "verify_fail": n_fail_verify += 1

    # Write CSV cleanly
    runs_rows.sort(key=lambda x: (x[0], x[1])) 
    if not args.dry_run:
        with runs_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["circuit", "lock_percent", "seed", "status", "key_width", "locked_gates", 
                        "lock_runtime_sec", "verify_runtime_sec", "output_file"])
            w.writerows(runs_rows)

    print("\n" + "-" * 78)
    summary = f"  done: {n_ok} ok, {n_fail_lock} lock-failed"
    if args.verify: summary += f", {n_fail_verify} verify-failed"
    print(f"{summary}, {n_skip} skipped, {len(runs_rows)} total combinations.")
    
    if not args.dry_run:
        print(f"  bookkeeping: {out_dir/'metadata/seeds.json'}, {out_dir/'metadata/seeds.csv'}, {runs_path}")
        if args.verify:
            print(f"  verification report: {failures_path} "
                  f"({'empty — all passed' if n_fail_verify == 0 else str(n_fail_verify) + ' failures listed'})")
                  
    sys.exit(0 if (n_fail_lock == 0 and n_fail_verify == 0) else 1)

if __name__ == "__main__":
    main()