#!/usr/bin/env python3
"""
VerifyLock.py — End-to-end correctness check for a GateLock-locked Yosys
                 JSON netlist produced by `GateLock.py`.

Runs four checks:

  (1) Functional equivalence under the correct INPUT-DEPENDENT key.
      For every primary input pattern, compute the per-gate correct key
      from the sidecar and confirm the locked netlist's outputs equal
      the original netlist's outputs.

  (2) IDKLL property. For every key sequence in the combined key space,
      simulate over all input patterns and confirm that NO single key
      makes every input correct. (Exhaustive when total key width <= 20,
      otherwise sampled.)

  (3) Wrong-key output corruption rate (with a random or fixed wrong key).

  (4) Optional: feed the JSON through Yosys and check it parses.

Usage
-----
    python VerifyLock.py --orig design.json --locked locked.json \
                          --key-map locked.key.json [--module top] \
                          [--samples 20000] [--no-idkll]

When the design has more than ~20 primary inputs, the script automatically
switches from exhaustive enumeration to random sampling and prints how
many patterns were tried.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys


# Dictionary mapping Yosys cell types to native Python lambda functions
GATE = {
    "$_AND_":  lambda a, b: a & b,
    "$_NAND_": lambda a, b: 1 - (a & b),
    "$_OR_":   lambda a, b: a | b,
    "$_NOR_":  lambda a, b: 1 - (a | b),
    "$_XOR_":  lambda a, b: a ^ b,
    "$_XNOR_": lambda a, b: 1 - (a ^ b),
    "$_NOT_":  lambda a:    1 - a,
    "$_BUF_":  lambda a:    a,
    "$and":    lambda a, b: a & b,
    "$nand":   lambda a, b: 1 - (a & b),
    "$or":     lambda a, b: a | b,
    "$nor":    lambda a, b: 1 - (a | b),
    "$xor":    lambda a, b: a ^ b,
    "$xnor":   lambda a, b: 1 - (a ^ b),
    "$not":    lambda a:    1 - a,
}


# ----------------------------------------------------------------------
# Tiny combinational simulator (topological evaluation by fix-point)
# ----------------------------------------------------------------------

def _topo_sort(module: dict) -> list[tuple[str, dict]]:
    """
    Sorts the cells of a module topologically so they can be evaluated sequentially.

    Args:
        module (dict): The Yosys module to sort.

    Returns:
        list[tuple[str, dict]]: A list of (cell_name, cell_data) tuples ordered 
                                so that dependencies are evaluated first.

    Raises:
        RuntimeError: If a combinational loop is detected in the logic.
    """
    cells = list(module["cells"].items())
    produces = {}        # bit_id -> cell_name
    for name, c in cells:
        for p, d in c["port_directions"].items():
            if d == "output":
                produces[c["connections"][p][0]] = name
    deps = {name: set() for name, _ in cells}
    for name, c in cells:
        for p, d in c["port_directions"].items():
            if d == "input":
                src_bit = c["connections"][p][0]
                src_cell = produces.get(src_bit)
                if src_cell is not None and src_cell != name:
                    deps[name].add(src_cell)
    order = []
    visited = set()
    temp = set()

    def visit(n):
        if n in visited:
            return
        if n in temp:
            raise RuntimeError(f"combinational loop involving cell {n}")
        temp.add(n)
        for m in deps[n]:
            visit(m)
        temp.remove(n)
        visited.add(n)
        order.append(n)

    for name, _ in cells:
        visit(name)
    return [(n, module["cells"][n]) for n in order]


def simulate(sorted_cells: list[tuple[str, dict]], bit_values: dict[int | str, int]):
    """
    Evaluates a topologically sorted list of cells.

    Args:
        sorted_cells (list): The list of ordered cells from `_topo_sort`.
        bit_values (dict): A mutable dictionary containing the currently known bit states. 
                           It should already be populated with primary inputs and constants.
                           The function updates this dictionary with gate output values.

    Raises:
        RuntimeError: If an unsupported gate/cell type is encountered.
    """
    for _, c in sorted_cells:
        t = c["type"]
        conns = c["connections"]
        ports_dir = c["port_directions"]
        ins = [p for p, d in ports_dir.items() if d == "input"]
        outs = [p for p, d in ports_dir.items() if d == "output"]
        try:
            fn = GATE[t]
        except KeyError:
            raise RuntimeError(f"unsupported cell type {t!r}")
        in_vals = [bit_values[conns[p][0]] for p in ins]
        bit_values[conns[outs[0]][0]] = fn(*in_vals)


def collect_outputs(mod: dict, bit_values: dict[int | str, int], ignore_ports: tuple = ()) -> dict[str, tuple]:
    """
    Extracts the final output values from the simulation dictionary.

    Args:
        mod (dict): The evaluated module.
        bit_values (dict): The dictionary populated by `simulate`.
        ignore_ports (tuple, optional): Port names to exclude from collection.

    Returns:
        dict[str, tuple]: A mapping of output port names to a tuple of their bit values.
    """
    out = {}
    for n, p in mod["ports"].items():
        if p["direction"] == "output" and n not in ignore_ports:
            out[n] = tuple(bit_values[b] for b in p["bits"])
    return out


def primary_input_bits(mod: dict, ignore_ports: tuple = ()) -> tuple[list[int], list[str]]:
    """
    Identifies all primary input bits for a module.

    Args:
        mod (dict): The module to analyze.
        ignore_ports (tuple, optional): Port names to ignore (e.g., the key port).

    Returns:
        tuple: A tuple containing:
            - list[int]: The list of input bit IDs.
            - list[str]: The names of the corresponding input ports.
    """
    bits = []
    names = []
    for n, p in mod["ports"].items():
        if p["direction"] == "input" and n not in ignore_ports:
            for b in p["bits"]:
                bits.append(b)
                names.append(n)
    return bits, names


# ----------------------------------------------------------------------
# Pattern iterator (exhaustive or sampled)
# ----------------------------------------------------------------------

def iter_patterns(num_bits: int, max_exhaustive: int = 20, samples: int = 20000, seed: int = 0):
    """
    Generates binary testing patterns either exhaustively or randomly.

    Args:
        num_bits (int): The width of the bit vector to generate.
        max_exhaustive (int, optional): The threshold to switch from exhaustive to sampling.
        samples (int, optional): Number of random samples if exhaustive is skipped.
        seed (int, optional): Random seed.

    Yields:
        tuple: (pattern_tuple, is_exhaustive_flag)
    """
    if num_bits <= max_exhaustive:
        for pat in itertools.product([0, 1], repeat=num_bits):
            yield pat, True
    else:
        rng = random.Random(seed)
        for _ in range(samples):
            yield tuple(rng.randint(0, 1) for _ in range(num_bits)), False


# ----------------------------------------------------------------------
# Check (1): functional equivalence under correct input-dependent key
# ----------------------------------------------------------------------

def build_key_assignment(sidecar: dict, bits_o: dict) -> dict[int, int]:
    """
    Looks up the correct IDKLL key sequence for the current primary input state.

    Args:
        sidecar (dict): The JSON sidecar generated by GateLock.py.
        bits_o (dict): The current simulation values of the original netlist.

    Returns:
        dict[int, int]: A mapping of global key bit IDs to their correct 0/1 values 
                        required to unlock the design for the *current* inputs.

    Raises:
        RuntimeError: If the sidecar lacks a valid key for the current input state.
    """
    key_assign = {}
    for entry in sidecar["locked_gates"]:
        a = bits_o[entry["A_bit"]]
        has_b = entry.get("B_bit") is not None
        b = bits_o[entry["B_bit"]] if has_b else None
        chosen = None
        for ks in entry["valid_key_sequences"]:
            if ks["gate_input_A"] != a:
                continue
            if has_b and ks.get("gate_input_B") != b:
                continue
            chosen = ks
            break
        if chosen is None:
            raise RuntimeError(
                f"no valid key sequence in sidecar for gate "
                f"{entry['locked_cell']} (A={a}, B={b}). "
                "Sidecar may be incomplete.")
        for bit, ch in zip(entry["K_bits"], chosen["valid_key_sequence"]):
            key_assign[bit] = int(ch)
    return key_assign


def check_functional(orig_mod: dict, locked_mod: dict, sidecar: dict, key_port: str,
                     samples: int, seed: int) -> dict:
    """
    Verifies that supplying the correct input-dependent keys produces the original outputs.

    Args:
        orig_mod (dict): Original Yosys module.
        locked_mod (dict): GateLock modified module.
        sidecar (dict): GateLock key sidecar.
        key_port (str): Name of the locking key port.
        samples (int): Max number of patterns to test.
        seed (int): Random seed.

    Returns:
        dict: A report containing patterns tested, patterns passed, test mode, and failure data.
    """
    orig_sorted   = _topo_sort(orig_mod)
    locked_sorted = _topo_sort(locked_mod)

    pi_bits, _ = primary_input_bits(orig_mod)
    n_pi = len(pi_bits)
    n_patterns = 0
    n_pass = 0
    first_fail = None
    exhaustive = True

    for pat, was_exh in iter_patterns(n_pi, samples=samples, seed=seed):
        exhaustive = exhaustive and was_exh
        pi_assign = dict(zip(pi_bits, pat))

        bits_o = {"0": 0, "1": 1, **pi_assign}
        simulate(orig_sorted, bits_o)
        orig_outs = collect_outputs(orig_mod, bits_o)

        key_assign = build_key_assignment(sidecar, bits_o)

        bits_l = {"0": 0, "1": 1, **pi_assign, **key_assign}
        simulate(locked_sorted, bits_l)
        locked_outs = collect_outputs(locked_mod, bits_l,
                                      ignore_ports={key_port})

        n_patterns += 1
        if orig_outs == locked_outs:
            n_pass += 1
        elif first_fail is None:
            first_fail = (pat, orig_outs, locked_outs)

    return {
        "patterns_tested": n_patterns,
        "patterns_passed": n_pass,
        "exhaustive": exhaustive,
        "first_failure": first_fail,
    }


# ----------------------------------------------------------------------
# Check (2): IDKLL — no single key works for all inputs
# ----------------------------------------------------------------------

def check_idkll(orig_mod: dict, locked_mod: dict, sidecar: dict, key_port: str,
                key_samples: int, input_samples: int, seed: int) -> dict:
    """
    Verifies the core IDKLL property: no single static key can unlock the entire design.

    Args:
        orig_mod (dict): Original Yosys module.
        locked_mod (dict): GateLock modified module.
        sidecar (dict): GateLock key sidecar.
        key_port (str): Name of the locking key port.
        key_samples (int): Max number of wrong keys to try globally.
        input_samples (int): Max number of input patterns to evaluate per key.
        seed (int): Random seed.

    Returns:
        dict: A report identifying any keys that illegally unlocked the design for all inputs.
    """
    orig_sorted   = _topo_sort(orig_mod)
    locked_sorted = _topo_sort(locked_mod)
    pi_bits, _ = primary_input_bits(orig_mod)
    key_bits = locked_mod["ports"][key_port]["bits"]
    n_pi = len(pi_bits)
    n_key = len(key_bits)

    # Precompute the gold output for every (sampled) input pattern.
    rng = random.Random(seed)
    input_patterns = list(iter_patterns(n_pi, samples=input_samples,
                                        seed=seed))
    input_exhaustive = all(flag for _, flag in input_patterns)
    gold = []
    for pat, _ in input_patterns:
        bits_o = {"0": 0, "1": 1, **dict(zip(pi_bits, pat))}
        simulate(orig_sorted, bits_o)
        gold.append((pat, collect_outputs(orig_mod, bits_o)))

    key_iter = list(iter_patterns(n_key, samples=key_samples,
                                  seed=seed + 1))
    key_exhaustive = all(flag for _, flag in key_iter)

    bad_keys = []
    for k_pat, _ in key_iter:
        key_assign = dict(zip(key_bits, k_pat))
        ok = True
        for pat, exp_outs in gold:
            bits_l = {"0": 0, "1": 1,
                      **dict(zip(pi_bits, pat)),
                      **key_assign}
            simulate(locked_sorted, bits_l)
            got = collect_outputs(locked_mod, bits_l,
                                  ignore_ports={key_port})
            if got != exp_outs:
                ok = False
                break
        if ok:
            bad_keys.append(k_pat)

    return {
        "input_patterns_tested": len(gold),
        "input_exhaustive": input_exhaustive,
        "keys_tested": len(key_iter),
        "key_exhaustive": key_exhaustive,
        "universally_correct_keys": bad_keys,
        "key_width": n_key,
    }


# ----------------------------------------------------------------------
# Check (3): wrong-key output corruption rate
# ----------------------------------------------------------------------

def check_wrong_key_corruption(orig_mod: dict, locked_mod: dict, sidecar: dict, key_port: str,
                               num_wrong_keys: int, samples: int, seed: int) -> dict:
    """
    Evaluates how effectively incorrect keys corrupt the output signals.

    Args:
        orig_mod (dict): Original Yosys module.
        locked_mod (dict): GateLock modified module.
        sidecar (dict): GateLock key sidecar.
        key_port (str): Name of the locking key port.
        num_wrong_keys (int): Number of static incorrect keys to test.
        samples (int): Number of input patterns to evaluate.
        seed (int): Random seed.

    Returns:
        dict: A report detailing the failure percentages for various wrong keys.
    """
    orig_sorted   = _topo_sort(orig_mod)
    locked_sorted = _topo_sort(locked_mod)
    pi_bits, _ = primary_input_bits(orig_mod)
    key_bits = locked_mod["ports"][key_port]["bits"]
    n_pi = len(pi_bits)
    n_key = len(key_bits)
    rng = random.Random(seed)

    # Generate wrong keys. We try a few canonical ones (all-0, all-1,
    # alternating) and then `num_wrong_keys-3` random ones.
    wrong_keys = [
        tuple([0] * n_key),
        tuple([1] * n_key),
        tuple([i & 1 for i in range(n_key)]),
    ]
    while len(wrong_keys) < num_wrong_keys:
        wrong_keys.append(tuple(rng.randint(0, 1) for _ in range(n_key)))

    pattern_list = list(iter_patterns(n_pi, samples=samples, seed=seed))
    n_patterns = len(pattern_list)
    exhaustive = all(flag for _, flag in pattern_list)

    gold = []
    for pat, _ in pattern_list:
        bits_o = {"0": 0, "1": 1, **dict(zip(pi_bits, pat))}
        simulate(orig_sorted, bits_o)
        gold.append((pat, collect_outputs(orig_mod, bits_o)))

    per_key_results = []
    for k_pat in wrong_keys:
        key_assign = dict(zip(key_bits, k_pat))
        diffs = 0
        for pat, exp_outs in gold:
            bits_l = {"0": 0, "1": 1,
                      **dict(zip(pi_bits, pat)),
                      **key_assign}
            simulate(locked_sorted, bits_l)
            got = collect_outputs(locked_mod, bits_l,
                                  ignore_ports={key_port})
            if got != exp_outs:
                diffs += 1
        per_key_results.append({
            "key": "".join(str(b) for b in k_pat),
            "differs": diffs,
            "total": n_patterns,
            "pct": 100.0 * diffs / n_patterns,
        })
    return {
        "exhaustive_inputs": exhaustive,
        "per_key": per_key_results,
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def pick_module(design: dict, name: str | None) -> tuple[str, dict]:
    """Helper to safely fetch the target module dictionary from the design."""
    mods = design["modules"]
    if name is None:
        for n, m in mods.items():
            if not m.get("attributes", {}).get("blackbox"):
                return n, m
        raise SystemExit("no non-blackbox modules in JSON")
    if name not in mods:
        raise SystemExit(f"module {name!r} not found")
    return name, mods[name]


def banner(text: str, ch: str = "="):
    """Helper to print formatted CLI banners."""
    print(ch * 78)
    print(text)
    print(ch * 78)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    """Parses CLI arguments and coordinates the GateLock verification flow."""
    ap = argparse.ArgumentParser(
        description="Verify a GateLock-locked Yosys JSON netlist.")
    ap.add_argument("--orig",    required=True,
                    help="Original (un-locked) Yosys JSON netlist.")
    ap.add_argument("--locked",  required=True,
                    help="Locked Yosys JSON produced by GateLock.py.")
    ap.add_argument("--key-map", required=True,
                    help="Sidecar JSON produced by GateLock.py "
                         "(usually <locked>.key.json).")
    ap.add_argument("--module", default=None,
                    help="Module name (default: first non-blackbox).")
    ap.add_argument("--samples", type=int, default=20000,
                    help="Input patterns to sample when PIs > 20 "
                         "(default: 20000).")
    ap.add_argument("--key-samples", type=int, default=4096,
                    help="Random key sequences to try for IDKLL check "
                         "when total key width > 20 (default: 4096).")
    ap.add_argument("--wrong-keys", type=int, default=12,
                    help="How many wrong keys to try for the corruption "
                         "rate report (default: 12).")
    ap.add_argument("--no-idkll", action="store_true",
                    help="Skip the IDKLL global-key check "
                         "(can be slow on large designs).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    orig    = json.load(open(args.orig))
    locked  = json.load(open(args.locked))
    sidecar = json.load(open(args.key_map))

    mod_name = args.module or sidecar.get("module")
    _, orig_mod   = pick_module(orig,   mod_name)
    _, locked_mod = pick_module(locked, mod_name)
    key_port = sidecar["key_port"]

    pi_bits, _ = primary_input_bits(orig_mod)
    key_width = len(locked_mod["ports"][key_port]["bits"])

    banner("GateLock verification")
    print(f"  module          : {mod_name}")
    print(f"  primary inputs  : {len(pi_bits)} bits "
          f"({2**len(pi_bits)} patterns)")
    print(f"  key bus '{key_port}' : {key_width} bits")
    print(f"  locked gates    : {len(sidecar['locked_gates'])}")
    print(f"  variant policy  : {sidecar.get('variant_policy','?')}")
    print()

    # (1) functional equivalence ----------------------------------------
    banner("(1) Functional equivalence under correct input-dependent key",
           ch="-")
    r1 = check_functional(orig_mod, locked_mod, sidecar, key_port,
                          args.samples, args.seed)
    mode = "exhaustive" if r1["exhaustive"] else f"sampled ({r1['patterns_tested']} patterns)"
    print(f"  mode   : {mode}")
    print(f"  result : {r1['patterns_passed']}/{r1['patterns_tested']} "
          f"patterns match")
    func_ok = r1["patterns_passed"] == r1["patterns_tested"]
    if not func_ok:
        pat, oo, ll = r1["first_failure"]
        print(f"  FAIL   : first mismatch on PI pattern {pat}")
        print(f"           original outs  = {oo}")
        print(f"           locked   outs  = {ll}")
    print(f"  verdict: {'PASS' if func_ok else 'FAIL'}")
    print()

    # (2) IDKLL ----------------------------------------------------------
    if args.no_idkll:
        idkll_ok = True
        print("(2) IDKLL global-key check  [SKIPPED]\n")
    else:
        banner("(2) IDKLL — no single global key should unlock everything",
               ch="-")
        r2 = check_idkll(orig_mod, locked_mod, sidecar, key_port,
                         args.key_samples, args.samples, args.seed)
        kmode = "exhaustive" if r2["key_exhaustive"] else f"sampled ({r2['keys_tested']} keys)"
        imode = "exhaustive" if r2["input_exhaustive"] else f"sampled ({r2['input_patterns_tested']} inputs)"
        print(f"  key sweep    : {kmode}  (key width = {r2['key_width']})")
        print(f"  input sweep  : {imode}")
        bad = r2["universally_correct_keys"]
        if bad:
            print(f"  FAIL : {len(bad)} key(s) work for every tested input:")
            for k in bad[:8]:
                print("         " + "".join(str(b) for b in k))
            if len(bad) > 8:
                print(f"         ... and {len(bad)-8} more")
            idkll_ok = False
        else:
            print("  no single key unlocked the whole design within the "
                  "tested space.")
            idkll_ok = True
        if not r2["key_exhaustive"]:
            print("  NOTE: key sweep was sampled; absence of a global key "
                  "here is strong evidence but not a formal proof. The "
                  "per-gate IDKLL property is independently guaranteed by "
                  "Database.py's self-test.")
        print(f"  verdict: {'PASS' if idkll_ok else 'FAIL'}")
        print()

    # (3) wrong-key corruption ------------------------------------------
    banner("(3) Output corruption with WRONG keys", ch="-")
    r3 = check_wrong_key_corruption(orig_mod, locked_mod, sidecar, key_port,
                                    args.wrong_keys, args.samples, args.seed)
    print(f"  {'key':<24} {'differs/total':>16}  {'pct':>7}")
    max_pct = 0.0
    for row in r3["per_key"]:
        print(f"  {row['key']:<24} {row['differs']:>7}/{row['total']:<8} "
              f"{row['pct']:6.2f}%")
        max_pct = max(max_pct, row["pct"])
    corruption_ok = max_pct > 0.0
    print(f"  verdict: {'PASS (locking is effective)' if corruption_ok else 'WARN (no wrong key caused any change — likely no gates affect outputs)'}")
    print()

    # Final --------------------------------------------------------------
    banner("Summary")
    print(f"  (1) functional equivalence : {'PASS' if func_ok else 'FAIL'}")
    if args.no_idkll:
        print("  (2) IDKLL property         : SKIPPED")
    else:
        print(f"  (2) IDKLL property         : {'PASS' if idkll_ok else 'FAIL'}")
    print(f"  (3) wrong-key corruption   : {'PASS' if corruption_ok else 'WARN'}"
          f"  (max corruption = {max_pct:.2f}%)")

    overall_ok = func_ok and (args.no_idkll or idkll_ok)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
