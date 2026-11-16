#!/usr/bin/env python3
"""
GateLock.py — Apply GateLock (IDKLL) logic locking to a Yosys JSON netlist,
drawing locked-gate variants from `Database.VARIANTS`.

Each chosen design gate is replaced by an IDKLL-locked sub-network picked from 
the corresponding entry in `Database.VARIANTS[gate_type]`. Different variants can 
use different key widths (2, 3, 4 bits ...), so the script allocates a fresh slice 
of the new primary input bus `lock_key[...]` per replaced gate. Sidecar JSON 
records which slice belongs to which gate and what its valid IDKLL key sequences are.

Reference
---------
V. S. Rathor, M. Singh, K. S. Sahoo, S. P. Mohanty,
"GateLock: Input-Dependent Key-Based Locked Gates for SAT Resistant
 Logic Locking", IEEE TVLSI, 2024.  DOI: 10.1109/TVLSI.2023.3340350
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import Database  # local file Database.py


# ----------------------------------------------------------------------
# Cell-library detection
# ----------------------------------------------------------------------

INTERNAL_TYPES = {
    "$_AND_":  "AND",
    "$_NAND_": "NAND",
    "$_OR_":   "OR",
    "$_NOR_":  "NOR",
    "$_XOR_":  "XOR",
    "$_XNOR_": "XNOR",
    "$_NOT_":  "NOT",
}

RTLIL_TYPES = {
    "$and":  "AND",
    "$or":   "OR",
    "$xor":  "XOR",
    "$xnor": "XNOR",
    "$not":  "NOT",
}

ONE_INPUT_KINDS = {"NOT", "BUF"}


def detect_style(module: dict) -> str:
    """
    Detects the cell library style used in the Yosys module.

    Parameters:
        module (dict): A dictionary representing a single Yosys module.

    Returns:
        str: 'rtlil' if the module uses RTLIL cell types (e.g., '$and'), 
             otherwise defaults to 'internal' (e.g., '$_AND_').
    """
    saw_internal = saw_rtlil = False
    for cell in module.get("cells", {}).values():
        t = cell.get("type", "")
        if t in INTERNAL_TYPES:
            saw_internal = True
        elif t in RTLIL_TYPES:
            saw_rtlil = True
    if saw_rtlil and not saw_internal:
        return "rtlil"
    return "internal"


def _param_int(val) -> int:
    """
    Converts a Yosys parameter value to an integer.

    Yosys JSON can represent integers as actual ints, string representations 
    of base-10 integers, or binary strings.

    Parameters:
        val (int | str): The parameter value to convert.

    Returns:
        int: The integer representation of the value.
    """
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val, 2)
        except ValueError:
            return int(val)
    return int(val)


def cell_kind(cell: dict, style: str) -> str | None:
    """
    Determines the generic logic kind of a specific cell based on the detected style.

    Parameters:
        cell (dict): A dictionary representing a single cell/gate.
        style (str): The detected style ('internal' or 'rtlil').

    Returns:
        str | None: The generic logic kind (e.g., 'AND', 'NOT') or None if the 
                    cell is not a recognizable single-bit logic gate.
    """
    t = cell.get("type", "")
    if style == "internal":
        return INTERNAL_TYPES.get(t)
    if t in RTLIL_TYPES:
        p = cell.get("parameters", {})
        is_unary = t == "$not"
        if is_unary:
            if (_param_int(p.get("A_WIDTH", 1)) == 1 and
                    _param_int(p.get("Y_WIDTH", 1)) == 1):
                return RTLIL_TYPES[t]
            return None
        # Verify it's a single-bit gate
        if (_param_int(p.get("A_WIDTH", 1)) == 1 and
                _param_int(p.get("B_WIDTH", 1)) == 1 and
                _param_int(p.get("Y_WIDTH", 1)) == 1):
            return RTLIL_TYPES[t]
    return None


# ----------------------------------------------------------------------
# Bit-id management
# ----------------------------------------------------------------------

def max_bit_id(design: dict) -> int:
    """
    Scans the entire Yosys JSON design to find the highest integer bit ID currently in use.

    This is required to safely allocate new wire IDs without colliding with existing nets.

    Parameters:
        design (dict): The complete Yosys JSON design.

    Returns:
        int: The maximum integer bit ID found in the design.
    """
    biggest = 1

    def scan(bits):
        nonlocal biggest
        for b in bits:
            if isinstance(b, int) and b > biggest:
                biggest = b

    for mod in design["modules"].values():
        for p in mod.get("ports", {}).values():
            scan(p["bits"])
        for n in mod.get("netnames", {}).values():
            scan(n["bits"])
        for c in mod.get("cells", {}).values():
            for conn in c.get("connections", {}).values():
                scan(conn)
    return biggest


# ----------------------------------------------------------------------
# Cell emitter
# ----------------------------------------------------------------------

class CellEmitter:
    """
    A utility class to dynamically generate and wire new logic cells in a Yosys module.

    Parameters:
        module (dict): The target Yosys module where cells will be added.
        style (str): The detected cell style ('internal' or 'rtlil').
        next_bit (int): The starting point for allocating new bit IDs.
        name_prefix (str, optional): Prefix for new cell names. Defaults to "lk".
    """
    def __init__(self, module: dict, style: str, next_bit: int,
                 name_prefix: str = "lk"):
        self.module = module
        self.style = style
        self.next_bit = next_bit
        self.prefix = name_prefix
        self.counter = 0
        self.not_cache: dict = {}

    def new_bit(self) -> int:
        """Allocates and returns a fresh, unused bit ID for net generation."""
        self.next_bit += 1
        return self.next_bit

    def new_name(self, kind: str) -> str:
        """Generates a unique cell name within the module."""
        self.counter += 1
        name = f"{self.prefix}_{kind}_{self.counter}"
        # Ensure no accidental collisions with existing user cells
        while name in self.module["cells"]:
            self.counter += 1
            name = f"{self.prefix}_{kind}_{self.counter}"
        return name

    def add_cell(self, kind: str, inputs: list[int], output: int):
        """
        Instantiates a new logic cell into the module dictionary.

        Parameters:
            kind (str): The logical operation ("AND", "OR", "NOT").
            inputs (list[int]): List of input bit IDs.
            output (int): The output bit ID.
        """
        name = self.new_name(kind.lower())
        if self.style == "internal":
            if kind == "NOT":
                cell = {
                    "hide_name": 1, "type": "$_NOT_",
                    "parameters": {}, "attributes": {},
                    "port_directions": {"A": "input", "Y": "output"},
                    "connections": {"A": [inputs[0]], "Y": [output]},
                }
            else:
                cell = {
                    "hide_name": 1,
                    "type": "$_AND_" if kind == "AND" else "$_OR_",
                    "parameters": {}, "attributes": {},
                    "port_directions": {"A": "input", "B": "input",
                                        "Y": "output"},
                    "connections": {"A": [inputs[0]], "B": [inputs[1]],
                                    "Y": [output]},
                }
        else:
            if kind == "NOT":
                cell = {
                    "hide_name": 1, "type": "$not",
                    "parameters": {"A_SIGNED": "0",
                                   "A_WIDTH": "1",
                                   "Y_WIDTH": "1"},
                    "attributes": {},
                    "port_directions": {"A": "input", "Y": "output"},
                    "connections": {"A": [inputs[0]], "Y": [output]},
                }
            else:
                cell = {
                    "hide_name": 1,
                    "type": "$and" if kind == "AND" else "$or",
                    "parameters": {"A_SIGNED": "0", "A_WIDTH": "1",
                                   "B_SIGNED": "0", "B_WIDTH": "1",
                                   "Y_WIDTH": "1"},
                    "attributes": {},
                    "port_directions": {"A": "input", "B": "input",
                                        "Y": "output"},
                    "connections": {"A": [inputs[0]], "B": [inputs[1]],
                                    "Y": [output]},
                }
        self.module["cells"][name] = cell

    # ---- AST realisation ------------------------------------------------
    def realise(self, expr, leaves: dict):
        """
        Recursively translates an AST expression into hardware cells.

        Parameters:
            expr: The AST expression to realize (from Database.py).
            leaves (dict): Mapping of leaf names (like 'A', 'K1') to their wire bit IDs.

        Returns:
            int | tuple: A bit ID holding the value of `expr`, or a ('const', 0|1) tuple.
        """
        if isinstance(expr, str):
            return leaves[expr]
        op = expr[0]
        if op == "const":
            return expr  # propagate as a constant marker
        
        # Optimize INVERTER chains via cache
        if op == "not":
            sub = self.realise(expr[1], leaves)
            if isinstance(sub, tuple) and sub[0] == "const":
                return ("const", 1 - sub[1])
            if sub in self.not_cache:
                return self.not_cache[sub]
            out = self.new_bit()
            self.add_cell("NOT", [sub], out)
            self.not_cache[sub] = out
            return out

        # AND / OR Logic Generation
        bits = []
        for t in expr[1:]:
            v = self.realise(t, leaves)
            if isinstance(v, tuple) and v[0] == "const":
                if op == "and" and v[1] == 0:
                    return ("const", 0)
                if op == "or" and v[1] == 1:
                    return ("const", 1)
                # neutral element — drop it
                continue
            bits.append(v)
            
        if not bits:
            # all operands were neutral
            return ("const", 1 if op == "and" else 0)
            
        kind = "AND" if op == "and" else "OR"
        cur = bits[0]
        # Chain multi-input operations into 2-input gates
        for nxt in bits[1:]:
            out = self.new_bit()
            self.add_cell(kind, [cur, nxt], out)
            cur = out
        return cur

    def realise_into(self, expr, leaves: dict, final_bit: int):
        """
        Realises `expr` and drives its final output into a pre-allocated `final_bit`. 
        
        Handles constant outputs safely to avoid breaking original net drivers by
        modeling a simple buffer if needed.

        Parameters:
            expr: The AST expression.
            leaves (dict): The variable-to-wire mapping.
            final_bit (int): The destination bit ID to drive the result into.
        """
        result = self.realise(expr, leaves)
        if isinstance(result, tuple) and result[0] == "const":
            # Drive `final_bit` to a constant by OR-ing a fresh wire with
            # itself, where the wire is forced 0 or 1.
            const_bit = self._make_const_bit(result[1])
            self.add_cell("OR", [const_bit, const_bit], final_bit)
            return
        if result == final_bit:
            return
        # Buffer: final_bit = result | result
        self.add_cell("OR", [result, result], final_bit)

    def _make_const_bit(self, value: int) -> int:
        """
        Create an internal bit that always holds `value`.  We do this
        by feeding the Yosys-JSON constant strings '0'/'1' through a NOT
        of a NOT — but the cleanest portable way is just to allocate a
        fresh bit driven by AND(x,!x) / OR(x,!x) of some existing wire.
        Here we use the simplest trick: ask for a buffered constant via
        a dedicated cell that has '0'/'1' on its input.
        Parameters:
            value (int): 0 or 1.

        Returns:
            int: The new bit ID carrying the constant value.
        """
        out = self.new_bit()
        # NOT of constant string driver — Yosys accepts string bits in
        # cell connections, so we can wire '0'/'1' directly into a NOT.
        name = self.new_name("constdrv")
        if self.style == "internal":
            self.module["cells"][name] = {
                "hide_name": 1, "type": "$_NOT_",
                "parameters": {}, "attributes": {},
                "port_directions": {"A": "input", "Y": "output"},
                "connections": {"A": [str(1 - value)], "Y": [out]},
            }
        else:
            self.module["cells"][name] = {
                "hide_name": 1, "type": "$not",
                "parameters": {"A_SIGNED": "0",
                               "A_WIDTH": "1", "Y_WIDTH": "1"},
                "attributes": {},
                "port_directions": {"A": "input", "Y": "output"},
                "connections": {"A": [str(1 - value)], "Y": [out]},
            }
        return out


# ----------------------------------------------------------------------
# Variant selection policies
# ----------------------------------------------------------------------

def make_variant_chooser(policy: str, rng: random.Random):
    """
    Returns a selection function that dictates how locked gate variants are chosen.

    This factory function reads the requested policy and returns a specialized 
    `chooser(gate_kind)` function. The generated chooser is then used during the 
    locking phase to pick the specific IDKLL sub-network to replace a target gate.

    Available Policies:
        * `random`: Uniformly picks a random variant from the database for the 
          given gate type. Uses the seeded RNG to ensure reproducible builds.
        * `paper`: Always selects the baseline variant (index 0) originally 
          proposed in the reference Rathor et al. paper.
        * `lightest`: Dynamically analyzes all available variants for a gate type 
          and selects the one with the fewest logical operations. This is ideal 
          when you need to minimize area and power overhead.
        * `heaviest`: Selects the variant with the absolute most logical operations, 
          maximizing the obfuscation footprint and structural complexity.
        * `fixed:<index>`: Bypasses all dynamic checks and strictly forces the 
          selection of the variant at the specified zero-based integer index 
          (e.g., `fixed:2`). Useful for strict, manual architectural control.

    Parameters:
        policy (str): The variant selection policy to apply.
        rng (random.Random): Instantiated random number generator.

    Returns:
        Callable[[str],LockedVariant]: A function that takes a logic kind 
                                        (e.g., `AND`) and returns the chosen 
                                        `LockedVariant` object.
    
    Raises:
        SystemExit: If an unknown policy string is requested, or if a `fixed:` 
                    index is out of bounds for the requested gate type.
    """
    if policy.startswith("fixed:"):
        try:
            idx = int(policy.split(":", 1)[1])
        except ValueError:
            raise SystemExit(f"bad --variant-policy {policy!r}")

        def chooser(kind):
            variants = Database.VARIANTS[kind]
            if idx >= len(variants):
                raise SystemExit(
                    f"--variant-policy fixed:{idx} but only "
                    f"{len(variants)} variants exist for {kind}")
            return variants[idx]
        return chooser

    if policy == "paper":
        def chooser(kind):
            return Database.VARIANTS[kind][0]   # V0 == paper-original
        return chooser

    if policy == "random":
        def chooser(kind):
            return rng.choice(Database.VARIANTS[kind])
        return chooser

    if policy == "lightest":
        def chooser(kind):
            return min(Database.VARIANTS[kind],
                       key=lambda v: Database._count_ops(v.expression))
        return chooser

    if policy == "heaviest":
        def chooser(kind):
            return max(Database.VARIANTS[kind],
                       key=lambda v: Database._count_ops(v.expression))
        return chooser

    raise SystemExit(
        f"unknown --variant-policy {policy!r}; "
        "use paper | random | lightest | heaviest | fixed:<index>")


# ----------------------------------------------------------------------
# Main locking routine
# ----------------------------------------------------------------------

def lock_design(design: dict, *, module_name: str, num_gates: int, percent: float, seed: int,
                key_port_name: str, variant_policy: str, skip_nand_nor: bool, name_prefix: str = "lk"):
    """
    The core logic locking routine. It iterates over a Yosys netlist, identifies 
    candidate gates, swaps them for IDKLL-locked sub-networks, and builds the sidecar map.

    Parameters:
        design (dict): The complete Yosys JSON design structure.
        module_name (str): The specific module within the design to lock (None defaults to the first non-blackbox).
        num_gates (int): Explicit number of gates to lock.
        percent (float): Percentage of candidate gates to lock (used if num_gates is None).
        seed (int): RNG seed for reproducible locking decisions.
        key_port_name (str): The name to assign to the new top-level key input port.
        variant_policy (str): The strategy for selecting locking architectures.
        skip_nand_nor (bool): Flag to exclude NAND and NOR gates from being candidates.
        name_prefix (str, optional): Prefix for new cell names. Defaults to "lk".

    Returns:
        tuple: A 4-tuple containing:
            - dict: The modified Yosys design JSON.
            - list[dict]: The generated sidecar mapping of locking keys/sequences.
            - str: The name of the module that was locked.
            - str: The name of the injected key port.
    
    Raises:
        SystemExit: If the target module is missing or no valid candidate gates exist.
    """
    rng = random.Random(seed)
    mods = design["modules"]

    if module_name is None:
        non_blackbox = [n for n, m in mods.items()
                        if not m.get("attributes", {}).get("blackbox")]
        if not non_blackbox:
            raise SystemExit("No non-blackbox modules found in input JSON.")
        module_name = non_blackbox[0]
    if module_name not in mods:
        raise SystemExit(f"Module '{module_name}' not found in input JSON.")
    
    module = mods[module_name]
    style = detect_style(module)
    chooser = make_variant_chooser(variant_policy, rng)

    # 1. Map candidate gates
    candidates = []
    for cname, cell in module.get("cells", {}).items():
        kind = cell_kind(cell, style)
        if kind is None:
            continue
        if skip_nand_nor and kind in ("NAND", "NOR"):
            continue
        if kind not in Database.VARIANTS:
            continue  # no variants for this gate type in the database
        candidates.append(cname)

    if not candidates:
        raise SystemExit(
            f"No replaceable gates found in module '{module_name}' "
            f"(style={style}).  Make sure the design has been mapped to "
            "single-bit gate cells with, e.g.\n"
            '    yosys -p "synth; abc -g AND,OR,XOR,NAND,NOR,XNOR; '
            'write_json out.json"')

    # 2. Determine target count and choose variants
    if num_gates is None:
        num_gates = max(1, round(len(candidates) * percent / 100.0))
    num_gates = min(num_gates, len(candidates))

    chosen = rng.sample(candidates, num_gates)

    # Pick the variant per chosen gate up-front so we know the total key width needed
    variant_per_gate = []
    total_key_bits = 0
    for cname in chosen:
        kind = cell_kind(module["cells"][cname], style)
        v = chooser(kind)
        variant_per_gate.append(v)
        total_key_bits += v.key_width

    print(f"[GateLock] module '{module_name}' (style={style}): "
          f"{len(candidates)} candidate gates, locking {num_gates}; "
          f"total key width = {total_key_bits} bits.", file=sys.stderr)

    # 3. Allocate key bits & expose the bus as a primary input
    next_bit = max_bit_id(design)
    key_bits = []
    for _ in range(total_key_bits):
        next_bit += 1
        key_bits.append(next_bit)

    module.setdefault("ports", {})[key_port_name] = {
        "direction": "input",
        "bits": list(key_bits),
    }
    module.setdefault("netnames", {})[key_port_name] = {
        "hide_name": 0,
        "bits": list(key_bits),
        "attributes": {},
    }

    emitter = CellEmitter(module, style, next_bit, name_prefix)

    # 4. Replace Original Gates with Sub-networks
    key_map = []
    bit_cursor = 0
    for cname, variant in zip(chosen, variant_per_gate):
        cell = module["cells"][cname]
        kind = cell_kind(cell, style)
        slice_bits = key_bits[bit_cursor: bit_cursor + variant.key_width]
        bit_cursor += variant.key_width

        # Wire up the AST leaves
        leaves = {f"K{i+1}": slice_bits[i] for i in range(variant.key_width)}
        if variant.num_inputs == 1:
            a_bit = cell["connections"]["A"][0]
            leaves["A"] = a_bit
            a_for_sidecar = a_bit
            b_for_sidecar = None
        else:
            a_bit = cell["connections"]["A"][0]
            b_bit = cell["connections"]["B"][0]
            leaves["A"] = a_bit
            leaves["B"] = b_bit
            a_for_sidecar = a_bit
            b_for_sidecar = b_bit
        y_bit = cell["connections"]["Y"][0]

        # Remove the original gate and synthesise the locked sub-network
        del module["cells"][cname]
        emitter.realise_into(variant.expression, leaves, y_bit)

        # Translate the variant's valid key map into a sidecar entry.
        sidecar_valid = []
        for inp, key in variant.valid_key_map.items():
            entry = {
                "gate_input_A": inp[0],
                "valid_key_sequence": "".join(str(b) for b in key),
                "lock_key_bit_indices": list(range(
                    bit_cursor - variant.key_width, bit_cursor)),
            }
            if variant.num_inputs == 2:
                entry["gate_input_B"] = inp[1]
            sidecar_valid.append(entry)

        key_map.append({
            "locked_cell": cname,
            "original_type": cell["type"],
            "logic_kind": kind,
            "variant_partition": variant.partition_desc,
            "variant_fill_strategy": variant.fill_strategy,
            "key_width": variant.key_width,
            "A_bit": a_for_sidecar,
            "B_bit": b_for_sidecar,
            "Y_bit": y_bit,
            "K_bits": list(slice_bits),
            "valid_key_sequences": sidecar_valid,
        })

    return design, key_map, module_name, key_port_name


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    """Parses CLI arguments, invokes the lock_design function, and saves outputs."""
    ap = argparse.ArgumentParser(
        description="Apply GateLock (IDKLL) logic locking to a Yosys JSON "
                    "netlist using variants drawn from Database.VARIANTS."
    )
    
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Name (with path) of the input Yosys JSON netlist file.")
    
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Name (with path) of the output Yosys JSON netlist file.")
    
    # Mutually exclusive group: You can specify either a fixed number or a percentage, but not both.
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--num-gates", type=int,
                     help="The exact number of candidate logic gates to lock.")
    grp.add_argument("--lock-percent", type=float,
                     help="The percentage (0-100) of candidate logic gates to lock.")
    
    ap.add_argument("--module", default=None,
                    help="The specific module within the design to lock. "
                         "If not specified, defaults to the first non-blackbox module.")
    
    ap.add_argument("--seed", type=int, default=0,
                    help="Integer seed for the random number generator to ensure "
                         "reproducible locking across runs. Default is 0.")
    
    ap.add_argument("--key-port", default="lock_key",
                    help="The name to assign to the new top-level input port "
                         "that will receive the locking keys. Default is 'lock_key'.")
    
    ap.add_argument("--key-map", default=None,
                    help="Path to save the generated sidecar JSON file containing "
                         "the valid key mapping. If omitted, defaults to the output "
                         "path with a '.key.json' extension.")
    
    ap.add_argument("--variant-policy", default="random",
                    help="How to pick a locked variant per chosen gate: "
                         "'paper' (always V0), 'random', 'lightest', "
                         "'heaviest', or 'fixed:<index>'. Default is 'random'.")
    
    ap.add_argument("--no-nand-nor", action="store_true",
                    help="Flag to exclude NAND and NOR cells from being locked.")
    
    ap.add_argument("--name-prefix", help="Prefix to be appended before the counter for  new cell names.")
    args = ap.parse_args()
    args = ap.parse_args()

    with open(args.in_path) as f:
        design = json.load(f)

    locked, key_map, mod_name, key_port = lock_design(
        design,
        module_name=args.module,
        num_gates=args.num_gates,
        percent=args.lock_percent,
        seed=args.seed,
        key_port_name=args.key_port,
        variant_policy=args.variant_policy,
        skip_nand_nor=args.no_nand_nor,
        name_prefix=args.name_prefix,
    )

    with open(args.out_path, "w") as f:
        json.dump(locked, f, indent=2)

    if args.key_map is None:
        base = args.out_path
        if base.endswith(".json"):
            base = base[:-5]
        args.key_map = base + ".key.json"

    sidecar = {
        "module": mod_name,
        "key_port": key_port,
        "key_width": sum(e["key_width"] for e in key_map),
        "method": "GateLock (IDKLL), variant-driven",
        "variant_policy": args.variant_policy,
        "reference": "Rathor et al., IEEE TVLSI 2024, "
                     "DOI 10.1109/TVLSI.2023.3340350",
        "note": (
            "Each locked gate consumes its own slice of '{kp}' "
            "(see 'K_bits' per entry).  The IDKLL key is INPUT-DEPENDENT: "
            "look up 'valid_key_sequences' for that gate using the gate's "
            "current A (and B) input values to find which key sequence is "
            "valid for that input.  In a real deployment the LUT-based "
            "tamper-evident memory described in the paper would drive these "
            "key bits at run-time."
        ).format(kp=key_port),
        "locked_gates": key_map,
    }
    with open(args.key_map, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[GateLock] wrote locked netlist   -> {args.out_path}",
          file=sys.stderr)
    print(f"[GateLock] wrote key-info sidecar -> {args.key_map}",
          file=sys.stderr)


if __name__ == "__main__":
    main()