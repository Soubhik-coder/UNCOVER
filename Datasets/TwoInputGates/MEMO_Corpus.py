#!/usr/bin/env python3
"""
Database.py — A catalogue of GateLock / IDKLL locked variants of basic gates.
This database relies entirely on basic wire splits (P_A and P_B), sweeping
through key widths from k=2 to k=5 and cycling through every fill policy to
test basic structural obfuscation overheads.

Each entry in `VARIANTS[gate_type]` is a `LockedVariant` object containing:
  * gate_type    : one of "AND", "OR", "NOT", "XOR", "XNOR"
  * key_width    : how many key bits this variant uses (varies per variant)
  * valid_key_map: input pattern  ->  the key sequence that is "correct"
                   for that pattern (input-dependent key, the heart of IDKLL)
  * expression   : a small AST realising the locked Boolean function
                   (leaves are "A", optionally "B", and "K1", "K2", ..., "Kk")
  * partition_desc: a human-readable note about how inputs are partitioned

You can use this database from `GateLock.py` to lock a Yosys-JSON netlist with a different IDKLL variant per gate (instead of the
single paper-default encoding).  Add or tweak variants by editing `_generate_*_variants()` near the bottom of this file.

How a variant is built (see `build_variant()` below)
----------------------------------------------------
For a gate with n primary inputs and a k-bit key:

  1.  The 2^n input patterns are split into m sets XS_1, ..., XS_m.
  2.  m distinct key sequences KS_1, ..., KS_m (out of 2^k possible) are chosen, one per set, and declared "valid" for that set.
  3.  A truth table over (key, input) is built row by row:
        - (KS_i, x) with x in XS_i  ->  original_gate(x)        (correct)
        - (KS_i, x) with x not in XS_i -> NOT original_gate(x)  (so KS_i cannot work for inputs outside its assigned set)
        - (K,   x)  for K not in {KS_1,...,KS_m} (invalid key) is filled according to `fill_strategy`.
  4.  The truth table is minimised via Quine-McCluskey to give a compact sum-of-products AST.
  5.  The resulting variant is verified to satisfy the two IDKLL properties.

Reference: Rathor, Singh, Sahoo, Mohanty, "GateLock: Input-Dependent Key-
Based Locked Gates for SAT Resistant Logic Locking", IEEE TVLSI 2024.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Tuple, Iterable, Any

# ----------------------------------------------------------------------
# Original gate semantics
# ----------------------------------------------------------------------


def original_gate(gate_type: str, inputs: Tuple[int, ...]) -> int:
    """
    Computes the reference Boolean value of an un-locked logic gate.

    Parameters:
        gate_type (str): The type of the gate (e.g., "AND", "NOT", "XOR").
        inputs (Tuple[int, ...]): A tuple containing the input bit values (0 or 1).

    Returns:
        int: The correctly evaluated output bit (0 or 1).

    Raises:
        ValueError: If an unsupported or unknown `gate_type` is provided.
    """
    if gate_type == "NOT":
        (a,) = inputs
        return 1 - a
    if gate_type == "BUF":
        (a,) = inputs
        return a
    a, b = inputs
    if gate_type == "AND":
        return a & b
    if gate_type == "NAND":
        return 1 - (a & b)
    if gate_type == "OR":
        return a | b
    if gate_type == "NOR":
        return 1 - (a | b)
    if gate_type == "XOR":
        return a ^ b
    if gate_type == "XNOR":
        return 1 - (a ^ b)
    raise ValueError(f"unknown gate type: {gate_type!r}")


def num_inputs_of(gate_type: str) -> int:
    """
    Returns the expected number of inputs for a given logic gate.

    Parameters:
        gate_type (str): The logic gate identifier.

    Returns:
        int: 1 for unary gates (NOT, BUF), 2 for binary gates.
    """
    return 1 if gate_type in ("NOT", "BUF") else 2


# ----------------------------------------------------------------------
# AST helpers — same dialect that GateLock.py understands
#   leaves          : strings ("A", "B", "K1", ...)
#   ("not", x)      : unary NOT
#   ("and", *xs)    : n-ary AND
#   ("or",  *xs)    : n-ary OR
#   ("const", 0|1)  : constant
# ----------------------------------------------------------------------


def eval_ast(ast: Any, env: Dict[str, int]) -> int:
    """Evaluates an AST Boolean expression against a variable assignment environment.

    Fully compatible with strictly 2-input binary tuples (op, left, right) as
    well as legacy multi-input tuples (op, *xs).

    Args:
        ast: The AST node to evaluate (string leaf, const, not, and, or).
        env: A dictionary mapping variable names (e.g., 'A', 'K1') to {0, 1}.

    Returns:
        The evaluated Boolean result (0 or 1).
    """
    if isinstance(ast, str):
        return env[ast]

    tag = ast[0]
    if tag == "const":
        return ast[1]
    if tag == "not":
        return 1 - eval_ast(ast[1], env)

    if tag == "and":
        # Strictly binary: ast[1] & ast[2]
        if len(ast) == 3:
            return eval_ast(ast[1], env) & eval_ast(ast[2], env)
        # Fallback support for n-ary nodes
        for sub in ast[1:]:
            if not eval_ast(sub, env):
                return 0
        return 1

    if tag == "or":
        # Strictly binary: ast[1] | ast[2]
        if len(ast) == 3:
            return eval_ast(ast[1], env) | eval_ast(ast[2], env)
        # Fallback support for n-ary nodes
        for sub in ast[1:]:
            if eval_ast(sub, env):
                return 1
        return 0

    raise ValueError(f"Unknown AST node: {ast!r}")


# ----------------------------------------------------------------------
# Quine-McCluskey minimisation (small / textbook)
# ----------------------------------------------------------------------
# Works fine for our cases: at most  (key_width + 2)  variables, i.e.
# up to 2^6 = 64 rows for the biggest variants we ship.


def _to_bits(n: int, num_vars: int) -> Tuple[int, ...]:
    """Helper to convert an integer to a zero-padded tuple of bits."""
    return tuple((n >> (num_vars - 1 - i)) & 1 for i in range(num_vars))


def qm_minimize(
    num_vars: int, minterms: Iterable[int], dont_cares: Iterable[int] = ()
) -> List[Tuple]:
    """
    Performs Quine-McCluskey boolean minimization to find a sum-of-products expression.

    Parameters:
        num_vars (int): The total number of boolean variables.
        minterms (Iterable[int]): Truth table row indices that evaluate to 1.
        dont_cares (Iterable[int], optional): Truth table row indices that are don't-cares. Defaults to ().

    Returns:
        List[Tuple]: A list of prime implicants representing the greedy cover of `minterms`.
                     Each implicant is a tuple of length `num_vars` with entries 0, 1, or None (don't-care).
                     Returns an empty list for constant-0, and `[all-None tuple]` for constant-1.
    """
    minterms = set(minterms)
    dont_cares = set(dont_cares)
    if not minterms:
        return []  # constant 0
    if (minterms | dont_cares) >= set(range(2**num_vars)):
        return [tuple([None] * num_vars)]  # constant 1

    # Round 0: every minterm + don't-care becomes a fully-defined implicant.
    current = {_to_bits(n, num_vars) for n in (minterms | dont_cares)}
    primes: set = set()

    while current:
        next_round: set = set()
        used: set = set()
        impls = list(current)
        for i in range(len(impls)):
            a = impls[i]
            for j in range(i + 1, len(impls)):
                b = impls[j]
                diff_pos = -1
                ok = True
                for k in range(num_vars):
                    if a[k] == b[k]:
                        continue
                    # different here.  Must both be 0/1 (not None).
                    if a[k] is None or b[k] is None:
                        ok = False
                        break
                    if diff_pos != -1:
                        ok = False
                        break
                    diff_pos = k
                if ok and diff_pos != -1:
                    merged = list(a)
                    merged[diff_pos] = None
                    next_round.add(tuple(merged))
                    used.add(a)
                    used.add(b)
        for impl in current:
            if impl not in used:
                primes.add(impl)
        current = next_round

    # Greedy cover over real minterms (don't-cares are free help, not goals).
    minterm_bits = {_to_bits(n, num_vars) for n in minterms}

    def covers(impl, mt):
        return all(i is None or i == m for i, m in zip(impl, mt))

    primes_list = list(primes)
    selected = []
    uncovered = set(minterm_bits)
    while uncovered:
        best = max(primes_list, key=lambda p: sum(1 for m in uncovered if covers(p, m)))
        selected.append(best)
        uncovered = {m for m in uncovered if not covers(best, m)}
    return selected


def _build_binary_tree(op: str, nodes: List[Any]) -> Any:
    """Combines a list of AST nodes into a balanced tree of 2-input operations.

    Recursively splits the input node list in half to maintain minimum logic
    depth (O(log2 N)), ensuring that every generated compound node is strictly a
    2-input binary tuple of the form (op, left, right).

    Args:
        op: The logic operator string, either 'and' or 'or'.
        nodes: A list of AST sub-nodes (strings or tuples) to combine.

    Returns:
        An AST node conforming to the GateLock dialect, containing strictly
        2-input operations.
    """
    if not nodes:
        # Neutral identity elements for Boolean operators
        return ("const", 0 if op == "or" else 1)
    if len(nodes) == 1:
        return nodes[0]
    if len(nodes) == 2:
        return (op, nodes[0], nodes[1])

    mid = len(nodes) // 2
    left = _build_binary_tree(op, nodes[:mid])
    right = _build_binary_tree(op, nodes[mid:])
    return (op, left, right)


def _implicant_to_ast(implicant: Tuple, var_names: List[str]) -> Any:
    """Translates a QM prime implicant tuple into a strictly 2-input AST node.

    Extracts non-None literals from the Quine-McCluskey implicant tuple (applying
    unary NOT as necessary) and composes them into a balanced binary tree of
    2-input AND gates.

    Args:
        implicant: A Quine-McCluskey prime implicant tuple containing values
          from {0, 1, None}, where None indicates a eliminated literal ('-').
        var_names: The variable name string for each bit index (e.g., ['A', 'B',
          'K1', 'K2']).

    Returns:
        An AST node representing the product term composed entirely of 2-input
        AND gates and unary NOT gates.
    """
    if all(v is None for v in implicant):
        return ("const", 1)

    literals = []
    for i, v in enumerate(implicant):
        if v is None:
            continue
        literals.append(var_names[i] if v == 1 else ("not", var_names[i]))

    if not literals:
        return ("const", 1)
    if len(literals) == 1:
        return literals[0]

    # Decompose into a balanced binary tree of 2-input AND gates
    return _build_binary_tree("and", literals)


def sop_to_ast(implicants: List[Tuple], var_names: List[str]) -> Any:
    """Translates a list of QM prime implicants into a final 2-input SOP AST.

    Converts each prime implicant into a balanced 2-input AND sub-tree, and then
    combines all generated product terms into a balanced binary tree of 2-input
    OR gates.

    Args:
        implicants: A list of Quine-McCluskey prime implicant tuples representing
          the minimal cover.
        var_names: The variable name string for each bit index (e.g., ['A', 'B',
          'K1', 'K2']).

    Returns:
        A sum-of-products AST root node composed strictly of 2-input AND and
        2-input OR operations.
    """
    if not implicants:
        return ("const", 0)

    # Convert each prime implicant into a 2-input product sub-tree
    product_terms = [_implicant_to_ast(imp, var_names) for imp in implicants]

    if len(product_terms) == 1:
        return product_terms[0]

    # Decompose the top-level sum into a balanced binary tree of 2-input OR gates
    return _build_binary_tree("or", product_terms)


# ----------------------------------------------------------------------
# LockedVariant
# ----------------------------------------------------------------------


class LockedVariant:
    """
    Represents a single IDKLL-locked structural variant of a basic logic gate.

    Attributes:
        gate_type (str): The base logic type (e.g., "AND", "OR").
        key_width (int): The number of individual key bits this variant requires.
        valid_key_map (Dict[Tuple, Tuple]): A dictionary mapping a tuple of input bits
                                            to the tuple of valid key bits required to unlock it.
        expression (tuple): The AST representation of the obfuscated logic function.
        partition_desc (str): A human-readable note describing the logic partitioning strategy.
        fill_strategy (str): How invalid key rows in the generated truth table were handled.
    """

    def __init__(
        self,
        gate_type,
        key_width,
        valid_key_map,
        expression,
        partition_desc,
        fill_strategy,
    ):
        self.gate_type = gate_type
        self.key_width = key_width
        self.valid_key_map = valid_key_map
        self.expression = expression
        self.partition_desc = partition_desc
        self.fill_strategy = fill_strategy

    @property
    def num_inputs(self) -> int:
        """int: The number of primary data inputs required for this gate."""
        return num_inputs_of(self.gate_type)

    @property
    def input_var_names(self) -> List[str]:
        """List[str]: Expected variable names for the primary inputs."""
        return ["A"] if self.num_inputs == 1 else ["A", "B"]

    @property
    def key_var_names(self) -> List[str]:
        """List[str]: Expected variable names for the key inputs."""
        return [f"K{i+1}" for i in range(self.key_width)]

    def to_dict(self):
        """Converts the variant metadata into a JSON-serializable dictionary."""
        return {
            "gate_type": self.gate_type,
            "key_width": self.key_width,
            "partition_desc": self.partition_desc,
            "fill_strategy": self.fill_strategy,
            "valid_key_map": {
                "".join(str(b) for b in inp): "".join(str(b) for b in key)
                for inp, key in self.valid_key_map.items()
            },
        }


# ----------------------------------------------------------------------
# Variant builder
# ----------------------------------------------------------------------


def build_variant(
    gate_type: str,
    key_width: int,
    valid_key_map: Dict[Tuple[int, ...], Tuple[int, ...]],
    fill_strategy: str = "complement",
    seed: int = 0,
    partition_desc: str = "",
) -> "LockedVariant":  # Assuming LockedVariant is defined elsewhere
    """
    Constructs a `LockedVariant` object from an IDKLL partition specification.

    This function generates the required obfuscated truth table by injecting valid
    keys for specific inputs, incorrectly evaluating inputs given mismatched valid keys,
    and resolving completely invalid keys using the provided `fill_strategy`.

    Parameters:
        gate_type (str): "AND", "OR", "NOT", "XOR", "XNOR", etc.
        key_width (int): The number of key bits required (k).
        valid_key_map (Dict): Maps an input pattern tuple to the valid key sequence tuple.
        fill_strategy (str, optional): Strategy for evaluating fully invalid key rows.
            Available strategies:
            - "complement": Outputs the inverse of the original gate. Maximizes Distinguishing
                            Input Patterns (DIPs) to heavily resist SAT solvers.
            - "original": Outputs the correct gate behavior anyway. (Low security, fallback).
            - "random": Assigns deterministic noise (0 or 1). Breaks structural
                                         symmetries to confuse graph-based machine learning attacks.
            - "fixed_0": Forces output to 0. Creates dense pull-down networks (AND/NOR heavy).
            - "fixed_1": Forces output to 1. Creates dense pull-up networks (OR/NAND heavy).
            - "pass_through_a": Leaks the A data input directly.
            - "pass_through_b": Leaks the B data input directly.
            - "dontcare_min": Marks invalid keys as Don't Cares ('X'). Allows logic minimization
                              (Quine-McCluskey) to drastically shrink the area overhead, but is
                              highly vulnerable to key-aliasing (SAT solvers finding alternate keys).
            Defaults to "complement".
        seed (int, optional): Random seed utilized if fill_strategy is "random". Defaults to 0.
        partition_desc (str, optional): Label explaining the variant structure. Defaults to "".

    Returns:
        LockedVariant: The generated and optimized locked logic block.

    Raises:
        ValueError: If an unknown `fill_strategy` is requested.
    """
    n_in = num_inputs_of(gate_type)
    rng = random.Random(seed)

    valid_keys = set(valid_key_map.values())
    key_to_inputs: Dict[Tuple[int, ...], set] = {}
    for inp, key in valid_key_map.items():
        key_to_inputs.setdefault(key, set()).add(inp)

    num_vars = key_width + n_in
    all_inputs = list(itertools.product([0, 1], repeat=n_in))
    all_keys = list(itertools.product([0, 1], repeat=key_width))

    minterms: List[int] = []
    dont_cares: List[int] = []

    for K in all_keys:
        for inp in all_inputs:
            row = 0
            for bit in K + inp:
                row = (row << 1) | bit

            if K in valid_keys:
                if inp in key_to_inputs[K]:
                    y = original_gate(gate_type, inp)
                else:
                    # Force this valid key to be WRONG outside its assigned
                    # input set, otherwise it could also unlock other inputs.
                    y = 1 - original_gate(gate_type, inp)
            else:
                # Handle all INVALID key sequences using the selected fill strategy
                if fill_strategy == "complement":
                    y = 1 - original_gate(gate_type, inp)
                elif fill_strategy == "original":
                    y = original_gate(gate_type, inp)
                elif fill_strategy == "random":
                    y = rng.randint(0, 1)
                elif fill_strategy == "fixed_0":
                    y = 0
                elif fill_strategy == "fixed_1":
                    y = 1
                elif fill_strategy == "pass_through_a":
                    y = inp[0]
                elif fill_strategy == "pass_through_b":
                    # Fallback to A if it's a 1-input gate (like NOT)
                    y = inp[1] if n_in > 1 else inp[0]
                elif fill_strategy == "dontcare_min":
                    dont_cares.append(row)
                    continue
                else:
                    raise ValueError(f"Unknown fill_strategy: '{fill_strategy}'")

            if y == 1:
                minterms.append(row)

    implicants = qm_minimize(num_vars, minterms, dont_cares)
    var_names = [f"K{i+1}" for i in range(key_width)] + (
        ["A"] if n_in == 1 else ["A", "B"]
    )
    expression = sop_to_ast(implicants, var_names)

    return LockedVariant(
        gate_type=gate_type,
        key_width=key_width,
        valid_key_map=valid_key_map,
        expression=expression,
        partition_desc=partition_desc,
        fill_strategy=fill_strategy,
    )


# ----------------------------------------------------------------------
# IDKLL verification
# ----------------------------------------------------------------------


def verify_variant(variant: LockedVariant) -> Tuple[bool, bool, List[Tuple[int, ...]]]:
    """
    Simulates the variant AST to ensure it strictly conforms to IDKLL properties.

    Parameters:
        variant (LockedVariant): The generated obfuscated gate to check.

    Returns:
        Tuple[bool, bool, List]: A 3-tuple containing:
            - correctness_ok (bool): True if providing the valid input-dependent key
                                     always evaluates to the original expected output.
            - idkll_ok (bool): True if NO single key sequence works for all inputs.
            - universally_correct_keys (List): A list of offending universal keys (empty if successful).
    """
    n_in = variant.num_inputs
    k = variant.key_width

    # (1) input-dependent key correctness
    correctness_ok = True
    for inp, key in variant.valid_key_map.items():
        env = {f"K{i+1}": key[i] for i in range(k)}
        env["A"] = inp[0]
        if n_in == 2:
            env["B"] = inp[1]
        if eval_ast(variant.expression, env) != original_gate(variant.gate_type, inp):
            correctness_ok = False
            break

    # (2) no single sufficient key
    sufficient = []
    for K in itertools.product([0, 1], repeat=k):
        ok = True
        for inp in itertools.product([0, 1], repeat=n_in):
            env = {f"K{i+1}": K[i] for i in range(k)}
            env["A"] = inp[0]
            if n_in == 2:
                env["B"] = inp[1]
            if eval_ast(variant.expression, env) != original_gate(
                variant.gate_type, inp
            ):
                ok = False
                break
        if ok:
            sufficient.append(K)

    return correctness_ok, len(sufficient) == 0, sufficient


def build_variant_robust(
    gate_type,
    key_width,
    valid_key_map,
    fill_strategy,
    partition_desc,
    max_retries=16,
    metaseed=0,
):
    """
    Generates a variant with fallback safety for non-deterministic fill strategies.

    If a probabilistic `fill_strategy` like 'random' or 'dontcare_min' violates IDKLL
    properties, this function will automatically retry with varying seeds (range(max_retries)),
    ultimately falling back to the safe 'complement' deterministic strategy if needed.

    Parameters:
        gate_type (str): "AND", "OR", "NOT", "XOR", "XNOR", etc.
        key_width (int): The number of key bits required (k).
        valid_key_map (Dict): Maps an input pattern tuple to the valid key sequence tuple.
        fill_strategy (str, optional): Strategy for invalid key rows ("complement", "original",
                                       "random", or "dontcare_min"). Defaults to "complement".
        partition_desc (str, optional): Label explaining the variant structure. Defaults to "".
        metaseed: Seed for the sequence that is to be tried as seed for build_variant(). Defaults to 0.

    Returns:
        LockedVariant: The generated and optimized locked logic block.
    """
    random.seed(metaseed)
    for seed in [random.random() for _ in range(max_retries)]:
        v = build_variant(
            gate_type,
            key_width,
            valid_key_map,
            fill_strategy,
            seed=seed,
            partition_desc=partition_desc,
        )
        corr, idkll, _ = verify_variant(v)
        if corr and idkll:
            return v
    return build_variant(
        gate_type,
        key_width,
        valid_key_map,
        "complement",
        seed=0,
        partition_desc=partition_desc + " [fallback]",
    )


# ----------------------------------------------------------------------
# Partition helpers
# ----------------------------------------------------------------------


def _all_inputs(n):
    """Returns a list of all binary tuples of length n."""
    return list(itertools.product([0, 1], repeat=n))


def _partition_by_A(n_in):
    """Divides inputs into two sets based on the state of bit A (index 0)."""
    return [
        [t for t in _all_inputs(n_in) if t[0] == 0],
        [t for t in _all_inputs(n_in) if t[0] == 1],
    ]


def _partition_by_B(n_in):
    """Divides 2-bit inputs into two sets based on the state of bit B (index 1)."""
    assert n_in == 2
    return [
        [t for t in _all_inputs(2) if t[1] == 0],
        [t for t in _all_inputs(2) if t[1] == 1],
    ]


def _partition_by_xor(n_in):
    """Divides 2-bit inputs into two sets based on A XOR B."""
    assert n_in == 2
    return [
        [t for t in _all_inputs(2) if (t[0] ^ t[1]) == 0],
        [t for t in _all_inputs(2) if (t[0] ^ t[1]) == 1],
    ]


def _partition_by_xnor(n_in):
    """Divides 2-bit inputs into two sets based on A XNOR B."""
    assert n_in == 2
    return [
        [t for t in _all_inputs(2) if (1 - (t[0] ^ t[1])) == 0],
        [t for t in _all_inputs(2) if (1 - (t[0] ^ t[1])) == 1],
    ]


def _partition_4way(n_in):
    """Divides inputs into individual unique partitions."""
    return [[t] for t in _all_inputs(n_in)]


def _make_variant(gate_type, key_width, partition, valid_keys, fill, desc, metaseed=0):
    """Helper method to orchestrate validation and compilation of a specific variant definition."""
    assert len(partition) == len(valid_keys), "need one valid key per input partition"
    assert all(
        len(k) == key_width for k in valid_keys
    ), "every valid key must have `key_width` bits"
    assert len(set(valid_keys)) == len(valid_keys), "valid keys must be distinct"
    valid_key_map = {}
    for input_set, key in zip(partition, valid_keys):
        for inp in input_set:
            valid_key_map[inp] = key
    return build_variant_robust(gate_type, key_width, valid_key_map, fill, desc, metaseed)


# ----------------------------------------------------------------------
# 8 variants per 2-input gate
# ----------------------------------------------------------------------


def _generate_2input_variants(gate_type: str) -> List[LockedVariant]:
    """Generates the database entries for standard binary input gates."""
    P_A = _partition_by_A(2)
    P_B = _partition_by_B(2)
    P_XOR = _partition_by_xor(2)
    P_XNOR = _partition_by_xnor(2)
    P_4 = _partition_4way(2)

    vs = [
            # V0: A-split
            _make_variant(
                gate_type,
                3,
                P_A,
                [(1, 1, 0), (0, 0, 1)],
                "complement",
                desc="k=3 | split-by-A | KS=(110,001) | complement",
            ),
            # V1: XOR-split
            _make_variant(
                gate_type,
                3,
                P_XOR,
                [(1, 0, 1), (0, 1, 0)],
                "dontcare_min",
                desc="k=3 | split-by-XOR | KS=(101,010) | dontcare_min",
            ),
            # V2: B-split
            _make_variant(
                gate_type,
                4,
                P_B,
                [(1, 0, 0, 1), (0, 1, 1, 0)],
                "fixed_0",
                desc="k=4 | split-by-B | KS=(1001,0110) | fixed_0",
            ),
            # V3: 4-way split
            _make_variant(
                gate_type,
                4,
                P_4,
                [(1, 1, 1, 1), (1, 1, 0, 0), (0, 0, 1, 1), (0, 0, 0, 0)],
                "random",
                desc="k=4 | 4-way | KS=(1111,1100,0011,0000) | random",
                metaseed=15,
            ),
            # V4: XNOR-split
            _make_variant(
                gate_type,
                5,
                P_XNOR,
                [(1, 0, 1, 0, 1), (0, 1, 0, 1, 0)],
                "fixed_1",
                desc="k=5 | split-by-XNOR | KS=(10101,01010) | fixed_1",
            ),
            # V5: A-split wide
            _make_variant(
                gate_type,
                5,
                P_A,
                [(1, 1, 0, 0, 0), (0, 0, 1, 1, 1)],
                "complement",
                desc="k=5 | split-by-A | KS=(11000,00111) | complement",
            ),
            # V6: XOR-split ultra wide
            _make_variant(
                gate_type,
                6,
                P_XOR,
                [(1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)],
                "random",
                desc="k=6 | split-by-XOR | KS=(111000,000111) | random",
                metaseed=16,
            ),
            # V7: 4-way ultra wide
            _make_variant(
                gate_type,
                6,
                P_4,
                [
                    (1, 0, 0, 0, 0, 1),
                    
                    (0, 1, 0, 0, 1, 0),
                    (0, 0, 1, 1, 0, 0),
                    (1, 1, 1, 1, 1, 1),
                ],
                "dontcare_min",
                desc="k=6 | 4-way | KS=(100001,010010,001100,111111) | dontcare_min",
            ),
        ]
    return vs


# ----------------------------------------------------------------------
# 8 variants for the 1-input NOT gate
# ----------------------------------------------------------------------


def _generate_NOT_variants() -> List[LockedVariant]:
    """Generates the database entries for unary NOT gates."""
    P = [[(0,)], [(1,)]]  # split by A
    vs = [
        _make_variant(
            "NOT",
            2,
            P,
            [(0, 0), (1, 1)],
            "dontcare_min",
            desc="k=2 | split-by-A | KS=(00,11) | paper-style",
        ),
        _make_variant(
            "NOT",
            2,
            P,
            [(0, 1), (1, 0)],
            "complement",
            desc="k=2 | split-by-A | KS=(01,10)",
        ),
        _make_variant(
            "NOT",
            2,
            P,
            [(1, 0), (0, 1)],
            "complement",
            desc="k=2 | split-by-A | KS=(10,01)",
        ),
        _make_variant(
            "NOT",
            3,
            P,
            [(0, 0, 0), (1, 1, 1)],
            "complement",
            desc="k=3 | split-by-A | KS=(000,111)",
        ),
        _make_variant(
            "NOT",
            3,
            P,
            [(0, 1, 0), (1, 0, 1)],
            "complement",
            desc="k=3 | split-by-A | KS=(010,101)",
        ),
        _make_variant(
            "NOT",
            3,
            P,
            [(0, 0, 1), (1, 1, 0)],
            "complement",
            desc="k=3 | split-by-A | KS=(001,110)",
        ),
        _make_variant(
            "NOT",
            4,
            P,
            [(0, 0, 0, 0), (1, 1, 1, 1)],
            "complement",
            desc="k=4 | split-by-A | KS=(0000,1111)",
        ),
        _make_variant(
            "NOT",
            4,
            P,
            [(0, 1, 0, 1), (1, 0, 1, 0)],
            "complement",
            desc="k=4 | split-by-A | KS=(0101,1010)",
        ),
    ]
    return vs


# ----------------------------------------------------------------------
# Public catalogue
# ----------------------------------------------------------------------

VARIANTS: Dict[str, List[LockedVariant]] = {
    "AND": _generate_2input_variants("AND"),
    "OR": _generate_2input_variants("OR"),
    "XOR": _generate_2input_variants("XOR"),
    "XNOR": _generate_2input_variants("XNOR"),
    # NOT gate's locked Variants are availble upon request
    # But locking NOT is generally avoided because of disproportionate hardware
    # overhead, lower security return on investment and severe routing congestion
    # "NOT":  _generate_NOT_variants(),
    "NAND": _generate_2input_variants("NAND"),
    "NOR": _generate_2input_variants("NOR"),
}


# ----------------------------------------------------------------------
# Pretty-printer for the AST (handy when debugging the database)
# ----------------------------------------------------------------------

def ast_to_str(ast: Any) -> str:
    """Recursively formats an AST tuple into a readable string equation.

    Supports strictly 2-input binary tuples as well as legacy n-ary formats.

    Args:
        ast: The AST node to format.

    Returns:
        A human-readable algebraic expression string.
    """
    if isinstance(ast, str):
        return ast

    tag = ast[0]
    if tag == "const":
        return str(ast[1])
    if tag == "not":
        return f"!{ast_to_str(ast[1])}"

    if tag == "and":
        if len(ast) == 3:
            return f"({ast_to_str(ast[1])} & {ast_to_str(ast[2])})"
        return "(" + " & ".join(ast_to_str(t) for t in ast[1:]) + ")"

    if tag == "or":
        if len(ast) == 3:
            return f"({ast_to_str(ast[1])} | {ast_to_str(ast[2])})"
        return "(" + " | ".join(ast_to_str(t) for t in ast[1:]) + ")"

    raise ValueError(f"Unknown AST node: {ast!r}")


def _count_ops(ast) -> int:
    """
    Computes a rough heuristic count of operations in an AST.

    Useful for optimization policies (e.g., 'lightest' or 'heaviest')
    to gauge relative implementation footprints.
    """
    if isinstance(ast, str):
        return 1
    if ast[0] == "const":
        return 1
    if ast[0] == "not":
        return 1 + _count_ops(ast[1])
    return sum(_count_ops(t) for t in ast[1:]) + (len(ast) - 2)


# ----------------------------------------------------------------------
# Self-test  ($ python Database.py)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    total = passes = 0
    print("=" * 78)
    print("GateLock / IDKLL variant database — self-test")
    print("=" * 78)
    for gate_type, vlist in VARIANTS.items():
        print(f"\n[{gate_type}]   {len(vlist)} variants")
        for idx, v in enumerate(vlist):
            corr, idkll, suff = verify_variant(v)
            total += 1
            ok = corr and idkll
            passes += ok
            mark = "OK " if ok else "FAIL"
            cost = _count_ops(v.expression)
            print(
                f"  V{idx}  [{mark}]  k={v.key_width}  cost={cost:<3} "
                f"fill={v.fill_strategy:<13} {v.partition_desc}"
            )
            if not corr:
                print("        ! correctness check failed")
            if not idkll:
                print(f"        ! single sufficient keys exist: {suff}")
    print("\n" + "-" * 78)
    print(f"{passes}/{total} variants pass both IDKLL invariants.")
