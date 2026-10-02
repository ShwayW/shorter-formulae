"""feynman_pn_analysis.py

Converts every Feynman formula in datasets/feynman/FeynmanEquations.csv to the
grammar's canonical Polish-notation (PN) form, then computes:

  1. nb_bin  -- number of binary operator tokens in the canonical PN.
  2. nb_un   -- number of unary  operator tokens in the canonical PN.
  3. pn_len  -- total token count.
  4. cap_bin -- maximum nb_bin the online generator can produce for this formula's
               input_dim:  randint(input_dim-1, input_dim+5)  ->  max = input_dim+4.
  5. in_dist -- whether both nb_bin <= cap_bin AND nb_un <= reasonable_un_cap.
  6. p_param -- probability of sampling the exact (input_dim, nb_bin) pair from the
               generator's marginal distribution (ignoring the formula-conditional
               probability, which is intractable analytically).

Usage:
    python feynman_pn_analysis.py
"""
import ast
import re
import sys
import csv
sys.path.insert(0, ".")
from simplifyFormula import simplify
from grammar import B, U

B_set = set(B)
U_set = set(U)

# -- int -> PN token list --------------------------------------------------------
def int_to_pn(n):
    if n < 0:
        return ["neg", *int_to_pn(-n)]
    if n == 0:  return ["0"]
    if n == 1:  return ["1"]
    if n == 2:  return ["2"]
    if n == 3:  return ["3"]
    if n == 4:  return ["sqr", "2"]
    if n == 5:  return ["++", "sqr", "2"]
    if n == 6:  return ["*", "2", "3"]
    if n == 7:  return ["++", "*", "2", "3"]
    if n == 8:  return ["pow3", "2"]
    if n == 9:  return ["sqr", "3"]
    if n == 16: return ["sqr", "sqr", "2"]
    return ["++", *int_to_pn(n - 1)]


# -- Python AST -> PN token list -------------------------------------------------
_FN_MAP = {
    "exp":    "exp",   "sqrt":   "sqrt",  "sin":    "sin",   "cos":    "cos",
    "tan":    "tan",   "arcsin": "arcsin","arccos": "arccos","arctan": "arctan",
    "tanh":   "tanh",  "log":    "ln",    "ln":     "ln",    "abs":    "abs",
}


def _ast_to_pn(node, var_map):
    if isinstance(node, ast.BinOp):
        left  = _ast_to_pn(node.left,  var_map)
        right = _ast_to_pn(node.right, var_map)
        op = type(node.op)
        if op is ast.Add:  return ["+",  *left, *right]
        if op is ast.Sub:  return ["-",  *left, *right]
        if op is ast.Mult: return ["*",  *left, *right]
        if op is ast.Div:  return ["/",  *left, *right]
        if op is ast.Pow:
            if isinstance(node.right, ast.Constant):
                e = node.right.value
                if e == 0.5:  return ["sqrt",   *left]
                if e == -0.5: return ["invert", "sqrt", *left]
                if e == -1:   return ["invert", *left]
                if e == -2:   return ["invert", "sqr", *left]
                if e == 2:    return ["sqr",    *left]
                if e == 3:    return ["pow3",   *left]
                if e == 4:    return ["sqr",    "sqr", *left]
            return ["pow", *left, *right]
        return ["ERR_OP", *left, *right]

    if isinstance(node, ast.UnaryOp):
        operand = _ast_to_pn(node.operand, var_map)
        if   isinstance(node.op, ast.USub): return ["neg", *operand]
        elif isinstance(node.op, ast.UAdd): return operand
        return ["ERR_UNARY", *operand]

    if isinstance(node, ast.Call):
        fn = node.func.id if isinstance(node.func, ast.Name) else ""
        if fn in _FN_MAP:
            return [_FN_MAP[fn], *_ast_to_pn(node.args[0], var_map)]
        return [f"ERR_FN_{fn}"]

    if isinstance(node, ast.Name):
        return [var_map.get(node.id, f"ERR_VAR_{node.id}")]

    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, (int, float)) and float(v) == int(float(v)):
            return int_to_pn(int(v))
        from fractions import Fraction
        f = Fraction(v).limit_denominator(100)
        if abs(float(f) - v) < 1e-9:
            n, d = f.numerator, f.denominator
            sign_toks = ["neg"] if n < 0 else []
            return [*sign_toks, "/", *int_to_pn(abs(n)), *int_to_pn(d)]
        return [repr(v)]

    return [f"ERR_{type(node).__name__}"]


def formula_to_pn(fml_str, var_names):
    """Convert a Python-syntax formula to canonical grammar PN using simplify()."""
    # Map variable names -> safe Python identifiers -> grammar var names
    safe_ids = {}
    var_map  = {}
    for i, vn in enumerate(var_names):
        sid = f"FVAR{i+1}F"
        safe_ids[vn] = sid
        var_map[sid]  = f"v{i+1}"
    var_map["piSYM"] = "pi"

    fml = fml_str

    # Sort by length (longest first) to avoid partial substitution
    for vn in sorted(safe_ids, key=len, reverse=True):
        fml = re.sub(rf"\b{re.escape(vn)}\b", safe_ids[vn], fml)

    # Protect symbolic pi
    fml = re.sub(r"\bpi\b", "piSYM", fml)

    try:
        tree = ast.parse(fml, mode = "eval")
    except SyntaxError as e:
        return None, f"PARSE_ERR: {e}"

    toks    = _ast_to_pn(tree.body, var_map)
    pn_raw  = " ".join(toks)

    if any(t.startswith("ERR") for t in toks):
        return pn_raw, "HAS_ERRORS"

    try:
        pn_canon = simplify(pn_raw)
    except Exception:
        pn_canon = pn_raw  # fallback; still report

    return pn_canon, "ok"


# -- Operator counts ------------------------------------------------------------
def count_ops(pn_str):
    toks   = pn_str.split()
    nb_bin = sum(1 for t in toks if t in B_set)
    nb_un  = sum(1 for t in toks if t in U_set)
    return nb_bin, nb_un, len(toks)


# -- Generation probability computation ---------------------------------------
#
# The generator makes these independent random choices:
#
#   1. input_dim  ~ Uniform(1, 10)
#   2. nb_bin     ~ Uniform(max(0,d-1), d+12)    [after fix; was d+5]
#   3. nb_un      ~ Uniform(0, 8)                [after fix; was 0..4]
#   4. binary tree STRUCTURE  (Catalan-uniform)  -> P = 1/C(nb_bin)
#   5. each binary op         (uniform over |B|=5) -> (1/5) each
#   6. nb_un unary insertions: each independently picks a child slot (uniform
#      over 2*nb_bin slots in the binary tree) and an operator (uniform over
#      |U|=18).  Insertions use replace=True, so nested unaries arise when the
#      same slot is chosen multiple times; the nesting ORDER matters (first
#      insertion wraps the original child, last gives the outermost wrapper).
#   7. leaves (nb_bin+1 leaf slots in the pre-unary binary tree):
#        - first input_dim leaves (after random shuffle): coverage vars v1..vd
#          -> P(specific permutation) = 1 / d!   (shuffle is uniform)
#        - remaining leaves: each independently is
#            a repeated variable  with prob (1-PROB_CONST) / d  (each of d vars)
#            a constant from C    with prob PROB_CONST / |C|  (ignoring composite)
#
# P(F) = P(params) * P(tree) * P(ops) * P(unary_slots) * P(leaves)
#
# This is a LOWER BOUND: many pre-simplification trees collapse to the same
# canonical form F, so the true probability is >= this value.  The dominant
# contribution for most formulas is this single "natural" tree path.

from math import comb, factorial

VAR_DIM     = 10    # len(V)
_PROB_CONST = 0.2   # from grammar.py
_N_BIN_OPS  = 5     # |B|
_N_UN_OPS   = 18    # |U|
_N_C        = 5     # |C| = {0,1,2,3,pi}

_VAR_SET   = {f"v{i}" for i in range(1, VAR_DIM + 1)}
_CONST_SET = {"0", "1", "2", "3", "pi"}


def _catalan(n):
    """C(2n,n)/(n+1)."""
    if n < 0: return 0
    if n == 0: return 1
    return comb(2 * n, n) // (n + 1)


def generation_probability(pn_str, nb_bin, nb_un, input_dim):
    """
    Estimate P(generator produces formula with PN pn_str).

    Returns (p_total, breakdown_dict) where breakdown_dict has the individual
    factors.  p_total = 0.0 if any required parameter is out of the generator's
    post-fix range.
    """
    # -- 1-3. Parameter probabilities (post-fix ranges) ------------------------
    lo_bin = max(0, input_dim - 1)
    hi_bin = input_dim + 12          # exclusive (post-fix upper bound)
    lo_un, hi_un = 0, 8              # post-fix: randint(0,9) -> [0,8]

    in_range = (
        1 <= input_dim <= VAR_DIM
        and lo_bin <= nb_bin < hi_bin
        and lo_un  <= nb_un  <= hi_un
    )
    if not in_range:
        zero = {"p_params": 0, "p_tree": 0, "p_ops": 0,
                "p_unary_slots": 0, "p_leaves": 0}
        return 0.0, zero

    p_dim  = 1.0 / VAR_DIM
    p_bin  = 1.0 / (hi_bin - lo_bin)   # 1/12 for d>=1 (post-fix)
    p_un   = 1.0 / (hi_un - lo_un + 1) # 1/9
    p_params = p_dim * p_bin * p_un

    # -- 4. Binary tree structure -----------------------------------------------
    cat = _catalan(nb_bin)
    p_tree = 1.0 / cat if cat > 0 else 0.0

    # -- 5. Binary operator choices --------------------------------------------
    p_bin_ops = (1.0 / _N_BIN_OPS) ** nb_bin

    # -- 6. Unary insertions ---------------------------------------------------
    # The binary tree has 2*nb_bin child slots; each of the nb_un insertions
    # independently picks one slot (uniform) and one op (uniform).
    # We want all nb_un unaries at their correct slots with correct ops.
    n_slots = max(1, 2 * nb_bin)     # child slots in the binary tree
    p_unary_slots = (1.0 / (n_slots * _N_UN_OPS)) ** nb_un

    # -- 7. Leaf assignments ----------------------------------------------------
    toks = pn_str.split()
    var_leaves   = [t for t in toks if t in _VAR_SET]
    const_leaves = [t for t in toks if t in _CONST_SET]
    n_var_occ   = len(var_leaves)
    n_const_occ = len(const_leaves)

    # Coverage guarantee: first input_dim leaves (after random shuffle) are v1..vd.
    # Probability of the specific permutation: 1/d!
    # But if the formula uses fewer distinct variables than input_dim (shouldn't
    # happen after relabel_variables), guard below.
    n_distinct = len(set(var_leaves))
    if n_distinct < input_dim:
        # Formula uses fewer vars than input_dim -- wrong input_dim guess; 0.
        return 0.0, {"p_params": p_params, "p_tree": p_tree,
                     "p_ops": p_bin_ops * p_unary_slots,
                     "p_unary_slots": p_unary_slots, "p_leaves": 0.0}

    p_coverage = 1.0 / factorial(input_dim)

    # Extra variable occurrences beyond the coverage slot (each is a repeated var).
    n_extra_var = max(0, n_var_occ - input_dim)
    p_extra_var = ((1.0 - _PROB_CONST) / input_dim) ** n_extra_var

    # Each constant leaf (ignoring composite expansion; approximation).
    p_const_leaves = (_PROB_CONST / _N_C) ** n_const_occ

    p_leaves = p_coverage * p_extra_var * p_const_leaves

    p_total = p_params * p_tree * p_bin_ops * p_unary_slots * p_leaves

    breakdown = {
        "p_params":       p_params,
        "p_tree":         p_tree,
        "p_bin_ops":      p_bin_ops,
        "p_unary_slots":  p_unary_slots,
        "p_leaves":       p_leaves,
    }
    return p_total, breakdown


# -- Parse CSV -----------------------------------------------------------------
def parse_feynman_csv(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Filename", "").strip()
            if not name:
                continue
            fml  = row.get("Formula", "").strip()
            nvars_str = row.get("# variables", "0").strip()
            nvars = int(nvars_str) if nvars_str.isdigit() else 0
            var_names = []
            for k in range(1, 11):
                vname = row.get(f"v{k}_name", "").strip()
                if vname:
                    var_names.append(vname)
            rows.append((name, nvars, var_names, fml))
    return rows


# -- Main ----------------------------------------------------------------------
def main():
    path  = "datasets/feynman/FeynmanEquations.csv"
    rows  = parse_feynman_csv(path)

    # Post-fix caps (after the two-line change to data_process.py):
    #   nb_bin: randint(max(0,d-1), d+12)  ->  max = d+11
    #   nb_un:  randint(0, 9)              ->  max = 8
    BIN_EXTRA  = 11    # input_dim + BIN_EXTRA is the new max binary ops
    UN_CAP_NEW = 8     # new max unary ops

    print(f"{'Name':<12} {'nv':>3} {'nBin':>5} {'nUn':>5} {'len':>5} {'capBin':>7} "
          f"{'status':>7} {'p_gen':>12}  PN")
    print("-" * 140)

    n_ok = n_over_bin = n_over_un = n_both = n_err = 0

    results = []
    for name, nvars, var_names, fml in rows:
        if not fml or not var_names:
            continue

        pn, status = formula_to_pn(fml, var_names)
        if pn is None or "ERR" in (pn or ""):
            n_err += 1
            print(f"{name:<12} {nvars:>3}  CONVERSION ERROR: {status}")
            results.append((name, nvars, None, None, None, None, None, None, None, status))
            continue

        nb_bin, nb_un, pn_len = count_ops(pn)
        # Use actual distinct variable count from the PN, not CSV metadata
        # (some CSV rows declare more variables than the formula actually uses)
        actual_dim = len({t for t in pn.split() if t in _VAR_SET})
        cap_bin_new = actual_dim + BIN_EXTRA   # new max binary ops
        lo_bin_new  = max(0, actual_dim - 1)   # generator lower bound
        over_bin = nb_bin > cap_bin_new or nb_bin < lo_bin_new
        over_un  = nb_un  > UN_CAP_NEW
        in_dist  = (not over_bin) and (not over_un)

        p_gen, breakdown = generation_probability(pn, nb_bin, nb_un, actual_dim)
        status_str = "OK" if in_dist else (
            "BIN+UN" if (over_bin and over_un) else
            "BIN"    if over_bin               else "UN"
        )

        if   in_dist:                n_ok       += 1
        elif over_bin and over_un:   n_both     += 1
        elif over_bin:               n_over_bin += 1
        else:                        n_over_un  += 1

        results.append((name, actual_dim, nb_bin, nb_un, pn_len, cap_bin_new,
                        in_dist, p_gen, breakdown, status_str, pn))
        print(f"{name:<12} {actual_dim:>3} {nb_bin:>5} {nb_un:>5} {pn_len:>5} {cap_bin_new:>7} "
              f"{status_str:>7} {p_gen:>12.3e}  {pn[:70]}")

    print()
    n_total = n_ok + n_over_bin + n_over_un + n_both + n_err
    print(f"Total formulas processed: {n_total}")
    print(f"  In distribution (OK, new caps):   {n_ok:3d}  ({100*n_ok/n_total:.0f}%)")
    print(f"  Over binary op cap only:           {n_over_bin:3d}")
    print(f"  Over unary op cap only (>{UN_CAP_NEW}):    {n_over_un:3d}")
    print(f"  Over both caps:                    {n_both:3d}")
    print(f"  Conversion errors:                 {n_err:3d}")
    print()

    # Sort by p_gen descending
    ranked = [(r[0], r[2], r[3], r[4], r[7], r[9]) for r in results if r[2] is not None]
    ranked.sort(key=lambda x: -(x[4] or 0))
    print("Top 10 highest generation probability (p_gen = P(params)*P(tree)*P(ops)*P(leaves)):")
    for name, nb, nu, pn_len, pg, st in ranked[:10]:
        print(f"  {name:<12}  nb_bin={nb:2d}  nb_un={nu:2d}  len={pn_len:2d}  p_gen={pg:.3e}  {st}")

    print()
    print("Probability breakdown for a sample formula (I.12.1: F=mu*Nn):")
    for r in results:
        if r[0] == "I.12.1" and r[2] is not None:
            name, nvars, nb_bin, nb_un, pn_len, _, _, p_gen, bkd, st, pn = r
            print(f"  PN: {pn}")
            for k, v in bkd.items():
                print(f"  {k:<18} = {v:.4e}")
            print(f"  p_gen (product)  = {p_gen:.4e}")
            break

    print()
    print("Formulas outside generator distribution (post-fix):")
    for r in results:
        name, nvars, nb_bin, nb_un, pn_len, cap_bin, in_dist, p_gen, bkd, status_str, pn = r
        if nb_bin is None:
            continue
        if status_str != "OK":
            print(f"  {name:<12}  nv={nvars:2d}  nb_bin={nb_bin:2d} (cap={cap_bin:2d})  "
                  f"nb_un={nb_un:2d}  p_gen={p_gen:.3e}  status={status_str}")
            if pn:
                print(f"    PN: {pn}")


if __name__ == "__main__":
    main()
