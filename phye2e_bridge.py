#!/usr/bin/env python3
"""phye2e_bridge.py -- translate OUR Polish-notation formulae into PhyE2E's expression
trees, so their divide-and-conquer Oracle can be driven with our transformer as the
backbone instead of theirs.

WHY THIS EXISTS
---------------
Their Oracle (PhysicsRegression/Oracle/oracle.py) is two-phase with plain arrays in the
middle, which is what makes the swap possible at all:

    oracle_fit(x, y, ...)  ->  res_x, res_y, res_hints        # surrogate + separability
            <some symbolic model solves those subproblems>    # <- the seam
    oracle.reverse(original_gens, oracle_gens)  ->  recombined expression

`oracle_fit` never touches a symbolic model: it trains the surrogate net, reads the
separability structure off its derivatives, and emits pseudo-data as numpy.  So any
solver can sit in the middle.  What it will NOT accept is a bare string: `reverse` calls
`safely_refine` -> `generator.function_to_skeleton` and `translate_expr_str` -> `str(node)`
on real `Node` objects.  Hence this module: our PN string -> their `Node` tree.

WHAT HAS TO BE TRANSLATED
-------------------------
1. OPERATOR NAMES.  Ours are symbols and short words, theirs are words with fixed arity
   (symbolicregression/envs/operators.py).  Mostly 1:1, except the four below.
2. TOKENS WITH NO COUNTERPART.  `++`/`--` (increment/decrement) and `sqr` do not exist in
   their vocabulary and are EXPANDED: `++ x` -> add(x, 1), `-- x` -> sub(x, 1),
   `sqr x` -> pow2(x).  This is the only place the translation is not structure-preserving,
   and it is exact rather than approximate.
3. CONSTANTS.  Our grammar is mantissa-free: the only literals are {0,1,2,3,pi} and every
   other value is BUILT (4 is spelled `sqr 2`).  Their `_decode` accepts any token that
   parses as a float as a numeric leaf, so integers pass through as plain `Node("2")`
   and `pi` goes through as the symbolic constant they already carry in math_constants.
   Nothing needs their sign/mantissa/exponent float encoding, which is why this builds
   `Node`s directly instead of going through `equation_encoder.decode`.
4. PI IS NOT THE SAME NUMBER ON BOTH SIDES.  Our C evaluator (extEvalPN) uses
   pi = 3.1415 -- five significant digits -- while their Node uses full math.pi.  A
   translated formula containing pi therefore evaluates ~3e-5 (relatively) differently
   on their side.  This module keeps the SYMBOL `pi` rather than baking in our truncated
   literal: their value is the correct one, their own `translate_expr_str` substitutes
   3.14159265 anyway, and BFGS refits constants downstream.  Verified numerically: 16/18
   translated formulae agree to 1e-9 or exactly, and the only two that do not are the
   two containing pi, off by exactly the 3.1415-vs-math.pi gap.
5. VARIABLE INDEXING.  Ours are 1-based (`v1`), theirs 0-based (`x_0`).  Subformulae are
   GROUP-LOCAL: for a group of k variables our solver returns v1..vk, which map to
   x_0..x_{k-1}.  Their `translate_expr_str` then remaps those to the parent's indices
   using the group it recorded, so this module must NOT do that remapping itself.

Usage:
    from phye2e_bridge import pn_to_node
    node = pn_to_node("* v1 sqr v2", params)     # params = their model's params object
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _their_node_cls():
    """Import their Node lazily -- PhysicsRegression/ has to be on sys.path first, and
    eval_phye2e.py's _setup_imports() does that only when their model is being used."""
    pkg = os.path.join(SCRIPT_DIR, "PhysicsRegression")
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    from symbolicregression.envs.node import Node
    return Node


# ---- our token -> their token -------------------------------------------------------
# Straight renames: same arity, same semantics.
_DIRECT = {
    "+": "add", "-": "sub", "*": "mul", "/": "div", "pow": "pow",
    "neg": "neg", "invert": "inv", "sqrt": "sqrt", "exp": "exp", "ln": "log",
    "sin": "sin", "cos": "cos", "tan": "tan", "abs": "abs", "tanh": "tanh",
    "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
    "pow2": "pow2", "pow3": "pow3",
}
# Ours only; rewritten during translation (see EXPANSIONS in the docstring).
_EXPAND = {"sqr", "++", "--"}

_ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2, "pow": 2}   # everything else is unary


class BridgeError(ValueError):
    """Raised when a PN string cannot be represented in their grammar."""


def pn_to_node(pn: str, params, var_prefix: str = "x_"):
    """Our PN string -> their Node tree.  Raises BridgeError on anything untranslatable.

    `params` is their params namespace (Node stores it and reads it when printing).
    """
    Node = _their_node_cls()
    toks = pn.split()
    if not toks:
        raise BridgeError("empty formula")

    def build(i):
        """(node, next_index) for the subtree starting at toks[i]."""
        if i >= len(toks):
            raise BridgeError(f"ran off the end of {pn!r}")
        t = toks[i]

        # -- expansions: tokens of ours with no counterpart in their vocabulary --
        if t == "sqr":
            child, j = build(i + 1)
            n = Node("pow2", params); n.push_child(child)
            return n, j
        if t in ("++", "--"):
            child, j = build(i + 1)
            one = Node("1", params)
            n = Node("add" if t == "++" else "sub", params)
            n.push_child(child); n.push_child(one)
            return n, j

        # -- direct operator renames --
        if t in _DIRECT:
            their = _DIRECT[t]
            n = Node(their, params)
            j = i + 1
            for _ in range(_ARITY.get(their, 1)):
                child, j = build(j)
                n.push_child(child)
            return n, j

        # -- leaves --
        if t == "pi":
            return Node("pi", params), i + 1
        if t.startswith("v") and t[1:].isdigit():
            return Node(f"{var_prefix}{int(t[1:]) - 1}", params), i + 1   # 1-based -> 0-based
        try:
            float(t)                       # our numeric literals 0..3, and any BFGS-fitted value
        except ValueError:
            raise BridgeError(f"token {t!r} has no PhyE2E counterpart (in {pn!r})")
        return Node(t, params), i + 1

    node, used = build(0)
    if used != len(toks):
        raise BridgeError(f"trailing tokens after a complete parse: {toks[used:]!r}")
    return node


def as_oracle_gen(pn: str, params):
    """Wrap a PN string in the dict shape `Oracle.reverse` expects for one subproblem.

    Their code reads "predicted_tree" and tolerates None (it records the failure and
    carries on), so an untranslatable formula degrades to a missed subproblem rather
    than killing the whole run."""
    try:
        tree = pn_to_node(pn, params)
    except BridgeError:
        tree = None
    return {"predicted_tree": tree, "relabed_predicted_tree": tree, "message": "ours"}


# ---- their token -> our token -------------------------------------------------------
# The return leg.  `Oracle.reverse` hands back their `Node` trees, but every consumer
# downstream of eval_phyedc.py -- eval_mymodels.predict/compute_r2, score_sym_acc.py,
# the augment_*_caches scripts -- speaks our prefix notation.  Returning their infix
# instead would not raise: `predict()` would simply yield NaN for every row and the arm
# would score a silent R^2 = 0.  So the trip has to be a round trip.
_INVERSE = {
    "add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "pow",
    "neg": "neg", "inv": "invert", "sqrt": "sqrt", "exp": "exp", "log": "ln",
    "sin": "sin", "cos": "cos", "tan": "tan", "abs": "abs", "tanh": "tanh",
    "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
    "pow2": "pow2", "pow3": "pow3",
}
# In their unary list but not in our grammar (grammar.U).  pow4/pow5 are exact as a
# binary power -- our evaluator accepts any float literal, so `pow x 4` needs no new
# token and avoids duplicating the child subtree the way `pow2 pow2 x` would.
_AS_POW = {"pow4": "4", "pow5": "5"}
_THEIR_BINARY = {"add", "sub", "mul", "div", "pow"}

# Their `E` leaf is Euler's number; our grammar has no symbol for it, and it is a
# constant, so it goes through as a literal rather than failing the whole formula.
_E = "2.718281828459045"


def node_to_pn(node, var_prefix: str = "x_") -> str:
    """Their Node tree -> our PN string.  Raises BridgeError on anything untranslatable.

    Inverse of `pn_to_node` up to the expansions that leg performs: `sqr`/`++`/`--` come
    back as `pow2` / `+ x 1` / `- x 1`, which are the same functions written the way our
    grammar prefers.  sinh/cosh have no counterpart of ours and raise.
    """
    out = []

    def walk(n):
        v = str(n.value)
        kids = list(getattr(n, "children", []) or [])

        if v in _INVERSE:
            expected = 2 if v in _THEIR_BINARY else 1
            if len(kids) != expected:
                raise BridgeError(f"{v} has {len(kids)} children, expected {expected}")
            out.append(_INVERSE[v])
            for c in kids:
                walk(c)
            return
        if v in _AS_POW:
            if len(kids) != 1:
                raise BridgeError(f"{v} has {len(kids)} children, expected 1")
            out.append("pow")
            walk(kids[0])
            out.append(_AS_POW[v])
            return
        if kids:
            raise BridgeError(f"operator {v!r} has no counterpart in our grammar")

        # -- leaves --
        if v == "pi":
            out.append("pi")
            return
        if v == "E":
            out.append(_E)
            return
        if v.startswith(var_prefix) and v[len(var_prefix):].isdigit():
            out.append(f"v{int(v[len(var_prefix):]) + 1}")        # 0-based -> 1-based
            return
        try:
            float(v)
        except ValueError:
            raise BridgeError(f"leaf {v!r} has no counterpart in our grammar")
        out.append(v)

    walk(node)
    return " ".join(out)
