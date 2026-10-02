#!/usr/bin/env python3
"""
llmsr_complexity.py -- symbolic-skeleton complexity for LLMSR `def equation(...)`
programs, made comparable to compare_srbench.formula_complexity (sympy node count).

An LLMSR discovered "formula" is a full numpy program padded with numerical-stability
guards (np.maximum/np.where/np.finfo, default-value fallbacks, intermediate vars).
Counting its AST would conflate defensive code with the mathematics. Instead we
symbolically EXECUTE the program with sympy symbols for the inputs and params, mapping
numpy ops to sympy and the guards to pass-throughs (np.maximum(a,b)->a, np.abs->identity,
np.where(c,x,y)->y, np.finfo(...).eps->0), then count nodes of the returned sympy
expression with the same preorder_traversal counter formula_complexity uses.

Returns NaN when the program can't be parsed/executed symbolically (excluded upstream).
"""
import ast
import math

import sympy as sp
from sympy import preorder_traversal


# The LLMSR searcher optimizes a fixed-length params array (llm_methods/llmsr/searcher.py:
# MAX_NPARAMS = 10), so a solved program only ever touches params[0..9].  Modelling the
# TRUE length keeps `*rest = params`, `for p in params`, and `range(len(params))` bounded;
# an unbounded stand-in (formerly len 1e6, indices never out of range) makes the starred-
# unpack / range forms build ~1e6 symbols and OOM the process.
MAX_NPARAMS = 10


class _Params:
    """Stand-in for the optimized `params` array: params[i] -> a distinct constant symbol
    c_i for 0 <= i < MAX_NPARAMS; out-of-range (and negative-from-end past the start)
    indices raise IndexError exactly like the real length-10 numpy array, so iteration and
    starred unpacking terminate.  Never None."""
    def __init__(self):
        self._syms = {}

    def __getitem__(self, i):
        i = int(i)                      # slices raise TypeError here -> NaN upstream (as before)
        if i < 0:
            i += MAX_NPARAMS            # numpy-style negative indexing
        if not 0 <= i < MAX_NPARAMS:
            raise IndexError(i)
        return self._syms.setdefault(i, sp.Symbol(f"c{i}"))

    def __iter__(self):
        return (self[i] for i in range(MAX_NPARAMS))

    def __len__(self):
        return MAX_NPARAMS


def _passthrough_max(a, *_):   # np.maximum/np.minimum/np.clip: strip the guard, keep the value
    return a


def _where(cond, x, y):        # np.where(cond, x, y): keep the normal (else) branch
    return y


class _Finfo:
    eps = sp.Integer(0)
    tiny = sp.Integer(0)
    max = sp.oo


def _identity(x, *_, **__):
    return x


class _FakeNP:
    """math ops -> sympy, guards -> pass-through; any UNKNOWN attribute (ndarray,
    dtypes, novel guard fns) falls back to a pass-through so def-time annotations and
    defensive calls don't crash the symbolic execution."""
    def __init__(self, known):
        object.__setattr__(self, "_known", known)

    def __getattr__(self, name):
        known = object.__getattribute__(self, "_known")
        if name in known:
            return known[name]
        return _identity   # unknown fn/type -> keep the value (also valid as an annotation)


def _make_np():
    def power(a, b):
        return a ** b

    known = dict(
        # elementary functions
        sin=sp.sin, cos=sp.cos, tan=sp.tan, arcsin=sp.asin, arccos=sp.acos,
        arctan=sp.atan, arctan2=lambda y, x: sp.atan2(y, x), sinh=sp.sinh, cosh=sp.cosh,
        tanh=sp.tanh, exp=sp.exp, log=sp.log, log10=lambda x: sp.log(x, 10),
        log2=lambda x: sp.log(x, 2), log1p=lambda x: sp.log(1 + x),
        sqrt=sp.sqrt, cbrt=lambda x: sp.Pow(x, sp.Rational(1, 3)),
        square=lambda x: x ** 2, power=power, float_power=power, reciprocal=lambda x: 1 / x,
        sign=sp.sign, abs=_identity, absolute=_identity, fabs=_identity,
        # constants
        pi=sp.pi, e=sp.E, euler_gamma=sp.EulerGamma, inf=sp.oo,
        # guards / reductions -> pass-through
        maximum=_passthrough_max, minimum=_passthrough_max, fmax=_passthrough_max,
        fmin=_passthrough_max, clip=_passthrough_max, where=_where, nan_to_num=_identity,
        real=_identity, sum=_identity, mean=_identity, prod=_identity,
        finfo=lambda *_a, **_k: _Finfo(),
    )
    return _FakeNP(known)


def _input_names(program: str):
    """The equation's input args (everything except `params`)."""
    tree = ast.parse(program)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "equation":
            return [a.arg for a in node.args.args if a.arg != "params"]
    raise ValueError("no equation() def")


def llmsr_complexity(program) -> float:
    """sympy node count of the LLMSR program's symbolic skeleton; NaN on failure."""
    if not isinstance(program, str) or "def equation" not in program:
        return float("nan")
    try:
        names = _input_names(program)
        ns = {"np": _make_np(), "math": math}
        exec(compile(program, "<llmsr>", "exec"), ns)  # noqa: S102 -- trusted local artifacts
        eq = ns.get("equation")
        if eq is None:
            return float("nan")
        inputs = [sp.Symbol(n) for n in names]
        result = eq(*inputs, _Params())
        expr = sp.sympify(result)
        return float(sum(1 for _ in preorder_traversal(expr)))
    except Exception:
        return float("nan")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from results_io import load_results
    import numpy as np
    _m, rows = load_results(
        "results/llmsr/noise_0/llmsrbench/llmsr-gemini35flash_lsr_transform/results.pkl.gz")
    cs = [llmsr_complexity(r.get("discovered_equation")) for r in rows]
    ok = [c for c in cs if c == c]
    print(f"parsed {len(ok)}/{len(cs)} programs "
          f"({100*len(ok)/len(cs):.0f}%); node count: "
          f"median={np.median(ok):.0f} mean={np.mean(ok):.1f} "
          f"min={min(ok):.0f} max={max(ok):.0f}")
