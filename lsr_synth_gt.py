#!/usr/bin/env python3
"""
lsr_synth_gt.py -- recover an evaluable ground truth for every LSR-Synth problem.

The LSR-Synth `expression` strings in the LLM-SRBench parquet files are not directly
parseable, but every one of the 129 problems has a well-defined ground truth:

    bio_pop_growth  24  state variables in function notation, P(t)      -> write as P
    chem_react      36  A(t), plus constants whose NAMES were mangled by a value
                        substitution: "k_z" was written out as "<value of k>_z", so
                        "0.1899..._z" is a separate free constant k_z, not a number
    matsci          25  parse as they are
    phys_osc        44  x(t), v(t), plus named parameters (F0, beta, omega0, alpha,
                        mu, gamma, ...) with no values in the metadata

Free constants (the mangled chem_react names and the phys_osc parameters) are fitted
by least squares to the benchmark's own `train` split.  The fitted formula is then
checked against the benchmark's `id_test` and `ood_test` splits, which it never saw:
a recovered ground truth that reproduces the benchmark's own out-of-domain targets is
the right formula, not merely a good fit.

    python lsr_synth_gt.py            # table of every problem + fit / check R^2
"""
import re

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

DOMAINS = ["bio_pop_growth", "chem_react", "matsci", "phys_osc"]

# Names sympify must NOT turn into symbols (they are the functions the formulas use).
_FUNCS = {"sin", "cos", "tan", "exp", "log", "sqrt", "Abs", "tanh", "sinh", "cosh",
          "asin", "acos", "atan", "pi", "E"}
_MANGLED = re.compile(r"\b\d+\.\d+(?:e[-+]?\d+)?_([A-Za-z]\w*)\b")


def _r2(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    ok = np.isfinite(p)
    if ok.sum() < 2:
        return float("-inf")
    y, p = y[ok], p[ok]
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def parse(expression, input_symbols):
    """sympy expression with inputs as plain symbols and free constants as symbols."""
    e = expression
    for v in input_symbols:                       # P(t) -> P, x(t) -> x, ...
        e = re.sub(r"\b%s\(t\)" % re.escape(v), v, e)
    e = _MANGLED.sub(r"k_\1", e)                  # "0.18997..._z" -> k_z
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", e)) - _FUNCS
    return sp.sympify(e, locals={n: sp.Symbol(n) for n in names})


def recover(expression, input_symbols, train, n_starts=16, seed=0):
    """(expr, info): the ground truth with every free constant replaced by its fitted
    value, and a dict with the fitted constants and the train R^2.

    `train` follows the LLM-SRBench layout: column 0 = y, columns 1: = inputs in the
    order of `input_symbols`.
    """
    expr = parse(expression, input_symbols)
    ins = [sp.Symbol(s) for s in input_symbols]
    free = sorted((s for s in expr.free_symbols if s not in ins), key=str)
    X, y = train[:, 1:].astype(np.float64), train[:, 0].astype(np.float64)
    f = sp.lambdify(free + ins, expr, "numpy")
    if not free:
        with np.errstate(all="ignore"):
            return expr, {"constants": {}, "train_r2": _r2(y, f(*X.T))}

    def resid(c):
        with np.errstate(all="ignore"):
            r = np.asarray(f(*c, *X.T), np.float64) - y
        return np.where(np.isfinite(r), r, 1e6)

    rng = np.random.default_rng(seed)
    starts = [np.full(len(free), v) for v in (0.1, 0.5, 1.0, 2.0)]
    starts += [rng.uniform(-2, 2, len(free)) for _ in range(n_starts - len(starts))]
    best = min((least_squares(resid, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15,
                              max_nfev=20000) for x0 in starts), key=lambda r: r.cost)
    consts = {str(s): float(v) for s, v in zip(free, best.x)}
    fitted = expr.subs({s: sp.Float(v, 17) for s, v in zip(free, best.x)})
    with np.errstate(all="ignore"):
        tr = _r2(y, f(*best.x, *X.T))
    return fitted, {"constants": consts, "train_r2": tr}


def recover_problem(p):
    """recover() for one eval_mymodels.load_problems() entry, plus the check R^2 on
    the benchmark's id_test / ood_test splits.  Returns (expr_str, input_symbols, info)."""
    input_symbols = list(p["symbols"][1:])
    expr, info = recover(p["expression"], input_symbols, p["train"])
    g = sp.lambdify([sp.Symbol(s) for s in input_symbols], expr, "numpy")
    for split in ("test", "id_test", "ood_test"):
        arr = p.get(split)
        if arr is None or len(arr) == 0:
            continue
        with np.errstate(all="ignore"):
            info[f"{split}_r2"] = _r2(arr[:, 0], g(*arr[:, 1:].T))
    return sp.sstr(expr, full_prec=True), input_symbols, info


def main():
    from eval_mymodels import load_problems
    worst = {}
    for dom in DOMAINS:
        for p in load_problems(f"lsr_synth_{dom}"):
            s, ins, info = recover_problem(p)
            keys = [k for k in info if k.endswith("_r2")]
            print(f"{p['name']:>9}  " + "  ".join(f"{k}={info[k]:.8f}" for k in keys)
                  + (f"  consts={ {k: round(v, 6) for k, v in info['constants'].items()} }"
                     if info["constants"] else ""))
            for k in keys:
                worst[k] = min(worst.get(k, 1.0), info[k])
    print("\nworst over all 129 problems:", {k: round(v, 10) for k, v in worst.items()})


if __name__ == "__main__":
    main()
