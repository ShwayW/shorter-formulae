#!/usr/bin/env python3
"""
gt_eval.py -- evaluate a ground-truth formula numerically.

Both OOD generators need the same thing: a callable mapping an (N, D) array of
inputs to N ground-truth outputs, with NaN wherever the formula is undefined.

Feynman used to route its formulas through the repo's prefix grammar and the C++
evaluator instead.  That evaluator hardcodes pi = 3.1415 (cpp/extEvalPN.cpp:180),
which is right for the model's vocabulary and wrong for ground truth: it put a
2.9e-5 relative error into all 31 pi-containing Feynman formulas, reaching 1.6% at
gap 128 for feynman_III_8_54, where the error sits inside a sin whose argument
grows as the OOD box moves out.  Both benchmarks now evaluate through sympy.

The prefix path is untouched everywhere else.  load_feynman_pn_targets() still
serves eval_e2e.py, eval_e2e_tpsr.py, eval_phye2e.py and the symbolic-accuracy
scoring, where matching the model's own grammar is exactly the point, and the
C++ pi must stay at 3.1415 because the model was trained against it.
"""

import numpy as np


def build_evaluator(expression, variable_names, verbose=True):
    """Compile `expression` into evaluate(X) -> y, with NaN where undefined.

    X columns must be ordered to match `variable_names`.  Returns None if the
    expression cannot be parsed, so callers can skip the problem.
    """
    try:
        import sympy as sp
        symbols = [sp.Symbol(name) for name in variable_names]
        expr = sp.sympify(expression, locals=dict(zip(variable_names, symbols)))
        compiled = sp.lambdify(symbols, expr, modules="numpy")
    except Exception as exc:
        if verbose:
            print(f"    [warn] lambdify failed ({exc.__class__.__name__}: {exc})")
        return None

    n_vars = len(variable_names)

    def evaluate(X):
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                y = np.asarray(compiled(*[X[:, i] for i in range(n_vars)]))
        except Exception:
            return np.full(len(X), np.nan)

        # A formula that ignores its inputs lambdifies to a scalar.
        if y.ndim == 0:
            y = np.full(len(X), y.item())

        # Real inputs normally give real outputs -- numpy returns NaN for
        # sqrt(negative) rather than going complex.  Guard anyway, and keep only
        # the rows that came back real rather than discarding the whole batch.
        if np.iscomplexobj(y):
            y = np.where(np.abs(y.imag) < 1e-12, y.real, np.nan)

        return y.astype(np.float64, copy=False)

    return evaluate
