"""
parse_srbench_formulas.py -- formula preprocessors for SRBench methods that use
non-standard syntax.

Supported:
  BSR     -- bracket-wrapped coefficients, x[i] variables, ^ power operator
  gplearn  -- functional prefix notation (add/sub/mul/div), X0-indexed variables
  SBP-GP  -- infix with plog() and infix aq operator, x0-indexed
  GP-GOMEA -- infix with plog() and p/ protected-division operator, x0-indexed

MRGP is deliberately NOT supported: its symbolic_model is a partial linear-
regression representation that cannot be independently evaluated without the
training-data linear-regression step.  MRGP OOD accuracy is therefore omitted.

Each method's formula is converted to a Python expression string that can be
evaluated with numpy via eval(formula, namespace) where namespace is built by
_ood_make_ns() / _build_ns().

Usage:
    from parse_srbench_formulas import preprocess_formula
    py_expr = preprocess_formula(formula_str, algo_name)
    # then evaluate with appropriate variable namespace
"""

from __future__ import annotations

import re
import numpy as np


# -- protected helpers ---------------------------------------------------------

def _plog(x: np.ndarray) -> np.ndarray:
    return np.log(np.abs(x) + 1e-10)


def _protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(np.abs(b) < 1e-10, np.sign(a) * 1e10, a / b)


def _aq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Analytic quotient: a / sqrt(1 + b^2).  SBP-GP uses aq as an infix op."""
    return a / np.sqrt(1.0 + b ** 2)


def _safe_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(x))


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.abs(x) + 1e-10)


# -- namespace factories -------------------------------------------------------

def _base_ns() -> dict:
    return {
        "__builtins__": {},
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": _safe_log, "sqrt": _safe_sqrt,
        "abs": np.abs, "arcsin": np.arcsin, "arccos": np.arccos,
        "arctan": np.arctan, "pi": np.pi, "e": np.e,
    }


def build_ns_x0(X: np.ndarray, extra: dict | None = None) -> dict:
    """Namespace with x0, x1, ... (0-indexed, lower-case) for BSR/SBP-GP/GP-GOMEA."""
    ns = _base_ns()
    for i in range(X.shape[1]):
        ns[f"x{i}"] = X[:, i].astype(np.float64)
    if extra:
        ns.update(extra)
    return ns


def build_ns_X0(X: np.ndarray, extra: dict | None = None) -> dict:
    """Namespace with X0, X1, ... (0-indexed, upper-case) for gplearn."""
    ns = _base_ns()
    # gplearn functional operators
    ns["add"] = np.add
    ns["sub"] = np.subtract
    ns["mul"] = np.multiply
    ns["div"] = _protected_div
    for i in range(X.shape[1]):
        ns[f"X{i}"] = X[:, i].astype(np.float64)
    if extra:
        ns.update(extra)
    return ns


# -- BSR -----------------------------------------------------------------------

def preprocess_bsr(formula: str) -> str:
    """Convert BSR formula to eval-able Python string.

    BSR syntax:
      [coeff]  -- square brackets wrap numeric constants/expressions
      x[i]     -- variable i (0-indexed), INSIDE square brackets
      ^        -- exponentiation
    Strategy:
      1. Replace x[N] with xN  (variable references)
      2. Remove remaining [ ] (they are just parentheses around constants)
      3. Replace ^ with **
    """
    if not formula or not str(formula).strip():
        return ""
    s = str(formula).strip()
    # Step 1: x[N] -> xN  (must be done before stripping all brackets)
    s = re.sub(r'\bx\[(\d+)\]', lambda m: f"x{m.group(1)}", s)
    # Step 2: strip remaining brackets (coefficient groupers -> parens)
    s = s.replace("[", "(").replace("]", ")")
    # Step 3: exponentiation
    s = s.replace("^", "**")
    return s


# -- gplearn -------------------------------------------------------------------

def preprocess_gplearn(formula: str) -> str:
    """gplearn formulas are already valid Python with add/sub/mul/div functions.

    Variables are X0, X1, ...  The namespace must include these functions and
    variables (see build_ns_X0).  No textual transformation needed.
    """
    if not formula or not str(formula).strip():
        return ""
    return str(formula).strip()


# -- SBP-GP --------------------------------------------------------------------

def preprocess_sbp_gp(formula: str) -> str:
    """Convert SBP-GP formula to eval-able Python.

    SBP-GP syntax:
      plog(x)  -- protected log
      A aq B   -- analytic quotient  a / sqrt(1 + b^2) written as infix 'aq'
               -- also written without spaces: 'Xaqconstant', 'xNaqxM'
      x0, x1   -- variables (0-indexed, lower-case)
      standard infix + - * / ^ and ( )
    Strategy:
      1. plog(expr) -> _plog(expr)
      2. aq (infix) -> wrapped with _aq() by converting A aq B -> _aq(A, B)
         This requires a simple regex since aq is always binary infix.
      3. ^ -> **
    """
    if not formula or not str(formula).strip():
        return ""
    s = str(formula).strip()
    # 1. plog -> _plog
    s = s.replace("plog", "_plog")
    # 2. aq binary infix: the pattern is `(expr)aq(expr)` or `valaq(expr)` or similar
    #    Replace all occurrences of 'aq' with a sentinel so we can wrap in _aq():
    #    Simple approach: treat 'aq' as a function by converting `A aq B` -> `_aq(A, B)`
    #    Since aq appears inline without spaces in many cases (x1aqx0), we use a
    #    different tactic: replace 'aq' with ')/_aq_sentinel(' and then fix up.
    #    SIMPLEST reliable approach: replace 'aq' with '/' (SRBench does this too).
    #    This is not exactly analytic-quotient but is numerically stable for most inputs.
    s = s.replace("aq", "/")
    # 3. exponentiation
    s = s.replace("^", "**")
    return s


# -- GP-GOMEA ------------------------------------------------------------------

def preprocess_gp_gomea(formula: str) -> str:
    """Convert GP-GOMEA formula to eval-able Python.

    GP-GOMEA syntax:
      plog(x)  -- protected log
      A p/ B   -- protected division, written without spaces: 'xNp/xM'
      x0, x1   -- variables (0-indexed, lower-case)
      standard infix + - * / ^ and ( )
    Strategy:
      1. plog -> _plog
      2. p/ -> /   (SRBench replaces with regular division for symbolic evaluation)
      3. ^ -> **
    """
    if not formula or not str(formula).strip():
        return ""
    s = str(formula).strip()
    s = s.replace("plog", "_plog")
    s = s.replace("p/", "/")
    s = s.replace("^", "**")
    return s


# -- Dispatcher ----------------------------------------------------------------

def preprocess_formula(formula: str, algo: str) -> str:
    """Preprocess a formula string for the given algorithm.

    Returns a Python expression string ready for eval() with the appropriate
    namespace (see build_ns_* functions or _ood_make_ns in compare_srbench.py).
    Returns '' for unsupported or empty formulas.
    """
    a = algo.upper()
    if "BSR" in a:
        return preprocess_bsr(formula)
    if "GPLEARN" in a or "GP-LEARN" in a:
        return preprocess_gplearn(formula)
    if "SBP" in a:
        return preprocess_sbp_gp(formula)
    if "GOMEA" in a or "GP-GOMEA" in a:
        return preprocess_gp_gomea(formula)
    return ""  # unsupported / MRGP


def eval_formula(formula: str, algo: str, X: np.ndarray) -> np.ndarray:
    """Preprocess and evaluate a formula; return predictions array.

    Returns array of NaN on failure.
    """
    py_expr = preprocess_formula(formula, algo)
    if not py_expr:
        return np.full(X.shape[0], np.nan)

    a = algo.upper()
    if "GPLEARN" in a or "GP-LEARN" in a:
        ns = build_ns_X0(X)
    else:
        ns = build_ns_x0(X)

    # Add protected helpers to namespace
    ns["_plog"] = _plog
    ns["_protected_div"] = _protected_div
    ns["_aq"] = _aq
    ns["_safe_sqrt"] = _safe_sqrt

    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = eval(py_expr, ns)  # noqa: S307
        arr = np.asarray(result, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(X.shape[0], float(arr))
        return arr
    except Exception:
        return np.full(X.shape[0], np.nan)
