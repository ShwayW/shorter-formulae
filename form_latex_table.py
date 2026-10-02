#!/usr/bin/env python3
"""
form_latex_table.py -- render the 145M model's recovered formulae next to the
ground-truth targets as a LaTeX table.

The model emits formulae in standard Polish (prefix) notation; the targets live in
the two benchmark suites:

  * SRBench     -- Feynman ground-truth problems.  Targets are the Feynman equations
                   CSV (datasets/feynman/FeynmanEquations.csv), keyed by the PMLB
                   dataset name feynman_I_6_2 etc.  Model predictions come from
                   <results-root>/<group>/[seed<seed>/]noise_<tau>/eval_tf_<model>.pkl.gz.
  * LLM-SRBench -- targets are the split parquet (datasets/llmsrbench/data/<split>*.parquet),
                   keyed by problem name (I.10.7_1_0 ...).  Predictions come from
                   <results-root>/<group>/[seed<seed>/]noise_<tau>/llmsrbench/<model>_<split>/results.pkl.gz.

Both sides are converted to LaTeX math with sympy (which is already a dependency of
the repo's comparison scripts).  Variables are kept anonymous (v1..v10) on both sides,
and float constants are rounded to a small number of significant digits so the table
stays readable.

The script only ever READS results and benchmark files; it writes a single .tex file.

Usage
-----
    python form_latex_table.py                                # both benches, seed 42, noise 0
    python form_latex_table.py --bench srbench                # SRBench only
    python form_latex_table.py --bench llmsrbench             # LLM-SRBench only
    python form_latex_table.py --seed 45 --noise 0.01 --out results/formulae_table.tex
"""
import argparse
import csv
import gzip
import os
import pickle
import re

import sympy as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Grammar vocabulary + operator semantics (mirrors grammar.py; self-contained so
# this script does not import grammar.py, which loads the compiled C extension
# via funcWrappers).
# ---------------------------------------------------------------------------

V = ["v%d" % i for i in range(1, 11)]
C = ["0", "1", "2", "3", "pi"]
U = ["++", "--", "neg", "sqr", "sqrt", "exp", "ln", "sin", "cos", "tan",
     "abs", "arcsin", "arccos", "arctan", "invert", "pow2", "pow3", "tanh"]
B = ["+", "-", "*", "/", "pow"]

_V_SET = set(V)

_BINARY = {
    "+":   lambda a, b: a + b,
    "-":   lambda a, b: a - b,
    "*":   lambda a, b: a * b,
    "/":   lambda a, b: a / b,
    "pow": lambda a, b: a ** b,
}
_UNARY = {
    "++":     lambda a: a + 1,
    "--":     lambda a: a - 1,
    "neg":    lambda a: -a,
    "sqr":    lambda a: a ** 2,
    "sqrt":   sp.sqrt,
    "exp":    sp.exp,
    "ln":     sp.log,
    "sin":    sp.sin,
    "cos":    sp.cos,
    "tan":    sp.tan,
    "abs":    sp.Abs,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "invert": lambda a: 1 / a,
    "pow2":   lambda a: a ** 2,
    "pow3":   lambda a: a ** 3,
    "tanh":   sp.tanh,
}
_CONST = {"pi": sp.pi,
          "0": sp.Integer(0), "1": sp.Integer(1),
          "2": sp.Integer(2), "3": sp.Integer(3)}

# Functions that may appear in the *target* infix formulas.  `log` (LLM-SRBench) and
# `ln` (Feynman) both mean the natural logarithm.
_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sqrt": sp.sqrt, "abs": sp.Abs,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "tanh": sp.tanh,
}


# ---------------------------------------------------------------------------
# Polish-notation -> LaTeX
# ---------------------------------------------------------------------------

def _sig_round(x, sig):
    """Round a float to `sig` significant digits."""
    if x == 0.0:
        return 0.0
    from math import floor, log10
    d = sig - 1 - floor(log10(abs(x)))
    return round(x, d)


def pn_to_sympy(tokens):
    """Parse a Polish-notation token list -> (sympy expr, n_tokens_consumed).

    Variables v1..v10 become symbols v1..v10 (kept anonymous).  Raises ValueError
    on malformed or unknown input.
    """
    if not tokens:
        raise ValueError("empty token stream")
    tok, rest = tokens[0], tokens[1:]
    if tok in _BINARY:
        a, na = pn_to_sympy(rest)
        b, nb = pn_to_sympy(rest[na:])
        return _BINARY[tok](a, b), 1 + na + nb
    if tok in _UNARY:
        a, na = pn_to_sympy(rest)
        return _UNARY[tok](a), 1 + na
    if tok in _CONST:
        return _CONST[tok], 1
    if tok in _V_SET:
        return sp.Symbol(tok), 1
    try:
        f = float(tok)
    except (TypeError, ValueError):
        raise ValueError("unknown token: %r" % (tok,))
    return (sp.Integer(int(f)) if f == int(f) else sp.Float(f)), 1


def round_floats(expr, sig):
    """Round every Float leaf to `sig` significant digits (None = leave untouched).

    Iterated to a fixed point because sympy auto-folds numeric coefficients on
    substitution (e.g. rounding 88.8 * 0.0113 reproduces a full-precision
    Float on the next pass).
    """
    if sig is None:
        return expr
    for _ in range(64):
        changed = False
        for a in list(sp.preorder_traversal(expr)):
            if isinstance(a, sp.Float):
                v = _sig_round(float(a), sig)
                new = sp.Integer(int(v)) if v == int(v) else sp.Float(v)
                if new != a:
                    expr = expr.subs(a, new)
                    changed = True
        if not changed:
            break
    return expr


def pn_to_latex(pn, sig=None):
    """Polish-notation formula string -> LaTeX math string ('' on failure)."""
    pn = (pn or "").strip()
    if not pn:
        return ""
    try:
        expr, n = pn_to_sympy(pn.split())
        if n != len(pn.split()):
            raise ValueError("trailing tokens -- not clean prefix")
        expr = round_floats(expr, sig)
        return sp.latex(expr)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Target infix -> LaTeX
# ---------------------------------------------------------------------------

def infix_to_latex(formula, symbols, sig=None):
    """Python-infix target formula -> LaTeX math string ('' on failure).

    `symbols` is the list of variable names in input-column order; each is
    declared as an explicit symbol named v1..vN (so the target shares the model's
    anonymous variable naming), and the explicit declaration keeps names that
    collide with sympy built-ins (I, C, E, N, ...) treated as variables.
    """
    formula = (formula or "").strip()
    if not formula:
        return ""
    ns = {name: sp.Symbol("v%d" % (i + 1)) for i, name in enumerate(symbols)}
    ns.update(_FUNCS)
    ns["pi"] = sp.pi
    try:
        expr = sp.sympify(formula, locals=ns)
        expr = round_floats(expr, sig)
        return sp.latex(expr)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def _noise_dir(noise):
    return "noise_%g" % (noise,)


def _first_existing(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _load_pkl_gz(path):
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def _extract_r2(r):
    """R^2 for one result row (srbench: `r2`; llmsrbench: id_metrics['r2'])."""
    if "r2" in r:
        return r.get("r2")
    m = r.get("id_metrics") or {}
    return m.get("r2")


def _pick_by_seed(results, seed):
    """Return {problem: (predicted_pn, r2)} for one seed (first row per problem)."""
    out = {}
    for r in results:
        if r.get("seed") != seed:
            continue
        key = r.get("dataset") or r.get("equation_id")
        pn = r.get("predicted_formula") or r.get("discovered_equation") or ""
        if key and key not in out:
            out[key] = (pn, _extract_r2(r))
    return out


def load_srbench_predictions(results_root, group, model, noise, seed):
    """{feynman_X_Y_Z: (predicted_pn, r2)} for the requested seed."""
    nd = _noise_dir(noise)
    path = _first_existing(
        os.path.join(results_root, group, nd, "eval_tf_%s.pkl.gz" % model),
        os.path.join(results_root, group, "seed%d" % seed, nd,
                     "eval_tf_%s.pkl.gz" % model),
    )
    if not path:
        raise SystemExit(
            "No SRBench results for model %r (group %r, noise %g). Looked under:\n  %s"
            % (model, group, noise,
               os.path.join(results_root, group, nd, "eval_tf_%s.pkl.gz" % model)))
    data = _load_pkl_gz(path)
    return _pick_by_seed(data.get("results", []), seed)


def load_llmsrbench_predictions(results_root, group, model, noise, seed, split):
    """{equation_id: (discovered_equation, r2)} for the requested seed."""
    nd = _noise_dir(noise)
    method = "%s_%s" % (model, split)
    path = _first_existing(
        os.path.join(results_root, group, nd, "llmsrbench", method, "results.pkl.gz"),
        os.path.join(results_root, group, "seed%d" % seed, nd,
                     "llmsrbench", method, "results.pkl.gz"),
    )
    if not path:
        raise SystemExit(
            "No LLM-SRBench results for method %r (group %r, noise %g). Looked under:\n  %s"
            % (method, group, noise,
               os.path.join(results_root, group, nd, "llmsrbench", method, "results.pkl.gz")))
    data = _load_pkl_gz(path)
    return _pick_by_seed(data.get("results", []), seed)


# ---------------------------------------------------------------------------
# Target loading (SRBench = Feynman CSV, LLM-SRBench = split parquet)
# ---------------------------------------------------------------------------

def load_feynman_targets(feynman_csv):
    """{feynman_X_Y_Z: (formula, [var_names_in_order])} from the Feynman CSV."""
    targets = {}
    if not os.path.isfile(feynman_csv):
        return targets
    with open(feynman_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fname = (row.get("Filename") or "").strip()
            formula = (row.get("Formula") or "").strip()
            if not fname or not formula:
                continue
            var_names = [(row.get("v%d_name" % k) or "").strip()
                         for k in range(1, 11)]
            var_names = [n for n in var_names if n]
            dataset = "feynman_" + fname.replace(".", "_")
            targets[dataset] = (formula, var_names)
    return targets


def load_llmsrbench_targets(llmsrbench_dir, split):
    """{name: (expression, [symbols])} from the split parquet."""
    import pandas as pd

    parquet = os.path.join(llmsrbench_dir, "data",
                           "%s-00000-of-00001.parquet" % split)
    if not os.path.isfile(parquet):
        return {}
    meta = pd.read_parquet(parquet)
    return {str(r["name"]): (str(r["expression"]), list(r["symbols"]))
            for _, r in meta.iterrows()}


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------

def _nat_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


def _tex_escape(text):
    return (text.replace("\\", r"\textbackslash{}")
                .replace("_", r"\_")
                .replace("&", r"\&")
                .replace("%", r"\%")
                .replace("#", r"\#")
                .replace("$", r"\$")
                .replace("{", r"\{")
                .replace("}", r"\}")
                .replace("~", r"\textasciitilde{}")
                .replace("^", r"\textasciicircum{}"))


def _math_or_raw(latex, raw):
    """$latex$ when available, else the raw string in a typewriter font."""
    if latex:
        return "$%s$" % latex
    if raw:
        return r"\texttt{%s}" % _tex_escape(raw)
    return ""


def _fmt_r2(r2):
    """Format an R^2 value for the table ('' when missing / not a number)."""
    if r2 is None:
        return ""
    try:
        v = float(r2)
    except (TypeError, ValueError):
        return ""
    if v != v:  # NaN
        return ""
    return "%.3f" % v


def build_rows(preds_sr, targets_sr, preds_lsr, targets_lsr, sig):
    """Return a list of (benchmark, problem, model_latex, target_latex, r2) tuples."""
    rows = []

    for dataset in sorted(preds_sr, key=_nat_key):
        pred_pn, r2 = preds_sr[dataset]
        target = targets_sr.get(dataset)
        if target is None:
            # e.g. feynman_test_* -- no ground-truth formula shipped with the repo.
            target_latex = ""
        else:
            formula, var_names = target
            target_latex = infix_to_latex(formula, var_names, sig)
        model_latex = pn_to_latex(pred_pn, sig)
        rows.append(("SRBench", dataset, model_latex, target_latex, _fmt_r2(r2)))

    for name in sorted(preds_lsr, key=_nat_key):
        pred_pn, r2 = preds_lsr[name]
        target = targets_lsr.get(name)
        if target is None:
            target_latex = ""
        else:
            expression, symbols = target
            target_latex = infix_to_latex(expression, symbols[1:], sig)
        model_latex = pn_to_latex(pred_pn, sig)
        rows.append(("LLM-SRBench", name, model_latex, target_latex, _fmt_r2(r2)))

    return rows


def write_tex(rows, label, out_path, landscape=True):
    header = (r"\toprule" "\n"
              r"Benchmark & Problem & %s & Target & $R^2$ \\" "\n"
              r"\midrule" "\n"
              r"\endhead" "\n") % _tex_escape(label)

    body = []
    for benchmark, problem, model_latex, target_latex, r2 in rows:
        body.append("%s & %s & %s & %s & %s \\\\" % (
            _tex_escape(benchmark),
            _tex_escape(problem),
            _math_or_raw(model_latex, None),
            _math_or_raw(target_latex, None),
            r2,
        ))
    body.append(r"\bottomrule")

    cls = "landscape" if landscape else "article"
    document = r"""\documentclass[%s]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\begin{document}
\footnotesize
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.5}

\begin{longtable}{@{} l l >{\raggedright\arraybackslash}p{3.2in} >{\raggedright\arraybackslash}p{3.2in} c @{}}
%s
%s
\end{longtable}

\end{document}
""" % (cls, header, "\n".join(body))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(document)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="results",
                    help="Results root directory. Default: results.")
    ap.add_argument("--group", default="mymodels",
                    help="Group directory under --results-root holding the model's "
                         "runs. Default: mymodels.")
    ap.add_argument("--model", default="145M_40_simp1",
                    help="Model name (srbench file stem / llmsrbench method prefix). "
                         "Default: 145M_40_simp1.")
    ap.add_argument("--label", default="145M",
                    help="Display label for the model column. Default: 145M.")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Target-noise level of the results to read. Default: 0.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Which evaluation seed's formula to show. Default: 42.")
    ap.add_argument("--bench", choices=["both", "srbench", "llmsrbench"],
                    default="both",
                    help="Which benchmark(s) to include. Default: both.")
    ap.add_argument("--split", default="lsr_transform",
                    help="LLM-SRBench split. Default: lsr_transform.")
    ap.add_argument("--feynman-csv",
                    default=os.path.join(SCRIPT_DIR, "datasets", "feynman",
                                         "FeynmanEquations.csv"),
                    help="Path to the SRBench (Feynman) ground-truth CSV.")
    ap.add_argument("--llmsrbench-dir",
                    default=os.path.join(SCRIPT_DIR, "datasets", "llmsrbench"),
                    help="Path to the LLM-SRBench dataset directory.")
    ap.add_argument("--round", type=int, default=3,
                    help="Significant digits for float constants in the model "
                         "formula. Default: 3.")
    ap.add_argument("--no-round", action="store_true",
                    help="Keep float constants at full precision.")
    ap.add_argument("--portrait", action="store_true",
                    help="Render the table in portrait orientation (default: landscape).")
    ap.add_argument("--out", default=None,
                    help="Output .tex file. Default: results/formulae_table.tex")
    args = ap.parse_args()

    sig = None if args.no_round else args.round

    results_root = os.path.join(SCRIPT_DIR, args.results_root)
    preds_sr = {}
    preds_lsr = {}

    if args.bench in ("both", "srbench"):
        preds_sr = load_srbench_predictions(
            results_root, args.group, args.model, args.noise, args.seed)
        print("[srbench] %d predictions (seed %d, noise %g)"
              % (len(preds_sr), args.seed, args.noise), flush=True)

    if args.bench in ("both", "llmsrbench"):
        preds_lsr = load_llmsrbench_predictions(
            results_root, args.group, args.model, args.noise, args.seed, args.split)
        print("[llmsrbench] %d predictions (seed %d, noise %g)"
              % (len(preds_lsr), args.seed, args.noise), flush=True)

    targets_sr = load_feynman_targets(args.feynman_csv) if preds_sr else {}
    targets_lsr = (load_llmsrbench_targets(args.llmsrbench_dir, args.split)
                   if preds_lsr else {})

    rows = build_rows(preds_sr, targets_sr, preds_lsr, targets_lsr, sig)

    if args.out is None:
        args.out = os.path.join("results", "formulae_table.tex")
    out_path = os.path.join(SCRIPT_DIR, args.out)
    write_tex(rows, args.label, out_path, landscape=not args.portrait)

    n_sr = sum(1 for r in rows if r[0] == "SRBench")
    n_lsr = sum(1 for r in rows if r[0] == "LLM-SRBench")
    n_no_target = sum(1 for r in rows if not r[3])
    print("\n%s rows written -> %s" % (len(rows), args.out))
    print("  SRBench: %d  |  LLM-SRBench: %d  |  no target formula: %d"
          % (n_sr, n_lsr, n_no_target))


if __name__ == "__main__":
    main()
