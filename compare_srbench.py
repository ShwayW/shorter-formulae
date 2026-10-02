#!/usr/bin/env python3
"""
compare_srbench.py -- Compare the E2E transformer against SRBench methods on
the Feynman benchmark.

Usage:
    python compare_srbench.py [--results results/results_feynman.csv]
                              [--noise 0.0]
                              [--r2-threshold 0.99]
                              [--output plots/srbench_comparison.png]
                              [--ood-data datasets/feynman_ood_g5.pkl.gz]
                              [--reuse-ood-csv] [--no-ood]
                              [--no-plot]  [--show-plot]

The transformer results CSV is produced by eval_e2e.py.  If the file does not
exist the script still runs and shows only the SRBench baseline numbers.

Out-of-distribution (OOD) R^2 for the right-hand scatter is computed in-process
by re-evaluating each method's predicted formula on --ood-data (formerly the
standalone compare_ood.py -> results/ood_multi.csv hand-off).  The computed table
is still written to --ood-csv as an artifact; --reuse-ood-csv reads it back
instead of recomputing, and --no-ood skips OOD altogether.

Noise levels in ground-truth_results.feather: 0.0, 0.001, 0.01, 0.1

Two accuracy metrics are reported:
  Solve Rate  -- mean over datasets of (fraction of seeds that solved it).
                Directly comparable between transformer and SRBench.
  Strict Acc  -- fraction of datasets where the *mean* R^2 over seeds >= threshold.
                Requires the algorithm to solve an equation consistently.
"""

import argparse
import csv
import gzip
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
import sympy as sp
from sympy import preorder_traversal, Float, Integer

from grammar import V   # grammar variables (single source of truth)
import results_io
import phye2e_addon
import pysr_addon
from plot_io import save_fig

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SRBENCH_GT   = os.path.join(SCRIPT_DIR, "srbench", "results", "ground-truth_results.feather")
FEYNMAN_CSV  = os.path.join(SCRIPT_DIR, "datasets", "feynman", "FeynmanEquations.csv")

# -- Formula complexity (SRBench-compatible) ----------------------------------
# SRBench's `simplified_complexity` = number of nodes in the sympy expression
# tree after round_floats + simplify(ratio=1) (srbench/experiment/symbolic_utils.py:
# complexity() counts preorder_traversal nodes).  We reproduce that here so our
# transformer's complexity is measured on the same scale as the baselines.
#
# Our predicted formulae come in prefix (Polish) notation (eval_mymodels.py) with the
# grammar.py vocabulary, or infix (eval_e2e.py).  Operator semantics mirror
# bfgs_optimizer.py / extEvalPN.cpp.
_SP_BINARY = {
    "+":   lambda a, b: a + b,
    "-":   lambda a, b: a - b,
    "*":   lambda a, b: a * b,
    "/":   lambda a, b: a / b,
    "pow": lambda a, b: a ** b,
}
_SP_UNARY = {
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
_SP_CONST = {"pi": sp.pi}
_VAR_SET  = set(V)   # grammar variables (single source of truth: grammar.py)


def _prefix_to_sympy(tokens: list):
    """Build a sympy expr from a prefix token list -> (expr, n_tokens_consumed).
    Raises ValueError on malformed / unknown input."""
    if not tokens:
        raise ValueError("empty token stream")
    tok, rest = tokens[0], tokens[1:]
    if tok in _SP_BINARY:
        a, na = _prefix_to_sympy(rest)
        b, nb = _prefix_to_sympy(rest[na:])
        return _SP_BINARY[tok](a, b), 1 + na + nb
    if tok in _SP_UNARY:
        a, na = _prefix_to_sympy(rest)
        return _SP_UNARY[tok](a), 1 + na
    if tok in _SP_CONST:
        return _SP_CONST[tok], 1
    if tok in _VAR_SET:
        return sp.Symbol(tok), 1
    try:
        f = float(tok)
    except (TypeError, ValueError):
        raise ValueError(f"unknown token: {tok!r}")
    return (Integer(int(f)) if f == int(f) else Float(f)), 1


def _round_floats(expr):
    """Mirror SRBench round_floats: zero out tiny floats, round others to 3 dp."""
    out = expr
    for a in preorder_traversal(expr):
        if isinstance(a, Float):
            if abs(a) < 1e-4:
                out = out.subs(a, Integer(0))
            else:
                out = out.subs(a, Float(round(float(a), 3), 3))
    return out


def formula_complexity(formula) -> float:
    """Complexity = node count of the parsed sympy expression tree, WITHOUT
    simplification (constants are rounded via _round_floats, which does not change
    the node count).  Measured identically for predicted and ground-truth
    formulae so they are directly comparable.  Accepts prefix (preferred) or
    infix strings.  Returns NaN when the formula is missing or unparseable."""
    if formula is None or (isinstance(formula, float) and pd.isna(formula)):
        return float("nan")
    s = str(formula).strip()
    if not s:
        return float("nan")
    try:
        toks = s.split()
        try:
            expr, n = _prefix_to_sympy(toks)
            if n != len(toks):
                raise ValueError("trailing tokens -- not clean prefix")
        except ValueError:
            expr = sp.sympify(s)                  # fall back to infix
        expr = _round_floats(expr)
        # NOTE: no sp.simplify() -- count the raw parsed tree (per user request).
        return float(sum(1 for _ in preorder_traversal(expr)))
    except Exception:
        return float("nan")

# -- Data loading ---------------------------------------------------------------

# Result artifacts are .pkl.gz since the results_io migration; .csv is the legacy
# form still found in archived trees (results/_old).  os.path.splitext is useless
# here -- it splits "x.pkl.gz" into ("x.pkl", ".gz") -- so the two helpers below
# treat ".pkl.gz" as one extension.
_ARTIFACT_EXTS = (".pkl.gz", ".pkl", ".csv")


def _split_artifact_ext(path: str):
    for ext in _ARTIFACT_EXTS:
        if path.endswith(ext):
            return path[:-len(ext)], ext
    return os.path.splitext(path)


def _resolve_artifact(path: str) -> str:
    """`path` if it exists, else the same stem with another known extension.

    Lets --e2e-results default to the current .pkl.gz while an archived .csv run
    still loads. Returns `path` unchanged when nothing exists, so the caller's
    os.path.exists() guard still decides whether to include the method.
    """
    if os.path.exists(path):
        return path
    stem, _ = _split_artifact_ext(path)
    for ext in _ARTIFACT_EXTS:
        alt = stem + ext
        if os.path.exists(alt):
            return alt
    return path


def _seed_std_of_mean(df, seed_col: str, value_col: str) -> float:
    """+/-1 std, across seeds, of the per-seed mean of `value_col`.

    Each seed contributes one number (its mean over the datasets it solved); the
    spread of those is the uncertainty we report. 0.0 when there is a single seed.
    Used for the complexity bars, mirroring the solve-rate bands.
    """
    if seed_col not in df.columns or value_col not in df.columns:
        return 0.0
    per_seed = df.groupby(seed_col)[value_col].mean().dropna().to_numpy(dtype=float)
    return float(np.std(per_seed)) if per_seed.size >= 2 else 0.0


def _load_pkl_gz(pkl_path: str, with_complexity: bool = True) -> "pd.DataFrame | None":
    """Load eval_mymodels.py .pkl.gz results into a flat DataFrame with columns
    dataset, seed, r2, predicted_formula, target_formula.

    When --failed-only re-evaluation rounds are present, multiple rows exist
    per (dataset, seed). Keep only the best R^2 row per pair so that a dataset
    solved in any round counts as solved.

    with_complexity=False skips the (sympy, per-formula) complexity computation --
    callers that only need the time/R^2 columns (e.g. collect_timing) avoid parsing
    every formula, which otherwise dominates their runtime.
    """
    with gzip.open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    results = data.get("results", [])
    if not results:
        print(f"[warn] {pkl_path} contains no results.")
        return None
    df = pd.DataFrame(results)
    # Deduplicate: keep the row with the highest R^2 per (dataset, seed).
    if "seed" in df.columns:
        df = (df.loc[df.groupby(["dataset", "seed"])["r2"].idxmax()]
                .reset_index(drop=True))
    if with_complexity and "predicted_formula" in df.columns:
        df["complexity"] = df["predicted_formula"].apply(formula_complexity)
    return df


def load_transformer_results(path: str, r2_thr: float,
                             with_complexity: bool = True) -> "pd.DataFrame | None":
    """Per-dataset aggregated transformer results.

    Accepts either:
      * a CSV from eval_e2e.py  (columns: dataset, r2, seed, [accuracy])
      * a .pkl.gz from eval_mymodels.py (columns: dataset, r2, seed, ...)

    with_complexity=False skips the sympy complexity pass for callers that only use
    the timing/R^2 aggregates (the resulting frame simply omits the complexity_*
    columns).
    """
    if not os.path.exists(path):
        print(f"[warn] Transformer results not found: {path}")
        print("       Run eval_mymodels.py (or eval_e2e.py) first.  Showing SRBench baselines only.")
        return None

    if path.endswith(".pkl.gz") or path.endswith(".pkl"):
        df = _load_pkl_gz(path, with_complexity=with_complexity)
    else:
        df = pd.read_csv(path)

    if df is None or df.empty:
        print(f"[warn] {path} is empty -- skipping transformer.")
        return None

    # Normalize time-column aliases so eval_e2e.py CSVs (which emit
    # 'predict_time_s') get an 'Avg Time (s)' like the eval_mymodels pkls do.
    df = df.rename(columns={"predict_time_s": "time"})

    # CSV path (eval_e2e.py): compute SRBench-style complexity from the formula
    # if not already present. The .pkl.gz path computes it in _load_pkl_gz.
    if with_complexity and "complexity" not in df.columns and "predicted_formula" in df.columns:
        df["complexity"] = df["predicted_formula"].apply(formula_complexity)

    if "accuracy" not in df.columns:
        df["accuracy"] = (df["r2"] >= r2_thr).astype(float)
    agg_cols = {
        "r2_mean":    ("r2", "max"),       # best R^2 ever achieved for this dataset
        "r2_std":     ("r2", "std"),
        "solve_rate": ("accuracy", "max"), # solved if solved in any round/seed
        "n_seeds":    ("seed", "count"),
    }
    if "time" in df.columns:
        agg_cols["time_mean"] = ("time", "mean")
        agg_cols["time_std"]  = ("time", "std")
    if "complexity" in df.columns:
        agg_cols["complexity_mean"] = ("complexity", "mean")
        agg_cols["complexity_std"]  = ("complexity", "std")
        # Complexity restricted to solved formulae (R^2 >= threshold). NaN for
        # unsolved rows is skipped by pandas' mean/std.
        df["complexity_solved"] = df["complexity"].where(df["accuracy"] == 1)
        agg_cols["complexity_solved_mean"] = ("complexity_solved", "mean")
        agg_cols["complexity_solved_std"]  = ("complexity_solved", "std")
    agg = df.groupby("dataset").agg(**agg_cols).reset_index()
    # Seed-to-seed spread of the mean solved complexity. Constant per algorithm, so it
    # survives build_comparison_table's mean-over-datasets aggregation untouched.
    agg["complexity_seed_std"] = _seed_std_of_mean(df, "seed", "complexity_solved") \
        if "complexity_solved" in df.columns else 0.0
    # Same quantity for the run time, so the time bars report seed-to-seed spread like
    # the complexity ones instead of problem-difficulty spread (collect_timing).
    agg["time_seed_std"] = _seed_std_of_mean(df, "seed", "time") \
        if "time" in df.columns else 0.0
    return agg


def load_tpsr_results(path: str, r2_thr: float, noise: float = 0.0) -> "pd.DataFrame | None":
    """Load TPSR Feynman CSV as a per-dataset agg frame compatible with
    build_comparison_table (rows at the given target_noise, one row per dataset)."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["target_noise"] == noise].copy()
    if df.empty:
        return None
    df["dataset"]              = df["problem"]
    df["r2_mean"]              = df["r2_predict"].clip(0.0, 1.0)
    df["r2_std"]               = float("nan")
    df["solve_rate"]           = (df["r2_predict"] >= r2_thr).astype(float)
    df["n_seeds"]              = 1
    df["time_mean"]            = df["time"]
    df["time_std"]             = float("nan")
    df["complexity_mean"]      = df["_complexity_predict"]
    df["complexity_std"]       = float("nan")
    df["complexity_solved_mean"] = df["_complexity_predict"].where(df["solve_rate"] == 1)
    df["complexity_solved_std"]  = float("nan")
    df["complexity_seed_std"]    = 0.0   # authors' CSV is a single run -- no seed axis
    df["time_seed_std"]          = 0.0   # likewise
    return df[["dataset", "r2_mean", "r2_std", "solve_rate", "n_seeds",
               "time_mean", "time_std", "complexity_mean", "complexity_std",
               "complexity_solved_mean", "complexity_solved_std", "complexity_seed_std",
               "time_seed_std"]]


def load_srbench_gt(feather_path: str, noise: float, r2_thr: float) -> pd.DataFrame:
    """
    Load SRBench ground-truth Feynman results at the given noise level.
    Returns per-(algorithm, dataset) aggregated frame with the same columns
    as load_transformer_results, so the two can be compared directly.
    """
    df = pd.read_feather(feather_path)
    mask = (df["data_group"] == "Feynman") & (df["target_noise"] == noise)
    df = df[mask].copy()
    # Clip to [0, 1] -- standard reporting convention; extreme negatives skew means.
    df["r2_clipped"] = df["r2_test"].clip(0.0, 1.0)
    df["solved"]     = (df["r2_test"] >= r2_thr).astype(float)   # unclipped for accuracy
    # Complexity restricted to solved formulae (R^2 >= threshold).
    df["complexity_solved"] = df["simplified_complexity"].where(df["solved"] == 1)
    agg = (
        df.groupby(["algorithm", "dataset"])
        .agg(
            r2_mean=("r2_clipped", "mean"),
            r2_std=("r2_clipped", "std"),
            solve_rate=("solved", "mean"),        # fraction of seeds that solved it
            symbolic_solution_rate=("symbolic_solution", "mean"),
            n_seeds=("random_state", "count"),
            time_mean=("training time (s)", "mean"),
            time_std=("training time (s)", "std"),
            complexity_mean=("simplified_complexity", "mean"),
            complexity_std=("simplified_complexity", "std"),
            complexity_solved_mean=("complexity_solved", "mean"),
            complexity_solved_std=("complexity_solved", "std"),
        )
        .reset_index()
    )
    # Seed-to-seed spread of the mean solved complexity, per algorithm. SRBench's seed
    # column is `random_state`. Attached as a constant column per algorithm so
    # build_comparison_table's mean-over-datasets leaves it unchanged.
    seed_std = {
        alg: _seed_std_of_mean(grp, "random_state", "complexity_solved")
        for alg, grp in df.groupby("algorithm")
    }
    agg["complexity_seed_std"] = agg["algorithm"].map(seed_std).fillna(0.0)
    # Same for the training time -- the time bars report seed spread, not the spread
    # across datasets (see collect_timing).
    time_seed_std = {
        alg: _seed_std_of_mean(grp, "random_state", "training time (s)")
        for alg, grp in df.groupby("algorithm")
    }
    agg["time_seed_std"] = agg["algorithm"].map(time_seed_std).fillna(0.0)
    return agg


def feynman_truth_complexity(feather_path: str, noise: float, datasets: set):
    """Complexity of the *true* Feynman formulae over the given datasets, using
    the same (unsimplified) formula_complexity() as everything else.

    Returns a dict with the distribution's mean/std and quantiles:
      {mean, std, q25, q50 (median), q75, series}
    so callers can show the spread (a variance band), not just a point."""
    df = pd.read_feather(feather_path)
    mask = (df["data_group"] == "Feynman") & (df["target_noise"] == noise)
    df = df[mask]
    # One true_model per dataset (identical across algorithms/seeds).
    truth = df.drop_duplicates("dataset").set_index("dataset")["true_model"]
    truth = truth[truth.index.isin(datasets)]
    cplx = truth.apply(formula_complexity).dropna()
    if cplx.empty:
        nan = float("nan")
        return {"mean": nan, "std": nan, "q25": nan, "q50": nan, "q75": nan, "series": cplx}
    return {
        "mean": float(cplx.mean()), "std": float(cplx.std()),
        "q25": float(cplx.quantile(0.25)), "q50": float(cplx.quantile(0.50)),
        "q75": float(cplx.quantile(0.75)), "series": cplx,
    }


# -- Benchmark-level statistics -------------------------------------------------

def benchmark_stats(per_dataset: pd.DataFrame, r2_thr: float) -> dict:
    """
    Accepts a per-dataset frame with 'r2_mean' and 'solve_rate' columns.

    solve_rate   -- fraction of seeds that solved each dataset (already per-dataset).
    mean_solve   -- mean of solve_rate over all datasets (= overall solve rate).
    strict_acc   -- fraction of datasets where mean R^2 >= r2_thr.
    """
    strict_solved = int((per_dataset["r2_mean"] >= r2_thr).sum())
    n = len(per_dataset)
    return {
        "mean_r2":      float(per_dataset["r2_mean"].mean()),
        "median_r2":    float(per_dataset["r2_mean"].median()),
        "mean_solve":   float(per_dataset["solve_rate"].mean()),   # primary metric
        "strict_acc":   float(strict_solved / n),
        "n_strict":     strict_solved,
        "n_total":      n,
    }


# -- Comparison table -----------------------------------------------------------

def build_comparison_table(
    tf_results:       "list[tuple[pd.DataFrame, str]]",
    srbench_agg:      pd.DataFrame,
    common_datasets:  set,
    r2_thr:           float,
    noise:            float,
) -> pd.DataFrame:
    """tf_results is a list of (per-dataset agg DataFrame, label) pairs."""
    rows = []

    # SRBench methods -- fill datasets the algorithm didn't evaluate with R^2=0 / solved=0
    # so that all methods are compared on the same full set of common_datasets.
    n_common = len(common_datasets)
    filtered = srbench_agg[srbench_agg["dataset"].isin(common_datasets)]
    for alg, grp in filtered.groupby("algorithm"):
        missing = common_datasets - set(grp["dataset"])
        if missing:
            fill = pd.DataFrame({
                "dataset":                list(missing),
                "r2_mean":                0.0,
                "r2_std":                 float("nan"),
                "solve_rate":             0.0,
                "symbolic_solution_rate": float("nan"),
                "n_seeds":                0,
            })
            grp = pd.concat([grp, fill], ignore_index=True)
        s = benchmark_stats(grp, r2_thr)
        has_sym = grp["symbolic_solution_rate"].notna().any()
        rows.append({
            "Algorithm":     alg,
            "Mean R^2":       s["mean_r2"],
            "Median R^2":     s["median_r2"],
            "Solve Rate":    s["mean_solve"],
            "Strict Acc":    s["strict_acc"],
            "Strict Solved": f"{s['n_strict']}/{n_common}",
            "Sym. Sol. Rate": grp["symbolic_solution_rate"].mean() if has_sym else float("nan"),
            "Avg Time (s)":  grp["time_mean"].mean(),
            "Time Std (s)":  grp["time_mean"].std(),
            "Complexity":     grp["complexity_mean"].mean(),
            "Complexity Std": grp["complexity_mean"].std(),
            "Complexity (solved)":     grp["complexity_solved_mean"].mean(),
            "Complexity Std (solved)": grp["complexity_solved_mean"].std(),
            # Constant per algorithm (see load_srbench_gt); .mean() just reads it back.
            "Complexity Seed Std (solved)": (grp["complexity_seed_std"].mean()
                                             if "complexity_seed_std" in grp.columns else 0.0),
            "Source": "SRBench",
        })

    # Transformer models
    for tf_agg, label in tf_results:
        if tf_agg is None:
            continue
        tf_f = tf_agg[tf_agg["dataset"].isin(common_datasets)]
        s = benchmark_stats(tf_f, r2_thr)
        has_time = "time_mean" in tf_f.columns
        rows.append({
            "Algorithm":     label,
            "Mean R^2":       s["mean_r2"],
            "Median R^2":     s["median_r2"],
            "Solve Rate":    s["mean_solve"],
            "Strict Acc":    s["strict_acc"],
            "Strict Solved": f"{s['n_strict']}/{n_common}",
            "Sym. Sol. Rate": float("nan"),
            "Avg Time (s)":  tf_f["time_mean"].mean() if has_time else float("nan"),
            "Time Std (s)":  tf_f["time_mean"].std()  if has_time else float("nan"),
            "Complexity":     tf_f["complexity_mean"].mean() if "complexity_mean" in tf_f.columns else float("nan"),
            "Complexity Std": tf_f["complexity_mean"].std()  if "complexity_mean" in tf_f.columns else float("nan"),
            "Complexity (solved)":     tf_f["complexity_solved_mean"].mean() if "complexity_solved_mean" in tf_f.columns else float("nan"),
            "Complexity Std (solved)": tf_f["complexity_solved_mean"].std()  if "complexity_solved_mean" in tf_f.columns else float("nan"),
            "Complexity Seed Std (solved)": (tf_f["complexity_seed_std"].mean()
                                             if "complexity_seed_std" in tf_f.columns else 0.0),
            "Source": "Ours",
        })

    table = pd.DataFrame(rows)
    table = table.sort_values("Solve Rate", ascending=False).reset_index(drop=True)
    return table


# -- Plots ----------------------------------------------------------------------

_BASELINE_COLOR = "#457b9d"   # steel blue (all SRBench baselines share this)
_TRANSFORMER_PALETTE = [
    "#e63946", "#f4a261", "#e9c46a", "#2a9d8f",   # red, sandy orange, muted yellow, teal
    "#8338ec", "#fb5607", "#ff006e", "#3a86ff",   # violet, bright orange, magenta/pink, azure blue
]


def _build_color_map(tf_labels: list) -> dict:
    """Map each transformer model label to a distinct colour."""
    return {lbl: _TRANSFORMER_PALETTE[i % len(_TRANSFORMER_PALETTE)]
            for i, lbl in enumerate(tf_labels)}


_FLOAT_COLOR = "#db2777"   # rose -- the 89M float/continuous model (m89float_res), distinct from
                           # baselines (blue) and the mantissa-free 89M/145M models (green/orange).


# NOTE: solve-rate uncertainty is the seed-to-seed spread (+/-1 std over the 10 seeds),
# never a binomial/Wilson interval. A Wilson CI would treat the datasets as i.i.d.
# Bernoulli trials, which they are not -- the benchmark is fixed, and the only thing
# we actually resample is the seed.


def _color(alg: str, tf_color_map: dict) -> str:
    return tf_color_map.get(alg, _BASELINE_COLOR)


def plot_comparison(table: pd.DataFrame, r2_thr: float, noise: float, out_path: str,
                    tf_color_map: "dict | None" = None,
                    truth_complexity: "dict | None" = None,
                    ood_df: "pd.DataFrame | None" = None,
                    gap=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if tf_color_map is None:
        tf_color_map = {}

    FS_TITLE  = 16
    FS_LABEL  = 14
    FS_TICK   = 13
    FS_ANNOT  = 11
    FS_LEGEND = 12
    FS_SUPER  = 17

    # -- Pre-compute which algorithms have OOD data (filters both subplots) ------
    ood_rate:   dict[str, float] = {}
    ood_count:  dict[str, tuple] = {}
    ood_ci_lo:  dict[str, float] = {}
    ood_ci_hi:  dict[str, float] = {}
    if ood_df is not None:
        # Common denominator: every method's solve rate is out of the full dataset
        # universe (datasets it didn't evaluate / failed on count as unsolved), so
        # coverage gaps (e.g. SRBench's 116 or AIFeynman's 62 vs our 119) don't
        # inflate a method's rate relative to one evaluated on every dataset.
        n_universe = int(ood_df["dataset"].nunique())
        if "seed" in ood_df.columns:
            # Multi-seed data: the point is the MEAN solve rate across seeds and the
            # error bar spans +/-1 seed-to-seed standard deviation -- the same statistic
            # plot_ood_vs_gap.py shades.  It measures how much the rate wobbles between
            # random seeds, which is what we can actually resample; a binomial/Wilson
            # interval would instead assume the datasets are i.i.d. Bernoulli draws.
            for alg, alg_grp in ood_df.groupby("algorithm"):
                rates = np.array(
                    [int((sg["ood_r2"] >= r2_thr).sum()) / n_universe
                     for _, sg in alg_grp.groupby("seed")],
                    dtype=float,
                ) if n_universe else np.array([])
                if rates.size == 0:
                    continue
                mean = float(np.mean(rates))
                std  = float(np.std(rates))       # population std (0 for a single seed)
                ood_count[alg] = (int(round(mean * n_universe)), n_universe)
                ood_rate[alg]  = mean
                ood_ci_lo[alg] = max(0.0, mean - std)
                ood_ci_hi[alg] = min(1.0, mean + std)
        else:
            # One best formula per dataset: count distinct datasets solved (dedupe by
            # dataset so pooled-seed sources like m89float aren't counted repeatedly)
            # over the full universe.  With no seed axis there is no spread to show.
            for alg, grp in ood_df.groupby("algorithm"):
                n_correct = int((grp.groupby("dataset")["ood_r2"].max() >= r2_thr).sum())
                rate = n_correct / n_universe if n_universe else float("nan")
                ood_count[alg] = (n_correct, n_universe)
                ood_rate[alg]  = rate
                ood_ci_lo[alg] = ood_ci_hi[alg] = rate

    timed = table.dropna(subset=["Avg Time (s)"]).copy()
    if ood_df is not None:
        print(f"[plot] OOD CSV algorithms  : {sorted(ood_count.keys())}")
    print(f"[plot] Table algorithms (have time data): {list(timed['Algorithm'])}")
    timed["ood_x"] = timed["Algorithm"].map(ood_rate)
    no_match = timed[timed["ood_x"].isna()]["Algorithm"].tolist()
    if no_match:
        print(f"[plot] No OOD entry -- excluded from both subplots: {no_match}")
    timed = timed.dropna(subset=["ood_x"])
    print(f"[plot] {len(timed)} algorithms in both subplots: {list(timed['Algorithm'])}")

    fig, ax_s = plt.subplots(1, 1, figsize=(11, max(6, len(timed) * 0.55 + 2)))
    gap_str = f", OOD shift={gap}" if gap is not None else ""
    fig.suptitle(
        f"SRBench Feynman Benchmark  (noise={noise},  R^2-threshold={r2_thr}{gap_str})",
        fontsize=FS_SUPER, fontweight="bold", y=1.01,
    )

    # -- OOD R^2 vs. Speed scatter --------------------------------------------
    # Formula complexity is now its own cross-noise figure (plot_complexity_vs_noise.py).

    # Set log scale BEFORE adding data so autoscaling applies correctly.
    ax_s.set_yscale("log")

    if timed.empty:
        ax_s.text(0.5, 0.5, "No OOD data matched\n(check --ood-data / --no-ood)",
                  ha="center", va="center", transform=ax_s.transAxes,
                  fontsize=FS_LABEL, color="grey")
    else:
        for _, row in timed.iterrows():
            alg   = row["Algorithm"]
            color = _color(alg, tf_color_map)
            x     = float(row["ood_x"])
            yerr  = row["Time Std (s)"] if pd.notna(row["Time Std (s)"]) else 0.0
            lo    = ood_ci_lo.get(alg, x)
            hi    = ood_ci_hi.get(alg, x)
            ax_s.errorbar(
                x, float(row["Avg Time (s)"]),
                xerr=[[x - lo], [hi - x]],
                yerr=float(yerr),
                fmt="o", color=color,
                markersize=10, markeredgecolor="white", markeredgewidth=0.8,
                elinewidth=1.4, capsize=4, ecolor=color, alpha=0.85,
            )
            label = str(alg)
            ax_s.annotate(
                label,
                xy=(x, float(row["Avg Time (s)"])),
                xytext=(6, 4), textcoords="offset points",
                fontsize=FS_ANNOT, color=color,
            )
        ax_s.relim()
        ax_s.autoscale_view()

        # Reference lines at human-readable compute budgets (y-axis is seconds).
        time_marks = [(60, "1 min"), (600, "10 min"), (1800, "30 min"), (3600, "1 hour"), (10800, "3 hours"), (36000, "10 hours")]
        for ysec, lbl in time_marks:
            ax_s.axhline(ysec, color="#444", lw=1.2, ls=":", alpha=0.7, zorder=1)
            ax_s.text(
                0.995, ysec, lbl, transform=ax_s.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=FS_ANNOT - 1, color="#444",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )
        # Keep every reference line within view regardless of the data's range.
        ymin, ymax = ax_s.get_ylim()
        ax_s.set_ylim(min(ymin, time_marks[0][0] * 0.8),
                      max(ymax, time_marks[-1][0] * 1.2))

    ax_s.set_xlabel(f"OOD solve rate   (fraction of all datasets with R^2>={r2_thr};  "
                    f"error bars = +/-1 std across seeds)", fontsize=FS_LABEL)
    ax_s.set_ylabel("Avg time per formula (s)  [log scale, error bars = +/-1 std across datasets]",
                    fontsize=FS_LABEL)
    ax_s.set_xlim(-0.05, 1.05)
    ax_s.set_title("OOD Solve Rate vs. Speed  [mean over seeds * x-bars = +/-1 seed std]",
                   fontsize=FS_TITLE)
    ax_s.tick_params(axis="both", labelsize=FS_TICK)
    ax_s.grid(linestyle="--", alpha=0.4)
    ax_s.spines[["top", "right"]].set_visible(False)

    legend_patches = [Patch(facecolor=c, label=lbl) for lbl, c in tf_color_map.items()]
    legend_patches.append(Patch(facecolor=_BASELINE_COLOR, label="SRBench baselines"))
    ncol = min(4, len(legend_patches))
    fig.legend(handles=legend_patches, loc="lower center", ncol=ncol, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_fig(fig, out_path)
    return fig


def plot_per_equation(
    tf_agg:       "pd.DataFrame | None",
    srbench_agg:  pd.DataFrame,
    common_datasets: list,
    r2_thr:       float,
    out_dir:      str,
    noise_suffix: str = "",
):
    """Scatter: transformer R^2 vs best-of-srbench R^2, one point per equation.

    The figure is in-distribution only (gap-independent), so it is tagged by noise
    alone.  Without that tag every noise level wrote the same file and the last run
    won -- and concurrent per-noise runs raced on it.
    """
    if tf_agg is None:
        return
    import matplotlib.pyplot as plt

    best_srb = (
        srbench_agg[srbench_agg["dataset"].isin(common_datasets)]
        .groupby("dataset")["r2_mean"].max()
        .reset_index()
        .rename(columns={"r2_mean": "srb_best"})
    )
    tf_f = (
        tf_agg[tf_agg["dataset"].isin(common_datasets)][["dataset", "r2_mean"]]
        .rename(columns={"r2_mean": "tf_r2"})
    )
    merged = tf_f.merge(best_srb, on="dataset")

    fig, ax = plt.subplots(figsize=(6, 6))
    delta = 0.02
    below = merged["tf_r2"] < merged["srb_best"] - delta
    above = merged["tf_r2"] > merged["srb_best"] + delta
    equal = ~below & ~above

    ax.scatter(merged.loc[below, "srb_best"], merged.loc[below, "tf_r2"],
               color=_BASELINE_COLOR, alpha=0.7, s=30, label="SRBench better (>+0.02)")
    ax.scatter(merged.loc[above, "srb_best"], merged.loc[above, "tf_r2"],
               color=_TRANSFORMER_PALETTE[0], alpha=0.7, s=30, label="Transformer better (>+0.02)")
    ax.scatter(merged.loc[equal, "srb_best"], merged.loc[equal, "tf_r2"],
               color="#2d6a4f", alpha=0.5, s=20, label=f"Similar (+/-{delta})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.axvline(r2_thr, color="grey", lw=0.6, ls=":")
    ax.axhline(r2_thr, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("Best SRBench R^2 (max over all algorithms)", fontsize=10)
    ax.set_ylabel("Transformer R^2 (mean over seeds)", fontsize=10)
    ax.set_title("Per-equation R^2 comparison -- Feynman benchmark", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    path = os.path.join(out_dir, f"srbench_per_equation{noise_suffix}.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    save_fig(fig, path)


# -- OOD R^2 computation (merged from the former compare_ood.py) ------------------
# Re-evaluate each method's predicted formula on an out-of-distribution dataset
# (datasets/feynman_ood_g*.pkl.gz) and return per-(algorithm, dataset) OOD R^2.
# This replaces the old compare_ood.py -> results/ood_multi.csv -> compare_srbench.py
# hand-off: the OOD numbers are now produced in-process by compute_ood_r2().
#
# Style string encodes each tool's variable-naming convention:
#   x_0 -> "x_{i}" (AFP, AFP_FE, EPLEX, FEAT, e2e) * x0 -> "x{i}" (AIFeynman)
#   x1 -> "x{i+1}" (DSR) * X0 -> "X{i}" (FFX) * X1 -> "X{i+1}" (Operon)
#   itea -> x as full matrix (ITEA)
_OOD_ALGO_STYLE = {
    "AFP": "x_0", "AFP_FE": "x_0", "AIFeynman": "x0", "DSR": "x1",
    "EPLEX": "x_0", "FFX": "X0", "ITEA": "itea",
    "Operon": "X1", "e2e": "x_0",
    # Newly added: parsed from symbolic_model strings.
    # MRGP, FEAT, SBP-GP, GP-GOMEA omitted -- formula strings cannot be independently
    # evaluated to reproduce model predictions:
    #   MRGP: requires a separate linear-regression step over training data.
    #   FEAT: formula is in internally standardized variable space.
    #   SBP-GP / GP-GOMEA: likely run on standardized data in SRBench; per-dataset
    #       scaler parameters are not stored in the feather, so raw-input evaluation
    #       gives systematically wrong predictions (~47-69% vs SRBench's ~97%).
    "BSR": "bsr",          # bracket-wrapped constants, x[i] vars, ^ power
    "gplearn": "gplearn",  # functional prefix (add/sub/mul/div), X0-indexed
}
_OOD_BASE_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
    "sqrt": np.sqrt, "arctan": np.arctan, "arcsin": np.arcsin, "arccos": np.arccos,
    "tanh": np.tanh, "abs": np.abs,
    "pi": np.pi, "e": np.e, "max": np.maximum, "min": np.minimum,
    "log10": np.log10,  # FFX uses log10
    "sqrtAbs": lambda x: np.sqrt(np.abs(x)),  # ITEA
    # gplearn functional operators
    "add": np.add, "sub": np.subtract, "mul": np.multiply,
    # protected helpers for BSR/SBP-GP/GP-GOMEA
    "_plog": lambda x: np.log(np.abs(x) + 1e-10),
    "_safe_sqrt": lambda x: np.sqrt(np.abs(x)),
}
_OOD_EVAL_GLOBALS = {"__builtins__": {}}
_OOD_N_VARS = len(V)


def _ood_protected_div(a, b):
    return np.where(np.abs(b) < 1e-10, np.sign(a) * 1e10, a / b)


def _ood_analytic_quotient(a, b):
    """SBP-GP analytic quotient: a / sqrt(1 + b^2). Always finite."""
    return a / np.sqrt(1.0 + np.asarray(b, np.float64) ** 2)


# Formula preprocessors for the newly supported algorithms.
def _preprocess_bsr(formula: str) -> str:
    import re
    s = str(formula).strip()
    s = re.sub(r'\bx\[(\d+)\]', lambda m: f"x{m.group(1)}", s)
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("^", "**")
    return s


def _convert_infix_op(s: str, op: str, fn: str) -> str:
    """Replace infix 'A op B' with 'fn(A, B)' by tracing balanced parens.

    Scans left-to-right and updates the scan position after each replacement
    so the replacement text itself is never re-processed.
    """
    op_len = len(op)
    scan_from = 0
    while True:
        idx = s.find(op, scan_from)
        if idx == -1:
            break

        # Skip if this occurrence was produced by a previous replacement
        # (e.g., 'aq' inside '_aq(...)' is preceded by '_').
        before = s[idx - 1] if idx > 0 else '\x00'
        if before == '_':
            scan_from = idx + op_len
            continue

        # --- find left operand: ends at idx - 1 ---
        end_l = idx - 1
        if end_l >= 0 and s[end_l] == ')':
            depth, i = 1, end_l - 1
            while i >= 0 and depth > 0:
                if s[i] == ')': depth += 1
                elif s[i] == '(': depth -= 1
                i -= 1
            start_l = i + 1  # position of matching '('
            # Include any identifier immediately before '(' (e.g. function name).
            j = start_l - 1
            while j >= 0 and (s[j].isalnum() or s[j] in '._'):
                j -= 1
            start_l = j + 1
        else:
            i = end_l
            while i >= 0 and (s[i].isalnum() or s[i] in '._'):
                i -= 1
            start_l = i + 1
        # Include a leading unary minus if the token is a negative numeric literal
        # (e.g. '-19.83' -- only when preceded by '(' or ',', not binary subtraction).
        if (start_l > 0 and s[start_l - 1] == '-' and
                (start_l < 2 or s[start_l - 2] in '(,')):
            start_l -= 1

        # --- find right operand: starts at idx + op_len ---
        start_r = idx + op_len
        if start_r < len(s) and s[start_r] == '(':
            depth, i = 1, start_r + 1
            while i < len(s) and depth > 0:
                if s[i] == '(': depth += 1
                elif s[i] == ')': depth -= 1
                i += 1
            end_r = i
        else:
            i = start_r
            # Handle leading unary minus for negative number literals.
            if i < len(s) and s[i] == '-':
                i += 1
            while i < len(s) and (s[i].isalnum() or s[i] in '._'):
                i += 1
            # If identifier is immediately followed by '(', include the function call.
            while i < len(s) and s[i] == '(':
                depth, i = 1, i + 1
                while i < len(s) and depth > 0:
                    if s[i] == '(': depth += 1
                    elif s[i] == ')': depth -= 1
                    i += 1
            end_r = i

        left        = s[start_l:end_l + 1]
        right       = s[start_r:end_r]
        replacement = f"{fn}({left},{right})"
        s           = s[:start_l] + replacement + s[end_r:]
        # Advance past the replacement so it is never re-scanned.
        scan_from   = start_l + len(replacement)
    return s


def _convert_infix_op_all(s: str, op: str, fn: str) -> str:
    """Apply _convert_infix_op in passes until no more occurrences remain."""
    for _ in range(50):  # max iterations guard
        s_prev = s
        s = _convert_infix_op(s, op, fn)
        if s == s_prev:
            break
    return s


def _preprocess_sbp_gp(formula: str) -> str:
    s = str(formula).strip()
    s = s.replace("plog", "_plog")
    # Convert infix 'aq' -> _aq(A, B)  [analytic quotient: A/sqrt(1+B^2)]
    # Multi-pass needed for nested operators in right-side operands.
    s = _convert_infix_op_all(s, "aq", "_aq")
    s = s.replace("^", "**")
    return s


def _preprocess_gp_gomea(formula: str) -> str:
    s = str(formula).strip()
    s = s.replace("plog", "_plog")
    # Convert infix 'p/' -> _pdiv(A, B)  [protected division]
    s = _convert_infix_op_all(s, "p/", "_pdiv")
    s = s.replace("^", "**")
    return s


def _preprocess_caret_pipe(formula: str) -> str:
    """AFP / AFP_FE / EPLEX / FEAT: ^ exponentiation, |...| absolute value."""
    s = str(formula).strip()
    s = s.replace("|", "")   # strip abs-value pipes (SRBench convention)
    s = s.replace("^", "**")
    return s


def _preprocess_caret(formula: str) -> str:
    """Operon / FFX: ^ exponentiation only (no pipe abs-value)."""
    s = str(formula).strip()
    s = s.replace("^", "**")
    return s


_OOD_FORMULA_PREPROCESSORS = {
    "bsr":      _preprocess_bsr,
    "sbp_gp":   _preprocess_sbp_gp,
    "gp_gomea": _preprocess_gp_gomea,
    # AFP / AFP_FE / EPLEX / FEAT use x_0 style with ^ and |...|
    "x_0":      _preprocess_caret_pipe,
    # Operon uses X1 style with ^ only
    "X1":       _preprocess_caret,
    # FFX uses X0 style with ^ only
    "X0":       _preprocess_caret,
}


# Relative tolerance for the constant-target guard below: a target counts as constant
# only when its spread is this small COMPARED WITH ITS OWN MAGNITUDE (1e-8 of the RMS is
# double-precision noise), never at some fixed absolute size.
_OOD_R2_RTOL = 1e-8


def _ood_r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if not np.isfinite(ss_res):
        return 0.0
    # Constant-target guard.  This used to be np.isclose(ss_tot, 0.0), whose default
    # atol=1e-8 is ABSOLUTE: on the OOD gap series the inputs move far from the origin,
    # so any decaying Feynman formula has targets around 1e-9, and both sums of squares
    # slip under that tolerance.  The branch then returned 1.0 for predictions that were
    # wrong by orders of magnitude and even anti-correlated with the truth -- 12 of 98
    # datasets at gap 128, and the artefact grew with the gap, tilting every OOD curve
    # upward.  Scaling the tolerance by the target's own magnitude keeps the intent (a
    # genuinely constant target cannot be scored by R^2) without the scale trap.
    tol = (_OOD_R2_RTOL ** 2) * y_true.size * float(np.mean(y_true ** 2))
    if ss_tot <= tol:
        return 1.0 if ss_res <= tol else 0.0
    return float(max(0.0, 1.0 - min(ss_res / ss_tot, 1.0)))


def _ood_make_ns(style: str, X: np.ndarray) -> dict:
    ns = dict(_OOD_BASE_FUNCS)
    ns["div"]   = _ood_protected_div
    ns["_pdiv"] = _ood_protected_div
    ns["_aq"]   = _ood_analytic_quotient
    n = X.shape[1]
    X64 = X.astype(np.float64)
    if style == "x_0":
        # AFP/AFP_FE/EPLEX/FEAT use |...| for abs-value protection; after stripping
        # the pipes we need protected log/sqrt so OOD inputs don't produce NaN.
        ns["log"]  = lambda x: np.log(np.abs(x) + 1e-10)
        ns["sqrt"] = lambda x: np.sqrt(np.abs(x))
        for i in range(n): ns[f"x_{i}"] = X64[:, i]
    elif style in ("x0", "bsr", "sbp_gp", "gp_gomea"):
        for i in range(n): ns[f"x{i}"] = X64[:, i]
    elif style == "x1":
        for i in range(n): ns[f"x{i+1}"] = X64[:, i]
    elif style in ("X0", "gplearn"):
        for i in range(n): ns[f"X{i}"] = X64[:, i]
    elif style == "X1":
        for i in range(n): ns[f"X{i+1}"] = X64[:, i]
    elif style == "v1":
        for i in range(n): ns[f"v{i+1}"] = X64[:, i]
    elif style == "itea":
        ns["x"] = X64
    return ns


def _ood_eval_infix(formula: str, style: str, X: np.ndarray, y: np.ndarray) -> float:
    if not formula or not str(formula).strip():
        return 0.0
    try:
        # Apply method-specific formula preprocessing before eval.
        if style in _OOD_FORMULA_PREPROCESSORS:
            formula = _OOD_FORMULA_PREPROCESSORS[style](str(formula))
        ns = _ood_make_ns(style, X)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            preds = eval(formula, _OOD_EVAL_GLOBALS, ns)  # noqa: S307
        preds = np.asarray(preds, np.float64)
        if not np.isfinite(preds).all():
            return 0.0
        return _ood_r2_score(y, preds)
    except Exception:
        return 0.0


def _ood_eval_pn(formula: str, X: np.ndarray, y: np.ndarray, pn_eval) -> float:
    if not formula or not str(formula).strip():
        return 0.0
    try:
        X64 = X.astype(np.float64)
        if X64.shape[1] < _OOD_N_VARS:
            pad = np.zeros((X64.shape[0], _OOD_N_VARS - X64.shape[1]), np.float64)
            X64 = np.concatenate([X64, pad], axis=1)
        preds, stacklefts = pn_eval(formula, X64)
        if np.any(stacklefts != 0) or not np.isfinite(preds).all():
            return 0.0
        return _ood_r2_score(y, preds)
    except Exception:
        return 0.0


def _ood_load_feather_best(path: str, noise: float = 0.0) -> dict:
    """{algo: {dataset: best_row}} for OOD-evaluable SRBench algorithms.

    Algorithms in _OOD_ALGO_STYLE are included; MRGP is excluded because its
    symbolic_model requires a separate linear-regression step and cannot be
    independently re-evaluated.

    `noise` selects the target_noise level of the SRBench rows to re-evaluate, so
    the baselines' OOD is measured on formulas fit at the SAME noise as our models.
    """
    df = pd.read_feather(path)
    f = df[(df["data_group"] == "Feynman") & (df["target_noise"] == noise)].copy()
    f = f[f["algorithm"].isin(_OOD_ALGO_STYLE) & f["symbolic_model"].notna()]
    best = f.loc[f.groupby(["algorithm", "dataset"])["r2_test"].idxmax()]
    out: dict = {}
    for _, row in best.iterrows():
        out.setdefault(row["algorithm"], {})[row["dataset"]] = row.to_dict()
    return out


def _ood_load_feather_all(path: str, noise: float = 0.0) -> list:
    """All (algo, dataset, seed, formula) tuples for OOD-evaluable SRBench algorithms.

    Returns a list of dicts with keys: algorithm, dataset, seed, symbolic_model, r2_test.
    Used for multi-seed OOD evaluation and variance estimation.  `noise` selects the
    target_noise level of the SRBench rows (default 0.0 = noise-free).
    """
    df = pd.read_feather(path)
    f = df[(df["data_group"] == "Feynman") & (df["target_noise"] == noise)].copy()
    f = f[f["algorithm"].isin(_OOD_ALGO_STYLE) & f["symbolic_model"].notna()]
    return f[["algorithm", "dataset", "random_state", "symbolic_model", "r2_test"]].rename(
        columns={"random_state": "seed"}
    ).to_dict("records")


def _ood_load_e2e(path: str) -> dict:
    """{dataset: best-R^2 row} from an eval_e2e artifact.

    The merged E2E group artifact (results/e2e/noise_<tau>/results_feynman_e2e.pkl.gz)
    holds one row per (dataset, seed), so keep the highest-r2 row per dataset -- e2e
    contributes a single formula per dataset in the OOD step (seed=-1). A legacy .csv
    (one row per dataset) still loads via the csv fallback."""
    if path.endswith(".pkl.gz") or path.endswith(".pkl"):
        with gzip.open(path, "rb") as fh:
            rows = pickle.load(fh).get("results", [])
        best: dict = {}
        for r in rows:
            ds = r.get("dataset")
            if ds is None:
                continue
            try:
                r2 = float(r.get("r2"))
            except (TypeError, ValueError):
                r2 = float("-inf")
            if ds not in best or r2 > best[ds][0]:
                best[ds] = (r2, r)
        return {ds: r for ds, (_r2, r) in best.items()}
    with open(path, newline="") as fh:
        return {r["dataset"]: r for r in csv.DictReader(fh)}


def _ood_load_e2e_all(path: str) -> dict:
    """{dataset: [(seed, formula), ...]} -- all seeds from an eval_e2e artifact.

    Mirrors _ood_load_tf_all so e2e contributes one OOD row per (dataset, seed) in
    all_seeds mode, giving it the same 10-seed spread as the transformer models.
    A legacy .csv (one formula per dataset) yields a single pseudo-seed (-1)."""
    if path.endswith(".pkl.gz") or path.endswith(".pkl"):
        with gzip.open(path, "rb") as fh:
            rows = pickle.load(fh).get("results", [])
        by_ds: dict = {}
        for r in rows:
            ds = r.get("dataset")
            if ds is None:
                continue
            by_ds.setdefault(ds, []).append((r.get("seed", -1), r.get("predicted_formula")))
        for v in by_ds.values():
            v.sort(key=lambda t: (t[0] is None, t[0]))
        return by_ds
    with open(path, newline="") as fh:
        return {r["dataset"]: [(-1, r.get("predicted_formula"))] for r in csv.DictReader(fh)}


def _ood_load_tf_rows(path: str) -> dict:
    """{dataset: row} from an eval_mymodels pkl (last row per dataset, matching the
    former compare_ood.py behaviour)."""
    with gzip.open(path, "rb") as fh:
        data = pickle.load(fh)
    return {r["dataset"]: r for r in data["results"]}


def _ood_load_tf_all(path: str) -> dict:
    """{dataset: [(seed, formula), ...]} -- all seeds from an eval_mymodels pkl.

    Multi-seed pkls return one entry per seed so callers can compute variance.
    """
    with gzip.open(path, "rb") as fh:
        data = pickle.load(fh)
    by_ds: dict = {}
    for r in data["results"]:
        by_ds.setdefault(r["dataset"], []).append(
            (r["seed"], r["predicted_formula"])
        )
    for rows in by_ds.values():
        rows.sort(key=lambda t: t[0])
    return by_ds


def _ood_load_tf_best(path: str) -> dict:
    """{dataset: (seed, formula)} -- best-training-R^2 seed per dataset.

    Mirrors _ood_load_feather_best for SRBench methods: one formula per dataset
    so that n_total = number of datasets (not dataset * seeds).
    """
    with gzip.open(path, "rb") as fh:
        data = pickle.load(fh)
    best: dict = {}
    for r in data["results"]:
        ds = r["dataset"]
        r2 = r.get("r2", float("-inf"))
        if ds not in best or r2 > best[ds][0]:
            best[ds] = (r2, r["seed"], r["predicted_formula"])
    return {ds: (seed, formula) for ds, (_, seed, formula) in best.items()}


def _ood_load_tpsr(path: str, noise: float = 0.0) -> dict:
    """{dataset: row_dict} from a TPSR Feynman CSV at the given target_noise.

    The *_allnoise.csv variants carry all of {0, 0.001, 0.01, 0.1}; single-noise
    CSVs carry only 0.0.  A noise level absent from the file yields no rows (TPSR
    is then simply absent from the OOD plot at that noise).
    """
    df = pd.read_csv(path)
    df = df[df["target_noise"] == noise]
    return {row["problem"]: row.to_dict() for _, row in df.iterrows()}


# TPSR uses its own prefix notation: binary ops are 'add','mul','sub';
# unary ops are 'inv','exp','sin','cos','tan','sqrt','abs','arctan','pow2','pow3'.
_TPSR_BINARY_OPS = {"add": np.add, "mul": np.multiply, "sub": np.subtract}
_TPSR_UNARY_OPS = {
    "abs": np.abs, "arctan": np.arctan, "cos": np.cos, "exp": np.exp,
    "inv": lambda x: 1.0 / x, "pow2": lambda x: x ** 2, "pow3": lambda x: x ** 3,
    "sin": np.sin, "sqrt": np.sqrt, "tan": np.tan,
}


def _tpsr_prefix_eval(tokens: list, pos: int, X: np.ndarray):
    tok = tokens[pos]
    if tok in _TPSR_BINARY_OPS:
        a, pos = _tpsr_prefix_eval(tokens, pos + 1, X)
        b, pos = _tpsr_prefix_eval(tokens, pos, X)
        return _TPSR_BINARY_OPS[tok](a, b), pos
    if tok in _TPSR_UNARY_OPS:
        a, pos = _tpsr_prefix_eval(tokens, pos + 1, X)
        return _TPSR_UNARY_OPS[tok](a), pos
    if tok.startswith("x_"):
        return X[:, int(tok[2:])].astype(np.float64), pos + 1
    return np.full(len(X), float(tok), dtype=np.float64), pos + 1


def _ood_eval_tpsr_prefix(formula_prefix: str, X: np.ndarray, y: np.ndarray) -> float:
    if not formula_prefix or not str(formula_prefix).strip():
        return 0.0
    tokens = [t.strip() for t in formula_prefix.split(",")]
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            preds, _ = _tpsr_prefix_eval(tokens, 0, X)
        preds = np.asarray(preds, np.float64)
        if not np.isfinite(preds).all():
            return 0.0
        return _ood_r2_score(y, preds)
    except Exception:
        return 0.0


def _ood_load_gt_pn(feynman_csv: str = FEYNMAN_CSV) -> dict:
    """{dataset_name: pn_str} from FeynmanEquations.csv.

    Uses the same formula_to_pn pipeline as gen_ood_data.py so that the GT
    prefix strings are bit-for-bit identical to the ones used to generate the
    OOD y values -- guaranteeing R^2=1.0 rather than the near-1.0 values that
    result from the rounded constants in the feather's true_model column.
    """
    from feynman_pn_analysis import formula_to_pn
    out = {}
    with open(feynman_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fname = row.get("Filename", "").strip()
            fml   = row.get("Formula",  "").strip()
            if not fname or not fml:
                continue
            var_names = [row.get(f"v{k}_name", "").strip()
                         for k in range(1, 11)
                         if row.get(f"v{k}_name", "").strip()]
            if not var_names:
                continue
            pn, status = formula_to_pn(fml, var_names)
            if pn and "ERR" not in pn:
                out["feynman_" + fname.replace(".", "_")] = pn
    return out


def _ood_pn_evaluator():
    """Lazy-import the C PN evaluator; return None (with a warning) if the
    extension is unavailable so the rest of the plot still renders."""
    import sys
    sys.path.insert(0, SCRIPT_DIR)
    try:
        from funcWrappers import wrapExtEvalPN
        return wrapExtEvalPN
    except Exception as exc:
        print(f"[ood] PN evaluator unavailable ({exc.__class__.__name__}: {exc}); "
              f"transformer OOD R^2 will be skipped.")
        return None


def compute_ood_r2(ood_data_path: str, tf_files: list,
                   feather_path: str = SRBENCH_GT,
                   e2e_path: "str | None" = None,
                   tpsr_path: "str | None" = None,
                   phye2e_path: "str | None" = None,
                   pysr_path: "str | None" = None,
                   include_gt: bool = True,
                   all_seeds: bool = False,
                   noise: float = 0.0) -> "tuple[pd.DataFrame, object]":
    """Compute per-(algorithm, dataset) OOD R^2 on an OOD dataset pkl.

    tf_files: list of (label, path) for transformer eval_mymodels pkls. The label is
    the model_name used in the comparison table, so OOD rows join to it by the
    same key (no eval_mymodels<->model-name mismatch).

    all_seeds: when True, evaluate ALL seed formulas from the feather (not just best)
    and include a 'seed' column in the returned DataFrame. Our models always
    contribute one row per dataset (seed=-1) since they have a single formula.

    Returns (DataFrame[algorithm, dataset, ood_r2[, seed]], gap).
    """
    # At noise>0 the e2e baseline must be its noise-matched run, the merged E2E
    # group artifact at results/e2e/noise_<tau>/results_feynman_e2e.pkl.gz; if that
    # file was not generated it simply does not exist and e2e is skipped below
    # (os.path.exists guard), so a noisy plot never silently reuses the noise-0 curve.
    if e2e_path is None:
        e2e_path = _resolve_artifact(
            results_io.e2e_srbench_pkl(os.path.join(SCRIPT_DIR, "results"), noise))

    with gzip.open(ood_data_path, "rb") as fh:
        ood_file = pickle.load(fh)
    ood_datasets = ood_file["datasets"]
    gap = ood_file.get("gap")

    # collect (algo, dataset, formula, style, is_pn, seed) entries from every source
    entries: list = []
    if all_seeds:
        for row in _ood_load_feather_all(feather_path, noise=noise):
            entries.append((row["algorithm"], row["dataset"], row["symbolic_model"],
                            _OOD_ALGO_STYLE[row["algorithm"]], False, int(row["seed"])))
    else:
        for alg, ds_map in _ood_load_feather_best(feather_path, noise=noise).items():
            for ds, row in ds_map.items():
                entries.append((alg, ds, row["symbolic_model"], _OOD_ALGO_STYLE[alg], False, -1))
    for label, path in tf_files:
        if all_seeds:
            for ds, seed_fmls in _ood_load_tf_all(path).items():
                for seed, formula in seed_fmls:
                    entries.append((label, ds, formula, "", True, seed))
        else:
            for ds, (seed, formula) in _ood_load_tf_best(path).items():
                entries.append((label, ds, formula, "", True, -1))
    if os.path.exists(e2e_path):
        if all_seeds:
            for ds, seed_fmls in _ood_load_e2e_all(e2e_path).items():
                for seed, formula in seed_fmls:
                    entries.append(("e2e", ds, formula, "x_0", False, seed))
        else:
            for ds, r in _ood_load_e2e(e2e_path).items():
                entries.append(("e2e", ds, r["predicted_formula"], "x_0", False, -1))
    if tpsr_path and os.path.exists(tpsr_path):
        for ds, row in _ood_load_tpsr(tpsr_path, noise=noise).items():
            entries.append(("TPSR", ds, row["predicted_tree_prefix"], "tpsr_pn", False, -1))
    # PhyE2E: eval_phye2e.py writes the SAME row schema as eval_e2e.py (dataset, seed,
    # predicted_formula) and the same infix x_0..x_N formula convention, so it reuses
    # the e2e loader and the "x_0" evaluator style verbatim -- nothing new to teach.
    if phye2e_path and os.path.exists(phye2e_path):
        _phye2e_label = phye2e_addon.DISPLAY_LABEL
        if all_seeds:
            for ds, seed_fmls in _ood_load_e2e_all(phye2e_path).items():
                for seed, formula in seed_fmls:
                    entries.append((_phye2e_label, ds, formula, "x_0", False, seed))
        else:
            for ds, r in _ood_load_e2e(phye2e_path).items():
                entries.append((_phye2e_label, ds, r["predicted_formula"], "x_0", False, -1))
    # PySR: same story as PhyE2E one block up.  eval_pysr.py writes eval_e2e.py's row
    # schema (dataset, seed, predicted_formula) and the same infix x_0..x_N formulas, so
    # it reuses that loader and the "x_0" evaluator style verbatim.  Note this curve is
    # OURS-RUN on both benchmarks: PySR is absent from the SRBench feather entirely, so
    # there is no published row for it to be confused with.  See pysr_addon.py.
    if pysr_path and os.path.exists(pysr_path):
        _pysr_label = pysr_addon.DISPLAY_LABEL
        if all_seeds:
            for ds, seed_fmls in _ood_load_e2e_all(pysr_path).items():
                for seed, formula in seed_fmls:
                    entries.append((_pysr_label, ds, formula, "x_0", False, seed))
        else:
            for ds, r in _ood_load_e2e(pysr_path).items():
                entries.append((_pysr_label, ds, r["predicted_formula"], "x_0", False, -1))
    if include_gt and os.path.exists(FEYNMAN_CSV):
        for ds, pn in _ood_load_gt_pn().items():
            entries.append(("GT", ds, pn, "", True, -1))

    pn_eval = _ood_pn_evaluator() if any(is_pn for *_, is_pn, _s in entries) else None

    records = []
    for algo, ds, formula, style, is_pn, seed in entries:
        if ds not in ood_datasets:
            continue
        X, y = ood_datasets[ds]["X"], ood_datasets[ds]["y"]
        if style == "tpsr_pn":
            r2 = _ood_eval_tpsr_prefix(formula, X, y)
        elif is_pn:
            if pn_eval is None:
                continue
            r2 = _ood_eval_pn(formula, X, y, pn_eval)
        else:
            r2 = _ood_eval_infix(formula, style, X, y)
        rec = {"algorithm": algo, "dataset": ds, "ood_r2": r2}
        if all_seeds:
            rec["seed"] = seed
        records.append(rec)

    cols = ["algorithm", "dataset", "ood_r2"] + (["seed"] if all_seeds else [])
    return pd.DataFrame(records, columns=cols), gap


# -- PMLB test-split evaluation (gap = 0) --------------------------------------

PMLB_DIR = os.path.join(SCRIPT_DIR, "datasets", "pmlb", "datasets")
# SRBench uses train_size=0.75 / test_size=0.25 (see srbench/experiment/evaluate_model.py).
_PMLB_TEST_SIZE = 0.25
# Fixed seed for our models (matches SEEDS[0] in srbench/experiment/seeds.py).
_PMLB_FIXED_SEED = 23654
# First 10 seeds from srbench/experiment/seeds.py -- used for multi-seed evaluation.
_PMLB_SEEDS = [23654, 15795, 860, 5390, 16850, 29910, 4426, 21962, 14423, 28020]


def _pmlb_test_split(dataset: str, seed: int,
                     pmlb_dir: str = PMLB_DIR) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """Load PMLB dataset and return (X_test, y_test) for the 25% test split at `seed`."""
    path = os.path.join(pmlb_dir, dataset, f"{dataset}.tsv.gz")
    if not os.path.exists(path):
        return None, None
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(path, sep="\t")
    y = df["target"].values.astype(np.float64)
    X = df.drop(columns=["target"]).values.astype(np.float64)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=_PMLB_TEST_SIZE, random_state=seed)
    return X_test, y_test


def compute_pmlb_r2(
    tf_files: list,
    feather_path: str = SRBENCH_GT,
    pmlb_dir: str = PMLB_DIR,
    e2e_path: "str | None" = None,
    tpsr_path: "str | None" = None,
    include_gt: bool = True,
    seeds: "list | None" = None,
) -> pd.DataFrame:
    """Compute per-(algorithm, dataset, seed) R^2 using the PMLB 25% test split.

    SRBench algorithms: symbolic_model for every feather row (all seeds) is re-evaluated
    on the PMLB 25% test split matching that row's random_state, using the same
    _ood_eval_infix path as gap>0 evaluation for consistent comparisons.

    Our models (eval_mymodels, m89float, e2e): each formula is evaluated on 10 PMLB test splits
    (one per seed in `seeds`, default _PMLB_SEEDS) so variance is comparable to SRBench.

    Returns DataFrame[algorithm, dataset, seed, ood_r2].
    """
    if e2e_path is None:
        e2e_path = _resolve_artifact(
            results_io.e2e_srbench_pkl(os.path.join(SCRIPT_DIR, "results"), 0.0))
    if seeds is None:
        seeds = _PMLB_SEEDS

    records = []

    pn_eval = _ood_pn_evaluator()
    _split_cache: dict = {}

    def _get_split(ds, seed):
        key = (ds, seed)
        if key not in _split_cache:
            _split_cache[key] = _pmlb_test_split(ds, seed, pmlb_dir)
        return _split_cache[key]

    # -- SRBench algorithms: re-evaluate symbolic_model on matching PMLB split ----
    df = pd.read_feather(feather_path)
    f = df[(df["data_group"] == "Feynman") & (df["target_noise"] == 0.0)].copy()
    f = f[f["algorithm"].isin(_OOD_ALGO_STYLE) & f["symbolic_model"].notna()]
    for _, row in f.iterrows():
        alg  = row["algorithm"]
        ds   = row["dataset"]
        seed = int(row["random_state"])
        X, y = _get_split(ds, seed)
        if X is None:
            continue
        r2 = _ood_eval_infix(row["symbolic_model"], _OOD_ALGO_STYLE[alg], X, y)
        records.append({
            "algorithm": alg,
            "dataset":   ds,
            "seed":      seed,
            "ood_r2":    r2,
        })

    # -- Our models: evaluate each formula on 10 different PMLB test splits -------
    # eval_mymodels models (PN formulas)
    for label, path in tf_files:
        with gzip.open(path, "rb") as _fh:
            _pkl = pickle.load(_fh)
        # Group all rows by dataset, sort by seed so index -> PMLB seed mapping is stable.
        _by_ds: dict = {}
        for r in _pkl.get("results", []):
            _by_ds.setdefault(r["dataset"], []).append(r)
        for _rows in _by_ds.values():
            _rows.sort(key=lambda r: r["seed"])

        for ds, _rows in _by_ds.items():
            if len(_rows) == 1:
                # Single-seed pkl: evaluate the one formula on every PMLB split (old behaviour).
                r = _rows[0]
                for seed in seeds:
                    X, y = _get_split(ds, seed)
                    if X is None or pn_eval is None:
                        continue
                    r2 = _ood_eval_pn(r["predicted_formula"], X, y, pn_eval)
                    records.append({"algorithm": label, "dataset": ds, "seed": seed, "ood_r2": r2})
            else:
                # Multi-seed pkl: pair formula_i with seeds[i] PMLB test split -- mirrors
                # SRBench's per-run train/test split exactly (one split per algorithm run).
                for i, r in enumerate(_rows[:len(seeds)]):
                    pmlb_seed = seeds[i % len(seeds)]
                    X, y = _get_split(ds, pmlb_seed)
                    if X is None or pn_eval is None:
                        continue
                    r2 = _ood_eval_pn(r["predicted_formula"], X, y, pn_eval)
                    records.append({"algorithm": label, "dataset": ds, "seed": pmlb_seed, "ood_r2": r2})

    # e2e (infix, x_0 style)
    if os.path.exists(e2e_path):
        for ds, r in _ood_load_e2e(e2e_path).items():
            for seed in seeds:
                X, y = _get_split(ds, seed)
                if X is None:
                    continue
                r2 = _ood_eval_infix(r["predicted_formula"], "x_0", X, y)
                records.append({"algorithm": "e2e", "dataset": ds, "seed": seed, "ood_r2": r2})

    # TPSR (prefix notation)
    if tpsr_path and os.path.exists(tpsr_path):
        for ds, row in _ood_load_tpsr(tpsr_path).items():
            for seed in seeds:
                X, y = _get_split(ds, seed)
                if X is None:
                    continue
                r2 = _ood_eval_tpsr_prefix(row["predicted_tree_prefix"], X, y)
                records.append({"algorithm": "TPSR", "dataset": ds, "seed": seed, "ood_r2": r2})

    # GT (PN formulas from FeynmanEquations.csv)
    if include_gt and os.path.exists(FEYNMAN_CSV):
        gt_pns = _ood_load_gt_pn()
        for ds, pn in gt_pns.items():
            for seed in seeds:
                X, y = _get_split(ds, seed)
                if X is None or pn_eval is None:
                    continue
                r2 = _ood_eval_pn(pn, X, y, pn_eval)
                records.append({"algorithm": "GT", "dataset": ds, "seed": seed, "ood_r2": r2})

    return pd.DataFrame(records, columns=["algorithm", "dataset", "seed", "ood_r2"])


# -- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare the E2E transformer against SRBench on Feynman benchmark."
    )
    parser.add_argument(
        "--results", default=None, nargs="+",
        help=(
            "One or more transformer results files (.pkl.gz from eval_mymodels.py). "
            "If omitted, all eval_tf*.pkl.gz files in results/ are used automatically "
            "(eval_tf_ is the on-disk artifact prefix, unchanged by the script rename)."
        ),
    )
    parser.add_argument(
        "--exclude", default=[], nargs="*", metavar="MODEL",
        help="Model names to exclude from plots and table. Default: none.",
    )
    parser.add_argument(
        "--include-phye2e", action="store_true",
        help="Also load PhyE2E from results/phye2e/noise_<tau>/"
             "results_feynman_phye2e.pkl.gz (eval_phye2e.py) into the table and both "
             "plots. Off by default. Their 'e2e' ablation with beam search, NOT the "
             "full D&C+MCTS pipeline -- hence the '(E2E)' label. --phye2e-group picks "
             "a different tree (e.g. phye2e_units).")
    parser.add_argument(
        "--include-pysr", action="store_true",
        help="Also load PySR from results/pysr/noise_<tau>/results_feynman_pysr.pkl.gz "
             "(eval_pysr.py) into the table and both plots. Off by default. PySR has no "
             "published SRBench results -- it is absent from ground-truth_results.feather "
             "-- so this curve is entirely our own run. --pysr-group picks a different "
             "tree (e.g. pysr_nostop).")
    parser.add_argument(
        "--e2e-results", default=None,
        help="eval_e2e.py results artifact, loaded as the 'e2e' baseline so it "
             "appears in the table and both plots. Default: the merged E2E group "
             "artifact results/e2e/noise_<tau>/results_feynman_e2e.pkl.gz for the "
             "chosen --noise (skipped if missing or via --exclude e2e). Pass an "
             "explicit path to override; a legacy .csv of the same stem is a fallback.",
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        choices=[0.0, 0.001, 0.01, 0.1],
        help="Noise level to compare against (must match noise used in eval_e2e.py). Default: 0.0",
    )
    parser.add_argument(
        "--r2-threshold", type=float, default=0.99, dest="r2_thr",
        help="R^2 threshold for counting an equation as solved. Default: 0.99",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save the summary bar chart. Defaults to "
             "plots/srbench_comparison_g<gap>.png when --gap is given, "
             "else plots/srbench_comparison.png.",
    )
    parser.add_argument("--no-plot",   action="store_true", help="Skip generating plots.")
    parser.add_argument("--show-plot", action="store_true", help="Show plot interactively.")
    parser.add_argument(
        "--ood-data", default=None,
        help="OOD dataset pkl (feynman_ood_g*.pkl.gz) to evaluate predicted "
             "formulae on. OOD R^2 is computed in-process (no separate "
             "compare_ood.py step). Defaults to datasets/feynman_ood_g<gap>.pkl.gz "
             "when --gap is given, else datasets/feynman_ood_g10.pkl.gz.",
    )
    parser.add_argument(
        "--ood-csv", default="results/ood_multi.csv",
        help="Where to write the computed per-(algorithm,dataset) OOD R^2 table "
             "(artifact). With --reuse-ood-csv it is read back instead of "
             "recomputed. Default: results/ood_multi.csv",
    )
    parser.add_argument(
        "--reuse-ood-csv", action="store_true",
        help="Skip OOD computation and read an existing --ood-csv instead.",
    )
    parser.add_argument(
        "--reuse-ood", action="store_true",
        help="Reuse the raw OOD R^2 table plot_ood_vs_gap.py cached "
             "(results/ood_raw_noise<tau>.csv, all gaps/seeds), filtered to --gap, "
             "instead of recomputing -- turns the ~minutes OOD step into seconds. "
             "Falls back to computing when the gap is not in the cache.",
    )
    parser.add_argument("--no-ood", action="store_true",
                        help="Skip OOD evaluation entirely (left scatter shows no data).")
    parser.add_argument(
        "--tpsr-results", default="TPSR/srbench_results/feynman_tpsr_l0.1_allnoise.csv",
        help="TPSR Feynman results CSV; rows at --noise are used. The *_allnoise "
             "variant carries all of {0,0.001,0.01,0.1} so TPSR is shown "
             "noise-matched. Default: TPSR/srbench_results/feynman_tpsr_l0.1_allnoise.csv",
    )
    parser.add_argument(
        "--gap", default=None,
        help="OOD gap value shown in the plot title (e.g. 5 or 10). "
             "Auto-detected from the OOD dataset if omitted.",
    )
    phye2e_addon.add_group_arg(parser)
    pysr_addon.add_group_arg(parser)
    args = parser.parse_args()
    phye2e_addon.apply_group_arg(args)
    pysr_addon.apply_group_arg(args)

    if args.ood_data is None or args.output is None:
        gap_suffix = args.gap if args.gap is not None else "10"
        # noise>0 -> tag the plot file so it never overwrites the noise-free version.
        noise_suffix = "" if args.noise == 0.0 else f"_noise{args.noise:g}"
        if args.ood_data is None:
            args.ood_data = f"datasets/feynman_ood_g{gap_suffix}.pkl.gz"
        if args.output is None:
            args.output = f"plots/srbench_comparison_g{gap_suffix}{noise_suffix}.png"

    if args.no_plot or not args.show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402 -- must come after matplotlib.use()

    # -- Load data --------------------------------------------------------------
    print(f"\nLoading SRBench ground-truth results (noise={args.noise}) ...")
    srbench_agg = load_srbench_gt(SRBENCH_GT, args.noise, args.r2_thr)
    srbench_datasets = set(srbench_agg["dataset"])
    print(f"  Algorithms : {sorted(srbench_agg['algorithm'].unique())}")
    print(f"  Datasets   : {len(srbench_datasets)}")

    # Resolve transformer result files.  The cluster jobs write per-group/seed trees
    # (results/mymodels/seed<N>/noise_<tau>/) that merge_seed_results.py folds into
    # results/mymodels/noise_<tau>/eval_tf_<m>.pkl.gz (all seeds in one artifact; a
    # local eval_mymodels.py run writes it directly); auto-discovery and the m145-ensure
    # step read that merged group dir, at the matching noise level.  The file/label
    # names are identical across noise levels.
    _RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")
    _tf_dir = results_io.group_noise_dir(_RESULTS_ROOT, "mymodels", args.noise)
    # Auto-discovery skips non_bpe and per-model TPSR wrappers;
    # the canonical TPSR baseline is loaded separately from --tpsr-results.
    _AUTO_SKIP = ("non_bpe", "89M_40_simp1_tpsr", "145M_40_simp1_tpsr",
                  "89M_40_tpsr", "145M_40_tpsr",
                  "m89_tpsr", "m145_tpsr")   # all three rename generations
    if args.results is None:
        result_paths = [
            p for p in results_io.list_srbench_pkls(_RESULTS_ROOT, args.noise, "mymodels")
            if not any(s in os.path.basename(p) for s in _AUTO_SKIP)
        ]
        if not result_paths:
            result_paths = [os.path.join(SCRIPT_DIR, "results/results_feynman.csv")]
    else:
        result_paths = [os.path.join(SCRIPT_DIR, p) for p in args.results]

    # Ensure m145 is present (auto-discovery may miss it if result files were
    # listed explicitly).  m145_unscale (the --unscale run) is added the same way.
    for _fname in ("eval_tf_145M_40.pkl.gz", "eval_tf_145M_40_unscale.pkl.gz"):
        _p = os.path.join(_tf_dir, _fname)
        if os.path.isfile(_p) and _p not in result_paths:
            result_paths.append(_p)

    # Build label->path map first so _remap_scale_labels can see all labels at once.
    # m89float -> "89M-float": our own 89M model trained on the floating-point-constant
    # ("continuous") grammar, evaluated here from results/eval_tf_m89float.pkl.gz like
    # the other models (NOT the old SRBench-JSON baseline, which has been removed).
    # Both the pre- and post-rename checkpoint labels (see plot_ood_vs_gap).
    _MODEL_RENAME = {"145M_80_simp1_float_ftnoise_48h": "145M-len80-float_ftnoise_48h",
                     # The --unscale arm of that same run.  _remap_scale_labels inserts "_unscaled"
                     # after the model prefix BEFORE this map is applied, so the key is that
                     # post-insertion spelling; without it the label stays
                     # "145M_unscaled_80_simp1_float_ftnoise_48h", which is both ugly and caught
                     # by make_requested_plots.sh's SRBENCH_DROP "145M_unscaled" filter.
                     "145M_unscaled_80_simp1_float_ftnoise_48h": "145M-len80-float_ftnoise_48h_unscaled",
                     "145M_80_simp1_float_ftnoise_e80": "145M-len80-float_ftnoise_e80",
                     "145M_80_simp1_float_ftnoise_e27": "145M-len80-float_ftnoise_e27",
                     "145M_80_simp1_float": "145M-len80-float",
                     "m145": "145M", "m89": "89M", "m89float": "89M-float",
                     "145M_40_simp1": "145M", "89M_40_simp1": "89M",
                     "145M_80_simp0": "145M-len80",
                     "145M_40": "145M", "89M_40": "89M", "145M_80": "145M-len80"}

    def _raw_label(path):
        lbl = os.path.basename(path).replace("eval_tf_", "").replace(".pkl.gz", "")
        try:
            with gzip.open(path, "rb") as fh:
                raw = pickle.load(fh)
            lbl = raw.get("model_name", lbl)
        except Exception:
            pass
        return "ours" if lbl == "non_bpe" else lbl

    def _remap_scale_labels(raw_labels):
        # New convention (eval_mymodels.py --unscale): the "_unscale" file is the
        # run with unscaling APPLIED, the plain file the one without it. Map
        # "_unscale"->"<model>_unscaled" and plain->"<model>" when both are present;
        # a lone variant collapses to the canonical model name.
        groups = {}
        for lbl in raw_labels:
            groups.setdefault(lbl.replace("_unscale", ""), set()).add(lbl)
        remap = {}
        for lbl in raw_labels:
            base  = lbl.replace("_unscale", "")
            grp   = groups[base]
            paired = (any("_unscale" in g for g in grp)
                      and any("_unscale" not in g for g in grp))
            if "_unscale" in lbl and paired:
                model, _, rest = base.partition("_")
                remapped = f"{model}_unscaled" + (f"_{rest}" if rest else "")
            else:
                remapped = base
            for src, dst in _MODEL_RENAME.items():
                remapped = remapped.replace(src, dst, 1)
            remap[lbl] = remapped
        return remap

    raw_labels = [_raw_label(p) for p in result_paths
                  if os.path.exists(p) and _raw_label(p) not in args.exclude]
    scale_remap = _remap_scale_labels(raw_labels)

    print(f"\nLoading {len(result_paths)} transformer result file(s)...")
    tf_results = []
    tf_file_labels = []          # (label, path) for in-process OOD evaluation
    all_tf_datasets = set()
    for path in result_paths:
        # Complexity is now its own figure (plot_complexity_vs_noise.py); this plot
        # only needs the OOD scatter, so skip the (slow, sympy) complexity pass.
        agg = load_transformer_results(path, args.r2_thr, with_complexity=False)
        if agg is None:
            continue
        raw_lbl = _raw_label(path)
        label = scale_remap.get(raw_lbl, raw_lbl)
        if raw_lbl in args.exclude or label in args.exclude:
            print(f"  {label:<20} excluded (--exclude)")
            continue
        print(f"  {label:<20} datasets={len(agg)}")
        tf_results.append((agg, label))
        tf_file_labels.append((label, path))
        all_tf_datasets |= set(agg["dataset"])

    # e2e baseline (eval_e2e.py CSV): different file + formula format, loaded as a
    # model so it shows in the table and both subplots. Its OOD R^2 is computed
    # separately (infix) by compute_ood_r2(), so it is NOT added to tf_file_labels.
    # The e2e curve must be its noise-matched run: the merged E2E group artifact
    # lives at results/e2e/noise_<tau>/results_feynman_e2e.pkl.gz, one per noise
    # level.  An explicit --e2e-results overrides the group default (a legacy .csv
    # of the same stem is a fallback via _resolve_artifact).  e2e is dropped from
    # the plot when the noise-matched artifact was not generated.
    if args.e2e_results is None:
        e2e_path = _resolve_artifact(results_io.e2e_srbench_pkl(_RESULTS_ROOT, args.noise))
    else:
        e2e_path = _resolve_artifact(os.path.join(SCRIPT_DIR, args.e2e_results))
    if os.path.exists(e2e_path) and "e2e" not in args.exclude:
        e2e_agg = load_transformer_results(e2e_path, args.r2_thr, with_complexity=False)
        if e2e_agg is not None:
            print(f"  {'e2e':<20} datasets={len(e2e_agg)}")
            tf_results.append((e2e_agg, "e2e"))
            all_tf_datasets |= set(e2e_agg["dataset"])

    # PhyE2E: same artifact schema and same infix x_0 convention as e2e, so it loads
    # through the same path and its OOD R^2 is re-evaluated in-process by
    # compute_ood_r2(phye2e_path=...).  Opt-in, so no existing figure changes.
    phye2e_path = None
    if getattr(args, "include_phye2e", False):
        phye2e_path = phye2e_addon.srbench_phye2e_pkl(_RESULTS_ROOT, args.noise)
        _lbl = phye2e_addon.DISPLAY_LABEL
        if phye2e_path and _lbl not in args.exclude:
            phye2e_agg = load_transformer_results(phye2e_path, args.r2_thr,
                                                 with_complexity=False)
            if phye2e_agg is not None:
                print(f"  {_lbl:<20} datasets={len(phye2e_agg)}")
                tf_results.append((phye2e_agg, _lbl))
                all_tf_datasets |= set(phye2e_agg["dataset"])
        elif not phye2e_path:
            print(f"  [warn] --include-phye2e: no artifact under "
                  f"{phye2e_addon.PHYE2E_GROUP}/ at noise {args.noise:g}")

    # PySR: identical plumbing to PhyE2E above -- same schema, same x_0 infix, OOD R^2
    # re-evaluated in-process by compute_ood_r2(pysr_path=...).  Opt-in.
    pysr_path = None
    if getattr(args, "include_pysr", False):
        pysr_path = pysr_addon.srbench_pysr_pkl(_RESULTS_ROOT, args.noise)
        _lbl = pysr_addon.DISPLAY_LABEL
        if pysr_path and _lbl not in args.exclude:
            pysr_agg = load_transformer_results(pysr_path, args.r2_thr,
                                                with_complexity=False)
            if pysr_agg is not None:
                print(f"  {_lbl:<20} datasets={len(pysr_agg)}")
                tf_results.append((pysr_agg, _lbl))
                all_tf_datasets |= set(pysr_agg["dataset"])
        elif not pysr_path:
            print(f"  [warn] --include-pysr: no artifact under "
                  f"{pysr_addon.PYSR_GROUP}/ at noise {args.noise:g}")

    # TPSR: loaded like e2e -- in-distribution R^2 from CSV, OOD R^2 re-evaluated
    # in-process by compute_ood_r2() via the TPSR prefix evaluator.
    tpsr_csv = os.path.join(SCRIPT_DIR, args.tpsr_results)
    if os.path.exists(tpsr_csv) and "TPSR" not in args.exclude:
        tpsr_agg = load_tpsr_results(tpsr_csv, args.r2_thr, noise=args.noise)
        if tpsr_agg is not None:
            print(f"  {'TPSR':<20} datasets={len(tpsr_agg)}")
            tf_results.append((tpsr_agg, "TPSR"))
            all_tf_datasets |= set(tpsr_agg["dataset"])
        else:
            # No TPSR rows at this noise level -> drop it from the OOD step too.
            print(f"  {'TPSR':<20} (no rows at noise={args.noise:g} -- skipped)")
            tpsr_csv = None
    else:
        tpsr_csv = None

    # Use all transformer-evaluated datasets as the benchmark universe.
    # SRBench methods that were not evaluated on a dataset are filled with
    # R^2=0 / solved=0 in build_comparison_table (see missing-fill logic there).
    common_datasets = sorted(all_tf_datasets)
    missing_from_srbench = all_tf_datasets - srbench_datasets
    print(f"  Benchmark universe : {len(common_datasets)} datasets (all transformer-evaluated)")
    if missing_from_srbench:
        print(f"  Not in SRBench     : {len(missing_from_srbench)} dataset(s) "
              f"-> counted as unsolved for SRBench methods")

    # Build colour map for transformer models
    tf_labels    = [label for _, label in tf_results]
    tf_color_map = _build_color_map(tf_labels)
    # Pin every one of our methods to a fixed colour matching the ood_vs_gap plots
    # (dark = input-unscaled/_unscaled, light = raw), so nothing collides with a palette
    # slot and the colours are identical across every plot family.
    _FIXED_MODEL_COLORS = {
        "145M":          "#4ade80",  "145M_unscaled": "#15803d",   # green (light default / dark --unscale)
        "145M-len80":    "#15803d",                                 # same green family, dark shade
        "89M":           "#fb923c",  "89M_unscaled":  "#c2410c",   # orange
        "89M-float":     _FLOAT_COLOR,                              # rose
        "e2e":           "#7c3aed",                                 # purple
        "TPSR":          "#dc2626",                                 # bold red (distinct from baseline blue)
    }
    for _lbl, _c in _FIXED_MODEL_COLORS.items():
        if _lbl in tf_color_map:
            tf_color_map[_lbl] = _c

    # -- Build and print comparison table --------------------------------------
    table = build_comparison_table(
        tf_results, srbench_agg, set(common_datasets), args.r2_thr, args.noise
    )

    print(f"\n{'='*90}")
    srb_note = (f"  ({len(missing_from_srbench)} not in SRBench -> counted as unsolved)"
                if missing_from_srbench else "")
    print(f"Feynman Benchmark -- noise={args.noise}  |  R^2-threshold={args.r2_thr}  |  "
          f"{len(common_datasets)} datasets{srb_note}")
    print(f"{'='*90}")
    print("  Solve Rate  = mean over datasets of (fraction of seeds that solved it)")
    print(f"  Strict Acc  = fraction of datasets where mean R^2 over seeds >= {args.r2_thr}")
    print("  Sym. Sol.   = fraction of trials with exact symbolic recovery (SRBench only)")
    print(f"{'-'*90}")

    display = table[["Algorithm", "Mean R^2", "Median R^2", "Solve Rate",
                      "Strict Acc", "Strict Solved", "Sym. Sol. Rate"]].copy()
    display["Mean R^2"]      = display["Mean R^2"].map(lambda x: f"{x:.4f}")
    display["Median R^2"]    = display["Median R^2"].map(lambda x: f"{x:.4f}")
    display["Solve Rate"]   = display["Solve Rate"].map(lambda x: f"{x*100:.1f}%")
    display["Strict Acc"]   = display["Strict Acc"].map(lambda x: f"{x*100:.1f}%")
    display["Sym. Sol. Rate"] = display["Sym. Sol. Rate"].map(
        lambda x: f"{x*100:.1f}%" if not np.isnan(x) else "--"
    )
    print(display.to_string(index=False))
    print(f"{'='*90}\n")

    # Complexity distribution of the true Feynman formulae (reference for the plot).
    truth_cplx = feynman_truth_complexity(SRBENCH_GT, args.noise, set(common_datasets))
    if np.isfinite(truth_cplx["q50"]):
        print(f"True Feynman formula complexity (unsimplified node count): "
              f"median={truth_cplx['q50']:.1f}  IQR={truth_cplx['q25']:.0f}-{truth_cplx['q75']:.0f}  "
              f"mean={truth_cplx['mean']:.2f}  std={truth_cplx['std']:.2f}  "
              f"(over {len(common_datasets)} datasets)\n")

    # Save table CSV
    out_dir = os.path.dirname(os.path.join(SCRIPT_DIR, args.output))
    os.makedirs(out_dir, exist_ok=True)
    _tbl_suffix = "" if args.noise == 0.0 else f"_noise{args.noise:g}"
    table_csv = os.path.join(out_dir, f"srbench_comparison_table{_tbl_suffix}.csv")
    table.to_csv(table_csv, index=False)
    print(f"[table] Saved -> {table_csv}")

    if not args.no_plot:
        # -- OOD R^2: compute in-process (or reuse a cached CSV) ----------------
        # noise>0 -> tag the artifact so a noise sweep does not overwrite the
        # noise-free results/ood_multi.csv (or collide across noise levels).
        if args.noise == 0.0:
            ood_csv_path = os.path.join(SCRIPT_DIR, args.ood_csv)
        else:
            _oc_base, _oc_ext = os.path.splitext(args.ood_csv)
            ood_csv_path = os.path.join(SCRIPT_DIR, f"{_oc_base}_noise{args.noise:g}{_oc_ext}")
        ood_df = None
        gap = args.gap
        if args.no_ood:
            print("[ood]  Skipped (--no-ood).")
        elif args.reuse_ood_csv:
            if os.path.exists(ood_csv_path):
                ood_df = pd.read_csv(ood_csv_path)
                print(f"[ood]  Reused {len(ood_df)} rows from {ood_csv_path}")
            else:
                print(f"[warn] --reuse-ood-csv set but {ood_csv_path} not found -- skipping OOD.")
        else:
            ood_data_path = os.path.join(SCRIPT_DIR, args.ood_data)
            detected_gap = gap
            # Fast path: reuse the raw OOD R^2 table plot_ood_vs_gap.py cached (all
            # gaps, all seeds), filtered to this gap.  Its labels are raw model names
            # (m145, ...), so remap to display names (145M, ...) like the table.
            _cache = os.path.join(SCRIPT_DIR, f"results/ood_raw_noise{args.noise:g}.csv")
            _gap_int = int(gap) if gap is not None else None       # args.gap is a str
            if args.reuse_ood and _gap_int is not None and os.path.exists(_cache):
                _sub = pd.read_csv(_cache)
                _sub = _sub[_sub["gap"] == _gap_int].drop(columns=["gap"]).copy()
                if not _sub.empty:
                    _sub["algorithm"] = _sub["algorithm"].map(lambda a: scale_remap.get(a, a))
                    ood_df = _sub
                    print(f"[ood]  Reused {len(_sub)} rows from {os.path.basename(_cache)} (gap {gap})")
                else:
                    print(f"[ood]  gap {gap} not in {os.path.basename(_cache)} -- computing.")
            # gap 0 uses datasets/feynman_ood_g0.pkl.gz (the synthetic in-distribution
            # set, same 99-equation universe as every other gap), NOT the PMLB split --
            # so this plot is consistent with the whole gap series and with ood_vs_gap.
            if ood_df is None and not os.path.exists(ood_data_path):
                print(f"[warn] OOD dataset not found: {ood_data_path} -- skipping OOD subplot.")
                detected_gap = None
            elif ood_df is None:
                print(f"[ood]  Computing OOD R^2 on {os.path.basename(ood_data_path)} "
                      f"for {len(tf_file_labels)} transformer model(s) + SRBench/e2e/TPSR ...")
                ood_df, detected_gap = compute_ood_r2(
                    ood_data_path, tf_file_labels, tpsr_path=tpsr_csv,
                    e2e_path=e2e_path, phye2e_path=phye2e_path,
                    pysr_path=pysr_path,
                    all_seeds=True,                      # per-seed rows -> x error bars
                    include_gt="GT" not in args.exclude, noise=args.noise)
            if gap is None and detected_gap is not None:
                gap = detected_gap
            if ood_df is not None:
                os.makedirs(os.path.dirname(ood_csv_path), exist_ok=True)
                ood_df.to_csv(ood_csv_path, index=False)
                print(f"[ood]  Computed {len(ood_df)} rows -> {ood_csv_path}")
                if not ood_df.empty:
                    summ = (ood_df.groupby("algorithm")["ood_r2"]
                            .agg(n="count", mean_ood_r2="mean")
                            .sort_values("mean_ood_r2", ascending=False))
                    print(summ.to_string(float_format=lambda v: f"{v:.4f}"))

        plot_comparison(
            table, args.r2_thr, args.noise,
            os.path.join(SCRIPT_DIR, args.output),
            tf_color_map=tf_color_map,
            truth_complexity=truth_cplx,
            ood_df=ood_df,
            gap=gap,
        )
        first_tf_agg = tf_results[0][0] if tf_results else None
        plot_per_equation(
            first_tf_agg, srbench_agg, common_datasets, args.r2_thr,
            os.path.join(SCRIPT_DIR, os.path.dirname(args.output)),
            noise_suffix=_tbl_suffix,
        )
        if args.show_plot:
            plt.show()


if __name__ == "__main__":
    main()
