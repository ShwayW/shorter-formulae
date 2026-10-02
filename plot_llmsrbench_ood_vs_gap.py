#!/usr/bin/env python3
"""
plot_llmsrbench_ood_vs_gap.py -- OOD accuracy vs gap for LLM-SRBench (curves only).

Evaluates each transformer method's discovered equations (from results.jsonl) on
pre-generated OOD datasets (datasets/llmsrbench_ood_g{gap}.pkl.gz) and plots the
fraction of problems with OOD R^2 >= threshold vs gap (out of that gap's OOD set).
Search time is now its own cross-noise figure, plot_llmsr_time_vs_noise.py.

By default includes all transformer-based methods in results/llmsrbench/ for the
lsr_transform split (including m89float -> "89M-float"), excluding gemma/tpsr variants.

Usage:
    python plot_llmsrbench_ood_vs_gap.py [--gaps 0 1 2 4 8 16 32 64 128]
                                         [--split lsr_transform]
                                         [--output plots/llmsrbench_ood_vs_gap.png]
"""

import argparse
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import results_io
import tpsr_addon
import e2e_tpsr_addon
import devncon_addon
import aifeynman_addon
import phye2e_addon
import pretrained_addon
import pysr_addon
import llmsr_overlay
import oneshot_overlay
import ood_transform
from plot_io import (save_fig, GRID_LEGEND_KW, GRID_LEGEND_NCOL, bottom_legend, above_legend,
                     sort_legend, display_labels, font_scale, GRID_FIG_W_IN,
                     GRID_ROW_FIG_H_IN)

# Font sizes below are written as the point size the text should have ON THE PAGE and
# multiplied by _FS -- the grid is 10 in wide and LaTeX shrinks it to \textwidth, so a
# bare "12" would reach the reader at under 7 pt.  See plot_io.font_scale.
_FS = font_scale(GRID_FIG_W_IN)
from plot_markers import method_marker, HOLLOW, CURVE_MS, CURVE_LW
# One implementation of the OOD R^2, shared with the SRBench path (see _r2_score).
from compare_srbench import _ood_r2_score
from results_io import load_rows_any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DEFAULT_GAPS   = [0, 1, 2, 4, 8, 16, 32, 64, 128]
_CURVE_LW = CURVE_LW   # one width for every curve, overlays included
DEFAULT_R2_THR = 0.99


# Substrings in dir names that mark methods to skip by default.
# m89float is our own 89M float/continuous model (89M-float) -- it is NOT skipped.
_SKIP_SUBSTR = ("gemma", "tpsr", "oracle")

# Model name -> display label (number of parameters).
# m89float -> "89M-float": our own 89M model on the floating-point-constant grammar.
# Both the pre- and post-rename checkpoint labels (see plot_ood_vs_gap).
# The 2026-08-17 checkpoint rename (`<name>_res` -> `<name>_simp{0,1}_res`, recording
# whether targets were canonicalised) changed the --model string, and the model name
# flows into artifact filenames -- so both spellings name the SAME weights and must map
# to the same display label.  Longer keys FIRST: the remap applies every key in order,
# so a bare "145M_40" listed first would eat the prefix of "145M_40_simp1".
# "simplipy_145M_80l" FIRST: it contains "145M_80" as a substring, so the bare key
# listed earlier would eat it and yield "simplipy_145M-len80l".  Same weights family as
# 145M-len80, but canonicalised with the SimpliPy library rather than simplifyFormula.py,
# which is what the suffix records.
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
                 "simplipy_145M_80l": "145M-len80-simplipy",
                 "m145": "145M", "m89": "89M", "m89float": "89M-float",
                 "145M_40_simp1": "145M", "89M_40_simp1": "89M",
                 "145M_80_simp0": "145M-len80",
                 "145M_40": "145M", "89M_40": "89M", "145M_80": "145M-len80"}

# Colour scheme: model family by hue, shade by whitening.
#   145M (m145)  : green  -- dark (#15803d) = input-unscaled (unscaled), light (#4ade80) = raw
#   89M (m89) : orange -- dark (#c2410c) = input-unscaled (unscaled), light (#fb923c) = raw
#   89M-float (m89float) : rose (#db2777) -- the floating-point-constant 89M model
#   e2e: purple
def _color(label):
    # "<model> + TPSR" combined runs -- own dark-shade colours (checked before the
    # "89M"/"145M" prefixes, which they also start with).
    if label in tpsr_addon.TPSR_COLORS:
        return tpsr_addon.TPSR_COLORS[label]
    # "<model> + D&C" combined runs
    if label in devncon_addon.DEVNCON_COLORS:
        return devncon_addon.DEVNCON_COLORS[label]
    # "89M-float" must be checked before the "89M" prefix (it also starts with "89M").
    if label == "89M-float":
        return "#db2777"                                          # rose / deep pink
    if label.startswith("89M"):
        return "#c2410c" if "unscaled" in label else "#fb923c"    # burnt orange (dark) / light orange
    # The SimpliPy-canonicalised len-80 145M.  Lime rather than a third green shade:
    # it also starts with "145M", and green-400/green-700 are already spoken for by
    # 145M and 145M-len80, which two near-identical greens could not separate.
    if label == "145M-len80-simplipy":
        return "#65a30d"                                          # lime-600
    # 145M-len80 takes the dark green the retired input-unscaled arms used, so it does
    # not collide with 145M's light green (it also starts with "145M").
    if label == "145M-len80":
        return "#15803d"
    # The 145M float-constant model: dark rose, pairing with 89M-float's rose so the two
    # restrict_consts=0 arms read as a family.  Before the "145M" prefix test, which
    # would otherwise give it plain 145M's light green.
    if label == "145M-len80-float":
        return "#9d174d"                                          # rose-800 (dark)
    # The NOISE-FINETUNED 145M float arm (epoch-27 snapshot) -- RED, swapped with
    # 145M+D&C on 2026-09-17: it is the headline arm now, and red is the most legible
    # line on a crowded panel.  Equality test, so it must precede the
    # startswith("145M") fallback below.
    if label == "145M-len80-float_ftnoise_e27":
        return "#dc2626"                                          # red
    # The same arm after 23 h of finetuning (e80) rather than 7.6 h (e27): red-800,
    # a darker step of e27's red, so the pair reads as one arm at two budgets.
    if label == "145M-len80-float_ftnoise_e80":
        return "#991b1b"                                          # red-800
    # The same arm after 48 h of finetuning: red-900 for the plain beam curve, and a
    # darker step again for its --unscale arm, keeping the repo convention that the
    # unscaled curve is the dark one of a pair.
    if label == "145M-len80-float_ftnoise_48h":
        return "#b91c1c"                                          # red-700
    if label == "145M-len80-float_ftnoise_48h_unscaled":
        return "#7f1d1d"                                          # red-900


    if label.startswith("145M"):
        return "#15803d" if "unscaled" in label else "#4ade80"    # forest green (dark) / light green
    # The three pretrained-transformer baselines, sharing the colours they take on the
    # SRBench figures (plot_ood_vs_gap._tf_color) so a method reads the same in both
    # families.  Opt-in only (--include-pretrained), so sets 3/4 are unaffected.
    if pretrained_addon.is_pretrained(label):
        return pretrained_addon.color(label)
    # PhyE2E before the e2e test: distinct method, distinct colour, and its label must
    # never be confused with the plain "e2e" baseline's.
    if phye2e_addon.is_phye2e(label):
        return phye2e_addon.color(label)
    # PySR -- amber, the same identity it carries on the SRBench figures
    # (plot_ood_vs_gap._tf_color), so one method reads the same in every figure family.
    if pysr_addon.is_pysr(label):
        return pysr_addon.color(label)
    # The standalone TPSR baseline ("e2e+TPSR" in the legend) -- the same teal it has on
    # the SRBench figures (plot_ood_vs_gap._tf_color), so one method reads the same in
    # every figure family.  Exact-match, so it cannot steal 89M+TPSR / 145M+TPSR, which
    # are handled by the TPSR_COLORS lookup at the top of this function.
    if e2e_tpsr_addon.is_e2e_tpsr(label):
        return e2e_tpsr_addon.color(label)
    if label.startswith("e2e"):
        return "#7c3aed"                                          # violet / purple
    # AI Feynman baseline -- same steel blue / pentagon / dashed identity it already
    # has on the SRBench figures (plot_ood_vs_gap._tf_color), so one method reads the
    # same in every figure family.  See aifeynman_addon.
    if aifeynman_addon.is_aifeynman(label):
        return aifeynman_addon.COLOR                              # steel blue
    # The one-shot LLM arms (blue / amber) -- only the best-seed figure routes their
    # labels through here; the grid draws them from oneshot_overlay, which colours its
    # own lines.
    if oneshot_overlay.is_oneshot(label):
        return oneshot_overlay.color(label)
    return "#6b7280"                                              # medium grey (fallback)


def _linestyle(label):
    # Dashed for the AI Feynman baseline (SRBench baselines are dashed on the
    # SRBench figures too) and for PhyE2E, the other external published model.
    if aifeynman_addon.is_aifeynman(label):
        return aifeynman_addon.LINESTYLE
    if phye2e_addon.is_phye2e(label):
        return phye2e_addon.LINESTYLE
    # PySR is SOLID: dashed here means "external PUBLISHED baseline", and SRBench
    # publishes no PySR results at all -- both halves of this curve are our own run.
    if pysr_addon.is_pysr(label):
        return pysr_addon.LINESTYLE
    # Dashed for the one-shot LLM arms too, so the "one LLM call" family reads as a
    # family against the solid search-based curves (oneshot_overlay's convention, which
    # the best-seed figure inherits through this helper).
    if oneshot_overlay.is_oneshot(label):
        return oneshot_overlay.LINESTYLE
    return "-"


# -- Data loading --------------------------------------------------------------

def _method_artifact(d):
    """The per-method result file inside dir `d`: results.pkl.gz, or the legacy
    results.jsonl for archived runs. None when neither exists."""
    for name in ("results.pkl.gz", "results.jsonl"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def _load_jsonl(path):
    return load_rows_any(path)[1]


def _remap_scale_labels(raw_labels):
    """raw (plain) -> canonical, input-unscaled (_unscale) -> _unscaled; then rename m145/m89 to param counts."""
    groups = {}
    for lbl in raw_labels:
        groups.setdefault(lbl.replace("_unscale", ""), set()).add(lbl)
    remap = {}
    for lbl in raw_labels:
        base = lbl.replace("_unscale", "")
        grp  = groups[base]
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


def discover_methods_from_dirs(dirs, split, exclude_substrings=_SKIP_SUBSTR):
    """Return {display_label: jsonl_path} for the valid method dirs in `dirs`."""
    candidates = []
    for d in sorted(dirs):
        jl = _method_artifact(d) if os.path.isdir(d) else None
        if jl is None:
            continue
        raw = os.path.basename(d).replace(f"_{split}", "") or os.path.basename(d)
        if any(s in raw for s in exclude_substrings):
            continue
        candidates.append((raw, jl))
    remap = _remap_scale_labels([r for r, _ in candidates])
    return {remap[r]: jl for r, jl in candidates}


def discover_methods(results_dir, split, exclude_substrings=_SKIP_SUBSTR):
    """Return {display_label: jsonl_path} for all valid method dirs in one flat dir."""
    import glob
    return discover_methods_from_dirs(
        glob.glob(os.path.join(results_dir, f"*_{split}*")), split, exclude_substrings)


def load_method_data(jsonl_path):
    """Return {seed: {equation_id: (discovered_equation, search_time)}} from results.jsonl.

    The merged multi-seed jsonl holds one row per (seed, problem); grouping by the
    'seed' field keeps the seeds separate so the plot can show the seed-to-seed
    spread.  Rows written before the multi-seed protocol carry no 'seed' field and
    all fall under seed 0 (single-seed behaviour -> band collapses to the line).
    """
    from collections import defaultdict
    rows = _load_jsonl(jsonl_path)
    out: dict = defaultdict(dict)
    for r in rows:
        seed = int(r.get("seed", 0) or 0)
        name = r.get("equation_id", "")
        eq   = r.get("discovered_equation", "") or ""
        t    = r.get("search_time", float("nan"))
        out[seed][name] = (eq.strip(), float(t) if t is not None else float("nan"))
    return dict(out)


def _r2_score(y_true, y_pred):
    """Drop non-finite predictions, then score with the SRBench OOD scorer.

    This used to carry its own copy of the scoring arithmetic, including the
    np.isclose(ss_tot, 0.0) constant-target guard whose ABSOLUTE 1e-8 tolerance handed
    a free 1.0 to any prediction on a small-magnitude target (see
    compare_srbench._ood_r2_score).  Delegating keeps one implementation, so the two
    benchmarks cannot drift apart on what an OOD R^2 means.  The finite-prediction mask
    below is this path's own and is unchanged.
    """
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    mask   = np.isfinite(y_pred)
    if mask.sum() < 2:
        return 0.0
    return _ood_r2_score(y_true[mask], y_pred[mask])


_INFIX_EVAL_NS_BASE = {
    "__builtins__": {},
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "pi": np.pi, "e": np.e,
}


import re as _re
_INFIX_VAR_RE = _re.compile(r"\bx_\d+\b")


def _is_infix(formula: str) -> bool:
    """Heuristic: infix formulas start with '(' or contain it early.

    The `x_<digit>` test is the reliable one and is checked first: PREFIX formulas
    (our transformers) name their inputs v1..vN and can never contain "x_",
    while every infix producer here (e2e, AI Feynman via
    eval_aifeynman.normalize_vars) names them x_0..x_N.  The original
    paren heuristic alone misroutes a paren-free infix expression such as
    "x_0*x_1 + 2.0*x_2" -- common in AI Feynman's sympy output -- into
    wrapExtEvalPN, which then returns NaN for the whole method.
    """
    s = formula.lstrip()
    if _INFIX_VAR_RE.search(s):
        return True
    return s.startswith("(") or s.startswith("-") or ("(" in s[:20])


def _eval_infix(formula: str, X: np.ndarray) -> np.ndarray:
    """Evaluate an infix formula with x_0, x_1, ... variables."""
    ns = dict(_INFIX_EVAL_NS_BASE)
    for i in range(X.shape[1]):
        ns[f"x_{i}"] = X[:, i].astype(np.float64)
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            y = np.asarray(eval(formula, ns), dtype=np.float64)  # noqa: S307
        if y.ndim == 0:
            y = np.full(X.shape[0], float(y))
        return y
    except Exception:
        return np.full(X.shape[0], np.nan)


# -- OOD evaluation ------------------------------------------------------------

def eval_ood_raw_for_gap(ood_path, methods_data, gap_override=None):
    """Per-(algorithm, seed, equation_id) OOD R^2 for one gap -- raw, threshold-free.

    This is the expensive step: each method's discovered equation (per seed) is
    re-evaluated on the gap's OOD point cloud.  Handles both prefix (v1/v2/..., via
    wrapExtEvalPN) and infix (x_0/x_1/..., via numpy eval) formula formats.

    `gap_override` stamps the rows' x-axis value instead of the file's payload gap.
    Callers pass the same value the file already stores, so it is a no-op today; it is
    kept because the caches key on this column.
    """
    from eval_mymodels import predict   # prefix notation, v1/v2/... via wrapExtEvalPN

    with gzip.open(ood_path, "rb") as fh:
        payload = pickle.load(fh)
    gap      = float(gap_override if gap_override is not None
                     else payload.get("gap", float("nan")))
    ood_data = payload["datasets"]  # {name: {X, y}}

    rows = []
    for label, seed_map in methods_data.items():
        for seed, prob_map in seed_map.items():
            for name, (formula, _) in prob_map.items():
                if name not in ood_data:
                    continue
                X_ood = ood_data[name]["X"].astype(np.float64)
                y_ood = ood_data[name]["y"].astype(np.float64)
                if _is_infix(formula):
                    y_pred = _eval_infix(formula, X_ood)
                else:
                    y_pred = predict(formula, X_ood)
                rows.append({
                    "gap":         gap,
                    "algorithm":   label,
                    "seed":        int(seed),
                    "equation_id": name,
                    "ood_r2":      _r2_score(y_ood, y_pred),
                })
    return rows


def ood_raw_table(methods_data, gaps):
    """Concatenated per-(gap, algorithm, seed, equation_id) OOD R^2 -- the slow part.

    Threshold-independent, so main() caches it to results/llmsr_ood_raw_noise<tau>.csv
    and re-thresholds cheaply on later re-plots (as plot_ood_vs_gap.py does for SRBench).

    """
    frames = []
    for g in gaps:
        ood_path = os.path.join(SCRIPT_DIR, ood_transform.llmsr_ood_path(g))
        if not os.path.exists(ood_path):
            print(f"[warn] Missing: {ood_path}")
            continue
        print(f"[gap={g:>3}] evaluating {len(methods_data)} methods...", flush=True)
        frames.extend(eval_ood_raw_for_gap(ood_path, methods_data, gap_override=g))
    return pd.DataFrame(frames, columns=["gap", "algorithm", "seed", "equation_id", "ood_r2"])


def accuracy_from_raw(raw_df, r2_thr, monotone=True):
    """Threshold the raw OOD R^2 table into per-(gap, algorithm, seed) solve counts:
    a list of {gap, algorithm, seed, n_correct, n_total}.

    monotone=True applies ood_transform.monotone_gap first: an equation that has
    already failed at a smaller gap stays failed at every larger one."""
    if monotone:
        # Restrict FIRST (same problem set at every gap), then prefix-AND.
        raw_df = ood_transform.restrict_to_common(raw_df)
        raw_df = ood_transform.monotone_gap(raw_df, r2_thr)
    records = []
    key = ood_transform.problem_key(raw_df) or "equation_id"
    for g, gap_df in raw_df.groupby("gap"):
        # DENOMINATOR = the problem universe at this gap (the union over algorithms),
        # NOT the rows this method happens to have.  A method evaluated on only part of
        # the benchmark -- NeSymReS (<=3 vars), tf4sr (<=6), SymFormer (<=2) -- has no
        # row for the rest, and counting only its own rows would score it against its
        # easiest slice: SymFormer read 44% when it had solved 2.2 of 111 problems.
        # Problems a method never attempted count as UNSOLVED, which is what
        # common_problems' docstring already promised and what the SRBench twin
        # (plot_ood_vs_gap.accuracy_from_raw, n_universe) has always done.
        # Every method that predates this carried full 111/111 coverage, so sets 3-5
        # are numerically unchanged.
        n_universe = int(gap_df[key].nunique())
        for (alg, seed), grp in gap_df.groupby(["algorithm", "seed"]):
            n_correct = int((grp.groupby(key)["ood_r2"].max() >= r2_thr).sum())
            records.append({
                "gap":       g,
                "algorithm": alg,
                "seed":      int(seed),
                "n_correct": n_correct,
                "n_total":   n_universe,
            })
    return records


def collect_ood_accuracy(methods_data, gaps, r2_thr):
    """Compute raw OOD R^2 then threshold (no caching -- see main() for the cache)."""
    return accuracy_from_raw(ood_raw_table(methods_data, gaps), r2_thr)


def collect_timing(methods_data):
    """Return {label: avg_search_time_s} from results.jsonl data (over all seeds)."""
    timing = {}
    for label, seed_map in methods_data.items():
        times = [t for prob_map in seed_map.values()
                 for _, (_, t) in prob_map.items() if np.isfinite(t)]
        if times:
            timing[label] = float(np.mean(times))
    return timing


# -- Plot ----------------------------------------------------------------------

def _prepare_noise_llmsr(noise, args, gaps):
    """Per-noise curve data: (acc[(gap,alg)]->{mean,lo,hi}, algs_sorted).  Discovers
    method dirs, loads results, and reads/writes the OOD R^2 cache like main() does."""
    from collections import defaultdict
    if args.results_dir is None:
        method_dirs = results_io.list_llmsr_method_dirs(
            os.path.join(SCRIPT_DIR, "results"), noise, args.split)
        methods_paths = discover_methods_from_dirs(method_dirs, args.split, _SKIP_SUBSTR)
    else:
        methods_paths = discover_methods(os.path.join(SCRIPT_DIR, args.results_dir),
                                         args.split, exclude_substrings=_SKIP_SUBSTR)
    # Opt-in: the "<model> + TPSR" combined runs (their dirs carry 'tpsr', so the
    # default discovery skips them).
    if getattr(args, "include_tpsr_combo", False):
        methods_paths.update(tpsr_addon.llmsr_tpsr_methods(
            os.path.join(SCRIPT_DIR, "results"), noise, args.split))
    if getattr(args, "include_devncon", False):
        methods_paths.update(devncon_addon.llmsr_devncon_methods(
            os.path.join(SCRIPT_DIR, "results"), noise, args.split))
    # Opt-in: the AI Feynman baseline (its own results/aifeynman/ tree, which is not
    # in results_io.GROUPS, so the default discovery never sees it).
    if getattr(args, "include_aifeynman", False):
        methods_paths.update(aifeynman_addon.llmsr_aifeynman_methods(
            os.path.join(SCRIPT_DIR, "results"), noise, args.split))
    # Opt-in: the standalone TPSR baseline (results/tpsr/).  Its group IS in
    # results_io.GROUPS, but its dir name carries 'tpsr' so _SKIP_SUBSTR drops it from
    # the default discovery -- see e2e_tpsr_addon.
    if getattr(args, "include_e2e_tpsr", False):
        methods_paths.update(e2e_tpsr_addon.llmsr_methods(
            os.path.join(SCRIPT_DIR, "results"), noise, args.split))
    # Opt-in: PhyE2E (its own results/phye2e/ tree, likewise outside results_io.GROUPS).
    if getattr(args, "include_phye2e", False):
        methods_paths.update(phye2e_addon.llmsr_methods_for_groups(
            os.path.join(SCRIPT_DIR, "results"), noise, args.split,
            phye2e_addon.active_groups(args)))
    # Opt-in: PySR (its own results/pysr/ tree, likewise outside results_io.GROUPS).
    if getattr(args, "include_pysr", False):
        methods_paths.update(pysr_addon.llmsr_methods_for_groups(
            os.path.join(SCRIPT_DIR, "results"), noise, args.split,
            pysr_addon.active_groups(args)))
    # Opt-in: NeSymReS / tf4sr / SymFormer (their own trees, outside results_io.GROUPS
    # like AI Feynman and PhyE2E).  See plot_transformer_methods.py for why they are
    # not in sets 3/4.
    if getattr(args, "include_pretrained", False):
        methods_paths.update(pretrained_addon.llmsr_methods(
            os.path.join(SCRIPT_DIR, "results"), noise, args.split))
    methods_paths = {k: v for k, v in methods_paths.items() if k not in args.exclude}
    # --only: keep just the listed methods (the SRBench twin plot_ood_vs_gap.py has
    # carried this since the camera-ready subset figures; sets 3/4 pass --exclude and
    # never set it, so they are unaffected).
    if getattr(args, "only", None):
        keep = set(args.only)
        methods_paths = {k: v for k, v in methods_paths.items() if k in keep}
    if not methods_paths:
        return {}, []
    methods_data = {label: load_method_data(jl) for label, jl in methods_paths.items()}

    ood_cache = os.path.join(SCRIPT_DIR, "results",
                             ood_transform.llmsr_cache_name(noise))
    if args.reuse_ood and os.path.exists(ood_cache):
        raw = pd.read_csv(ood_cache)
        missing = [g for g in gaps if float(g) not in set(raw["gap"].unique())]
        if missing:
            raw = pd.concat([raw, ood_raw_table(methods_data, missing)],
                            ignore_index=True)
            raw.to_csv(ood_cache, index=False)
        print(f"[ood] reused OOD R^2 from {os.path.basename(ood_cache)} (noise {noise:g})")
    else:
        raw = ood_raw_table(methods_data, gaps)
        if not raw.empty:
            os.makedirs(os.path.dirname(ood_cache), exist_ok=True)
            raw.to_csv(ood_cache, index=False)
            print(f"[ood] wrote OOD R^2 cache -> {ood_cache}")
    records = accuracy_from_raw(raw[raw["gap"].isin([float(g) for g in gaps])], args.r2_thr,
                                monotone=not args.no_monotone_gap)

    by_key = defaultdict(list)
    for r in records:
        if r["n_total"]:
            by_key[(r["gap"], r["algorithm"])].append(r["n_correct"] / r["n_total"])
    acc = {}
    for key, rates in by_key.items():
        arr = np.asarray(rates, dtype=np.float64)
        m, s = float(arr.mean()), float(arr.std())
        acc[key] = {"mean": m, "lo": max(0.0, m - s), "hi": min(1.0, m + s)}

    _base = ood_transform.baseline_gap()
    def _rate0(alg):
        e = acc.get((float(_base), alg)) or acc.get((_base, alg))
        return e["mean"] if e else 0.0
    algs_sorted = sorted(methods_data, key=lambda a: -_rate0(a))
    return acc, algs_sorted


def _yerr(ys, lo, hi):
    """Asymmetric yerr for ax.errorbar from a (mean, lo, hi) band, clipped nonneg."""
    ys, lo, hi = np.asarray(ys, float), np.asarray(lo, float), np.asarray(hi, float)
    return np.vstack([np.clip(ys - lo, 0, None), np.clip(hi - ys, 0, None)])


def _draw_curves_llmsr(ax, acc, algs_sorted, gaps, r2_thr, *, xlabel=True, ylabel=True):
    """Draw the LLM-SRBench OOD-accuracy-vs-gap curves (+ seed-std error bars) onto `ax`.

    The +/-1 seed-std spread is a per-point error bar, not the translucent band this
    used to shade -- with this many methods per panel the bands washed into each other
    (same change as plot_ood_vs_gap._draw_curves)."""
    gap_pos = {g: i for i, g in enumerate(gaps)}
    for alg in algs_sorted:
        xs, ys, lo_arr, hi_arr = [], [], [], []
        for g in gaps:
            entry = acc.get((float(g), alg)) or acc.get((g, alg))
            if entry is not None:
                xs.append(gap_pos[g])
                ys.append(entry["mean"])
                lo_arr.append(entry["lo"])
                hi_arr.append(entry["hi"])
        if not xs:
            continue
        # Curves are fractions in [0, 1] and are drawn as such, on a fixed 0-1 axis --
        # matching sets 1/2 so every OOD-vs-gap figure shares one y scale (2026-09-16).
        col = _color(alg)
        ax.plot(xs, ys, label=alg, color=col, lw=_CURVE_LW, linestyle=_linestyle(alg),
                marker=method_marker(alg), markersize=CURVE_MS, **HOLLOW)
        ax.errorbar(xs, ys, yerr=_yerr(ys, np.array(lo_arr).clip(0), hi_arr),
                    fmt="none", ecolor=col, elinewidth=1.0, capsize=2.5,
                    capthick=1.0, alpha=0.85, zorder=1.8)
    ax.set_ylim(0, 1)
    if xlabel:
        ax.set_xlabel("k", fontsize=12 * _FS)
    if ylabel:
        ax.set_ylabel(f"Fraction with $R^2 \\geq$ {r2_thr}", fontsize=12 * _FS)
    ax.set_xticks(range(len(gaps)))
    ax.set_xticklabels(["0" if g == 0 else str(g) for g in gaps])
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11 * _FS)


GRID_NOISES = [0.0, 0.001, 0.01, 0.1]


def draw_grid_row_llmsr(axes, args, gaps, titles=True):
    """Draw the four LLM-SRBench noise panels into `axes` (an iterable of 4 axes).

    Split out of _plot_grid_llmsr so the combined figure (plot_combined_curves_grid.py)
    can hang this row under the SRBench one and share a single legend.  Returns
    {raw algorithm label: legend handle}.
    """
    axes = list(axes)          # iterated twice: drawn, then read back for the legend
    for ax, noise in zip(axes, GRID_NOISES):
        acc, algs_sorted = _prepare_noise_llmsr(noise, args, gaps)
        _draw_curves_llmsr(ax, acc, algs_sorted, gaps, args.r2_thr,
                           xlabel=False, ylabel=False)
        # Four panels across a 10 in canvas leave ~2.3 in each -- not room for all nine
        # gap labels, which collide.  Every tick (and its gridline) stays; only alternate
        # LABELS are hidden, so both endpoints (0 and 128) survive.
        for _lbl in ax.get_xticklabels()[1::2]:
            _lbl.set_visible(False)
        # Opt-in single-seed LLMSR-Llama overlay (from score_llmsr_ood.py caches).
        if getattr(args, "include_llmsr", False):
            # One curve per requested backbone; the default list is just Llama, so a run
            # without --llmsr-backbones draws exactly what it drew before.
            for _lbl in (getattr(args, "llmsr_backbones", None)
                         or [llmsr_overlay.DEFAULT_LABEL]):
                llmsr_overlay.draw_overlay(ax, "lsr_transform", noise, gaps, args.r2_thr,
                                           monotone=not args.no_monotone_gap,
                                           as_percent=False, label=_lbl)
        # Opt-in one-shot LLM arms (from score_oneshot_ood.py caches), same story:
        # one dashed line per arm, with +/-1 seed-std error bars for the arms that ran
        # more than one seed (Llama-3.3-70B did; the other two are seed 0).
        # --exclude also thins the one-shot family here: the arms are appended after the
        # methods_paths filtering, so without this they would survive an --exclude that
        # names them (make_requested_plots.sh drops OneShot-Llama / -Llama70B from the
        # curve grids, keeping the Gemini arm).
        if getattr(args, "include_oneshot", False):
            oneshot_overlay.draw_overlay(ax, "lsr_transform", noise, gaps, args.r2_thr,
                                         monotone=not args.no_monotone_gap, as_percent=False,
                                         labels=oneshot_overlay.visible_labels(
                                             getattr(args, "exclude", None)))
        if titles:
            ax.set_title(f"noise = {noise:g}", fontsize=13 * _FS)

    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    return seen


def _plot_grid_llmsr(args, gaps):
    """Single row of four LLM-SRBench OOD-accuracy-vs-gap panels, one per noise level."""
    # Canvas matches the SRBench figure (plot_ood_vs_gap._plot_grid): the two face each
    # other in the paper, so they are emitted at the same size and LaTeX scales both by
    # the same factor at width=\textwidth.  ONE ROW of four panels (2026-09-21; was a
    # 2x2 grid), sharey so only the leftmost panel carries y tick labels -- with four
    # panels across a 10 in canvas there is no room for four sets of them.
    fig, axes = plt.subplots(1, 4, figsize=(GRID_FIG_W_IN, GRID_ROW_FIG_H_IN),
                             sharex=True, sharey=True)
    seen = draw_grid_row_llmsr(list(axes.flat), args, gaps)

    fig.supylabel(f"Fraction with $R^2 \\geq$ {args.r2_thr}", fontsize=12 * _FS)

    # Alphabetical by the PRINTED label -- after rename_legend, so the standalone TPSR
    # sorts under "e2e+TPSR" (same rule as the SRBench grid).
    _handles, _labels = sort_legend(
        list(seen.values()),
        display_labels(e2e_tpsr_addon.rename_legend(seen.keys())))
    # The legend is a STRIP UNDER the panel row, spanning the full canvas width, with the
    # shared x-label sitting just above it; tight_layout shrinks the axes to the measured
    # height that is left, so the box can never land on data however many arms are added.
    # bottom_legend drops a column at a time until the strip fits the canvas, so it
    # cannot overhang and inflate the saved bbox the way the old fixed two-row strip did.
    # Same treatment as the SRBench grid (plot_ood_vs_gap._plot_grid).
    _leg, frac = bottom_legend(fig, _handles, _labels, ncol=GRID_LEGEND_NCOL,
                               **GRID_LEGEND_KW)
    fig.supxlabel("Extrapolation Constant (k)", fontsize=12 * _FS, y=above_legend(frac, fig))
    fig.tight_layout(rect=(0, frac, 1, 1))
    save_fig(fig, os.path.join(SCRIPT_DIR, args.output or "plots/llmsrbench_ood_vs_gap_grid.png"))
    if args.show:
        plt.show()


def build_parser():
    """The CLI, split out of main() so plot_combined_curves_grid.py can parse exactly the
    same LLM-SRBench flags for the bottom row of the combined figure."""
    ap = argparse.ArgumentParser(
        description="Plot OOD accuracy and time vs gap for LLM-SRBench.")
    ap.add_argument("--results-dir", default=None,
                    help="Flat dir of <method>_<split> result dirs. Default: gather "
                         "them from every group under results/<group>/noise_<tau>/"
                         "llmsrbench/ for the chosen --noise.")
    ap.add_argument("--split", default="lsr_transform")
    ap.add_argument("--gaps", nargs="+", type=int, default=None,
                    help="x-axis gap values. Default: 0 1 2 4 8 16 32 64 128.")
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    ap.add_argument("--no-monotone-gap", action="store_true",
                    help="Score each gap independently, allowing a formula that failed at a smaller gap to count as solved at a larger one. Off by default: 'solved at gap k' normally means 'solved at every gap <= k' (ood_transform.monotone_gap).")
    ap.add_argument("--output", default=None,
                    help="Output PNG. Default: plots/llmsrbench_ood_vs_gap[_noise<tau>].png")
    ap.add_argument("--noise", type=float, default=0.0, choices=[0.0, 0.001, 0.01, 0.1],
                    help="Target-noise level: reads results from "
                         "results/llmsrbench/noise_<tau>/ and tags the output. Default: 0.0.")
    ap.add_argument("--exclude", nargs="*", default=[])
    pretrained_addon.add_arguments(ap)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Keep ONLY these methods (applied after --exclude). Mirrors "
                         "plot_ood_vs_gap.py --only.")
    ap.add_argument("--reuse-ood", action="store_true",
                    help="Reuse the cached OOD R^2 table "
                         "(results/llmsr_ood_raw_noise<tau>.csv) instead of recomputing "
                         "-- instant re-plots (layout / threshold tweaks). Missing gaps "
                         "are computed and appended; the cache is always (re)written.")
    ap.add_argument("--grid", action="store_true",
                    help="Combine all four noise levels into one 2x2 figure "
                         "(one subplot per noise) with a shared legend. "
                         "Default output: plots/llmsrbench_ood_vs_gap_grid.png.")
    ap.add_argument("--include-tpsr-combo", action="store_true",
                    help="Also draw the '<model> + TPSR' combined runs "
                         "(89M+TPSR, 145M+TPSR) from results/mymodels_tpsr/. Off by "
                         "default; their OOD rows come from augment_tpsr_caches.py.")
    ap.add_argument("--include-devncon", action="store_true",
                    help="Also draw the '<model> + D&C' combined runs "
                         "(89M+D&C, 145M+D&C) from results/mymodels_devncon/. Off by "
                         "default.")
    ap.add_argument("--include-phye2e", action="store_true",
                    help="Also draw PhyE2E (Nature MI 2025, run by eval_phye2e.py) from "
                         "results/phye2e/. Off by default. This is their 'e2e' ablation "
                         "decoded with beam search, NOT their full D&C+MCTS pipeline -- "
                         "hence the '(E2E)' in the label.")
    ap.add_argument("--include-pysr", action="store_true",
                    help="Also draw PySR (Cranmer 2023, run by eval_pysr.py) from "
                         "results/pysr/. Off by default. Amber, solid -- solid because "
                         "SRBench publishes no PySR numbers, so this is our own run on "
                         "both benchmarks, not a published baseline.")
    ap.add_argument("--include-aifeynman", action="store_true",
                    help="Also draw the AI Feynman 2.0 baseline (the original authors' "
                         "code, run by eval_aifeynman.py --benchmark llmsrbench) from "
                         "results/aifeynman/. Off by default. Steel blue, dashed, "
                         "pentagon -- matching its SRBench-figure identity.")
    e2e_tpsr_addon.add_include_arg(
        ap, " Its OOD rows come from augment_e2e_tpsr_caches.py.")
    ap.add_argument("--include-llmsr", action="store_true",
                    help="Overlay the single-seed LLMSR-Llama curve (lsr_transform) from "
                         "the score_llmsr_ood.py caches (results/llmsr_llama_ood_raw_"
                         "lsr_transform_noise<tau>.csv). One line per panel, no band. "
                         "--grid only.")
    ap.add_argument("--llmsr-backbones", nargs="+", default=None,
                    metavar="LABEL",
                    choices=list(llmsr_overlay.LLMSR_LABELS),
                    help="Which LLM-SR backbones --include-llmsr overlays. "
                         "Default: LLMSR-Llama only, which is what every "
                         "figure drew before this flag existed.")
    oneshot_overlay.add_include_arg(
        ap, " lsr_transform; one dashed line per arm per panel, with +/-1 seed-std bars "
            "for the multi-seed arms. "
            "--grid only.")
    ap.add_argument("--show", action="store_true")
    aifeynman_addon.add_group_arg(ap)
    phye2e_addon.add_group_arg(ap)
    pysr_addon.add_group_arg(ap)
    return ap


def parse_args(argv=None):
    """Parse `argv` (default sys.argv) and apply the addons' group-arg post-processing."""
    args = build_parser().parse_args(argv)
    aifeynman_addon.apply_group_arg(args)
    phye2e_addon.apply_group_arg(args)
    pysr_addon.apply_group_arg(args)
    return args


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    if args.gaps is None:
        args.gaps = ood_transform.default_gaps()
    gaps = sorted(set(args.gaps))

    if args.grid:
        _plot_grid_llmsr(args, gaps)
        return

    # -- single noise level ---------------------------------------------------
    if args.output is None:
        args.output = ("plots/llmsrbench_ood_vs_gap.png" if args.noise == 0.0
                       else f"plots/llmsrbench_ood_vs_gap_noise{args.noise:g}.png")

    acc, algs_sorted = _prepare_noise_llmsr(args.noise, args, gaps)
    if not acc:
        raise SystemExit(f"No method dirs found for split '{args.split}' at noise {args.noise:g}.")

    fig_h = max(6, len(algs_sorted) * 0.4 + 2)
    fig, ax_a = plt.subplots(1, 1, figsize=(12, fig_h))
    _draw_curves_llmsr(ax_a, acc, algs_sorted, gaps, args.r2_thr)

    # Compact boxed legend in the blank upper-right corner of the panel, matching
    # the grid figure (see _plot_grid_llmsr) and the ood_vs_gap plots.
    _h, _l = ax_a.get_legend_handles_labels()
    handles, labels = sort_legend(
        _h, display_labels(e2e_tpsr_addon.rename_legend(_l)))  # printed order
    ax_a.legend(handles, labels, loc="upper right",
                fontsize=10 * _FS, frameon=True, framealpha=0.9, ncol=1)

    fig.tight_layout()
    save_fig(fig, os.path.join(SCRIPT_DIR, args.output))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
