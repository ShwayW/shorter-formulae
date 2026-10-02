#!/usr/bin/env python3
"""
plot_ood_vs_gap.py -- OOD accuracy and timing vs distribution-shift gap.

Produces a two-subplot figure:
  Left:  avg time per formula (y, log-scale) vs gap (x) -- flat lines per algorithm
  Right: # formulas with OOD R^2 >= threshold (y) vs gap (x)

Usage:
    python plot_ood_vs_gap.py \
        --results results/eval_tf_89M_40.pkl.gz results/eval_tf_145M_40.pkl.gz \
        --output plots/ood_vs_gap.png
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import results_io
import tpsr_addon
import devncon_addon
import phye2e_addon
import pysr_addon
import llmsr_overlay
import oneshot_overlay
import ood_transform
from plot_io import (save_fig, GRID_LEGEND_KW, GRID_LEGEND_NCOL, bottom_legend, above_legend,
                     sort_legend, display_labels, font_scale, GRID_FIG_W_IN,
                     GRID_ROW_FIG_H_IN)
from plot_markers import method_marker, HOLLOW, CURVE_MS, CURVE_LW

from compare_srbench import (
    SRBENCH_GT,
    _BASELINE_COLOR,
    _FLOAT_COLOR,
    _resolve_artifact,
    compute_ood_r2,
    load_srbench_gt,
    load_transformer_results,
    load_tpsr_results,
)

# Font sizes below are written as the point size the text should have ON THE PAGE and
# multiplied by _FS -- the grid is 10 in wide and LaTeX shrinks it to \textwidth, so a
# bare "12" would reach the reader at under 7 pt.  See plot_io.font_scale.
_FS = font_scale(GRID_FIG_W_IN)

# Model name -> display label (number of parameters).
# m89float -> "89M-float": our own 89M model on the floating-point-constant grammar,
# evaluated from results/eval_tf_m89float.pkl.gz like the other models.
# 2026-08-17: checkpoint dirs renamed to <params>_<max tokens per formula>, so runs from
# now on are labelled 89M_40/145M_40 while every EXISTING artifact is labelled
# m89/m145.  Both spellings must map to the same display label.
# 145M_80 is the same 145M architecture trained with fml_len_range [1, 80] instead of
# [1, 40], so the length budget -- not the parameter count -- is what distinguishes it;
# "145M" alone would collide with 145M_40.
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

# Colour scheme: after remapping, "145M" is the default arm and "145M_unscaled" the
# --unscale arm.  NAMING, since the two are easy to invert: the transformer sees
# WHITENED inputs, so its formula is written in whitened variables; --unscale is the
# post-decode pass (eval_mymodels.apply_input_unscaling) that substitutes
# v_i -> (v_i - mu_i)/sigma_i to express it in raw-variable space.  So "_unscaled"
# names the arm the substitution was APPLIED to -- it is not the whitened-space curve.
# The default arm skips the substitution and evaluates the formula as decoded.
#   145M (m145)  : green  -- dark (#15803d) = input-unscaled (_unscaled), light (#4ade80) = raw
#   89M (m89) : orange -- dark (#c2410c) = input-unscaled (_unscaled), light (#fb923c) = raw
#   TPSR : teal   e2e: purple
def _tf_color(label):
    # PhyE2E first: its label ends in "(E2E)", so any substring test for the plain
    # "e2e" baseline would cross-fire.  is_phye2e is an exact/prefix match.
    if phye2e_addon.is_phye2e(label):
        return phye2e_addon.color(label)
    # PySR: exact/prefix match for the same reason -- a bare substring test on a name
    # this short would cross-fire against any future label containing it.
    if pysr_addon.is_pysr(label):
        return pysr_addon.color(label)
    # "<model> + TPSR" combined runs -- their own dark-shade colours (checked first,
    # since "89M+TPSR" also starts with "89M").
    if label in tpsr_addon.TPSR_COLORS:
        return tpsr_addon.TPSR_COLORS[label]
    # "<model> + D&C" combined runs
    if label in devncon_addon.DEVNCON_COLORS:
        return devncon_addon.DEVNCON_COLORS[label]
    # "89M-float" must be checked before the "89M" prefix (it also starts with "89M").
    if label == "89M-float":
        return _FLOAT_COLOR                                       # rose / deep pink (see compare_srbench)
    # The SimpliPy-canonicalised len-80 145M.  Lime rather than a third green shade:
    # it also starts with "145M", and green-400/green-700 are already spoken for by
    # 145M and 145M-len80, which two near-identical greens could not separate.
    if label == "145M-len80-simplipy":
        return "#65a30d"                                          # lime-600
    # 145M-len80 keeps the 145M family's green but takes the dark shade, which the
    # retired input-unscaled (_unscaled) arms used to own -- checked before the "145M"
    # prefix, which would otherwise hand it the same light green as 145M itself.
    if label == "145M-len80":
        return "#15803d"                                          # forest green (dark)
    # The 145M float-constant model: dark rose, pairing it with 89M-float's rose so the
    # two restrict_consts=0 arms read as a family.  Must precede the "145M" prefix test,
    # which would otherwise hand it plain 145M's light green.
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
        return "#15803d" if "unscaled" in label else "#4ade80"   # forest green (dark) / light green
    if label.startswith("89M"):
        return "#c2410c" if "unscaled" in label else "#fb923c"   # burnt orange (dark) / light orange
    if label == "e2e":
        return "#7c3aed"                                          # violet / purple
    if label == "TPSR":
        return "#0d9488"                                          # teal -- distinct from 89M-float (rose) and 145M (green)
    # Give the two headline SRBench baselines distinct colours so they are
    # tellable apart (both are dashed, so colour is the only cue).
    if label == "AIFeynman":
        return "#457b9d"                                          # steel blue
    if label == "Operon":
        return "#8c564b"                                          # brown
    # The three pretrained-transformer baselines.  They are absent from set1..set5
    # (see plot_transformer_methods.py for why), so these entries change no existing
    # figure -- they exist so the transformer-only figure does not draw three curves
    # in the same _BASELINE_COLOR, where colour is the only cue that separates them.
    if label == "nesymres":
        return "#0284c7"                                          # sky blue
    if label == "tf4sr":
        return "#b45309"                                          # dark amber
    if label == "symformer":
        return "#475569"                                          # slate
    return None  # any other SRBench baseline -> use _BASELINE_COLOR (steel blue)


def _remap_scale_labels(raw_labels):
    """raw (plain) -> canonical, input-unscaled (_unscale) -> _unscaled; then rename m145/m89 to param counts."""
    groups = {}
    for lbl in raw_labels:
        groups.setdefault(lbl.replace("_unscale", ""), set()).add(lbl)
    remap = {}
    for lbl in raw_labels:
        base   = lbl.replace("_unscale", "")
        grp    = groups[base]
        paired = (any("_unscale" in g for g in grp)
                  and any("_unscale" not in g for g in grp))
        if "_unscale" in lbl and paired:
            model, _, rest = base.partition("_")
            remapped = f"{model}_unscaled" + (f"_{rest}" if rest else "")
        else:
            remapped = base
        # Rename model prefix to parameter-count label.
        for src, dst in _MODEL_RENAME.items():
            remapped = remapped.replace(src, dst, 1)
        remap[lbl] = remapped
    return remap

_CURVE_LW = CURVE_LW   # one width for every curve, overlays included

DEFAULT_GAPS   = [0, 1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_R2_THR = 0.99
DEFAULT_NOISE  = 0.0


def _label_for_path(path: str) -> str:
    label = os.path.basename(path).replace("eval_tf_", "").replace(".pkl.gz", "")
    try:
        with gzip.open(path, "rb") as fh:
            raw = pickle.load(fh)
        label = raw.get("model_name", label)
    except Exception:
        pass
    return "ours" if label == "non_bpe" else label


def _seed_std(agg):
    """The per-algorithm seed-to-seed time spread carried on an aggregated frame.

    compare_srbench attaches `time_seed_std` as a CONSTANT column per algorithm (it is
    a spread across seeds, not across datasets), so any row reads it back.  0.0 for a
    single-run source, exactly as `complexity_seed_std` is."""
    if "time_seed_std" not in agg.columns or agg.empty:
        return 0.0
    v = float(agg["time_seed_std"].dropna().mean()) if agg["time_seed_std"].notna().any() else 0.0
    return v if np.isfinite(v) else 0.0


def collect_timing(result_paths, srbench_agg, r2_thr, tpsr_path=None, noise=0.0):
    """Return {algorithm: (mean_seconds, std_seconds)} from all sources.

    The mean is the average over datasets of each dataset's mean-across-seeds time.
    The std is the SEED-TO-SEED spread of the per-seed mean time (compare_srbench's
    _seed_std_of_mean), the same quantity the complexity bars report -- not the spread
    across datasets, which measures problem difficulty rather than run-to-run
    variability and stayed wide however reproducible a method was.  A single-run
    source (SRBench's published TPSR csv) has no seed axis and reports 0.0.
    """
    timing = {}  # {alg: (mean, std)}

    for path in result_paths:
        if not os.path.exists(path):
            continue
        agg = load_transformer_results(path, r2_thr, with_complexity=False)
        if agg is None or "time_mean" not in agg.columns:
            continue
        times = agg["time_mean"].dropna()
        timing[_label_for_path(path)] = (float(times.mean()), _seed_std(agg))

    # e2e baseline (merged E2E group artifact, noise-matched)
    e2e_csv = _resolve_artifact(
        results_io.e2e_srbench_pkl(os.path.join(SCRIPT_DIR, "results"), noise))
    if os.path.exists(e2e_csv):
        e2e_agg = load_transformer_results(e2e_csv, r2_thr, with_complexity=False)
        if e2e_agg is not None and "time_mean" in e2e_agg.columns:
            times = e2e_agg["time_mean"].dropna()
            timing["e2e"] = (float(times.mean()), _seed_std(e2e_agg))

    # SRBench baselines: mean over datasets, std over their random_state seeds
    if "time_mean" in srbench_agg.columns:
        for alg, grp in srbench_agg.groupby("algorithm"):
            times = grp["time_mean"].dropna()
            if not times.empty:
                timing[alg] = (float(times.mean()), _seed_std(grp))

    # TPSR
    if tpsr_path and os.path.exists(tpsr_path):
        tpsr_agg = load_tpsr_results(tpsr_path, r2_thr, noise=noise)
        if tpsr_agg is not None and "time_mean" in tpsr_agg.columns:
            times = tpsr_agg["time_mean"].dropna()
            timing["TPSR"] = (float(times.mean()), _seed_std(tpsr_agg))

    return timing


def collect_ood_accuracy(result_paths, gaps, r2_thr, tpsr_path=None, noise=0.0):
    """Return DataFrame[gap, algorithm, seed, n_correct, n_total, solve_rate].

    One row per (gap, algorithm, seed): that seed's OOD solve rate, counted over
    the full dataset universe at that gap (datasets the seed didn't solve / wasn't
    evaluated on count as unsolved for that seed).  Aggregating these across seeds
    in main() gives the mean curve and the seed-to-seed spread shaded on the plot.

    Every gap (gap=0 included) uses the synthetic datasets/feynman_ood_g{gap}.pkl.gz.
    Within a seed the per-dataset best formula is thresholded, matching the
    aggregation of compare_srbench.py's srbench_comparison_g{gap}.png.  The per-seed
    models (our transformers and e2e) contribute one row per (dataset, seed) so their
    band spans the 10-seed spread; TPSR (authors' single run) has no seed axis, so it
    contributes one pseudo-seed (-1) and its band collapses to the line.
    """
    raw = ood_raw_table(result_paths, gaps, tpsr_path, noise)
    return accuracy_from_raw(raw, r2_thr)


def ood_raw_table(result_paths, gaps, tpsr_path=None, noise=0.0, phye2e_path=None,
                  pysr_path=None):
    """Concatenated per-(gap, algorithm, dataset, seed) OOD R^2 -- the expensive part.

    This is where the time goes: every method's predicted formula (ours x 10 seeds,
    the SRBench baselines x their seeds, e2e, TPSR) is re-evaluated on each OOD point
    cloud at each gap.  The result is threshold-independent, so main() caches it to
    results/ood_raw_noise<tau>.csv and re-thresholds cheaply on later re-plots.

    """
    tf_file_labels = [
        (_label_for_path(p), p) for p in result_paths if os.path.exists(p)
    ]
    tpsr = tpsr_path if (tpsr_path and os.path.exists(tpsr_path)) else None

    frames = []
    for g in gaps:
        print(f"[gap={g:>3}] computing OOD R^2...", flush=True)
        ood_path = os.path.join(SCRIPT_DIR, ood_transform.feynman_ood_path(g))
        if not os.path.exists(ood_path):
            print(f"[warn] Missing OOD dataset: {ood_path}")
            continue
        ood_df, _ = compute_ood_r2(
            ood_path, tf_file_labels, tpsr_path=tpsr, phye2e_path=phye2e_path,
            pysr_path=pysr_path,
            include_gt=False, noise=noise,
            all_seeds=True,   # keep per-seed rows so the shaded band = seed variance
        )
        if ood_df.empty:
            continue
        ood_df = ood_df.copy()
        ood_df["gap"] = g
        frames.append(ood_df)
    if not frames:
        return pd.DataFrame(columns=["gap", "algorithm", "dataset", "seed", "ood_r2"])
    return pd.concat(frames, ignore_index=True)


def accuracy_from_raw(raw_df, r2_thr, monotone=True):
    """Threshold a raw OOD R^2 table into per-(gap, algorithm, seed) solve rates.

    Common denominator: every seed's rate is out of the full dataset universe at
    that gap (datasets it didn't solve / wasn't evaluated on count as unsolved).

    monotone=True applies ood_transform.monotone_gap first, so a dataset that has
    already failed at a smaller gap stays failed at every larger one (see that
    function for why the raw per-gap scores are not monotone on their own).
    """
    if monotone:
        # Order matters: restrict FIRST so the prefix-AND runs over the same problem set
        # it will be counted on, then monotonise.
        raw_df = ood_transform.restrict_to_common(raw_df)
        raw_df = ood_transform.monotone_gap(raw_df, r2_thr)
    records = []
    for g, gap_df in raw_df.groupby("gap"):
        n_universe = int(gap_df["dataset"].nunique())
        for (alg, seed), grp in gap_df.groupby(["algorithm", "seed"]):
            n_correct = int((grp.groupby("dataset")["ood_r2"].max() >= r2_thr).sum())
            records.append({
                "gap":        g,
                "algorithm":  alg,
                "seed":       int(seed),
                "n_correct":  n_correct,
                "n_total":    n_universe,
                "solve_rate": n_correct / n_universe if n_universe > 0 else float("nan"),
            })
    if not records:
        return pd.DataFrame(
            columns=["gap", "algorithm", "seed", "n_correct", "n_total", "solve_rate"]
        )
    return pd.DataFrame(records)


def _prepare_noise(noise, args, gaps, results_root):
    """Per-noise curve data: (agg_acc[gap,algorithm,solve_rate,band_lo,band_hi],
    algs_sorted, tf_label_set).  Reads/writes the OOD R^2 cache like main() does."""
    if args.results is None:
        result_paths = results_io.list_base_srbench_pkls(results_root, noise)
    else:
        result_paths = [os.path.join(SCRIPT_DIR, p) for p in args.results]
    # Opt-in: the "<model> + TPSR" combined runs (results/mymodels_tpsr/).
    if getattr(args, "include_tpsr_combo", False):
        result_paths = list(result_paths) + tpsr_addon.srbench_tpsr_pkls(results_root, noise)
    # Opt-in: the "<model> + D&C" combined runs (results/mymodels_devncon/).
    if getattr(args, "include_devncon", False):
        result_paths = list(result_paths) + devncon_addon.srbench_devncon_pkls(results_root, noise)
    tpsr_path = os.path.join(SCRIPT_DIR, args.tpsr_results)
    # Opt-in: PhyE2E (results/phye2e/).  Its own artifact, not a transformer pkl, so it
    # is threaded to compute_ood_r2 separately rather than appended to result_paths.
    phye2e_arms = (phye2e_addon.srbench_pkls_for_groups(
                       results_root, noise, phye2e_addon.active_groups(args))
                   if getattr(args, "include_phye2e", False) else {})
    # compute_ood_r2 takes ONE artifact; the cache already holds every arm's rows, so
    # the live path covers the first arm and --reuse-ood covers the rest.
    phye2e_path = next(iter(phye2e_arms.values()), None)
    # Opt-in: PySR (results/pysr/).  Same shape as the PhyE2E block above -- its own
    # artifact rather than a transformer pkl, so it is threaded to compute_ood_r2
    # separately rather than appended to result_paths.
    pysr_arms = (pysr_addon.srbench_pkls_for_groups(
                     results_root, noise, pysr_addon.active_groups(args))
                 if getattr(args, "include_pysr", False) else {})
    pysr_path = next(iter(pysr_arms.values()), None)

    ood_cache = os.path.join(results_root,
                             ood_transform.srbench_cache_name(noise))
    if args.reuse_ood and os.path.exists(ood_cache):
        raw = pd.read_csv(ood_cache)
        missing = [g for g in gaps if g not in set(raw["gap"].unique())]
        if missing:
            raw = pd.concat([raw, ood_raw_table(result_paths, missing, tpsr_path,
                                                noise=noise, phye2e_path=phye2e_path,
                                                pysr_path=pysr_path)],
                            ignore_index=True)
            raw.to_csv(ood_cache, index=False)
        print(f"[ood] reused OOD R^2 from {os.path.basename(ood_cache)} (noise {noise:g})")
    else:
        raw = ood_raw_table(result_paths, gaps, tpsr_path, noise=noise,
                            phye2e_path=phye2e_path, pysr_path=pysr_path)
        if not raw.empty:
            os.makedirs(os.path.dirname(ood_cache), exist_ok=True)
            raw.to_csv(ood_cache, index=False)
            print(f"[ood] wrote OOD R^2 cache -> {ood_cache}")
    acc_df = accuracy_from_raw(raw[raw["gap"].isin(gaps)], args.r2_thr,
                               monotone=not args.no_monotone_gap)

    raw_tf_labels = [_label_for_path(p) for p in result_paths if os.path.exists(p)]
    scale_remap = _remap_scale_labels(raw_tf_labels)
    # "<model> + TPSR" combined runs get their display label from their addon
    # (the generic remap would only turn 'm89_tpsr' into '89M_tpsr').
    def _combo_display(k, v):
        if tpsr_addon.is_tpsr_combo(k):
            return tpsr_addon.display_label(k)
        return v
    scale_remap = {k: _combo_display(k, v) for k, v in scale_remap.items()}
    if not acc_df.empty:
        acc_df["algorithm"] = acc_df["algorithm"].map(lambda x: scale_remap.get(x, x))
    # When not opted in, drop any combo rows the cache may carry, so every other
    # figure is byte-for-byte unchanged.
    if not getattr(args, "include_tpsr_combo", False) and not acc_df.empty:
        acc_df = acc_df[~acc_df["algorithm"].map(tpsr_addon.is_tpsr_combo)]
    if args.exclude:
        acc_df = acc_df[~acc_df["algorithm"].isin(args.exclude)]

    tf_labels = list(scale_remap.values())
    if os.path.exists(tpsr_path):
        tf_labels.append("TPSR")
    if os.path.exists(_resolve_artifact(results_io.e2e_srbench_pkl(results_root, noise))):
        tf_labels.append("e2e")
    tf_labels.extend(phye2e_arms)
    tf_labels.extend(pysr_arms)
    tf_label_set = set(tf_labels)

    agg_acc = pd.DataFrame()
    if not acc_df.empty:
        var_records = []
        for (alg, gap), grp in acc_df.groupby(["algorithm", "gap"]):
            rates = grp["solve_rate"].dropna().to_numpy()
            if rates.size == 0:
                continue
            mean = float(np.mean(rates))
            std  = float(np.std(rates))
            var_records.append({"algorithm": alg, "gap": gap, "solve_rate": mean,
                                "band_lo": max(0.0, mean - std),
                                "band_hi": min(1.0, mean + std)})
        agg_acc = pd.DataFrame(var_records)

    gap0_mean = {}
    if not agg_acc.empty:
        gap0_mean = agg_acc[agg_acc["gap"] == 0].set_index("algorithm")["solve_rate"].to_dict()
    algs = agg_acc["algorithm"].unique().tolist() if not agg_acc.empty else []
    algs_sorted = sorted(algs, key=lambda a: -gap0_mean.get(a, 0))
    # --only: keep just the listed methods (e.g. a camera-ready subset).
    if getattr(args, "only", None):
        keep = set(args.only)
        algs_sorted = [a for a in algs_sorted if a in keep]
    return agg_acc, algs_sorted, tf_label_set


def _yerr(ys, lo, hi):
    """Asymmetric yerr for ax.errorbar from a (mean, lo, hi) band, clipped nonneg."""
    ys, lo, hi = np.asarray(ys, float), np.asarray(lo, float), np.asarray(hi, float)
    return np.vstack([np.clip(ys - lo, 0, None), np.clip(hi - ys, 0, None)])


def _draw_curves(ax, agg_acc, algs_sorted, gaps, tf_label_set, r2_thr,
                 *, direct_labels, xlabel=True, ylabel=True):
    """Draw the OOD-accuracy-vs-gap curves (+ seed-std error bars) onto `ax`.

    The +/-1 seed-std spread is drawn as a per-point error bar rather than the
    translucent band this used to shade: with a dozen methods on one panel the bands
    overlapped into a wash that hid the curves themselves.

    direct_labels=True writes each method's name past its last point (single-panel
    style); False labels the lines instead (label=) for a shared figure legend."""
    def _color(alg):
        return _tf_color(alg) or _BASELINE_COLOR

    def _ls(alg):
        if phye2e_addon.is_phye2e(alg):
            return phye2e_addon.linestyle(alg)   # we RUN PhyE2E; never "published"
        if pysr_addon.is_pysr(alg):
            return pysr_addon.linestyle(alg)     # ours-run too: PySR has no published
                                                 # SRBench rows at all (see pysr_addon)
        # The "<model> + D&C" combos are OUR checkpoints under a different decoder, not
        # a published SRBench baseline -- they are absent from tf_label_set only because
        # they reach the cache through augment_devncon_caches.py, so name them here or
        # the dashed-baseline rule below would steal 145M+D&C's solid line.
        if devncon_addon.is_devncon_combo(alg):
            return "-"
        if alg not in tf_label_set:
            return "--"   # SRBench baseline
        return "-"

    gap_pos = {g: i for i, g in enumerate(gaps)}
    curve_ends = []
    for alg in algs_sorted:
        sub = agg_acc[agg_acc["algorithm"] == alg].sort_values("gap")
        if sub.empty:
            continue
        xs  = np.array([gap_pos[g] for g in sub["gap"].values])
        ys  = sub["solve_rate"].values
        # A missing band edge (single-seed arm) collapses onto the mean -> zero-length
        # bar, not a bar down to 0 as fillna(0) would have given.
        lo  = sub["band_lo"].fillna(sub["solve_rate"]).values.clip(0)
        hi  = sub["band_hi"].fillna(sub["solve_rate"]).values
        col = _color(alg)
        ax.plot(xs, ys, color=col, lw=_CURVE_LW, ls=_ls(alg),
                marker=method_marker(alg), markersize=CURVE_MS, **HOLLOW,
                label=(None if direct_labels else alg))
        ax.errorbar(xs, ys, yerr=_yerr(ys, lo, hi), fmt="none", ecolor=col,
                    elinewidth=1.0, capsize=2.5, capthick=1.0, alpha=0.85, zorder=1.8)
        curve_ends.append((float(xs[-1]), float(ys[-1]), alg, col))

    if xlabel:
        ax.set_xlabel("k", fontsize=12 * _FS)
    if ylabel:
        ax.set_ylabel(f"Fraction with $R^2 \\geq$ {r2_thr}", fontsize=12 * _FS)
    ax.set_xticks(range(len(gaps)))
    ax.set_xticklabels(["0" if g == 0 else str(g) for g in gaps])
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11 * _FS)

    if direct_labels and curve_ends:
        x_end = float(len(gaps) - 1)
        x_lbl = x_end + 0.3
        ax.set_xlim(right=x_end + 1.2)
        curve_ends.sort(key=lambda t: t[1])
        label_ys = [t[1] for t in curve_ends]
        MIN_GAP = 0.022
        for i in range(1, len(label_ys)):
            if label_ys[i] < label_ys[i - 1] + MIN_GAP:
                label_ys[i] = label_ys[i - 1] + MIN_GAP
        for i in range(len(label_ys) - 2, -1, -1):
            if label_ys[i + 1] - label_ys[i] < MIN_GAP:
                label_ys[i] = label_ys[i + 1] - MIN_GAP
        for (x_last, y_last, alg, col), y_lbl in zip(curve_ends, label_ys):
            ax.annotate(alg, xy=(x_last, y_last), xytext=(x_lbl, y_lbl),
                        color=col, fontsize=10 * _FS, va="center", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.7, relpos=(0, 0.5))
                        if abs(y_lbl - y_last) > 0.01 else None)


# Legend TEXT only remap (the authors' standalone TPSR is E2E decoded with the TPSR
# search); the algorithm identity stays "TPSR" so colour + the --only filter are unchanged.
GRID_LEGEND_RENAME = {"TPSR": "e2e+TPSR"}
GRID_NOISES = [0.0, 0.001, 0.01, 0.1]


def draw_grid_row(axes, args, gaps, results_root, titles=True):
    """Draw the four SRBench noise panels into `axes` (an iterable of 4 axes).

    Split out of _plot_grid so the combined SRBench-over-LSR-Transform figure
    (plot_combined_curves_grid.py) can put this row on top of the LLM-SRBench one and
    share a single legend.  Returns {raw algorithm label: legend handle}.
    """
    axes = list(axes)          # iterated twice: drawn, then read back for the legend
    for ax, noise in zip(axes, GRID_NOISES):
        agg_acc, algs_sorted, tf_label_set = _prepare_noise(noise, args, gaps, results_root)
        _draw_curves(ax, agg_acc, algs_sorted, gaps, tf_label_set, args.r2_thr,
                     direct_labels=False, xlabel=False, ylabel=False)
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
                llmsr_overlay.draw_overlay(ax, "feynman", noise, gaps, args.r2_thr,
                                           monotone=not args.no_monotone_gap,
                                           as_percent=False, label=_lbl)
        # Opt-in one-shot LLM arms (from score_oneshot_ood.py caches): one dashed line
        # per arm, with +/-1 seed-std error bars for the arms that ran more than one seed
        # (Llama-3.3-70B did; Gemini and Llama-3.1-8B are seed 0, so their bars vanish).
        # --exclude also thins the one-shot family here: the arms are appended after
        # the acc_df filtering, so without this they would survive an --exclude that
        # names them (make_requested_plots.sh drops OneShot-Llama / -Llama70B from the
        # curve grids, keeping the Gemini arm).
        if getattr(args, "include_oneshot", False):
            oneshot_overlay.draw_overlay(ax, "feynman", noise, gaps, args.r2_thr,
                                         monotone=not args.no_monotone_gap, as_percent=False,
                                         labels=oneshot_overlay.visible_labels(
                                             getattr(args, "exclude", None)))
        if titles:
            ax.set_title(f"noise = {noise:g}", fontsize=13 * _FS)

    # Union of handles across the row (same method set at every noise).
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    return seen


def _plot_grid(args, gaps, results_root):
    """Single row of four OOD-accuracy-vs-gap panels, one per noise level, shared legend."""
    # ONE ROW of four panels (2026-09-21; was a 2x2 grid).  sharey keeps the y tick
    # labels on the leftmost panel only, which is what makes four panels fit across a
    # 10 in canvas; the height is just one row of panels plus the measured legend strip.
    fig, axes = plt.subplots(1, 4, figsize=(GRID_FIG_W_IN, GRID_ROW_FIG_H_IN),
                             sharex=True, sharey=True)
    seen = draw_grid_row(list(axes.flat), args, gaps, results_root)
    _LEGEND_RENAME = GRID_LEGEND_RENAME

    fig.supylabel(f"Fraction with $R^2 \\geq$ {args.r2_thr}", fontsize=12 * _FS)

    # Alphabetical by the PRINTED label (so "TPSR" sorts as "e2e+TPSR"), not by the
    # solve-rate plot order, which reshuffled the legend on every figure.
    _handles, _labels = sort_legend(
        list(seen.values()),
        display_labels([_LEGEND_RENAME.get(l, l) for l in seen.keys()]))
    # The legend is a STRIP UNDER the panel row, spanning the full canvas width, with the
    # shared x-label sitting just above it; tight_layout shrinks the axes to the measured
    # height that is left, so the box can never land on data however many arms are added.
    # bottom_legend drops a column at a time until the strip fits the canvas, so it
    # cannot overhang and inflate the saved bbox the way the old fixed two-row strip did.
    # Same treatment as the LLM-SRBench grid (plot_llmsrbench_ood_vs_gap._plot_grid_llmsr);
    # the two figures sit side by side in the paper and should read the same way.
    _leg, frac = bottom_legend(fig, _handles, _labels, ncol=GRID_LEGEND_NCOL,
                               **GRID_LEGEND_KW)
    fig.supxlabel("Extrapolation Constant (k)", fontsize=12 * _FS, y=above_legend(frac, fig))
    fig.tight_layout(rect=(0, frac, 1, 1))

    out = os.path.join(SCRIPT_DIR, args.output or "plots/ood_vs_gap_grid.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_fig(fig, out)
    if args.show:
        plt.show()


def _plot_overlay(args, gaps, results_root):
    """Single panel: our models only, all four noise levels overlaid on one axes.
    Colour = model (matching every other plot), line style = noise level.  No
    shaded bands -- with 4 noises overlaid they would be an unreadable tangle."""
    from matplotlib.lines import Line2D
    noises = [0.0, 0.001, 0.01, 0.1]
    ls_map = {0.0: "-", 0.001: "--", 0.01: ":", 0.1: "-."}

    def _is_ours(a):
        return a.startswith("145M") or a.startswith("89M") or a == "e2e"

    fig, ax = plt.subplots(figsize=(12, 8))
    gap_pos = {g: i for i, g in enumerate(gaps)}
    models_seen = {}                          # model -> colour, in plot order
    for noise in noises:
        agg_acc, algs_sorted, _tf_label_set = _prepare_noise(noise, args, gaps, results_root)
        if agg_acc.empty:
            continue
        for alg in algs_sorted:
            if not _is_ours(alg):
                continue
            sub = agg_acc[agg_acc["algorithm"] == alg].sort_values("gap")
            if sub.empty:
                continue
            xs = [gap_pos[g] for g in sub["gap"].values]
            col = _tf_color(alg) or _BASELINE_COLOR
            ax.plot(xs, sub["solve_rate"].values, color=col, lw=2, ls=ls_map[noise],
                    marker="o", markersize=4, **HOLLOW)
            # +/-1 seed-std band (same variance the grid/per-noise plots shade).
            ax.fill_between(xs, sub["band_lo"].fillna(0).clip(0).values,
                            sub["band_hi"].fillna(0).values,
                            color=col, alpha=0.12, linewidth=0)
            models_seen.setdefault(alg, col)

    ax.set_xlabel("OOD gap", fontsize=12)
    ax.set_ylabel(f"Fraction with $R^2 \\geq$ {args.r2_thr}", fontsize=12)
    ax.set_xticks(range(len(gaps)))
    ax.set_xticklabels(["no gap" if g == 0 else str(g) for g in gaps])
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    # Two legends: colour -> model, line style -> noise.
    model_h = [Line2D([0], [0], color=c, lw=2, label=m) for m, c in models_seen.items()]
    noise_h = [Line2D([0], [0], color="0.3", lw=2, ls=ls_map[n],
                      label=("0 (clean)" if n == 0 else f"{n:g}")) for n in noises]
    leg1 = ax.legend(handles=model_h, title="Model", loc="upper right",
                     fontsize=10, title_fontsize=10, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=noise_h, title="Target noise", loc="lower left",
              fontsize=10, title_fontsize=10, frameon=True)

    fig.tight_layout()
    save_fig(fig, os.path.join(SCRIPT_DIR, args.output or "plots/ood_vs_gap_overlay.png"))
    if args.show:
        plt.show()


def build_parser():
    """The CLI, split out of main() so plot_combined_curves_grid.py can parse exactly the
    same SRBench flags for the top row of the combined figure."""
    ap = argparse.ArgumentParser(
        description="Plot OOD accuracy and avg time vs distribution-shift gap."
    )
    ap.add_argument(
        "--results", nargs="+", default=None,
        help="Transformer .pkl.gz result files. Default: auto-discover the merged "
             "eval_tf_*.pkl.gz artifacts under results/mymodels/noise_<tau>/ for the "
             "chosen --noise.",
    )
    ap.add_argument("--gaps", nargs="+", type=int, default=None,
                    help="x-axis gap values. Default: 0 1 2 4 8 16 32 64 128.")
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    ap.add_argument("--output", default=None,
                    help="Output PNG. Default: plots/ood_vs_gap[_noise<tau>].png")
    ap.add_argument("--noise", type=float, default=0.0, choices=[0.0, 0.001, 0.01, 0.1],
                    help="Target-noise level: reads transformer results from "
                         "results/noise_<tau>/, matches SRBench/TPSR baselines at tau, "
                         "and tags the output filename. Default: 0.0.")
    ap.add_argument("--tpsr-results",
                    default="TPSR/srbench_results/feynman_tpsr_l0.1_allnoise.csv")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--no-monotone-gap", action="store_true",
                    help="Score each gap independently, allowing a formula that failed at a smaller gap to count as solved at a larger one. Off by default: 'solved at gap k' normally means 'solved at every gap <= k' (ood_transform.monotone_gap).")
    ap.add_argument("--reuse-ood", action="store_true",
                    help="Reuse the cached OOD R^2 table (results/ood_raw_noise<tau>.csv) "
                         "instead of recomputing -- instant re-plots (layout / threshold "
                         "tweaks). Gaps missing from the cache are computed and appended. "
                         "The cache is always (re)written, so the first run enables later "
                         "--reuse-ood runs.")
    ap.add_argument("--grid", action="store_true",
                    help="Combine all four noise levels into one 2x2 figure "
                         "(one subplot per noise) with a shared legend, instead of a "
                         "single-noise plot. Default output: plots/ood_vs_gap_grid.png.")
    ap.add_argument("--overlay", action="store_true",
                    help="Single panel overlaying all four noise levels for our models "
                         "only (colour = model, line style = noise). "
                         "Default output: plots/ood_vs_gap_overlay.png.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Keep only these method display labels (e.g. a camera-ready "
                         "subset: --only 145M 89M 89M-float e2e TPSR AIFeynman Operon).")
    ap.add_argument("--include-tpsr-combo", action="store_true",
                    help="Also draw the '<model> + TPSR' combined runs from "
                         "results/mymodels_tpsr/ (89M+TPSR, 145M+TPSR). Off by "
                         "default so every other figure is unchanged; their OOD rows "
                         "come from augment_tpsr_caches.py.")
    ap.add_argument("--include-devncon", action="store_true",
                    help="Also draw the '<model> + D&C' combined runs from "
                         "results/mymodels_devncon/ (89M+D&C, 145M+D&C). Off by "
                         "default.")
    ap.add_argument("--include-phye2e", action="store_true",
                    help="Also draw PhyE2E from results/phye2e/ (the authors' 'e2e' "
                         "ablation decoded with beam search -- NOT their full "
                         "D&C+MCTS pipeline, hence the '(E2E)' in the label). Off by "
                         "default.")
    ap.add_argument("--include-pysr", action="store_true",
                    help="Also draw PySR from results/pysr/ (eval_pysr.py). Off by "
                         "default. PySR is absent from SRBench's published feather, so "
                         "this curve is entirely our own run.")
    ap.add_argument("--include-llmsr", action="store_true",
                    help="Overlay the single-seed LLMSR-Llama curve (Feynman) from the "
                         "score_llmsr_ood.py caches (results/llmsr_llama_ood_raw_feynman_"
                         "noise<tau>.csv). One line per panel, no band (single seed). "
                         "--grid only.")
    ap.add_argument("--llmsr-backbones", nargs="+", default=None,
                    metavar="LABEL",
                    choices=list(llmsr_overlay.LLMSR_LABELS),
                    help="Which LLM-SR backbones --include-llmsr overlays. "
                         "Default: LLMSR-Llama only, which is what every "
                         "figure drew before this flag existed.")
    oneshot_overlay.add_include_arg(
        ap, " Feynman; one dashed line per arm per panel, no band (single seed). "
            "--grid only.")
    ap.add_argument("--show", action="store_true")
    phye2e_addon.add_group_arg(ap)
    pysr_addon.add_group_arg(ap)
    return ap


def parse_args(argv=None):
    """Parse `argv` (default sys.argv) and apply the addons' group-arg post-processing."""
    args = build_parser().parse_args(argv)
    phye2e_addon.apply_group_arg(args)
    pysr_addon.apply_group_arg(args)
    return args


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    _RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")
    if args.gaps is None:
        args.gaps = ood_transform.default_gaps()
    gaps = sorted(set(args.gaps))

    if args.grid:
        _plot_grid(args, gaps, _RESULTS_ROOT)
        return
    if args.overlay:
        _plot_overlay(args, gaps, _RESULTS_ROOT)
        return

    # -- single noise level ---------------------------------------------------
    if args.noise != 0.0 and args.output is None:
        args.output = f"plots/ood_vs_gap_noise{args.noise:g}.png"
    if args.output is None:
        args.output = "plots/ood_vs_gap.png"

    agg_acc, algs_sorted, tf_label_set = _prepare_noise(args.noise, args, gaps, _RESULTS_ROOT)

    fig_h = max(7, len(algs_sorted) * 0.4 + 2)
    fig, ax_a = plt.subplots(1, 1, figsize=(12, fig_h))
    _draw_curves(ax_a, agg_acc, algs_sorted, gaps, tf_label_set, args.r2_thr,
                 direct_labels=True)

    fig.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_fig(fig, out_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
