#!/usr/bin/env python3
"""
plot_llmsrbench_curves_best_seed.py -- LLM-SRBench OOD accuracy vs gap at noise 0,
comparing e2e / 89M / 145M against the LLMSR backbones (Gemini and Llama-3.1-8B), where
each method is drawn as its SINGLE BEST SEED (the seed with the highest gap-0 solve rate).

Each LLMSR backbone is a single run (one seed), so scoring every other method by its best
seed makes this a one-seed-vs-one-seed comparison rather than mean-vs-single.  There is no
shaded band -- each curve is one seed.  Every method attempted all 111 LLM-SRBench
equations, so the solve rate is out of the full 111-equation universe (no common-subset
restriction, unlike SRBench/Feynman).

Reads the per-(gap, method, seed, equation) OOD R^2 caches (all at noise 0):
  * results/llmsr_ood_raw_noise0.csv                      -- e2e / 89M / 145M / 89M-float
        (written by plot_llmsrbench_ood_vs_gap.py --noise 0)
  * results/llmsr_gemini_ood_raw_lsr_transform_noise0.csv -- LLMSR-Gemini (param-refit)
  * results/llmsr_llama_ood_raw_lsr_transform_noise0.csv  -- LLMSR-Llama  (param-refit)
        (both written by score_llmsr_ood.py; a missing backbone cache is skipped)

Usage:
    python plot_llmsrbench_curves_best_seed.py
    PLOT_FORMAT=pdf python plot_llmsrbench_curves_best_seed.py \
        --output plots/llmsrbench_curves_noise0_bestseed.pdf
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from plot_io import save_fig, sort_legend, display_labels
from plot_markers import method_marker, HOLLOW, CURVE_MS, CURVE_LW
import aifeynman_addon
import phye2e_addon
import pysr_addon
import e2e_tpsr_addon
import devncon_addon
import oneshot_overlay
import ood_transform
from plot_llmsrbench_ood_vs_gap import (
    _color as _tf_color, _linestyle, DEFAULT_GAPS, DEFAULT_R2_THR,
)

# e2e / 89M / 145M / 89M-float (already display-labelled by the writer):
_BASE_CACHE   = "llmsr_ood_raw_noise0.csv"

# LLMSR backbones overlaid on this figure.  display label -> attributes:
#   cache    : score_llmsr_ood.py OOD-R^2 CSV under the results root
#   src      : the 'algorithm' label inside that CSV (mapped to the display label)
#   color    : curve colour (Gemini black, Llama azure -- matches plot_ood_common.py;
#              Llama was rose #e11d48, which is now 145M+D&C's red -- see devncon_addon)
#   artifact : the run's results.pkl.gz (relative to the results root) -- used by set 6
_LLMSR_BACKBONES = {
    "LLMSR-Gemini": {
        "backbone": "gemini",
        "cache":    "llmsr_gemini_ood_raw_lsr_transform_noise0.csv",
        "src":      "llmsr-gemini",
        "color":    "#111111",
        "artifact": "llmsr/noise_0/llmsrbench/llmsr-gemini35flash_lsr_transform/results.pkl.gz",
    },
    "LLMSR-Llama": {
        "backbone": "llama",
        "cache":    "llmsr_llama_ood_raw_lsr_transform_noise0.csv",
        "src":      "llmsr-llama",
        "color":    "#0369a1",
        "artifact": "llmsr_llama/noise_0/llmsrbench/llmsr-llama31-8b-vllm_lsr_transform/results.pkl.gz",
    },
}
LLMSR_LABELS = tuple(_LLMSR_BACKBONES)

# The one-shot LLM baselines (ONE LLM call per problem, eval_oneshot.py), whose OOD rows
# live in their own score_oneshot_ood.py caches -- the same per-backbone arrangement as
# the LLMSR runs above, so load_combined_raw folds them in the same way.  Opt-in
# (--include-oneshot), so DEFAULT_METHODS is unchanged and the default figure is
# byte-identical to before.  Identity (colour, dashed line, artifact path) is
# oneshot_overlay's, shared with the sets 1-4 overlays.
ONESHOT_LABELS = oneshot_overlay.LABELS

# The "<model> + TPSR" combined runs (our checkpoints decoded with TPSR search); their
# per-seed OOD rows already live in the base cache (appended by augment_tpsr_caches.py),
# and their set-6 artifacts are wired in via tpsr_addon.
TPSR_COMBO_LABELS = ("89M+TPSR", "145M+TPSR")

# The "<model> + D&C" combined runs (our checkpoints decoded with the DEVNCON
# divide-and-conquer search); same wiring as the TPSR combos above -- their per-seed
# OOD rows reach the base cache through the plot scripts' non---reuse-ood path, and their
# set-6 artifacts come from devncon_addon.llmsr_devncon_methods.
DEVNCON_COMBO_LABELS = ("89M+D&C", "145M+D&C")

# The AI Feynman 2.0 baseline (the original authors' code).
# Its per-seed OOD rows reach the base cache via augment_aifeynman_caches.py, exactly
# like the TPSR combos.  Opt-in only (--include-aifeynman), so DEFAULT_METHODS is
# unchanged and every existing figure renders identically.
AIFEYNMAN_LABEL = aifeynman_addon.LABEL

# The PhyE2E baseline (Ying et al. 2025), same opt-in wiring as AI Feynman: its per-seed
# OOD rows reach the base cache via augment_phye2e_caches.py, and --include-phye2e
# appends the label to --only so the default figure is unchanged.
PHYE2E_LABEL = phye2e_addon.LABEL

# The PySR baseline (Cranmer 2023), same opt-in wiring: its per-seed OOD rows reach the
# base cache via augment_pysr_caches.py, and --include-pysr appends the label to --only
# so the default figure is unchanged.
PYSR_LABEL = pysr_addon.LABEL

# The standalone TPSR baseline (the authors' MCTS decoding over the E2E transformer),
# legend text "e2e+TPSR".  Same opt-in wiring as AI Feynman / PhyE2E: its per-seed OOD
# rows reach the base cache via augment_e2e_tpsr_caches.py, and --include-e2e-tpsr
# appends the label to --only so the default figure is unchanged.
E2E_TPSR_LABEL = e2e_tpsr_addon.LABEL

# The methods this figure compares, in default legend order.
# "145M-len80-simplipy" is a base arm like 145M/89M, not an opt-in addon, so it belongs
# in the default list rather than behind an --include-* flag: sets 1-4 reach it through
# SRBENCH_KEEP / their --exclude blacklists, and this is the only figure that filters on
# DEFAULT_METHODS instead.  Callers that do not want it pass --exclude.
# "145M-len80-float" added 2026-09-15: set 5 draws from this list rather than from
# make_requested_plots.sh's SRBENCH_KEEP whitelist, so an arm missing here is absent from set 5
# even when sets 1-4 show it.  ("145M-len80-simplipy" and "89M" stay listed but are
# filtered out again by that script's DROP_ARMS --exclude.)
DEFAULT_METHODS = ["145M-len80-float", "145M-len80-float_ftnoise_48h",
                   "145M-len80-float_ftnoise_48h_unscaled",
                   "145M-len80-float_ftnoise_48h+D&C", "145M-len80-float_ftnoise_48h+TPSR",
                   "145M-len80-simplipy", "89M", "e2e",
                   *TPSR_COMBO_LABELS, *DEVNCON_COMBO_LABELS, *LLMSR_LABELS]


def _is_llmsr(alg):
    return alg in _LLMSR_BACKBONES


def _color(alg):
    if alg in _LLMSR_BACKBONES:
        return _LLMSR_BACKBONES[alg]["color"]
    if oneshot_overlay.is_oneshot(alg):
        return oneshot_overlay.color(alg)
    # _tf_color already knows "AIFeynman" (steel blue) -- shared with the SRBench
    # figures -- so this is only a guard for a suffixed variant tree's label.
    if aifeynman_addon.is_aifeynman(alg):
        return aifeynman_addon.COLOR
    # _tf_color does not know PhyE2E -- it is not an SRBench-published baseline.
    if phye2e_addon.is_phye2e(alg):
        return phye2e_addon.color(alg)
    # _tf_color DOES know PySR (we taught it in plot_ood_vs_gap), but go through the
    # addon anyway so a suffixed variant tree gets its darker shade here too.
    if pysr_addon.is_pysr(alg):
        return pysr_addon.color(alg)
    return _tf_color(alg)


def load_combined_raw(root):
    """Combined per-(gap, algorithm, seed, equation_id, ood_r2) table: the base-cache
    methods plus every available LLMSR backbone, at noise 0, each backbone's source
    label mapped to its display label.  A backbone whose cache is missing is skipped
    (with a warning) so the figure still renders.

    """
    base_name = ood_transform.llmsr_cache_name(0)
    base = pd.read_csv(os.path.join(root, base_name))
    frames = [base[["gap", "algorithm", "seed", "equation_id", "ood_r2"]].copy()]

    for label, spec in _LLMSR_BACKBONES.items():
        cache_name = ood_transform.llmsr_backbone_cache_name(
            spec["backbone"], "lsr_transform", 0)
        path = os.path.join(root, cache_name)
        if not os.path.exists(path):
            print(f"[warn] missing LLMSR cache, skipping {label}: {path}")
            continue
        df = pd.read_csv(path)
        if "equation_id" not in df.columns and "dataset" in df.columns:
            df = df.rename(columns={"dataset": "equation_id"})
        df = df[["gap", "algorithm", "seed", "equation_id", "ood_r2"]].copy()
        df["algorithm"] = df["algorithm"].replace({spec["src"]: label})
        frames.append(df)

    # The one-shot arms, same schema -- but read through oneshot_overlay.load_cache, so
    # the problems score_oneshot_ood.py dropped (no equation / unparseable) come back as
    # UNSOLVED rows and the solve rate is divided by all 111 equations, as every other
    # arm's is.  Silently skipped when absent: unlike the LLMSR backbones these are
    # opt-in (--include-oneshot), so a missing cache is the normal state of a figure that
    # never asked for them, not something to warn about.
    for label, spec in oneshot_overlay.BACKBONES.items():
        df = oneshot_overlay.load_cache(label, "lsr_transform", 0)
        if df is None or df.empty:
            continue
        df = df[["gap", "algorithm", "seed", "equation_id", "ood_r2"]].copy()
        df["algorithm"] = df["algorithm"].replace({spec["src"]: label})
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["gap"] = df["gap"].astype(float)
    return df


def _seed_rate(grp, thr):
    """Solve rate of one (algorithm, seed, gap) group over its equation universe."""
    n_total = grp["equation_id"].nunique()
    if n_total == 0:
        return float("nan")
    n_correct = int((grp.groupby("equation_id")["ood_r2"].max() >= thr).sum())
    return n_correct / n_total


def apply_monotone_gap(raw, thr, monotone=True):
    """Prefix-AND the gap axis before any solve rate is computed (see
    ood_transform.monotone_gap).  Call this ONCE on the raw table; best_seed_per_method
    and best_seed_curves then inherit the rule through their own counting."""
    if not monotone:
        return raw
    return ood_transform.monotone_gap(ood_transform.restrict_to_common(raw), thr)


def best_seed_per_method(raw, thr, baseline=0.0):
    """{algorithm: seed} choosing, per method, the seed with the highest solve rate at
    the in-distribution baseline (gap 0; ties -> smallest seed id)."""
    best = {}
    g0 = raw[raw["gap"] == float(baseline)]
    for alg, alg_df in g0.groupby("algorithm"):
        rates = {int(seed): _seed_rate(sd, thr) for seed, sd in alg_df.groupby("seed")}
        if not rates:
            continue
        best[alg] = max(sorted(rates), key=lambda s: rates[s])
    return best


def best_seed_curves(raw, thr, best):
    """{algorithm: {gap: solve_rate}} for each method's best seed."""
    curves = {}
    for alg, seed in best.items():
        sub = raw[(raw["algorithm"] == alg) & (raw["seed"] == seed)]
        curves[alg] = {g: _seed_rate(gd, thr) for g, gd in sub.groupby("gap")}
    return curves


def draw_best_seed_curves(ax, curves, algs_sorted, gaps, r2_thr, *,
                          title="LLM-SRBench OOD accuracy vs gap (noise 0, best seed)",
                          legend=True, fs=1.0, xtick_rotation=0):
    """Render the best-seed OOD-accuracy-vs-gap curves onto `ax`.

    Shared by the standalone set-5 figure (this script) and the combined 3-panel set-6
    figure (plot_llmsrbench_time_complexity_best_seed.py), so the curves look identical
    in both.  One distinct marker per method (plot_markers); one legend, upper-right.

    `fs` multiplies every font size (plot_io.font_scale); the combined figure is 16.5 in
    wide and passes >1 so its text prints at the same height as the 10 in curve grids.
    That leaves this panel narrower, in printed terms, than the ones in the 2x2 grids, and
    at nine gap ticks "64" and "128" then collide -- `xtick_rotation` (45 in the combined
    figure) tilts them clear instead of dropping any of the nine values."""
    gap_pos = {g: i for i, g in enumerate(gaps)}
    for alg in algs_sorted:
        xs, ys = [], []
        for g in gaps:
            v = curves[alg].get(float(g))
            if v is not None and np.isfinite(v):
                xs.append(gap_pos[g])
                ys.append(v)
        if not xs:
            continue
        ax.plot(xs, ys, label=alg, color=_color(alg), linestyle=_linestyle(alg),
                lw=CURVE_LW, marker=method_marker(alg), markersize=CURVE_MS, **HOLLOW,
                zorder=5 if _is_llmsr(alg) else 3)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Extrapolation Constant (k)", fontsize=12 * fs)
    ax.set_ylabel(f"Fraction with $R^2 \\geq$ {r2_thr}", fontsize=12 * fs)
    ax.set_xticks(range(len(gaps)))
    ax.set_xticklabels([ood_transform.xticklabel(g) for g in gaps],
                       rotation=xtick_rotation,
                       ha="right" if xtick_rotation else "center",
                       rotation_mode="anchor" if xtick_rotation else None)
    if title:
        ax.set_title(title, fontsize=13 * fs)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11 * fs)
    if legend:
        # Sorted by the PRINTED label (post-rename), as in sets 1/3.
        _h, _l = ax.get_legend_handles_labels()
        handles, labels = sort_legend(
            _h, display_labels(e2e_tpsr_addon.rename_legend(_l)))
        ax.legend(handles, labels,
                  loc="upper right", fontsize=10 * fs, frameon=True, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gaps", nargs="+", type=int, default=None,
                    help="x-axis gap values. Default: 0 1 2 4 8 16 32 64 128.")
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    ap.add_argument("--no-monotone-gap", action="store_true",
                    help="Score each gap independently, allowing a formula that "
                         "failed at a smaller gap to count as solved at a larger "
                         "one. Off by default (ood_transform.monotone_gap).")
    ap.add_argument("--only", nargs="*", default=DEFAULT_METHODS,
                    help="Method display labels to plot (default: 145M 89M e2e LLMSR).")
    ap.add_argument("--include-aifeynman", action="store_true",
                    help="Also draw the AI Feynman 2.0 baseline (results/aifeynman/, "
                         "rows added to the base cache by augment_aifeynman_caches.py). "
                         "Off by default.")
    ap.add_argument("--include-phye2e", action="store_true",
                    help="Also draw the PhyE2E baseline (results/phye2e/, rows added to "
                         "the base cache by augment_phye2e_caches.py). Off by default.")
    ap.add_argument("--include-pysr", action="store_true",
                    help="Also draw the PySR baseline (results/pysr/, rows added to the "
                         "base cache by augment_pysr_caches.py). Off by default.")
    e2e_tpsr_addon.add_include_arg(
        ap, " Its rows reach the base cache via augment_e2e_tpsr_caches.py.")
    oneshot_overlay.add_include_arg(
        ap, " Their rows come from their own per-backbone caches, folded in by "
            "load_combined_raw.")
    ap.add_argument("--output", default="plots/llmsrbench_curves_noise0_bestseed.png")
    ap.add_argument("--show", action="store_true")
    aifeynman_addon.add_group_arg(ap)
    phye2e_addon.add_group_arg(ap)
    pysr_addon.add_group_arg(ap)
    args = ap.parse_args()
    aifeynman_addon.apply_group_arg(args)
    phye2e_addon.apply_group_arg(args)
    pysr_addon.apply_group_arg(args)
    # Opt-in append rather than a DEFAULT_METHODS entry, so the default figure is
    # byte-identical to before.  Uses the ACTIVE tree's (possibly suffixed) label.
    if args.include_aifeynman and aifeynman_addon.DISPLAY_LABEL not in args.only:
        args.only = list(args.only) + [aifeynman_addon.DISPLAY_LABEL]
    if args.include_phye2e:
        # Every arm's rows already sit in the base cache (augment_phye2e_caches.py, run
        # once per tree); --only is what decides which of them get drawn.
        _labels = ([phye2e_addon.label_for_group(g) for g in args.phye2e_groups]
                   if args.phye2e_groups else [phye2e_addon.DISPLAY_LABEL])
        args.only = list(args.only) + [l for l in _labels if l not in args.only]
    if args.include_pysr:
        # Same story as PhyE2E: the rows are already cached per tree, --only selects.
        _labels = ([pysr_addon.label_for_group(g) for g in args.pysr_groups]
                   if args.pysr_groups else [pysr_addon.DISPLAY_LABEL])
        args.only = list(args.only) + [l for l in _labels if l not in args.only]
    if args.include_e2e_tpsr and E2E_TPSR_LABEL not in args.only:
        args.only = list(args.only) + [E2E_TPSR_LABEL]
    if args.include_oneshot:
        args.only = list(args.only) + [l for l in ONESHOT_LABELS if l not in args.only]

    if not args.show:
        matplotlib.use("Agg")

    root = os.path.join(SCRIPT_DIR, "results")
    if args.gaps is None:
        args.gaps = ood_transform.default_gaps()
    gaps = sorted(set(args.gaps))
    baseline = float(ood_transform.baseline_gap())

    raw = load_combined_raw(root)
    keep = set(args.only)
    raw = raw[raw["algorithm"].isin(keep)]
    raw = raw[raw["gap"].isin([float(g) for g in gaps])]
    raw = apply_monotone_gap(raw, args.r2_thr, monotone=not args.no_monotone_gap)

    best = best_seed_per_method(raw, args.r2_thr, baseline)
    curves = best_seed_curves(raw, args.r2_thr, best)
    if not curves:
        raise SystemExit("No methods to plot -- check the OOD caches and --only.")

    print(f"[best seed by baseline (gap={baseline:g}) solve rate]")
    for alg in sorted(curves, key=lambda a: -curves[a].get(baseline, 0.0)):
        print(f"  {alg:8} seed={best[alg]:<3} base={100*curves[alg].get(baseline, float('nan')):.1f}%")

    # Legend order = baseline solve rate, highest first.
    algs_sorted = sorted(curves, key=lambda a: -curves[a].get(baseline, 0.0))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    draw_best_seed_curves(ax, curves, algs_sorted, gaps, args.r2_thr)

    fig.tight_layout()
    save_fig(fig, os.path.join(SCRIPT_DIR, args.output))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
