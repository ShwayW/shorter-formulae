#!/usr/bin/env python3
"""
plot_ablation_ood_vs_gap.py -- OOD solve rate vs extrapolation gap for the
data-generation ablation arms, against the 145M reference model.

One panel, set-1 styling (plot_ood_vs_gap._draw_curves): x is the extrapolation gap k
(categorical positions, so 0..128 are evenly spaced), y is the fraction of the 99
Feynman equations whose recovered formula still reaches OOD R^2 >= 0.99 after the
inputs are shifted by k window-widths.  Seed variation is drawn as mean +/- std error
bars, exactly as sets 1/3/5 do -- not a filled band, which is plot_ood_common's style.

Both benchmarks, selected with --bench: `srbench` (the 119 Feynman equations, the
default) and `llmsr` (the 111 LSR-Transform equations).  The two differ only in which
caches are read, which column keys a problem (`dataset` vs `equation_id`) and which
accuracy_from_raw does the thresholding -- the curves, colours and seed handling are
shared, so the two figures stay visually comparable.

Two caches, because the ablation rows are deliberately kept out of the paper's main
cache (they would otherwise be mixed into every existing figure):
  * base      -- results/ood_raw_noise<tau>.csv          (145M and every other method)
  * ablation  -- results/ood_ablation_t120M_noise<tau>.csv (the arms, from
                 augment_base_ood_caches.py run against results/ood_t120M_root)
and under --bench llmsr, their LLM-SRBench twins:
  * base      -- results/llmsr_ood_raw_noise<tau>.csv
  * ablation  -- results/ood_ablation_t120M_llmsr_noise<tau>.csv

Both caches must cover the same equation universe at each gap or the solve rates are
not comparable; the script checks this and refuses to plot a mismatch.

SEEDS: every arm now carries the full 10 seeds (42-51) on both benchmarks, matching the
145M reference -- the 3-seed cluster run this script was written against was superseded
on 2026-09-12 (it is archived under _old/).  The shared-seed restriction below is
therefore a no-op in the normal case and is kept only so a partially-finished rerun
still produces a like-for-like figure.  --all-seeds uses whatever each model has.

Usage:
    python plot_ablation_ood_vs_gap.py
    python plot_ablation_ood_vs_gap.py --bench llmsr
    python plot_ablation_ood_vs_gap.py --all-seeds --output plots/my_name
    python plot_ablation_ood_vs_gap.py --models 145M_40_simp1 prefac_on_t120M
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import ood_transform
from plot_io import (save_fig, GRID_LEGEND_KW, GRID_LEGEND_NCOL, bottom_legend,
                     above_legend, sort_legend, display_labels, font_scale,
                     GRID_FIG_W_IN)
from plot_markers import method_marker, HOLLOW, CURVE_MS
from plot_ood_vs_gap import (
    accuracy_from_raw, _tf_color, _yerr, _CURVE_LW,
    DEFAULT_GAPS, DEFAULT_R2_THR, _BASELINE_COLOR,
)
from plot_llmsrbench_ood_vs_gap import accuracy_from_raw as _llmsr_accuracy_from_raw

# Printed type size, exactly as sets 1 and 3 do it: a size written as N below prints at
# N * FONT_TRIM points on the paper's textwidth, so a bare "12" on this 10-inch canvas
# would otherwise reach the reader at under 7 pt.  See plot_io.font_scale.
_FS = font_scale(GRID_FIG_W_IN)


def _llmsr_accuracy_df(df, r2_thr):
    """The LLM-SRBench thresholder, shaped like the SRBench one.

    plot_llmsrbench_ood_vs_gap.accuracy_from_raw returns a list of dicts carrying
    n_correct/n_total but no solve_rate; _aggregate below needs a DataFrame with the
    rate already divided out, which is what its SRBench twin returns.
    """
    rec = pd.DataFrame(_llmsr_accuracy_from_raw(df, r2_thr))
    if rec.empty:
        return pd.DataFrame(
            columns=["gap", "algorithm", "seed", "n_correct", "n_total", "solve_rate"])
    rec["solve_rate"] = rec["n_correct"] / rec["n_total"].where(rec["n_total"] > 0)
    return rec

# Raw cache label -> the name drawn in the legend.  "145M" is the name the rest of the
# figures use for 145M_40_simp1, so it keeps its family colour and square marker.
DISPLAY = {
    "145M_40_simp1":     "145M",
    # Legend names follow the PAPER's terminology: affine subtree transformations
    # (use_prefactors in the code) and target simplification (simplify_targets).
    "prefac_on_t120M":   "affine=1 simplify=1",
    "simp_off_t120M":    "affine=0 simplify=0",
    # room for the other two arms once they finish training
    "prefac_off_t120M":       "affine=0 simplify=1",
    "prefac_on_nosimp_t120M": "affine=1 simplify=0",
    # The --unscale decode of each arm (noise 0 only, evaluated 2026-09-23).  TWO keys
    # each: the raw artifact label the SRBench cache stores, and the mangled spelling the
    # LLM-SRBench cache gets from _remap_scale_labels -- see the DEFAULT_MODELS_LLMSR note.
    "prefac_on_t120M_unscale":        "affine=1 simplify=1 (unscaled)",
    "simp_off_t120M_unscale":         "affine=0 simplify=0 (unscaled)",
    "prefac_off_t120M_unscale":       "affine=0 simplify=1 (unscaled)",
    "prefac_on_nosimp_t120M_unscale": "affine=1 simplify=0 (unscaled)",
    "prefac_unscaled_on_t120M":        "affine=1 simplify=1 (unscaled)",
    "simp_unscaled_off_t120M":         "affine=0 simplify=0 (unscaled)",
    "prefac_unscaled_off_t120M":       "affine=0 simplify=1 (unscaled)",
    "prefac_unscaled_on_nosimp_t120M": "affine=1 simplify=0 (unscaled)",
    # THE reference model (2026-09-17; it used to be a second curve beside 145M).  The
    # spelling must match the LLM-SRBench cache's display label exactly -- that half
    # stores "145M-len80-float" -- or the shared legend of --bench both lists this one
    # model twice, once per half, under two names.
    "145M_80_simp1_float":    "145M-len80-float",
    # The 48 h finetune, now the reference curve.  BOTH spellings map to one name so
    # --bench both lists it once rather than once per half (the same reason the raw
    # reference above needed its mapping).
    "145M_80_simp1_float_ftnoise_48h": "145M-len80-float_ftnoise_48h",
    "145M-len80-float_ftnoise_48h":    "145M-len80-float_ftnoise_48h",
    # The --unscale variant of that same checkpoint.  Both curves must come from ONE
    # checkpoint or the figure confounds the flag with training progress, which is why
    # the float cache below points at the FINAL c4 run and not the mid-chain snapshot.
    # It also points at the 400-POINT bags: --sample-size was silently dropped on this
    # code path until 2026-09-06 (evaluate_xy took no sample_size_override), so the
    # earlier caches under ood_float_root / ood_floatfinal_root are 200-point runs of a
    # model trained with num_io_pairs_range=[50, 400].  Both remain on disk; pass
    # --float-cache to draw them.
    "145M_80_simp1_float_unscale": "145M-len80-float (unscale)",
}

# _tf_color knows nothing about the ablation arms and would hand every one of them the
# same steel blue, so name them here.  Chosen clear of the palette the other figures
# already spend: greens (145M), oranges (89M), #db2777 rose (89M-float), #7c3aed violet
# (e2e), #0d9488 teal (TPSR), #457b9d steel blue (AIFeynman), #8c564b brown (Operon).
ABLATION_COLORS = {
    "affine=1 simplify=1": "#1d4ed8",   # royal blue
    "affine=0 simplify=0": "#be123c",   # crimson
    "affine=0 simplify=1": "#a21caf",   # purple
    "affine=1 simplify=0": "#0891b2",   # cyan
    # Each arm's --unscale decode: same family, darker shade, following this file's own
    # plain/unscaled idiom.  NOTE the pair is then separated by COLOUR ALONE -- the
    # marker is shared deliberately so the two settings read as one arm, which is the
    # one place in these figures where monochrome printing loses information.
    "affine=1 simplify=1 (unscaled)": "#1e3a8a",   # blue-900
    "affine=0 simplify=0 (unscaled)": "#881337",   # rose-900
    "affine=0 simplify=1 (unscaled)": "#701a75",   # fuchsia-900
    "affine=1 simplify=0 (unscaled)": "#155e75",   # cyan-800
    # Slate: the second reference curve.  Deliberately desaturated so the four
    # saturated arm colours stay the eye's focus, and clear of every family colour
    # listed above (notably the rose that 89M-float already owns).
    # Lighter shade of the same slate, following this repo's existing idiom for a
    # plain/input-unscaled pair (145M #4ade80/#15803d, 89M #fb923c/#c2410c): same family,
    # two shades, so the pair reads as one model's two settings.  An explicit entry is
    # REQUIRED -- _tf_color would otherwise match the leading "145M" and hand this the
    # 145M family green (its "unscaled" test does not fire on "unscale").
}

# method_marker's fallback pool for unknown labels is one octagon plus rotated polygons,
# and at markersize 6 a 6- or 7-sided polygon is indistinguishable from a circle -- three
# of the four arms came out as circles, which is exactly the colour-only encoding
# plot_markers.py exists to prevent.  These four labels appear ONLY in this figure, so
# reusing shapes that 89M (^ v) and e2e (D) own elsewhere cannot create a clash here.
ABLATION_MARKERS = {
    "affine=1 simplify=1": "^",
    "affine=1 simplify=0": "v",
    "affine=0 simplify=1": "D",
    "affine=0 simplify=0": "o",
    "affine=1 simplify=1 (unscaled)": "^",
    "affine=1 simplify=0 (unscaled)": "v",
    "affine=0 simplify=1 (unscaled)": "D",
    "affine=0 simplify=0 (unscaled)": "o",
    # Star: unused by the arms (^ v D o) and by 145M's square, so the two reference
    # curves stay distinguishable from each other in monochrome print.
}

# The four noise levels --grid panels, in reading order -- the same set and order as
# set 1 (plot_ood_vs_gap._plot_grid).
GRID_NOISES = [0.0, 0.001, 0.01, 0.1]

_ARMS = ["prefac_on_t120M", "prefac_off_t120M",
         "prefac_on_nosimp_t120M", "simp_off_t120M"]

# ONE reference curve on each half: the 145M model the arms are meant to be read
# against, and nothing else.
#
# 2026-09-17: switched from 145M_40_simp1 to the len-80 FLOAT arm, so that "ours" here
# means the same model it means in sets 1-5 (which took the same switch).  The two
# differ in GRAMMAR, not just length: 145M_40_simp1 predates the knobs and so takes
# data_process's default restrict_consts=True -- constant leaves confined to the
# symbolic set {0,1,2,3,pi} (grammar.py:14) -- while 145M_80_simp1_float is the
# unrestricted (restrict_consts=0, float-constant) model.  Reading the prefactor /
# simplify arms against a restricted reference while the headline figures used an
# unrestricted one made "ours" two different models across the paper.
#   python plot_ablation_ood_vs_gap.py --models 145M_40_simp1 <arms...>   # old reference
# The --unscale arms, in each cache's own spelling (see the DISPLAY note): SRBench keeps
# the raw artifact label, LLM-SRBench stores the _remap_scale_labels output.
_ARMS_UNSCALE_SR = [a + "_unscale" for a in _ARMS]
_ARMS_UNSCALE_LLMSR = ["prefac_unscaled_on_t120M", "prefac_unscaled_off_t120M",
                       "prefac_unscaled_on_nosimp_t120M", "simp_unscaled_off_t120M"]

# The reference curve is the 48 H FINETUNED model (2026-09-23), not the raw
# 145M_80_simp1_float it used to be: that arm was dropped from every other figure when
# the finetune became THE "ours" curve, so drawing the raw one only here made the
# ablation's reference a model the rest of the paper no longer shows.  As elsewhere, the
# two halves spell it differently -- SRBench keeps the raw artifact label, LLM-SRBench
# the display one.
# The --unscale arms are NOT in the default lists (2026-09-23): the figure is about the
# 2x2 data-generation ablation, and doubling it to eight curves buried that comparison.
# Their cache rows, DISPLAY names, colours and markers are all still registered, so
#     --models 145M_80_simp1_float_ftnoise_48h prefac_off_t120M prefac_off_t120M_unscale
# draws any subset of them on demand.
DEFAULT_MODELS = ["145M_80_simp1_float_ftnoise_48h"] + _ARMS

# The LLM-SRBench cache stores DISPLAY names ("145M") where the SRBench one stores raw
# artifact labels ("145M_40_simp1") -- the two halves were built by different augment
# paths.  Asking for the SRBench spellings there drops the reference curve with only a
# [warn], which is how the first llmsr figures came out arm-only, so --bench picks the
# right spelling.
DEFAULT_MODELS_LLMSR = ["145M-len80-float_ftnoise_48h"] + _ARMS


def _color_for(label):
    return ABLATION_COLORS.get(label) or _tf_color(label) or _BASELINE_COLOR


def _marker_for(label):
    return ABLATION_MARKERS.get(label) or method_marker(label)


def _load(path, label, key_col):
    if not os.path.exists(path):
        sys.exit(f"[error] {label} cache not found: {path}")
    df = pd.read_csv(path)
    missing = {"algorithm", key_col, "ood_r2", "seed", "gap"} - set(df.columns)
    if missing:
        sys.exit(f"[error] {path} is missing columns: {sorted(missing)}")
    return df


def _aggregate(acc):
    """per (gap, algorithm): mean solve rate over seeds, +/- one std as the band."""
    rec = []
    for (alg, gap), grp in acc.groupby(["algorithm", "gap"]):
        rates = grp["solve_rate"].dropna().to_numpy()
        if rates.size == 0:
            continue
        m, s = float(np.mean(rates)), float(np.std(rates))
        rec.append({"algorithm": alg, "gap": gap, "solve_rate": m, "n_seeds": rates.size,
                    "band_lo": max(0.0, m - s), "band_hi": min(1.0, m + s)})
    return pd.DataFrame(rec)


def _prepare_noise(args, noise, llmsr, verbose=True):
    """Curve data for ONE noise level: (agg[gap, algorithm, solve_rate, band_*],
    algs in legend order, size of the equation universe).

    Everything that depends on tau lives here so --grid can call it four times; main()
    below is then just the single-panel special case.
    """
    tau = f"{float(noise):g}"
    # The problem key differs per benchmark (ood_transform.problem_key), and it is fixed
    # by --bench rather than sniffed from the frame so a cache carrying the wrong column
    # fails loudly in _load instead of silently grouping by the other benchmark's key.
    key_col = "equation_id" if llmsr else "dataset"
    _acc = _llmsr_accuracy_df if llmsr else accuracy_from_raw
    base_name = ood_transform.llmsr_cache_name(tau) if llmsr else f"ood_raw_noise{tau}.csv"
    abl_name = (f"ood_ablation_t120M_llmsr_noise{tau}.csv" if llmsr
                else f"ood_ablation_t120M_noise{tau}.csv")

    # The explicit --*-cache overrides name ONE file, so they only make sense for a
    # single-noise run; --grid reads the four default paths per noise level.
    base_path = (args.base_cache if (args.base_cache and not args.grid)
                 else os.path.join(SCRIPT_DIR, "results", base_name))
    abl_path  = (args.ablation_cache if (args.ablation_cache and not args.grid)
                 else os.path.join(SCRIPT_DIR, "results", abl_name))
    flt_path  = (args.float_cache if (args.float_cache and not args.grid)
                 else os.path.join(SCRIPT_DIR, "results/ood_ss400_root", base_name))

    frames = [_load(base_path, "base", key_col), _load(abl_path, "ablation", key_col)]
    if os.path.exists(flt_path):
        frames.append(_load(flt_path, "float", key_col))
    elif args.float_cache and not args.grid:
        # An explicitly named cache that is missing is a typo, not an absent run.
        sys.exit(f"[error] float cache not found: {flt_path}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["algorithm"].isin(args.models) & df["gap"].isin(args.gaps)]
    found = set(df["algorithm"].unique())
    for m in args.models:
        if m not in found and verbose:
            print(f"[warn] noise {tau}: no rows for '{m}' -- absent from the figure")
    if df.empty:
        sys.exit(f"[error] noise {tau}: no rows for any requested model")

    # Same equation universe per gap, or the solve-rate denominators differ per curve.
    for g, gap_df in df.groupby("gap"):
        universes = {a: frozenset(s[key_col]) for a, s in gap_df.groupby("algorithm")}
        if len(set(universes.values())) > 1:
            sizes = {a: len(u) for a, u in universes.items()}
            sys.exit(f"[error] noise {tau}, gap {g}: models cover different equation "
                     f"sets {sizes} -- solve rates would not be comparable")

    seeds_by_alg = {a: set(s["seed"].unique()) for a, s in df.groupby("algorithm")}
    if not args.all_seeds:
        shared = set.intersection(*seeds_by_alg.values())
        if not shared:
            sys.exit(f"[error] noise {tau}: models share no seed: {seeds_by_alg}")
        df = df[df["seed"].isin(shared)]
        if verbose:
            print(f"[seeds] noise {tau}: restricted to the {len(shared)} shared seeds: "
                  f"{sorted(int(s) for s in shared)}")
    elif verbose:
        print(f"[seeds] noise {tau}: per-model: "
              + ", ".join(f"{a}={len(s)}" for a, s in sorted(seeds_by_alg.items())))

    n_universe = df[df["gap"] == min(args.gaps)][key_col].nunique()
    agg = _aggregate(_acc(df, args.r2_thr))
    agg["algorithm"] = agg["algorithm"].map(lambda a: DISPLAY.get(a, a))
    if agg.empty:
        sys.exit(f"[error] noise {tau}: nothing to plot")

    # Legend/draw order: best at the smallest gap first, as the set-1 panels do.
    g_min = agg["gap"].min()
    rank = agg[agg["gap"] == g_min].set_index("algorithm")["solve_rate"].to_dict()
    algs = sorted(agg["algorithm"].unique(), key=lambda a: -rank.get(a, 0.0))
    return agg, algs, n_universe


def _draw_panel(ax, agg, algs, gaps, r2_thr):
    """One set-1-style panel: categorical gap positions, mean +/- std error bars."""
    gap_pos = {g: i for i, g in enumerate(gaps)}
    for alg in algs:
        sub = agg[agg["algorithm"] == alg].sort_values("gap")
        xs = np.array([gap_pos[g] for g in sub["gap"].values])
        ys = sub["solve_rate"].values
        lo = sub["band_lo"].fillna(sub["solve_rate"]).values.clip(0)
        hi = sub["band_hi"].fillna(sub["solve_rate"]).values
        col = _color_for(alg)
        ax.plot(xs, ys, color=col, lw=_CURVE_LW, ls="-",
                marker=_marker_for(alg), markersize=CURVE_MS, **HOLLOW, label=alg)
        ax.errorbar(xs, ys, yerr=_yerr(ys, lo, hi), fmt="none", ecolor=col,
                    elinewidth=1.0, capsize=2.5, capthick=1.0, alpha=0.85, zorder=1.8)
    ax.set_xticks(range(len(gaps)))
    ax.set_xticklabels(["0" if g == 0 else str(g) for g in gaps])
    ax.set_ylim(-0.02, 1.02)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11 * _FS)


def _print_table(agg, algs, r2_thr, n_universe, tau):
    piv = agg.pivot(index="algorithm", columns="gap", values="solve_rate").reindex(algs)
    print(f"\n=== fraction with OOD R^2 >= {r2_thr} "
          f"({n_universe} equations, noise {tau}) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(piv.to_string())


BENCH_TITLE = {"srbench": "SRBench", "llmsr": "LSR-Transform"}


def _plot_both(args, gaps, user_models):
    """1x2: Feynman | LSR-Transform at ONE noise level, sharing a legend.

    The two benchmarks were separate files (ablation_ood_vs_gap_grid.pdf and its
    llmsrbench twin), each a 2x2 over noise.  This is the pair as one figure and one
    noise level -- two subplots, not two grids -- which also reads less like sets 1 and
    3 than another 2x2 does.  Sweep noise with --noise, or use --grid for the per-
    benchmark 2x2.
    """
    tau = f"{float(args.noise):g}"
    fig, axes = plt.subplots(1, 2, figsize=(GRID_FIG_W_IN, 4.9), sharex=True, sharey=True)
    seen = {}
    for ax, bench in zip(axes, ("srbench", "llmsr")):
        llmsr = bench == "llmsr"
        # Each half spells the reference model differently (see DEFAULT_MODELS_LLMSR),
        # so the per-bench default is re-resolved here; an explicit --models wins on both.
        args.models = user_models or (DEFAULT_MODELS_LLMSR if llmsr else DEFAULT_MODELS)
        agg, algs, n_universe = _prepare_noise(args, args.noise, llmsr, verbose=True)
        _draw_panel(ax, agg, algs, gaps, args.r2_thr)
        # Benchmark name only.  The two halves score against different universes (99 vs
        # 111 equations) -- that belongs in the caption, not in the panel titles, and the
        # per-panel counts are still printed to stdout with each table.
        ax.set_title(BENCH_TITLE[bench], fontsize=13 * _FS)
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
        _print_table(agg, algs, args.r2_thr, n_universe, tau)

    fig.supylabel(f"Fraction with $R^2 \\geq$ {args.r2_thr}", fontsize=12 * _FS)
    _handles, _labels = sort_legend(list(seen.values()), display_labels(list(seen.keys())))
    _leg, frac = bottom_legend(fig, _handles, _labels, ncol=GRID_LEGEND_NCOL,
                               **GRID_LEGEND_KW)
    fig.supxlabel("Extrapolation Constant (k)", fontsize=12 * _FS,
                  y=above_legend(frac, fig))
    fig.tight_layout(rect=(0, frac, 1, 1))

    out = args.output or os.path.join(
        SCRIPT_DIR, "plots",
        f"ablation_ood_vs_gap_both{'' if args.noise == 0 else f'_noise{tau}'}")
    save_fig(fig, out)


def _plot_grid(args, llmsr, gaps):
    """2x2 grid, one subplot per noise level, shared legend -- set 1's layout
    (plot_ood_vs_gap._plot_grid), so the ablation figure can sit beside it."""
    # Same canvas as sets 1 and 3: the taller 8.6 in buys back the vertical room the
    # paper-sized fonts cost, so the y axis still shows more than two ticks.
    fig, axes = plt.subplots(2, 2, figsize=(GRID_FIG_W_IN, 8.6), sharex=True, sharey=True)
    seen = {}
    for ax, noise in zip(axes.flat, GRID_NOISES):
        agg, algs, n_universe = _prepare_noise(args, noise, llmsr, verbose=True)
        _draw_panel(ax, agg, algs, gaps, args.r2_thr)
        ax.set_title(f"noise = {noise:g}", fontsize=13 * _FS)
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
        _print_table(agg, algs, args.r2_thr, n_universe, f"{noise:g}")

    fig.supylabel(f"Fraction with $R^2 \\geq$ {args.r2_thr}", fontsize=12 * _FS)
    # Legend as a strip UNDER the grid, with the shared x-label just above it -- the
    # same bottom_legend treatment sets 1 and 3 use, so the box can never land on data.
    _handles, _labels = sort_legend(list(seen.values()), display_labels(list(seen.keys())))
    _leg, frac = bottom_legend(fig, _handles, _labels, ncol=GRID_LEGEND_NCOL,
                               **GRID_LEGEND_KW)
    fig.supxlabel("Extrapolation Constant (k)", fontsize=12 * _FS,
                  y=above_legend(frac, fig))
    fig.tight_layout(rect=(0, frac, 1, 1))

    out = args.output or os.path.join(
        SCRIPT_DIR, "plots",
        f"ablation_{'llmsrbench_' if llmsr else ''}ood_vs_gap_grid")
    save_fig(fig, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", choices=["srbench", "llmsr", "both"], default="srbench",
                    help="which benchmark to draw: srbench = the 119 Feynman equations "
                         "(default), llmsr = the 111 LSR-Transform equations, both = one "
                         "1x2 figure with a panel per benchmark at a single --noise "
                         "(plots/ablation_ood_vs_gap_both).  Selects the cache pair, the "
                         "problem key and the output name.")
    ap.add_argument("--noise", type=float, default=0.0, metavar="TAU",
                    help="target-noise level; selects BOTH caches so the curves are "
                         "noise-matched (default 0)")
    ap.add_argument("--base-cache", default=None,
                    help="default results/ood_raw_noise<tau>.csv")
    ap.add_argument("--ablation-cache", default=None,
                    help="default results/ood_ablation_t120M_noise<tau>.csv")
    ap.add_argument("--float-cache", default=None,
                    help="third cache, for the 145M-80 float reference; default "
                         "results/ood_ss400_root/ood_raw_noise<tau>.csv.  Unlike the "
                         "other two this one is OPTIONAL -- absent, the figure just "
                         "drops that curve, so the script still runs on a checkout "
                         "that has never evaluated the float model.")
    ap.add_argument("--models", nargs="+", default=None,
                    help="cache labels to draw (default: 145M + all four ablation arms, "
                         "spelled as the selected --bench cache spells them)")
    ap.add_argument("--gaps", nargs="+", type=int, default=DEFAULT_GAPS)
    ap.add_argument("--r2-thr", type=float, default=DEFAULT_R2_THR)
    ap.add_argument("--grid", action="store_true",
                    help="Combine all four noise levels into one 2x2 figure (one "
                         "subplot per noise) with a shared legend, in the style of set 1 "
                         "in make_requested_plots.sh, instead of a single-noise plot.  "
                         "Default output: plots/ablation_ood_vs_gap_grid.{pdf,png} "
                         "(ablation_llmsrbench_ood_vs_gap_grid under --bench llmsr).  "
                         "--noise is ignored; the explicit --*-cache overrides are too, "
                         "since each names a single file.")
    ap.add_argument("--all-seeds", action="store_true",
                    help="use every seed each model has instead of the shared subset")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    llmsr = args.bench == "llmsr"
    gaps = sorted(args.gaps)

    if args.bench == "both":
        # --grid is the noise axis and "both" is the benchmark axis; together they would
        # be the 8-panel figure this deliberately is not.  Refuse rather than silently
        # honour one of them.
        if args.grid:
            sys.exit("[error] --grid and --bench both are different figures: --grid is a "
                     "2x2 over noise for ONE benchmark, --bench both is 1x2 over the two "
                     "benchmarks at ONE noise.  Pick one.")
        _plot_both(args, gaps, args.models)
        return

    if args.models is None:
        args.models = DEFAULT_MODELS_LLMSR if llmsr else DEFAULT_MODELS

    if args.grid:
        _plot_grid(args, llmsr, gaps)
        return

    tau = f"{float(args.noise):g}"
    agg, algs, n_universe = _prepare_noise(args, args.noise, llmsr, verbose=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _draw_panel(ax, agg, algs, gaps, args.r2_thr)
    ax.set_xlabel("k", fontsize=12 * _FS)
    ax.set_ylabel(f"Fraction with $R^2 \\geq$ {args.r2_thr}", fontsize=12 * _FS)
    ax.legend(fontsize=11 * _FS, framealpha=0.9)

    out = args.output or os.path.join(
        SCRIPT_DIR, "plots",
        f"ablation_{'llmsrbench_' if llmsr else ''}ood_vs_gap"
        f"{'' if args.noise == 0 else f'_noise{tau}'}")
    save_fig(fig, out)
    _print_table(agg, algs, args.r2_thr, n_universe, tau)


if __name__ == "__main__":
    main()
