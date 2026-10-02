#!/usr/bin/env python3
"""
plot_ablation_complexity.py -- per-formula inference time and complexity for the
data-generation ablation arms and the 145M reference, in the set-2 dot-plot style.

Set 2 (plot_time_complexity_vs_noise.py) is a 1x2 figure -- inference time left,
formula complexity right -- sharing one method (y) axis, so a method's time and
complexity read off the same row.  Each panel puts methods on y, the metric on a log x
axis, one hollow marker per target noise with +/-1 std error bars, our models' row
labels bold, and (complexity panel only) the ground-truth median as a dashed line over
a shaded IQR.  This script is that figure for the ablation arms.

All four target-noise levels are drawn in ONE figure -- a marker per level on each
method's row, as set 2 does -- so a row shows how that arm's time and complexity move
with noise; --panels draws either panel alone when the other is not wanted.

Neither metric is recomputed here.  Both come from
compare_srbench.load_transformer_results, so they match the comparison table and set 2
exactly:
  * complexity -- unsimplified sympy node count (formula_complexity).  Per dataset,
    complexity_solved_mean is the mean over the seeds that SOLVED it (R^2 >= thr); per
    method, the mean of that over datasets, as build_comparison_table's
    "Complexity (solved)".  Error bar: complexity_seed_std.
  * time -- per dataset, time_mean is the mean inference time over seeds; per method,
    mean and std of that ACROSS DATASETS, as collect_timing and the table's
    "Avg Time (s)" / "Time Std (s)".  That std is problem-difficulty spread, not seed
    noise, so the two panels' error bars do not mean the same thing.

BOLD ROWS: set 2 bolds "our" models against published SRBench baselines.  Every row here
is ours, so the bolding instead marks the four ABLATION arms, leaving the 145M reference
in regular weight -- it is the yardstick, not a cell of the 2x2.

ROW ORDER is by complexity, descending, in both panels -- set 2 shares one row order
across its panels, which keeps a method's two points on the same line.

Usage:
    python plot_ablation_complexity.py                      # both panels, all four taus
    python plot_ablation_complexity.py --noises 0           # clean run only
    python plot_ablation_complexity.py --panels complexity  # complexity only
    python plot_ablation_complexity.py --panels time
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import results_io
from plot_io import save_fig, font_scale
from compare_srbench import (
    SRBENCH_GT, load_transformer_results, feynman_truth_complexity,
)
from plot_time_vs_noise import _noise_style

# Printed type size, as set 2 (plot_time_complexity_vs_noise.py) does it: sizes below are
# written as the point size the text should have ON THE PAGE, and _FS converts them to
# canvas points for this 10 in-wide figure, which LaTeX then shrinks to \textwidth.
_FIG_W_BOTH, _FIG_W_ONE = 10.0, 6.2
_FS = font_scale(_FIG_W_BOTH)
# One source of truth for the legend names, so this figure and the OOD-vs-gap figure
# never drift apart on what "pref=0 simp=1" is called.
from plot_ablation_ood_vs_gap import DISPLAY

# Marker sizing -- kept in step with plot_time_complexity_vs_noise.py (set 2).
_MS, _ELW, _CAPS, _MEW = 11, 2.4, 3, 2.0
# Vertical inches reserved at the bottom for the legend strip.  A single-column,
# single-entry legend fitted in 0.75; the four-noise strip is a taller box (entries +
# title) and at 0.75 it sat ON the panels' x-axis labels.
_LEGEND_IN = 1.15
_ZORDER = {0.01: 3, 0.001: 4, 0.0: 5, 0.1: 6}

DEFAULT_R2_THR = 0.99
# The paper's ablation figure shows the CLEAN runs only: the noise story is carried by
# the ablation curve figures, and four markers per row made this one hard to read next to
# them.  The sweep is one flag away -- `--noises 0 0.001 0.01 0.1` restores it, and
# _draw_panel still takes any number of levels (set 2's own shape).
DEFAULT_NOISES = [0.0]
# label -> results tree.  The reference lives in the paper's own tree; the arms in the
# ablation tree they were evaluated into (results/ablation_t120M).  The noise dir is
# composed per call from --noise: these paths used to be hardcoded to noise_0, from when
# the arms existed at tau=0 only, so --noise 0.1 silently scored the tau=0 predictions
# against the tau=0.1 ground-truth reference and labelled the result "noise 0.1".
DEFAULT_SOURCES = [
    ("145M_40_simp1",          "results/mymodels"),
    ("prefac_on_t120M",        "results/ablation_t120M"),
    ("prefac_off_t120M",       "results/ablation_t120M"),
    ("prefac_on_nosimp_t120M", "results/ablation_t120M"),
    ("simp_off_t120M",         "results/ablation_t120M"),
]


def _source_paths(noise):
    """(raw_label, pkl path) for every row at this target noise."""
    nd = results_io.noise_dir("", noise).lstrip("/") or f"noise_{float(noise):g}"
    return [(label, os.path.join(root, nd, f"eval_tf_{label}.pkl.gz"))
            for label, root in DEFAULT_SOURCES]


def _readable_xticks(ax):
    """Legible ticks for a log axis that spans well under a decade.

    A log decade's minor ticks are 2..9 x 10^k, so a range like 10-20 carries no tick at
    all and the decade ticks alone would label one point on the whole axis.  Two cases:
      * narrow (< 3x): plain integers across the range, thinned to at most eight.
      * wider: a log-friendly 1/2/3/5/7 ladder, which stays evenly spaced ON the log
        axis -- integers there bunch up at the high end and overprint.
    Scale and dot-plot style are unchanged either way.
    """
    xlo, xhi = ax.get_xlim()
    xlo = max(xlo, 1e-9)
    if xhi / xlo < 3:
        cand = [t for t in range(max(1, int(np.floor(xlo))), int(np.ceil(xhi)) + 1)]
        # CEILING division: floor gave step=1 for 9..16 candidates, so "at most eight"
        # silently emitted up to 16 ticks and the labels overprinted at the crowded end
        # (visible as "15 16 17 18 19" running together on the complexity panel).
        step = max(1, -(-len(cand) // 8))
        ticks = cand[::step]
    else:
        ladder = [1, 2, 3, 5, 7]
        ticks = [v * 10 ** k for k in range(-3, 4) for v in ladder]
        ticks = [t for t in ticks if xlo <= t <= xhi]
    if not ticks:
        return
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())


def _draw_panel(ax, series, methods, ypos, noises, nstyle, title, xlabel,
                *, show_ylabels, bold, readable_ticks):
    """One panel: methods on y, the metric on a log x axis, ONE MARKER PER NOISE per row.

    `series` is {method: {tau: (value, err)}} -- set 2's shape
    (plot_time_complexity_vs_noise._draw_panel), so a row's four markers overlap and the
    hollow faces let the ones underneath show through.
    """
    ax.set_axisbelow(True)
    for m in methods:
        ax.axhline(ypos[m], color="0.78", linewidth=0.9, zorder=1)
    for m in methods:
        for t in noises:
            if m not in series or t not in series[m]:
                continue
            val, err = series[m][t]
            if not np.isfinite(val):
                continue
            mk, col = nstyle[t]
            ax.errorbar(val, ypos[m], xerr=err, marker=mk, markersize=_MS,
                        linestyle="none", color=col, ecolor=col, elinewidth=_ELW,
                        capsize=_CAPS, markerfacecolor="none", markeredgecolor=col,
                        markeredgewidth=_MEW, zorder=_ZORDER.get(t, 3))
    ax.set_xscale("log")
    if readable_ticks:
        _readable_xticks(ax)
    ax.set_yticks(range(len(methods)))
    ax.set_ylim(-0.6, len(methods) - 0.4)
    if show_ylabels:
        ax.set_yticklabels(methods)
        for m, lbl in zip(methods, ax.get_yticklabels()):
            if m in bold:
                lbl.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    # pad clears the "GT median" annotation, which sits just above the axes.
    ax.set_title(title, fontsize=13 * _FS, pad=16)
    ax.set_xlabel(xlabel, fontsize=11 * _FS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11 * _FS)


def _tables_for_noise(noise, r2_thr, verbose=True):
    """(per-row metrics DataFrame, shared dataset set) at one target noise."""
    per_model, datasets = {}, None
    for raw_label, rel in _source_paths(noise):
        df = load_transformer_results(os.path.join(SCRIPT_DIR, rel), r2_thr)
        if df is None or df.empty:
            if verbose:
                print(f"[warn] noise {noise:g}: no results for {raw_label} at {rel}")
            continue
        per_model[DISPLAY.get(raw_label, raw_label)] = df
        ds = set(df["dataset"])
        datasets = ds if datasets is None else (datasets & ds)
    if not per_model:
        return None, None

    # Same dataset universe for every row, or a method could look simpler (or faster)
    # purely by having been scored on an easier subset.
    rows = []
    for label, df in per_model.items():
        sub = df[df["dataset"].isin(datasets)]
        times = (sub["time_mean"].dropna() if "time_mean" in sub.columns
                 else pd.Series(dtype=float))
        std = (sub["complexity_seed_std"].mean()
               if "complexity_seed_std" in sub.columns else 0.0)
        rows.append({
            "method":     label,
            "complexity": float(sub["complexity_solved_mean"].mean()),   # NaN skipped
            "cplx_err":   float(std) if np.isfinite(std) else 0.0,
            "time":       float(times.mean()) if len(times) else float("nan"),
            "time_err":   float(times.std()) if len(times) > 1 else 0.0,
            "n_solved":   int(sub["complexity_solved_mean"].notna().sum()),
        })
    return pd.DataFrame(rows), datasets


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noises", nargs="+", type=float, default=DEFAULT_NOISES,
                    metavar="TAU",
                    help="target-noise levels to draw, one marker per level per row "
                         "(default: 0 0.001 0.01 0.1, as set 2 sweeps them).  The FIRST "
                         "level also selects the ground-truth complexity reference.")
    ap.add_argument("--r2-thr", type=float, default=DEFAULT_R2_THR,
                    help="a seed 'solved' a dataset at R^2 >= this (default 0.99)")
    ap.add_argument("--panels", choices=["both", "time", "complexity"], default="both")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    noises = sorted(set(args.noises))
    # {method: {tau: (value, err)}} for each metric -- set 2's series shape.
    st, sc, tables = {}, {}, {}
    universe = None
    for tau in noises:
        tab, datasets = _tables_for_noise(tau, args.r2_thr)
        if tab is None:
            print(f"[warn] noise {tau:g}: no model results -- level skipped")
            continue
        tables[tau] = tab
        # ONE universe across every noise level too, not just across rows: a level
        # evaluated on a different dataset set would shift its markers for a reason that
        # has nothing to do with noise.
        universe = datasets if universe is None else (universe & datasets)
        for _, r in tab.iterrows():
            st.setdefault(r["method"], {})[tau] = (r["time"], r["time_err"])
            sc.setdefault(r["method"], {})[tau] = (r["complexity"], r["cplx_err"])
    if not tables:
        sys.exit("[error] no model results found at any requested noise")
    drawn = sorted(tables)
    print(f"[datasets] {len(universe)} shared across "
          f"{len(st)} models x {len(drawn)} noise levels")

    # Row order: slowest at top, by each row's LARGEST time over the drawn noise levels
    # -- set 2's rule (plot_time_complexity_vs_noise._rep), so the two figures' rows read
    # the same way.  ypos 0 is the bottom row.
    def _rep(m):
        vals = [st[m][t][0] for t in drawn if t in st.get(m, {}) and np.isfinite(st[m][t][0])]
        return max(vals) if vals else 0.0
    methods = sorted(st, key=_rep)
    ypos = {m: i for i, m in enumerate(methods)}
    ablation = {m for m in methods if m != "145M"}
    nstyle = _noise_style(drawn)
    # GT complexity is a property of the ground-truth FORMULAE, so it does not move with
    # tau; drawn from the first level, as set 2 does.
    gt = feynman_truth_complexity(SRBENCH_GT, drawn[0], universe)

    # Canvas height: the noise-sweep version reserves a legend strip at the bottom and
    # needs the extra inch so the panels keep their height; without the legend the figure
    # is that much shorter rather than carrying a band of white space.
    fig_h = 4.3 if len(noises) > 1 else 3.3
    want_time = args.panels in ("both", "time")
    want_cplx = args.panels in ("both", "complexity")
    n_panels = int(want_time) + int(want_cplx)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(_FIG_W_BOTH if n_panels == 2 else _FIG_W_ONE, fig_h))
    axes = list(np.atleast_1d(axes))

    if want_time:
        _draw_panel(axes.pop(0), st, methods, ypos, drawn, nstyle,
                    "Inference time (seconds)", "Avg time per formula (s)",
                    show_ylabels=True, bold=ablation, readable_ticks=True)
    if want_cplx:
        ax = axes.pop(0)
        # GT median dashed line + IQR band behind the points (complexity panel only).
        if gt is not None and np.isfinite(gt.get("q50", float("nan"))):
            q25, q50, q75 = gt["q25"], gt["q50"], gt["q75"]
            ax.axvspan(q25, q75, color="#222", alpha=0.10, zorder=0)
            ax.axvline(q50, color="#222", lw=1.4, ls="--", zorder=1)
            ax.text(q50, 1.0, "GT median", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10 * _FS, color="#222",
                    fontweight="bold")
        _draw_panel(ax, sc, methods, ypos, drawn, nstyle,
                    "Formula complexity", "Avg formula complexity (solved)",
                    show_ylabels=not want_time, bold=ablation, readable_ticks=True)

    # A one-level figure needs no key: the legend would spend a strip of canvas saying
    # that the single marker shape means the single noise level, which belongs in the
    # caption.  Drawn (and its space reserved) only when there is something to tell apart.
    legend_in = 0.0
    if len(drawn) > 1:
        handles = [Line2D([0], [0], marker=nstyle[t][0], linestyle="none",
                          color=nstyle[t][1], markersize=_MS, markerfacecolor="none",
                          markeredgewidth=_MEW, label=f"{t:g}") for t in drawn]
        fig.legend(handles=handles, title="Target Noise", loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
                   fontsize=13 * _FS, title_fontsize=13 * _FS,
                   frameon=True, framealpha=0.9)
        legend_in = _LEGEND_IN

    fig.tight_layout(rect=(0, legend_in / fig_h, 1, 1))
    stem = {"both": "ablation_time_complexity", "time": "ablation_time",
            "complexity": "ablation_complexity"}[args.panels]
    save_fig(fig, args.output or os.path.join(SCRIPT_DIR, "plots", stem))

    for tau in drawn:
        tab = tables[tau].set_index("method").reindex(list(reversed(methods))).reset_index()
        print(f"\n=== per-formula time and complexity "
              f"({len(universe)} datasets, R^2 >= {args.r2_thr}, noise {tau:g}) ===")
        with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
            print(tab[["method", "time", "time_err", "complexity", "cplx_err",
                       "n_solved"]].to_string(index=False))
    if gt is not None and np.isfinite(gt.get("q50", float("nan"))):
        print(f"\nground truth complexity: median {gt['q50']:.1f}  "
              f"IQR {gt['q25']:.0f}-{gt['q75']:.0f}  mean {gt.get('mean', float('nan')):.1f}")


if __name__ == "__main__":
    main()
