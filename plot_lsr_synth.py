#!/usr/bin/env python
"""LSR-Synth solve rate vs target noise -- in-distribution and the benchmark's own OOD split.

WHY THIS IS NOT rho_k (corrected 2026-09-23).  An earlier version said rho_k is
unavailable because 104 of the 129 ground truths do not parse.  That reason was wrong:
every ground truth is recoverable (lsr_synth_gt.py -- function notation, and free
constants that are either mangled names like "0.1899..._z" or named parameters, fitted
to `train`; the result reproduces the benchmark's own ood_test targets to R^2 >=
0.999999).  The real obstacle is the INPUTS.  Each LSR-Synth problem is sampled along a
single path: the ODE domains along one trajectory (the state is a function of time),
matsci along one strain-temperature path (|Spearman(epsilon, T)| = 1.00 in the median
problem).  The training points fill only 5-10 % of a 20^D grid over their own bounding
box, so a box-shaped D^k_test (ood_domain / gen_ood_lsr_synth.py) is off the data's
curve even at k = 0, where the ground truth is not identified by D_train at all.
Scored that way (plot_lsr_synth_curves.py), even PySR (1m) drops from 92 % on the
benchmark's in-distribution split to rho_0 = 0.36.  So rho_k is not a meaningful
measure here, and this figure uses the benchmark's own splits instead.

What LSR-Synth DOES ship is its own out-of-distribution test split (500 points per
problem, alongside 4000 train and 500 in-distribution test) -- the only benchmark here
that does.  That is a single fixed shift of unknown magnitude, NOT a point on the k axis,
so it is plotted on its own axes and must not be read as "rho_k at some k".

    python plot_lsr_synth.py                      # -> plots/lsr_synth_noise.pdf
    python plot_lsr_synth.py --r2-threshold 0.999 # stricter match
    python plot_lsr_synth.py --per-domain         # one column per domain instead of pooled
"""
import argparse
import collections
import glob
import gzip
import os
import pickle
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from plot_io import (save_fig, display_labels, sort_legend, bottom_legend,
                     font_scale, GRID_FIG_W_IN)
from plot_markers import HOLLOW, CURVE_LW, method_marker

DOMAINS = ["bio_pop_growth", "chem_react", "matsci", "phys_osc"]
NOISES = ["0", "0.001", "0.01", "0.1"]
MODEL = "145M_80_simp1_float_ftnoise_48h"

# One entry per curve, IN DRAW ORDER: draw_bars enumerates this list, so the sequence
# here is the left-to-right order of the bars inside each noise group.  It must match
# plot_io.LEGEND_ORDER or the legend and the bars disagree -- ours, ours (unscaled),
# then PySR by ascending budget (10s, 1m, 30m).  Each arm names its OWN results group and method-dir spelling,
# because they do not share either: the transformer arms live under
# mymodels_ftnoise_48h_synth as "<model>_lsr_synth_<dom>[_unscale]", while PySR writes
# to its own budget group (pysr_60) as "pysr_lsr_synth_<dom>".  `label` is the internal
# name the shared helpers key on -- plot_io maps it for the legend, plot_ood_vs_gap
# ._tf_color for the colour and plot_markers for the shape -- so every curve here is
# drawn exactly as the same method is drawn in sets 1-3.
ARMS = [
    {"label": "145M-len80-float_ftnoise_48h",
     "group": "mymodels_ftnoise_48h_synth", "dir": f"{MODEL}_lsr_synth_{{dom}}"},
    {"label": "145M-len80-float_ftnoise_48h_unscaled",
     "group": "mymodels_ftnoise_48h_synth", "dir": f"{MODEL}_lsr_synth_{{dom}}_unscale"},
    # PySR at the 60 s and 10 s per-fit budgets -- "PySR (1m)" / "PySR (10s)" elsewhere.
    # Evaluated on our cluster 2026-09-23, 10 seeds x 4 noise, same 129 problems.
    # NOTE the 10 s arm is deliberately absent from sets 1-3 (dropped 2026-09-22 via
    # CURVE_DROP + --pysr-groups) but IS drawn here: on LSR-Synth the cheapest search
    # budget is the interesting comparison, since it is where the transformer is weakest.
    {"label": "PySR-10", "group": "pysr_10", "dir": "pysr_lsr_synth_{dom}"},
    {"label": "PySR-60", "group": "pysr_60", "dir": "pysr_lsr_synth_{dom}"},
    # The 30 min/fit budget (on our cluster 2026-09-23).  With 10 s / 1 m / 30 m the three
    # form a compute ladder, which is the comparison that matters here: PySR passes our
    # best arm at the SHORTEST budget already, so the interesting axis is cost, not
    # whether search wins.
    {"label": "PySR-1800", "group": "pysr_1800", "dir": "pysr_lsr_synth_{dom}"},
]


# Bars carry BOTH channels (2026-09-24): the method's own colour -- the same one that
# method's curve is drawn in throughout sets 1-3, via plot_ood_vs_gap._tf_color -- AND a
# distinct hatch.  Colour alone would break the figure in greyscale and for colour-blind
# readers (which is why it was hatch-only from 2026-09-23); hatch alone made this the one
# figure where an arm did not wear its usual colour.  With both, either channel is enough.
#
# Patterns are ordered by increasing visual weight so the eye can rank them, and are
# deliberately dissimilar in STROKE DIRECTION (none / diagonal / dots / cross) rather
# than in density alone, which is what washes out at small print sizes.
BAR_HATCH = ["", "///", "...", "xxx", "\\\\", "+++", "ooo"]
# Fills are lightened towards white before drawing: the hatch and the bar outline are
# black, and at full saturation the dark arms (PySR's blues) swallow both, leaving a flat
# block where the texture should be.  The tint keeps the hue recognisable next to the
# curves in sets 1-3 while leaving the strokes readable.
BAR_TINT = 0.55               # 0 = the method's colour as-is, 1 = white
BAR_FILL_FALLBACK = "#d9d9d9" # neutral grey for any arm _tf_color does not know


def _bar_fill(label):
    """The method's canonical colour, lightened by BAR_TINT towards white."""
    import matplotlib.colors as mcolors
    import plot_ood_vs_gap as _sr

    col = _sr._tf_color(label) or BAR_FILL_FALLBACK
    r, g, b = mcolors.to_rgb(col)
    t = BAR_TINT
    return (r + (1.0 - r) * t, g + (1.0 - g) * t, b + (1.0 - b) * t)

# ONE font size for every text element -- ticks, axis labels, panel titles, legend.
# Nothing here set a size before, so everything fell back to matplotlib's 10 pt default,
# which is small against this figure's 10 in canvas and left the legend looking like a
# footnote.  Scaled by font_scale() the same way the other figures are, so this stays in
# proportion if GRID_FIG_W_IN ever changes.
FONT = 13 * font_scale(GRID_FIG_W_IN)


def _rows(results_root, arm, tau, domain):
    p = os.path.join(results_root, arm["group"], f"noise_{tau}", "llmsrbench",
                     arm["dir"].format(dom=domain), "results.pkl.gz")
    if not os.path.isfile(p):
        return []
    try:
        with gzip.open(p, "rb") as fh:
            return pickle.load(fh)["results"]
    except Exception as exc:                      # a half-written artifact, not a silent 0
        print(f"  WARNING: unreadable {p}: {type(exc).__name__}: {exc}", flush=True)
        return []


def solve_rate(rows, which, thr):
    """(mean, std, n_seeds) solve rate over problems, averaged across seeds.

    `which` picks the metric block: "id_metrics" is the benchmark's in-distribution test
    split, "ood_metrics" its own OOD split.  Seed-level first, then mean +/- 1 std ACROSS
    SEEDS -- never a binomial/Wilson interval, which would understate the real spread.
    """
    per_seed = collections.defaultdict(list)
    for r in rows:
        m = r.get(which) or {}
        v = m.get("r2")
        ok = v is not None and np.isfinite(v) and v > thr
        per_seed[r["seed"]].append(1.0 if ok else 0.0)
    means = [100.0 * sum(v) / len(v) for v in per_seed.values() if v]
    if not means:
        return float("nan"), 0.0, 0
    return (statistics.mean(means),
            statistics.pstdev(means) if len(means) > 1 else 0.0,
            len(means))


def draw_bars(ax, results_root, domains, thr, which, title):
    """Grouped bars: one group per noise level, one bar per method.

    Bars rather than lines (2026-09-23): with four methods x {in-distribution, OOD} the
    line version put eight series on one axes, and the two metrics of a method share a
    colour, so the eye had to separate them by dash pattern alone.  Splitting the metric
    across PANELS and the methods across BARS means nothing is encoded by line style at
    all, and the four noise levels read as discrete categories -- which they are.
    """
    arms = [a for a in ARMS
            if any(_rows(results_root, a, tau, d) for tau in NOISES for d in domains)]
    n = len(arms)
    width = 0.8 / max(n, 1)
    x = np.arange(len(NOISES))
    for i, arm in enumerate(arms):
        label = arm["label"]
        ys, es = [], []
        for tau in NOISES:
            rows = [r for d in domains for r in _rows(results_root, arm, tau, d)]
            m, sd, _ = solve_rate(rows, which, thr)
            ys.append(m); es.append(sd)
        # Centre the group on the tick: offset by (i - (n-1)/2) widths.
        ax.bar(x + (i - (n - 1) / 2) * width, ys, width * 0.9, yerr=es, capsize=2,
               facecolor=_bar_fill(label), edgecolor="black", linewidth=0.7,
               hatch=BAR_HATCH[i % len(BAR_HATCH)],
               error_kw={"lw": 0.8, "ecolor": "black"},
               label=display_labels([label])[0])
    ax.set_xticks(x)
    # Plain strings, NOT mathtext: at one point size the math renderer draws noticeably
    # bigger than the regular one (measured 28.6 vs 21.0 glyph units here), so $0.001$ on
    # the x axis did not match the plain "100" on the y axis even though both were set to
    # FONT.  The epsilon in the axis label stays mathtext -- it is a symbol, not a number.
    ax.set_xticklabels(["0", "0.001", "0.01", "0.1"], fontsize=FONT)
    ax.set_xlabel(r"target noise $\epsilon$", fontsize=FONT)
    ax.set_title(title, fontsize=FONT)
    ax.tick_params(axis="both", labelsize=FONT)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=os.path.join(SCRIPT_DIR, "results"))
    ap.add_argument("--r2-threshold", type=float, default=0.99, dest="thr",
                    help="near-exact match threshold; 0.99 matches the paper's tau")
    ap.add_argument("--per-domain", action="store_true",
                    help="one panel per domain instead of all 129 problems pooled")
    ap.add_argument("--output", default=os.path.join(SCRIPT_DIR, "plots", "lsr_synth_noise.pdf"))
    args = ap.parse_args()

    absent = [a["group"] for a in ARMS
              if not os.path.isdir(os.path.join(args.results_root, a["group"]))]
    if absent:
        print(f"  [warn] results group(s) not on disk, their arms are skipped: "
              f"{', '.join(sorted(set(absent)))}", flush=True)

    cols = ([(d, [d]) for d in DOMAINS] if args.per_domain
            else [("LSR-Synth (129 problems)", DOMAINS)])
    # One ROW per metric: the benchmark's own in-distribution test split, and its own
    # out-of-distribution split.  The latter is a single fixed shift, NOT a point on the
    # rho_k axis of sets 1/3 -- see the module docstring.
    rows_spec = [("id_metrics", "in-distribution"), ("ood_metrics", "out-of-distribution")]
    fig, axes = plt.subplots(len(rows_spec), len(cols),
                             figsize=(GRID_FIG_W_IN, 2.9 * len(rows_spec)),
                             squeeze=False, sharey=True)
    for r, (which, mname) in enumerate(rows_spec):
        for c, (title, doms) in enumerate(cols):
            head = "" if len(cols) == 1 else (title if r == 0 else "")
            draw_bars(axes[r][c], args.results_root, doms, args.thr, which, head)
            if r < len(rows_spec) - 1:
                axes[r][c].set_xlabel("")
        axes[r][0].set_ylabel(f"{mname}\nsolve rate (%)", fontsize=FONT)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles, labels = sort_legend(handles, labels)   # already display form, see draw_panel
    _leg, bottom = bottom_legend(fig, handles, labels, ncol=5, fontsize=FONT)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    print(save_fig(fig, args.output))


if __name__ == "__main__":
    main()
