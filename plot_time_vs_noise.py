#!/usr/bin/env python3
"""
plot_time_vs_noise.py -- "Time per formula" for every method, across noise levels.

Styled after Fig. 5 (the "Inference time (seconds)" panel) of Kamienny et al.,
"End-to-end symbolic regression with transformers":
  * methods on the y-axis; avg time per formula on a log x-axis
  * each method sits on one faint horizontal row line; its FOUR points (one per
    target noise) all lie on that line, distinguished by MARKER SHAPE
    (x = 0.0, o = 0.001, s = 0.01, + = 0.1) in a purple -> red -> orange palette
  * +/-1 std across datasets as the horizontal error bar
  * our own models' y-labels are shown in bold (as "Ours" is in the paper)

This is the old ood_vs_gap_noise<tau>.png left panel (one bar per method) turned into
a single cross-noise figure; the timing numbers are the same ones collect_timing()
produced there, noise-matched per tau.

Usage:
    python plot_time_vs_noise.py [--noises 0 0.001 0.01 0.1]
                                 [--r2-threshold 0.99]
                                 [--output plots/time_per_formula_vs_noise.png]
                                 [--exclude e2e ...] [--show]
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import results_io
from plot_io import save_fig
from compare_srbench import SRBENCH_GT, load_srbench_gt
from plot_ood_vs_gap import collect_timing, _remap_scale_labels, _label_for_path

DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]
DEFAULT_R2_THR = 0.99

# Fig. 5 style: marker SHAPE encodes the noise level, colours run purple -> red ->
# orange (plasma-like).  All four points of a method sit on one faint horizontal
# row line.  (tau, marker, colour)
_NOISE_STYLE = [
    (0.0,   "x", "#6c2b9d"),
    (0.001, "o", "#9c2f8f"),
    (0.01,  "s", "#d43d2a"),
    (0.1,   "+", "#f59a30"),
]


def _noise_style(noises):
    """{tau: (marker, colour)} -- the paper's shapes for the standard four levels,
    a cycled marker + autumn colour for any non-standard level."""
    base = {t: (mk, c) for t, mk, c in _NOISE_STYLE}
    extra_markers = ["^", "v", "P", "*", "<", ">"]
    out, j = {}, 0
    for i, t in enumerate(noises):
        if t in base:
            out[t] = base[t]
        else:
            import matplotlib.cm as cm
            out[t] = (extra_markers[j % len(extra_markers)],
                      cm.autumn(i / max(1, len(noises) - 1)))
            j += 1
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noises", nargs="+", type=float, default=DEFAULT_NOISES)
    ap.add_argument("--results", nargs="+", default=None,
                    help="Transformer .pkl.gz files; default: auto-discover per noise "
                         "under results/mymodels/noise_<tau>/.")
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    ap.add_argument("--tpsr-results",
                    default="TPSR/srbench_results/feynman_tpsr_l0.1_allnoise.csv")
    ap.add_argument("--output", default="plots/time_per_formula_vs_noise.png")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")
    tpsr_path = os.path.join(SCRIPT_DIR, args.tpsr_results)
    noises = sorted(set(args.noises))

    # {method: {tau: (mean_time, std_time)}} -- one point per (method, noise).
    # my_models = the display names coming from the mymodels group (our transformers),
    # which get bold y-labels like "Ours" in the paper.
    series: dict = defaultdict(dict)
    my_models: set = set()
    for tau in noises:
        if args.results is None:
            result_paths = results_io.list_srbench_pkls(RESULTS_ROOT, tau, "mymodels")
        else:
            result_paths = [os.path.join(SCRIPT_DIR, p) for p in args.results]
        srbench_agg = load_srbench_gt(SRBENCH_GT, tau, args.r2_thr)
        timing = collect_timing(result_paths, srbench_agg, args.r2_thr, tpsr_path, noise=tau)
        # Our raw model labels (m145, m89, ...) -> display names (145M, ...).
        raw_tf = [_label_for_path(p) for p in result_paths if os.path.exists(p)]
        remap = _remap_scale_labels(raw_tf)
        my_models.update(remap.values())
        for k, (mean, std) in timing.items():
            label = remap.get(k, k)
            if label in args.exclude or not np.isfinite(mean):
                continue
            series[label][tau] = (mean, float(std) if np.isfinite(std) else 0.0)

    if not series:
        raise SystemExit("No timing data found -- check results/ and --noises.")

    # -- Plot: methods on y, time on x (log), 4 shaped points/method (per noise) --
    nstyle = _noise_style(noises)

    # Order methods by their slowest noise-level time; slowest ends up at the top.
    def _rep_time(m):
        vals = [series[m][t][0] for t in noises if t in series[m]]
        return max(vals) if vals else 0.0
    methods = sorted(series, key=_rep_time)          # ascending -> fastest at bottom
    ypos = {m: i for i, m in enumerate(methods)}

    fig_h = max(5, len(methods) * 0.32 + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    # One clearly-visible horizontal line per method (full width, from the label
    # across); all four noise points sit on it, as in the paper.
    ax.set_axisbelow(True)
    for m in methods:
        ax.axhline(ypos[m], color="0.78", linewidth=0.9, zorder=0)

    # Draw the filled markers (square, circle) first and the line markers (x, +)
    # on top, so none is hidden when a method's four noise times cluster together.
    _ZORDER = {0.01: 3, 0.001: 4, 0.0: 5, 0.1: 6}
    for m in methods:
        y0 = ypos[m]
        for t in noises:
            if t not in series[m]:
                continue
            mean, std = series[m][t]
            mk, col = nstyle[t]
            ax.errorbar(
                mean, y0, xerr=std, marker=mk, markersize=11,
                linestyle="none", color=col, ecolor=col, elinewidth=2.4, capsize=3,
                markeredgecolor=col, markeredgewidth=2.0, zorder=_ZORDER.get(t, 3),
            )

    ax.set_xscale("log")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    # Bold our own models' y-labels (the paper bolds "Ours").
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in my_models:
            lbl.set_fontweight("bold")

    # Paper style: metric name as the panel title, minimal chrome, no vertical grid.
    ax.set_title("Inference time (seconds)", fontsize=13)
    ax.set_xlabel("Avg time per formula  [log scale, error bars = +/-1 std across datasets]",
                  fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    # Legend: marker shape -> noise level.
    handles = [
        Line2D([0], [0], marker=nstyle[t][0], linestyle="none", color=nstyle[t][1],
               markersize=11, markeredgewidth=2.0, label=(f"{t:g}" if t != 0 else "0.0"))
        for t in noises
    ]
    ax.legend(handles=handles, title="Target Noise", loc="lower right",
              fontsize=15, title_fontsize=15, frameon=True)

    fig.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_fig(fig, out_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
