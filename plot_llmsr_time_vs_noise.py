#!/usr/bin/env python3
"""
plot_llmsr_time_vs_noise.py -- LLM-SRBench "search time per problem" across noise
levels.  The LLM-SRBench counterpart of plot_time_vs_noise.py, styled after Fig. 5
(the inference-time panel) of Kamienny et al.

Layout (identical to plot_time_vs_noise.py):
  * methods on the y-axis; mean search time per problem on a log x-axis
  * each method sits on one faint horizontal row line; its points (one per target
    noise) lie on that line, distinguished by MARKER SHAPE
    (x = 0.0, o = 0.001, s = 0.01, + = 0.1) in a purple -> red -> orange palette
  * +/-1 std across problems as the horizontal error bar
  * our own models' y-labels are bold

This is the left panel that used to live inside llmsrbench_ood_vs_gap_noise<tau>.png,
turned into one cross-noise figure.

Usage:
    python plot_llmsr_time_vs_noise.py [--noises 0 0.001 0.01 0.1]
                                       [--split lsr_transform]
                                       [--output plots/llmsr_time_per_problem_vs_noise.png]
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
from plot_llmsrbench_ood_vs_gap import (
    discover_methods_from_dirs, load_method_data, _SKIP_SUBSTR,
)
# Reuse the exact Fig-5 marker shapes / colours + sizing from the SRBench time plot.
from plot_time_vs_noise import _NOISE_STYLE, _noise_style  # noqa: F401

DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]

# Marker sizing kept in step with plot_time_vs_noise.py.
_MS = 11
_ELW = 2.4
_CAPS = 3
_MEW = 2.0
_ZORDER = {0.01: 3, 0.001: 4, 0.0: 5, 0.1: 6}


def _is_ours(label):
    """Our transformers (145M*/89M*/89M-float); e2e and any baseline are not."""
    return label.startswith("145M") or label.startswith("89M")


def _time_stats(method_data):
    """(mean, std) of the search time per problem for one method.

    method_data = {seed: {equation_id: (equation, search_time)}}.  Each SEED
    contributes one number -- its mean search time over the problems it ran -- and the
    reported std is the spread of those, matching the complexity bars
    (plot_llmsr_time_complexity_vs_noise._method_complexity) and the SRBench side's
    _seed_std_of_mean.  It used to be the spread ACROSS PROBLEMS, which measures
    problem difficulty rather than run-to-run variability.  A single-seed run reports
    0.0, as the complexity bars do."""
    per_seed = []
    for _seed, probs in method_data.items():
        ts = [float(t) for _eq, t in probs.values() if t is not None and np.isfinite(t)]
        if ts:
            per_seed.append(float(np.mean(ts)))
    if not per_seed:
        return None
    return float(np.mean(per_seed)), (float(np.std(per_seed)) if len(per_seed) >= 2 else 0.0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noises", nargs="+", type=float, default=DEFAULT_NOISES)
    ap.add_argument("--split", default="lsr_transform")
    ap.add_argument("--output", default="plots/llmsr_time_per_problem_vs_noise.png")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")
    noises = sorted(set(args.noises))

    series: dict = defaultdict(dict)       # {method: {tau: (mean_time, std_time)}}
    my_models: set = set()
    for tau in noises:
        dirs = results_io.list_llmsr_method_dirs(RESULTS_ROOT, tau, args.split)
        methods_paths = discover_methods_from_dirs(dirs, args.split, _SKIP_SUBSTR)
        for label, jl in methods_paths.items():
            if label in args.exclude:
                continue
            stats = _time_stats(load_method_data(jl))
            if stats is None:
                continue
            series[label][tau] = stats
            if _is_ours(label):
                my_models.add(label)

    if not series:
        raise SystemExit("No LLM-SRBench timing found -- check results/ and --noises.")

    # -- Plot -----------------------------------------------------------------
    nstyle = _noise_style(noises)

    def _rep(m):
        vals = [series[m][t][0] for t in noises if t in series[m]]
        return max(vals) if vals else 0.0
    methods = sorted(series, key=_rep)            # ascending -> slowest at top
    ypos = {m: i for i, m in enumerate(methods)}

    fig_h = max(5, len(methods) * 0.42 + 1.4)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    ax.set_axisbelow(True)
    for m in methods:
        ax.axhline(ypos[m], color="0.78", linewidth=0.9, zorder=1)

    for m in methods:
        y0 = ypos[m]
        for t in noises:
            if t not in series[m]:
                continue
            mean, std = series[m][t]
            mk, col = nstyle[t]
            ax.errorbar(
                mean, y0, xerr=std, marker=mk, markersize=_MS,
                linestyle="none", color=col, ecolor=col, elinewidth=_ELW, capsize=_CAPS,
                markeredgecolor=col, markeredgewidth=_MEW, zorder=_ZORDER.get(t, 3),
            )

    ax.set_xscale("log")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in my_models:
            lbl.set_fontweight("bold")

    ax.set_title("LLM-SRBench search time (seconds)", fontsize=13)
    ax.set_xlabel("Avg search time per problem (s)  [log scale, error bars = +/-1 std across problems]",
                  fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    handles = [
        Line2D([0], [0], marker=nstyle[t][0], linestyle="none", color=nstyle[t][1],
               markersize=_MS, markeredgewidth=_MEW,
               label=(f"{t:g}" if t != 0 else "0.0"))
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
