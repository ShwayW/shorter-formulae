#!/usr/bin/env python3
"""
plot_complexity_vs_noise.py -- "Formula complexity" for every method, across noise
levels.  Styled after Fig. 5 (the "Formula complexity" panel) of Kamienny et al.,
"End-to-end symbolic regression with transformers".

Layout (identical to plot_time_vs_noise.py, its sibling for the inference-time panel):
  * methods on the y-axis; mean solved-formula complexity on a log x-axis
  * each method sits on one faint horizontal row line; its FOUR points (one per
    target noise) lie on that line, distinguished by MARKER SHAPE
    (x = 0.0, o = 0.001, s = 0.01, + = 0.1) in a purple -> red -> orange palette
  * +/-1 std across seeds as the horizontal error bar
  * our own models' y-labels are bold
  * the ground-truth Feynman complexity is shown as a dashed vertical line (median)
    with a shaded band spanning the IQR

Complexity numbers come from build_comparison_table() -- the same source the old
srbench_comparison_*.png left panel used -- so the two agree.

Usage:
    python plot_complexity_vs_noise.py [--noises 0 0.001 0.01 0.1]
                                       [--r2-threshold 0.99]
                                       [--output plots/complexity_per_formula_vs_noise.png]
                                       [--exclude e2e ...] [--show]
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import results_io
from plot_io import save_fig
from compare_srbench import (
    SRBENCH_GT, load_srbench_gt, load_transformer_results, load_tpsr_results,
    build_comparison_table, feynman_truth_complexity, _resolve_artifact,
)
from plot_ood_vs_gap import _remap_scale_labels, _label_for_path
# Reuse the exact Fig-5 marker shapes / colours from the inference-time plot so the
# two panels look like one figure split in two.
from plot_time_vs_noise import _NOISE_STYLE, _noise_style  # noqa: F401

DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]
DEFAULT_R2_THR = 0.99

# Shared marker sizing (kept in step with plot_time_vs_noise.py).
_MS = 11
_ELW = 2.4
_CAPS = 3
_MEW = 2.0
_ZORDER = {0.01: 3, 0.001: 4, 0.0: 5, 0.1: 6}


def _load_complexity(result_paths, tpsr_csv, e2e_path, r2_thr, noise):
    """Per-method (mean solved complexity, seed std) at one noise level, plus the
    datasets seen -- built via build_comparison_table so the numbers match the
    srbench_comparison table exactly."""
    tf_results, datasets = [], set()
    raw = [_label_for_path(p) for p in result_paths if os.path.exists(p)]
    remap = _remap_scale_labels(raw)
    my_models = set(remap.values())
    for path in result_paths:
        agg = load_transformer_results(path, r2_thr)          # with complexity
        if agg is None:
            continue
        label = remap.get(_label_for_path(path), _label_for_path(path))
        tf_results.append((agg, label))
        datasets |= set(agg["dataset"])
    if e2e_path and os.path.exists(e2e_path):
        e2e_agg = load_transformer_results(e2e_path, r2_thr)
        if e2e_agg is not None:
            tf_results.append((e2e_agg, "e2e"))
            datasets |= set(e2e_agg["dataset"])
    if tpsr_csv and os.path.exists(tpsr_csv):
        tpsr_agg = load_tpsr_results(tpsr_csv, r2_thr, noise=noise)
        if tpsr_agg is not None:
            tf_results.append((tpsr_agg, "TPSR"))
            datasets |= set(tpsr_agg["dataset"])

    srbench_agg = load_srbench_gt(SRBENCH_GT, noise, r2_thr)
    common = sorted(datasets)
    table = build_comparison_table(tf_results, srbench_agg, set(common), r2_thr, noise)
    out = {}
    for _, row in table.iterrows():
        c = row.get("Complexity (solved)")
        if c is None or not np.isfinite(c):
            continue
        std = row.get("Complexity Seed Std (solved)", 0.0)
        out[row["Algorithm"]] = (float(c), float(std) if np.isfinite(std) else 0.0)
    return out, my_models, common


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noises", nargs="+", type=float, default=DEFAULT_NOISES)
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    ap.add_argument("--tpsr-results",
                    default="TPSR/srbench_results/feynman_tpsr_l0.1_allnoise.csv")
    ap.add_argument("--output", default="plots/complexity_per_formula_vs_noise.png")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--reuse-cache", action="store_true",
                    help="Reuse the cached per-method complexity "
                         "(results/complexity_noise<tau>.csv) instead of recomputing it "
                         "-- the complexity is a (slow, sympy) formula-node count, so "
                         "this turns a re-plot from minutes into ~1s. Missing noise "
                         "levels are computed and cached; the cache is always written.")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")
    tpsr_csv = os.path.join(SCRIPT_DIR, args.tpsr_results)
    noises = sorted(set(args.noises))

    series: dict = defaultdict(dict)       # {method: {tau: (complexity, std)}}
    my_models: set = set()
    all_datasets: set = set()
    for tau in noises:
        # Per-method complexity is the slow part (sympy node count on every predicted
        # formula).  Cache it per noise so a re-plot (layout/size tweak) is instant.
        cache = os.path.join(RESULTS_ROOT, f"complexity_noise{tau:g}.csv")
        if args.reuse_cache and os.path.exists(cache):
            cdf = pd.read_csv(cache)
            cplx = {r.algorithm: (float(r.complexity), float(r.std)) for r in cdf.itertuples()}
            mine = set(cdf.loc[cdf["is_ours"], "algorithm"])
            print(f"[cplx] reused {os.path.basename(cache)}")
        else:
            result_paths = results_io.list_srbench_pkls(RESULTS_ROOT, tau, "mymodels")
            e2e_path = _resolve_artifact(results_io.e2e_srbench_pkl(RESULTS_ROOT, tau))
            cplx, mine, common = _load_complexity(result_paths, tpsr_csv, e2e_path,
                                                  args.r2_thr, tau)
            all_datasets.update(common)
            pd.DataFrame(
                [{"algorithm": a, "complexity": c, "std": s, "is_ours": a in mine}
                 for a, (c, s) in cplx.items()]
            ).to_csv(cache, index=False)
            print(f"[cplx] wrote {os.path.basename(cache)}")
        my_models.update(mine)
        for label, val in cplx.items():
            if label in args.exclude:
                continue
            series[label][tau] = val

    if not series:
        raise SystemExit("No complexity data found -- check results/ and --noises.")

    # Ground-truth Feynman complexity (noise-independent: the true formula is fixed).
    # Cheap (one feather read + ~119 sympy parses), so it is always recomputed; when
    # every noise came from cache we lack the evaluated universe, so fall back to all
    # Feynman datasets in the feather at this noise.
    if not all_datasets:
        _gtdf = pd.read_feather(SRBENCH_GT)
        all_datasets = set(_gtdf[(_gtdf["data_group"] == "Feynman")
                                 & (_gtdf["target_noise"] == noises[0])]["dataset"].unique())
    gt = feynman_truth_complexity(SRBENCH_GT, noises[0], set(all_datasets))

    # -- Plot -----------------------------------------------------------------
    nstyle = _noise_style(noises)

    def _rep(m):
        vals = [series[m][t][0] for t in noises if t in series[m]]
        return max(vals) if vals else 0.0
    methods = sorted(series, key=_rep)            # ascending -> most complex at top
    ypos = {m: i for i, m in enumerate(methods)}

    fig_h = max(5, len(methods) * 0.32 + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    # GT median (dashed) + IQR band (shaded), behind everything.
    if gt is not None and np.isfinite(gt.get("q50", float("nan"))):
        q25, q50, q75 = gt["q25"], gt["q50"], gt["q75"]
        ax.axvspan(q25, q75, color="#222", alpha=0.10, zorder=0)
        ax.axvline(q50, color="#222", lw=1.4, ls="--", zorder=1)
        ax.text(q50, 1.0, f"GT median ~ {q50:.1f}  (IQR {q25:.0f}-{q75:.0f})",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=10, color="#222", fontweight="bold")

    # One faint horizontal row line per method.
    ax.set_axisbelow(True)
    for m in methods:
        ax.axhline(ypos[m], color="0.78", linewidth=0.9, zorder=1)

    for m in methods:
        y0 = ypos[m]
        for t in noises:
            if t not in series[m]:
                continue
            cval, std = series[m][t]
            mk, col = nstyle[t]
            ax.errorbar(
                cval, y0, xerr=std, marker=mk, markersize=_MS,
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

    ax.set_title("Formula complexity", fontsize=13)
    ax.set_xlabel("Avg formula complexity (solved)  [log scale, error bars = +/-1 std across seeds]",
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
