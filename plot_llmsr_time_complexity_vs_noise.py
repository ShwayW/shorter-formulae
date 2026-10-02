#!/usr/bin/env python3
"""
plot_llmsr_time_complexity_vs_noise.py -- the LLM-SRBench counterpart of
plot_time_complexity_vs_noise.py: one 1x2 figure with LLM-SRBench search time (left)
and formula complexity (right) for every method across noise levels, sharing a single
method (y) axis.

Each panel is the usual Fig-5 dot plot (methods on y, metric on log x, one point per
target noise, +/-1 std error bars, our models bold).  The complexity panel keeps the
ground-truth median dashed line + IQR band, computed from the benchmark's gt_equation.

Complexity is the sympy node count of each method's discovered_equation on the
problems it solved (id_metrics r2 >= threshold); it is cached per noise in
results/llmsr_complexity_noise<tau>.csv so re-plots are instant with --reuse-cache.

Usage:
    python plot_llmsr_time_complexity_vs_noise.py [--noises 0 0.001 0.01 0.1]
                                                  [--split lsr_transform]
                                                  [--r2-threshold 0.99] [--reuse-cache]
                                                  [--output plots/llmsr_time_complexity_vs_noise.png]
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
import tpsr_addon
import e2e_tpsr_addon
import devncon_addon
import aifeynman_addon
import phye2e_addon
import pretrained_addon
import pysr_addon
import llmsr_overlay
import oneshot_overlay
from plot_io import save_fig
from results_io import load_rows_any
from compare_srbench import formula_complexity
from plot_llmsrbench_ood_vs_gap import discover_methods_from_dirs, load_method_data, _SKIP_SUBSTR
from plot_llmsr_time_vs_noise import _time_stats, _is_ours
from plot_time_vs_noise import _noise_style
from plot_time_complexity_vs_noise import (_draw_panel, noise_legend_handles,
                                           _FIG_W, _FIG_H, _MS, _MEW, _FS,
                                            _LEGEND_IN, _GT_MARK_TOP)

DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]
DEFAULT_R2_THR = 0.99

# _MS/_MEW (the legend key's marker size) come from the SRBench twin too, so the dots in
# the two figures cannot drift apart.
# _FIG_W/_FIG_H come from the SRBench twin (imported above) so the two figures cannot
# drift apart in size -- see the note in main().
# Sentinel default for --row-height: "no explicit row height, use the fixed _FIG_H".
# A real number here goes back to the per-row height budget.
_ROW_H_AUTO = None


def _time_series(noises, split, root, include_tpsr=False,
                 include_aifeynman=False, include_devncon=False,
                 include_phye2e=False, include_e2e_tpsr=False,
                 include_pysr=False):
    """{method: {tau: (mean_time, std_time)}} -- search time per problem."""
    series = defaultdict(dict)
    for tau in noises:
        dirs = results_io.list_llmsr_method_dirs(root, tau, split)
        methods = dict(discover_methods_from_dirs(dirs, split, _SKIP_SUBSTR))
        if include_tpsr:
            methods.update(tpsr_addon.llmsr_tpsr_methods(root, tau, split))
        if include_aifeynman:
            methods.update(aifeynman_addon.llmsr_aifeynman_methods(root, tau, split))
        if include_phye2e:
            methods.update(phye2e_addon.llmsr_methods_for_groups(
                root, tau, split, include_phye2e))
        if include_pysr:
            methods.update(pysr_addon.llmsr_methods_for_groups(
                root, tau, split, include_pysr))
        if include_devncon:
            methods.update(devncon_addon.llmsr_devncon_methods(root, tau, split))
        if include_e2e_tpsr:
            methods.update(e2e_tpsr_addon.llmsr_methods(root, tau, split))
        for label, jl in methods.items():
            st = _time_stats(load_method_data(jl))
            if st is not None:
                series[label][tau] = st
    return series


def _method_complexity(jl, r2_thr):
    """(mean solved complexity, +/-1 seed std) for one method's discovered eqs.

    Per seed, take the mean node count over the problems it solved (r2 >= thr);
    report the mean and std of those per-seed means -- mirroring the SRBench
    'Complexity (solved)' / seed-std used in plot_complexity_vs_noise.py."""
    _meta, rows = load_rows_any(jl)
    by_seed = defaultdict(list)
    for r in rows:
        r2 = (r.get("id_metrics") or {}).get("r2")
        if r2 is None or not np.isfinite(r2) or r2 < r2_thr:
            continue
        c = formula_complexity(r.get("discovered_equation"))
        if np.isfinite(c):
            by_seed[r.get("seed")].append(c)
    per_seed = [float(np.mean(v)) for v in by_seed.values() if v]
    if not per_seed:
        return None
    return float(np.mean(per_seed)), (float(np.std(per_seed)) if len(per_seed) >= 2 else 0.0)


def _complexity_series(noises, split, root, r2_thr, reuse_cache, include_tpsr=False,
                       include_aifeynman=False, include_devncon=False,
                       include_phye2e=False, include_e2e_tpsr=False,
                       include_pysr=False):
    """{method: {tau: (complexity, std)}}, cached per noise (sympy is the slow part)."""
    series = defaultdict(dict)
    for tau in noises:
        cache = os.path.join(root, f"llmsr_complexity_noise{tau:g}.csv")
        if reuse_cache and os.path.exists(cache):
            for r in pd.read_csv(cache).itertuples():
                # combo rows live in the cache (from augment_tpsr_caches.py) but
                # are only shown when opted in, so every other figure is unchanged.
                if tpsr_addon.is_tpsr_combo(r.algorithm) and not include_tpsr:
                    continue
                if devncon_addon.is_devncon_combo(r.algorithm) and not include_devncon:
                    continue
                # Same deal for the AI Feynman baseline row that
                # augment_aifeynman_caches.py adds.
                if aifeynman_addon.is_aifeynman(r.algorithm) and not include_aifeynman:
                    continue
                # ... and the PhyE2E row that augment_phye2e_caches.py adds.
                if phye2e_addon.is_phye2e(r.algorithm) and not include_phye2e:
                    continue
                # ... and the PySR row that augment_pysr_caches.py adds.
                if pysr_addon.is_pysr(r.algorithm) and not include_pysr:
                    continue
                # ... and the standalone-TPSR row from augment_e2e_tpsr_caches.py.
                if e2e_tpsr_addon.is_e2e_tpsr(r.algorithm) and not include_e2e_tpsr:
                    continue
                series[r.algorithm][tau] = (float(r.complexity), float(r.std))
        else:
            dirs = results_io.list_llmsr_method_dirs(root, tau, split)
            methods = dict(discover_methods_from_dirs(dirs, split, _SKIP_SUBSTR))
            if include_tpsr:
                methods.update(tpsr_addon.llmsr_tpsr_methods(root, tau, split))
            if include_aifeynman:
                methods.update(aifeynman_addon.llmsr_aifeynman_methods(root, tau, split))
            if include_phye2e:
                methods.update(phye2e_addon.llmsr_methods_for_groups(
                    root, tau, split, include_phye2e))
            if include_pysr:
                methods.update(pysr_addon.llmsr_methods_for_groups(
                    root, tau, split, include_pysr))
            if include_devncon:
                methods.update(devncon_addon.llmsr_devncon_methods(root, tau, split))
            if include_e2e_tpsr:
                methods.update(e2e_tpsr_addon.llmsr_methods(root, tau, split))
            out = []
            for label, jl in methods.items():
                st = _method_complexity(jl, r2_thr)
                if st is not None:
                    series[label][tau] = st
                    out.append({"algorithm": label, "complexity": st[0], "std": st[1],
                                "is_ours": _is_ours(label)})
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            pd.DataFrame(out).to_csv(cache, index=False)
    return series


def _gt_complexity(noise, split, root):
    """{q25,q50,q75} of the benchmark's ground-truth formula complexity (one per
    problem, noise-independent)."""
    dirs = results_io.list_llmsr_method_dirs(root, noise, split)
    methods = discover_methods_from_dirs(dirs, split, _SKIP_SUBSTR)
    if not methods:
        return None
    _meta, rows = load_rows_any(next(iter(methods.values())))
    gt = {}
    for r in rows:
        gt.setdefault(r.get("equation_id"), r.get("gt_equation"))
    cplx = np.array([formula_complexity(g) for g in gt.values()], dtype=float)
    cplx = cplx[np.isfinite(cplx)]
    if cplx.size == 0:
        return None
    return {"q25": float(np.quantile(cplx, 0.25)),
            "q50": float(np.quantile(cplx, 0.50)),
            "q75": float(np.quantile(cplx, 0.75))}


def build_parser():
    """The CLI, split out of main() so plot_combined_time_complexity.py can parse exactly
    the same LLM-SRBench flags for the bottom row of the combined figure."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noises", nargs="+", type=float, default=DEFAULT_NOISES)
    ap.add_argument("--split", default="lsr_transform")
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    ap.add_argument("--output", default="plots/llmsr_time_complexity_vs_noise.png")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--reuse-cache", action="store_true",
                    help="Reuse the cached per-noise complexity "
                         "(results/llmsr_complexity_noise<tau>.csv) instead of recomputing.")
    ap.add_argument("--include-tpsr-combo", action="store_true",
                    help="Also include the '<model> + TPSR' combined runs "
                         "(89M+TPSR, 145M+TPSR) from results/mymodels_tpsr/. Off by "
                         "default; complexity comes from the cache row augment_tpsr_caches.py adds.")
    ap.add_argument("--include-devncon", action="store_true",
                    help="Also include the '<model> + D&C' combined runs "
                         "(89M+D&C, 145M+D&C) from results/mymodels_devncon/. Off by "
                         "default.")
    ap.add_argument("--include-aifeynman", action="store_true",
                    help="Also include the AI Feynman 2.0 baseline from "
                         "results/aifeynman/. Off by default; with --reuse-cache the "
                         "complexity comes from the row augment_aifeynman_caches.py adds.")
    ap.add_argument("--include-phye2e", action="store_true",
                    help="Also include the PhyE2E baseline from results/phye2e/. Off by "
                         "default; with --reuse-cache the complexity comes from the row "
                         "augment_phye2e_caches.py adds.")
    ap.add_argument("--include-pysr", action="store_true",
                    help="Also include the PySR baseline from results/pysr/. Off by "
                         "default; with --reuse-cache the complexity comes from the row "
                         "augment_pysr_caches.py adds.")
    e2e_tpsr_addon.add_include_arg(
        ap, " With --reuse-cache its complexity comes from the row "
            "augment_e2e_tpsr_caches.py adds.")
    ap.add_argument("--include-llmsr", action="store_true",
                    help="Add a single-seed LLMSR-Llama row (search time + "
                         "llmsr_complexity), computed live from results/llmsr_llama/.")
    oneshot_overlay.add_include_arg(
        ap, " Adds one row per arm: search time + formula complexity, computed live "
            "from results/oneshot_<backbone>/.")
    ap.add_argument("--row-height", type=float, default=_ROW_H_AUTO,
                    help="Vertical inches allotted to each method row.  Unset (the "
                         "default), the figure takes the fixed canvas height it shares "
                         "with the SRBench twin; give a number (0.42 was the old "
                         "default) to size the figure by its row count instead.  Each "
                         "method sits on one data unit of the y axis, so this is what "
                         "actually sets how far apart the markers of neighbouring "
                         "methods are drawn.")
    ap.add_argument("--fig-height", type=float, default=None,
                    help="Total figure height in inches; overrides --row-height.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Keep only these method display labels. Applied after "
                         "--exclude, and before the opt-in overlay rows (LLMSR, "
                         "one-shot, pretrained), exactly as on the SRBench side.")
    ap.add_argument("--llmsr-backbones", nargs="+", default=None,
                    metavar="LABEL",
                    choices=list(llmsr_overlay.LLMSR_LABELS),
                    help="Which LLM-SR backbones --include-llmsr draws a row for. "
                         "Default: LLMSR-Llama only, which is what every figure drew "
                         "before this flag existed. Mirrors the curve grids' flag of "
                         "the same name.")
    ap.add_argument("--show", action="store_true")
    pretrained_addon.add_arguments(ap)
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


def draw_row(axL, axR, args, *, title_prefix="", fs=None, ms=None,
             time_xlabel="mean search time per instance (s)",
             cplx_xlabel="mean formula size (solved)"):
    """Draw the LLM-SRBench search-time (axL) and formula-size (axR) panels into the
    given axes -- the twin of plot_time_complexity_vs_noise.draw_row, split out for the
    same reason (plot_combined_time_complexity.py stacks the two rows under one
    Target-Noise legend).  Returns (noises, nstyle).
    """
    ms = _MS if ms is None else ms
    # The combined figure (plot_combined_time_complexity.py) draws on a wider canvas
    # than this script's own, so it passes the font scale that canvas needs.
    fs = _FS if fs is None else fs
    root = os.path.join(SCRIPT_DIR, "results")
    noises = sorted(set(args.noises))

    _phye2e_groups = phye2e_addon.active_groups(args) if args.include_phye2e else ()
    _pysr_groups = pysr_addon.active_groups(args) if args.include_pysr else ()
    st = _time_series(noises, args.split, root, args.include_tpsr_combo,
                      args.include_aifeynman, args.include_devncon,
                      _phye2e_groups, args.include_e2e_tpsr, _pysr_groups)
    sc = _complexity_series(noises, args.split, root, args.r2_thr, args.reuse_cache,
                            args.include_tpsr_combo,
                            args.include_aifeynman, args.include_devncon,
                            _phye2e_groups, args.include_e2e_tpsr, _pysr_groups)
    for ex in args.exclude:
        st.pop(ex, None)
        sc.pop(ex, None)
    # --only: restrict to the listed methods, as on the SRBench side.
    if getattr(args, "only", None):
        keep = set(args.only)
        st = {k: v for k, v in st.items() if k in keep}
        sc = {k: v for k, v in sc.items() if k in keep}
    # Opt-in single-seed LLMSR-Llama row (search time + llmsr_complexity), computed live.
    if getattr(args, "include_llmsr", False):
        # One row per requested backbone, as on the SRBench side.
        for _lbl in (getattr(args, "llmsr_backbones", None)
                     or [llmsr_overlay.DEFAULT_LABEL]):
            for tau in noises:
                t_stat, c_stat = llmsr_overlay.time_complexity_stat(
                    args.split, tau, args.r2_thr, label=_lbl)
                if t_stat is not None:
                    st.setdefault(_lbl, {})[tau] = t_stat
                if c_stat is not None:
                    sc.setdefault(_lbl, {})[tau] = c_stat
    # Opt-in one-shot rows, computed live from results/oneshot_<backbone>/ (nothing
    # about them is cached).
    if getattr(args, "include_oneshot", False):
        # See the note in plot_time_complexity_vs_noise.py: one-shot rows bypass the
        # dataframe-level --exclude, so the filtered label list must be passed here.
        oneshot_overlay.add_time_complexity_rows(
            st, sc, args.split, noises, args.r2_thr,
            labels=oneshot_overlay.visible_labels(args.exclude))
    # Opt-in NeSymReS / tf4sr / SymFormer rows, read live from their own trees (they
    # are not in results_io.GROUPS, so discovery never finds them).
    if getattr(args, "include_pretrained", False):
        common = (pretrained_addon.llmsr_common_problems(root, noises, args.split)
                  if getattr(args, "pretrained_common_subset", False) else None)
        pretrained_addon.add_llmsr_time_complexity_rows(
            st, sc, root, noises, args.split, args.r2_thr, common=common)
    if not st and not sc:
        raise SystemExit("No LLM-SRBench time/complexity found -- check results/ and --noises.")

    methods_set = set(st) | set(sc)

    def _rep(m):
        vals = [st[m][t][0] for t in noises if m in st and t in st[m]]
        return max(vals) if vals else 0.0
    methods = sorted(methods_set, key=_rep)
    ypos = {m: i for i, m in enumerate(methods)}
    my_models = {m for m in methods if _is_ours(m)}
    nstyle = _noise_style(noises)
    gt = _gt_complexity(noises[0], args.split, root)

    _draw_panel(axL, st, methods, ypos, noises, nstyle, my_models,
                f"{title_prefix}Synthesis time (seconds)",
                time_xlabel,
                show_ylabels=True, fs=fs, ms=ms)

    if gt is not None and np.isfinite(gt.get("q50", float("nan"))):
        q25, q50, q75 = gt["q25"], gt["q50"], gt["q75"]
        # The line and the IQR band stop short of the top (ymax in AXES fraction) so the
        # "GT median" label above them sits against clear panel, not on top of the dashes
        # and the grey.  _TOP_HEADROOM in _draw_panel is what keeps that strip empty of
        # data, and this keeps the annotation's own marks out of it too.
        axR.axvspan(q25, q75, ymax=_GT_MARK_TOP, color="#222", alpha=0.10, zorder=0)
        axR.axvline(q50, ymax=_GT_MARK_TOP, color="#222", lw=1.4, ls="--", zorder=1)
        #axR.text(q50, 0.98, f"GT median: {q50:.1f}  (IQR {q25:.0f}-{q75:.0f})",
        # Sits inside the top of the panel, in the headroom _draw_panel leaves free
        # (_TOP_HEADROOM), so it clears both the subplot title and the top row's points.
        # LEFT-aligned a few points to the RIGHT of the line, not centred on it: the GT
        # median sits near the low end of the formula-size axis, so a centred label spills
        # past x=0 and lands on the y axis / its tick labels (2026-09-23).  Anchoring the
        # left edge to the line keeps the whole string inside the panel whatever q50 is.
        axR.annotate("GT median", xy=(q50, 0.99),
                     xycoords=axR.get_xaxis_transform(),
                     xytext=(4, 0), textcoords="offset points",
                     ha="left", va="top",
                     fontsize=10 * fs, color="#222", fontweight="bold")
    _draw_panel(axR, sc, methods, ypos, noises, nstyle, my_models,
                f"{title_prefix}Formula size",
                cplx_xlabel,
                show_ylabels=False, fs=fs, ms=ms)
    return noises, nstyle, methods


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    # Canvas: the SAME size as the SRBench time/complexity figure
    # (plot_time_complexity_vs_noise.py, _FIG_W x _FIG_H).  The two face each other in
    # the paper, so emitting them at one size means LaTeX scales both by the same factor
    # at width=\textwidth and their fonts and markers print at matching sizes -- this
    # figure used to go out 13 x ~6.4 in against the other's 10 x 3.8 and printed
    # noticeably smaller.
    #
    # --row-height/--fig-height still override, for a one-off taller run: --row-height
    # gives `len(methods)` rows at that many inches each, plus the bottom legend strip
    # and the title/xlabel margins.
    # The row count only becomes known once the row is drawn, so the canvas starts at
    # the shared height and is resized afterwards when --row-height/--fig-height ask for
    # something else; tight_layout below lays the figure out at the final size.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H))
    noises, nstyle, methods = draw_row(axL, axR, args)
    fig_h = (args.fig_height if args.fig_height
             else (max(4.0, len(methods) * args.row_height + _LEGEND_IN + 1.05)
                   if args.row_height != _ROW_H_AUTO else _FIG_H))
    if fig_h != _FIG_H:
        fig.set_size_inches(_FIG_W, fig_h)

    handles = noise_legend_handles(noises, nstyle)
    fig.legend(handles=handles, title="Target Noise", loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               fontsize=11 * _FS, title_fontsize=11 * _FS, frameon=True, framealpha=0.9)

    fig.tight_layout(rect=(0, _LEGEND_IN / fig_h, 1, 1))
    save_fig(fig, os.path.join(SCRIPT_DIR, args.output))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
