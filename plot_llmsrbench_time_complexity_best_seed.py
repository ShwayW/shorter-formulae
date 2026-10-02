#!/usr/bin/env python3
"""
plot_llmsrbench_time_complexity_best_seed.py -- combined LLM-SRBench best-seed figure with
THREE panels: OOD accuracy vs gap (left, the set-5 curves), search time (middle), and
formula complexity (right), at noise 0 for e2e / 89M / 145M / 89M+TPSR / 145M+TPSR and the
LLMSR backbones (Gemini, Llama-3.1-8B).  Each method is taken from its SINGLE BEST SEED
(highest gap-0 OOD solve rate) -- exactly the seed plot_llmsrbench_curves_best_seed.py
draws; the left panel reuses that script's draw_best_seed_curves so the curves match.

Both panels are the usual dot plot: methods on y, metric on a log x-axis, one point per
method with +/-1 std error bars (std over that seed's per-problem values), our models
bold.  The complexity panel keeps the ground-truth median dashed line + IQR band.

  * search time = the best seed's search_time per problem (mean +/- std over problems)
  * complexity  = node count of the best seed's discovered_equation on the problems it
                  solved (id_metrics r2 >= threshold); LLMSR programs are scored with
                  llmsr_complexity (symbolic skeleton of the `def equation` body), the
                  others with compare_srbench.formula_complexity.

Usage:
    python plot_llmsrbench_time_complexity_best_seed.py
    PLOT_FORMAT=pdf python plot_llmsrbench_time_complexity_best_seed.py \
        --output plots/llmsrbench_time_complexity_noise0_bestseed.pdf
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import results_io
import tpsr_addon
import aifeynman_addon
import phye2e_addon
import pysr_addon
import e2e_tpsr_addon
import devncon_addon
import oneshot_overlay
from plot_io import (save_fig, sort_legend, display_labels, GRID_LEGEND_KW,
                     font_scale, fit_rect)
from results_io import load_rows_any
from compare_srbench import formula_complexity
from llmsr_complexity import llmsr_complexity
from plot_llmsrbench_ood_vs_gap import discover_methods_from_dirs, _SKIP_SUBSTR
from plot_time_vs_noise import _noise_style
from plot_time_complexity_vs_noise import (_draw_panel, _is_ours, _MS_SINGLE,
                                            _GT_MARK_TOP)
from plot_llmsr_time_complexity_vs_noise import _gt_complexity
from plot_llmsrbench_curves_best_seed import (
    load_combined_raw, best_seed_per_method, best_seed_curves, draw_best_seed_curves,
    DEFAULT_METHODS, DEFAULT_R2_THR, DEFAULT_GAPS, _LLMSR_BACKBONES, _is_llmsr,
    E2E_TPSR_LABEL, ONESHOT_LABELS, apply_monotone_gap)

# Canvas width, and the font scale that goes with it.  Every font size below is written
# as the point size the text should have ON THE PAGE, times _FS, so all three figures
# print at matching sizes -- and at the paper's own 10-13 pt -- once LaTeX has scaled each
# to \textwidth.  See plot_io.font_scale.
_FIG_W = 10.0
_FS = font_scale(_FIG_W)

# Panel-frame type size, shared with sets 1 and 2 (see the override in main()).
LABEL_FS = 9

# Row pitch of the two dot panels, and of the legend strip.  The figure height follows
# from them, so adding an arm grows the figure instead of squeezing rows.  0.34 in is
# what a row label set to print at 11 pt needs on this 10 in canvas.
_ROW_IN = 0.28         # inches per method row in a dot panel
_CURVE_IN = 3.5        # height of the curve panel's row

# Height the legend cell needs, derived from GRID_LEGEND_KW rather than guessed, so that
# restyling the shared legend cannot silently make the box outgrow its cell.
_LEG_FS    = LABEL_FS * _FS        # the strip is drawn at this size, see main()
_LEG_PITCH = _LEG_FS * (1.0 + GRID_LEGEND_KW["labelspacing"]) / 72.0    # per entry
_LEG_PAD   = 2.0 * GRID_LEGEND_KW["borderpad"] * _LEG_FS / 72.0        # per box
# Columns the legend strip starts at; _draw_legend_strip drops one at a time until the
# box fits the canvas, exactly as plot_io.bottom_legend does for the grids.
_LEG_NCOL = 3


def _legend_cell_in(n_entries, ncol=_LEG_NCOL):
    """Inches of row height the legend strip needs for `n_entries` methods."""
    rows = -(-n_entries // max(1, ncol))
    return rows * _LEG_PITCH + _LEG_PAD + 0.25


def _draw_legend_strip(fig, ax, handles, labels, **kw):
    """Draw the method key centred in its own full-width row, narrowing it by a column
    at a time until it fits the canvas.  A box wider than the figure would be caught by
    save_fig's tight bbox and widen the saved page -- which LaTeX pays back as a harder
    downscale and smaller print, the very thing the paper sizes are here to avoid."""
    ncol = _LEG_NCOL
    while True:
        leg = ax.legend(handles, labels, loc="center", ncol=ncol, **kw)
        fig.canvas.draw()
        if ncol == 1 or leg.get_window_extent().width <= fig.get_figwidth() * fig.dpi:
            return leg
        leg.remove()
        ncol -= 1


def _method_artifacts(root, keep, phye2e_groups=None, pysr_groups=None):
    """{display_label: results_artifact_path} for the requested methods at noise 0.

    The LLMSR backbones (Gemini, Llama) live in the 'llmsr'/'llmsr_llama' groups, and the
    '<model> + TPSR' combined runs live in 'mymodels_tpsr' (dir names carry 'tpsr'), none
    of which the default discovery (mymodels/e2e) sweeps -- so all are wired in by hand."""
    out = {}
    dirs = results_io.list_llmsr_method_dirs(root, 0.0, "lsr_transform")
    for label, jl in discover_methods_from_dirs(dirs, "lsr_transform", _SKIP_SUBSTR).items():
        if label in keep:
            out[label] = jl
    for label, spec in _LLMSR_BACKBONES.items():
        if label in keep:
            p = os.path.join(root, spec["artifact"])
            if os.path.exists(p):
                out[label] = p
    # "89M+TPSR" / "145M+TPSR": scored with formula_complexity
    # (prefix-notation transformer output), not llmsr_complexity -- _is_llmsr() is False.
    for label, art in tpsr_addon.llmsr_tpsr_methods(root, 0.0, "lsr_transform").items():
        if label in keep:
            out[label] = art
    # "89M+D&C" / "145M+D&C": same story -- transformer output, so formula_complexity.
    for label, art in devncon_addon.llmsr_devncon_methods(root, 0.0, "lsr_transform").items():
        if label in keep:
            out[label] = art
    # The AI Feynman baseline lives in its own 'aifeynman' group, also outside the
    # default discovery.  Its formulas are infix sympy, which formula_complexity
    # handles via its sympify fallback -- _is_llmsr() is False, so that is what runs.
    for label, art in aifeynman_addon.llmsr_aifeynman_methods(
            root, 0.0, "lsr_transform").items():
        if label in keep:
            out[label] = art
    # PhyE2E: likewise its own group, outside the default discovery.  Its formulas are
    # infix in x_0..x_N (the e2e convention), so formula_complexity is the right counter
    # -- _is_llmsr() is False for this label.
    for label, art in phye2e_addon.llmsr_methods_for_groups(
            root, 0.0, "lsr_transform", phye2e_groups or (phye2e_addon.PHYE2E_GROUP,)).items():
        if label in keep:
            out[label] = art
    # PySR: likewise its own group, outside the default discovery.  Its formulas are
    # infix in x_0..x_N (the e2e convention), so formula_complexity is the right counter
    # -- _is_llmsr() is False for this label.  NB this counts the sympy node count, NOT
    # the `complexity` field eval_pysr.py stores from PySR's own Pareto front, which is
    # a different metric and must never be mixed into these figures (see pysr_addon).
    for label, art in pysr_addon.llmsr_methods_for_groups(
            root, 0.0, "lsr_transform", pysr_groups or (pysr_addon.PYSR_GROUP,)).items():
        if label in keep:
            out[label] = art
    # The standalone TPSR baseline lives in the 'tpsr' group.  That group IS swept by
    # the default discovery above, but _SKIP_SUBSTR drops its dir on the 'tpsr'
    # substring, so it is wired in by hand like the rest.  Its formulas are infix in
    # x_0..x_N (the e2e convention), so formula_complexity is the right counter --
    # _is_llmsr() is False for this label.
    for label, art in e2e_tpsr_addon.llmsr_methods(root, 0.0, "lsr_transform").items():
        if label in keep:
            out[label] = art
    # The one-shot LLM arms live in their own 'oneshot_<backbone>' groups, outside the
    # default discovery.  Their formulas are concrete infix sympy, so formula_complexity
    # is the right counter -- _is_llmsr() is False for these labels.
    for label in ONESHOT_LABELS:
        if label not in keep:
            continue
        p = oneshot_overlay.artifact(label, "lsr_transform", 0.0)
        if os.path.exists(p):
            out[label] = p
    return out


def _time_complexity_for_seed(jl, seed, thr, is_llmsr):
    """(time_mean, time_std), (cplx_mean, cplx_std) for one method's best seed.

    time  = search_time over every problem that seed ran; complexity = node count over
    the problems it solved (id r2 >= thr).  std is across problems (single seed)."""
    _meta, rows = load_rows_any(jl)
    cplx_fn = llmsr_complexity if is_llmsr else formula_complexity
    times, cplx = [], []
    for r in rows:
        if int(r.get("seed", 0) or 0) != seed:
            continue
        t = r.get("search_time")
        if t is not None and np.isfinite(t):
            times.append(float(t))
        r2 = (r.get("id_metrics") or {}).get("r2")
        if r2 is not None and np.isfinite(r2) and r2 >= thr:
            c = cplx_fn(r.get("discovered_equation"))
            if np.isfinite(c):
                cplx.append(float(c))
    t_stat = (float(np.mean(times)), float(np.std(times))) if times else None
    c_stat = (float(np.mean(cplx)), float(np.std(cplx))) if cplx else None
    return t_stat, c_stat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    ap.add_argument("--no-monotone-gap", action="store_true",
                    help="Score each gap independently, allowing a formula that "
                         "failed at a smaller gap to count as solved at a larger "
                         "one. Off by default (ood_transform.monotone_gap).")
    ap.add_argument("--only", nargs="*", default=DEFAULT_METHODS,
                    help="Method display labels to include (default: 145M 89M e2e LLMSR).")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Method display labels to drop, applied AFTER --only and after "
                         "the --include-* addons have appended their labels. Lets a "
                         "caller thin the default list without restating it.")
    ap.add_argument("--include-aifeynman", action="store_true",
                    help="Also include the AI Feynman 2.0 baseline (results/aifeynman/). "
                         "Off by default.")
    ap.add_argument("--include-phye2e", action="store_true",
                    help="Also include the PhyE2E baseline (results/phye2e/). "
                         "Off by default.")
    ap.add_argument("--include-pysr", action="store_true",
                    help="Also include the PySR baseline (results/pysr/). Off by default.")
    e2e_tpsr_addon.add_include_arg(
        ap, " Its rows reach the base cache via augment_e2e_tpsr_caches.py.")
    oneshot_overlay.add_include_arg(
        ap, " Their OOD rows come from their own per-backbone caches; time and "
            "complexity are computed live from results/oneshot_<backbone>/.")
    ap.add_argument("--output",
                    default="plots/llmsrbench_time_complexity_noise0_bestseed.png")
    ap.add_argument("--show", action="store_true")
    aifeynman_addon.add_group_arg(ap)
    phye2e_addon.add_group_arg(ap)
    pysr_addon.add_group_arg(ap)
    args = ap.parse_args()
    aifeynman_addon.apply_group_arg(args)
    phye2e_addon.apply_group_arg(args)
    pysr_addon.apply_group_arg(args)
    if args.include_aifeynman and aifeynman_addon.DISPLAY_LABEL not in args.only:
        args.only = list(args.only) + [aifeynman_addon.DISPLAY_LABEL]
    if args.include_phye2e:
        _labels = ([phye2e_addon.label_for_group(g) for g in args.phye2e_groups]
                   if args.phye2e_groups else [phye2e_addon.DISPLAY_LABEL])
        args.only = list(args.only) + [l for l in _labels if l not in args.only]
    if args.include_pysr:
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
    keep = set(args.only) - set(args.exclude)

    # Best seed per method = highest gap-0 OOD solve rate (same choice as the curves).
    raw = load_combined_raw(root)
    raw = raw[raw["algorithm"].isin(keep)]
    # Prefix-AND the gap axis ONCE, here: both the best-seed choice below and the curve
    # panel's rates are counted off this table, so applying it at the source keeps the
    # two consistent (see ood_transform.monotone_gap).
    raw = apply_monotone_gap(raw, args.r2_thr, monotone=not args.no_monotone_gap)
    best = best_seed_per_method(raw, args.r2_thr)

    artifacts = _method_artifacts(root, keep, args.phye2e_groups, args.pysr_groups)
    st, sc = defaultdict(dict), defaultdict(dict)
    for label, jl in artifacts.items():
        seed = best.get(label)
        if seed is None:
            print(f"[warn] no best seed for {label} (missing from OOD cache) -- skipping")
            continue
        t_stat, c_stat = _time_complexity_for_seed(
            jl, seed, args.r2_thr, is_llmsr=_is_llmsr(label))
        if t_stat is None and c_stat is None:
            print(f"[warn] no rows for {label} at seed {seed} "
                  f"(seed mismatch between OOD cache and results.pkl.gz?) -- skipping")
            continue
        if t_stat is not None:
            st[label][0.0] = t_stat
        if c_stat is not None:
            sc[label][0.0] = c_stat
        t_str = f"{t_stat[0]:.3g}s" if t_stat else "n/a"
        c_str = f"{c_stat[0]:.1f}" if c_stat else "n/a"
        print(f"  {label:8} seed={seed:<3} time={t_str}  complexity={c_str}")

    if not st and not sc:
        raise SystemExit("No time/complexity data -- check results/ and the OOD caches.")

    methods_set = set(st) | set(sc)

    def _rep(m):
        return st[m][0.0][0] if m in st and 0.0 in st[m] else 0.0
    methods = sorted(methods_set, key=_rep)          # slowest at top
    ypos = {m: i for i, m in enumerate(methods)}
    my_models = {m for m in methods if _is_ours(m)}
    noises = [0.0]
    nstyle = _noise_style(noises)
    gt = _gt_complexity(0.0, "lsr_transform", root)

    # Best-seed OOD-accuracy-vs-gap curves for the LEFT panel -- same data and best-seed
    # choice as the standalone set-5 figure, so all three subplots live in one figure.
    curves = best_seed_curves(raw, args.r2_thr, best)
    gaps = sorted(DEFAULT_GAPS)
    curve_algs = sorted(curves, key=lambda a: -curves[a].get(0.0, 0.0))

    # Three full-width rows:
    #
    #     OOD accuracy vs gap
    #     legend  (the method key, serving all three panels)
    #     search time  |  formula complexity
    #
    # The old 1x3 row had to be 16.5 in wide to fit three panels plus the method labels,
    # which is 1.65x more than sets 1/3 -- and since every figure goes in at
    # width=\textwidth, that width is paid straight back as a harder downscale and
    # smaller print.  The 2x2 that replaced it stood the legend BESIDE the curves, which
    # worked only while the key was set at 9 pt; sized to print at the paper's 10 pt, two
    # columns of method names no longer fit next to a panel, so the key gets a row of its
    # own like the grids' bottom strip.
    #
    # The two variable rows are driven by the number of methods -- the dot panels need a
    # row each, and the key needs one line per arm it names -- so an extra arm makes the
    # figure taller rather than making its rows collide.
    n_meth = len(methods)
    h_leg = _legend_cell_in(n_meth)
    h_bot = max(2.4, n_meth * _ROW_IN + 1.25)
    fig = plt.figure(figsize=(_FIG_W, _CURVE_IN + h_leg + h_bot))
    gs = fig.add_gridspec(3, 1, height_ratios=[_CURVE_IN, h_leg, h_bot])
    # wspace above the default: the two x labels are centred under their own panels and
    # at paper point sizes they meet in the middle of the row.
    gs_bot = gs[2].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.32)
    axCurve = fig.add_subplot(gs[0])
    axLegend = fig.add_subplot(gs[1])
    axTime = fig.add_subplot(gs_bot[0])
    axCplx = fig.add_subplot(gs_bot[1])
    axLegend.set_axis_off()          # a cell that holds only the legend

    # No panel title (2026-09-21): the axis labels already say what the panel shows, and
    # the caption names the benchmark/noise/seed -- same reasoning as the missing suptitle.
    draw_best_seed_curves(axCurve, curves, curve_algs, gaps, args.r2_thr,
                          title=None, legend=False, fs=_FS)

    _draw_panel(axTime, st, methods, ypos, noises, nstyle, my_models,
                "Synthesis time (seconds)",
                "mean synthesis time per instance (s)",
                show_ylabels=True, fs=_FS, ms=_MS_SINGLE, mew=2.0)

    if gt is not None and np.isfinite(gt.get("q50", float("nan"))):
        q25, q50, q75 = gt["q25"], gt["q50"], gt["q75"]
        # The line and the IQR band stop short of the top (ymax in AXES fraction) so the
        # "GT median" label above them sits against clear panel, not on top of the dashes
        # and the grey.  _TOP_HEADROOM in _draw_panel is what keeps that strip empty of
        # data, and this keeps the annotation's own marks out of it too.
        axCplx.axvspan(q25, q75, ymax=_GT_MARK_TOP, color="#222", alpha=0.10, zorder=0)
        axCplx.axvline(q50, ymax=_GT_MARK_TOP, color="#222", lw=1.4, ls="--", zorder=1)
        # Sits inside the top of the panel, in the headroom _draw_panel leaves free
        # (_TOP_HEADROOM), so it clears both the subplot title and the top row's points.
        axCplx.text(q50, 0.99, "GT median", transform=axCplx.get_xaxis_transform(),
                    ha="center", va="top", fontsize=10 * _FS, color="#222",
                    fontweight="bold")
    _draw_panel(axCplx, sc, methods, ypos, noises, nstyle, my_models,
                "Formula size",
                "mean formula size (solved)",
                show_ylabels=False, fs=_FS, ms=_MS_SINGLE, mew=2.0)

    # Panel titles and axis labels to one size, matching set 1's and set 2's LABEL_FS
    # (2026-09-22): the three figures share a paper, so their frame text prints the same.
    # The row drawers size their own (12 / 13 / 11 pt); this overrides them here only.
    for ax in (axCurve, axTime, axCplx):
        ax.title.set_fontsize(LABEL_FS * _FS)
        ax.xaxis.label.set_fontsize(LABEL_FS * _FS)
        ax.yaxis.label.set_fontsize(LABEL_FS * _FS)
        # And the tick numbers, drawn at 11 pt by the row drawers.
        ax.tick_params(axis="both", labelsize=LABEL_FS * _FS)

    _h, _l = axCurve.get_legend_handles_labels()
    # Sorted by the PRINTED label (post-rename), as in sets 1/3.
    m_handles, m_labels = sort_legend(
        _h, display_labels(e2e_tpsr_addon.rename_legend(_l)))
    # The legend row carries the method key only: it colours the curves AND names the
    # rows of the two dot panels, so it serves all three.  The dot panels get no key of
    # their own -- every point in them is the one style, and the caption says which
    # (noise 0, best seed per method).  Same styling as the grids' legends.
    _draw_legend_strip(fig, axLegend, m_handles, m_labels,
                       **{**GRID_LEGEND_KW, "handlelength": 1.8,
                          "fontsize": LABEL_FS * _FS})

    # No suptitle: the paper's caption already says which benchmark, noise and seed
    # selection this figure is, and dropping it buys back a strip of vertical space.
    fit_rect(fig, (0, 0, 1, 1))
    save_fig(fig, os.path.join(SCRIPT_DIR, args.output))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
