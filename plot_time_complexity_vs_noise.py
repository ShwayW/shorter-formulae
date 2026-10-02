#!/usr/bin/env python3
"""
plot_time_complexity_vs_noise.py -- combined Fig-5-style figure with two panels:
inference time (left) and formula complexity (right), for every method across noise
levels.  It merges plot_time_vs_noise.py and plot_complexity_vs_noise.py into one 1x2
figure sharing a single method (y) axis, so a method's time and complexity read off
the same row.

Each panel is the usual Fig-5 dot plot: methods on y, metric on a log x-axis, one
point per target noise (x = 0.0, o = 0.001, s = 0.01, + = 0.1), +/-1 std error bars,
our models' labels bold.  The complexity panel keeps the ground-truth median dashed
line + IQR band.  One shared Target-Noise legend.

Usage:
    python plot_time_complexity_vs_noise.py [--noises 0 0.001 0.01 0.1]
                                            [--r2-threshold 0.99] [--reuse-cache]
                                            [--output plots/time_complexity_vs_noise.png]
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
import devncon_addon
import phye2e_addon
import pretrained_addon
import pysr_addon
import llmsr_overlay
import oneshot_overlay
from plot_io import save_fig, display_labels, font_scale
from compare_srbench import (
    SRBENCH_GT, load_srbench_gt, feynman_truth_complexity, _resolve_artifact,
)
from plot_ood_vs_gap import collect_timing, _remap_scale_labels, _label_for_path
from plot_time_vs_noise import _noise_style
from plot_complexity_vs_noise import _load_complexity

DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]
DEFAULT_R2_THR = 0.99

# Marker sizing (kept in step with plot_time_vs_noise.py / plot_complexity_vs_noise.py).
# These are the FOUR-NOISE figures' sizes (sets 2 and 4): each row carries four markers
# that overlap, so they are drawn small enough to stay separable.  The best-seed figure
# (set 5) plots one marker per row and passes its own, larger `ms` to _draw_panel.
_MS, _ELW, _CAPS, _MEW = 8.5, 1.6, 2.2, 1.5
_MS_SINGLE = 12        # one-marker-per-row panels (set 5)
_LEGEND_IN = 0.85      # inches reserved for the bottom legend strip (paper-sized text).
                       # Sized when the canvas was 7.6 in tall; at _FIG_H 5.4 the old
                       # 1.15 was ~21% of the figure and the drawn legend did not fill
                       # it, leaving a visible empty band under the x label.
# Canvas size.  The LLM-SRBench twin (plot_llmsr_time_complexity_vs_noise.py) imports
# these rather than repeating them, so the two figures cannot drift apart: they face each
# other in the paper and must print at one scale at width=\textwidth.  The height is
# fixed rather than derived from the row count for that same reason -- sets 2 and 4 carry
# 15 and 14 rows, so a common canvas keeps their scale identical and their row pitch
# within a few percent.  Raised 3.8 -> 5.0 (2026-09-12) to open the rows up, and 5.0 ->
# 7.6 when the labels were resized to print at the paper's own point sizes: 14-15 method
# rows at ~20 canvas points each need the canvas to pay for them.
# _FIG_H sets the ROW PITCH of the dot panels: the method rows are spread over a
# fixed canvas height, so trimming it tightens the vertical gaps.  Shared with the
# LLM-SRBench twin (it imports _FIG_H) so sets 2 and 4 cannot drift apart.
_FIG_W, _FIG_H = 10.0, 5.4
# Font sizes are written as the point size the text should have ON THE PAGE; _FS turns
# them into canvas points for this 10 in canvas, which LaTeX shrinks to \textwidth.
# _draw_panel takes it as `fs`, so the best-seed figure (set 5) can pass its own.
_FS = font_scale(_FIG_W)
_ZORDER = {0.01: 3, 0.001: 4, 0.0: 5, 0.1: 6}


def _is_ours(label):
    return label.startswith("145M") or label.startswith("89M")


def _time_series(noises, r2_thr, tpsr_csv, root, include_tpsr=False, include_devncon=False,
                 include_phye2e=False, include_pysr=False):
    """{method: {tau: (mean_time, std_time)}} via collect_timing (fast, no sympy)."""
    series = defaultdict(dict)
    for tau in noises:
        result_paths = results_io.list_base_srbench_pkls(root, tau)
        if include_tpsr:
            result_paths = list(result_paths) + tpsr_addon.srbench_tpsr_pkls(root, tau)
        if include_devncon:
            result_paths = list(result_paths) + devncon_addon.srbench_devncon_pkls(root, tau)
        srbench_agg = load_srbench_gt(SRBENCH_GT, tau, r2_thr)
        timing = collect_timing(result_paths, srbench_agg, r2_thr, tpsr_csv, noise=tau)
        remap = _remap_scale_labels([_label_for_path(p) for p in result_paths
                                     if os.path.exists(p)])

        def _disp(k, v):
            if tpsr_addon.is_tpsr_combo(k):
                return tpsr_addon.display_label(k)
            if devncon_addon.is_devncon_combo(k):
                return devncon_addon.display_label(k)
            # _label_for_path yields the bare TREE name ('phye2e', 'phye2e_units');
            # the caches and every other figure use the display label, so map each arm
            # to its own.  Mapping only the active group left a second arm labelled
            # 'phye2e_units', which then failed to join its cached complexity row and
            # drew an empty row.
            if k in include_phye2e:
                return phye2e_addon.label_for_group(k)
            if k in include_pysr:
                return pysr_addon.label_for_group(k)
            return v
        remap = {k: _disp(k, v) for k, v in remap.items()}
        for k, (m, s) in timing.items():
            if np.isfinite(m):
                series[remap.get(k, k)][tau] = (m, float(s) if np.isfinite(s) else 0.0)
        # PhyE2E arms are timed ONE AT A TIME.  Every tree stores its SRBench artifact
        # under the same file name, and collect_timing keys off _label_for_path (the
        # file name), so passing two arms in one call silently collapses them onto a
        # single 'phye2e' entry -- one arm's row then comes out empty.
        for _lbl, _p in phye2e_addon.srbench_pkls_for_groups(
                root, tau, include_phye2e or ()).items():
            _t = collect_timing([_p], srbench_agg, r2_thr, tpsr_csv, noise=tau)
            # collect_timing ALSO returns every SRBench published baseline, not just the
            # artifact asked about, so the one row we want is picked out by key -- taking
            # whatever iterates last hands both arms some unrelated baseline's timing.
            _m, _s = _t.get(_label_for_path(_p), (float("nan"), float("nan")))
            if np.isfinite(_m):
                series[_lbl][tau] = (_m, float(_s) if np.isfinite(_s) else 0.0)
        # PySR arms: same one-at-a-time treatment and for the same two reasons --
        # every tree stores its artifact under the same file name, and collect_timing
        # returns the published baselines alongside the artifact asked about.
        for _lbl, _p in pysr_addon.srbench_pkls_for_groups(
                root, tau, include_pysr or ()).items():
            _t = collect_timing([_p], srbench_agg, r2_thr, tpsr_csv, noise=tau)
            _m, _s = _t.get(_label_for_path(_p), (float("nan"), float("nan")))
            if np.isfinite(_m):
                series[_lbl][tau] = (_m, float(_s) if np.isfinite(_s) else 0.0)
    return series


def _complexity_series(noises, r2_thr, tpsr_csv, root, reuse_cache, include_tpsr=False,
                       include_devncon=False, include_phye2e=False,
                       include_pysr=False):
    """{method: {tau: (complexity, std)}} from the per-noise complexity cache; a
    missing noise is computed (sympy) and cached, mirroring plot_complexity_vs_noise.py."""
    series = defaultdict(dict)
    for tau in noises:
        cache = os.path.join(root, f"complexity_noise{tau:g}.csv")
        if reuse_cache and os.path.exists(cache):
            for r in pd.read_csv(cache).itertuples():
                # combo rows live in the cache (from augment_tpsr_caches.py) but
                # are only shown when opted in, so every other figure is unchanged.
                if tpsr_addon.is_tpsr_combo(r.algorithm) and not include_tpsr:
                    continue
                if devncon_addon.is_devncon_combo(r.algorithm) and not include_devncon:
                    continue
                # Same deal for the PhyE2E row augment_phye2e_caches.py adds.
                if phye2e_addon.is_phye2e(r.algorithm) and not include_phye2e:
                    continue
                # ... and the PySR row augment_pysr_caches.py adds.
                if pysr_addon.is_pysr(r.algorithm) and not include_pysr:
                    continue
                series[r.algorithm][tau] = (float(r.complexity), float(r.std))
        else:
            result_paths = results_io.list_base_srbench_pkls(root, tau)
            if include_tpsr:
                result_paths = list(result_paths) + tpsr_addon.srbench_tpsr_pkls(root, tau)
            if include_devncon:
                result_paths = list(result_paths) + devncon_addon.srbench_devncon_pkls(root, tau)
            if include_phye2e:
                result_paths = list(result_paths) + list(
                    phye2e_addon.srbench_pkls_for_groups(root, tau, include_phye2e).values())
            if include_pysr:
                result_paths = list(result_paths) + list(
                    pysr_addon.srbench_pkls_for_groups(root, tau, include_pysr).values())
            e2e_path = _resolve_artifact(results_io.e2e_srbench_pkl(root, tau))
            cplx, mine, _ = _load_complexity(result_paths, tpsr_csv, e2e_path, r2_thr, tau)
            # _load_complexity emits '89M_tpsr'/'89M_devncon'-style labels; give them the
            # '+TPSR'/'+D&C' display form (a no-op for every base label).
            cplx = {a.replace("_tpsr", "+TPSR").replace("_devncon", "+D&C"): v
                    for a, v in cplx.items()}
            # ... and each PhyE2E tree -> its display label, matching the cached rows.
            for _g in (include_phye2e or ()):
                if _g in cplx:
                    cplx[phye2e_addon.label_for_group(_g)] = cplx.pop(_g)
            for _g in (include_pysr or ()):
                if _g in cplx:
                    cplx[pysr_addon.label_for_group(_g)] = cplx.pop(_g)
            pd.DataFrame(
                [{"algorithm": a, "complexity": c, "std": s, "is_ours": a in mine}
                 for a, (c, s) in cplx.items()]
            ).to_csv(cache, index=False)
            for a, (c, s) in cplx.items():
                series[a][tau] = (c, s)
    return series


def _feynman_universe(noise):
    df = pd.read_feather(SRBENCH_GT)
    return set(df[(df["data_group"] == "Feynman")
                  & (df["target_noise"] == noise)]["dataset"].unique())


# Blank rows of headroom left above the top method row, so the "GT median" label of the
# complexity panel has a strip of its own.  Applied to BOTH panels of a figure, since the
# two share the y axis' row positions and must stay aligned.
_TOP_HEADROOM = 1.2

# Where the GT-median line and IQR band stop, as a fraction of the panel height: just
# below the "GT median" label that is drawn at 0.99 with va="top".
_GT_MARK_TOP = 0.92


def _draw_panel(ax, series, methods, ypos, noises, nstyle, my_models,
                title, xlabel, show_ylabels, fs=1.0, ms=_MS, mew=_MEW):
    """`fs` multiplies every font size in the panel (plot_io.font_scale).  It exists for
    the 16.5 in-wide set-5 figure, which LaTeX shrinks ~1.6x harder than the 10 in curve
    grids and so needs correspondingly larger nominal sizes to print at the same height.
    The default 1.0 leaves sets 2 and 4 exactly as they were."""
    ax.set_axisbelow(True)
    for m in methods:
        ax.axhline(ypos[m], color="0.78", linewidth=0.9, zorder=1)
    for m in methods:
        y0 = ypos[m]
        for t in noises:
            if m not in series or t not in series[m]:
                continue
            val, std = series[m][t]
            mk, col = nstyle[t]
            # Hollow face, as in the curve figures (plot_markers.HOLLOW): the four
            # noise levels of a row overlap, and an outline lets the ones underneath
            # show through.  "x"/"+" have no face to begin with and are unaffected.
            ax.errorbar(val, y0, xerr=std, marker=mk, markersize=ms, linestyle="none",
                        color=col, ecolor=col, elinewidth=_ELW, capsize=_CAPS,
                        markerfacecolor="none",
                        markeredgecolor=col, markeredgewidth=mew, zorder=_ZORDER.get(t, 3))
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.9)  # 0.9 = 1 - 1/10 makes 0->10^0 exactly one decade
    ax.set_xlim(left=0.0)   # axis starts at 0; no 10^-1 / 10^-2 labels
    ax.set_yticks(range(len(methods)))
    # Extra headroom above the top row: the complexity panel prints its "GT median"
    # label in that strip (just under the subplot title), so it collides with neither
    # the title nor the top method's points.
    ax.set_ylim(-0.6, len(methods) - 0.4 + _TOP_HEADROOM)
    if show_ylabels:
        # Rows print the DISPLAY name (plot_io.LEGEND_DISPLAY), the same text the curve
        # figures' legends use -- set 5's legend doubles as the key for these rows, so a
        # row saying "145M+TPSR" against a legend saying "TPSR (145M)" would not match.
        # The bold test still runs on the INTERNAL label, by row index.
        ax.set_yticklabels(display_labels(methods))
        for m, lbl in zip(methods, ax.get_yticklabels()):
            if m in my_models:
                lbl.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=13 * fs)
    ax.set_xlabel(xlabel, fontsize=11 * fs)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11 * fs)


def build_parser():
    """The CLI, split out of main() so plot_combined_time_complexity.py can parse exactly
    the same SRBench flags for the top row of the combined figure."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noises", nargs="+", type=float, default=DEFAULT_NOISES)
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    ap.add_argument("--tpsr-results",
                    default="TPSR/srbench_results/feynman_tpsr_l0.1_allnoise.csv")
    ap.add_argument("--output", default="plots/time_complexity_vs_noise.png")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None,
                    help="Keep only these method display labels (e.g. our models plus "
                         "the headline baselines: --only 145M 89M 89M-float e2e TPSR "
                         "AIFeynman Operon). Applied after --exclude.")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="Reuse the cached per-noise complexity "
                         "(results/complexity_noise<tau>.csv) instead of recomputing it.")
    ap.add_argument("--include-tpsr-combo", action="store_true",
                    help="Also include the '<model> + TPSR' combined runs "
                         "(89M+TPSR, 145M+TPSR) from results/mymodels_tpsr/. Off by "
                         "default; complexity comes from the cache row augment_tpsr_caches.py adds.")
    ap.add_argument("--include-devncon", action="store_true",
                    help="Also include the '<model> + D&C' combined runs "
                         "(89M+D&C, 145M+D&C) from results/mymodels_devncon/. Off by "
                         "default.")
    ap.add_argument("--include-phye2e", action="store_true",
                    help="Also include the PhyE2E baseline from results/phye2e/. Off by "
                         "default; with --reuse-cache the complexity comes from the row "
                         "augment_phye2e_caches.py adds.")
    ap.add_argument("--include-llmsr", action="store_true",
                    help="Add a single-seed LLMSR-Llama row (Feynman): search time + "
                         "llmsr_complexity, computed live from results/llmsr_llama/.")
    oneshot_overlay.add_include_arg(
        ap, " Adds one Feynman row per arm: search time + formula complexity, computed "
            "live from results/oneshot_<backbone>/.")
    ap.add_argument("--include-pysr", action="store_true",
                    help="Also include the PySR baseline from results/pysr/. Off by "
                         "default; with --reuse-cache the complexity comes from the row "
                         "augment_pysr_caches.py adds.")
    ap.add_argument("--llmsr-backbones", nargs="+", default=None,
                    metavar="LABEL",
                    choices=list(llmsr_overlay.LLMSR_LABELS),
                    help="Which LLM-SR backbones --include-llmsr draws a row for. "
                         "Default: LLMSR-Llama only, which is what every figure drew "
                         "before this flag existed. Mirrors the curve grids' flag of "
                         "the same name.")
    ap.add_argument("--show", action="store_true")
    phye2e_addon.add_group_arg(ap)
    pretrained_addon.add_arguments(ap)
    pysr_addon.add_group_arg(ap)
    return ap


def parse_args(argv=None):
    """Parse `argv` (default sys.argv) and apply the addons' group-arg post-processing."""
    args = build_parser().parse_args(argv)
    phye2e_addon.apply_group_arg(args)
    pysr_addon.apply_group_arg(args)
    return args


def draw_row(axL, axR, args, *, title_prefix="", fs=None, ms=None,
             time_xlabel="mean search time per instance (s)",
             cplx_xlabel="mean formula size (solved)"):
    """Draw the SRBench search-time (axL) and formula-size (axR) panels into given axes.

    Split out of main() so the combined time/complexity figure
    (plot_combined_time_complexity.py) can stack this row over the LLM-SRBench one and
    share a single Target-Noise legend.  `title_prefix` names the benchmark on a shared
    canvas ("SRBench: Search time (seconds)").  Returns (noises, nstyle), which is all
    the legend needs.
    """
    ms = _MS if ms is None else ms
    # The combined figure (plot_combined_time_complexity.py) draws on a wider canvas
    # than this script's own, so it passes the font scale that canvas needs.
    fs = _FS if fs is None else fs
    root = os.path.join(SCRIPT_DIR, "results")
    tpsr_csv = os.path.join(SCRIPT_DIR, args.tpsr_results)
    noises = sorted(set(args.noises))

    _phye2e_groups = phye2e_addon.active_groups(args) if args.include_phye2e else ()
    _pysr_groups = pysr_addon.active_groups(args) if args.include_pysr else ()
    st = _time_series(noises, args.r2_thr, tpsr_csv, root, args.include_tpsr_combo,
                      args.include_devncon, _phye2e_groups, _pysr_groups)
    sc = _complexity_series(noises, args.r2_thr, tpsr_csv, root, args.reuse_cache,
                            args.include_tpsr_combo, args.include_devncon,
                            _phye2e_groups, _pysr_groups)
    for ex in args.exclude:
        st.pop(ex, None)
        sc.pop(ex, None)
    # --only: restrict to the listed methods (e.g. a camera-ready subset).
    if args.only:
        keep = set(args.only)
        st = {k: v for k, v in st.items() if k in keep}
        sc = {k: v for k, v in sc.items() if k in keep}
    # Opt-in single-seed LLMSR-Llama row (search time + llmsr_complexity), added after
    # --only so the overlay is not filtered out; computed live from its raw artifacts.
    if getattr(args, "include_llmsr", False):
        # One row per requested backbone; the default list is just Llama, so a run
        # without --llmsr-backbones draws exactly what it drew before.
        for _lbl in (getattr(args, "llmsr_backbones", None)
                     or [llmsr_overlay.DEFAULT_LABEL]):
            for tau in noises:
                t_stat, c_stat = llmsr_overlay.time_complexity_stat(
                    "feynman", tau, args.r2_thr, label=_lbl)
                if t_stat is not None:
                    st.setdefault(_lbl, {})[tau] = t_stat
                if c_stat is not None:
                    sc.setdefault(_lbl, {})[tau] = c_stat
    # Opt-in one-shot rows, added after --only for the same reason, and computed live
    # from results/oneshot_<backbone>/ (nothing about them is cached).
    if getattr(args, "include_oneshot", False):
        # visible_labels(args.exclude): the one-shot arms are appended AFTER the
        # dataframe-level --only/--exclude filtering, so --exclude cannot reach them
        # on its own -- this figure has to ask for the filtered set explicitly.
        oneshot_overlay.add_time_complexity_rows(
            st, sc, "feynman", noises, args.r2_thr,
            labels=oneshot_overlay.visible_labels(args.exclude))
    # Opt-in NeSymReS / tf4sr / SymFormer rows.  Like the two overlays above they are
    # added AFTER --only, because neither the time nor the complexity cache knows these
    # trees -- the rows are read live from results/{nesymres,tf4sr,symformer}/.
    if getattr(args, "include_pretrained", False):
        common = (pretrained_addon.srbench_common_datasets(root, noises)
                  if getattr(args, "pretrained_common_subset", False) else None)
        pretrained_addon.add_srbench_time_complexity_rows(
            st, sc, root, noises, args.r2_thr, common=common)
    if not st and not sc:
        raise SystemExit("No time/complexity data found -- check results/ and --noises.")

    # Shared method axis (same rows in both panels), ordered by time (slowest at top).
    methods_set = set(st) | set(sc)

    def _rep(m):
        vals = [st[m][t][0] for t in noises if m in st and t in st[m]]
        return max(vals) if vals else 0.0
    methods = sorted(methods_set, key=_rep)
    ypos = {m: i for i, m in enumerate(methods)}
    my_models = {m for m in methods if _is_ours(m)}
    nstyle = _noise_style(noises)
    gt = feynman_truth_complexity(SRBENCH_GT, noises[0], _feynman_universe(noises[0]))

    _draw_panel(axL, st, methods, ypos, noises, nstyle, my_models,
                f"{title_prefix}Synthesis time (seconds)",
                time_xlabel,
                show_ylabels=True, fs=fs, ms=ms)

    # GT median dashed line + IQR band on the complexity panel, behind the points.
    if gt is not None and np.isfinite(gt.get("q50", float("nan"))):
        q25, q50, q75 = gt["q25"], gt["q50"], gt["q75"]
        # The line and the IQR band stop short of the top (ymax in AXES fraction) so the
        # "GT median" label above them sits against clear panel, not on top of the dashes
        # and the grey.  _TOP_HEADROOM in _draw_panel is what keeps that strip empty of
        # data, and this keeps the annotation's own marks out of it too.
        axR.axvspan(q25, q75, ymax=_GT_MARK_TOP, color="#222", alpha=0.10, zorder=0)
        axR.axvline(q50, ymax=_GT_MARK_TOP, color="#222", lw=1.4, ls="--", zorder=1)
        # Sits inside the top of the panel, in the headroom _draw_panel leaves free
        # (_TOP_HEADROOM), so it clears both the subplot title and the top row's points.
        #axR.text(q50, 1.0, f"GT median ~ {q50:.1f}  (IQR {q25:.0f}-{q75:.0f})",
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


def noise_legend_handles(noises, nstyle):
    """The Target-Noise key: one hollow marker per noise level.  Shared with the
    LLM-SRBench twin and with the combined figure, which draws it once for both rows."""
    return [
        Line2D([0], [0], marker=nstyle[t][0], linestyle="none", color=nstyle[t][1],
               markersize=_MS, markerfacecolor="none", markeredgewidth=_MEW,
               label=(f"{t:g}" if t != 0 else "0.0"))
        for t in noises
    ]


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    # No sharey: both panels use identical row positions (ypos) and ylim, so rows
    # align, but each keeps its own tick labels (left shows names, right hides them).
    fig_h = _FIG_H
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(_FIG_W, fig_h))
    noises, nstyle, _methods = draw_row(axL, axR, args)

    handles = noise_legend_handles(noises, nstyle)
    fig.legend(handles=handles, title="Target Noise", loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               fontsize=11 * _FS, title_fontsize=11 * _FS, frameon=True, framealpha=0.9)

    fig.tight_layout(rect=(0, _LEGEND_IN / fig_h, 1, 1))
    out_path = os.path.join(SCRIPT_DIR, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_fig(fig, out_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
