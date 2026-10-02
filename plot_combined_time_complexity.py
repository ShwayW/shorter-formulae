#!/usr/bin/env python
"""Combined search-time / formula-size figure: all four panels in ONE row.

This is set 2 of make_requested_plots.sh as of 2026-09-21, and it merges what used to be
sets 2 and 4:

    SRBench search time | SRBench formula size | LSR-Transform search time | LSR-Transform formula size

with ONE shared Target-Noise legend under the row -- the same merge, and the same
reason, as plot_combined_curves_grid.py does for the curve figures.

No plotting logic lives here: each row is drawn by the `draw_row` the corresponding
standalone script exposes, and each row's arms are selected by that script's OWN CLI
(SRBench takes an --only whitelist, LLM-SRBench an --exclude blacklist).  The two
command lines are therefore passed verbatim, separated by `--`:

    python plot_combined_time_complexity.py --output plots/set2.png \
        -- <srbench args ...> -- <llmsrbench args ...>

The --output of the two inner command lines is ignored (no per-row file is written).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_time_complexity_vs_noise as sr
import plot_llmsr_time_complexity_vs_noise as llm
from plot_time_complexity_vs_noise import (noise_legend_handles, _FIG_W, _FIG_H, _FS,
                                           _LEGEND_IN)
from plot_io import save_fig, GRID_FIG_W_IN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ONE ROW of four panels (2026-09-21; was 2x2, a row per benchmark).
#
# WIDER than the standalone figures' _FIG_W (10 in): two of the four panels carry method
# NAMES on their y axis ("ours (fine tuned, 23h)"), and at 10 in those labels leave the
# four panels a sliver of canvas each and the four titles overlap.
#
# NOTE this figure is ~2.5x as wide as it is tall, so at width=\textwidth it prints much
# smaller than the 2x2 version did -- give it a full-width float, or a landscape page.
COMBINED_FIG_W_IN = 18.0
# Height: one row of panels plus the legend strip.  This is the ROW PITCH knob -- the
# method rows are spread over whatever panel height is left, so lowering it tightens the
# vertical gaps between them.
COMBINED_FIG_H_IN = _FIG_H + 2.0

# Layout, in figure fractions.  NOT tight_layout: it gives every column the same gutter,
# and here only panels 1 and 3 carry method-name labels -- so the gutter the names need
# was being reserved four times over, leaving the panels barely half the canvas with
# dead space beside the two unlabelled ones.  A nested gridspec spends it where it is
# actually needed: a wide gap BETWEEN the two benchmark pairs (it holds the LSR-Transform
# method names), a narrow one INSIDE each pair.
# bottom is set by what has to fit UNDER the panels: tick labels, the x-axis label, and
# the one-row legend (~0.87 in) below it.  top is just the panel titles -- the benchmark
# headers sit above them at y=0.995.  Both were measured, not guessed: the legend used to
# be drawn over the second panel's x-axis label.
# left/_PAIR_GAP re-measured 2026-09-23, after the legend rename shortened every method
# name ("ours (fine tuned, unscaled)" -> "ours (unscaled)", "D&C (ours fine tuned)" ->
# "D&C (ours)").  The widest label now spans 117 of the 1175-unit page = 0.0996, where
# left=0.153 and a 0.46 pair gap had been sized for the old names -- about an inch of
# dead canvas on the left and as much again between the pairs.  Both now clear the
# measured label width with ~1% padding; re-measure if the names change again.
# Re-measured after the 1.55 font bump: the label column grew to 0.106 of the page,
# so left went 0.108 -> 0.118 to keep a real margin rather than 2 units of slack.
_MARGIN = dict(left=0.118, right=0.995, top=0.905, bottom=0.245)
_PAIR_GAP = 0.32       # between the two benchmark pairs, as a fraction of a pair's width
# The gap INSIDE a pair has to clear the two x-axis labels, not just the axes: at the
# type size below "mean search time per instance (s)" is wider than the panel it sits
# under, so it overhangs on both sides and would meet its neighbour's label.
_PANEL_GAP = 0.18      # between the two panels of one pair, as a fraction of panel width

# Shorter x-axis labels than the standalone figures use.  Theirs are ~3.7 in wide, which
# on a 2.5 in panel overhangs far enough to meet the neighbouring panel's label; the
# panel title above already says "Search time" / "Formula size", so the axis label only
# has to name the unit and the statistic.
_TIME_XLABEL = "mean per instance (s)"
_CPLX_XLABEL = "mean size (solved)"

# How far past the extreme data point each panel's x axis runs, as a multiplicative pad
# on a log axis.  _draw_panel starts every axis at 0, which on these panels leaves the
# left third to half of the box empty (the formula-size panels hold data in 10..100 but
# still draw the 0..10 stretch).  The combined figure is short of width, so it trims that
# instead: the axis starts just below the smallest point it has to show.  NOTE this is
# why this figure's axes do NOT begin at 0, unlike the standalone ones.
_X_PAD = 1.7

# Type size, as a multiple of the standalone figures' 10 in calibration (_FS).  Bumped
# 2026-09-21 so the labels carry on this wider canvas; the margins above pay for it --
# the method-name column and the gap between the pairs both hold text at this size.
# 1.55 since 2026-09-23 (was 1.25).  This figure is 18 in wide against set 1's 10, so at
# \textwidth it is scaled down ~1.8x more than set 1 is -- type calibrated to look right
# ON THE CANVAS therefore prints noticeably smaller than set 1's beside it.  The knob
# scales the method-name rows AND (through LABEL_FS * _FS_ROW) the titles, axis labels
# and ticks, so the margins below have to be re-measured whenever it changes.
_FS_SCALE = 1.55
_FS_ROW = _FS * _FS_SCALE

# Marker size for this figure only.  The row drawers default to their own _MS (8.5),
# sized for the standalone 10 in canvas; on this 18 in one -- scaled down ~1.8x more at
# \textwidth -- the four per-row markers print small.  Threaded through draw_row(ms=...)
# rather than raising _MS, so the standalone figures keep their calibration.
_MS_ROW = sr._MS * 1.35

# One size for the panel titles and axis labels, matching set 1's LABEL_FS (2026-09-22):
# the two figures sit next to each other in the paper, so their frame text prints the
# same.  draw_row sets its own 13 / 11 pt; those are overridden after the fact, below,
# so the standalone figures keep their sizes.
LABEL_FS = 9

# --- the STACKED variant (--layout stacked) ---------------------------------------
# Two rows, a benchmark each (SRBench over LSR-Transform), which is the shape this figure
# had before 2026-09-21 and which the appendix asked for again on 2026-09-24.  It is the
# right shape for a PORTRAIT float: at width=\textwidth the one-row form has to shrink
# ~2.5x, while this one only has to shrink the width of two panels.
#
# Narrower canvas, so the type reverts to the standalone 10 in calibration (_FS) instead
# of the one-row form's 1.25x bump -- see the note below on why the bump exists.
STACKED_FIG_W_IN = GRID_FIG_W_IN          # match the curve grids, so the appendix pair
                                          # prints at one width
# Per benchmark row: the standalone figure's panel height, plus room for its header and
# the row's x-axis label.  The legend strip is added on top of that once, not per row.
STACKED_ROW_H_IN = _FIG_H * 0.82
STACKED_FIG_H_IN = 2 * STACKED_ROW_H_IN + _LEGEND_IN
# Margins for the stacked form.  left is wider than a normal figure's because the LEFT
# panel of each row carries the method NAMES; right/top/bottom are what the x labels, the
# benchmark headers and the legend strip need.
STACKED_MARGIN = dict(left=0.26, right=0.985, top=0.93, bottom=0.115)
STACKED_ROW_GAP = 0.62     # between the two benchmark rows, as a fraction of a row's
                           # height: it holds the upper row's x-axis label AND the lower
                           # row's header
STACKED_PANEL_GAP = 0.10   # between the two panels of one row

# The type stays at the standalone figures' 10 in calibration (_FS) rather than being
# rescaled to this wider canvas: a panel here is ~3 in wide, and type sized for an 18 in
# canvas shrunk to \textwidth would print at ~34 pt titles that overlap their neighbours.
# The practical consequence is that this figure wants a WIDE float, not width=\textwidth
# on a portrait column.


def _tighten_x(ax, pad=_X_PAD):
    """Pull a dot panel's x limits in to the data it actually holds (plus `pad`x either
    side).  Reads the drawn artists rather than the series dicts, so it needs no second
    pass over the data and cannot disagree with what is on the canvas: the errorbar
    containers give the points, and a vertical line (the ground-truth median on a
    formula-size panel) must stay inside the box too."""
    xs = []
    for cont in ax.containers:                       # ErrorbarContainer per noise level
        xs.extend(x for x in cont.lines[0].get_xdata() if x and x > 0)
    for ln in ax.lines:                              # axvline: two identical x values
        xd = ln.get_xdata()
        if len(xd) == 2 and xd[0] == xd[1] and xd[0] > 0:
            xs.append(float(xd[0]))
    if not xs:
        return
    ax.set_xlim(min(xs) / pad, max(xs) * pad)


def _split_argv(argv):
    """argv -> (own, srbench, llmsrbench), split on the two bare `--` separators."""
    parts, cur = [], []
    for tok in argv:
        if tok == "--":
            parts.append(cur)
            cur = []
        else:
            cur.append(tok)
    parts.append(cur)
    if len(parts) != 3:
        sys.exit("usage: plot_combined_time_complexity.py [--output PATH] "
                 "-- <srbench args> -- <llmsrbench args>")
    return parts


def main():
    own, sr_argv, llm_argv = _split_argv(sys.argv[1:])

    out = "plots/combined_time_complexity.png"
    layout = "row"
    while own:
        if own[0] == "--output" and len(own) >= 2:
            out, own = own[1], own[2:]
        elif own[0] == "--layout" and len(own) >= 2:
            layout, own = own[1], own[2:]
            if layout not in ("row", "stacked"):
                sys.exit("--layout takes 'row' (default, one row of four panels) or "
                         "'stacked' (two rows, a benchmark each)")
        else:
            sys.exit("before the first `--` only --output PATH and --layout "
                     "row|stacked are accepted")

    sr_args = sr.parse_args(sr_argv)
    llm_args = llm.parse_args(llm_argv)

    # No sharing of any kind between the four panels: each carries its own method rows
    # (the two benchmarks rank the methods differently) and its own x scale (search time
    # spans different decades on the two benchmarks).  The benchmark name goes in each
    # panel's title, since the panels' y axis is already the method names.
    #
    # Panels 1 and 3 keep their method-name tick labels (draw_row gives its LEFT panel
    # show_ylabels=True), so the row reads as two benchmark pairs, not four loose panels.
    if layout == "stacked":
        # Two rows, a benchmark each.  Every row still holds its pair side by side, so
        # the row-drawers below are called exactly as in the one-row form -- only the
        # axes they are handed change.
        fig = plt.figure(figsize=(STACKED_FIG_W_IN, STACKED_FIG_H_IN))
        outer = fig.add_gridspec(2, 1, hspace=STACKED_ROW_GAP, **STACKED_MARGIN)
        axes = [fig.add_subplot(cell)
                for row in outer
                for cell in row.subgridspec(1, 2, wspace=STACKED_PANEL_GAP)]
        fs_row, ms_row = _FS, sr._MS      # 10 in canvas: the row drawers' own calibration
    else:
        fig = plt.figure(figsize=(COMBINED_FIG_W_IN, COMBINED_FIG_H_IN))
        outer = fig.add_gridspec(1, 2, wspace=_PAIR_GAP, **_MARGIN)
        axes = [fig.add_subplot(cell)
                for pair in outer
                for cell in pair.subgridspec(1, 2, wspace=_PANEL_GAP)]
        fs_row, ms_row = _FS_ROW, _MS_ROW

    # The benchmark names go ABOVE each pair, not into the two panel titles: prefixed
    # titles ("SRBench: Search time (seconds)") are wider than a panel here and run into
    # their neighbour.  The panel titles are then exactly the standalone figures'.
    _xlabels = dict(time_xlabel=_TIME_XLABEL, cplx_xlabel=_CPLX_XLABEL)
    noises, nstyle, _ = sr.draw_row(axes[0], axes[1], sr_args, fs=fs_row, ms=ms_row,
                                    **_xlabels)
    llm.draw_row(axes[2], axes[3], llm_args, fs=fs_row, ms=ms_row, **_xlabels)

    for ax in axes:
        _tighten_x(ax)
        # Titles and x labels to LABEL_FS, whatever draw_row sized them at.
        ax.title.set_fontsize(LABEL_FS * fs_row)
        ax.xaxis.label.set_fontsize(LABEL_FS * fs_row)
        # draw_row ticks at 11 pt; at 9 pt labels that made the numbers the loudest text.
        ax.tick_params(axis="both", labelsize=LABEL_FS * fs_row)
        # ...and the annotations draw_row adds itself.  "GT median" is a Text child, not a
        # title or an axis label, so the three overrides above miss it -- it was the one
        # string on this figure still at draw_row's own 10 pt while everything else had
        # been brought to LABEL_FS.
        for _t in ax.texts:
            _t.set_fontsize(LABEL_FS * fs_row)

    fig.canvas.draw()          # positions are only final once the figure has a renderer
    for (a, b), name in zip((axes[:2], axes[2:]), ("SRBench", "LSR-Transform")):
        x = 0.5 * (a.get_position().x0 + b.get_position().x1)
        # Stacked: each header sits just above ITS OWN row (the second one is halfway
        # down the canvas), so it cannot be pinned to the top of the figure the way the
        # one-row form's two headers both are.
        y = (a.get_position().y1 + 0.028) if layout == "stacked" else 0.995
        fig.text(x, y, name, ha="center", va="bottom" if layout == "stacked" else "top",
                 fontsize=LABEL_FS * fs_row, fontweight="bold")

    # One key for all four panels: the noise markers are the same in every one.  It stays
    # under the figure, but in a THIN strip: one row of four entries, not the two-row box
    # that used to cost ~1 in of height.  (It cannot go in the gap between the pairs --
    # the LSR-Transform method names fill that -- nor in the header band, where it lands
    # on the second panel's title.)
    handles = noise_legend_handles(noises, nstyle)
    fig.legend(handles=handles, title="Target Noise", loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               fontsize=LABEL_FS * fs_row, title_fontsize=LABEL_FS * fs_row, frameon=True,
               framealpha=0.9)

    # No tight_layout: _MARGIN above already leaves room for the titles, the x labels
    # and the legend strip, and letting tight_layout re-measure would undo the gridspec's
    # asymmetric gaps.
    out_path = os.path.join(SCRIPT_DIR, out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_fig(fig, out_path)


if __name__ == "__main__":
    main()
