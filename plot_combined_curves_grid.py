#!/usr/bin/env python
"""Combined OOD-accuracy-vs-gap figure: SRBench on top, LSR-Transform underneath.

This is set 1 of make_requested_plots.sh as of 2026-09-21.  It draws, on ONE canvas,
exactly what the two separate grids drew:

    row 1 (top)     the four SRBench noise panels        (plot_ood_vs_gap.draw_grid_row)
    row 2 (bottom)  the four LSR-Transform noise panels  (plot_llmsrbench_ood_vs_gap
                                                          .draw_grid_row_llmsr)

with ONE shared legend under both rows -- the point of merging them: the two benchmarks
draw largely the same arms, so a single figure spends one legend strip instead of two.

No plotting logic lives here.  Each row is drawn by the row-drawer the corresponding
standalone script exposes, and each row's arms are selected by that script's OWN CLI --
which matters, because the two filter differently: SRBench takes an --only whitelist,
LLM-SRBench an --exclude blacklist (see make_requested_plots.sh).  Rather than
re-declaring either flag set, this script takes the two command lines verbatim,
separated by `--`:

    python plot_combined_curves_grid.py --output plots/set1.png \
        -- <srbench args ...> -- <llmsrbench args ...>

Everything before the first `--` is this script's own; the --output of the two inner
command lines is ignored (nothing is written per row).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ood_transform
import plot_ood_vs_gap as sr
import plot_llmsrbench_ood_vs_gap as llm
import e2e_tpsr_addon
from plot_io import (save_fig, GRID_LEGEND_KW, GRID_LEGEND_NCOL, bottom_legend,
                     above_legend, sort_legend, display_labels, font_scale,
                     GRID_FIG_W_IN)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_FS = font_scale(GRID_FIG_W_IN)

# Two rows of panels plus the shared legend strip.  The legend's height is MEASURED
# (bottom_legend), so this only has to leave the panels room: taller than the single-row
# grids' GRID_ROW_FIG_H_IN by roughly one more row of panels.  Trimmed 8.6 -> 7.4
# (2026-09-21): the curves sit in the lower half of each panel, so a shorter panel costs
# no legibility and keeps the figure from eating a whole page column.
COMBINED_FIG_H_IN = 7.4

# Row identity goes in the leftmost panel's y-label rather than a row title: a title
# would sit between the two rows and read as if it belonged to the panels above it.
ROW_LABELS = ("SRBench", "LSR-Transform")

# One size for every piece of text that frames the panels: the shared x-label, the two
# row y-labels and the "noise = *" column titles.  They label the same grid, so they are
# set together here rather than inheriting the three different sizes the standalone
# scripts use (12 / 12 / 13) -- 2026-09-22.  Scoped to set 1; the standalone grids keep
# their own sizes in plot_ood_vs_gap.py / plot_llmsrbench_ood_vs_gap.py.
LABEL_FS = 9


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
        sys.exit("usage: plot_combined_curves_grid.py [--output PATH] "
                 "-- <srbench args> -- <llmsrbench args>")
    return parts


def main():
    own, sr_argv, llm_argv = _split_argv(sys.argv[1:])

    out = "plots/combined_curves_grid.png"
    if own:
        if own[0] != "--output" or len(own) != 2:
            sys.exit("the only option before the first `--` is --output PATH")
        out = own[1]

    sr_args = sr.parse_args(sr_argv)
    llm_args = llm.parse_args(llm_argv)
    gaps = sorted(set(sr_args.gaps or ood_transform.default_gaps()))
    llm_gaps = sorted(set(llm_args.gaps or ood_transform.default_gaps()))
    if gaps != llm_gaps:
        sys.exit(f"the two rows must share an x axis: --gaps {gaps} vs {llm_gaps}")

    # sharey is per ROW ('row'), not across the whole figure: the two benchmarks are
    # different problem sets and their solve rates are not on a common scale, which is
    # also how the two standalone figures drew them.
    fig, axes = plt.subplots(2, 4, figsize=(GRID_FIG_W_IN, COMBINED_FIG_H_IN),
                             sharex=True, sharey="row")

    seen_sr = sr.draw_grid_row(axes[0], sr_args, gaps,
                               os.path.join(SCRIPT_DIR, "results"))
    # Titles on the top row only -- the noise levels are the same four columns below.
    seen_llm = llm.draw_grid_row_llmsr(axes[1], llm_args, gaps, titles=False)

    for row, name in enumerate(ROW_LABELS):
        thr = (sr_args if row == 0 else llm_args).r2_thr
        axes[row][0].set_ylabel(f"{name}\nFraction with $R^2 \\geq$ {thr}",
                                fontsize=LABEL_FS * _FS)

    # The row-drawers sized the column titles and the tick numbers themselves; bring
    # both to LABEL_FS -- the ticks were 11 pt against 9 pt labels, which read as the
    # loudest text in the figure (2026-09-22).
    for ax in axes[0]:
        ax.title.set_fontsize(LABEL_FS * _FS)
    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=LABEL_FS * _FS)

    # One legend for both rows.  Keyed on the PRINTED label so an arm drawn in both rows
    # (most of them) contributes a single entry; the SRBench handle wins, and the two
    # rows draw each method in the same colour/marker, so the entry is right either way.
    # Each row's own rename runs first: both spell the standalone TPSR "e2e+TPSR".
    printed = {}
    for seen, rename in ((seen_sr, lambda ls: [sr.GRID_LEGEND_RENAME.get(l, l) for l in ls]),
                         (seen_llm, lambda ls: e2e_tpsr_addon.rename_legend(ls))):
        labels = display_labels(rename(list(seen.keys())))
        for lbl, h in zip(labels, seen.values()):
            printed.setdefault(lbl, h)

    _handles, _labels = sort_legend(list(printed.values()), list(printed.keys()))
    _leg, frac = bottom_legend(fig, _handles, _labels, ncol=GRID_LEGEND_NCOL,
                               **{**GRID_LEGEND_KW, "fontsize": LABEL_FS * _FS})
    fig.supxlabel("Extrapolation Constant (k)", fontsize=LABEL_FS * _FS,
                  y=above_legend(frac, fig))
    fig.tight_layout(rect=(0, frac, 1, 1))

    out_path = os.path.join(SCRIPT_DIR, out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_fig(fig, out_path)


if __name__ == "__main__":
    main()
