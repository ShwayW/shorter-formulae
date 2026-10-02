"""
plot_io.py -- one figure saver shared by every plotting script.

Output is ALWAYS PDF.  Whatever extension an --output path carries (.png from the
old defaults, .svg, or none at all) is rewritten to .pdf, so every script emits a
single vector file suitable for the paper.  The PLOT_FORMAT env var is no longer
consulted; setting it has no effect.

pil_kwargs={"optimize": True} is a PNG-only option (matplotlib's PDF canvas raises
a TypeError if it is passed), so it is not used.
"""
import os
import re

import matplotlib
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox

# --- paper font ---------------------------------------------------------------------
# The figures go into papers/iclr2027, whose preamble is
# "\usepackage{iclr2027_conference,times}" -- a 10 pt article set in Times.  Matplotlib's
# default DejaVu Sans reads as a different document once the PDF is dropped into that
# page, so every figure this module saves is drawn in a Times clone instead.
#
# "Liberation Serif" is metric-identical to Times New Roman and visually the same face as
# the "Nimbus Roman" pdflatex substitutes for \usepackage{times}, and it is a TrueType
# file -- Nimbus ships here as OTF/CFF, which matplotlib embeds under a Type 42 tag that
# makes poppler (and some print pipelines) warn about a font-type mismatch.  Nimbus is
# kept next in line for machines without Liberation.  Math text uses the STIX set, which
# is Times-metric too, so an "$R^2$" label matches the body font instead of switching to
# DejaVu.
#
# Set at IMPORT time, so any script that saves through save_fig picks it up without
# having to opt in.  Sizes are untouched -- each figure already scales its own.
PAPER_SERIF = ["Liberation Serif", "Times New Roman", "Nimbus Roman",
               "STIXGeneral", "DejaVu Serif"]
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": PAPER_SERIF,
    "mathtext.fontset": "stix",
    # Keep PDF text as text (Type 42/TrueType) rather than outlines, so the figure's
    # glyphs stay searchable and selectable in the compiled paper.
    "pdf.fonttype": 42,
})

_KNOWN = ("pdf", "svg", "png")
_FMT = "pdf"


def _decorations(ax):
    """The artists that stick out past an axes' own box: its title, its axis labels, and
    EVERY legend parented to it.  ax.get_legend() returns only the most recent one, so a
    cell holding two legends (the best-seed figure's legend panel) would have had its
    first one silently unmeasured -- and then clipped."""
    arts = [ax.title, ax.xaxis.label, ax.yaxis.label]
    arts += [c for c in ax.get_children() if isinstance(c, Legend)]
    return [a for a in arts
            if a.get_visible() and getattr(a, "get_text", lambda: "x")()]


_SAVE_PAD_IN = 0.1     # matplotlib's own default pad for bbox_inches="tight"


def _full_bbox(fig, pad_in=_SAVE_PAD_IN):
    """The figure's tight bbox, in inches, UNION the decorations fig.get_tightbbox()
    leaves out.

    get_tightbbox misses a subplot's x-axis label when that subplot carries no y tick
    labels, so bbox_inches="tight" would crop the page right through it -- which is how
    the best-seed figure's "Avg formula complexity (solved)" lost its closing bracket.
    Unioning the titles, axis labels and in-panel legends back in can only ever GROW the
    box, and only where something was about to be cut, so every other figure is
    byte-for-byte what it was.

    The box is padded here because savefig applies its `pad_inches` only to the string
    form "tight", not to an explicit Bbox -- without this every figure would come out
    0.2 in smaller than before and butt right up against its own frame."""
    fig.canvas.draw()
    bb = fig.get_tightbbox()
    to_in = fig.dpi_scale_trans.inverted()
    for ax in fig.axes:
        for art in _decorations(ax):
            bb = Bbox.union([bb, art.get_window_extent().transformed(to_in)])
    return bb.padded(pad_in)


def save_fig(fig, path, *, dpi=100, quiet=False):
    """Save `fig` to `path` as PDF, rewriting the extension if needed.  Returns the
    actual path written.  `dpi` is accepted for call-site compatibility and ignored."""
    # Only a RECOGNISED trailing token is an extension. A noise-suffixed stem such as
    # "ood_vs_gap_oneshot_feynman_noise0.001" splits to ext=".001", which is part of the
    # name -- stripping it silently collapsed every tau onto one "..._noise0.pdf" file.
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext not in _KNOWN:
        path = f"{path}.{_FMT}"         # append, never replace
    elif ext != _FMT:
        path = os.path.splitext(path)[0] + "." + _FMT

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches=_full_bbox(fig), pad_inches=0)
    if not quiet:
        print(f"[plot] Saved -> {path}")
    return path


PAPER_TEXTWIDTH_IN = 5.5   # \textwidth of papers/iclr2027/iclr2027_conference.sty
PAPER_BODY_PT = 10.0       # \normalsize of that 10 pt article -- the floor for any label
GRID_FIG_W_IN = 10.0       # canvas width of the curve grids (sets 1 and 3)
# Sets 1 and 3 are drawn as a SINGLE ROW of four noise panels (2026-09-21; they used to
# be a 2x2 grid).  The canvas width is unchanged -- widening it would only be scaled back
# down at width=\textwidth, and _FS/GRID_LEGEND_KW are both tied to GRID_FIG_W_IN -- so
# only the height changes: one row of panels plus the measured bottom legend strip.
GRID_ROW_FIG_H_IN = 5.6    # canvas height of those single-row grids
# One knob over every figure's type size, applied inside font_scale().  1.0 makes a size
# written as N print at exactly N points; below 1.0 trims the whole family down together.
# Kept just under 1 so the figures sit a touch under the body text rather than matching
# it -- raise or lower this one number rather than editing sizes figure by figure.
FONT_TRIM = 0.80


# --- shared legend styling for the 2x2 curve grids (sets 1 and 3) -------------------
# The entry text is PAPER_BODY_PT as printed (see font_scale below): the legend names the
# curves and has to be as readable as the paper's own body text.  The rest gives the
# handles/rows room to breathe; pass them to fig.legend / ax.legend.  These are for the
# 10 in grids -- a figure of another width would need its own font_scale factor.
GRID_LEGEND_KW = dict(fontsize=10 * FONT_TRIM * (GRID_FIG_W_IN / PAPER_TEXTWIDTH_IN),
                      markerscale=1.3, handlelength=2.8,
                      handletextpad=0.8, labelspacing=0.8, columnspacing=1.6,
                      borderpad=0.8, borderaxespad=0.6,
                      frameon=True, framealpha=0.9)

# Starting column count for the grids' bottom legend strip.  bottom_legend() drops a
# column at a time until the box fits the canvas width, so this is an upper bound, not a
# promise: it only decides how flat the strip is allowed to get when the labels are short.
GRID_LEGEND_NCOL = 4


# --- legend display names -----------------------------------------------------------
# Legend TEXT only, applied at the moment the legend is built.  The ALGORITHM IDENTITY is
# untouched: the caches, the --only/--exclude lists in make_requested_plots.sh and every
# colour/marker lookup key on the internal label, so renaming here cannot silently drop a
# method from a figure.  Same contract as e2e_tpsr_addon.LEGEND_RENAME, which maps the
# standalone TPSR to "e2e+TPSR" first -- this table then takes that to "TPSR (E2E)", so
# apply the two in that order.  Lookups are EXACT, never substring: "OneShot-Llama70B"
# must not be caught by the "OneShot-Llama" rule.
LEGEND_DISPLAY = {
    "e2e":                "E2E",
    "e2e+TPSR":           "TPSR (E2E)",
    # The dot panels (sets 2/4/5) print the raw algorithm name, which never passes
    # through e2e_tpsr_addon.rename_legend, so the standalone arm reaches this table as
    # "TPSR"; the curve legends reach it as "e2e+TPSR".  Both print the same text.
    "TPSR":               "TPSR (E2E)",
    "145M+TPSR":          "TPSR (145M)",
    # D&C follows TPSR's "<search> (<model>)" shape rather than devncon_addon's
    # "<model>+D&C", so the two combined-method families read the same way in one
    # legend.  The "145M" here becomes a bold "ours" in display_labels below, giving
    # "D&C (ours)".  The 89M / len-80 siblings are mapped too so the family cannot
    # print in two different shapes if either is ever re-enabled (both currently sit
    # in make_requested_plots.sh's DROP_ARMS).
    "145M+D&C":           "D&C (145M)",
    "145M-len80-float_ftnoise_e27+TPSR":  "TPSR (145M fine tuned)",
    "145M-len80-float_ftnoise_e27+D&C":   "D&C (145M fine tuned)",
    # No "fine tuned" qualifier (2026-09-23): all four arms drawn in sets 1-3 -- beam,
    # its --unscale decode, D&C and TPSR -- are the SAME 48 h checkpoint, and the beam
    # arm now prints as plain "ours", so the words distinguished nothing and appeared on
    # only two of the four.  "145M" spelling keeps the OURS_BOLD replace working.
    "145M-len80-float_ftnoise_48h+TPSR":  "TPSR (145M)",
    "145M-len80-float_ftnoise_48h+D&C":   "D&C (145M)",
    "89M+D&C":            "D&C (89M)",
    "145M-len80+D&C":     "D&C (145M-len80)",
    "LLMSR-Llama":        "LLM-SR (Llama-8B)",
    "LLMSR-Gemini":       "LLM-SR (Gemini)",
    # The self-hosted Qwen2.5-Coder-32B LLM-SR run: same family, same shape of name.
    # Without this entry the curve prints its INTERNAL label beside the prettified
    # Llama one, so one legend shows two naming conventions.
    "LLMSR-Qwen":         "LLM-SR (Qwen-32B)",
    # "one-shot <backbone>", not "OneShot (<backbone>)": the parenthesised form reads as
    # a model variant, while these arms are a METHOD (one LLM call per problem) applied
    # to a backbone.  The Llama arms are renamed to match even though only the Gemini one
    # is drawn (the other two sit in make_requested_plots.sh's CURVE_DROP), so the family
    # cannot print in two shapes if either is re-enabled.
    "OneShot-Gemini":     "one-shot Gemini",
    "OneShot-Llama":      "one-shot Llama-8B",
    "OneShot-Llama70B":   "one-shot Llama-70B",
    # The len-80 float arm is now THE "ours" curve (the plain 145M arm was dropped
    # from sets 1-5 on 2026-09-17), and its noise-augmented finetune prints beside it.
    # Values deliberately still spell "145M" so the OURS_BOLD replace below bolds them.
    "145M-len80-float":               "145M",
    "145M-len80-float_ftnoise_e27":   "145M (fine tuned, 7.6h)",
    "145M-len80-float_ftnoise_e80":   "145M (fine tuned, 23h)",
    # The 48 h checkpoint is now THE "ours" curve (2026-09-23): the un-finetuned
    # 145M-len80-float arm was dropped from the figures, so there is no second arm to
    # distinguish it from and "fine tuned" in the name only added noise.  Value spells
    # "145M" so the OURS_BOLD replace below bolds it, exactly as the base arm used to.
    # NOTE this is global -- set 3 and the LSR-Synth figure print it the same way, which
    # is why the base arm had to leave EVERY figure rather than just sets 1-2.
    "145M-len80-float_ftnoise_48h":   "145M",
    "145M-len80-float_ftnoise_48h_unscaled": "145M (unscaled)",
    "PhyE2E (E2E)":       "PhyE2E",
    "PySR-10":            "PySR (10s)",
    "PySR-60":            "PySR (1m)",
    "PySR-1800":          "PySR (30m)",
    "PhyE2E (E2E)-units": "PhyE2E-units",
}


# Our own model prints as a BOLD "ours" rather than "145M", in every figure and in every
# compound label ("ours+D&C", "TPSR (ours)", "ours-len80-float").  Applied here, after
# the LEGEND_DISPLAY lookup, so the substitution reaches the compound names that table
# builds as well as the bare one -- and so sets 1-5 and the ablation figure, which all
# funnel through this one function, cannot disagree.
#
# Bold via mathtext ($\bf{...}$), not a per-Text fontweight call: the label is a plain
# string by the time each script hands it to ax.legend / set_yticklabels, so markup is
# the only channel that survives.  Only the word itself is wrapped -- "&" and "+" in the
# compound names stay OUTSIDE math mode, where mathtext leaves them alone (there is no
# usetex here, so they need no escaping).
OURS_RAW   = "145M"
OURS_BOLD  = r"$\bf{ours}$"


def display_labels(labels):
    """Map a sequence of legend labels through LEGEND_DISPLAY (unknown labels pass through),
    then rename our own model to a bold "ours".

    Call this BEFORE sort_legend so the alphabetical order matches the printed text.
    """
    return [LEGEND_DISPLAY.get(l, l).replace(OURS_RAW, OURS_BOLD) for l in labels]


# --- legend ordering ----------------------------------------------------------------
# Explicit running order for the figures' legends, requested 2026-09-22.  It reads
# ours-first (base, then the fine-tuned arm and its decodes), then the transformer
# baselines, then the search/LLM methods -- i.e. by KINSHIP, which alphabetical order
# scatters.  Entries are the PRINTED text after display_labels(), with the mathtext
# markup around "ours" stripped, exactly as legend_sort_key sees them.
#
# A label absent from this list is NOT dropped: it sorts alphabetically after every
# listed one, so a newly added arm still appears (at the end) instead of vanishing.
# KEEP IN SYNC WITH LEGEND_DISPLAY ABOVE.  These are matched by exact printed text, so a
# renamed arm silently stops matching and falls back to alphabetical -- which is how the
# 2026-09-23 rename ("ours (fine tuned)" -> "ours", "D&C (ours fine tuned)" -> "D&C
# (ours)") scattered our own methods through the legend.
LEGEND_ORDER = [
    # ours first, and together: the same 48 h checkpoint under four decodes.
    "ours",
    "ours (unscaled)",
    "D&C (ours)",
    "TPSR (ours)",
    # then the other transformers, then the search / LLM methods.
    "E2E",
    "TPSR (E2E)",
    "PhyE2E",
    # PySR by ASCENDING compute budget, not alphabetically: the three are one method at
    # three costs, and on LSR-Synth the ladder (10s -> 1m -> 30m) is the comparison the
    # figure exists to make.
    "PySR (10s)",
    "PySR (1m)",
    "PySR (30m)",
    "AIFeynman",
    "one-shot Gemini",
]

# Legend entries are sorted by the text as PRINTED -- after any display rename (the
# standalone TPSR prints as "e2e+TPSR", so it sorts under "e", not "T").  Plot order is
# solve-rate order, which shuffles the legend from figure to figure and made a given
# method hard to find; alphabetical is stable across every panel and noise level.
def legend_sort_key(label):
    """Case-insensitive natural sort: digit runs compare numerically, so "89M" comes
    before "145M" and the numeric arms come before the alphabetic ones."""
    # Strip mathtext markup first: display_labels emits "$\\bf{ours}$", and sorting the
    # raw string would file it under "$" ahead of every real name.
    vis = re.sub(r"\$|\\bf\{|\}", "", str(label))
    # Explicit order wins; everything else keeps the old natural sort, after the listed
    # entries.  The rank is the first tuple element so the two regimes never interleave.
    if vis in LEGEND_ORDER:
        return [(0, LEGEND_ORDER.index(vis), "")]
    parts = re.split(r"(\d+)", vis)
    return [(1, 0, "")] + [(0, int(p), "") if p.isdigit() else (1, 0, p.casefold())
                           for p in parts]


def sort_legend(handles, labels):
    """(handles, labels) reordered by the printed label.  Both are returned as lists,
    so the result can be passed straight to fig.legend / ax.legend."""
    pairs = sorted(zip(list(labels), list(handles)), key=lambda p: legend_sort_key(p[0]))
    return [h for _l, h in pairs], [l for l, _h in pairs]


def bottom_legend(fig, handles, labels, *, ncol, pad_in=0.06, **kw):
    """Draw a shared legend under the whole figure and return (legend, bottom_frac):
    the fraction of the figure height tight_layout must reserve for it.

    The fraction is MEASURED from the drawn box rather than hard-coded, so growing the
    legend font or adding an arm cannot make the box creep over the x-axis labels the
    way a fixed rect= constant did.

    `ncol` is a STARTING point, not a promise: the box is redrawn with one column fewer
    until it fits the canvas width.  A legend wider than the figure hangs off the edge,
    and save_fig's bbox_inches="tight" then grows the saved page around the overhang --
    which is exactly what LaTeX turns back into small text, since a wider page is scaled
    down harder at width=\textwidth.  Wrapping to another row keeps the page at figsize.
    """
    def _draw(n):
        leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                         ncol=n, **kw)
        fig.canvas.draw()                # need a renderer before the box has a size
        return leg, leg.get_window_extent()

    leg, box = _draw(ncol)
    while ncol > 1 and box.width > fig.get_figwidth() * fig.dpi:
        leg.remove()
        ncol -= 1
        leg, box = _draw(ncol)

    frac = box.height / float(fig.get_figheight() * fig.dpi)
    return leg, min(0.45, frac + pad_in / fig.get_figheight())


def above_legend(frac, fig, gap_pt=6.0):
    """y (figure fraction) for a supxlabel that must clear a bottom_legend box whose
    reserved fraction is `frac`.  Sitting the label AT `frac` puts its baseline on the
    box's top edge, which the taller paper-sized fonts then overlap."""
    return frac + gap_pt / 72.0 / fig.get_figheight()


def right_legend(fig, handles, labels, *, ncol=1, pad_in=0.10, **kw):
    """Draw the shared legend as its OWN COLUMN down the right-hand side of the figure
    and return (legend, right_frac): the fraction of the figure width tight_layout must
    leave for the axes, i.e. pass rect=(0, 0, right_frac, 1).

    Why a right column at all: the old bottom strip was laid out as two very wide rows
    (ncol = ceil(n/2)), and save_fig's bbox_inches="tight" grows the saved bbox around
    anything that overhangs the canvas -- so a 10 in figure was written out ~15 in wide.
    Included at width=\\textwidth that got scaled to ~0.47x and every label printed at
    half its nominal size.  A single-column legend on the right is narrow enough to sit
    INSIDE the canvas, so the emitted PDF stays at its figsize and LaTeX barely scales it.

    The column is placed in two passes: draw it hanging off the right edge, measure the
    box that matplotlib actually produced, then move it back inside by its own width.
    Measuring (rather than hard-coding a rect constant) means adding an arm or growing
    the legend font can never push the box off the page or over the axes.
    """
    leg = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                     ncol=ncol, **kw)
    fig.canvas.draw()                    # need a renderer before the box has a size
    w = leg.get_window_extent().width / float(fig.get_figwidth() * fig.dpi)
    frac = max(0.45, 1.0 - w - pad_in / fig.get_figwidth())
    leg.set_bbox_to_anchor((frac, 0.5), transform=fig.transFigure)
    return leg, frac


# --- printed font size ---------------------------------------------------------------
# Every figure goes into the paper at width=\textwidth, so LaTeX scales it by
# (textwidth / its own width in inches).  A 10 in wide figure is therefore shrunk to
# 0.55x, and a nominally-11 pt tick label reaches the page at about 6 pt -- half the size
# of the 10 pt body text around it.
#
# The reference width is consequently the PAGE's text width, not any figure's own width:
# font_scale(w) is the factor a w-inch-wide figure's sizes need so that a size written as
# N in the code PRINTS at N points on the page.  Write every size as the point size it
# should have IN THE PAPER and multiply by font_scale of that figure's width -- then a
# figure keeps its printed typography no matter how wide its canvas is, and sizes are
# directly comparable with the document's own 10 pt.
def font_scale(fig_w_in, ref_w_in=PAPER_TEXTWIDTH_IN):
    """Factor a `fig_w_in`-wide figure's font sizes must carry so that a size written as
    N points in the code prints at N * FONT_TRIM points on a `ref_w_in`-wide page."""
    return FONT_TRIM * float(fig_w_in) / float(ref_w_in)


def fit_rect(fig, rect, *, max_iter=5, pad_px=4.0):
    """fig.tight_layout() into `rect`, pulling the left/right edges in until no axes
    title, axis label or in-panel legend hangs off the canvas.  Returns the rect used.

    tight_layout is supposed to make room for exactly these artists, but it under-measures
    a subplot's x-axis label when that subplot carries no y tick labels (as the right-hand
    dot panel of the best-seed figure does, show_ylabels=False) -- and fig.get_tightbbox()
    misses it too, so save_fig's bbox_inches="tight" does not grow the page around it
    either and the label is simply CUT at the page edge.  It only shows up once the fonts
    are large enough for the label to outgrow its panel, which is what font_scale does.

    Measuring the overflow and laying out again inside a narrower rect fixes it without
    widening the figure -- widening would just be scaled back down at width=\\textwidth.
    Iterated, because shrinking the panels also recentres the labels that overflowed.
    `pad_px` is the clear margin left at each edge, so the last glyph does not sit flush
    against it (the tight bbox is not always exact to the pixel).
    """
    x0, y0, x1, y1 = rect
    for _ in range(max_iter):
        fig.tight_layout(rect=(x0, y0, x1, y1))
        fig.canvas.draw()
        w_px = fig.get_figwidth() * fig.dpi
        over_r = over_l = 0.0
        for ax in fig.axes:
            for art in _decorations(ax):
                ext = art.get_window_extent()
                over_r = max(over_r, ext.x1 - (w_px - pad_px))
                over_l = max(over_l, pad_px - ext.x0)
        if over_r <= 1.0 and over_l <= 1.0:
            break
        x1 -= over_r / w_px
        x0 += over_l / w_px
    return (x0, y0, x1, y1)
