"""plot_markers.py -- one stable, colour-blind-friendly marker per method.

The OOD-accuracy-vs-gap curve figures (sets 1/3/5) used to draw every method with the
same 'o' marker, so *colour was the only channel* separating the lines -- unreadable for
colour-blind viewers and in greyscale.  This module maps each method's display label to a
distinct matplotlib marker, shared across all three curve scripts so a given method looks
the same in every figure.  Every marker is drawn HOLLOW (see HOLLOW below) -- the shapes
are high-contrast outlines that stay legible at small sizes.

Unknown labels (e.g. extra SRBench baselines that only appear in the camera-ready /
common-subset figures) get a deterministic marker from a fallback pool, so no figure
crashes on a new method and the choice is stable across runs.

The dot-plot figures (sets 2/4/6) are intentionally NOT changed: there the marker encodes
the *noise level* and the method is the row label, so method identity never relies on
colour to begin with.
"""
import zlib

# Every marker in the curve figures is drawn HOLLOW: the face is "none", so only the
# outline is stroked and whatever sits behind a point (another curve, an error bar)
# stays visible.  A hollow shape needs a heavier edge than the 0.5 pt white halo the
# filled markers used to carry, or it washes out at markersize 6.  The edge colour is
# deliberately left unset: matplotlib's "auto" default takes it from the line, so each
# method's outline keeps its own colour.
HOLLOW = {"markerfacecolor": "none", "markeredgewidth": 1.5}

# One marker size for every CURVE figure (sets 1/3/5 and the overlays that draw onto
# them), so the curves of a figure cannot end up wearing different sizes depending on
# which module drew them.  The dot panels of sets 2/4/5 size their own (_MS there).
CURVE_MS = 7
# Line width to match, for the same reason: the overlays (LLMSR, one-shot) used to draw
# at lw=2 against the 1.2 of every curve beside them, which read as emphasis rather than
# as just another method.  Thinned 1.2 -> 0.9 (2026-09-22): with a dozen arms per panel
# the thicker strokes merged where curves cross.
CURVE_LW = 0.9

# Distinct marker shape per known method.  Families are given clearly different shapes
# so, e.g., 89M (^) and its 89M+TPSR variant (X) never rely on the orange/dark-orange
# colour difference alone.
_METHOD_MARKER = {
    "145M":          "s",   # square
    # Named rather than left to the fallback pool, which picks by whatever shape is
    # untaken in a given figure and so would vary across sets 1/3/5 (see PySR below).
    # A rotated-polygon spec, not a letter shape: all fifteen of those are claimed
    # below, and the thin "tri" family is reserved for the one-shot arms.
    "145M-len80-simplipy": (5, 0, 180),  # pentagon, point-down
    "145M+TPSR":   "*",   # star
    "145M_unscaled": "d",   # thin diamond
    "89M":           "^",   # triangle up
    "89M+TPSR":    "X",   # x (filled)
    "89M_unscaled":  ">",   # triangle right
    "89M-float":     "v",   # triangle down
    # THE "OURS" FAMILY -- five arms drawn side by side in sets 1-3, so these have to be
    # the most separable shapes in the figure.  They used to be N-pointed stars (6, 9, 5
    # and 8 points) plus a heptagon: at markersize 7, hollow, a 6- and a 9-pointed star
    # are the same smudge, and the heptagon is just a circle.  Reassigned 2026-09-22 to
    # square / triangle-up / triangle-down / x / star -- four different outline FAMILIES
    # rather than four counts of the same one.
    #
    # Each shape's previous owner (145M, 89M, 89M-float, 89M+TPSR, 145M+TPSR) is dropped
    # from every figure these arms appear in, so nothing collides.  The one exception is
    # "*": make_llm_appendix_plots.sh still whitelists "145M+TPSR", so if that script is
    # ever pointed at the 48h arms too, give one of them another shape.
    "145M-len80-float": "s",                              # square      -- base arm
    "145M-len80-float_ftnoise_48h": "^",                  # triangle up -- fine tuned
    "145M-len80-float_ftnoise_48h_unscaled": "v",         # triangle down -- its --unscale arm
    "145M-len80-float_ftnoise_48h+D&C": "X",              # x
    "145M-len80-float_ftnoise_48h+TPSR": "*",             # star
    # The retired finetune checkpoints keep their old star specs: they are in DROP_ARMS,
    # so they are never drawn beside the arms above.
    "145M-len80-float_ftnoise_e27": (4, 1, 0),
    "145M-len80-float_ftnoise_e80": (7, 1, 0),
    "145M-len80-float_ftnoise_e27+TPSR": (8, 1, 0),   # eight-pointed star
    "145M-len80-float_ftnoise_e27+D&C": (7, 0, 0),   # heptagon
    "e2e":           "D",   # diamond
    "TPSR":          "P",   # plus (filled)
    "AIFeynman":     "p",   # pentagon
    "PhyE2E (E2E)":  "H",   # hexagon (vertical) -- the other published model
    # PySR: octagon.  Named explicitly rather than left to the fallback pool, which
    # assigns by whatever is untaken in a given figure and so would give this method a
    # different shape depending on which curves it is drawn beside.  "8" is the one
    # _FALLBACK shape no named method above claims, so adding it here costs nothing.
    "PySR":          "8",   # octagon
    "PySR-1800":     "8",   # octagon -- the 30m arm; plain "PySR" (1 h) is never drawn
    "Operon":        "h",   # hexagon
    "LLMSR-Gemini":  "o",   # circle
    "LLMSR-Llama":   "<",   # triangle left
    # The one-shot (single LLM call) arms share the thin three-spoke "tri" family, so
    # they read as a family -- as their dashed curves do (oneshot_overlay.LINESTYLE) --
    # and Llama's points left like LLMSR-Llama's "<".  Neither shape appears in
    # _FALLBACK below, so adding them leaves every fallback assignment untouched.
    "OneShot-Gemini": "1",  # tri_down
    "OneShot-Llama":  "3",  # tri_left
    # Named explicitly rather than left to method_marker's variant fallback: the label
    # starts with "OneShot-Llama", so the fallback would hand it an unrelated shape from
    # outside the tri family.
    "OneShot-Llama70B": "2",  # tri_up
}

# Deterministic fallback for any label not in the table above (kept disjoint-ish from the
# primaries so a stray baseline rarely clashes with a known method in the same figure).
_FALLBACK = ["8", "H", ">", "d", "*", "P", "X", "p", "h", "<", "s", "^", "v", "D", "o"]

# Every shape in _FALLBACK is also spoken for in _METHOD_MARKER above, so a variant
# label (a re-run tree's "89M+TPSR-revised", say) would otherwise be handed a
# marker a named method is already using IN THE SAME
# FIGURE.  These rotated-polygon specs, matplotlib's (numsides, style, angle) form,
# are high-contrast and distinct from all fifteen above.
_FALLBACK_EXTRA = [(5, 0, 180), (6, 0, 30), (7, 0, 0), (4, 0, 45),
                   (5, 1, 0), (6, 1, 30), (8, 0, 22.5), (3, 0, 180),
                   (7, 0, 25), (4, 1, 0), (5, 0, 36), (6, 0, 0)]


def method_marker(label):
    """The stable marker for a method's display label (see module docstring).

    Unknown labels draw from _FALLBACK, but only from the shapes NOT already
    assigned above -- otherwise a variant label (e.g. a re-run tree's
    "89M+TPSR-revised") could land on the same marker as a
    named method that appears in the very same figure.  Still deterministic: the
    choice is a crc32 of the label over the filtered pool.
    """
    mk = _METHOD_MARKER.get(label)
    if mk is not None:
        return mk
    label = str(label)
    taken = set(_METHOD_MARKER.values())
    pool = [m for m in _FALLBACK if m not in taken] + _FALLBACK_EXTRA

    # A variant label is a known method plus a suffix ("89M+TPSR" + "-revised").
    # Offsetting by the BASE method's rank guarantees two variants of DIFFERENT
    # models never collide with each other -- the case that actually arises, since
    # a comparison figure carries the same suffix for both models.
    keys = sorted(_METHOD_MARKER)
    bases = [k for k in keys if label.startswith(k) and label != k]
    if bases:
        base = max(bases, key=len)
        off = keys.index(base) * 5
        idx = (off + zlib.crc32(label[len(base):].encode())) % len(pool)
    else:
        idx = zlib.crc32(label.encode()) % len(pool)
    return pool[idx]
