"""phye2e_addon.py -- shared helpers for the PhyE2E baseline on both benchmarks.

PhyE2E is Ying et al., *A Neural Symbolic Model for Space Physics*, Nature Machine
Intelligence 2025 (vendored in ./PhysicsRegression/, driven by eval_phye2e.py).

WHAT THE CURVE IS -- "PhyE2E (E2E)", not "PhyE2E"
-------------------------------------------------
We run the authors' own **e2e** ablation: their end-to-end transformer with
Divide-and-Conquer, MCTS and GP all OFF, decoded with beam search.  That is the
like-for-like arm next to e2e / NeSymReS / our models, and it is what their own
eval_result CSV calls `e2e` (its `type` column is {e2e, oracle, oraclegp, oraclemcts}).
Their PAPER's Feynman numbers are the full `oraclemcts` pipeline, so this curve sits
below their published figures BY CONSTRUCTION.  The label says so, in the figure, so
nobody has to read the caption to avoid that mistake.

Unlike AIFeynman, PhyE2E is on BOTH benchmark families here: SRBench has no published
PhyE2E numbers either (their repo ships results for a 5-formula synthetic sample only),
so both halves come from our own run.

FORMULA FORMAT
--------------
eval_phye2e.py emits infix in x_0..x_N -- the e2e convention, NOT our prefix PN.  So the
SRBench OOD path scores it through compare_srbench._ood_eval_infix with style "x_0",
exactly as the e2e baseline is scored, and the LLM-SRBench path's _is_infix() heuristic
routes it the same way.  Nothing new had to be taught to either evaluator.

This is the TPSR/AIF2/AIFeynman addon pattern: the curve is opt-in behind
--include-phye2e (default off), so no existing figure changes unless asked.

RESULTS TREE
------------
    results/phye2e/[seed<N>/]noise_<tau>/results_feynman_phye2e.pkl.gz        (SRBench)
    results/phye2e/[seed<N>/]noise_<tau>/llmsrbench/phye2e_<split>/results.pkl.gz

"phye2e" is deliberately NOT added to results_io.GROUPS: that tuple drives the DEFAULT
discovery for every figure, and appending to it would silently add this curve to all of
them.  Discovery here passes groups=(GROUP,) explicitly, exactly as aifeynman_addon does.
"""
import os

import results_io

DEFAULT_GROUP = "phye2e"

# Display identity.  Fuchsia, deliberately NOT a violet: e2e is #7c3aed and the first
# choice here (#8b5cf6, violet-500) was close enough that the two labels read as the
# same colour in a rendered figure -- which is exactly the confusion the "(E2E)" in the
# label exists to prevent.  Everything else is spoken for: blue = SRBench baselines,
# orange = 89M, green = 145M, violet = e2e, teal = TPSR, cyan = D&C, brown = Operon.
# SOLID, like e2e -- on these figures dashed means "SRBench published baseline", and
# PhyE2E is a model we RUN ourselves on both benchmarks, exactly as e2e is.  (AIFeynman
# is dashed because it genuinely is a published SRBench baseline there, and keeps that
# identity on the LLM-SRBench figure for cross-family consistency.)  Solid in both
# families keeps this curve's meaning the same wherever it appears.
LABEL = "PhyE2E (E2E)"
COLOR = "#d946ef"          # fuchsia-500
LINESTYLE = "-"

SRBENCH_ARTIFACT = "results_feynman_phye2e.pkl.gz"

# --- module state, rebound by set_group() ---------------------------------------
PHYE2E_GROUP = DEFAULT_GROUP
LABEL_SUFFIX = ""
DISPLAY_LABEL = LABEL


def default_label_suffix(group: str) -> str:
    """The suffix a given results tree contributes to its display label.

    The default tree keeps the bare label; any other tree (a re-run, a different
    decode mode) gets "-<variant>" so its curves can never be confused with the
    original run's in an overlay.
    """
    if group == DEFAULT_GROUP:
        return ""
    tail = group[len(DEFAULT_GROUP):].lstrip("_-") if group.startswith(DEFAULT_GROUP) else group
    return f"-{tail}" if tail else ""


def set_group(group: str, label_suffix: "str | None" = None) -> None:
    """Point this addon at a results tree (and adjust the display label)."""
    global PHYE2E_GROUP, LABEL_SUFFIX, DISPLAY_LABEL
    PHYE2E_GROUP = group
    LABEL_SUFFIX = default_label_suffix(group) if label_suffix is None else label_suffix
    DISPLAY_LABEL = LABEL + LABEL_SUFFIX


def label_for_group(group: str, label_suffix: "str | None" = None) -> str:
    """The display label a given tree draws under, independent of module state.

    set_group() rebinds a global, which is fine when a figure shows ONE arm.  Overlaying
    several arms in one figure needs a per-group label instead, so this computes it
    without touching (or depending on) the active group.
    """
    suf = default_label_suffix(group) if label_suffix is None else label_suffix
    return LABEL + suf


def add_group_arg(parser) -> None:
    parser.add_argument("--phye2e-group", default=DEFAULT_GROUP, metavar="DIR",
                        help=f"Which results tree the PhyE2E curve reads "
                             f"(default: {DEFAULT_GROUP}). A non-default tree gets a "
                             f"'-<variant>' label suffix so an overlay stays readable.")
    parser.add_argument("--phye2e-label-suffix", default=None, metavar="S",
                        help="Override that suffix (pass '' to reuse the plain label).")
    parser.add_argument("--phye2e-groups", nargs="*", default=None, metavar="DIR",
                        help="Draw SEVERAL PhyE2E arms at once, one curve per results "
                             "tree (e.g. phye2e phye2e_units); each gets its own "
                             "'-<variant>' label, colour and marker. Default: just "
                             "--phye2e-group. Needed wherever a figure identifies "
                             "methods by ARTIFACT discovery rather than by cache row, "
                             "which is every figure except the pure --reuse-ood curves.")


def apply_group_arg(args) -> None:
    set_group(getattr(args, "phye2e_group", DEFAULT_GROUP),
              getattr(args, "phye2e_label_suffix", None))


def active_groups(args) -> tuple:
    """The PhyE2E trees a figure should draw, honouring --phye2e-groups then
    --phye2e-group.  Always a tuple, so a caller can pass it straight to
    llmsr_methods_for_groups / srbench_pkls_for_groups."""
    gs = getattr(args, "phye2e_groups", None)
    return tuple(gs) if gs else (getattr(args, "phye2e_group", DEFAULT_GROUP),)


def srbench_pkls_for_groups(root, tau, groups):
    """{display_label: artifact} for the SRBench half across several trees."""
    out = {}
    for g in groups:
        p = srbench_phye2e_pkl(root, tau, group=g)
        if p:
            out[label_for_group(g)] = p
    return out


def is_phye2e(label: str) -> bool:
    """Exact/prefix match, never a substring test -- so it cannot cross-fire with
    the plain "e2e" baseline, whose label is a substring of this one."""
    return str(label) == LABEL or str(label).startswith(LABEL + "-")


# A variant tree ("-units", a re-run, a different decode) is drawn in a DARKER shade of
# the same hue: same family says "this is PhyE2E", the shade says "different arm".  Both
# arms are legitimately ours-run, so both stay SOLID (dashed means "SRBench PUBLISHED
# baseline" on these figures).  Colour is never the only channel -- plot_markers hands a
# suffixed label its own shape from _FALLBACK_EXTRA.
VARIANT_COLOR = "#701a75"  # fuchsia-900 -- a LIGHTNESS gap, so the two arms stay
                           # separable in greyscale, not just by hue


def color(label: str = None):
    """Fuchsia for the default tree, a darker fuchsia for any suffixed variant.

    Called with no label (or the plain label) this is the original constant, so every
    single-arm figure renders exactly as before.
    """
    if label is not None and str(label).startswith(LABEL + "-"):
        return VARIANT_COLOR
    return COLOR


def linestyle(label: str = None):
    return LINESTYLE


def srbench_phye2e_pkl(root, tau, group=None) -> "str | None":
    """The merged SRBench PhyE2E artifact at noise tau, or None if absent."""
    d = results_io.group_noise_dir(root, group or PHYE2E_GROUP, tau)
    p = os.path.join(d, SRBENCH_ARTIFACT)
    return p if os.path.isfile(p) else None


def _method_artifact(d):
    """The per-method result file inside dir `d` (results.pkl.gz or legacy .jsonl)."""
    for name in ("results.pkl.gz", "results.jsonl"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def llmsr_phye2e_methods(root, tau, split, group=None):
    """{display_label: results_artifact} for the PhyE2E LLM-SRBench run at noise tau.

    Mirrors discover_methods_from_dirs' output so callers merge the two dicts. Empty
    when the tree does not exist, so a figure asked to --include-phye2e before the run
    has landed simply draws without it.  Shard dirs are skipped: they are partial by
    construction, and plotting one as the whole benchmark understates the solve rate.
    """
    out = {}
    dirs = results_io.list_llmsr_method_dirs(root, tau, split,
                                             groups=(group or PHYE2E_GROUP,))
    for d in dirs:
        base = os.path.basename(d)
        if "__shard" in base:
            continue
        art = _method_artifact(d)
        if art is None:
            continue
        # Label by the tree actually queried.  Falling back to DISPLAY_LABEL only when
        # no explicit group was passed keeps --phye2e-label-suffix working for the
        # single-arm case, while a multi-arm caller gets one label per tree instead of
        # every tree collapsing onto the active group's label.
        out[label_for_group(group) if group else DISPLAY_LABEL] = art
    return out


def llmsr_methods_for_groups(root, tau, split, groups):
    """{display_label: artifact} across SEVERAL PhyE2E trees at once.

    Each tree contributes its own label (see label_for_group), so an overlay of the
    plain and --units arms keeps them apart.  A missing tree contributes nothing.
    """
    out = {}
    for g in groups:
        out.update(llmsr_phye2e_methods(root, tau, split, group=g))
    return out
