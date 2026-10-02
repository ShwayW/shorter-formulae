"""pysr_addon.py -- shared helpers for the PySR baseline on both benchmarks.

PySR is Cranmer, *Interpretable Machine Learning for Science with PySR and
SymbolicRegression.jl* (2023), driven here by eval_pysr.py.

WHY THIS CURVE IS OURS TO RUN
-----------------------------
Unlike AIFeynman / Operon / FEAT, PySR has NO published SRBench numbers to reuse.
srbench/results/ground-truth_results.feather holds fourteen algorithms --

    AFP, AFP_FE, AIFeynman, BSR, DSR, EPLEX, FEAT, FFX, GP-GOMEA, ITEA, MRGP,
    Operon, SBP-GP, gplearn

-- and PySR is not among them.  It landed in the SRBench *repository* only after the
NeurIPS 2021 paper, in the post-paper algorithm collection (srbench/algorithms/pysr,
next to Brush / QLattice / TIR / uDSR, none of which are in the feather either).  So
SRBench ships an official PySR *configuration* but no official PySR *results*.

That has a consequence for how this curve is DRAWN: it is SOLID, not dashed.  On these
figures dashed means "SRBench published baseline", and both halves of this curve come
from our own run -- exactly like e2e and PhyE2E, and unlike AIFeynman, which is dashed
because it genuinely is published.  See eval_pysr.py's docstring for the budget we
chose and why it is not SRBench's effective 10 h/fit.

FORMULA FORMAT
--------------
eval_pysr.py emits infix in x_0..x_N -- the e2e convention, NOT our prefix PN.  So the
SRBench OOD path scores it through compare_srbench._ood_eval_infix with style "x_0",
exactly as e2e and PhyE2E are scored, and the LLM-SRBench path's _is_infix() heuristic
routes it the same way.  Nothing new had to be taught to either evaluator, and the
SRBench row schema (dataset, seed, predicted_formula) is byte-for-byte the one
_ood_load_e2e_all already reads.

A NOTE ON `complexity`
----------------------
eval_pysr.py stores a `complexity` field taken from PySR's OWN Pareto front (its node
count under its own operator weighting).  That is NOT the repo's metric and must never
be plotted as if it were.  It is safe only because every consumer recomputes complexity
from the formula string: compare_srbench._load_pkl_gz overwrites the column with
formula_complexity(), and plot_llmsr_time_complexity_vs_noise._method_complexity reads
`discovered_equation` and ignores the stored value.  The field is kept as provenance
(it is what PySR's model selection actually used); do not start reading it.

This is the TPSR/AIF2/AIFeynman/PhyE2E addon pattern: the curve is opt-in behind
--include-pysr (default off), so no existing figure changes unless asked.

RESULTS TREE
------------
    results/pysr/[seed<N>/]noise_<tau>/results_feynman_pysr.pkl.gz          (SRBench)
    results/pysr/[seed<N>/]noise_<tau>/llmsrbench/pysr_<split>/results.pkl.gz

"pysr" is deliberately NOT added to results_io.GROUPS: that tuple drives the DEFAULT
discovery for every figure, and appending to it would silently add this curve to all of
them.  Discovery here passes groups=(GROUP,) explicitly, exactly as phye2e_addon does.
"""
import os

import results_io

DEFAULT_GROUP = "pysr"

# Display identity.  AMBER -- the last clearly free slot in this palette.  Everything
# else is spoken for: steel blue = SRBench published baselines + AIFeynman, orange =
# 89M, green = 145M, violet = e2e, teal = TPSR, cyan = D&C, red = 145M+D&C, brown =
# Operon, fuchsia = PhyE2E, rose = 89M-float.  Amber is far enough from the 89M orange
# (#fb923c) in both hue and lightness to stay separable, including in greyscale.
# SOLID: see the module docstring -- dashed is reserved for published baselines, and
# both halves of this curve are our own run.
LABEL = "PySR"
COLOR = "#ca8a04"          # amber-600
LINESTYLE = "-"

SRBENCH_ARTIFACT = "results_feynman_pysr.pkl.gz"

# --- module state, rebound by set_group() ---------------------------------------
PYSR_GROUP = DEFAULT_GROUP
LABEL_SUFFIX = ""
DISPLAY_LABEL = LABEL


def default_label_suffix(group: str) -> str:
    """The suffix a given results tree contributes to its display label.

    The default tree keeps the bare label; any other tree (a re-run at a different
    budget, a no-early-stop arm) gets "-<variant>" so its curves can never be confused
    with the original run's in an overlay.
    """
    if group == DEFAULT_GROUP:
        return ""
    tail = group[len(DEFAULT_GROUP):].lstrip("_-") if group.startswith(DEFAULT_GROUP) else group
    return f"-{tail}" if tail else ""


def set_group(group: str, label_suffix: "str | None" = None) -> None:
    """Point this addon at a results tree (and adjust the display label)."""
    global PYSR_GROUP, LABEL_SUFFIX, DISPLAY_LABEL
    PYSR_GROUP = group
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
    parser.add_argument("--pysr-group", default=DEFAULT_GROUP, metavar="DIR",
                        help=f"Which results tree the PySR curve reads "
                             f"(default: {DEFAULT_GROUP}). A non-default tree gets a "
                             f"'-<variant>' label suffix so an overlay stays readable.")
    parser.add_argument("--pysr-label-suffix", default=None, metavar="S",
                        help="Override that suffix (pass '' to reuse the plain label).")
    parser.add_argument("--pysr-groups", nargs="*", default=None, metavar="DIR",
                        help="Draw SEVERAL PySR arms at once, one curve per results "
                             "tree (e.g. pysr pysr_nostop); each gets its own "
                             "'-<variant>' label, colour and marker. Default: just "
                             "--pysr-group. Needed wherever a figure identifies "
                             "methods by ARTIFACT discovery rather than by cache row, "
                             "which is every figure except the pure --reuse-ood curves.")


def apply_group_arg(args) -> None:
    set_group(getattr(args, "pysr_group", DEFAULT_GROUP),
              getattr(args, "pysr_label_suffix", None))


def active_groups(args) -> tuple:
    """The PySR trees a figure should draw, honouring --pysr-groups then --pysr-group.
    Always a tuple, so a caller can pass it straight to llmsr_methods_for_groups /
    srbench_pkls_for_groups."""
    gs = getattr(args, "pysr_groups", None)
    return tuple(gs) if gs else (getattr(args, "pysr_group", DEFAULT_GROUP),)


def is_pysr(label: str) -> bool:
    """Exact/prefix match, never a substring test.  "PySR" is short and would otherwise
    cross-fire against any future label containing it."""
    return str(label) == LABEL or str(label).startswith(LABEL + "-")


# A variant tree (a re-run, a different budget or early-stop setting) is drawn in a
# DARKER shade of the same hue: same family says "this is PySR", the shade says
# "different arm".  Both arms are ours-run, so both stay SOLID.  Colour is never the
# only channel -- plot_markers hands a suffixed label its own shape.
VARIANT_COLOR = "#713f12"  # amber-900 -- a LIGHTNESS gap, so the two arms stay
                           # separable in greyscale, not just by hue


# Per-arm overrides when several variants share a figure.  The 1800 s arm takes the
# default amber, free because the 1 h default tree is never drawn (2026-09-23).
LABEL_COLORS = {"PySR-1800": COLOR}


def color(label: str = None):
    """Amber for the default tree, a darker amber for any suffixed variant.

    Called with no label (or the plain label) this is the original constant, so every
    single-arm figure renders exactly as before.
    """
    if label is not None and str(label) in LABEL_COLORS:
        return LABEL_COLORS[str(label)]
    if label is not None and str(label).startswith(LABEL + "-"):
        return VARIANT_COLOR
    return COLOR


def linestyle(label: str = None):
    return LINESTYLE


def srbench_pysr_pkl(root, tau, group=None) -> "str | None":
    """The merged SRBench PySR artifact at noise tau, or None if absent."""
    d = results_io.group_noise_dir(root, group or PYSR_GROUP, tau)
    p = os.path.join(d, SRBENCH_ARTIFACT)
    return p if os.path.isfile(p) else None


def srbench_pkls_for_groups(root, tau, groups):
    """{display_label: artifact} for the SRBench half across several trees."""
    out = {}
    for g in groups:
        p = srbench_pysr_pkl(root, tau, group=g)
        if p:
            out[label_for_group(g)] = p
    return out


def _method_artifact(d):
    """The per-method result file inside dir `d` (results.pkl.gz or legacy .jsonl)."""
    for name in ("results.pkl.gz", "results.jsonl"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def llmsr_pysr_methods(root, tau, split, group=None):
    """{display_label: results_artifact} for the PySR LLM-SRBench run at noise tau.

    Mirrors discover_methods_from_dirs' output so callers merge the two dicts.  Empty
    when the tree does not exist, so a figure asked to --include-pysr before the run has
    landed simply draws without it.

    Shard dirs are skipped.  That matters more here than for most methods: eval_pysr.py
    runs 24 shards per cell and each cell's job merges its own into pysr_<split>/, but a
    cell killed mid-flight leaves pysr_<split>__shardNNofNN/ behind.  Plotting one of
    those as the whole benchmark would understate the solve rate ~24x.
    """
    out = {}
    dirs = results_io.list_llmsr_method_dirs(root, tau, split,
                                             groups=(group or PYSR_GROUP,))
    for d in dirs:
        base = os.path.basename(d)
        if "__shard" in base:
            continue
        art = _method_artifact(d)
        if art is None:
            continue
        out[label_for_group(group) if group else DISPLAY_LABEL] = art
    return out


def llmsr_methods_for_groups(root, tau, split, groups):
    """{display_label: artifact} across SEVERAL PySR trees at once.

    Each tree contributes its own label (see label_for_group), so an overlay of the
    default and a variant arm keeps them apart.  A missing tree contributes nothing.
    """
    out = {}
    for g in groups:
        out.update(llmsr_pysr_methods(root, tau, split, group=g))
    return out
