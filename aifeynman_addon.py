"""aifeynman_addon.py -- shared helpers for the AI Feynman 2.0 baseline on LLM-SRBench.

This is the ORIGINAL authors' AI Feynman (vendored in ./AI-Feynman/, driven by
eval_aifeynman.py --benchmark llmsrbench), NOT the repo's own aif2.py decomposition -- see
aif2_addon.py for that one.  The two are different methods and the labels keep them
apart: "AIFeynman" here vs "89M+AIF2" / "145M+AIF2" there.  Note that neither string
contains the other's discriminator, so no filter cross-fires: "AIFeynman" does not
contain "aif2" (aif2_addon.is_aif2_combo), and "89M+AIF2" does not equal "AIFeynman".

WHY THIS EXISTS
---------------
"AIFeynman" is ALREADY a method on the SRBench (Feynman) figures -- but there it comes
from SRBench's own published results JSON (compare_srbench.py), which covers only the
Feynman benchmark.  LLM-SRBench has no published AI Feynman numbers, so that curve was
simply absent from the LLM-SRBench figures.  eval_aifeynman.py --benchmark llmsrbench produces it by
running the real thing ourselves, and this module is what feeds it to the plots.

Because it is the same method at the same SRBench-tuned hyperparameters
(srbench/experiment/methods/tuned/params/_aifeynman.py), it deliberately reuses the
SRBench side's identity everywhere -- label "AIFeynman", steel-blue, pentagon marker,
dashed -- so one method looks the same in every figure family.  The provenance does
differ (published numbers on Feynman, our run on LLM-SRBench); that belongs in the
caption, not in a second visual identity.

This is the TPSR/AIF2 addon pattern: the curve is opt-in behind --include-aifeynman
(default off), so no existing figure changes unless asked.

RESULTS TREE
------------
    results/aifeynman/[seed<N>/]noise_<tau>/llmsrbench/aifeynman_<split>/results.pkl.gz

"aifeynman" is deliberately NOT added to results_io.GROUPS: that tuple drives the
DEFAULT discovery for every figure, and appending to it would silently add this curve
to all of them.  Discovery here passes groups=(GROUP,) explicitly instead, exactly as
aif2_addon does for its own tree.
"""
import os

import results_io

DEFAULT_GROUP = "aifeynman"

# Display identity, matched to the SRBench side so the method is recognisable across
# figure families:
#   label   -- compare_srbench.py / plot_ood_vs_gap.py already use "AIFeynman"
#   colour  -- plot_ood_vs_gap._tf_color: steel blue
#   marker  -- plot_markers._METHOD_MARKER: "p" (pentagon), already reserved
#   dashed  -- how the SRBench baselines are drawn there
LABEL = "AIFeynman"
COLOR = "#457b9d"          # steel blue -- plot_ood_vs_gap._tf_color("AIFeynman")
LINESTYLE = "--"

# --- module state, rebound by set_group() ---------------------------------------
AIFEYNMAN_GROUP = DEFAULT_GROUP
LABEL_SUFFIX = ""
DISPLAY_LABEL = LABEL


def default_label_suffix(group: str) -> str:
    """The suffix a given results tree contributes to its display label.

    The default tree contributes nothing.  A variant tree (e.g. "aifeynman_bf120",
    a re-run at a different BF_try_time) gets a suffix so two runs plotted together
    stay tellable apart -- the same convention as aif2_addon.default_label_suffix.
    """
    if group == DEFAULT_GROUP:
        return ""
    if group.startswith(DEFAULT_GROUP):
        tail = group[len(DEFAULT_GROUP):].strip("_")
        return f"-{tail}" if tail else ""
    return f"-{group}"


def set_group(group: str, label_suffix: str | None = None) -> None:
    """Point this module at results/<group>/ and rebuild the display label."""
    global AIFEYNMAN_GROUP, LABEL_SUFFIX, DISPLAY_LABEL
    AIFEYNMAN_GROUP = group
    LABEL_SUFFIX = default_label_suffix(group) if label_suffix is None else label_suffix
    DISPLAY_LABEL = LABEL + LABEL_SUFFIX


def add_group_arg(parser) -> None:
    """Add --aifeynman-group / --aifeynman-label-suffix to a script's parser."""
    parser.add_argument(
        "--aifeynman-group", default=DEFAULT_GROUP, metavar="GROUP",
        help=f"Results tree for the AI Feynman baseline: results/<GROUP>/. "
             f"Default: {DEFAULT_GROUP}.")
    parser.add_argument(
        "--aifeynman-label-suffix", default=None, metavar="S",
        help="Override the display-label suffix implied by --aifeynman-group. "
             "Pass an empty string to keep the plain 'AIFeynman' label.")


def apply_group_arg(args) -> None:
    """Apply --aifeynman-group / --aifeynman-label-suffix, if the script defined them."""
    group = getattr(args, "aifeynman_group", None)
    if group:
        set_group(group, getattr(args, "aifeynman_label_suffix", None))


def is_aifeynman(label: str) -> bool:
    """True for the AI Feynman baseline's display label (with or without a tree
    suffix).  Deliberately NOT a substring test on 'aif': "89M+AIF2" must not match."""
    return str(label) == LABEL or str(label).startswith(LABEL + "-")


def color(label: str = None):
    return COLOR


def linestyle(label: str = None):
    return LINESTYLE


def _method_artifact(d):
    """The per-method result file inside dir `d` (results.pkl.gz or legacy .jsonl)."""
    for name in ("results.pkl.gz", "results.jsonl"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def llmsr_aifeynman_methods(root, tau, split, group=None):
    """{display_label: results_artifact} for the AI Feynman LLM-SRBench run at noise
    tau -- mirrors discover_methods_from_dirs' output so callers merge the two dicts.

    Empty when the tree does not exist, so a figure asked to --include-aifeynman
    before the (long) run has landed simply draws without it.

    Shard dirs (<method>__shard<i>of<N>, written by a Slurm array before the
    --merge-shards pass) are skipped: they are partial by construction, and plotting
    one as if it were the whole benchmark would understate the solve rate.
    """
    out = {}
    dirs = results_io.list_llmsr_method_dirs(root, tau, split,
                                            groups=(group or AIFEYNMAN_GROUP,))
    for d in dirs:
        base = os.path.basename(d)
        if "__shard" in base:
            continue
        art = _method_artifact(d)
        if art is None:
            continue
        out[DISPLAY_LABEL] = art
    return out
