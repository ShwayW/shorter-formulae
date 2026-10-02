"""e2e_tpsr_addon.py -- the STANDALONE TPSR baseline on LLM-SRBench.

WHAT THE CURVE IS -- "TPSR" (legend: "e2e+TPSR"), not "<model>+TPSR"
-------------------------------------------------------------------
This is the original authors' TPSR (Shojaee et al.): MCTS decoding on top of the FAIR
E2E transformer, vendored in ./TPSR/ and driven by
`eval_e2e_tpsr.py --benchmark llmsrbench`.  It is the SAME method the SRBench figures
already draw from the authors' published csv (plot_ood_vs_gap.py appends the "TPSR"
label and renames it to "e2e+TPSR" in the legend) -- so it keeps that identity here:
same label, same teal, same filled-plus marker, same legend text.  One method reads the
same in every figure family.

Do NOT confuse it with tpsr_addon, which is "<our checkpoint> + TPSR" (89M+TPSR,
145M+TPSR) -- our encoder with the same search.  Those are a different arm and live in
results/mymodels_tpsr/.  is_e2e_tpsr() is an EXACT match on "TPSR" precisely so it can
never cross-fire with those labels, which end with the same four characters.

WHY IT NEEDS AN ADDON AT ALL
----------------------------
The run lands in the "tpsr" group, which IS in results_io.GROUPS, so
list_llmsr_method_dirs() finds its directory -- but every LLM-SRBench plot script filters
discovery through _SKIP_SUBSTR = ("gemma", "tpsr", "oracle"), whose "tpsr" entry exists
to keep the opt-in <model>+TPSR combos out of the default figures.  That filter catches
`e2e_tpsr_lsr_transform` too.  So this arm is wired in explicitly, behind
--include-e2e-tpsr, exactly as phye2e_addon / aifeynman_addon are: no existing figure
changes unless asked.

FORMULA FORMAT
--------------
The llmsrbench branch of eval_e2e_tpsr.py stores an E2E-style INFIX formula over
x_0..x_N (TPSR/tpsr_llmsrbench.py), not our prefix PN.  plot_llmsrbench_ood_vs_gap's
_is_infix() heuristic routes it to the numpy evaluator on the `x_<digit>` test, the same
path the e2e and PhyE2E baselines take, and compare_srbench.formula_complexity counts it
through its sympify fallback.  Nothing new had to be taught to either.

RESULTS TREE (seed-merged by merge_seed_results.py, as every other arm is)
    results/tpsr/noise_<tau>/llmsrbench/e2e_tpsr_<split>/results.pkl.gz
with the per-seed sources retained under results/tpsr/seed<N>/ .
"""
import os

import results_io

GROUP = "tpsr"
METHOD_DIRNAME = "e2e_tpsr"          # <METHOD_DIRNAME>_<split> inside llmsrbench/

# Display identity -- deliberately identical to the SRBench figures' standalone TPSR:
# teal (plot_ood_vs_gap._tf_color), filled plus (plot_markers._METHOD_MARKER["TPSR"]),
# solid.  Solid because we RUN this ourselves on LLM-SRBench, and because on the SRBench
# side it is in tf_label_set and so already draws solid there.
LABEL = "TPSR"
COLOR = "#0d9488"                    # teal-600
LINESTYLE = "-"

# Legend TEXT only.  The algorithm identity stays "TPSR" everywhere -- that is what the
# caches, --only filters and colour lookups key on -- but readers need to be told which
# backbone the search ran on, exactly as plot_ood_vs_gap._LEGEND_RENAME does.
LEGEND_RENAME = {LABEL: "e2e+TPSR"}


def is_e2e_tpsr(label) -> bool:
    """Exact match, never a substring test: "89M+TPSR" and "145M+TPSR" are a DIFFERENT
    arm (tpsr_addon) and both end in "TPSR"."""
    return str(label) == LABEL


def color(label: str = None):
    return COLOR


def linestyle(label: str = None):
    return LINESTYLE


def legend_label(label):
    """The text this label appears under in a legend ("TPSR" -> "e2e+TPSR")."""
    return LEGEND_RENAME.get(label, label)


def rename_legend(labels):
    """Map a whole sequence of legend labels through LEGEND_RENAME."""
    return [legend_label(l) for l in labels]


def add_include_arg(parser, extra: str = "") -> None:
    """The shared --include-e2e-tpsr flag, so its help text reads the same everywhere."""
    parser.add_argument(
        "--include-e2e-tpsr", action="store_true",
        help="Also draw the standalone TPSR baseline (the original authors' MCTS "
             "decoding over the E2E transformer, run by eval_e2e_tpsr.py --benchmark "
             "llmsrbench) from results/tpsr/. Off by default. Teal, filled plus, "
             "legend 'e2e+TPSR' -- matching its SRBench-figure identity." + extra)


def _method_artifact(d):
    """The per-method result file inside dir `d` (results.pkl.gz or legacy .jsonl)."""
    for name in ("results.pkl.gz", "results.jsonl"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def llmsr_methods(root, tau, split, group=GROUP):
    """{display_label: results_artifact} for the standalone TPSR run at noise tau.

    Mirrors discover_methods_from_dirs' output so callers merge the two dicts.  Empty
    when the tree does not exist, so a figure asked to --include-e2e-tpsr before the run
    has landed simply draws without it.

    Two dirs are deliberately excluded:
      * "__shard" dirs -- partial by construction (the run is sharded 16 ways over the
        111 problems); plotting one as the whole benchmark understates the solve rate.
      * anything that is not <METHOD_DIRNAME>_<split> -- the "tpsr" group also holds the
        SRBench artifact and could later hold other methods.
    """
    out = {}
    want = f"{METHOD_DIRNAME}_{split}"
    for d in results_io.list_llmsr_method_dirs(root, tau, split, groups=(group,)):
        base = os.path.basename(d)
        if base != want:
            continue
        art = _method_artifact(d)
        if art is not None:
            out[LABEL] = art
    return out
