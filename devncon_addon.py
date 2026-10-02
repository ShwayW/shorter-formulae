"""devncon_addon.py -- shared helpers for the "<model> + D&C" combined methods.

These are our own transformer checkpoints (m89 = 89M, m145 = 145M) decoded
with the DEVNCON decomposition search instead of plain beam search, run through the
same eval pipeline into results/mymodels_devncon/ (SRBench pkls + LLM-SRBench dirs,
10 seeds x 4 noises).

This module centralises their group name, display labels, colours, and discovery so
the plotting scripts opt them in with --include-devncon (default off).
"""
import os

import results_io

# The group dirs the D&C artifacts are read from, in PRIORITY order: when the same
# display label exists in more than one, the FIRST group wins and the others are
# ignored.  That matters because these trees are different PROTOCOLS, not just
# different runs -- mymodels_devncon3 is the complete 80-task array run
# (slurm/eval_mymodels_devncon_array.sh, job 4700017: 18400/18400 cells, no OOM holes),
# while the older single-job mymodels_devncon is holed by the CUDA-OOM losses that the
# array form exists to fix.  Merging them per label rather than per file keeps 89M/145M
# reading from devncon3 exactly as before, while letting an arm that exists in only one
# tree (145M_80, run on our cluster into mymodels_devncon) still be found.
#
# Comma-separated; override to compare trees, e.g. DEVNCON_GROUPS=mymodels_devncon.
DEVNCON_GROUPS = tuple(
    g.strip() for g in os.environ.get(
        "DEVNCON_GROUPS", os.environ.get("DEVNCON_GROUP",
                        "mymodels_devncon3,mymodels_devncon,mymodels_devncon_ftnoise,"
                        "mymodels_devncon_ftnoise_48h")
    ).split(",") if g.strip()
)
# Back-compat alias: the single highest-priority group.
DEVNCON_GROUP = DEVNCON_GROUPS[0]

# raw model_name (SRBench pkl) / stripped dir stem (LLM-SRBench) -> display label
# Note: includes both pre- and post-_remap_scale_labels versions
_RAW_TO_DISPLAY = {
    # Post-rename labels; m89/m145 kept for archived runs under _old/.
    # Post-simp-rename spellings first; the pre-rename ones name the SAME weights.
    "89M_40_simp1_devncon":   "89M+D&C",
    "145M_40_simp1_devncon":  "145M+D&C",
    "145M_80_simp0_devncon":  "145M-len80+D&C",
    "89M_40_devncon":   "89M+D&C",
    "145M_40_devncon":  "145M+D&C",
    "m89_devncon": "89M+D&C",
    "m145_devncon":   "145M+D&C",
    # The noise-finetuned 145M float arm (epoch-27 snapshot), evaluated 2026-09-18.
    "145M_80_simp1_float_ftnoise_e27_devncon": "145M-len80-float_ftnoise_e27+D&C",
    # The 48 h finetune, evaluated 2026-09-22.  Without this entry display_label()
    # passes the raw name through and the legend reads
    # "ours_80_simp1_float_ftnoise_48h_devncon" -- while the cache-fed curve for the
    # SAME arm draws correctly, so the figure shows it twice under two names.
    "145M_80_simp1_float_ftnoise_48h_devncon": "145M-len80-float_ftnoise_48h+D&C",
    "89M_devncon":      "89M+D&C",      # post-remap version
    "145M_devncon":     "145M+D&C",     # post-remap version
    # 145M_80: same 145M architecture, formula-length budget 80 instead of 40.
    "145M_80_devncon":    "145M-len80+D&C",
    "145M-len80_devncon": "145M-len80+D&C",   # post-remap version
}

# The combined methods actually shown
DISPLAY_LABELS = ("89M+D&C", "145M+D&C", "145M-len80+D&C")

# Colours: distinct hues, standing out from orange (89M), green (145M), and TPSR's
# dark shades.  145M+D&C -- the headline arm, and the only D&C curve the camera-ready
# figures draw -- is RED, the most legible line on a crowded panel; the other two keep
# the cyan family they always had.  The rose the LLMSR-Llama overlay used to own moved
# to #0369a1 (llmsr_overlay.py / plot_llmsrbench_curves_best_seed.py), so no two curves
# in these figures share a hue.
DEVNCON_COLORS = {
    "89M+D&C":  "#0891b2",    # cyan (close to teal, distinct from 89M orange family)
    "145M+D&C": "#f472b6",    # pink-400 -- swapped with "ours (fine tuned)" 2026-09-17
    "145M-len80+D&C": "#22d3ee",   # bright cyan -- same family, clearly lighter than both
    "145M-len80-float_ftnoise_e27+D&C": "#a16207",   # amber-700 -- the finetuned D&C arm
    "145M-len80-float_ftnoise_48h+D&C": "#a16207",   # amber-700 -- same, 48 h arm
}


def is_devncon_combo(label: str) -> bool:
    """True for a "<model> + D&C" combined method (raw label like 'm89_devncon' or
    display label like '89M+D&C'); False for everything else.

    Both spellings must match: the raw name is what the artifact filenames carry, while
    the caches (and every plot label) carry the display form, which has no "devncon" in
    it at all -- so a substring test alone let every display-labelled row slip past the
    --include-devncon filters and past plot_ood_vs_gap's is-this-ours line-style test."""
    low = label.lower()
    return "devncon" in low or "+d&c" in low


def display_label(raw: str) -> str:
    """Map a raw model_name / dir stem ('m89_devncon') to its display label
    ('89M+D&C'); pass anything unknown through unchanged."""
    return _RAW_TO_DISPLAY.get(raw, raw)


def srbench_devncon_pkls(root, tau):
    """The D&C SRBench eval_mymodels pkls at noise tau, one per arm.

    Walks DEVNCON_GROUPS in priority order and keeps the first file seen for each
    artifact name, so an arm present in several trees is read from exactly one of
    them (see the DEVNCON_GROUPS comment: those trees are different protocols)."""
    out = {}
    for group in DEVNCON_GROUPS:
        for p in results_io.list_srbench_pkls(root, tau, group):
            name = os.path.basename(p)
            if "_unscale" in name:
                continue
            out.setdefault(name, p)
    return list(out.values())


def _method_artifact(d):
    """The per-method result file inside dir `d` (results.pkl.gz or legacy .jsonl)."""
    for name in ("results.pkl.gz", "results.jsonl"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def llmsr_devncon_methods(root, tau, split):
    """{display_label: results_artifact} for the D&C LLM-SRBench dirs at noise tau --
    mirrors discover_methods_from_dirs' output so callers merge the two dicts.  (Those
    dirs carry 'devncon' in the name.)

    DEVNCON_GROUPS is walked in priority order and the first group to supply a given
    display label wins, so arms living in two trees are never silently mixed."""
    out = {}
    for group in DEVNCON_GROUPS:
        for d in results_io.list_llmsr_method_dirs(root, tau, split, groups=(group,)):
            if "_unscale" in os.path.basename(d):
                continue
            art = _method_artifact(d)
            if art is None:
                continue
            stem = os.path.basename(d).replace(f"_{split}", "")   # 145M_40_lsr_transform_devncon -> 145M_40_devncon
            out.setdefault(display_label(stem), art)
    return out
