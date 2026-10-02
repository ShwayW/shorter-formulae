"""tpsr_addon.py -- shared helpers for the "<model> + TPSR" combined methods.

These are our own transformer checkpoints (m89 = 89M, m145 = 145M) decoded
with the TPSR search instead of plain beam search, run through the same eval pipeline
into results/mymodels_tpsr/ (SRBench pkls + LLM-SRBench dirs, 10 seeds x 4 noises).

This module centralises their group name, display labels, colours, and discovery so the
six-figure plotting scripts opt them in with one call each, behind --include-tpsr-combo
(default off, so every other figure is unchanged).  The input-unscaled (_unscale) variants are
never shown -- only the two raw checkpoints, matching the base 89M / 145M curves.
"""
import os

import results_io

# Comma-separated, like devncon_addon.DEVNCON_GROUPS: the canonical 89M/145M tree plus
# the noise-finetuned arm (2026-09-18).  Both must be listed or the figures draw only
# whichever one is named -- artifact discovery is per-group.
TPSR_GROUPS = tuple(
    g.strip() for g in os.environ.get(
        "TPSR_GROUPS", os.environ.get("TPSR_GROUP",
                                      "mymodels_tpsr,mymodels_tpsr_ftnoise,mymodels_tpsr_ftnoise_48h")
    ).split(",") if g.strip()
)
# Back-compat alias: the single highest-priority group.
TPSR_GROUP = TPSR_GROUPS[0]

# raw model_name (SRBench pkl) / stripped dir stem (LLM-SRBench) -> display label
_RAW_TO_DISPLAY = {
    # Post-rename labels; the m89/m145 spellings are kept for archived runs
    # under _old/ (the live results tree was migrated 2026-08-17).
    "89M_40_simp1_tpsr":   "89M+TPSR",
    "145M_40_simp1_tpsr":  "145M+TPSR",
    "89M_40_tpsr":   "89M+TPSR",
    "145M_40_tpsr":  "145M+TPSR",
    "m89_tpsr": "89M+TPSR",
    "m145_tpsr":   "145M+TPSR",
    # The noise-finetuned 145M float arm (epoch-27 snapshot), evaluated 2026-09-18.
    "145M_80_simp1_float_ftnoise_e27_tpsr": "145M-len80-float_ftnoise_e27+TPSR",
    # The 48 h finetune, evaluated 2026-09-22 -- see the devncon_addon note.
    "145M_80_simp1_float_ftnoise_48h_tpsr": "145M-len80-float_ftnoise_48h+TPSR",
}

# The two combined methods actually shown (raw checkpoints only).
DISPLAY_LABELS = ("89M+TPSR", "145M+TPSR")

# Colours: the darker shade of each base model's hue (base 89M = light orange,
# 145M = light green).  These dark shades are otherwise used only for the input-unscaled
# variants, which are excluded from these figures, so there is no clash.
TPSR_COLORS = {
    "89M+TPSR":  "#c2410c",   # burnt orange (89M family)
    "145M+TPSR": "#15803d",   # forest green  (145M family)
    # TPSR over the noise-finetuned 145M float arm.  Indigo: the TPSR family colours are
    # taken by the model families they sit on, and this arm's own red (#dc2626) belongs
    # to its plain beam curve.
    "145M-len80-float_ftnoise_e27+TPSR": "#4338ca",   # indigo-700
    "145M-len80-float_ftnoise_48h+TPSR": "#4338ca",   # indigo-700 (48 h arm)
}


def is_tpsr_combo(label: str) -> bool:
    """True for a "<model> + TPSR" combined method (raw label like 'm89_tpsr' or
    display label like '89M+TPSR'); False for the authors' standalone 'TPSR'
    baseline and everything else.  Used to filter these methods OUT of every figure
    unless --include-tpsr-combo is set."""
    if label == "TPSR":
        return False
    return "tpsr" in label.lower()


def display_label(raw: str) -> str:
    """Map a raw model_name / dir stem ('m89_tpsr') to its display label
    ('89M+TPSR'); pass anything unknown through unchanged."""
    return _RAW_TO_DISPLAY.get(raw, raw)


def srbench_tpsr_pkls(root, tau):
    """The two raw (non-unscaled) mymodels_tpsr SRBench eval_mymodels pkls at noise tau."""
    out = []
    for group in TPSR_GROUPS:
        out.extend(p for p in results_io.list_srbench_pkls(root, tau, group)
                   if "_unscale" not in os.path.basename(p))
    return out


def _method_artifact(d):
    """The per-method result file inside dir `d` (results.pkl.gz or legacy .jsonl)."""
    for name in ("results.pkl.gz", "results.jsonl"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def llmsr_tpsr_methods(root, tau, split):
    """{display_label: results_artifact} for the raw mymodels_tpsr LLM-SRBench dirs at
    noise tau -- mirrors discover_methods_from_dirs' output so callers merge the two
    dicts.  (Those dirs carry 'tpsr' in the name, which the default discovery skips.)"""
    out = {}
    dirs = results_io.list_llmsr_method_dirs(root, tau, split, groups=TPSR_GROUPS)
    for d in dirs:
        if "_unscale" in os.path.basename(d):
            continue
        art = _method_artifact(d)
        if art is None:
            continue
        stem = os.path.basename(d).replace(f"_{split}", "")   # m89_lsr_transform_tpsr -> m89_tpsr
        out[display_label(stem)] = art
    return out
