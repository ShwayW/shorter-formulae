#!/usr/bin/env python3
"""
ood_transform.py -- names and x-axis conventions for the OOD-vs-gap plots.

    additive   X_ood = X + k*W   ->  box [min + k*W, max + k*W]
               (gen_ood_data.py, gen_ood_llmsrbench.py)

This module centralises the dataset paths, cache basenames and axis labels that the
OOD plotting/scoring scripts share, so a rename happens in one place:
plot_ood_vs_gap.py, plot_llmsrbench_ood_vs_gap.py, plot_llmsrbench_curves_best_seed.py,
score_llmsr_ood.py, augment_tpsr_caches.py,
augment_aifeynman_caches.py, llmsr_overlay.py.

HISTORY: this module used to abstract over TWO distribution-shift families -- the
additive one above and a multiplicative twin (X_ood = k*X, caches tagged "_mult",
driven by make_mult_ood_plots.sh and gen_ood_*_mult.py).  The multiplicative family
was discarded, so the `transform` parameter and the "_mult" cache tag are gone and
every name below is unconditional.  The module is kept because the additive paths
still benefit from having their names defined once.
"""

GAPS = [0, 1, 2, 4, 8, 16, 32, 64, 128]

# Retained under its old name so callers reading `ood_transform.ADDITIVE_GAPS` keep
# working; there is only one family now.
ADDITIVE_GAPS = GAPS


def default_gaps():
    return list(GAPS)


def _fmt(g):
    """Render a gap value as the integer token used in filenames (2, not 2.0)."""
    g = float(g)
    return int(g) if g == int(g) else g


def feynman_ood_path(g):
    """Relative path to the Feynman OOD point-cloud for gap g."""
    return f"datasets/feynman_ood_g{_fmt(g)}.pkl.gz"


def llmsr_ood_path(g):
    """Relative path to the LLM-SRBench OOD point-cloud for gap g."""
    return f"datasets/llmsrbench_ood_g{_fmt(g)}.pkl.gz"


def ood_basename(split, g):
    """score_llmsr_ood.py basename for (split, gap).  split in {feynman, lsr_transform}."""
    stem = "feynman_ood" if split == "feynman" else "llmsrbench_ood"
    return f"{stem}_g{_fmt(g)}.pkl.gz"


def srbench_cache_name(tau):
    """Basename of the SRBench OOD-R^2 cache (joined with the results root by callers)."""
    return f"ood_raw_noise{float(tau):g}.csv"


def llmsr_cache_name(tau):
    """Basename of the LLM-SRBench OOD-R^2 cache."""
    return f"llmsr_ood_raw_noise{float(tau):g}.csv"


def srbench_complexity_cache_name(tau):
    """Basename of the SRBench formula-complexity cache."""
    return f"complexity_noise{float(tau):g}.csv"


def llmsr_complexity_cache_name(tau):
    """Basename of the LLM-SRBench formula-complexity cache."""
    return f"llmsr_complexity_noise{float(tau):g}.csv"


def llmsr_backbone_cache_name(backbone, split, tau):
    """Basename of a score_llmsr_ood.py per-backbone cache (llmsr / gemini / llama)."""
    return f"llmsr_{backbone}_ood_raw_{split}_noise{float(tau):g}.csv"


def oneshot_backbone_cache_name(backbone, split, tau):
    """Basename of a score_oneshot_ood.py per-backbone cache (llama / gemini35flash).

    Same schema and directory as llmsr_backbone_cache_name, different prefix so the
    one-shot arms never collide with the LLMSR ones in results/."""
    return f"oneshot_{backbone}_ood_raw_{split}_noise{float(tau):g}.csv"


def baseline_gap():
    """The in-distribution x value.  Used to rank best seeds and to drop combo rows
    when a method is not opted in."""
    return 0


def xlabel():
    return "OOD gap"


def xticklabel(g):
    g = _fmt(g)
    return "0" if g == 0 else str(g)


# ---------------------------------------------------------------------------------
# Monotone-gap rule
# ---------------------------------------------------------------------------------
# A formula that has already failed at gap k cannot become correct again at a LARGER
# gap: the gap-k box is nested inside the gap-(k+1) box, so "still right at k+1" is
# only meaningful if it was right at every gap up to there.  Scored independently per
# gap, the raw caches DO produce non-monotone curves -- a structurally wrong formula
# can, by coincidence, track the target better in a farther-out region than a nearer
# one, which is how a bigger k ends up sitting above a smaller k on the plots.
#
# So "solved at gap k" is defined here as "solved at EVERY gap <= k": a prefix-AND
# over the gap axis, applied per (algorithm, seed, problem).  Once a series fails, it
# stays failed.  Note this includes the in-distribution baseline (gap 0, the smallest
# k): a formula that is already wrong on its own training distribution is not credited
# with extrapolating anywhere.
#
# Applied to the RAW long-form table, before any thresholding, so every downstream
# solve-rate counter -- plot_ood_vs_gap, plot_llmsrbench_ood_vs_gap,
# plot_llmsrbench_curves_best_seed, llmsr_overlay, oneshot_overlay -- inherits it
# without changing its own arithmetic.  Failures are marked by setting the R^2 to
# -inf rather than by dropping rows, so per-gap denominators (n_total, and
# plot_ood_vs_gap's dataset universe) are untouched: a monotonised failure counts as
# unsolved, exactly like a missing or below-threshold row.

MONOTONE_FAIL = float("-inf")


def problem_key(df):
    """The per-problem column of a raw OOD table: SRBench keys on `dataset`,
    LLM-SRBench on `equation_id`."""
    for c in ("dataset", "equation_id"):
        if c in df.columns:
            return c
    return None


def monotone_gap(df, r2_thr, *, key=None, gap_col="gap", value_col="ood_r2",
                 by=("algorithm", "seed")):
    """Enforce the monotone-gap rule on a raw OOD R^2 table; returns a new DataFrame.

    Rows whose (algorithm, seed, problem) series has already dropped below `r2_thr` at
    some smaller gap get `value_col` set to MONOTONE_FAIL.  Grouping columns that are
    absent are skipped, so this works on a single-method or single-seed table too.
    """
    import numpy as _np

    if df is None or len(df) == 0 or value_col not in df.columns or gap_col not in df.columns:
        return df
    key = key or problem_key(df)
    group = [c for c in list(by) + ([key] if key else []) if c in df.columns]
    if not group:
        return df

    out = df.copy()
    # A (group, gap) cell counts as solved if ANY of its rows clears the threshold --
    # the same "max over duplicate rows" the callers' own counters already use.  NaN
    # compares False, so a missing score is a failure and propagates like any other.
    cell = (out[value_col] >= r2_thr).groupby(
        [out[c] for c in group + [gap_col]]).transform("max")
    # Prefix-AND along the gap axis: walk each group's gaps in ascending order and
    # cummin, so the first False turns every later gap False.  Indexing through the
    # sorted view keeps the caller's original row order in the result.
    ordered = out.sort_values(group + [gap_col])
    keep = cell.loc[ordered.index].groupby([ordered[c] for c in group]).cummin()
    out.loc[keep[~keep.astype(bool)].index, value_col] = MONOTONE_FAIL
    return out


def common_problems(df, *, key=None, gap_col="gap"):
    """The problem keys present at EVERY gap in `df` (an empty set if there are none).

    Membership is taken over the UNION of algorithms at each gap, not per algorithm, so
    this stays the shared "dataset universe" that plot_ood_vs_gap's common denominator
    already means -- a problem a given method never attempted still belongs to the
    universe and still counts against it as unsolved.
    """
    key = key or problem_key(df)
    if df is None or len(df) == 0 or key is None or gap_col not in df.columns:
        return set()
    per_gap = [set(g[key]) for _, g in df.groupby(gap_col)]
    return set.intersection(*per_gap) if per_gap else set()


def restrict_to_common(df, *, key=None, gap_col="gap"):
    """Drop problems that are missing at one or more gaps, so the solve-rate
    DENOMINATOR is the same at every gap.

    Why this is needed on top of monotone_gap: a problem disappears from a gap when its
    GROUND TRUTH is undefined over the shifted input box -- gen_ood_data.py keeps only
    finite GT outputs and drops a problem when both shift directions come up short.  On
    SRBench that costs exactly one problem (feynman_I_26_2, Snell's law: arcsin(n*sin
    theta2) leaves arcsin's [-1,1] domain once n's box shifts past gap 2).  On
    LLM-SRBench it costs 13, because the lsr_transform split ALGEBRAICALLY INVERTS the
    Feynman equations and inversion manufactures partial functions -- III.10.19 forward
    is mom*sqrt(Bx^2+By^2+Bz^2) (total, survives to gap 128), while solved for Bx it is
    sqrt(E_n^2/mom^2 - By^2 - Bz^2), which needs E_n^2/mom^2 >= By^2+Bz^2 and dies at
    gap 4.  Six such equations go at gap 2 and seven more at gap 4.

    Those are problems where NO valid OOD test set exists at that gap, for any method --
    not problems anyone failed.  Left in, the shrinking denominator lets a solve RATE
    rise while the solved COUNT falls (which is how a larger k outranks a smaller one on
    the plots even after monotone_gap); counted as failures instead, every curve in the
    figure inherits a ~12-point cliff between gap 2 and gap 4 that is pure artifact.
    Restricting to the common subset asks one well-posed question across the whole x
    axis, and moves nothing at gap >= 4.
    """
    key = key or problem_key(df)
    if df is None or len(df) == 0 or key is None:
        return df
    keep = common_problems(df, key=key, gap_col=gap_col)
    if not keep:
        return df
    return df[df[key].isin(keep)]
