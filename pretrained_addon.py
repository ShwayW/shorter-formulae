#!/usr/bin/env python3
"""
pretrained_addon.py -- the NeSymReS / tf4sr / SymFormer twin of aifeynman_addon.py
and phye2e_addon.py.

Each of these three pretrained-transformer baselines lives in its OWN results tree
(results/nesymres/, results/tf4sr/, results/symformer/), none of which is in
results_io.GROUPS -- so the default LLM-SRBench discovery never sees them and a
figure cannot draw them without an explicit opt-in.  This module supplies the
{display_label: artifact} mapping that plot_llmsrbench_ood_vs_gap.py and its
time/complexity twin merge into their discovered methods, exactly as the AI Feynman
and PhyE2E addons do.

OPT-IN ONLY.  Nothing here changes an existing figure: sets 1-5 never pass
--include-pretrained, so their method lists are untouched.  These three are drawn
by plot_transformer_methods.py, which explains in its docstring why they are kept
out of the set figures (they cover 16-105 of the benchmark's problems, not all of
it, so a solve rate against them is not comparable to a full-coverage arm's).

Colours match plot_ood_vs_gap._tf_color so a method keeps one colour across every
figure it appears in.
"""
import os

import results_io

# label -> results group.  The label is also the key these methods carry in the OOD
# caches (augment_pretrained_ood_caches.py writes them), so the two agree by
# construction rather than by a lookup table that could drift.
GROUPS = {
    "nesymres":  "nesymres",
    "tf4sr":     "tf4sr",
    "symformer": "symformer",
}

# Kept in step with plot_ood_vs_gap._tf_color.
COLORS = {
    "nesymres":  "#0284c7",     # sky blue
    "tf4sr":     "#b45309",     # dark amber
    "symformer": "#475569",     # slate
}

# SRBench artifact basename per group.
SRBENCH_ARTIFACTS = {
    "nesymres":  "eval_nesymres.pkl.gz",
    "tf4sr":     "eval_tf4sr.pkl.gz",
    "symformer": "eval_symformer.pkl.gz",
}


def is_pretrained(label) -> bool:
    return str(label) in GROUPS


def color(label):
    return COLORS.get(str(label))


def add_arguments(parser):
    """Register --include-pretrained on a figure's parser."""
    parser.add_argument("--include-pretrained", action="store_true",
                        help="Also include the NeSymReS / tf4sr / SymFormer baselines "
                             "from results/{nesymres,tf4sr,symformer}/. Off by default; "
                             "these cover only part of the benchmark -- see "
                             "plot_transformer_methods.py.")
    parser.add_argument("--pretrained-common-subset", action="store_true",
                        help="Restrict the three baselines' time/complexity rows to the "
                             "problems all three attempted (the SymFormer-sized set). "
                             "Note it restricts THOSE THREE ROWS only -- the other arms "
                             "come from caches that hold one aggregate row each.")


def llmsr_methods(root, tau, split, labels=None):
    """{display_label: results artifact} for the LLM-SRBench runs at noise tau.

    Mirrors discover_methods_from_dirs' output so callers merge the two dicts.  Empty
    for any tree that does not exist, so a figure asked to include these before a run
    lands simply draws without them rather than failing.

    Each label owns ONE method dir (<label>_<split>), and the per-seed artifacts under
    it are pooled by the caller's loader -- the same shape the other addons return.
    """
    out = {}
    for label, group in GROUPS.items():
        if labels is not None and label not in labels:
            continue
        d = results_io.find_llmsr_method_dir(root, tau, f"{label}_{split}",
                                             groups=(group,))
        if not d:
            continue
        art = os.path.join(d, results_io.RESULTS_NAME)
        if os.path.exists(art):
            out[label] = art
    return out


def srbench_pkls(root, tau, labels=None):
    """{display_label: [per-seed SRBench artifact, ...]} at noise tau."""
    out = {}
    for label, group in GROUPS.items():
        if labels is not None and label not in labels:
            continue
        art = SRBENCH_ARTIFACTS[label]
        paths = []
        for seed in range(42, 52):
            p = os.path.join(
                results_io.noise_dir(
                    results_io.seed_dir(os.path.join(root, group), seed), tau), art)
            if os.path.exists(p):
                paths.append(p)
        if paths:
            out[label] = paths
    return out


# -- time / complexity rows ---------------------------------------------------------
#
# The dot-plot figures (plot_time_complexity_vs_noise.py and its LLM-SRBench twin)
# source every other method through a cache or a discovery pass that does not know
# these three trees.  These helpers read the per-seed artifacts directly and hand back
# the same {label: {noise: (mean, std)}} tables those figures build, so a caller only
# has to merge the dicts -- the same contract as oneshot_overlay.add_time_complexity_rows.
#
# They live here rather than in plot_transformer_methods.py (where they were written)
# so that the COMBINED set-1/set-2-style figures can draw these arms through the shared
# row-drawers instead of re-implementing the layout.

def _seed_srbench_paths(root, group, art, tau):
    for seed in range(42, 52):
        p = os.path.join(
            results_io.noise_dir(
                results_io.seed_dir(os.path.join(root, group), seed), tau), art)
        if os.path.exists(p):
            yield p


def _seed_llmsr_paths(root, group, label, split, tau):
    for seed in range(42, 52):
        p = os.path.join(
            results_io.llmsr_dir(
                results_io.seed_dir(os.path.join(root, group), seed),
                tau, f"{label}_{split}"), results_io.RESULTS_NAME)
        if os.path.exists(p):
            yield p


def srbench_common_datasets(root, noises):
    """Datasets all three baselines attempted (in practice the SymFormer-sized set)."""
    sets = []
    for label, group in GROUPS.items():
        seen = set()
        for tau in noises:
            for p in _seed_srbench_paths(root, group, SRBENCH_ARTIFACTS[label], tau):
                try:
                    _m, rows = results_io.load_rows_any(p)
                except Exception:
                    continue
                seen |= {r.get("dataset") for r in rows if r.get("dataset")}
        if seen:
            sets.append(seen)
    return set.intersection(*sets) if sets else None


def llmsr_common_problems(root, noises, split):
    """equation_ids all three baselines attempted."""
    sets = []
    for label, group in GROUPS.items():
        seen = set()
        for tau in noises:
            for p in _seed_llmsr_paths(root, group, label, split, tau):
                try:
                    _m, rows = results_io.load_rows_any(p)
                except Exception:
                    continue
                seen |= {r.get("equation_id") for r in rows if r.get("equation_id")}
        if seen:
            sets.append(seen)
    return set.intersection(*sets) if sets else None


def add_srbench_time_complexity_rows(st, sc, root, noises, thr, common=None):
    """Merge the three SRBench rows into a set-2 style figure's series dicts.

    Read from the per-seed artifacts rather than the complexity cache: the cache stores
    ONE aggregate row per method, so there would be nothing to restrict when `common`
    asks for a sub-population.  Complexity averages SOLVED formulae only (r2 >= thr),
    matching plot_complexity_vs_noise._load_complexity; time averages every attempted
    problem, matching collect_timing.  Both spreads are ACROSS PROBLEMS, as on the
    SRBench side generally.
    """
    from compare_srbench import formula_complexity
    import numpy as np

    for label, group in GROUPS.items():
        for tau in noises:
            t_vals, c_vals = [], []
            for p in _seed_srbench_paths(root, group, SRBENCH_ARTIFACTS[label], tau):
                try:
                    _meta, rows = results_io.load_rows_any(p)
                except Exception:
                    continue
                for r in rows:
                    if common is not None and r.get("dataset") not in common:
                        continue
                    tv = r.get("time")
                    if tv is not None and np.isfinite(tv):
                        t_vals.append(float(tv))
                    if (r.get("r2") or 0.0) >= thr:
                        c = formula_complexity(r.get("predicted_formula"))
                        if c is not None and np.isfinite(c):
                            c_vals.append(float(c))
            if t_vals:
                st.setdefault(label, {})[tau] = (float(np.mean(t_vals)),
                                                 float(np.std(t_vals)))
            if c_vals:
                sc.setdefault(label, {})[tau] = (float(np.mean(c_vals)),
                                                 float(np.std(c_vals)))


def add_llmsr_time_complexity_rows(st, sc, root, noises, split, thr, common=None):
    """The LLM-SRBench twin of add_srbench_time_complexity_rows.

    The LLM-SRBench artifacts use that benchmark's row schema -- `search_time`,
    `discovered_equation`, r2 nested under `id_metrics` -- so this cannot share the
    SRBench reader.  Per-seed means are averaged and the std is ACROSS SEEDS, matching
    _method_complexity on the LLM-SRBench side.
    """
    from compare_srbench import formula_complexity
    import numpy as np

    for label, group in GROUPS.items():
        for tau in noises:
            per_seed_t, per_seed_c = [], []
            for p in _seed_llmsr_paths(root, group, label, split, tau):
                try:
                    _m, rows = results_io.load_rows_any(p)
                except Exception:
                    continue
                ts, cs = [], []
                for r in rows:
                    if common is not None and r.get("equation_id") not in common:
                        continue
                    tv = r.get("search_time")
                    if tv is not None and np.isfinite(tv):
                        ts.append(float(tv))
                    idm = r.get("id_metrics") or {}
                    if float(idm.get("r2") or 0.0) >= thr:
                        c = formula_complexity(r.get("discovered_equation"))
                        if c is not None and np.isfinite(c):
                            cs.append(float(c))
                if ts:
                    per_seed_t.append(float(np.mean(ts)))
                if cs:
                    per_seed_c.append(float(np.mean(cs)))
            if per_seed_t:
                st.setdefault(label, {})[tau] = (float(np.mean(per_seed_t)),
                                                 float(np.std(per_seed_t)))
            if per_seed_c:
                sc.setdefault(label, {})[tau] = (float(np.mean(per_seed_c)),
                                                 float(np.std(per_seed_c)))
