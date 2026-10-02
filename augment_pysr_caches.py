#!/usr/bin/env python3
"""
augment_pysr_caches.py -- the PySR twin of augment_phye2e_caches.py /
augment_tpsr_caches.py.  Compute the OOD R^2 and formula complexity of the PySR run and
append them to the plotting caches, so the existing --reuse-ood / --reuse-cache paths
draw the curve without recomputing everything.

WHY THIS IS NEEDED (and not optional)
-------------------------------------
`plot_ood_vs_gap.py --include-pysr` computes PySR live, so a fresh plot is fine.  But
with --reuse-ood the plot scripts read the cached table and only recompute GAPS that are
missing -- never ALGORITHMS that are missing (plot_ood_vs_gap.py:267-274,
plot_llmsrbench_ood_vs_gap.py:365-371).  So without this script a --reuse-ood figure
silently DROPS the PySR curve rather than failing, which is the worst kind of wrong.
Run this once after the run lands, exactly as make_requested_plots.sh does for the
TPSR / D&C / AI Feynman / PhyE2E curves.

Like PhyE2E and unlike AI Feynman, PySR has BOTH halves -- we run it on SRBench
(Feynman) and LLM-SRBench ourselves, because SRBench publishes no PySR results at all
(see pysr_addon.py).  So this writes to four cache families:

    results/ood_raw_noise<tau>.csv              SRBench OOD R^2      (per gap/dataset/seed)
    results/complexity_noise<tau>.csv           SRBench complexity
    results/llmsr_ood_raw_..._noise<tau>.csv    LLM-SRBench OOD R^2
    results/llmsr_complexity_noise<tau>.csv     LLM-SRBench complexity

FORMULA FORMAT
--------------
PySR emits infix in x_0..x_N (the e2e convention), NOT our prefix PN -- so the SRBench
half evaluates through compare_srbench._ood_eval_infix with style "x_0", exactly as e2e
and PhyE2E do.  Getting this wrong would not crash; it would quietly score every formula
as NaN, so the style is passed explicitly rather than sniffed.

Idempotent: rerunning replaces this method's rows rather than duplicating them, so it is
safe to run repeatedly as more of the sweep lands.

USAGE
    python augment_pysr_caches.py                 # all 4 noises (skips missing)
    python augment_pysr_caches.py --noises 0      # just the finished noise
    python augment_pysr_caches.py --bench srbench # one half only
"""
import argparse
import gzip
import os
import pickle
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import ood_transform
import pysr_addon
from augment_tpsr_caches import _append_idempotent, _llmsr_ood_rows
from compare_srbench import _ood_eval_infix, _ood_load_e2e_all
from plot_ood_vs_gap import _label_for_path
from plot_complexity_vs_noise import _load_complexity
from plot_llmsrbench_ood_vs_gap import load_method_data
from plot_llmsr_time_complexity_vs_noise import _method_complexity

DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]
DEFAULT_R2_THR = 0.99


# -- SRBench -------------------------------------------------------------------

def _srbench_ood_rows(pysr_pkl, gaps, label):
    """Per-(gap, algorithm, dataset, seed) OOD R^2 for the PySR SRBench artifact.

    The infix branch of compare_srbench.compute_ood_r2, with no baselines / e2e / GT, so
    only this one method is evaluated.  eval_pysr.py writes the same row schema as
    eval_e2e.py (dataset, seed, predicted_formula), which is exactly what
    _ood_load_e2e_all reads -- hence no PySR-specific loader.
    """
    rows = []
    for g in gaps:
        ood_path = os.path.join(SCRIPT_DIR, ood_transform.feynman_ood_path(g))
        if not os.path.exists(ood_path):
            print(f"  [warn] missing {ood_path}")
            continue
        with gzip.open(ood_path, "rb") as fh:
            ood_datasets = pickle.load(fh)["datasets"]
        for ds, seed_fmls in _ood_load_e2e_all(pysr_pkl).items():
            if ds not in ood_datasets:
                continue
            X, y = ood_datasets[ds]["X"], ood_datasets[ds]["y"]
            for seed, formula in seed_fmls:
                rows.append({"algorithm": label, "dataset": ds,
                             "ood_r2": _ood_eval_infix(formula, "x_0", X, y),
                             "seed": int(seed), "gap": g})
        print(f"  [gap={g:>3}] SRBench OOD done", flush=True)
    return pd.DataFrame(rows, columns=["algorithm", "dataset", "ood_r2", "seed", "gap"])


def _srbench_complexity_rows(pysr_pkl, tau, r2_thr, label):
    """(algorithm, complexity, std, is_ours) for the PySR SRBench artifact.

    _load_complexity ALSO returns every SRBench published baseline (FEAT, Operon, ...),
    not just the pkl we asked about, so the one row we want has to be picked out by key
    -- relabelling everything it returns would write ~15 bogus PySR rows.  Same trap
    augment_tpsr_caches._srbench_complexity_rows guards with its '+TPSR' filter.

    The complexity here is the REPO's metric (sympy node count via formula_complexity),
    not the `complexity` field eval_pysr.py stores from PySR's own Pareto front --
    _load_pkl_gz overwrites that column on load.  See pysr_addon's docstring.

    is_ours=False: PySR is an external published method we happen to run ourselves, not
    one of our checkpoints -- the complexity figures style the two groups differently.
    """
    cplx, _mine, _common = _load_complexity([pysr_pkl], None, None, r2_thr, tau)
    key = _label_for_path(pysr_pkl)          # 'pysr' (from the artifact's model_name)
    if key not in cplx:
        print(f"  [warn] no complexity row for {key!r} "
              f"(got {sorted(cplx)[:4]}...) -- skipping")
        return pd.DataFrame(columns=["algorithm", "complexity", "std", "is_ours"])
    c, s = cplx[key]
    return pd.DataFrame([{"algorithm": label, "complexity": c, "std": s,
                          "is_ours": False}],
                        columns=["algorithm", "complexity", "std", "is_ours"])


def augment_srbench(root, noises, gaps, r2_thr):
    label = pysr_addon.DISPLAY_LABEL
    for tau in noises:
        print(f"[SRBench] noise {tau:g}")
        pkl = pysr_addon.srbench_pysr_pkl(root, tau)
        if not pkl:
            print("  no results/pysr SRBench artifact at this noise -- skipping")
            continue
        ood = _srbench_ood_rows(pkl, gaps, label)
        if not ood.empty:
            _append_idempotent(
                os.path.join(root, ood_transform.srbench_cache_name(tau)), ood)
        cplx = _srbench_complexity_rows(pkl, tau, r2_thr, label)
        if not cplx.empty:
            _append_idempotent(
                os.path.join(root, ood_transform.srbench_complexity_cache_name(tau)),
                cplx)


# -- LLM-SRBench ---------------------------------------------------------------

def augment_llmsr(root, noises, gaps, split, r2_thr):
    for tau in noises:
        print(f"[LLM-SRBench] noise {tau:g}")
        methods = pysr_addon.llmsr_pysr_methods(root, tau, split)
        if not methods:
            print("  no results/pysr llmsrbench dirs at this noise -- skipping")
            continue
        methods_data = {label: load_method_data(art) for label, art in methods.items()}

        ood = _llmsr_ood_rows(methods_data, gaps)
        if not ood.empty:
            _append_idempotent(
                os.path.join(root, ood_transform.llmsr_cache_name(tau)), ood)

        recs = []
        for label, art in methods.items():
            st = _method_complexity(art, r2_thr)
            if st is not None:
                recs.append({"algorithm": label, "complexity": st[0], "std": st[1],
                             "is_ours": False})
        if recs:
            _append_idempotent(
                os.path.join(root, ood_transform.llmsr_complexity_cache_name(tau)),
                pd.DataFrame(recs, columns=["algorithm", "complexity", "std", "is_ours"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noises", nargs="+", type=float, default=DEFAULT_NOISES)
    ap.add_argument("--gaps", nargs="+", type=int, default=None,
                    help="Gap values. Default: 0 1 2 4 8 16 32 64 128.")
    ap.add_argument("--split", default="lsr_transform")
    ap.add_argument("--bench", choices=["all", "srbench", "llmsr"], default="all",
                    help="Which half to augment. Default: all.")
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    ap.add_argument("--results-root", default=None, metavar="DIR",
                    help="Results tree to read AND write caches in. Default: "
                         "<repo>/results. Point it at a copy to dry-run the append "
                         "without touching the live caches.")
    pysr_addon.add_group_arg(ap)
    args = ap.parse_args()
    pysr_addon.apply_group_arg(args)

    root = args.results_root or os.path.join(SCRIPT_DIR, "results")
    noises = sorted(set(args.noises))
    if args.gaps is None:
        args.gaps = ood_transform.default_gaps()
    gaps = sorted(set(args.gaps))

    if args.bench in ("all", "srbench"):
        augment_srbench(root, noises, gaps, args.r2_thr)
    if args.bench in ("all", "llmsr"):
        augment_llmsr(root, noises, gaps, args.split, args.r2_thr)
    print("Done.")


if __name__ == "__main__":
    main()
