#!/usr/bin/env python3
"""
augment_e2e_tpsr_caches.py -- the standalone-TPSR twin of augment_phye2e_caches.py /
augment_tpsr_caches.py.  Compute the OOD R^2 and formula complexity of the authors' TPSR
run on LLM-SRBench and append them to the plotting caches, so the existing --reuse-ood /
--reuse-cache paths draw the curve without recomputing everything.

WHY THIS IS NEEDED (and not optional)
-------------------------------------
`plot_llmsrbench_ood_vs_gap.py --include-e2e-tpsr` computes the arm live, so a fresh plot
is fine.  But with --reuse-ood the plot scripts read the cached table and only recompute
GAPS that are missing -- never ALGORITHMS that are missing.  Without this script a
--reuse-ood figure silently drops the curve instead of failing, which is the worst kind of
wrong.  Run it once after a run lands, exactly as make_requested_plots.sh does for the
TPSR-combo / D&C / AI Feynman / PhyE2E curves.

THREE cache families, not four:

    results/llmsr_ood_raw_noise<tau>.csv        LLM-SRBench OOD R^2 (per gap/eq/seed)
    results/llmsr_complexity_noise<tau>.csv     LLM-SRBench complexity
    results/ood_raw_noise<tau>.csv              SRBench (Feynman) OOD R^2

The SRBench half is a DIFFERENT run of the same method: the authors' own published
Feynman csv (TPSR/srbench_results/feynman_tpsr_l0.1_allnoise.csv), which is what the
SRBench figures have always meant by "TPSR".  plot_ood_vs_gap.py computes it live and
appends the label itself -- but only on the non---reuse-ood path, and every figure in
make_requested_plots.sh passes --reuse-ood, so the curve was silently absent from set 1
for exactly the reason this script exists.  Appending it here fixes that.  It is ONE run
per problem (no seeds), so its rows are all seed 0 and its band collapses to the line --
unlike the LLM-SRBench half, which is 3 seeds and gets a real +/-1 std band.

The SRBench COMPLEXITY cache already carries a "TPSR" row (compare_srbench computes it
from the same csv), so set 2 needs nothing from here.

FORMULA FORMAT
--------------
The llmsrbench branch emits infix in x_0..x_N (the e2e convention), NOT our prefix PN, so
the OOD evaluation goes through plot_llmsrbench_ood_vs_gap's _is_infix -> numpy path --
the same one e2e and PhyE2E take.  Getting this wrong would not crash; it would quietly
score every formula as NaN.

SEEDS
-----
The run is 3 seeds (42/43/44) x 4 noises, seed-merged by merge_seed_results.py into one
artifact per noise.  eval_ood_raw_for_gap keys its rows by (algorithm, seed, equation_id),
so all three seeds survive into the cache -- which is what gives the curve its +/-1 std
band in set 3 and lets sets 5/6 pick a best seed.  Running this against an UNMERGED tree
(results/tpsr/seed42/...) would cache a single seed and silently flatten the band.

Idempotent: rerunning replaces this method's rows rather than duplicating them.

USAGE
    python augment_e2e_tpsr_caches.py                 # both halves, all 4 noises
    python augment_e2e_tpsr_caches.py --noises 0      # just one noise
    python augment_e2e_tpsr_caches.py --bench llmsr   # one half only
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
import e2e_tpsr_addon
from augment_tpsr_caches import _append_idempotent, _llmsr_ood_rows
from compare_srbench import _ood_load_tpsr, _ood_eval_tpsr_prefix
from plot_llmsrbench_ood_vs_gap import load_method_data
from plot_llmsr_time_complexity_vs_noise import _method_complexity

DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]
DEFAULT_R2_THR = 0.99
DEFAULT_TPSR_CSV = "TPSR/srbench_results/feynman_tpsr_l0.1_allnoise.csv"


# -- SRBench (the authors' published Feynman run) ------------------------------

def _srbench_ood_rows(csv_path, tau, gaps, label):
    """Per-(gap, algorithm, dataset, seed) OOD R^2 for the published TPSR csv at noise tau.

    The TPSR branch of compare_srbench.compute_ood_r2, with no baselines / e2e / GT, so
    only this one method is evaluated.  Formulas are in TPSR's OWN prefix notation
    ('add', 'mul', 'inv', 'pow2', ...), so they go through _ood_eval_tpsr_prefix -- the
    same evaluator the in-distribution comparison table uses, which keeps these numbers
    consistent with it.  seed 0 throughout: the published run has no seed axis.
    """
    preds = _ood_load_tpsr(csv_path, noise=float(tau))
    if not preds:
        print(f"  no TPSR rows at target_noise={tau:g} in {os.path.basename(csv_path)}")
        return pd.DataFrame(columns=["algorithm", "dataset", "ood_r2", "seed", "gap"])
    rows = []
    for g in gaps:
        ood_path = os.path.join(SCRIPT_DIR, ood_transform.feynman_ood_path(g))
        if not os.path.exists(ood_path):
            print(f"  [warn] missing {ood_path}")
            continue
        with gzip.open(ood_path, "rb") as fh:
            ood_datasets = pickle.load(fh)["datasets"]
        n = 0
        for ds, row in preds.items():
            if ds not in ood_datasets:
                continue
            X, y = ood_datasets[ds]["X"], ood_datasets[ds]["y"]
            rows.append({"algorithm": label, "dataset": ds,
                         "ood_r2": _ood_eval_tpsr_prefix(row["predicted_tree_prefix"],
                                                         X, y),
                         "seed": 0, "gap": g})
            n += 1
        print(f"  [gap={g:>3}] SRBench OOD done ({n} problems)", flush=True)
    return pd.DataFrame(rows, columns=["algorithm", "dataset", "ood_r2", "seed", "gap"])


def augment_srbench(root, noises, gaps, csv_path):
    label = e2e_tpsr_addon.LABEL
    if not os.path.exists(csv_path):
        print(f"[SRBench] published TPSR csv not found ({csv_path}) -- skipping")
        return
    for tau in noises:
        print(f"[SRBench] noise {tau:g}")
        ood = _srbench_ood_rows(csv_path, tau, gaps, label)
        if not ood.empty:
            _append_idempotent(
                os.path.join(root, ood_transform.srbench_cache_name(tau)), ood)


def augment_llmsr(root, noises, gaps, split, r2_thr):
    for tau in noises:
        print(f"[LLM-SRBench] noise {tau:g}")
        methods = e2e_tpsr_addon.llmsr_methods(root, tau, split)
        if not methods:
            print("  no results/tpsr llmsrbench dir at this noise -- skipping")
            continue
        methods_data = {label: load_method_data(art) for label, art in methods.items()}
        n_seeds = {len(sm) for sm in methods_data.values()}
        print(f"  {sorted(methods)} -- {n_seeds} seed(s) "
              f"({sum(len(p) for sm in methods_data.values() for p in sm.values())} rows)")

        ood = _llmsr_ood_rows(methods_data, gaps)
        if not ood.empty:
            _append_idempotent(
                os.path.join(root, ood_transform.llmsr_cache_name(tau)), ood)

        recs = []
        for label, art in methods.items():
            st = _method_complexity(art, r2_thr)
            if st is not None:
                # is_ours=False: the authors' TPSR over the authors' E2E transformer is
                # an external published method, not one of our checkpoints -- the
                # complexity figures style the two groups differently.
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
    ap.add_argument("--tpsr-csv", default=DEFAULT_TPSR_CSV,
                    help="The authors' published SRBench Feynman results (prefix "
                         "notation) -- the SRBench half's source.")
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    ap.add_argument("--results-root", default=None, metavar="DIR",
                    help="Results tree to read AND write caches in. Default: "
                         "<repo>/results. Point it at a copy to dry-run the append "
                         "without touching the live caches.")
    args = ap.parse_args()

    root = args.results_root or os.path.join(SCRIPT_DIR, "results")
    noises = sorted(set(args.noises))
    if args.gaps is None:
        args.gaps = ood_transform.default_gaps()
    gaps = sorted(set(args.gaps))

    if args.bench in ("all", "srbench"):
        augment_srbench(root, noises, gaps,
                        os.path.join(SCRIPT_DIR, args.tpsr_csv))
    if args.bench in ("all", "llmsr"):
        augment_llmsr(root, noises, gaps, args.split, args.r2_thr)
    print("Done.")


if __name__ == "__main__":
    main()
