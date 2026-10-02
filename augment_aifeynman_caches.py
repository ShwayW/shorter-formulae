#!/usr/bin/env python3
"""
augment_aifeynman_caches.py -- the AI Feynman twin of augment_tpsr_caches.py /
augment_tpsr_caches.py.  Compute the OOD R^2 and formula complexity of the AI Feynman
2.0 baseline (results/aifeynman/, produced by eval_aifeynman.py --benchmark llmsrbench) and APPEND
them to the LLM-SRBench plotting caches, so the existing --reuse-ood / --reuse-cache
plot paths pick the method up without recomputing everything else.

Caches written (one row-set per noise tau in --noises), each idempotent (rows for this
method are dropped and rewritten on re-run):

    results/llmsr_ood_raw_noise<tau>.csv        (+ display label "AIFeynman")
    results/llmsr_complexity_noise<tau>.csv     (+ display label "AIFeynman")

LLM-SRBench ONLY -- unlike the TPSR/AIF2 twins there is no SRBench half here.  The
SRBench (Feynman) figures already carry an "AIFeynman" curve, taken from SRBench's own
published results JSON via compare_srbench.py; LLM-SRBench is the side that had no
published numbers, which is the whole reason we run it ourselves.  Writing SRBench rows
from this tree would silently replace published numbers with ours.

Why this exists at all: sets 5/6 (plot_llmsrbench_curves_best_seed.py,
plot_llmsrbench_time_complexity_best_seed.py) read the BASE caches rather than
discovering method dirs, so a method outside results_io.GROUPS is invisible to them
until its rows are in those caches.  Set 3 (plot_llmsrbench_ood_vs_gap.py) can also
discover it live via --include-aifeynman, but reads the cache when --reuse-ood is set --
so run this once after a run lands, exactly as make_requested_plots.sh does for AIF2.

A noise level with no AI Feynman artifact (e.g. only noise 0 has been run -- the
default, since the method costs ~1-2 h/problem) is skipped, so this is safe to run on a
partial sweep: only the noises that exist get rows.

Usage:
    python augment_aifeynman_caches.py                  # all 4 noises (skips missing)
    python augment_aifeynman_caches.py --noises 0       # just the finished noise
"""
import argparse
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import aifeynman_addon
import ood_transform
from augment_tpsr_caches import _append_idempotent, _llmsr_ood_rows
from plot_llmsrbench_ood_vs_gap import load_method_data
from plot_llmsr_time_complexity_vs_noise import _method_complexity

DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]
DEFAULT_R2_THR = 0.99


def augment_llmsr(root, noises, gaps, split, r2_thr):
    for tau in noises:
        print(f"[LLM-SRBench] noise {tau:g}")
        methods = aifeynman_addon.llmsr_aifeynman_methods(root, tau, split)
        if not methods:
            print("  no results/aifeynman dirs at this noise -- skipping")
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
                # is_ours=False: AI Feynman is an external baseline, not one of our
                # models -- the complexity figures style the two groups differently.
                recs.append({"algorithm": label, "complexity": st[0], "std": st[1],
                             "is_ours": False})
        if recs:
            _append_idempotent(
                os.path.join(root,
                             ood_transform.llmsr_complexity_cache_name(tau)),
                pd.DataFrame(recs, columns=["algorithm", "complexity", "std", "is_ours"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noises", nargs="+", type=float, default=DEFAULT_NOISES)
    ap.add_argument("--gaps", nargs="+", type=int, default=None,
                    help="Gap values. Default: 0 1 2 4 8 16 32 64 128.")
    ap.add_argument("--split", default="lsr_transform")
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    aifeynman_addon.add_group_arg(ap)
    args = ap.parse_args()
    aifeynman_addon.apply_group_arg(args)

    root = os.path.join(SCRIPT_DIR, "results")
    noises = sorted(set(args.noises))
    if args.gaps is None:
        args.gaps = ood_transform.default_gaps()
    gaps = sorted(set(args.gaps))

    augment_llmsr(root, noises, gaps, args.split, args.r2_thr)
    print("Done.")


if __name__ == "__main__":
    main()
