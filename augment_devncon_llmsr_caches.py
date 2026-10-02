#!/usr/bin/env python3
"""
augment_devncon_llmsr_caches.py -- append the LLM-SRBench OOD R^2 + complexity rows for
the "<model> + D&C" (DEVNCON) arms to the plotting caches.

Why this exists: augment_devncon_caches.py writes the SRBench OOD cache and NOTHING else
(make_requested_plots.sh documents that; its docstring's mention of complexity is
aspirational).  The canonical D&C arms got their LLM-SRBench rows the only other way
available -- a full, non---reuse-ood run of plot_llmsrbench_ood_vs_gap.py, which
RECOMPUTES EVERY METHOD and then does raw.to_csv(cache), i.e. it OVERWRITES the cache
with only the methods that run discovered.  Any arm not rediscovered in that run is
silently wiped, which is the failure [[ood-cache-rebuild-trap]] warns about.

This script computes just the D&C rows and appends them idempotently (the same
_append_idempotent the TPSR augment uses), leaving every other method's rows untouched.

Usage:
    DEVNCON_GROUPS=mymodels_devncon_ftnoise python augment_devncon_llmsr_caches.py
    python augment_devncon_llmsr_caches.py --noises 0 --gaps 0 1 2
"""
import argparse
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import devncon_addon
import ood_transform
from augment_tpsr_caches import (
    _append_idempotent, _llmsr_ood_rows, DEFAULT_NOISES, DEFAULT_GAPS, DEFAULT_R2_THR)
from plot_llmsrbench_ood_vs_gap import load_method_data
from plot_llmsr_time_complexity_vs_noise import _method_complexity


def augment_llmsr(root, noises, gaps, split, r2_thr):
    for tau in noises:
        print(f"[LLM-SRBench] noise {tau:g}")
        methods = devncon_addon.llmsr_devncon_methods(root, tau, split)
        if not methods:
            print("  no devncon dirs -- skipping"); continue
        print(f"  labels: {sorted(methods)}")
        methods_data = {label: load_method_data(art) for label, art in methods.items()}

        ood = _llmsr_ood_rows(methods_data, gaps)
        if not ood.empty:
            _append_idempotent(os.path.join(root, ood_transform.llmsr_cache_name(tau)), ood)

        recs = []
        for label, art in methods.items():
            st = _method_complexity(art, r2_thr)
            if st is not None:
                recs.append({"algorithm": label, "complexity": st[0],
                             "std": st[1], "is_ours": True})
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
    ap.add_argument("--r2-threshold", type=float, default=DEFAULT_R2_THR, dest="r2_thr")
    ap.add_argument("--results-root", default=os.path.join(SCRIPT_DIR, "results"))
    args = ap.parse_args()

    gaps = args.gaps if args.gaps is not None else DEFAULT_GAPS
    print(f"devncon groups: {devncon_addon.DEVNCON_GROUPS}")
    augment_llmsr(args.results_root, args.noises, gaps, args.split, args.r2_thr)
    print("Devncon LLM-SRBench augmentation complete.")


if __name__ == "__main__":
    main()
