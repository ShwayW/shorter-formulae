#!/usr/bin/env python3
"""
augment_tpsr_caches.py -- compute the OOD R^2 and formula complexity of the two
"<model> + TPSR" combined methods (results/mymodels_tpsr/: m89=89M, m145=145M
decoded with TPSR) and APPEND them to the four plotting caches, so the existing
--reuse-ood / --reuse-cache plot paths pick them up without recomputing everything else.

Caches written (one row-set per noise tau in --noises), each idempotent (rows for these
methods are dropped and rewritten on re-run):

  SRBench:
    results/ood_raw_noise<tau>.csv        (+ raw labels m89_tpsr / m145_tpsr)
    results/complexity_noise<tau>.csv     (+ display labels 89M+TPSR / 145M+TPSR)
  LLM-SRBench:
    results/llmsr_ood_raw_noise<tau>.csv        (+ display labels)
    results/llmsr_complexity_noise<tau>.csv     (+ display labels)

The SRBench OOD cache stores RAW model labels (remapped to display labels at plot time),
so these rows use m89_tpsr / m145_tpsr; every other cache stores display labels.

Usage:
    python augment_tpsr_caches.py                       # all four caches, 4 noises
    python augment_tpsr_caches.py --noises 0            # just noise 0
    python augment_tpsr_caches.py --which srbench       # only the SRBench caches
"""
import argparse
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import tpsr_addon
import ood_transform
from compare_srbench import (
    _ood_load_tf_all, _ood_pn_evaluator, _ood_eval_pn, formula_complexity,
)
from plot_complexity_vs_noise import _load_complexity
from plot_ood_vs_gap import _label_for_path
from plot_llmsrbench_ood_vs_gap import (
    load_method_data, eval_ood_raw_for_gap,
)
from plot_llmsr_time_complexity_vs_noise import _method_complexity

DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]
DEFAULT_GAPS   = [0, 1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_R2_THR = 0.99


def _append_idempotent(cache_path, new_df, key_col="algorithm"):
    """Rewrite cache_path with new_df's rows replacing any existing rows whose
    key_col value is in new_df (leaving all other methods untouched)."""
    if os.path.exists(cache_path):
        old = pd.read_csv(cache_path)
        old = old[~old[key_col].isin(new_df[key_col].unique())]
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_csv(cache_path, index=False)
    print(f"  wrote {os.path.basename(cache_path)}: "
          f"+{len(new_df)} rows for {sorted(new_df[key_col].unique())}")


# -- SRBench -------------------------------------------------------------------

def _srbench_ood_rows(tpsr_pkls, gaps):
    """Per-(gap, raw_label, dataset, seed) OOD R^2 for the tpsr pkls -- the transformer-
    only branch of compare_srbench.compute_ood_r2 (no baselines / e2e / GT), so only
    these two methods are evaluated."""
    pn_eval = _ood_pn_evaluator()
    rows = []
    for g in gaps:
        ood_path = os.path.join(SCRIPT_DIR, ood_transform.feynman_ood_path(g))
        if not os.path.exists(ood_path):
            print(f"  [warn] missing {ood_path}")
            continue
        with gzip.open(ood_path, "rb") as fh:
            ood_datasets = pickle.load(fh)["datasets"]
        for path in tpsr_pkls:
            label = _label_for_path(path)          # m89_tpsr / m145_tpsr
            for ds, seed_fmls in _ood_load_tf_all(path).items():
                if ds not in ood_datasets:
                    continue
                X, y = ood_datasets[ds]["X"], ood_datasets[ds]["y"]
                for seed, formula in seed_fmls:
                    rows.append({"algorithm": label, "dataset": ds,
                                 "ood_r2": _ood_eval_pn(formula, X, y, pn_eval),
                                 "seed": int(seed), "gap": g})
        print(f"  [gap={g:>3}] SRBench OOD done", flush=True)
    return pd.DataFrame(rows, columns=["algorithm", "dataset", "ood_r2", "seed", "gap"])


def _srbench_complexity_rows(root, tau, r2_thr):
    """(display_label, complexity, std, is_ours) for the tpsr pkls -- computed by the
    same _load_complexity path the base 89M/145M use, then relabelled to '+TPSR'."""
    tpsr_pkls = tpsr_addon.srbench_tpsr_pkls(root, tau)
    if not tpsr_pkls:
        return pd.DataFrame(columns=["algorithm", "complexity", "std", "is_ours"])
    cplx, _mine, _common = _load_complexity(tpsr_pkls, None, None, r2_thr, tau)
    recs = []
    for lbl, (c, s) in cplx.items():
        disp = lbl.replace("_tpsr", "+TPSR")          # 89M_tpsr -> 89M+TPSR
        if not disp.endswith("+TPSR"):
            continue                                    # skip stray baseline rows
        recs.append({"algorithm": disp, "complexity": c, "std": s, "is_ours": True})
    return pd.DataFrame(recs, columns=["algorithm", "complexity", "std", "is_ours"])


def augment_srbench(root, noises, gaps, r2_thr):
    for tau in noises:
        print(f"[SRBench] noise {tau:g}")
        tpsr_pkls = tpsr_addon.srbench_tpsr_pkls(root, tau)
        if not tpsr_pkls:
            print("  no mymodels_tpsr pkls -- skipping"); continue
        ood = _srbench_ood_rows(tpsr_pkls, gaps)
        if not ood.empty:
            _append_idempotent(os.path.join(root,
                               ood_transform.srbench_cache_name(tau)), ood)
        cplx = _srbench_complexity_rows(root, tau, r2_thr)
        if not cplx.empty:
            _append_idempotent(os.path.join(root,
                               ood_transform.srbench_complexity_cache_name(tau)),
                               cplx)


# -- LLM-SRBench ---------------------------------------------------------------

def _llmsr_ood_rows(methods_data, gaps):
    frames = []
    for g in gaps:
        ood_path = os.path.join(SCRIPT_DIR, ood_transform.llmsr_ood_path(g))
        if not os.path.exists(ood_path):
            print(f"  [warn] missing {ood_path}"); continue
        frames.extend(eval_ood_raw_for_gap(ood_path, methods_data, gap_override=g))
        print(f"  [gap={g:>3}] LLM-SRBench OOD done", flush=True)
    return pd.DataFrame(frames, columns=["gap", "algorithm", "seed", "equation_id", "ood_r2"])


def augment_llmsr(root, noises, gaps, split, r2_thr):
    for tau in noises:
        print(f"[LLM-SRBench] noise {tau:g}")
        methods = tpsr_addon.llmsr_tpsr_methods(root, tau, split)
        if not methods:
            print("  no mymodels_tpsr dirs -- skipping"); continue
        methods_data = {label: load_method_data(art) for label, art in methods.items()}
        ood = _llmsr_ood_rows(methods_data, gaps)
        if not ood.empty:
            _append_idempotent(os.path.join(root,
                               ood_transform.llmsr_cache_name(tau)), ood)
        recs = []
        for label, art in methods.items():
            st = _method_complexity(art, r2_thr)
            if st is not None:
                recs.append({"algorithm": label, "complexity": st[0], "std": st[1],
                             "is_ours": True})
        if recs:
            _append_idempotent(os.path.join(root,
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
    ap.add_argument("--which", choices=["all", "srbench", "llmsr"], default="all")
    args = ap.parse_args()

    root = os.path.join(SCRIPT_DIR, "results")
    noises = sorted(set(args.noises))
    if args.gaps is None:
        args.gaps = ood_transform.default_gaps()
    gaps = sorted(set(args.gaps))

    if args.which in ("all", "srbench"):
        augment_srbench(root, noises, gaps, args.r2_thr)
    if args.which in ("all", "llmsr"):
        augment_llmsr(root, noises, gaps, args.split, args.r2_thr)
    print("Done.")


if __name__ == "__main__":
    main()
