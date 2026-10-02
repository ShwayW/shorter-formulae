#!/usr/bin/env python3
"""
gen_ood_lsr_synth.py -- D^k_test point clouds for the 129 LSR-Synth problems.

DIAGNOSTIC ONLY: the box construction leaves LSR-Synth's one-dimensional input path
even at k = 0; see the docstring of plot_lsr_synth.py.

The LSR-Synth twin of gen_ood_llmsrbench.py (LSR-Transform), built the same way:
the training range of each problem is the benchmark's `train` split, the region is
placed by ood_domain.build_region with the "freeze" method, 10,000 points per
problem, seed 0 -- the settings datasets/llmsrbench_ood_g*.pkl.gz were made with.

The only difference is the ground truth: LSR-Synth expressions do not parse as
shipped, so each is recovered by lsr_synth_gt.recover_problem (function notation
removed, free constants fitted to `train` and checked against the benchmark's own
ood_test split).  All four domains go into ONE file per gap; problem names (BPG*,
CRK*, MatSci*, PO*) do not collide.

    python gen_ood_lsr_synth.py --gap 8
    for g in 0 1 2 4 8 16 32 64 128; do python gen_ood_lsr_synth.py --gap $g & done
"""
import argparse
import gzip
import os
import pickle
import warnings

import numpy as np

import gt_eval
import lsr_synth_gt
import ood_domain
import ood_transform

warnings.filterwarnings("ignore")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def lsr_synth_ood_path(g):
    return f"datasets/lsr_synth_ood_g{ood_transform._fmt(g)}.pkl.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gap", type=float, required=True)
    ap.add_argument("--n-points", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=ood_domain.METHODS, default="freeze")
    ap.add_argument("--budget", type=int, default=ood_domain.DEFAULT_BUDGET)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    out = args.output or os.path.join(SCRIPT_DIR, lsr_synth_ood_path(args.gap))

    from eval_mymodels import load_problems
    rng = np.random.default_rng(args.seed)
    datasets, regions, gts, skipped = {}, {}, {}, []
    for dom in lsr_synth_gt.DOMAINS:
        for p in load_problems(f"lsr_synth_{dom}"):
            name = p["name"]
            expr, ins, info = lsr_synth_gt.recover_problem(p)
            evaluate = gt_eval.build_evaluator(expr, ins)
            if evaluate is None:
                skipped.append((name, "parse")); continue
            X_ref = p["train"][:, 1:].astype(np.float64)
            vmin, vmax, widths = ood_domain.training_ranges(X_ref)
            X, y, kappa, sign = ood_domain.build_region(
                args.method, evaluate, vmin, vmax, widths, args.gap, args.n_points,
                rng, gaps=ood_transform.GAPS, seed=args.seed, budget=args.budget)
            if len(X) == 0:
                skipped.append((name, "no valid points")); continue
            datasets[name] = {"X": X, "y": y}
            regions[name] = {"kappa": None if kappa is None else kappa.tolist(),
                             "sign": None if sign is None else sign.tolist()}
            gts[name] = {"expression": expr, "inputs": ins, "domain": dom, **info}
            print(f"{name:>9} ({dom}): {len(X)} pts  "
                  f"{ood_domain.describe(args.method, kappa, sign)}", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with gzip.open(out, "wb") as fh:
        pickle.dump({"gap": args.gap, "n_points": args.n_points, "seed": args.seed,
                     "method": args.method, "regions": regions, "datasets": datasets,
                     "ground_truth": gts}, fh)
    print(f"\nSaved {len(datasets)} datasets -> {out}  ({len(skipped)} skipped: {skipped})")


if __name__ == "__main__":
    main()
