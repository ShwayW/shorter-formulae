#!/usr/bin/env python3
"""
gen_ood_llmsrbench.py -- Generate OOD evaluation datasets for LLM-SRBench (lsr_transform split).

The OOD region is built by ood_domain.py; see that module for the two methods and
why the plain all-positive shift is not enough on its own.

Output format is identical to gen_ood_data.py (Feynman):
    {
        "gap":      float,
        "n_points": int,
        "seed":     int,
        "method":   str,
        "regions":  {"<problem_name>": {"kappa": [...], "sign": [...]}},
        "datasets": {
            "<problem_name>": {"X": np.ndarray (N, D), "y": np.ndarray (N,)},
            ...
        },
    }

Usage:
    python gen_ood_llmsrbench.py --gap 1 --output datasets/llmsrbench_ood_g1.pkl.gz
    python gen_ood_llmsrbench.py --gap 128 --method freeze
    python gen_ood_llmsrbench.py --gap 128 --only I.24.6_1_1 --merge-into datasets/llmsrbench_ood_g128.pkl.gz
"""

import os
import gzip
import pickle
import argparse
import warnings

import numpy as np

import gt_eval
import ood_domain
import ood_transform

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# -- Per-problem setup ----------------------------------------------------------

def problem_inputs(p):
    """Reference inputs and input symbol names for one LLM-SRBench problem.

    Returns (X_ref, input_symbols) or (None, reason) when the problem is unusable.
    """
    train = p.get("train")
    if train is None or train.shape[0] == 0:
        return None, "no train data"

    # col 0 = y, cols 1: = X  (LLM-SRBench convention)
    X_ref = train[:, 1:].astype(np.float64)
    syms = p.get("symbols", [])

    # symbols[0] is the output variable (symbol_properties[0]=='O');
    # the remaining symbols are the inputs, matching X_ref columns.
    input_symbols = syms[1:] if len(syms) > 1 else syms
    if X_ref.shape[1] != len(input_symbols):
        input_symbols = syms          # rare edge case: treat all symbols as inputs
    if X_ref.shape[1] != len(input_symbols):
        return None, f"dim mismatch ({X_ref.shape[1]} cols vs {len(syms)} symbols)"
    if not p.get("expression", ""):
        return None, "no GT expression"
    return X_ref, input_symbols


# -- Main -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate OOD datasets for LLM-SRBench (lsr_transform split)."
    )
    ap.add_argument("--gap", type=float, default=1.0,
                    help="Shift multiplier: gap * per-variable range. Default: 1.0")
    ap.add_argument("--n-points", type=int, default=10000,
                    help="Target OOD points per problem. Default: 10000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="lsr_transform",
                    help="LLM-SRBench split to use. Default: lsr_transform")
    ap.add_argument("--method", choices=ood_domain.METHODS, default="legacy",
                    help="How to place the OOD region: legacy reproduces the "
                         "published datasets and drops 13 problems at gap >= 4; "
                         "freeze keeps all 111. Default: legacy")
    ap.add_argument("--budget", type=int, default=ood_domain.DEFAULT_BUDGET,
                    help="Max candidate draws per problem before giving up.")
    ap.add_argument("--positive-only", action="store_true",
                    help="freeze only: move every variable in the +W direction, as "
                         "legacy does, instead of letting a variable flip when it "
                         "cannot advance.  Loses no problem, but 3 variables in 3 "
                         "problems then never leave their training range.")
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME",
                    help="Regenerate only these problems (use with --merge-into).")
    ap.add_argument("--merge-into", default=None, metavar="PKL",
                    help="Start from this existing file and overwrite only the "
                         "problems generated in this run, leaving the rest byte "
                         "identical.  Use to add problems without disturbing "
                         "results already cached for the others.")
    ap.add_argument("--output", default=None,
                    help="Output .pkl.gz path. Default: datasets/llmsrbench_ood_g<gap>.pkl.gz")
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join(SCRIPT_DIR, ood_transform.llmsr_ood_path(args.gap))

    rng = np.random.default_rng(args.seed)

    print(f"Loading LLM-SRBench '{args.split}' problems ...", flush=True)
    from eval_mymodels import load_problems
    problems = load_problems(args.split)
    print(f"  {len(problems)} problems loaded.", flush=True)

    if args.only:
        wanted = set(args.only)
        problems = [p for p in problems if p["name"] in wanted]
        missing = wanted - {p["name"] for p in problems}
        if missing:
            raise SystemExit(f"--only named unknown problems: {sorted(missing)}")
        print(f"  restricted to {len(problems)} problem(s).", flush=True)

    ood_dict, regions = {}, {}
    if args.merge_into:
        with gzip.open(args.merge_into, "rb") as fh:
            existing = pickle.load(fh)
        ood_dict = dict(existing.get("datasets", {}))
        regions = dict(existing.get("regions", {}))
        print(f"  merging into {args.merge_into} ({len(ood_dict)} existing).", flush=True)

    n_skipped = 0
    for i, p in enumerate(problems):
        name = p["name"]
        X_ref, info = problem_inputs(p)
        if X_ref is None:
            print(f"[{i+1:3d}/{len(problems)}] {name}: {info} -- skipping.")
            n_skipped += 1
            continue
        input_symbols = info

        evaluate = gt_eval.build_evaluator(p["expression"], input_symbols)
        if evaluate is None:
            print(f"[{i+1:3d}/{len(problems)}] {name}: expression parse failed -- skipping.")
            n_skipped += 1
            continue

        vmin, vmax, widths = ood_domain.training_ranges(X_ref)

        X_ood, y_ood, kappa, sign = ood_domain.build_region(
            args.method, evaluate, vmin, vmax, widths, args.gap, args.n_points,
            rng, gaps=ood_transform.GAPS, seed=args.seed, budget=args.budget,
            positive_only=args.positive_only)

        if len(X_ood) == 0:
            print(f"[{i+1:3d}/{len(problems)}] {name}: no valid points -- skipping.")
            n_skipped += 1
            continue

        ood_dict[name] = {"X": X_ood, "y": y_ood}
        regions[name] = {"kappa": None if kappa is None else kappa.tolist(),
                         "sign":  None if sign  is None else sign.tolist()}
        print(f"[{i+1:3d}/{len(problems)}] {name}: {len(X_ood)} pts  "
              f"(gap={args.gap}, {ood_domain.describe(args.method, kappa, sign)})",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    payload = {
        "gap":      args.gap,
        "n_points": args.n_points,
        "seed":     args.seed,
        "method":   args.method,
        "regions":  regions,
        "datasets": ood_dict,
    }
    with gzip.open(args.output, "wb") as fh:
        pickle.dump(payload, fh)

    print(f"\nSaved {len(ood_dict)} datasets -> {args.output}  ({n_skipped} skipped).",
          flush=True)


if __name__ == "__main__":
    main()
