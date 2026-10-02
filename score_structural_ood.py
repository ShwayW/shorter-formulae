#!/usr/bin/env python3
"""
score_structural_ood.py -- OOD R^2 on affine-stripped formulae ("structural recovery").

The ordinary OOD score compares a predicted formula against the ground-truth y.
A model can lose that comparison purely because it fitted different affine
prefactors, even when it recovered the right structure.  This script strips the
affine wrappers from BOTH sides with utils.remove_affine and scores the two
skeletons against each other, so what is left is whether the structure matches.

For each (model, dataset, seed, gap):
    y_struct = eval(remove_affine(target_pn),        X_ood)
    y_pred   = eval(remove_affine(predicted_formula), X_ood)
    ood_r2   = R^2(y_struct, y_pred)

Output has the same schema as results/ood_raw_noise<tau>.csv -- algorithm,
dataset, ood_r2, seed, gap -- so the existing plotting code can read it.

Usage:
    python score_structural_ood.py --models 145M_40_simp1
    python score_structural_ood.py --models 145M_40_simp1 abl145M_p0s0 --noise 0.0
"""

import argparse
import glob
import gzip
import os
import pickle

import numpy as np
import pandas as pd

import ood_transform
from compare_srbench import _ood_r2_score, _ood_pn_evaluator, _OOD_N_VARS
from utils import remove_affine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_predictions(path):
    """{dataset: (target_pn, [(seed, predicted_formula), ...])} from one eval pkl."""
    with gzip.open(path, "rb") as fh:
        data = pickle.load(fh)
    by_ds = {}
    for r in data["results"]:
        ds = r["dataset"]
        if ds not in by_ds:
            by_ds[ds] = (r.get("target_pn"), [])
        by_ds[ds][1].append((r["seed"], r.get("predicted_formula")))
    return by_ds


def eval_pn(formula, X, pn_eval):
    """Evaluate a PN formula on X, or None when it does not evaluate everywhere."""
    if not formula or not str(formula).strip():
        return None
    try:
        X64 = X.astype(np.float64)
        if X64.shape[1] < _OOD_N_VARS:                       # pad to the grammar's arity
            pad = np.zeros((X64.shape[0], _OOD_N_VARS - X64.shape[1]), np.float64)
            X64 = np.concatenate([X64, pad], axis=1)
        preds, stacklefts = pn_eval(formula, X64)
        if np.any(stacklefts != 0) or not np.isfinite(preds).all():
            return None
        return preds
    except Exception:
        return None


def score(model_files, gaps, pn_eval):
    """One row per (model, dataset, seed, gap)."""
    rows = []
    for gap in gaps:
        ood_path = os.path.join(SCRIPT_DIR, ood_transform.feynman_ood_path(gap))
        if not os.path.exists(ood_path):
            print(f"  [warn] missing {ood_path}")
            continue
        with gzip.open(ood_path, "rb") as fh:
            ood_datasets = pickle.load(fh)["datasets"]

        for label, paths in model_files.items():
            for path in paths:
                for ds, (target_pn, preds) in load_predictions(path).items():
                    if ds not in ood_datasets or not target_pn:
                        continue
                    X = ood_datasets[ds]["X"]

                    # The stripped target defines the structure to match.  If it
                    # does not evaluate on this box there is nothing to score.
                    y_struct = eval_pn(remove_affine(target_pn), X, pn_eval)
                    if y_struct is None:
                        continue

                    for seed, formula in preds:
                        y_pred = eval_pn(remove_affine(formula or ""), X, pn_eval)
                        r2 = 0.0 if y_pred is None else _ood_r2_score(y_struct, y_pred)
                        rows.append({"algorithm": label, "dataset": ds, "ood_r2": r2,
                                     "seed": int(seed), "gap": gap})
        print(f"  [gap={gap:>3}] done ({len(rows)} rows so far)", flush=True)
    return pd.DataFrame(rows, columns=["algorithm", "dataset", "ood_r2", "seed", "gap"])


def find_eval_files(model, noise, root="results"):
    """Every per-seed eval_mymodels artifact for one model at one noise level."""
    tag = f"noise_{noise:g}" if noise else "noise_0"
    return sorted(glob.glob(os.path.join(root, "*", "seed*", tag, f"eval_tf_{model}.pkl.gz")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="model names as they appear in eval_tf_<model>.pkl.gz")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--gaps", nargs="+", type=int, default=None)
    ap.add_argument("--output", default=None,
                    help="default results/ood_structural_noise<tau>.csv")
    args = ap.parse_args()

    gaps = args.gaps if args.gaps is not None else ood_transform.default_gaps()
    out = args.output or os.path.join(
        SCRIPT_DIR, f"results/ood_structural_noise{args.noise:g}.csv")

    model_files = {}
    for m in args.models:
        files = find_eval_files(m, args.noise)
        if not files:
            print(f"  [warn] no eval artifacts for {m!r} at noise {args.noise:g} -- skipped")
            continue
        model_files[m] = files
        print(f"  {m}: {len(files)} seed files")
    if not model_files:
        raise SystemExit("[error] no models to score")

    pn_eval = _ood_pn_evaluator()
    if pn_eval is None:
        raise SystemExit("[error] PN evaluator unavailable")

    df = score(model_files, gaps, pn_eval)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df)} rows -> {out}")
    for m in df["algorithm"].unique():
        sub = df[df["algorithm"] == m]
        print(f"  {m}: {sub['dataset'].nunique()} datasets, {sub['seed'].nunique()} seeds")


if __name__ == "__main__":
    main()
