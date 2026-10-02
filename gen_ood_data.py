#!/usr/bin/env python3
"""
gen_ood_data.py -- Pre-generate OOD evaluation datasets for the Feynman benchmark.

For each of the ~100 Feynman equations that has a known ground-truth formula in
FeynmanEquations.csv, this script:
  1. Loads the corresponding PMLB dataset to determine per-variable ranges.
  2. Builds an OOD region at the requested gap (see ood_domain.py for the two
     methods and why a plain all-positive shift is not enough on its own).
  3. Evaluates the ground-truth formula there (via sympy, see gt_eval.py) and
     retains up to n_points valid (finite) output points.

Black-box datasets (not present in FeynmanEquations.csv) are automatically skipped.

The output is a single .pkl.gz file with the following structure:
    {
        "gap":      float,
        "n_points": int,
        "seed":     int,
        "method":   str,
        "regions":  {"<dataset_name>": {"kappa": [...], "sign": [...]}},
        "datasets": {
            "<dataset_name>": {"X": np.ndarray (N, D), "y": np.ndarray (N,)},
            ...
        },
    }

Usage
-----
python gen_ood_data.py
python gen_ood_data.py --gap 2.0 --n-points 5000 --output ./datasets/feynman_ood_g2.pkl.gz
python gen_ood_data.py --gap 128 --method freeze
python gen_ood_data.py --gap 128 --only feynman_I_26_2 --merge-into ./datasets/feynman_ood_g128.pkl.gz
"""

import os
import gzip
import csv
import pickle
import argparse

import numpy as np

import gt_eval
import ood_domain
import ood_transform
from feynman_pn_analysis import formula_to_pn as _feynman_to_pn


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(tsv_gz_path: str):
    """Load a PMLB .tsv.gz file; returns (X, y) as float64 arrays."""
    with gzip.open(tsv_gz_path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header row
        rows = list(reader)
    data = np.array(rows, dtype=np.float64)
    return data[:, :-1], data[:, -1]


# ---------------------------------------------------------------------------
# Feynman GT formula loading
# ---------------------------------------------------------------------------
def load_feynman_pn_targets(
    feynman_csv: str = "./datasets/feynman/FeynmanEquations.csv",
) -> dict:
    """Return {dataset_name: pn_str} for each row in FeynmanEquations.csv.

    Rows whose formula cannot be converted to canonical prefix notation are
    omitted.  Black-box datasets (absent from the CSV) never appear here.
    """
    pn_targets: dict = {}
    if not os.path.isfile(feynman_csv):
        raise FileNotFoundError(f"Feynman CSV not found: {feynman_csv}")
    with open(feynman_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("Filename", "").strip()
            fml   = row.get("Formula",  "").strip()
            if not fname or not fml:
                continue
            var_names = [row.get(f"v{k}_name", "").strip()
                         for k in range(1, 11)
                         if row.get(f"v{k}_name", "").strip()]
            if not var_names:
                continue
            pn, _status = _feynman_to_pn(fml, var_names)
            if pn and "ERR" not in pn:
                dataset_name = "feynman_" + fname.replace(".", "_")
                pn_targets[dataset_name] = pn
    return pn_targets


def load_feynman_formulas(
    feynman_csv: str = "./datasets/feynman/FeynmanEquations.csv",
) -> dict:
    """Return {dataset_name: (formula_str, variable_names)} from the CSV.

    The formula is used verbatim for evaluation, rather than being converted to
    prefix notation first -- see gt_eval.py for why that conversion corrupted the
    ground truth of every pi-containing formula.

    variable_names are in v1..v10 order, which matches the PMLB input column order
    for all 99 datasets (checked); gt_eval.build_evaluator relies on that.

    Rows that sympy cannot parse are omitted.  That selects the same 101 formulas
    as the prefix-notation filter in load_feynman_pn_targets (checked), so the
    dataset universe is unchanged.
    """
    formulas: dict = {}
    if not os.path.isfile(feynman_csv):
        raise FileNotFoundError(f"Feynman CSV not found: {feynman_csv}")
    with open(feynman_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("Filename", "").strip()
            fml   = row.get("Formula",  "").strip()
            if not fname or not fml:
                continue
            var_names = [row.get(f"v{k}_name", "").strip()
                         for k in range(1, 11)
                         if row.get(f"v{k}_name", "").strip()]
            if not var_names:
                continue
            if gt_eval.build_evaluator(fml, var_names, verbose=False) is None:
                continue
            formulas["feynman_" + fname.replace(".", "_")] = (fml, var_names)
    return formulas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate OOD evaluation datasets for the Feynman benchmark."
    )
    parser.add_argument(
        "--datasets",
        default="./datasets/pmlb/datasets",
        help="Root directory containing one sub-folder per PMLB dataset.",
    )
    parser.add_argument(
        "--feynman-csv",
        default="./datasets/feynman/FeynmanEquations.csv",
        help="Path to FeynmanEquations.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pkl.gz path.  Default: datasets/feynman_ood_g<gap>.pkl.gz",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=1.0,
        metavar="G",
        help="Shift multiplier: variables are moved G * W beyond the training "
             "range.  G=1 means the OOD region starts one full range-width "
             "beyond the training box.  Default: 1.0.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=10000,
        metavar="N",
        help="Target number of valid OOD points per dataset.  Default: 10000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for input generation.  Default: 0.",
    )
    parser.add_argument(
        "--method",
        choices=ood_domain.METHODS,
        default="legacy",
        help="How to place the OOD region: legacy reproduces the published "
             "datasets and drops feynman_I_26_2 at gap >= 4; freeze keeps all 99.  "
             "Default: legacy.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=ood_domain.DEFAULT_BUDGET,
        help="Max candidate draws per dataset before giving up.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="freeze only: move every variable in the +W direction, as legacy does, "
             "instead of letting a variable flip when it cannot advance.  Loses no "
             "dataset on this benchmark.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Regenerate only these datasets (use with --merge-into).",
    )
    parser.add_argument(
        "--merge-into",
        default=None,
        metavar="PKL",
        help="Start from this existing file and overwrite only the datasets "
             "generated in this run, leaving the rest byte identical.  Use to add "
             "datasets without disturbing results already cached for the others.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = "./" + ood_transform.feynman_ood_path(args.gap)

    rng = np.random.default_rng(args.seed)

    print(f"Loading Feynman GT formulas from {args.feynman_csv} ...", flush=True)
    formulas = load_feynman_formulas(args.feynman_csv)
    print(f"  {len(formulas)} datasets with known ground-truth formula.", flush=True)

    names = sorted(formulas.keys())
    if args.only:
        wanted = set(args.only)
        missing = wanted - set(names)
        if missing:
            raise SystemExit(f"--only named unknown datasets: {sorted(missing)}")
        names = [n for n in names if n in wanted]
        print(f"  restricted to {len(names)} dataset(s).", flush=True)

    ood_dict, regions = {}, {}
    if args.merge_into:
        with gzip.open(args.merge_into, "rb") as fh:
            existing = pickle.load(fh)
        ood_dict = dict(existing.get("datasets", {}))
        regions = dict(existing.get("regions", {}))
        print(f"  merging into {args.merge_into} ({len(ood_dict)} existing).", flush=True)

    n_skipped = 0
    for i, name in enumerate(names):
        dataset_dir = os.path.join(args.datasets, name)

        if not os.path.isdir(dataset_dir):
            print(f"[{i+1:3d}/{len(names)}] {name}: dataset directory not found -- skipping.",
                  flush=True)
            n_skipped += 1
            continue

        tsv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".tsv.gz")]
        if not tsv_files:
            print(f"[{i+1:3d}/{len(names)}] {name}: no .tsv.gz file -- skipping.", flush=True)
            n_skipped += 1
            continue

        X_ref, _ = load_dataset(os.path.join(dataset_dir, tsv_files[0]))
        formula, var_names = formulas[name]
        if len(var_names) != X_ref.shape[1]:
            print(f"[{i+1:3d}/{len(names)}] {name}: {len(var_names)} CSV variables vs "
                  f"{X_ref.shape[1]} data columns -- skipping.", flush=True)
            n_skipped += 1
            continue

        evaluate = gt_eval.build_evaluator(formula, var_names)
        vmin, vmax, widths = ood_domain.training_ranges(X_ref)

        X_ood, y_ood, kappa, sign = ood_domain.build_region(
            args.method, evaluate, vmin, vmax, widths, args.gap, args.n_points,
            rng, gaps=ood_transform.GAPS, seed=args.seed, budget=args.budget,
            positive_only=args.positive_only)

        if len(X_ood) == 0:
            print(f"[{i+1:3d}/{len(names)}] {name}: no valid OOD points -- skipping.",
                  flush=True)
            n_skipped += 1
            continue

        ood_dict[name] = {"X": X_ood, "y": y_ood}
        regions[name] = {"kappa": None if kappa is None else kappa.tolist(),
                         "sign":  None if sign  is None else sign.tolist()}
        print(f"[{i+1:3d}/{len(names)}] {name}: {len(X_ood)} points  "
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
    with gzip.open(args.output, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved {len(ood_dict)} datasets to {args.output}  "
          f"({n_skipped} skipped).", flush=True)


if __name__ == "__main__":
    main()
