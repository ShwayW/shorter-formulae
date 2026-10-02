#!/usr/bin/env python3
"""
score_tpsr_ood.py -- OOD R^2 cache for the standalone TPSR baseline (TPSR decoding on
top of the E2E transformer, Shojaee et al.), from the authors' published SRBench Feynman
results.

This is the *baseline* TPSR -- E2E + MCTS decoding -- not "<our model> + TPSR", which is
a different arm and already sits in the base cache as 145M_tpsr / 89M_tpsr. The base OOD
cache (results/ood_raw_noise0.csv) has no rows for it: compare_srbench.py evaluates it
live from TPSR/srbench_results/*.csv and never persisted it, so nothing that reads the
caches (plot_ood_common.py, plot_ood_full_suite.py) could draw it. This writes it out
once, in the same schema as score_llmsr_ood.py / score_oneshot_ood.py.

TPSR's published run is a SINGLE run per problem (no seeds), so every row is seed 0 and
any plot of it gets a bare line with no variance band. Its formulas are in TPSR's own
prefix notation, evaluated by compare_srbench._ood_eval_tpsr_prefix -- the same evaluator
the in-distribution comparison uses, so the numbers are consistent with that table.

On LLM-SRBench there is no published csv: TPSR is run here by
`eval_e2e_tpsr.py --benchmark llmsrbench`, whose artifact stores an E2E-style infix
formula over x_0..x_N. Point --from-artifact at that results.pkl.gz and the same OOD
clouds are scored through compare_srbench._ood_eval_infix -- the evaluator the e2e and
PhyE2E baselines already use for that formula convention.

Writes: <output-dir>/tpsr_ood_raw_<split>_noise<tau>.csv
columns: gap, algorithm, dataset, seed, ood_r2

Usage:
    python score_tpsr_ood.py                                  # feynman, published csv
    python score_tpsr_ood.py --split lsr_transform \
        --from-artifact results/tpsr/noise_0/llmsrbench/e2e_tpsr_lsr_transform/results.pkl.gz
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

from compare_srbench import (_ood_load_tpsr, _ood_eval_tpsr_prefix,  # noqa: E402
                             _ood_eval_infix)
from results_io import load_results  # noqa: E402
import ood_transform  # noqa: E402

DEFAULT_TPSR_CSV = "TPSR/srbench_results/feynman_tpsr_l0.1_allnoise.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tpsr-csv", default=DEFAULT_TPSR_CSV,
                    help="Published SRBench Feynman results (prefix notation).")
    ap.add_argument("--from-artifact", default=None,
                    help="Score an eval_e2e_tpsr.py --benchmark llmsrbench artifact "
                         "(results.pkl.gz, infix x_0 formulas) instead of --tpsr-csv.")
    ap.add_argument("--split", default="feynman",
                    help="Which OOD clouds to score against (feynman / lsr_transform).")
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0])
    ap.add_argument("--gaps", nargs="+", type=int, default=ood_transform.default_gaps())
    ap.add_argument("--algorithm", default="TPSR",
                    help="Label written into the CSV (matches compare_srbench's).")
    ap.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "results"))
    args = ap.parse_args()

    if args.from_artifact is None:
        csv_path = os.path.join(SCRIPT_DIR, args.tpsr_csv)
        if not os.path.exists(csv_path):
            raise SystemExit(f"TPSR results csv not found: {csv_path}")
    else:
        csv_path = None

    for tau in args.noise:
        if csv_path is not None:
            # Published SRBench run: {dataset: row} keyed by problem, prefix formulas.
            preds = {k: (v["predicted_tree_prefix"], 0) for k, v in
                     _ood_load_tpsr(csv_path, noise=float(tau)).items()}
            evaluate = lambda f, X, y: _ood_eval_tpsr_prefix(f, X, y)
        else:
            # Our own llmsrbench run: infix x_0.. formulas, one row per (problem, seed).
            path = args.from_artifact
            path = path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)
            preds = {}
            for r in load_results(path)[1]:
                f = r.get("discovered_equation")
                if f:
                    preds[r["equation_id"]] = (f, int(r.get("seed", 0) or 0))
            evaluate = lambda f, X, y: _ood_eval_infix(f, "x_0", X, y)
        if not preds:
            print(f"[tau={tau:g}] no TPSR predictions at this noise -- skipped")
            continue
        rows = []
        for g in args.gaps:
            path = os.path.join(SCRIPT_DIR, "datasets",
                                ood_transform.ood_basename(args.split, g))
            if not os.path.exists(path):
                print(f"[tau={tau:g}] [gap={g:>3}] missing OOD dataset -- skipped")
                continue
            with gzip.open(path) as fh:
                ood = pickle.load(fh)["datasets"]
            n = 0
            for ds, (formula, seed) in preds.items():
                d = ood.get(ds)
                if d is None:
                    continue
                r2 = evaluate(formula, np.asarray(d["X"], dtype=np.float64),
                              np.asarray(d["y"], dtype=np.float64))
                rows.append({"gap": g, "algorithm": args.algorithm, "dataset": ds,
                             "seed": seed, "ood_r2": r2})
                n += 1
            print(f"[tau={tau:g}] [gap={g:>3}] scored {n}")
        out = os.path.join(args.output_dir,
                           f"tpsr_ood_raw_{args.split}_noise{float(tau):g}.csv")
        pd.DataFrame(rows, columns=["gap", "algorithm", "dataset", "seed", "ood_r2"]).to_csv(
            out, index=False)
        print(f"[tau={tau:g}] wrote {len(rows)} rows -> {out}\n")


if __name__ == "__main__":
    main()
