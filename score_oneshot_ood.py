#!/usr/bin/env python3
"""
score_oneshot_ood.py -- compute OOD R^2 for a one-shot run's discovered formulas, across
gaps / noise levels / benchmarks, and cache one CSV per (split, noise) in the schema the
comparison plots read.

Unlike score_llmsr_ood.py, nothing is re-fitted here. An LLMSR artifact stores a program
with symbolic params[0..9] and no values, so scoring it OOD first requires re-running its
BFGS fit on TRAIN. A one-shot artifact already stores a *concrete* formula: in direct mode
the model wrote the constants, and in skeleton mode BFGS fitted them on TRAIN at search
time (eval_oneshot.py). Re-fitting here would be scoring a different equation than the one
the method produced -- and, worse, silently repairing a structurally wrong formula whose
inability to extrapolate is exactly what the OOD axis is supposed to expose.

So the pipeline is: parse the stored expression, evaluate it on each
datasets/<bench>_ood_g<gap>.pkl.gz point cloud, record R^2. The stored TEST R^2 is
recomputed on the way through and reported as a consistency check -- it must reproduce
exactly (same formula, same data), so any mismatch means the expression or the column
order is being read wrong, not that the method changed.

Writes, for every (split, noise):
    <output-dir>/oneshot_<backbone>_ood_raw_<split>_noise<tau>.csv
columns: gap, algorithm, dataset, seed, ood_r2   (same as score_llmsr_ood.py)

Usage:
    python score_oneshot_ood.py --results-root results/oneshot \
        --method oneshot-fit-llama31-8b --algorithm oneshot-llama --backbone llama
    # narrow it:
    python score_oneshot_ood.py ... --split feynman --noise 0 --gaps 0 1 2 4
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
sys.path.insert(0, os.path.join(SCRIPT_DIR, "llm_methods"))

from eval_llmsr import load_feynman_problems, load_problems, output_metrics  # noqa: E402
from score_llmsr_ood import _cols, _load_ood  # noqa: E402  (shared, split-agnostic)
from oneshot.searcher import OneShotSearcher  # noqa: E402
import ood_transform  # noqa: E402

DEFAULT_SPLITS = ["feynman", "lsr_transform"]
DEFAULT_NOISES = [0.0]
DEFAULT_GAPS = ood_transform.default_gaps()

# Expression parsing/evaluation is the searcher's own, so a formula is read back exactly
# as it was written (same whitelist, same NaN-guarding lambdify). No LLM is constructed.
_PARSER = OneShotSearcher("score_oneshot_ood", llm=None)


def _artifact(results_root: str, method: str, split: str, noise: float) -> str:
    """Path to a one-shot run's artifact at target-noise `noise` (eval_oneshot.py layout,
    which is eval_llmsr.py's: flat file for Feynman, per-method dir for LLM-SRBench)."""
    nd = f"noise_{float(noise):g}"
    if split == "feynman":
        return os.path.join(results_root, nd, f"results_{method}_{split}.pkl.gz")
    return os.path.join(results_root, nd, "llmsrbench", f"{method}_{split}", "results.pkl.gz")


def _var_names(symbols) -> list:
    """The identifiers the searcher used for the input columns (same sanitisation)."""
    import re
    return [re.sub(r"\W", "_", str(v)) or f"x{i}" for i, v in enumerate(list(symbols)[1:])]


def _callable(expr_str: str, var_names: list):
    """Stored expression string -> f(X) over an (N, D) array; None if it will not parse."""
    expr, _params = _PARSER._parse(expr_str, var_names)
    if expr is None:
        return None
    return _PARSER._lambdify(expr, var_names)


def score_one(results_root, method, algorithm, backbone, split, noise, gaps,
              prob_by_id, ood_cache, out_dir):
    """Score one (split, noise) and write its CSV. Returns the path, or None if absent."""
    artifact = os.path.join(SCRIPT_DIR, _artifact(results_root, method, split, noise))
    tag = f"[{split} tau={float(noise):g}]"
    if not os.path.exists(artifact):
        print(f"{tag} MISSING artifact ({artifact}) -- skipped")
        return None
    with gzip.open(artifact) as f:
        recs = pickle.load(f)["results"]

    # Compile every stored formula, keyed (equation, seed) so multi-seed runs stay distinct.
    compiled, id_check, n_noeq, n_noparse = {}, [], 0, 0
    for r in recs:
        eqid, seed = r["equation_id"], int(r.get("seed", 0) or 0)
        prob = prob_by_id.get(eqid)
        eq_str = r.get("discovered_equation")
        if prob is None:
            continue
        if not eq_str:
            n_noeq += 1                    # the run itself failed on this problem
            continue
        names = _var_names(prob["symbols"])
        fn = _callable(eq_str, names)
        if fn is None:
            n_noparse += 1
            continue
        compiled[(eqid, seed)] = fn
        te = prob["test"]
        yp = fn(te[:, 1:])
        if yp is not None:
            id_check.append((output_metrics(yp, te[:, 0])["r2"],
                             (r.get("id_metrics") or {}).get("r2", np.nan)))
    if id_check:
        d = np.array([abs(a - b) for a, b in id_check if np.isfinite(a) and np.isfinite(b)])
        print(f"{tag} compiled {len(compiled)}/{len(recs)} "
              f"(no equation: {n_noeq}, unparseable: {n_noparse}); "
              f"id-R^2 vs stored: median|d|={np.median(d):.2e} max|d|={np.max(d):.2e}")

    rows = []
    for g in gaps:
        ood = _load_ood(split, g, ood_cache)
        if ood is None:
            print(f"{tag} [gap={g:>3}] missing OOD dataset -- skipped")
            continue
        n = 0
        for (eqid, seed), fn in compiled.items():
            ds = ood.get(eqid)
            if ds is None:
                continue
            X = np.asarray(ds["X"], dtype=np.float64)
            yp = fn(X)
            if yp is None or yp.shape[0] != X.shape[0]:
                continue
            r2 = output_metrics(yp, np.asarray(ds["y"], dtype=np.float64))["r2"]
            rows.append({"gap": g, "algorithm": algorithm, "dataset": eqid,
                         "seed": seed, "ood_r2": r2})
            n += 1
        print(f"{tag} [gap={g:>3}] scored {n}")

    out = os.path.join(out_dir,
                       ood_transform.oneshot_backbone_cache_name(backbone, split, noise))
    pd.DataFrame(rows, columns=["gap", "algorithm", "dataset", "seed", "ood_r2"]).to_csv(
        out, index=False)
    print(f"{tag} wrote {len(rows)} rows -> {out}\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", default=DEFAULT_SPLITS,
                    help="feynman and/or lsr_transform (default: both).")
    ap.add_argument("--noise", nargs="+", type=float, default=DEFAULT_NOISES,
                    help="Target-noise levels to score (default: 0).")
    ap.add_argument("--gaps", nargs="+", type=int, default=DEFAULT_GAPS)
    ap.add_argument("--results-root", default="results/oneshot",
                    help="Root holding noise_<tau>/ dirs (eval_oneshot.py --output).")
    ap.add_argument("--method", default="oneshot-fit-llama31-8b",
                    help="Method name in the artifact path (the config's lowercased name).")
    ap.add_argument("--algorithm", default="oneshot-llama",
                    help="Label written into the CSV's algorithm column (the plot legend).")
    ap.add_argument("--backbone", default=None,
                    help="Cache-name backbone tag (default: derived from --algorithm).")
    ap.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "results"))
    args = ap.parse_args()

    backbone = args.backbone or (args.algorithm.replace("oneshot-", "") or args.algorithm)
    os.makedirs(args.output_dir, exist_ok=True)
    ood_cache = {}
    for split in args.split:
        print(f"Loading problems for split '{split}' ...", flush=True)
        problems = (load_feynman_problems() if split == "feynman" else load_problems(split))
        prob_by_id = {q["name"]: q for q in problems}
        for noise in args.noise:
            score_one(args.results_root, args.method, args.algorithm, backbone,
                      split, noise, args.gaps, prob_by_id, ood_cache, args.output_dir)


if __name__ == "__main__":
    main()
