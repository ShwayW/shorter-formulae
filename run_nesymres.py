#!/usr/bin/env python3
"""
run_nesymres.py -- Run the pretrained NeSymReS 100M model (Biggio et al. 2021)
on either the SRBench Feynman benchmark or the LLM-SRBench benchmark.

    # SRBench (119 Feynman PMLB datasets, R^2 >= 0.99 counts as solved)
    python run_nesymres.py srbench

    # LLM-SRBench (default split lsr_transform; writes results.jsonl that
    # compare_llmsrbench.py can pick up as one method)
    python run_nesymres.py llmsrbench --split lsr_transform

The 100M checkpoint is loaded ONCE and reused across every problem (the srbench
regressor reloads it on each .fit(), which is far too slow for 100+ problems).
NeSymReS supports at most 3 input variables, so -- exactly like the srbench
regressor -- the top-3 features (by |r| with the target) are selected per problem;
the discovered equation is written in terms of x_1..x_k for those columns.

Model plumbing (config, tokenizer, BFGS refinement, feature selection, sympy
prediction) is reused verbatim from the srbench regressor we already wired up:
    srbench/experiment/methods/nesymres/regressor.py
so the two entry points stay in sync. Override the checkpoint with the
NESYMRES_WEIGHTS env var or --weights-path.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
_REGRESSOR_PATH = (REPO_ROOT / "srbench" / "experiment" / "methods"
                   / "nesymres" / "regressor.py")
# The vendored NeSymReS source lives at repo-root nesymres/ (pip install -e
# nesymres/src). The checkpoint is gitignored, so it lives in the repo-root
# weights/ dir next to e2e_model.pt. Override with NESYMRES_WEIGHTS or
# --weights-path.
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "nesymres" / "100M.ckpt"


# ---------------------------------------------------------------------------
# Reuse the NeSymReS config + helpers from the srbench regressor module.
# It imports with no side effects beyond building the omegaconf config once
# (no checkpoint load happens at import time -- that is inside .fit()).
# ---------------------------------------------------------------------------

def _load_regressor_module():
    if not _REGRESSOR_PATH.exists():
        sys.exit(f"[fatal] srbench regressor not found at {_REGRESSOR_PATH}")
    spec = importlib.util.spec_from_file_location(
        "nesymres_srbench_regressor", _REGRESSOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# One-time-loaded NeSymReS runner
# ---------------------------------------------------------------------------

class NeSymReSRunner:
    """Loads the 100M checkpoint once; fit()/predict() per problem."""

    MAX_VARS = 3  # the pretrained model supports at most 3 input variables

    def __init__(self, weights_path: str, device: str,
                 beam_size: int, n_restarts: int, max_fit_points: int):
        reg = _load_regressor_module()
        self._reg = reg
        self.cfg = reg.cfg
        self.eq_setting = reg.eq_setting
        self.device = device
        self.max_fit_points = max_fit_points

        # Apply the tunable inference knobs onto the shared config.
        self.cfg.inference.beam_size = beam_size
        self.cfg.inference.bfgs.n_restarts = n_restarts

        print(f"Loading NeSymReS 100M from {weights_path} ...", flush=True)
        # weights_only=False: torch>=2.6 otherwise refuses this trusted Lightning ckpt.
        self.model = reg.Model.load_from_checkpoint(
            weights_path, cfg=self.cfg.architecture, weights_only=False)
        self.model.eval()
        if device == "cuda" and torch.cuda.is_available():
            self.model.cuda()
        print("Model loaded.\n", flush=True)

    def _params_fit(self, n_vars: int):
        reg, cfg, eq = self._reg, self.cfg, self.eq_setting
        bfgs = reg.BFGSParams(
            activated=cfg.inference.bfgs.activated,
            n_restarts=cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=cfg.inference.bfgs.normalization_o,
            idx_remove=cfg.inference.bfgs.idx_remove,
            normalization_type=cfg.inference.bfgs.normalization_type,
            stop_time=cfg.inference.bfgs.stop_time,
        )
        return reg.FitParams(
            word2id=eq["word2id"],
            id2word={int(k): v for k, v in eq["id2word"].items()},
            una_ops=eq["una_ops"], bin_ops=eq["bin_ops"],
            total_variables=[f"x_{i+1}" for i in range(n_vars)],
            total_coefficients=eq["total_coefficients"],
            rewrite_functions=eq["rewrite_functions"],
            bfgs=bfgs, beam_size=cfg.inference.beam_size,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState):
        """Fit on (X, y); returns the discovered equation string or None.

        Stores the selected feature columns + parsed variables so predict()
        can be evaluated on held-out / OOD inputs with the same mapping.
        """
        self.top_k_idx_ = None
        self.formula_ = None
        self.pred_variables_ = []

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        # Keep at most 3 features (model limit), chosen by |correlation| with y.
        self.top_k_idx_ = self._reg.get_top_k_features(X, y, k=self.MAX_VARS)
        Xk = X[:, self.top_k_idx_]

        # Cap the point set fed to the set-transformer encoder (trained on <=~500).
        if len(Xk) > self.max_fit_points:
            sel = rng.choice(len(Xk), self.max_fit_points, replace=False)
            Xk, yk = Xk[sel], y[sel]
        else:
            yk = y

        params_fit = self._params_fit(Xk.shape[1])
        fitfunc = partial(self.model.fitfunc, cfg_params=params_fit)
        try:
            out = fitfunc(Xk, yk)
        except Exception as exc:
            print(f"   [fit failed: {exc}]", flush=True)
            return None

        # Prefer the refined equation (matches the srbench regressor); fall back
        # to the raw best BFGS prediction.
        formula = None
        try:
            eqs = self.model.get_equation()
            if eqs:
                formula = eqs[0]
        except Exception:
            pass
        if formula is None and isinstance(out, dict):
            preds = out.get("best_bfgs_preds") or out.get("all_bfgs_preds")
            if preds:
                formula = preds[0]

        if formula is not None:
            try:
                self.pred_variables_ = self._reg.get_variables(formula)
            except Exception:
                self.pred_variables_ = []
        self.formula_ = formula
        return formula

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the discovered equation on X; NaN vector on any failure."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        if self.formula_ is None:
            return np.full(n, np.nan)
        try:
            Xk = X[:, self.top_k_idx_]
            yp = self._reg.evaluate_func(self.formula_, self.pred_variables_, Xk)
            yp = np.asarray(yp, dtype=np.float64).reshape(-1)
            if yp.shape[0] == 1 and n > 1:      # constant prediction
                yp = np.full(n, yp[0])
            if yp.shape[0] != n or not np.all(np.isfinite(yp)):
                bad = np.full(n, np.nan)
                bad[:yp.shape[0]] = yp
                return bad
            return yp
        except Exception:
            return np.full(n, np.nan)


# ---------------------------------------------------------------------------
# SRBench (Feynman PMLB) benchmark
# ---------------------------------------------------------------------------

def _list_feynman(pmlb_dir: str):
    ddir = os.path.join(pmlb_dir, "datasets")
    return sorted(d for d in os.listdir(ddir)
                  if d.startswith("feynman_") and os.path.isdir(os.path.join(ddir, d)))


def _load_pmlb(pmlb_dir: str, name: str):
    import pandas as pd
    path = os.path.join(pmlb_dir, "datasets", name, f"{name}.tsv.gz")
    return pd.read_csv(path, sep="\t", compression="gzip")


def run_srbench(runner: NeSymReSRunner, args):
    import csv
    import gzip
    import pickle

    names = _list_feynman(args.pmlb_dir)
    if args.max_problems is not None:
        names = names[:args.max_problems]
    print(f"SRBench: {len(names)} Feynman datasets in {args.pmlb_dir}\n", flush=True)

    ood = {}
    if args.ood_data:
        with gzip.open(args.ood_data, "rb") as fh:
            ood = pickle.load(fh).get("datasets", {})
        print(f"Loaded OOD data for {len(ood)} datasets from {args.ood_data}\n", flush=True)

    out_csv = args.output or "results/nesymres_srbench.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    done = set()
    rows = []
    if os.path.exists(out_csv):
        with open(out_csv) as fh:
            for r in csv.DictReader(fh):
                f = (r.get("predicted_formula") or "").strip()
                if f and f not in ("None", "nan"):
                    rows.append(r)
                    done.add(r["dataset"])
        if done:
            print(f"Resuming: {len(done)} datasets already done in {out_csv}\n", flush=True)

    fields = ["dataset", "predicted_formula", "r2", "ood_r2",
              "accuracy", "n_vars", "fit_time_s", "seed"]
    rng = np.random.RandomState(args.seed)

    for i, name in enumerate(names):
        if name in done:
            print(f"[{i+1}/{len(names)}] {name} ... skipped (done)", flush=True)
            continue
        try:
            df = _load_pmlb(args.pmlb_dir, name)
        except Exception as exc:
            print(f"[{i+1}/{len(names)}] {name} ... load error: {exc}", flush=True)
            continue

        if len(df) > args.n_points:
            df = df.iloc[rng.choice(len(df), args.n_points, replace=False)].reset_index(drop=True)
        feat_cols = [c for c in df.columns if c != "target"]
        X = df[feat_cols].values.astype(np.float64)
        y = df["target"].values.astype(np.float64)
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            print(f"[{i+1}/{len(names)}] {name} ... non-finite data, skipped", flush=True)
            continue

        n_train = int(0.75 * len(df))
        X_tr, X_te = X[:n_train], X[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]

        if args.noise > 0 and len(y_tr):
            rms = float(np.sqrt(np.mean(y_tr ** 2)))
            if rms > 0:
                y_tr = y_tr + rng.normal(0.0, args.noise * rms, size=len(y_tr))

        t0 = time.perf_counter()
        formula = runner.fit(X_tr, y_tr, rng)
        dt = time.perf_counter() - t0

        yp = runner.predict(X_te)
        r2 = _clipped_r2(y_te, yp)
        ood_r2 = None
        if name in ood:
            ood_r2 = _clipped_r2(ood[name]["y"], runner.predict(ood[name]["X"]))

        row = {"dataset": name, "predicted_formula": formula, "r2": f"{r2:.6f}",
               "ood_r2": ("" if ood_r2 is None else f"{ood_r2:.6f}"),
               "accuracy": int(r2 >= args.r2_threshold),
               "n_vars": len(feat_cols), "fit_time_s": f"{dt:.2f}", "seed": args.seed}
        rows.append(row)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[{i+1}/{len(names)}] {name:<28} R^2={r2:.4f} "
              f"{'solved' if r2 >= args.r2_threshold else '      '} t={dt:.1f}s "
              f"| {formula}", flush=True)

    _summarize_srbench(rows, args.r2_threshold, out_csv)


def _clipped_r2(y, yp):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yp = np.asarray(yp, dtype=np.float64).reshape(-1)
    m = np.isfinite(y) & np.isfinite(yp)
    if m.sum() == 0:
        return 0.0
    try:
        return float(max(0.0, r2_score(y[m], yp[m])))
    except Exception:
        return 0.0


def _summarize_srbench(rows, thr, out_csv):
    r2s = np.array([float(r["r2"]) for r in rows], dtype=np.float64) if rows else np.array([])
    print(f"\n{'='*70}")
    print(f"NeSymReS 100M  *  SRBench Feynman  ({len(rows)} datasets)")
    if r2s.size:
        print(f"  Solve rate (R^2 >= {thr}) : {100*np.mean(r2s >= thr):.1f}%  "
              f"({int(np.sum(r2s >= thr))}/{len(rows)})")
        print(f"  Mean R^2                : {np.mean(r2s):.4f}   Median R^2: {np.median(r2s):.4f}")
    print(f"{'='*70}\n[results] {out_csv}\n", flush=True)


# ---------------------------------------------------------------------------
# LLM-SRBench benchmark
# ---------------------------------------------------------------------------

def run_llmsrbench(runner: NeSymReSRunner, args):
    # Reuse the exact loader + metrics used by eval_e2e.py --benchmark llmsrbench so the
    # results.jsonl is directly comparable via compare_llmsrbench.py.
    from eval_mymodels import load_problems, output_metrics

    print(f"Loading LLM-SRBench split '{args.split}' ...", flush=True)
    problems = load_problems(args.split)
    if args.problem:
        problems = [q for q in problems if q["name"] == args.problem]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    print(f"  {len(problems)} problems.\n", flush=True)

    base_out = args.output or "results/llmsrbench"
    if args.noise > 0:
        base_out = os.path.join(base_out, f"noise_{args.noise:g}")
    out_dir = os.path.join(base_out, f"nesymres_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "results.jsonl")

    done, rows = set(), []
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                eq = r.get("discovered_equation")
                if r.get("equation_id") in done or eq in (None, "None", ""):
                    continue
                done.add(r["equation_id"])
                rows.append(r)
        if done:
            print(f"Resuming: {len(done)} problems already done in {jsonl_path}\n", flush=True)

    rng = np.random.RandomState(args.seed)
    with open(jsonl_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, allow_nan=True) + "\n")
        fh.flush()

        for i, q in enumerate(problems):
            if q["name"] in done:
                continue
            train, test = q["train"], q["test"]
            n_vars = train.shape[1] - 1
            X_tr, y_tr = train[:, 1:], train[:, 0]      # col 0 = y (LLM-SRBench convention)
            X_te, y_te = test[:, 1:], test[:, 0]

            if args.noise > 0 and len(y_tr):
                rms = float(np.sqrt(np.mean(y_tr ** 2)))
                if rms > 0:
                    y_tr = y_tr + rng.normal(0.0, args.noise * rms, size=len(y_tr))

            t0 = time.perf_counter()
            formula = runner.fit(X_tr, y_tr, rng)
            dt = time.perf_counter() - t0

            id_m = output_metrics(runner.predict(X_te), y_te)
            ood_m = None
            if q["ood_test"] is not None:
                ood_m = output_metrics(
                    runner.predict(q["ood_test"][:, 1:]), q["ood_test"][:, 0])

            log = {
                "equation_id": q["name"],
                "gt_equation": q["expression"],
                "discovered_equation": formula,
                "n_vars": n_vars,
                "num_datapoints": int(len(train)),
                "num_eval_datapoints": int(len(test)),
                "search_time": dt,
                "id_metrics": id_m,
                "ood_metrics": ood_m,
            }
            fh.write(json.dumps(log, allow_nan=True) + "\n")
            fh.flush()
            rows.append(log)
            print(f"[{i+1}/{len(problems)}] {q['name']:<24} "
                  f"R^2={id_m['r2']:.4f}  t={dt:.1f}s", flush=True)

    r2s = np.array([r["id_metrics"]["r2"] for r in rows], dtype=np.float64)
    r2s = r2s[np.isfinite(r2s)]
    print(f"\n{'='*70}")
    print(f"NeSymReS 100M  *  LLM-SRBench {args.split}  ({len(rows)} problems)")
    if r2s.size:
        print(f"  Acc (R^2 >= 0.99) : {100*np.mean(r2s >= 0.99):.1f}%")
        print(f"  Mean R^2         : {np.mean(r2s):.4f}   Median R^2: {np.median(r2s):.4f}")
    print(f"{'='*70}\n[results] {jsonl_path}\n", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark", choices=["srbench", "llmsrbench"],
                   help="Which benchmark to run the NeSymReS 100M model on.")

    # Model / inference knobs (shared)
    p.add_argument("--weights-path", type=str,
                   default=os.environ.get("NESYMRES_WEIGHTS", str(DEFAULT_WEIGHTS)),
                   help="NeSymReS checkpoint (default: vendored weights/nesymres/100M.ckpt "
                        "or $NESYMRES_WEIGHTS).")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--beam-size", type=int, default=2,
                   help="Decoder beam size (accuracy/speed tradeoff). Default 2.")
    p.add_argument("--n-restarts", type=int, default=10,
                   help="BFGS restarts for constant refinement. Default 10.")
    p.add_argument("--max-fit-points", type=int, default=500,
                   help="Max points fed to the encoder per problem. Default 500.")
    p.add_argument("--max-problems", type=int, default=None,
                   help="Cap the number of problems (smoke test).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None,
                   help="Output path. srbench: CSV file "
                        "(default results/nesymres_srbench.csv). "
                        "llmsrbench: base dir (default results/llmsrbench).")
    p.add_argument("--noise", type=float, default=0.0, metavar="TAU",
                   help="Additive-RMS noise on TRAIN targets only "
                        "(y_train += N(0, TAU*rms(y))). Test/OOD stay clean.")

    # SRBench-only
    p.add_argument("--pmlb-dir", type=str, default="datasets/pmlb",
                   help="[srbench] Root of the PMLB mirror (datasets/feynman_*/).")
    p.add_argument("--n-points", type=int, default=1000,
                   help="[srbench] Rows subsampled per equation before 75/25 split.")
    p.add_argument("--r2-threshold", type=float, default=0.99,
                   help="[srbench] R^2 threshold counting a problem as solved.")
    p.add_argument("--ood-data", type=str, default=None,
                   help="[srbench] Optional OOD .pkl.gz from gen_ood_data.py.")

    # LLM-SRBench-only
    p.add_argument("--split", type=str, default="lsr_transform",
                   help="[llmsrbench] Benchmark split (e.g. lsr_transform).")
    p.add_argument("--problem", type=str, default=None,
                   help="[llmsrbench] Run a single problem by name.")

    args = p.parse_args()

    print(f"Device: {args.device}", flush=True)
    runner = NeSymReSRunner(
        weights_path=args.weights_path, device=args.device,
        beam_size=args.beam_size, n_restarts=args.n_restarts,
        max_fit_points=args.max_fit_points)

    if args.benchmark == "srbench":
        run_srbench(runner, args)
    else:
        run_llmsrbench(runner, args)


if __name__ == "__main__":
    main()
