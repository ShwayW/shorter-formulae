#!/usr/bin/env python3
"""
eval_symformer.py -- Evaluate SymFormer (Vastl et al., 2022; vendored in
symformer/) on the SRBench Feynman benchmark and on LLM-SRBench.

    python eval_symformer.py --benchmark srbench
    python eval_symformer.py --benchmark llmsrbench --split lsr_transform
    python eval_symformer.py --benchmark both --seeds 42 43 ... 51 \
        --target-noise 0 0.001 0.01 0.1

Artifacts follow the usual results_io layout, so the plotting picks this up as
one more method:
    results/symformer/seed<N>/noise_<tau>/eval_symformer.pkl.gz
    results/symformer/seed<N>/noise_<tau>/llmsrbench/symformer_<split>/results.pkl.gz

!! READ THIS BEFORE RUNNING -- TWO HARD CONSTRAINTS !!
------------------------------------------------------
1. TENSORFLOW 2.8 / PYTHON <= 3.10.  SymFormer is a TF2 model and its released
   checkpoints only load under the pinned stack in symformer/requirements.txt
   (tensorflow==2.8.0, keras==2.8.0, tensorflow-text==2.8.1, protobuf==3.19,
   aclick==0.2.0).  tensorflow 2.8 publishes wheels for cp37-cp310 ONLY, and
   TF >= 2.16 ships Keras 3, against which this code does not run unmodified.
   The cluster wheelhouse carries only 2.15.1/2.16.1/2.17.0, so on a
   cluster this needs a container (see slurm/eval_symformer.sh) or a hand-built
   py3.10 venv from PyPI.  It will NOT run in the repo's main env/ (py3.13).

2. AT MOST 2 VARIABLES.  The variable set is baked into each checkpoint: the
   tokenizer is built from config.dataset_config.variables and the encoder is
   built for a fixed point count.  There are exactly two released models --
       symformer-univariate  variables=['x']      num_points=100
       symformer-bivariate   variables=['x','y']  num_points=200
   so every problem with >= 3 variables is out of reach, and the two models are
   NOT interchangeable (different input shapes).  We route per problem on
   n_vars.  Measured coverage of the benchmarks:
       Feynman        16 / 99   (1 univariate + 15 bivariate)
       lsr_transform   5 / 111  (0 univariate + 5 bivariate)
   Solve rates must therefore be reported against those denominators, not 99
   and 111.  --coverage prints the routing table and exits.

DATA PATH.  SymFormer's own Runner.predict() takes a ground-truth *equation*
and samples its own points from it, which we cannot use -- our data comes from
PMLB / the LLM-SRBench HDF5.  We call the lower-level
`runner.search.batch_decode(points)` instead, which is exactly what predict()
does internally (runner.py:predict), passing our own points tensor of shape
[1, num_points, n_vars + 1] with the target in the last column.  The decoder's
BestFittingFilter convertor then selects among candidates using those same
points, so selection is on TRAIN data only -- never the reported test slice.

CONSTANTS.  Unlike NeSymReS/tf4sr skeletons, SymFormer's regression head emits
numeric constants directly, so there is no BFGS refit here; --optimization
selects the authors' own constant-refinement mode ("gradient" is their default).
"""

import argparse
import gzip
import os
import pickle
import sys
import time
import warnings

import numpy as np
import sympy

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "symformer"))

import results_io                                                    # noqa: E402
from eval_nesymres import (feynman_problems, llmsrbench_problems,    # noqa: E402
                           add_target_noise, seed_everything, shard)

METHOD_NAME = "symformer"
SRBENCH_ARTIFACT = "eval_symformer.pkl.gz"

MAX_MODEL_VARS = 2
CHECKPOINT_FOR_NVARS = {1: "symformer-univariate", 2: "symformer-bivariate"}


def _clipped_r2(y, yp):
    """R^2 clipped at 0, ignoring non-finite predictions (as run_nesymres.py)."""
    from sklearn.metrics import r2_score
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yp = np.asarray(yp, dtype=np.float64).reshape(-1)
    m = np.isfinite(y) & np.isfinite(yp)
    if m.sum() == 0:
        return 0.0
    try:
        return float(max(0.0, r2_score(y[m], yp[m])))
    except Exception:
        return 0.0


# -- runner -------------------------------------------------------------------

class SymformerRunner:
    """Holds one Runner per arity, loaded lazily; fit()/predict() per problem.

    Both checkpoints are ~1.1 GB, so they are only materialised if a problem of
    that arity is actually reached.
    """

    def __init__(self, weights_dir, optimization, num_equations, use_pool,
                 early_stopping):
        self.weights_dir = weights_dir
        self.optimization = optimization
        self.num_equations = num_equations
        self.use_pool = use_pool
        self.early_stopping = early_stopping
        self._runners = {}
        self.fitted_expr_ = None
        self.fitted_fn_ = None
        self.n_cols_ = None
        self.top_k_idx_ = None        # no feature selection; kept for schema parity

    def runner_for(self, n_vars):
        if n_vars not in CHECKPOINT_FOR_NVARS:
            return None
        if n_vars not in self._runners:
            from symformer.model.runner import Runner
            name = CHECKPOINT_FOR_NVARS[n_vars]
            # A path is passed, not a bare name: pull_model() returns any argument
            # containing "/" or naming an existing path as-is, so this uses the
            # local checkpoint and never reaches the network.
            local = os.path.join(self.weights_dir, name)
            path = local if os.path.exists(local) else name
            print(f"  [symformer] loading {name} from {path}", flush=True)
            self._runners[n_vars] = Runner.from_checkpoint(
                path,
                optimization_type=self.optimization,
                use_pool=self.use_pool,
                early_stopping=self.early_stopping,
                num_equations=self.num_equations,
            )
        return self._runners[n_vars]

    def num_points_for(self, n_vars):
        r = self.runner_for(n_vars)
        return int(r.config.dataset_config.num_points) if r else None

    def fit(self, X, y, rng):
        """Decode one expression from (X, y).  Returns the formula string or None."""
        import tensorflow as tf

        self.fitted_expr_ = None
        self.fitted_fn_ = None
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_cols = X.shape[1]
        self.n_cols_ = n_cols

        runner = self.runner_for(n_cols)
        if runner is None:
            return None
        n_points = int(runner.config.dataset_config.num_points)

        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        if int(ok.sum()) == 0:
            return None
        Xv, yv = X[ok], y[ok]
        # The encoder is built for exactly num_points rows; sample with
        # replacement only if the problem genuinely has fewer.
        idx = rng.choice(len(Xv), n_points, replace=len(Xv) < n_points)
        pts = np.concatenate([Xv[idx], yv[idx].reshape(-1, 1)], axis=1)
        if not np.all(np.isfinite(pts)):
            return None

        points = tf.convert_to_tensor(pts[None, ...], dtype=tf.float32)
        try:
            out = runner.search.batch_decode(points)
        except Exception as exc:
            print(f"    decode failed: {type(exc).__name__}: {exc}", flush=True)
            return None

        # runner.predict() reads these same two slots: [-1] symbolic, [-2] infix.
        expr = out[-1][0]
        if expr is None:
            return None
        try:
            from symformer.dataset.utils.sympy_functions import expr_to_func
            sym = sympy.sympify(str(expr))
            self.fitted_fn_ = expr_to_func(sym, runner.variables)
            self.fitted_expr_ = sym
        except Exception:
            return None
        return self._formula_string(runner.variables)

    def _formula_string(self, variables):
        """Rewrite SymFormer's x/y into our v1..vn column space."""
        if self.fitted_expr_ is None:
            return None
        try:
            subs = {sympy.Symbol(v): sympy.Symbol(f"v{i + 1}")
                    for i, v in enumerate(variables)}
            return str(sympy.sympify(self.fitted_expr_).xreplace(subs))
        except Exception:
            return str(self.fitted_expr_)

    def predict(self, X):
        if self.fitted_fn_ is None:
            return np.full(len(X), np.nan)
        Xs = np.asarray(X, dtype=np.float64)[:, :self.n_cols_]
        try:
            from symformer.dataset.utils.sympy_functions import evaluate_points
            with np.errstate(all="ignore"):
                return np.asarray(evaluate_points(self.fitted_fn_, Xs),
                                  dtype=np.float64).reshape(-1)
        except Exception:
            return np.full(len(Xs), np.nan)


# -- SRBench -------------------------------------------------------------------

def _load_ood(path):
    if not path or not os.path.exists(path):
        return {}
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh).get("datasets", {})


def run_srbench(runner, problems, seed, tau, args):
    out_path = os.path.join(
        results_io.noise_dir(
            results_io.seed_dir(os.path.join(args.output, METHOD_NAME), seed), tau),
        SRBENCH_ARTIFACT)

    meta, rows = results_io.load_results(out_path)
    done = {r["dataset"] for r in rows}
    if done:
        print(f"  resuming: {len(done)} datasets already in {out_path}", flush=True)

    ood = _load_ood(args.ood_data)
    import pandas as pd

    for i, (name, n_vars, formula, target_pn) in enumerate(problems, 1):
        if name in done:
            continue
        rng = seed_everything(name, seed)
        try:
            df = pd.read_csv(
                os.path.join(args.pmlb_dir, "datasets", name, f"{name}.tsv.gz"),
                sep="\t", compression="gzip")
        except Exception as exc:
            print(f"  [{i}/{len(problems)}] {name}: load error {exc}", flush=True)
            continue

        if len(df) > args.n_points:
            df = df.iloc[rng.choice(len(df), args.n_points, replace=False)]
            df = df.reset_index(drop=True)
        feat_cols = [c for c in df.columns if c != "target"]
        X = df[feat_cols].values.astype(np.float64)
        y = df["target"].values.astype(np.float64)
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            print(f"  [{i}/{len(problems)}] {name}: non-finite data, skipped", flush=True)
            continue

        n_train = int(0.75 * len(df))
        X_tr, X_te = X[:n_train], X[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]
        y_tr = add_target_noise(y_tr, tau, rng)

        t0 = time.perf_counter()
        pred = runner.fit(X_tr, y_tr, rng)
        dt = time.perf_counter() - t0

        r2 = _clipped_r2(y_te, runner.predict(X_te)) if pred else 0.0
        ood_r2 = None
        if pred and name in ood:
            ood_r2 = _clipped_r2(ood[name]["y"], runner.predict(ood[name]["X"]))

        rows.append({
            "dataset": name, "seed": seed, "round": 1,
            "r2": r2, "ood_r2": ood_r2, "time": dt,
            "predicted_formula": pred, "predicted_formula_raw": pred,
            "feature_idx": list(range(X.shape[1])),
            "target_formula": formula, "target_pn": target_pn,
            "n_vars": n_vars,
        })
        results_io.save_results(
            out_path, rows, model_name=METHOD_NAME,
            checkpoint=CHECKPOINT_FOR_NVARS.get(n_vars, "n/a"),
            target_noise=tau, seed=seed, max_vars=args.max_vars,
            optimization=args.optimization, num_equations=args.num_equations)
        print(f"  [{i}/{len(problems)}] {name:<28} R^2={r2:.4f} "
              f"{'solved' if r2 >= args.r2_threshold else '      '} "
              f"t={dt:.1f}s | {pred}", flush=True)
    return rows


# -- LLM-SRBench ---------------------------------------------------------------

def run_llmsrbench(runner, problems, seed, tau, args):
    import h5py
    from eval_mymodels import resolve_llmsrbench_file

    method_dir = f"{METHOD_NAME}_{args.split}"
    out_path = os.path.join(
        results_io.llmsr_dir(
            results_io.seed_dir(os.path.join(args.output, METHOD_NAME), seed),
            tau, method_dir),
        results_io.RESULTS_NAME)

    meta, rows = results_io.load_results(out_path)
    done = {r["equation_id"] for r in rows}
    if done:
        print(f"  resuming: {len(done)} problems already in {out_path}", flush=True)

    hdf5_path = resolve_llmsrbench_file("lsr_bench_data.hdf5")
    if args.split == "lsr_transform":
        group_of = lambda n: f"/lsr_transform/{n}"
    else:
        group_of = lambda n: f"/lsr_synth/{args.split[len('lsr_synth_'):]}/{n}"

    with h5py.File(hdf5_path, "r") as fh:
        for i, (name, n_vars, expression, syms) in enumerate(problems, 1):
            if name in done:
                continue
            rng = seed_everything(name, seed)
            try:
                g = fh[group_of(name)]
                samples = {k: g[k][...].astype(np.float64) for k in g.keys()}
            except Exception as exc:
                print(f"  [{i}/{len(problems)}] {name}: load error {exc}", flush=True)
                continue

            train, test = samples.get("train"), samples.get("test")
            if train is None or test is None:
                continue
            X_tr, y_tr = train[:, 1:], train[:, 0]
            X_te, y_te = test[:, 1:], test[:, 0]
            y_tr = add_target_noise(y_tr, tau, rng)

            t0 = time.perf_counter()
            pred = runner.fit(X_tr, y_tr, rng)
            dt = time.perf_counter() - t0

            r2 = _clipped_r2(y_te, runner.predict(X_te)) if pred else 0.0
            ood = samples.get("ood_test")
            ood_metrics = None
            if pred and ood is not None:
                ood_metrics = {"r2": _clipped_r2(ood[:, 0], runner.predict(ood[:, 1:]))}

            rows.append({
                "equation_id": name, "gt_equation": expression,
                "discovered_equation": pred, "discovered_equation_raw": pred,
                "feature_idx": list(range(X_tr.shape[1])), "n_vars": n_vars,
                "num_datapoints": int(len(train)), "num_eval_datapoints": int(len(test)),
                "search_time": dt, "seed": seed,
                "id_metrics": {"r2": r2}, "ood_metrics": ood_metrics,
                "symbols": list(map(str, syms)),
            })
            results_io.save_results(
                out_path, rows, model_name=METHOD_NAME, split=args.split,
                checkpoint=CHECKPOINT_FOR_NVARS.get(n_vars, "n/a"),
                target_noise=tau, seed=seed, max_vars=args.max_vars,
                optimization=args.optimization, num_equations=args.num_equations)
            print(f"  [{i}/{len(problems)}] {name:<28} R^2={r2:.4f} "
                  f"{'solved' if r2 >= args.r2_threshold else '      '} "
                  f"t={dt:.1f}s | {pred}", flush=True)
    return rows


# -- Reporting -----------------------------------------------------------------

def summarise(tag, rows, thr):
    vals = []
    for r in rows:
        v = r.get("r2")
        if v is None and isinstance(r.get("id_metrics"), dict):
            v = r["id_metrics"].get("r2")
        if v is not None:
            vals.append(float(v))
    a = np.asarray(vals, dtype=np.float64)
    print(f"  {tag}: n={a.size}"
          + (f"  solve={100 * np.mean(a >= thr):.1f}%  meanR2={np.mean(a):.4f}"
             if a.size else ""), flush=True)


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=["srbench", "llmsrbench", "both"],
                    default="both")
    ap.add_argument("--split", default="lsr_transform")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--target-noise", type=float, nargs="+", default=[0.0],
                    dest="noises")
    ap.add_argument("--max-vars", type=int, default=MAX_MODEL_VARS,
                    help="Model ceiling is 2 (univariate + bivariate checkpoints). "
                         "Raising it only adds problems that will be skipped.")
    ap.add_argument("--output", default="results")
    ap.add_argument("--pmlb-dir", default="datasets/pmlb")
    ap.add_argument("--feynman-csv", default="datasets/feynman/FeynmanEquations.csv")
    ap.add_argument("--ood-data", default=None)
    ap.add_argument("--n-points", type=int, default=10000,
                    help="Rows subsampled per Feynman equation before the 75/25 split.")
    ap.add_argument("--r2-threshold", type=float, default=0.99)
    # Model knobs
    ap.add_argument("--weights-dir", default=os.environ.get(
        "SYMFORMER_WEIGHTS", os.path.join(_HERE, "weights", "symformer")))
    ap.add_argument("--optimization", default="gradient",
                    choices=["gradient", "no_optimization"],
                    help="The authors' constant-refinement mode. Default gradient.")
    ap.add_argument("--num-equations", type=int, default=256,
                    help="Candidates decoded per problem (their default 256).")
    ap.add_argument("--no-pool", action="store_true",
                    help="Disable their multiprocessing pool (needed inside some "
                         "job launchers).")
    ap.add_argument("--early-stopping", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--list-problems", action="store_true")
    ap.add_argument("--coverage", action="store_true",
                    help="Print the per-arity routing table and exit (no TF needed).")
    args = ap.parse_args()

    do_sr = args.benchmark in ("srbench", "both")
    do_lsr = args.benchmark in ("llmsrbench", "both")

    sr_problems = (feynman_problems(args.pmlb_dir, args.feynman_csv, args.max_vars)
                   if do_sr else [])
    lsr_problems = (llmsrbench_problems(args.split, args.max_vars) if do_lsr else [])

    if args.coverage:
        import collections
        for tag, probs, total in (("Feynman", sr_problems, 99),
                                  ("lsr_transform", lsr_problems, 111)):
            c = collections.Counter(p[1] for p in probs)
            uni, bi = c.get(1, 0), c.get(2, 0)
            print(f"{tag}: reachable {uni + bi}/{total}  "
                  f"(univariate {uni}, bivariate {bi})")
        return

    if args.list_problems:
        for n, k, f, _ in sr_problems:
            print(f"srbench    {n:<28} {k}v  {f}")
        for n, k, e, _ in lsr_problems:
            print(f"llmsrbench {n:<28} {k}v  {e}")
        print(f"\nsrbench: {len(sr_problems)}   llmsrbench: {len(lsr_problems)}")
        return

    sr_problems = shard(sr_problems, args.shard, args.n_shards)
    lsr_problems = shard(lsr_problems, args.shard, args.n_shards)

    runner = SymformerRunner(
        weights_dir=args.weights_dir, optimization=args.optimization,
        num_equations=args.num_equations, use_pool=not args.no_pool,
        early_stopping=args.early_stopping)

    print(f"SymFormer | max_vars<={args.max_vars} | opt={args.optimization} | "
          f"num_equations={args.num_equations} | shard {args.shard}/{args.n_shards} | "
          f"srbench {len(sr_problems)} + llmsrbench {len(lsr_problems)} problems | "
          f"seeds {args.seeds} | noise {args.noises}", flush=True)

    for seed in args.seeds:
        for tau in args.noises:
            print(f"\n=== seed {seed} | noise {tau:g} ===", flush=True)
            if sr_problems:
                summarise("srbench",
                          run_srbench(runner, sr_problems, seed, tau, args),
                          args.r2_threshold)
            if lsr_problems:
                summarise("llmsrbench",
                          run_llmsrbench(runner, lsr_problems, seed, tau, args),
                          args.r2_threshold)


if __name__ == "__main__":
    main()
