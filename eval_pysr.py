#!/usr/bin/env python3
"""
eval_pysr.py -- Evaluate PySR (Cranmer 2023, github.com/MilesCranmer/PySR) on SRBench
(Feynman) and LLM-SRBench, writing results in the SAME artifacts every other eval_*.py
writes, so compare_srbench.py / compare_llmsrbench.py and the plotting stack can pick
PySR up as one more method.

WHY WE RUN IT OURSELVES (there are no published numbers to reuse)
-----------------------------------------------------------------
srbench/results/ground-truth_results.feather -- the published SRBench results this repo
compares against -- holds FOURTEEN algorithms:

    AFP, AFP_FE, AIFeynman, BSR, DSR, EPLEX, FEAT, FFX, GP-GOMEA, ITEA, MRGP,
    Operon, SBP-GP, gplearn

PySR is NOT among them.  It reached the SRBench *repository* only after the NeurIPS 2021
paper, as part of the post-paper algorithm collection (srbench/algorithms/pysr, alongside
Brush / QLattice / TIR / uDSR, none of which are in the feather either).  So SRBench ships
an official PySR *configuration* but no official PySR *results*.  Nothing exists to
compare against; the numbers have to be produced here.

THE BUDGET, AND WHY IT IS NOT SRBENCH'S
---------------------------------------
srbench/experiment/methods/pysr/regressor.py sets

    timeout_in_seconds=60*60 - 10*60,          # = 3000 s = 50 min

but that value NEVER TAKES EFFECT.  srbench/experiment/evaluate_model.py:156-158 does

    MAXTIME = fit_time_limit                                  # default 3600
    if len(y_train) > 1000 and not test and MAXTIME < 36000:
        MAXTIME = 36000                                       # 10 HOURS

and then at :172-174 overwrites `est.timeout_in_seconds = MAXTIME`.  The Feynman PMLB
files are 100,000 rows, so the branch always fires and SRBench's real PySR budget is
10 h/fit.  (The same override hits AI Feynman: its tuned params say max_time=7200, and
`est.max_time` is tested FIRST in that if-chain, so it too gets 36000.)

10 h x 230 problems x 10 seeds x 4 noise levels = 92,000 core-hours, which is not a run
anyone is doing.  And since there is no published PySR number waiting to be matched,
protocol fidelity buys nothing here -- the only comparability that matters is INTERNAL,
against the arms this repo ran itself.  So the protocol is:

    * SRBench's tuned operator set / constraints / population settings, verbatim
    * a --time-limit CEILING (default 3600 s) instead of 36000
    * --early-stop 1e-10 (a loss threshold), which SRBench does NOT set

The early stop is the important deviation and it is a deliberate one.  With
niterations=1e9 and no stopping condition, PySR burns the ENTIRE budget on
feynman_I_12_1 (F = m*a) exactly as it does on the hardest problem in the suite.  Stopping
once the loss is already at machine precision costs nothing in quality, moves the cost
onto the problems that actually need it, and makes `search_time` a real measurement --
which is what the time-vs-noise / time-vs-complexity figures plot.  Without it that axis
is a flat line at the budget and says nothing.

OUTPUT FORMAT
-------------
Both benchmarks store the discovered formula as INFIX with x_0, x_1, ... variables --
the same convention eval_aifeynman.normalize_vars produces and the same one
plot_ood_vs_gap / plot_llmsrbench_ood_vs_gap._is_infix routes to the numpy evaluator.
PySR's sympy printer emits `x0`, so it is rewritten here; getting this wrong sends the
whole method through wrapExtEvalPN and yields NaN for every row.

    SRBench     -> <results-dir>/noise_<TAU>/results_feynman_pysr.pkl.gz
    LLM-SRBench -> <results-dir>/noise_<TAU>/llmsrbench/pysr_<split>/results.pkl.gz

Row schemas match eval_e2e.py (SRBench) and eval_aifeynman.py (LLM-SRBench) field for
field, so merge_seed_results.py / check_results.py treat this group like any other.

Usage:
    # one (seed, noise) cell, both benchmarks, sharded across 16 workers
    python eval_pysr.py --benchmark both --seed 42 --target-noise 0 \
        --num-shards 16 --shard-id $k --results-dir results/pysr/seed42

    # fold the shards into the canonical per-seed artifacts
    python eval_pysr.py --benchmark both --seed 42 --target-noise 0 \
        --num-shards 16 --merge-shards --results-dir results/pysr/seed42

Each fit runs in a SUBPROCESS: PySR's own timeout governs its search loop but not Julia
compilation or the final constant optimisation, and a Julia-level crash would otherwise
take the whole worker (and its remaining problems) with it.  ~13 s of Julia startup per
fit is the price; against a 3600 s ceiling that is 0.4%.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# SRBench's tuned PySR configuration, from srbench/experiment/methods/pysr/regressor.py.
# Kept verbatim EXCEPT where PySR 2.x renamed things (noted inline) and except the two
# knobs the docstring above explains (timeout / early stop), which are CLI flags.
#
# `turbo=True` in SRBench's file is deliberately NOT carried over: it selects
# LoopVectorization kernels, is deprecated upstream, and trades numerical reproducibility
# for speed -- which fights --deterministic.  Everything else is unchanged.
SRBENCH_BINARY_OPS = ["+", "-", "*", "/"]
SRBENCH_UNARY_OPS = ["sin", "exp", "log", "sqrt"]
SRBENCH_CONSTRAINTS = {"sin": 9, "exp": 9, "log": 9, "sqrt": 9, "/": (-1, 9)}
SRBENCH_NESTED_CONSTRAINTS = {
    "sin": {"sin": 0, "exp": 1, "log": 1, "sqrt": 1},
    "exp": {"exp": 0, "log": 0},
    "log": {"exp": 0, "log": 0},
    "sqrt": {"sqrt": 0},
}


# PySR accepts early_stop_condition as a FLOAT (stop once the loss reaches it) or as a
# string holding a whole Julia function, "f(loss, complexity) = ...".  It is handed
# straight to jl.seval, so anything else -- notably the tempting bare predicate
# "loss < 1e-10" -- dies with `UndefVarError: loss not defined` AFTER paying ~13 s of
# Julia startup, once per fit.  Across a cell that is an hour of wasted CPU producing
# nothing but exit1 rows, so the shape is validated here, in the parent, before any
# worker is launched.
_JULIA_FN_RE = re.compile(r"^\s*\w+\s*\(\s*loss\s*,\s*complexity\s*\)\s*=", re.I)


def parse_early_stop(spec):
    """'' / None -> None (upstream behaviour).  A number -> float.  Otherwise a Julia
    'f(loss, complexity) = ...' definition, passed through."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    s = str(spec).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    if _JULIA_FN_RE.match(s):
        return s
    raise SystemExit(
        f"--early-stop {spec!r} is neither a number nor a Julia function definition.\n"
        f"  PySR seval's this verbatim, so a bare predicate fails with "
        f"'UndefVarError: loss not defined'.\n"
        f"  Use a threshold      : --early-stop 1e-10\n"
        f"  or a full definition : --early-stop 'f(loss, complexity) = loss < 1e-10 && complexity < 15'\n"
        f"  or '' for upstream's no-early-stop behaviour.")


def early_stop_for(args, y_train) -> "float | str | None":
    """The early_stop_condition for ONE problem, scaled to that problem's target.

    PySR's loss is a plain MSE, but mean(y^2) across these benchmarks spans FOURTEEN
    orders of magnitude (measured: 1.1 to 4.8e14 over lsr_transform).  So a fixed
    absolute threshold is not one criterion, it is 111 different ones: on a
    small-magnitude target `loss < 1e-10` is a sane "machine precision reached", while
    on a large-magnitude target it demands a relative error of ~1e-25, which float64
    cannot represent -- the stop never fires and the fit burns the whole ceiling even
    after recovering the equation exactly.  That is not hypothetical: the smoke run had
    feynman_III_13_18 at R^2=1.0 still running to its budget.

    So --early-stop is RELATIVE by default: the threshold handed to PySR is
    `--early-stop * mean(y_train^2)`, i.e. a normalised-MSE target that means the same
    thing on every problem.  --early-stop-absolute opts out.  A Julia function string
    is passed through untouched (we cannot scale someone else's expression).
    """
    spec = parse_early_stop(args.early_stop)
    if spec is None or isinstance(spec, str):
        return spec
    if args.early_stop_absolute:
        return spec
    scale = float(np.mean(np.square(np.asarray(y_train, dtype=np.float64))))
    if not np.isfinite(scale) or scale <= 0.0:
        return spec
    return spec * scale


def pysr_kwargs(args, seed: int, y_train=None) -> dict:
    """The PySRRegressor kwargs for one fit.  Single-core by construction: the sweep's
    parallelism is one PROCESS per (problem, seed, noise) cell, so letting Julia also
    fan out would oversubscribe every allocated core several times over."""
    return dict(
        niterations=args.niterations,
        # PySR 2.x renamed ncyclesperiteration -> ncycles_per_iteration.
        ncycles_per_iteration=2500,
        population_size=100,
        populations=2,
        maxsize=args.maxsize,
        maxdepth=args.maxdepth,
        binary_operators=list(SRBENCH_BINARY_OPS),
        unary_operators=list(SRBENCH_UNARY_OPS),
        constraints=dict(SRBENCH_CONSTRAINTS),
        nested_constraints={k: dict(v) for k, v in SRBENCH_NESTED_CONSTRAINTS.items()},
        batching=True,
        batch_size=50,
        weight_optimize=0.001,
        adaptive_parsimony_scaling=1000.0,
        parsimony=0.0,
        timeout_in_seconds=float(args.time_limit),
        early_stop_condition=early_stop_for(args, y_train),
        # PySR 2.x replaced procs/multithreading with `parallelism`.  "serial" is the
        # single-core equivalent of SRBench's procs=1, multithreading=False, and is
        # ALSO a hard requirement for deterministic=True.
        parallelism="serial",
        deterministic=True,
        random_state=int(seed) & 0x7FFFFFFF,
        verbosity=0,
        progress=False,
    )


# -- formula normalisation ----------------------------------------------------

_VAR_RE = re.compile(r"\bx(\d+)\b")
# sympy prints Euler's number as `E` and absolute value as `Abs`; the downstream numpy
# namespace (plot_llmsrbench_ood_vs_gap._INFIX_EVAL_NS_BASE) binds `e` and `abs`.
_E_RE = re.compile(r"\bE\b")
_ABS_RE = re.compile(r"\bAbs\b")
# Anything sympy emits for a degenerate expression.  These are not formulas and must not
# be stored as if they were -- downstream they would eval to a bare name and raise.
_DEGENERATE = ("zoo", "oo", "nan", "I")


def normalize_expr(expr) -> "str | None":
    """PySR's sympy string -> the repo's infix convention (x_0, x_1, ...).

    Returns None for an absent or degenerate expression.  See the module docstring:
    the x_<i> spelling is what routes this method to the numpy evaluator rather than
    to wrapExtEvalPN.
    """
    if expr is None:
        return None
    s = str(expr).strip()
    if not s:
        return None
    if any(re.search(rf"\b{re.escape(t)}\b", s) for t in _DEGENERATE):
        return None
    s = _VAR_RE.sub(r"x_\1", s)
    s = _ABS_RE.sub("abs", s)
    s = _E_RE.sub("e", s)
    return s


# The exact namespace plot_llmsrbench_ood_vs_gap._eval_infix uses.  Scoring here with
# the same evaluator the plots use means a formula that scores in this file cannot
# silently become NaN downstream.
_EVAL_NS = {
    "__builtins__": {},
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "pi": np.pi, "e": np.e,
}


def predict_expr(expr: "str | None", X: np.ndarray) -> np.ndarray:
    """Evaluate a normalized infix formula on X (N, D); NaN on any failure."""
    n = X.shape[0]
    if not expr:
        return np.full(n, np.nan)
    ns = dict(_EVAL_NS)
    for i in range(X.shape[1]):
        ns[f"x_{i}"] = X[:, i].astype(np.float64)
    try:
        with np.errstate(all="ignore"):
            out = eval(expr, ns)  # noqa: S307 -- our own sympy output, not user input
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 0:
            out = np.full(n, float(out))
        if out.shape[0] != n:
            return np.full(n, np.nan)
        return np.where(np.isfinite(out), out, np.nan)
    except Exception:
        return np.full(n, np.nan)


# -- the per-problem worker (its own process; see the module docstring) -------

def _worker_main(argv) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args(argv)
    with open(a.config) as fh:
        cfg = json.load(fh)

    import numpy as _np
    d = _np.load(cfg["data"])
    X, y = d["X"], d["y"]

    from pysr import PySRRegressor
    kw = dict(cfg["kwargs"])
    # Keep every Julia/PySR artifact inside the scratch dir; PySR otherwise litters
    # ./outputs/ in the CURRENT WORKING DIRECTORY, which for us is the repo root.
    kw["output_directory"] = cfg["workdir"]
    kw["run_id"] = "run"
    model = PySRRegressor(**kw)
    model.fit(X, y)

    # model_selection="best" (PySR's default) trades accuracy against complexity on the
    # Pareto front -- the same spirit as eval_aifeynman's --select test_error.
    out = {"expression": None, "complexity": None, "loss": None}
    try:
        out["expression"] = str(model.sympy())
    except Exception:
        pass
    try:
        best = model.get_best()
        out["complexity"] = int(best["complexity"])
        out["loss"] = float(best["loss"])
    except Exception:
        pass
    with open(cfg["result"], "w") as fh:
        json.dump(out, fh)
    return 0


def solve_one(X_train, y_train, workdir, args, seed) -> tuple:
    """Fit PySR on one problem in a subprocess. Returns (best_dict, elapsed, status)."""
    os.makedirs(workdir, exist_ok=True)
    data_path = os.path.join(workdir, "data.npz")
    np.savez(data_path, X=np.asarray(X_train, dtype=np.float64),
             y=np.asarray(y_train, dtype=np.float64))

    cfg_path = os.path.join(workdir, "config.json")
    res_path = os.path.join(workdir, "result.json")
    with open(cfg_path, "w") as fh:
        json.dump({"data": data_path, "workdir": workdir, "result": res_path,
                   "kwargs": pysr_kwargs(args, seed, y_train)}, fh)

    # Hard cap ABOVE PySR's own timeout_in_seconds.  PySR's timeout governs the search
    # loop only -- Julia compilation before it and the final constant optimisation after
    # it are both outside that clock, so a fit can legitimately overshoot. The grace is
    # what keeps a whole array task from being held hostage by one problem.
    hard_cap = args.time_limit * 1.5 + args.startup_grace if args.time_limit > 0 else None

    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--_worker",
           "--config", cfg_path]
    log_fh = open(os.path.join(workdir, "pysr.log"), "w") if args.quiet else None
    t0 = time.perf_counter()
    status = "ok"
    # start_new_session: kill the whole process group on timeout, so no Julia child
    # survives to keep burning a core after we have given up on it.
    proc = subprocess.Popen(cmd, cwd=workdir, start_new_session=True,
                            stdout=log_fh, stderr=subprocess.STDOUT if log_fh else None)
    try:
        rc = proc.wait(timeout=hard_cap)
        if rc != 0:
            status = f"exit{rc}"
    except subprocess.TimeoutExpired:
        status = "timeout"
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        proc.wait()
    finally:
        if log_fh is not None:
            log_fh.close()
    elapsed = time.perf_counter() - t0

    best = {}
    if os.path.exists(res_path):
        try:
            with open(res_path) as fh:
                best = json.load(fh)
        except Exception:
            pass
    if not best.get("expression") and status == "ok":
        status = "no_solution"
    return best, elapsed, status


def _save_worker_log_tail(workdir, out_dir, name, status, tail_bytes=64 * 1024):
    """Keep the tail of a failed fit's log in the RESULTS tree.  On a cluster the
    workdir is node-local scratch that vanishes with the allocation, so keeping the
    workdir is not enough to diagnose anything after the fact."""
    src = os.path.join(workdir, "pysr.log")
    if not os.path.exists(src):
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(src, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size > tail_bytes:
                fh.seek(size - tail_bytes)
            tail = fh.read()
        safe = str(name).replace("/", "_")
        with open(os.path.join(out_dir, f"{safe}.{status}.log"), "wb") as out:
            out.write(f"# {name}  status={status}  (tail of {size} bytes)\n".encode())
            out.write(tail)
    except Exception as e:
        print(f"  [warn] could not save worker log for {name}: {e}", flush=True)


# -- SRBench (Feynman) --------------------------------------------------------

def run_srbench(args, tau, shard_tag):
    """Mirrors eval_e2e.evaluate_equation's data handling exactly: n_points subsample,
    75/25 split, additive-RMS noise on TRAIN targets only (test/OOD stay clean)."""
    import pandas as pd
    from results_io import srbench_path, load_results, save_results

    name = f"results_feynman_pysr{shard_tag}.pkl.gz"
    out_path = srbench_path(args.results_dir, tau, name)

    datasets_dir = os.path.join(args.pmlb_dir, "datasets")
    dataset_names = sorted(d for d in os.listdir(datasets_dir)
                           if d.startswith("feynman_")
                           and os.path.isdir(os.path.join(datasets_dir, d)))
    if args.dataset:
        wanted = [d.strip() for d in args.dataset.split(",") if d.strip()]
        missing = [d for d in wanted if d not in dataset_names]
        if missing:
            sys.exit(f"--dataset: unknown dataset(s) {missing}")
        dataset_names = [d for d in dataset_names if d in wanted]
    if args.max_problems:
        dataset_names = dataset_names[:args.max_problems]
    all_names = list(dataset_names)
    if args.num_shards > 1:
        dataset_names = [d for k, d in enumerate(dataset_names)
                         if k % args.num_shards == args.shard_id]

    rows, done = _resume(load_results(out_path)[1], "dataset", args.retry_nulls,
                         "predicted_formula", out_path)

    scratch = _scratch_dir(args, f"srbench_seed{args.seed}_noise{tau:g}")
    diag = os.path.join(os.path.dirname(out_path), "pysr_diagnostics")

    def _save():
        save_results(out_path, rows, model_name="pysr", target_noise=tau,
                     hyperparams=_hparams(args))

    _save()
    for i, dname in enumerate(dataset_names):
        if (dname, args.seed) in done:
            continue
        # Index into the FULL list, not the shard, so a dataset's RNG stream does not
        # change when --num-shards does.  Otherwise re-running a cell at a different
        # shard count would silently produce different data.
        gi = all_names.index(dname)
        rng = np.random.RandomState(args.seed)

        try:
            df = pd.read_csv(os.path.join(datasets_dir, dname, f"{dname}.tsv.gz"),
                             sep="\t", compression="gzip")
        except Exception as e:
            rows.append({"dataset": dname, "predicted_formula": None, "r2": 0.0,
                         "ood_r2": None, "accuracy": 0, "seed": args.seed,
                         "predict_time_s": None, "status": "load_error", "error": str(e)})
            _save()
            continue

        if len(df) > args.n_points:
            df = df.iloc[rng.choice(len(df), args.n_points, replace=False)].reset_index(drop=True)
        feat = [c for c in df.columns if c != "target"]
        X = df[feat].values.astype(np.float64)
        y = df["target"].values.astype(np.float64)
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            rows.append({"dataset": dname, "predicted_formula": None, "r2": 0.0,
                         "ood_r2": None, "accuracy": 0, "seed": args.seed,
                         "predict_time_s": None, "status": "nonfinite"})
            _save()
            continue

        n_train = int(0.75 * len(df))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        if tau > 0.0 and len(y_train):
            _rms = float(np.sqrt(np.mean(np.square(y_train))))
            if _rms > 0.0:
                y_train = y_train + rng.normal(0.0, tau * _rms, size=len(y_train))
        if args.max_train_points and len(X_train) > args.max_train_points:
            X_train, y_train = X_train[:args.max_train_points], y_train[:args.max_train_points]

        wd = os.path.join(scratch, dname)
        best, elapsed, status = solve_one(X_train, y_train, wd, args, args.seed + gi)
        expr = normalize_expr(best.get("expression"))
        r2 = _r2(predict_expr(expr, X_test), y_test)

        rows.append({
            "dataset": dname,
            "predicted_formula": expr,
            "r2": r2,
            "ood_r2": None,          # filled post-hoc by the augment_*_caches.py sweep
            "accuracy": int(r2 >= args.r2_threshold),
            "seed": args.seed,
            "predict_time_s": elapsed,
            "search_time": elapsed,
            "complexity": best.get("complexity"),
            "train_loss": best.get("loss"),
            "status": status,
        })
        _save()
        if status != "ok":
            _save_worker_log_tail(wd, diag, dname, status)
        if not args.keep_workdir and not status.startswith("exit"):
            shutil.rmtree(wd, ignore_errors=True)
        print(f"[srbench {i+1}/{len(dataset_names)}] {dname:<28} R^2={r2:.4f} "
              f"t={elapsed:.0f}s {status}  {str(expr)[:60]}", flush=True)

    return out_path, rows


# -- LLM-SRBench --------------------------------------------------------------

def run_llmsrbench(args, tau, shard_tag):
    """Mirrors eval_aifeynman's LLM-SRBench handling: col 0 is y, per-seed row
    permutation, additive-RMS noise on TRAIN only, ood_test scored from the hdf5."""
    from results_io import llmsr_path, load_results, save_results
    from eval_mymodels import load_problems, output_metrics

    method = f"pysr_{args.split}{shard_tag}"
    out_path = llmsr_path(args.results_dir, tau, method)

    problems = load_problems(args.split)
    if args.problem:
        problems = [q for q in problems if q["name"] == args.problem]
    if args.max_problems:
        problems = problems[:args.max_problems]
    all_names = [q["name"] for q in problems]
    if args.num_shards > 1:
        problems = [q for k, q in enumerate(problems)
                    if k % args.num_shards == args.shard_id]

    rows, done = _resume(load_results(out_path)[1], "equation_id", args.retry_nulls,
                         "discovered_equation", out_path)

    scratch = _scratch_dir(args, f"llmsr_{args.split}_seed{args.seed}_noise{tau:g}")
    diag = os.path.join(os.path.dirname(os.path.dirname(out_path)), "pysr_diagnostics")

    def _save():
        save_results(out_path, rows, model_name="pysr", split=args.split, method=method,
                     target_noise=tau, hyperparams=_hparams(args))

    _save()
    for i, q in enumerate(problems):
        if (q["name"], args.seed) in done:
            continue
        gi = all_names.index(q["name"])
        train, test = q["train"], q["test"]
        X_train, y_train = train[:, 1:], train[:, 0]
        X_test, y_test = test[:, 1:], test[:, 0]

        rng = np.random.default_rng([args.seed, gi])
        perm = rng.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        if tau > 0.0 and len(y_train):
            _rms = float(np.sqrt(np.mean(np.square(y_train))))
            if _rms > 0.0:
                y_train = y_train + np.random.default_rng(
                    [args.seed, gi, 1]).normal(0.0, tau * _rms, size=len(y_train))
        if args.max_train_points and len(X_train) > args.max_train_points:
            X_train, y_train = X_train[:args.max_train_points], y_train[:args.max_train_points]

        wd = os.path.join(scratch, q["name"].replace("/", "_"))
        best, elapsed, status = solve_one(X_train, y_train, wd, args, args.seed + gi)
        expr = normalize_expr(best.get("expression"))

        id_m = output_metrics(predict_expr(expr, X_test), y_test)
        ood_m = None
        if q["ood_test"] is not None:
            ood_m = output_metrics(predict_expr(expr, q["ood_test"][:, 1:]),
                                   q["ood_test"][:, 0])

        rows.append({
            "equation_id": q["name"],
            "gt_equation": q["expression"],
            "discovered_equation": expr,
            "n_vars": int(train.shape[1] - 1),
            "num_datapoints": int(len(X_train)),
            "num_eval_datapoints": int(len(test)),
            "search_time": elapsed,
            "seed": args.seed,
            "id_metrics": id_m,
            "ood_metrics": ood_m,
            "status": status,
            "complexity": best.get("complexity"),
            "train_loss": best.get("loss"),
        })
        _save()
        if status != "ok":
            _save_worker_log_tail(wd, diag, q["name"], status)
        if not args.keep_workdir and not status.startswith("exit"):
            shutil.rmtree(wd, ignore_errors=True)
        print(f"[llmsr {i+1}/{len(problems)}] {q['name']:<28} R^2={id_m['r2']:.4f} "
              f"t={elapsed:.0f}s {status}  {str(expr)[:60]}", flush=True)

    return out_path, rows


# -- shared helpers -----------------------------------------------------------

def _r2(y_pred, y):
    mask = np.isfinite(y_pred) & np.isfinite(y)
    if not mask.any():
        return 0.0
    yp, yt = y_pred[mask], y[mask]
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - float(np.sum((yt - yp) ** 2)) / ss_tot)


def _resume(prior, key, retry_nulls, formula_key, path):
    """Shared resume policy (eval_aifeynman's, verbatim in spirit).

    A row with no formula is a REAL outcome -- PySR searched its whole budget and came
    back empty -- so by default it counts as done.  --retry-nulls drops those, which is
    what you want after CHANGING the budget and only then.
    """
    rows, done, n_null = [], set(), 0
    for r in prior:
        k = (r.get(key), r.get("seed"))
        if k in done:
            continue
        if r.get(formula_key) in (None, "None", ""):
            n_null += 1
            if retry_nulls:
                continue
        done.add(k)
        rows.append(r)
    if done or n_null:
        fate = "dropped for retry" if retry_nulls else "kept (use --retry-nulls to redo)"
        print(f"Resuming: {len(done)} rows kept, {n_null} null {fate} -- {path}",
              flush=True)
    return rows, done


def _scratch_dir(args, tag):
    root = args.workdir or os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
    d = os.path.join(root, f"pysr_{tag}")
    os.makedirs(d, exist_ok=True)
    return d


def _hparams(args):
    return {"time_limit": args.time_limit, "early_stop": args.early_stop,
            "niterations": args.niterations, "maxsize": args.maxsize,
            "maxdepth": args.maxdepth, "n_points": args.n_points,
            "max_train_points": args.max_train_points,
            "binary_operators": SRBENCH_BINARY_OPS,
            "unary_operators": SRBENCH_UNARY_OPS,
            "config_source": "srbench/experiment/methods/pysr/regressor.py",
            "deviations": ["timeout ceiling instead of srbench's 36000s override",
                           "early_stop_condition set (srbench sets none)",
                           "turbo=False (srbench True; deprecated + nondeterministic)"]}


def _merge(args, tau):
    """Fold every shard artifact for (seed, noise) into the canonical per-seed files."""
    from results_io import (srbench_path, llmsr_path, llmsr_dir, load_results,
                            save_results)
    n = args.num_shards
    if args.benchmark in ("srbench", "both"):
        merged = {}
        found = 0
        for i in range(n):
            p = srbench_path(args.results_dir, tau,
                             f"results_feynman_pysr__shard{i:02d}of{n:02d}.pkl.gz")
            srows = load_results(p)[1]
            found += bool(srows)
            for r in srows:
                k = (r.get("dataset"), r.get("seed"))
                prev = merged.get(k)
                if prev is None or prev.get("predicted_formula") in (None, "None", ""):
                    merged[k] = r
        canon = srbench_path(args.results_dir, tau, "results_feynman_pysr.pkl.gz")
        save_results(canon, list(merged.values()), model_name="pysr", target_noise=tau,
                     hyperparams=_hparams(args))
        print(f"[merge] srbench: {found}/{n} shards -> {len(merged)} rows -> {canon}")
        for i in range(n):
            p = srbench_path(args.results_dir, tau,
                             f"results_feynman_pysr__shard{i:02d}of{n:02d}.pkl.gz")
            if os.path.exists(p):
                os.remove(p)

    if args.benchmark in ("llmsrbench", "both"):
        merged = {}
        found = 0
        for i in range(n):
            m = f"pysr_{args.split}__shard{i:02d}of{n:02d}"
            srows = load_results(llmsr_path(args.results_dir, tau, m))[1]
            found += bool(srows)
            for r in srows:
                k = (r.get("equation_id"), r.get("seed"))
                prev = merged.get(k)
                if prev is None or prev.get("discovered_equation") in (None, "None", ""):
                    merged[k] = r
        canon = llmsr_path(args.results_dir, tau, f"pysr_{args.split}")
        save_results(canon, list(merged.values()), model_name="pysr", split=args.split,
                     method=f"pysr_{args.split}", target_noise=tau,
                     hyperparams=_hparams(args))
        print(f"[merge] llmsr: {found}/{n} shards -> {len(merged)} rows -> {canon}")
        for i in range(n):
            d = llmsr_dir(args.results_dir, tau,
                          f"pysr_{args.split}__shard{i:02d}of{n:02d}")
            if os.path.isdir(d):
                shutil.rmtree(d)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="both",
                   choices=["srbench", "llmsrbench", "both"])
    p.add_argument("--split", default="lsr_transform")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-noise", "--target_noise", dest="target_noise",
                   type=float, default=0.0, metavar="TAU",
                   help="Additive-RMS noise on TRAIN targets only (SRBench convention); "
                        "selects the noise_<TAU>/ subdir.")
    p.add_argument("--results-dir", "--output", dest="results_dir",
                   default="results/pysr",
                   help="Seed root; see results_io for the layout beneath it.")
    p.add_argument("--pmlb-dir", dest="pmlb_dir",
                   default=os.path.join(REPO_ROOT, "datasets", "pmlb"))
    p.add_argument("--dataset", default=None, help="[srbench] comma-separated subset")
    p.add_argument("--problem", default=None, help="[llmsrbench] single problem")
    p.add_argument("--max-problems", dest="max_problems", type=int, default=None)
    p.add_argument("--n-points", dest="n_points", type=int, default=20000,
                   help="[srbench] rows subsampled from the 100k PMLB file (75/25 split). "
                        "Matches eval_e2e.py's default.")
    p.add_argument("--max-train-points", dest="max_train_points", type=int, default=0,
                   help="Cap on TRAIN rows handed to PySR (0 = all). Independent of "
                        "--n-points, which sizes the split itself.")
    p.add_argument("--r2-threshold", dest="r2_threshold", type=float, default=0.99)
    p.add_argument("--num-shards", dest="num_shards", type=int, default=1)
    p.add_argument("--shard-id", dest="shard_id", type=int, default=0)
    p.add_argument("--merge-shards", dest="merge_shards", action="store_true")
    p.add_argument("--retry-nulls", dest="retry_nulls", action="store_true")
    # -- budget --------------------------------------------------------------
    p.add_argument("--time-limit", dest="time_limit", type=float, default=3600.0,
                   help="Per-fit ceiling in seconds (PySR timeout_in_seconds). SRBench "
                        "effectively uses 36000; see the module docstring for why this "
                        "defaults to 3600 instead.")
    p.add_argument("--early-stop", dest="early_stop", default="1e-10",
                   help="PySR early_stop_condition: a LOSS THRESHOLD (e.g. 1e-10) or a "
                        "full Julia definition ('f(loss, complexity) = ...'). A bare "
                        "predicate like 'loss < 1e-10' is NOT valid -- PySR seval's this "
                        "verbatim. SRBench sets none, so PySR there burns the full budget "
                        "even on F=ma; pass '' to reproduce that. A numeric threshold is "
                        "RELATIVE to mean(y_train^2) unless --early-stop-absolute.")
    p.add_argument("--early-stop-absolute", dest="early_stop_absolute",
                   action="store_true",
                   help="Treat --early-stop as a raw MSE instead of scaling it by "
                        "mean(y_train^2). Rarely what you want: target magnitudes here "
                        "span ~14 orders of magnitude, so one absolute threshold is a "
                        "different criterion on every problem (see early_stop_for).")
    p.add_argument("--startup-grace", dest="startup_grace", type=float, default=600.0,
                   help="Seconds added on top of 1.5*--time-limit for the hard "
                        "subprocess kill: Julia compile and final constant optimisation "
                        "both sit OUTSIDE PySR's own timeout.")
    p.add_argument("--niterations", type=int, default=1_000_000_000)
    p.add_argument("--maxsize", type=int, default=30)
    p.add_argument("--maxdepth", type=int, default=20)
    p.add_argument("--workdir", default=None)
    p.add_argument("--keep-workdir", dest="keep_workdir", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args, rest = p.parse_known_args()

    if args._worker:
        return _worker_main(rest)

    sys.path.insert(0, REPO_ROOT)
    tau = args.target_noise or 0.0

    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard-id must be in [0, {args.num_shards})")

    if args.merge_shards:
        _merge(args, tau)
        return 0

    shard_tag = ("" if args.num_shards == 1
                 else f"__shard{args.shard_id:02d}of{args.num_shards:02d}")

    print(f"PySR | benchmark={args.benchmark} seed={args.seed} noise={tau:g} "
          f"shard={args.shard_id}/{args.num_shards} "
          f"budget={args.time_limit:.0f}s early_stop={args.early_stop!r}", flush=True)

    if args.benchmark in ("srbench", "both"):
        path, rows = run_srbench(args, tau, shard_tag)
        r2 = np.array([r.get("r2", np.nan) for r in rows], dtype=np.float64)
        r2 = r2[np.isfinite(r2)]
        if r2.size:
            print(f"\n[srbench] {len(rows)} rows | acc(R^2>={args.r2_threshold})="
                  f"{100*np.mean(r2 >= args.r2_threshold):.1f}% | mean R^2={r2.mean():.4f}")
        print(f"[srbench] -> {path}\n", flush=True)

    if args.benchmark in ("llmsrbench", "both"):
        path, rows = run_llmsrbench(args, tau, shard_tag)
        r2 = np.array([(r.get("id_metrics") or {}).get("r2", np.nan) for r in rows],
                      dtype=np.float64)
        r2 = r2[np.isfinite(r2)]
        if r2.size:
            print(f"\n[llmsr] {len(rows)} rows | acc(R^2>=0.99)="
                  f"{100*np.mean(r2 >= 0.99):.1f}% | mean R^2={r2.mean():.4f}")
        print(f"[llmsr] -> {path}\n", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
