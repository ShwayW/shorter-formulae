#!/usr/bin/env python3
"""
eval_phye2e.py -- Evaluate PhyE2E (Ying et al., "A Neural Symbolic Model for Space
Physics", Nature Machine Intelligence 2025) on either benchmark, with BEAM SEARCH
decoding.  Pick with --benchmark:

  srbench     (default)  the 119 Feynman formulae from the local PMLB mirror.
                         Writes <results-dir>/noise_<TAU>/results_feynman_phye2e.pkl.gz.

  llmsrbench             LLM-SRBench, scored with the same loader + metrics as every
                         other method.  Writes <results-dir>/noise_<TAU>/llmsrbench/
                         phye2e_<split>/results.pkl.gz.

The model is the authors' 6M-formula pretrained checkpoint (374 MB), downloaded from
their Google Drive to weights/phye2e/phye2e_model.pt.  The vendored code is PhysicsRegression/.

DECODING -- read this before comparing numbers
----------------------------------------------
PhyE2E ships three decode paths, and only the first two were reachable upstream:

  greedy      one candidate per bag (`beam_type="sampling"`, `beam_size=1`)  <- the
              checkpoint's own saved default
  sampling    `beam_size` sampled candidates per bag at `beam_temperature`
  search      TRUE beam search -- upstream `ModelWrapper.forward` raised
              NotImplementedError for this branch

`--decode search` (the default here, since beam search is what this harness is for)
runs the beam branch we enabled in PhysicsRegression/symbolicregression/model/
model_wrapper.py; see the PATCH comment there for why the upstream dead code could
not have worked (it predates the units-aware decoder and never filled `outputs`).
`--decode sampling` / `--decode greedy` reproduce the authors' shipped behaviour.

Candidates from all bags are pooled and BFGS-refined by their own
SymbolicTransformerRegressor, and the best tree is reported -- the same
bagging + refinement structure as eval_e2e.py, so the curves are comparable.

WHAT IS DELIBERATELY OFF
------------------------
* Divide-and-Conquer (the oracle network), MCTS and GP refinement are all OFF.  They
  are separate search procedures layered on top of the transformer; including them
  would not be a beam-search evaluation.  Turn them on with --use-divide/--use-mcts/
  --use-gp if you want the full published pipeline instead.
* Physical units are NOT supplied.  PhyE2E can take per-variable units as a hint, and
  Feynman has them -- but no other method here gets that information, so passing them
  would not be a like-for-like comparison.  The model runs with units unknown.
* `rescale` is False, forced by their own PhyReg wrapper.

USAGE
    python eval_phye2e.py --dataset feynman_I_6_2
    python eval_phye2e.py --benchmark llmsrbench --split lsr_transform
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from results_io import load_results, save_results, srbench_path, llmsr_path
from bench_cli import bench_defaults, check_bench_flags

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHYSREG_DIR = os.path.join(_HERE, "PhysicsRegression")
DEFAULT_WEIGHTS = os.path.join(_HERE, "weights", "phye2e", "phye2e_model.pt")

# Their operator spelling -> infix, straight from PhysicsRegression.express_best_gens.
_REPLACE_OPS = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/", "neg": "-"}

PhyReg = None   # bound by _setup_imports()


def _setup_imports():
    """Put PhysicsRegression/ first on sys.path, then import PhyReg.

    Deferred out of module scope on purpose: PhysicsRegression/ ships its own
    `symbolicregression`, `parsers` and `Oracle` packages that collide with the repo
    root's (and with TPSR/'s) -- whichever is imported first wins for the whole
    process.  Same hazard eval_e2e_tpsr.py handles for the vendored TPSR tree.
    """
    global PhyReg
    if PhyReg is not None:
        return
    for k in [k for k in sys.modules
              if k in ("symbolicregression", "parsers", "Oracle")
              or k.startswith(("symbolicregression.", "Oracle."))]:
        del sys.modules[k]
    sys.path = [_PHYSREG_DIR] + [p for p in sys.path if p != _PHYSREG_DIR]
    from PhysicsRegression import PhyReg as _PhyReg
    PhyReg = _PhyReg


# ---------------------------------------------------------------------------
# Model + inference
# ---------------------------------------------------------------------------

def load_model(args):
    """Load the pretrained checkpoint and set the decode mode."""
    _setup_imports()
    if not os.path.isfile(args.weights_path):
        raise FileNotFoundError(
            f"{args.weights_path} not found.  Download the authors' pretrained model "
            f"(374 MB) with:\n"
            f"    python -c \"import gdown; gdown.download(id='1szeTQmqSkl8DPBq2aXoHzEOusg-R1sx0', "
            f"output='{args.weights_path}')\"")
    print(f"Loading PhyE2E from {args.weights_path} ...", flush=True)
    phyreg = PhyReg(path=args.weights_path, device=args.device)

    # The checkpoint saves beam_type="sampling", beam_size=1 (i.e. greedy).  Both the
    # params object and the already-built ModelWrapper have to be updated: the wrapper
    # copied these at construction time.
    beam_type = "sampling" if args.decode in ("greedy", "sampling") else "search"
    beam_size = 1 if args.decode == "greedy" else args.beam_size
    phyreg.params.beam_type = beam_type
    phyreg.params.beam_size = beam_size
    phyreg.mw.beam_type = beam_type
    phyreg.mw.beam_size = beam_size
    phyreg.params.max_input_points = args.max_input_points
    phyreg.dstr.max_input_points = args.max_input_points
    phyreg.dstr.max_number_bags = args.max_number_bags
    phyreg.dstr.n_trees_to_refine = args.n_trees_to_refine
    phyreg.mw.max_forward_batch = args.max_forward_batch or None
    print(f"Model loaded.  decode={args.decode} (beam_type={beam_type}, "
          f"beam_size={beam_size}), bags<={args.max_number_bags} x "
          f"{args.max_input_points} pts, fwd-batch={args.max_forward_batch}, "
          f"BFGS top-{args.n_trees_to_refine}\n", flush=True)
    return phyreg


# ---------------------------------------------------------------------------
# Physical units (SRBench / Feynman only)
# ---------------------------------------------------------------------------
# PhyE2E can take each variable's physical dimensions as a hint -- the mechanism the
# paper is built around.  Feynman ships them, so `--units feynman` supplies them and
# `--units none` (the default) does not, which is the ablation the two arms measure.
#
# NOT a neutral ablation, and worth knowing: their encode_units(None) returns the zero
# vector, which is the encoding of "kg0m0s0T0V0" = DIMENSIONLESS.  There is no
# "unknown" symbol on this path.  So --units none does not withhold information, it
# asserts that every variable is dimensionless.  That is what every other method here
# effectively sees, so it stays the default for like-for-like comparison, but it means
# the no-units arm is handicapped rather than merely uninformed.
_UNITS_CSV = os.path.join(_PHYSREG_DIR, "data", "units.csv")
_FEYNMAN_CSV = os.path.join(_HERE, "datasets", "feynman", "FeynmanEquations.csv")


def _unit_string(kg, m, s, T, V) -> str:
    """(kg, m, s, T, V) exponents -> the "kg1m1s-2T0V0" form encode_units() parses."""
    def _f(v):
        v = float(v)
        return str(int(v)) if v == int(v) else str(v)
    return f"kg{_f(kg)}m{_f(m)}s{_f(s)}T{_f(T)}V{_f(V)}"


def load_feynman_units():
    """{dataset: [unit_per_input_col..., output_unit]} for the Feynman datasets.

    Order matters twice over:
      * the list is [inputs in COLUMN order, then the OUTPUT last] -- ModelWrapper reads
        `un[:-1]` as the inputs, and fit() asserts len(units) == n_vars + 1;
      * units.csv's columns are (m, s, kg, T, V) but the model's string form is
        kg,m,s,T,V -- so it is read BY NAME.  Reading it positionally would silently
        swap metres for kilograms and hand the model confident nonsense.

    Column order is safe to rely on because get_top_k_features() returns range(n) when
    n_vars <= max_input_dimension (10) and Feynman has at most 9 -- if that ever stops
    holding, the selected columns would permute while these units did not.

    Datasets without a complete set (the 20 black-box feynman_test_* dirs, which have no
    FeynmanEquations.csv row) are omitted, and the caller passes units=None for them.
    """
    import pandas as pd

    table = _units_table()

    meta = pd.read_csv(_FEYNMAN_CSV)
    meta.columns = [c.strip().lstrip("﻿") for c in meta.columns]
    output_of = {str(r["Filename"]).strip(): str(r["Output"]).strip()
                 for _, r in meta.iterrows() if pd.notna(r.get("Filename"))}

    from eval_e2e import list_feynman_datasets, load_pmlb_dataset
    out = {}
    for name in list_feynman_datasets(os.path.join(_HERE, "datasets", "pmlb")):
        key = name.replace("feynman_", "").replace("_", ".")
        o = output_of.get(key)
        if o is None or o not in table:
            continue
        try:
            cols = list(load_pmlb_dataset(
                os.path.join(_HERE, "datasets", "pmlb"), name).columns)
        except Exception:
            continue
        ins = [c for c in cols if c != "target"]
        if any(c not in table for c in ins):
            continue
        out[name] = [table[c] for c in ins] + [table[o]]
    return out


def _units_table():
    """{variable name: unit string} from PhysicsRegression/data/units.csv (by NAME)."""
    import pandas as pd
    u = pd.read_csv(_UNITS_CSV)
    u.columns = [c.strip().lstrip("\ufeff") for c in u.columns]
    out = {}
    for _, r in u.iterrows():
        v = str(r.get("Variable", "")).strip()
        if v and not pd.isna(r.get("m")):
            out[v] = _unit_string(r["kg"], r["m"], r["s"], r["T"], r["V"])
    return out


def llmsr_units(problem, table):
    """[input units..., output unit] for one LLM-SRBench problem, or None.

    lsr_transform is FEYNMAN-DERIVED -- its problems are named after Feynman equations
    (II.6.15b_1_0) and `symbols` are Feynman variable names, so units.csv resolves all
    111 of them.  symbol_properties[0] == 'O': the FIRST symbol is the output and the
    rest are the inputs in X-column order (hdf5 column 0 is y, columns 1: are X in
    `symbols` order).  PhyE2E wants the output LAST, so it is moved to the end.

    Returns None if any symbol is unknown, so that problem falls back to no units
    rather than being told a wrong dimension.
    """
    syms = [str(x) for x in problem.get("symbols", [])]
    if len(syms) < 2 or any(x not in table for x in syms):
        return None
    return [table[x] for x in syms[1:]] + [table[syms[0]]]


def to_infix(tree) -> "str | None":
    """Their predicted_tree -> an infix string in x_0..x_N (the e2e convention)."""
    if tree is None:
        return None
    expr = str(tree)
    for op, sym in _REPLACE_OPS.items():
        expr = expr.replace(op, sym)
    return expr


def fit_one(phyreg, X_train, y_train, args, units=None) -> "str | None":
    """Run one PhyE2E fit and return the best formula as infix, or None.

    `units` is [input units..., output unit] or None (see load_feynman_units)."""
    import torch
    try:
        phyreg.fit(X_train, y_train,
                   units=units,
                   use_Divide=args.use_divide,
                   use_MCTS=args.use_mcts,
                   use_GP=args.use_gp,
                   use_const_optimization=args.use_const_optimization,
                   save_oracle_model=False,
                   verbose=False)
    except torch.cuda.OutOfMemoryError:
        # Never swallow a CUDA OOM into a None "result": that is how an entire e2e
        # sweep once turned into all-None garbage that still exited rc=0 (see
        # eval_e2e.run_inference).  Crash instead, so the row is retried rather than
        # recorded as a genuine R^2=0.  Lower --max-forward-batch or --max-number-bags.
        torch.cuda.empty_cache()
        raise
    except Exception as exc:
        print(f"    [fit failed] {type(exc).__name__}: {exc}", flush=True)
        return None
    gens = getattr(phyreg, "best_gens", None)
    if not gens:
        return None
    return to_infix(gens[0].get("predicted_tree"))


def predict(expr: "str | None", X: np.ndarray) -> np.ndarray:
    """Evaluate an infix x_0..x_N formula on X (N, D); NaN on any failure."""
    n = X.shape[0]
    if not expr:
        return np.full(n, np.nan)
    try:
        import sympy as sp
        syms = [sp.Symbol(f"x_{i}") for i in range(X.shape[1])]
        f = sp.lambdify(syms, expr, ["numpy", {"sigmoid": lambda z: 1 / (1 + np.exp(-z))}])
        with np.errstate(all="ignore"):
            out = f(*X.astype(np.float64).T)
        out = np.asarray(out, dtype=np.float64).reshape(-1)
        if out.size == 1 and n > 1:          # a constant formula broadcasts
            out = np.full(n, float(out[0]))
        return out if out.size == n else np.full(n, np.nan)
    except Exception:
        return np.full(n, np.nan)


def clipped_r2(expr, X, y) -> float:
    """R^2 clipped to [0, 1], 0.0 on any failure -- matches eval_e2e.predict_r2."""
    yp = predict(expr, X)
    if not np.all(np.isfinite(yp)):
        return 0.0
    try:
        from sklearn.metrics import r2_score
        return float(max(0.0, r2_score(y, yp)))
    except Exception:
        return 0.0


def add_target_noise(y_train, tau, rng):
    """y_train += N(0, tau*sqrt(mean(y^2))) -- SRBench/TPSR convention, TRAIN only."""
    if not tau or tau <= 0.0 or len(y_train) == 0:
        return y_train
    rms = float(np.sqrt(np.mean(np.square(np.asarray(y_train, dtype=np.float64)))))
    if rms <= 0.0:
        return y_train
    return y_train + rng.normal(0.0, tau * rms, size=y_train.shape)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
_BENCH_DEFAULTS = {"srbench": {"seed": 42}, "llmsrbench": {"seed": 0}}
_BENCH_ONLY_FLAGS = {
    "srbench": ["pmlb_dir", "dataset", "n_points", "r2_threshold"],
    "llmsrbench": ["split", "problem", "max_problems"],
}


def main():
    p = argparse.ArgumentParser(
        description=("Evaluate PhyE2E (Nature MI 2025) with beam search on SRBench "
                     "or LLM-SRBench. Pick with --benchmark."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="srbench", choices=["srbench", "llmsrbench"],
                   help="Which benchmark to run. Default: srbench.")
    p.add_argument("--weights-path", default=DEFAULT_WEIGHTS,
                   help="The authors' pretrained checkpoint. Default: weights/phye2e/phye2e_model.pt.")
    p.add_argument("--device", default=None,
                   help="cuda:0 / cpu. Default: cuda:0 when available.")

    # decoding
    p.add_argument("--decode", default="search", choices=["search", "sampling", "greedy"],
                   help="search = TRUE beam search (default, the point of this script); "
                        "sampling = their multi-candidate sampler; greedy = the "
                        "checkpoint's own saved default (1 candidate per bag).")
    p.add_argument("--beam-size", type=int, default=10,
                   help="Beam width (or #samples for --decode sampling). Matches "
                        "eval_e2e.py / eval_mymodels.py. Default: 10.")
    p.add_argument("--max-input-points", type=int, default=200,
                   help="Bag size: rows per forward pass. Default: 200.")
    p.add_argument("--max-number-bags", type=int, default=100,
                   help="Max bags pooled per problem (the e2e bagging trick). Default: 100.")
    p.add_argument("--n-trees-to-refine", type=int, default=10,
                   help="Top-K unique skeletons passed to BFGS. Default: 10.")
    p.add_argument("--max-forward-batch", type=int, default=10, metavar="N",
                   help="Bags decoded per GPU forward pass. Pure speed<->memory dial -- "
                        "results are identical. Upstream derives ~50, which OOMs a small "
                        "GPU under beam search. 0 = use their formula. Default: 10.")

    # the rest of the published pipeline -- off by default, see the module docstring
    p.add_argument("--use-divide", action="store_true", default=False,
                   help="Enable Divide-and-Conquer (trains an oracle net per problem). "
                        "OFF by default: not a beam-search evaluation.")
    p.add_argument("--use-mcts", action="store_true", default=False,
                   help="Enable MCTS refinement. OFF by default.")
    p.add_argument("--use-gp", action="store_true", default=False,
                   help="Enable genetic-programming refinement. OFF by default.")
    p.add_argument("--use-const-optimization", action="store_true", default=False,
                   help="Enable their extra constant optimisation pass. OFF by default.")

    # data / protocol
    p.add_argument("--fit-rows", "--fit_rows", dest="fit_rows", type=int, default=None,
                   metavar="N",
                   help="Cap the TRAIN rows handed to the model at N (both benchmarks). "
                        "The held-out test/OOD slices are untouched, so scoring stays "
                        "as precise as every other arm's. --fit-rows 200 reproduces the "
                        "authors' own evaluation budget (one 200-point bag per problem, "
                        "their evaluate.py protocol); default None = the full train "
                        "split, which is what the beam-search arms used.")
    p.add_argument("--pmlb-dir", "--pmlb_dir", dest="pmlb_dir", default="datasets/pmlb",
                   help="[srbench] Root of the local PMLB mirror.")
    p.add_argument("--dataset", default=None,
                   help="[srbench] Evaluate one dataset (or a comma-separated list).")
    p.add_argument("--n-points", "--n_points", dest="n_points", type=int, default=20000,
                   help="[srbench] Rows subsampled per equation before the 75/25 split.")
    p.add_argument("--r2-threshold", "--r2_threshold", dest="r2_threshold",
                   type=float, default=0.99,
                   help="[srbench] R^2 counting as solved. Default: 0.99.")
    p.add_argument("--units", choices=["none", "auto", "feynman"], default="none",
                   help="Supply each variable's physical units as a hint -- "
                        "the mechanism PhyE2E is built around. 'feynman' reads them "
                        "from PhysicsRegression/data/units.csv. srbench: covers 99 of "
                        "119 (the black-box feynman_test_* dirs fall back to none). "
                        "llmsrbench: lsr_transform is Feynman-derived, so all 111 "
                        "resolve. 'feynman' is a legacy alias for 'auto'. Default: "
                        "none, matching what every other method here sees -- NB that "
                        "encodes as all-dimensionless, not as unknown.")
    p.add_argument("--split", default="lsr_transform",
                   help="[llmsrbench] lsr_transform or lsr_synth_<domain>.")
    p.add_argument("--problem", default=None,
                   help="[llmsrbench] Evaluate only this problem (by name).")
    p.add_argument("--max-problems", "--max_problems", dest="max_problems",
                   type=int, default=None,
                   help="[llmsrbench] Cap the number of problems (smoke tests).")

    p.add_argument("--results-dir", "--results_dir", "--output", dest="results_dir",
                   default="./results",
                   help="Seed root; the noise_<TAU>/ subdir is composed automatically.")
    p.add_argument("--seed", type=int, default=None,
                   help="Base seed. Default: 42 for srbench, 0 for llmsrbench.")
    p.add_argument("--n-seeds", "--n_seeds", dest="n_seeds", type=int, default=1,
                   help="Seeds scored in ONE process (one checkpoint load).")
    p.add_argument("--target-noise", "--noise", dest="target_noise", type=float,
                   default=0.0, metavar="TAU",
                   help="Additive-RMS noise on the TRAIN targets only: "
                        "y_train += N(0, TAU*sqrt(mean(y^2))). Test/OOD stay clean.")

    args = p.parse_args()
    check_bench_flags(args, p, _BENCH_ONLY_FLAGS)
    bench_defaults(args, _BENCH_DEFAULTS, ("seed",))
    if args.device is None:
        import torch
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.benchmark == "llmsrbench":
        return run_llmsrbench(args)
    return run_srbench(args)


# ---------------------------------------------------------------------------
# SRBench (Feynman PMLB)
# ---------------------------------------------------------------------------
def run_srbench(args):
    """Writes <results-dir>/noise_<TAU>/results_feynman_phye2e.pkl.gz."""
    from eval_e2e import list_feynman_datasets, load_pmlb_dataset
    from gen_ood_data import load_feynman_pn_targets

    names = list_feynman_datasets(args.pmlb_dir)
    if args.dataset:
        wanted = [d.strip() for d in args.dataset.split(",") if d.strip()]
        missing = [d for d in wanted if d not in names]
        if missing:
            sys.exit(f"--dataset: unknown dataset(s) {missing}. "
                     f"{len(names)} available under {args.pmlb_dir}.")
        names = [d for d in names if d in wanted]
    print(f"Found {len(names)} Feynman datasets in {args.pmlb_dir}", flush=True)

    pn_targets = load_feynman_pn_targets()

    units_table = {}
    if args.units == "feynman":
        units_table = load_feynman_units()
        _n = sum(1 for n in names if n in units_table)
        print(f"[units] supplying physical units for {_n}/{len(names)} datasets in this "
              f"run (the rest fall back to none)", flush=True)

    phyreg = load_model(args)
    out_path = srbench_path(args.results_dir, args.target_noise, "results_feynman_phye2e.pkl.gz")

    # Resume: keep valid prior rows, redo the ones with no formula.
    rows, done = [], set()
    try:
        prior = load_results(out_path)[1]
        rows = [r for r in prior
                if str(r.get("predicted_formula") or "").strip() not in ("", "None", "nan")]
        done = {(r["dataset"], int(r["seed"])) for r in rows}
        if done:
            print(f"Resuming: {len(done)} valid (dataset, seed) rows in {out_path}", flush=True)
    except Exception:
        rows, done = [], set()

    def _save():
        save_results(out_path, rows, model_name="phye2e", target_noise=args.target_noise,
                     decode=args.decode, beam_size=args.beam_size, units=args.units,
                     fit_rows=args.fit_rows, use_divide=args.use_divide)

    _save()
    for seed in range(args.seed, args.seed + args.n_seeds):
        rng = np.random.RandomState(seed)
        print(f"\n--- Seed {seed} ---", flush=True)
        for i, name in enumerate(names):
            if (name, seed) in done:
                print(f"[{i+1}/{len(names)}] {name} ... skipped (already done)", flush=True)
                continue
            print(f"[{i+1}/{len(names)}] {name} ... ", end="", flush=True)
            try:
                df = load_pmlb_dataset(args.pmlb_dir, name)
            except Exception as exc:
                print(f"SKIP (load failed: {exc})", flush=True)
                continue
            if len(df) > args.n_points:
                idx = rng.choice(len(df), args.n_points, replace=False)
                df = df.iloc[idx].reset_index(drop=True)
            feats = [c for c in df.columns if c != "target"]
            X = df[feats].values.astype(np.float64)
            y = df["target"].values.astype(np.float64)
            if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
                print("SKIP (non-finite data)", flush=True)
                continue

            n_tr = int(0.75 * len(df))
            X_tr, X_te = X[:n_tr], X[n_tr:]
            y_tr, y_te = y[:n_tr], y[n_tr:]
            # Truncate BEFORE the noise draw, so tau scales against the RMS of the rows
            # the model actually sees -- what a natively 200-row run would do.
            if args.fit_rows:
                X_tr, y_tr = X_tr[:args.fit_rows], y_tr[:args.fit_rows]
            y_tr = add_target_noise(y_tr, args.target_noise, rng)

            t0 = time.perf_counter()
            formula = fit_one(phyreg, X_tr, y_tr, args, units=units_table.get(name))
            elapsed = time.perf_counter() - t0
            r2 = clipped_r2(formula, X_te, y_te)

            print(f"R^2={r2:.4f}  t={elapsed:.1f}s", flush=True)
            print(f"  pred:  {formula}", flush=True)
            if pn_targets.get(name):
                print(f"  GT:    {pn_targets[name]}", flush=True)

            rows.append({
                "dataset": name,
                "predicted_formula": formula,
                "r2": r2,
                "ood_r2": None,
                "accuracy": int(r2 >= args.r2_threshold),
                "seed": seed,
                "predict_time_s": elapsed,
            })
            done.add((name, seed))
            _save()

    _save()
    if rows:
        df = pd.DataFrame(rows)
        by_eq = df.groupby("dataset")
        print(f"\n=== PhyE2E ({args.decode}) -- {args.n_seeds} seed(s), "
              f"noise={args.target_noise} ===")
        print(f"  Mean R^2:                  {by_eq['r2'].mean().mean():.4f}")
        print(f"  Accuracy (R^2>={args.r2_threshold}):    "
              f"{by_eq['accuracy'].mean().mean():.4f} "
              f"({int(by_eq['accuracy'].mean().sum())}/{by_eq.ngroups} equations)")
    print(f"  Results written to: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# LLM-SRBench
# ---------------------------------------------------------------------------
def run_llmsrbench(args):
    """Writes <results-dir>/noise_<TAU>/llmsrbench/phye2e_<split>/results.pkl.gz.

    Loader and metrics come from eval_mymodels.py so PhyE2E is scored exactly like
    every other method on this benchmark (imported lazily -- an srbench run should
    not pay for loading that stack).
    """
    from eval_mymodels import load_problems, output_metrics

    print(f"Loading LLM-SRBench split '{args.split}' ...", flush=True)
    problems = load_problems(args.split)
    if args.problem is not None:
        problems = [q for q in problems if q["name"] == args.problem]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    print(f"  {len(problems)} problems.\n", flush=True)

    units_by_name = {}
    if args.units != "none":
        _tbl = _units_table()
        units_by_name = {q["name"]: llmsr_units(q, _tbl) for q in problems}
        _n = sum(1 for v in units_by_name.values() if v)
        print(f"[units] supplying physical units for {_n}/{len(problems)} problems "
              f"(the rest fall back to none)", flush=True)

    phyreg = load_model(args)
    method = f"phye2e_{args.split}"
    results_path = llmsr_path(args.results_dir, args.target_noise or 0.0, method)

    # Resume, keyed on (equation_id, seed); a row with no formula is redone.
    rows, done = [], set()
    for r in load_results(results_path)[1]:
        key = (r.get("equation_id"), r.get("seed"))
        if key in done or r.get("discovered_equation") in (None, "None", ""):
            continue
        done.add(key)
        rows.append(r)
    if done:
        print(f"Resuming: {len(done)} (problem, seed) pairs done in {results_path}", flush=True)

    def _save():
        save_results(results_path, rows, model_name="phye2e", split=args.split,
                     method=method, target_noise=args.target_noise or 0.0,
                     decode=args.decode, beam_size=args.beam_size, units=args.units,
                     fit_rows=args.fit_rows, use_divide=args.use_divide)

    _save()
    for seed in range(args.seed, args.seed + args.n_seeds):
        rng = np.random.default_rng(seed)
        for i, q in enumerate(problems):
            if (q["name"], seed) in done:
                continue
            train, test = q["train"], q["test"]
            n_vars = train.shape[1] - 1
            # Column 0 is y; columns 1: are X (LLM-SRBench convention).
            X_tr, y_tr = train[:, 1:], train[:, 0]
            X_te, y_te = test[:, 1:], test[:, 0]

            # Per-seed row permutation, so the consecutive-chunk bagging sees different
            # bags per seed -- the same device eval_e2e.py uses for its seed spread.
            perm = np.random.default_rng([seed, i]).permutation(len(X_tr))
            X_tr, y_tr = X_tr[perm], y_tr[perm]
            # Same cap as the SRBench path, applied after the permutation so each seed
            # draws a different 200 rows rather than the same prefix.
            if args.fit_rows:
                X_tr, y_tr = X_tr[:args.fit_rows], y_tr[:args.fit_rows]
            # Seed the noise stream from (seed, problem index) explicitly.  NOT hash(),
            # which Python salts per process -- that is what made eval_mymodels' per-
            # dataset RNG non-reproducible and confounded its paired A/Bs.
            y_tr = add_target_noise(y_tr, args.target_noise,
                                    np.random.default_rng([seed, i, 1]))

            t0 = time.perf_counter()
            formula = fit_one(phyreg, X_tr, y_tr, args,
                              units=units_by_name.get(q["name"]))
            elapsed = time.perf_counter() - t0

            id_m = output_metrics(predict(formula, X_te), y_te)
            ood_m = None
            if q["ood_test"] is not None:
                ood_m = output_metrics(predict(formula, q["ood_test"][:, 1:]),
                                       q["ood_test"][:, 0])

            rows.append({
                "equation_id": q["name"],
                "gt_equation": q["expression"],
                "discovered_equation": formula,
                "n_vars": n_vars,
                "num_datapoints": int(len(train)),
                "num_eval_datapoints": int(len(test)),
                "search_time": elapsed,
                "seed": seed,
                "id_metrics": id_m,
                "ood_metrics": ood_m,
            })
            done.add((q["name"], seed))
            _save()
            print(f"[seed {seed}] [{i+1}/{len(problems)}] {q['name']:<20} "
                  f"R^2={id_m['r2']:.4f}  t={elapsed:.1f}s  {formula}", flush=True)

    r2s = np.array([r["id_metrics"]["r2"] for r in rows], dtype=np.float64)
    r2s = r2s[np.isfinite(r2s)]
    print(f"\n{'='*70}")
    print(f"PhyE2E ({args.decode})  LLM-SRBench {args.split}  ({len(rows)} problems)")
    if r2s.size:
        print(f"  Acc (R^2 >= 0.99) : {100*np.mean(r2s >= 0.99):.1f}%")
        print(f"  Mean R^2          : {np.mean(r2s):.4f}   Median R^2: {np.median(r2s):.4f}")
    print(f"{'='*70}\n[results] {results_path}\n")


if __name__ == "__main__":
    main()
