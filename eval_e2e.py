#!/usr/bin/env python3
"""
eval_e2e.py -- Evaluate the E2E symbolic regression transformer (Kamienny et al.
2022) on either benchmark.  Pick with --benchmark:

  srbench     (default)  the 119 Feynman formulae from the local PMLB mirror
                         (datasets/pmlb/datasets/feynman_*/), the exact benchmark
                         used in the paper.  Writes <results-dir>/noise_<TAU>/
                         results_feynman_e2e.pkl.gz.

  llmsrbench             LLM-SRBench, in the SAME results.pkl.gz format as
                         eval_mymodels.py so compare_llmsrbench.py can include it
                         as one method.  Writes <results-dir>/noise_<TAU>/
                         llmsrbench/e2e_<split>/results.pkl.gz.

Both paths share the model + inference helpers below (load_raw_model /
build_model_wrapper / run_inference); the LLM-SRBench data loader and metrics are
imported from eval_mymodels.py so the two methods are scored identically.

Inference tricks (all four from Section 3 of the paper):
  - Scaling:    StandardScaler whitening of X before each forward pass;
                the best expression is rescaled back to original coordinates.
  - Bagging:    dataset split into bags of max_input_points rows; one
                forward pass per bag (model trained on <=200 pts/pass).
  - Decoding:   random sampling (beam_type='sampling') to draw beam_size
                structurally diverse skeleton candidates per bag.
  - Refinement: BFGS fine-tunes numerical constants in the top-K unique
                skeleton expressions selected across all bags.
"""

import argparse
import gzip
import os

from results_io import load_results, save_results, srbench_path, llmsr_path
from bench_cli import bench_defaults, check_bench_flags
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from symbolicregression.model.model_wrapper import ModelWrapper
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor
from gen_ood_data import load_feynman_pn_targets


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_raw_model(weights_path: str, device: str):
    raw = torch.load(weights_path, map_location=device, weights_only=False)
    raw.eval()
    return raw


def build_model_wrapper(
    raw_model, beam_type: str, beam_size: int, beam_temperature: float, max_forward_batch: int | None
) -> ModelWrapper:
    return ModelWrapper(
        env=raw_model.env,
        embedder=raw_model.embedder,
        encoder=raw_model.encoder,
        decoder=raw_model.decoder,
        beam_type=beam_type,
        beam_size=beam_size,
        beam_temperature=beam_temperature,
        beam_length_penalty=raw_model.beam_length_penalty,
        beam_early_stopping=raw_model.beam_early_stopping,
        max_generated_output_len=raw_model.max_generated_output_len,
        max_forward_batch=max_forward_batch,
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def list_feynman_datasets(pmlb_dir: str) -> list[str]:
    """Return sorted list of feynman_* dataset names under pmlb_dir/datasets/."""
    datasets_dir = os.path.join(pmlb_dir, "datasets")
    names = sorted(
        d for d in os.listdir(datasets_dir)
        if d.startswith("feynman_") and os.path.isdir(os.path.join(datasets_dir, d))
    )
    return names


def load_pmlb_dataset(pmlb_dir: str, name: str) -> pd.DataFrame:
    """Load a PMLB dataset from its local .tsv.gz file."""
    path = os.path.join(pmlb_dir, "datasets", name, f"{name}.tsv.gz")
    return pd.read_csv(path, sep="\t", compression="gzip")


# ---------------------------------------------------------------------------
# Inference with all four tricks
# ---------------------------------------------------------------------------

def run_inference(
    mw: ModelWrapper,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_input_points: int,
    max_number_bags: int,
    n_trees_to_refine: int,
) -> "SymbolicTransformerRegressor | None":
    """
    Fit SymbolicTransformerRegressor with:
      rescale=True      -> scaling trick (StandardScaler whitening + unscaling)
      max_input_points  -> bagging trick: rows per bag
      max_number_bags   -> bagging trick: max bags used
      n_trees_to_refine -> refinement trick: BFGS on top-K unique skeletons
    The decoding trick (random sampling) is encoded in the ModelWrapper via
    beam_type='sampling' and beam_size=K candidates per bag.
    """
    dstr = SymbolicTransformerRegressor(
        model=mw,
        max_input_points=max_input_points,
        max_number_bags=max_number_bags,
        n_trees_to_refine=n_trees_to_refine,
        rescale=True,
    )
    try:
        dstr.fit(X_train, y_train, verbose=False)
        return dstr
    except torch.cuda.OutOfMemoryError:
        # Do NOT swallow a CUDA OOM into a None "result" -- during the 2026-07-02
        # OOM cascade that silently turned entire e2e runs into all-None garbage
        # that still exited rc=0. Re-raise so the process crashes -> the parallel
        # scheduler's retry (and the GPU memory gate) handle it cleanly.
        raise
    except Exception:
        return None


def predict_r2(
    dstr: "SymbolicTransformerRegressor | None",
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Return clipped R^2 in [0, 1] on the test set, or 0.0 on any failure."""
    if dstr is None:
        return 0.0
    try:
        y_pred = dstr.predict(X_test)
        if y_pred is None:
            return 0.0
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if not np.all(np.isfinite(y_pred)):
            return 0.0
        return float(max(0.0, r2_score(y_test, y_pred)))
    except Exception:
        return 0.0


_REPLACE_OPS = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}


def get_predicted_formula(dstr: "SymbolicTransformerRegressor | None") -> "str | None":
    if dstr is None:
        return None
    try:
        info = dstr.retrieve_tree(with_infos=True)
        if info and info.get("predicted_tree") is not None:
            expr = info["predicted_tree"].infix()
            for op, sym in _REPLACE_OPS.items():
                expr = expr.replace(op, sym)
            return expr
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Per-equation evaluation
# ---------------------------------------------------------------------------

def evaluate_equation(
    dataset_name: str,
    pmlb_dir: str,
    mw: ModelWrapper,
    rng: np.random.RandomState,
    n_points: int,
    noise: float,
    max_input_points: int,
    max_number_bags: int,
    n_trees_to_refine: int,
    r2_threshold: float,
    seed: int,
    ood_data: tuple | None = None,
) -> dict:
    result = {
        "dataset": dataset_name,
        "predicted_formula": None,
        "r2": 0.0,
        "ood_r2": None,
        "accuracy": 0,
        "seed": seed,
        "predict_time_s": None,
    }

    try:
        df = load_pmlb_dataset(pmlb_dir, dataset_name)
    except Exception as e:
        result["error"] = str(e)
        return result

    # Randomly subsample n_points rows from the 100k available
    if len(df) > n_points:
        idx = rng.choice(len(df), n_points, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values.astype(np.float64)
    y = df["target"].values.astype(np.float64)

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        return result

    # 75 / 25 train-test split
    n_train = int(0.75 * len(df))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Additive-RMS target noise on the TRAINING targets only:
    #     y_train += N(0, noise * sqrt(mean(y_train^2)))
    # This replaces the previous *multiplicative* noise (y*(1+N(0,epsilon))) applied to
    # ALL of y (which also leaked into the test split).  The additive-RMS form,
    # train-only, matches eval_mymodels.py --target-noise, srbench evaluate_model.py,
    # TPSR/evaluate.py, and the ground-truth feather rows at target_noise==noise,
    # so the e2e curve is directly comparable to everything else on the plots.
    # The held-out test (and any OOD set) stay clean.
    if noise > 0.0 and len(y_train) > 0:
        _rms = float(np.sqrt(np.mean(np.square(y_train))))
        if _rms > 0.0:
            y_train = y_train + rng.normal(0.0, noise * _rms, size=len(y_train))

    t0 = time.perf_counter()
    dstr = run_inference(mw, X_train, y_train, max_input_points, max_number_bags, n_trees_to_refine)

    r2 = predict_r2(dstr, X_test, y_test)
    formula = get_predicted_formula(dstr)

    # OOD evaluation on held-out, far-from-training inputs.
    if ood_data is not None and dstr is not None:
        X_ood, y_ood = ood_data
        result["ood_r2"] = predict_r2(dstr, X_ood, y_ood)

    result["r2"] = r2
    result["accuracy"] = int(r2 >= r2_threshold)
    result["predicted_formula"] = formula
    result["predict_time_s"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Defaults that differ per benchmark (parsed as None so "unset" is detectable).
_BENCH_DEFAULTS = {"srbench": {"seed": 42}, "llmsrbench": {"seed": 0}}

# Flags only one benchmark's run_* function reads; passing one to the other
# benchmark is an error rather than a silent no-op.  See bench_cli.
_BENCH_ONLY_FLAGS = {
    "srbench": ["pmlb_dir", "dataset", "n_points", "r2_threshold", "ood_data"],
    "llmsrbench": ["split", "problem", "max_problems"],
}


def main():
    parser = argparse.ArgumentParser(
        description=("Evaluate the E2E symbolic regression transformer on SRBench "
                     "(119 Feynman formulae) or LLM-SRBench. Pick with --benchmark."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark", default="srbench", choices=["srbench", "llmsrbench"],
        help="Which benchmark to run. 'srbench' = Feynman PMLB datasets under "
             "--pmlb-dir; 'llmsrbench' = LLM-SRBench --split. Default: srbench.",
    )
    parser.add_argument("--weights-path", "--weights_path", dest="weights_path",
                        type=str, default="weights/e2e/e2e_model.pt")
    parser.add_argument("--pmlb-dir", "--pmlb_dir", dest="pmlb_dir",
                        type=str, default="datasets/pmlb",
                        help="[srbench] Root of local PMLB mirror containing datasets/feynman_*/.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="[srbench] Evaluate only this dataset (or a comma-separated "
                             "list), e.g. feynman_I_6_2. Default: all of them.")
    parser.add_argument("--split", type=str, default="lsr_transform",
                        help="[llmsrbench] Which split to evaluate: lsr_transform or "
                             "lsr_synth_<domain>. Default: lsr_transform.")
    parser.add_argument("--problem", type=str, default=None,
                        help="[llmsrbench] Evaluate only this problem (by name).")
    parser.add_argument("--max-problems", "--max_problems", dest="max_problems",
                        type=int, default=None,
                        help="[llmsrbench] Cap the number of problems (smoke tests).")
    parser.add_argument(
        "--results-dir", "--results_dir", "--output", dest="results_dir",
        type=str, default="./results",
        help="Seed root. srbench writes <results-dir>/noise_<TAU>/"
             "results_feynman_e2e.pkl.gz; llmsrbench writes <results-dir>/noise_<TAU>/"
             "llmsrbench/e2e_<split>/results.pkl.gz (see results_io).",
    )
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="PyTorch device string. Default: cuda when available.")

    # Data
    parser.add_argument(
        "--n-points", "--n_points", dest="n_points", type=int, default=20000,
        help="[srbench] Rows subsampled from the 100k-row PMLB file per equation "
             "(75/25 split).",
    )

    # Bagging trick
    parser.add_argument(
        "--max-input-points", "--max_input_points", dest="max_input_points",
        type=int, default=200,
        help="Bag size (rows per forward pass). Model was trained on <=200.",
    )
    parser.add_argument(
        "--max-number-bags", "--max_number_bags", dest="max_number_bags",
        type=int, default=100,
        help="Max number of bags. Paper reports SOTA with 100.",
    )

    # Decoding trick
    parser.add_argument(
        "--beam-size", "--beam_size", dest="beam_size", type=int, default=10,
        help="Beam width per bag (matches eval_mymodels.py BEAM_SIZE=10).",
    )
    parser.add_argument(
        "--beam-type", "--beam_type", dest="beam_type",
        type=str, default="search", choices=["sampling", "search"],
        help="'sampling' for structural diversity (paper default). 'search' uses beam search.",
    )
    parser.add_argument("--beam-temperature", "--beam_temperature",
                        dest="beam_temperature", type=float, default=1.0)
    parser.add_argument(
        "--max-forward-batch", "--max_forward_batch", dest="max_forward_batch",
        type=int, default=10,
        help="Max bags processed in one GPU call. Default 1 is safest; increase for speed if VRAM allows.",
    )

    # Refinement trick
    parser.add_argument(
        "--n-trees-to-refine", "--n_trees_to_refine", dest="n_trees_to_refine",
        type=int, default=10,
        help="Top-K unique skeletons passed to BFGS constant refinement.",
    )

    # Evaluation protocol (paper: noise in {0, 0.001, 0.01, 0.1}, 10 seeds)
    parser.add_argument(
        "--target-noise", "--noise", dest="target_noise", type=float, default=0.0,
        metavar="TAU",
        help="Additive-RMS noise on the TRAIN targets: "
             "y_train += N(0, TAU*sqrt(mean(y^2))) (SRBench/TPSR convention). "
             "Held-out test/OOD stay clean. Selects the noise_<TAU>/ output subdir. "
             "Paper tests 0, 0.001, 0.01, 0.1. Default: 0.0.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed. Default: 42 for srbench, 0 for llmsrbench.",
    )
    parser.add_argument(
        "--n-seeds", "--n_seeds", dest="n_seeds", type=int, default=1,
        help="Number of random seeds.",
    )
    parser.add_argument(
        "--r2-threshold", "--r2_threshold", dest="r2_threshold",
        type=float, default=0.99,
        help="[srbench] R^2 threshold for counting a problem as solved (paper uses 0.99).",
    )

    parser.add_argument(
        "--ood-data",
        default=None,
        metavar="PATH",
        help="[srbench] Path to a pre-generated OOD dataset file (.pkl.gz) produced by "
             "gen_ood_data.py.  When provided, each dataset's round-1 formula is "
             "also scored on the OOD test set and OOD R^2 is reported.",
    )

    args = parser.parse_args()
    check_bench_flags(args, parser, _BENCH_ONLY_FLAGS)
    bench_defaults(args, _BENCH_DEFAULTS, ("seed",))

    if args.benchmark == "llmsrbench":
        return run_llmsrbench(args)
    return run_srbench(args)


# ---------------------------------------------------------------------------
# SRBench (Feynman PMLB)
# ---------------------------------------------------------------------------
def run_srbench(args):
    """Writes <results-dir>/noise_<TAU>/results_feynman_e2e.pkl.gz."""
    device = args.device
    print(f"Device: {device}")
    print(f"Loading model from {args.weights_path} ...")
    raw_model = load_raw_model(args.weights_path, device)
    mw = build_model_wrapper(raw_model, args.beam_type, args.beam_size, args.beam_temperature, args.max_forward_batch)

    ood_dataset: dict = {}
    if args.ood_data is not None:
        with gzip.open(args.ood_data, "rb") as _fh:
            _ood_file = pickle.load(_fh)
        ood_dataset = _ood_file["datasets"]
        print(f"Loaded OOD data for {len(ood_dataset)} datasets from {args.ood_data} "
              f"(gap={_ood_file.get('gap', '?')}, "
              f"n_points={_ood_file.get('n_points', '?')}).")

    dataset_names = list_feynman_datasets(args.pmlb_dir)
    if args.dataset:
        # Scope a run to one dataset (or a few). Fail loudly on a typo rather than
        # silently evaluating nothing.
        wanted = [d.strip() for d in args.dataset.split(",") if d.strip()]
        missing = [d for d in wanted if d not in dataset_names]
        if missing:
            sys.exit(f"--dataset: unknown dataset(s) {missing}. "
                     f"{len(dataset_names)} available under {args.pmlb_dir}.")
        dataset_names = [d for d in dataset_names if d in wanted]
    print(f"Found {len(dataset_names)} Feynman datasets in {args.pmlb_dir}")
    print(
        f"Config -- noise={args.target_noise} | seeds={args.n_seeds} | n_points={args.n_points}\n"
        f"  [scaling ON] [bags: {args.max_number_bags}*{args.max_input_points}pts] "
        f"[decoding: {args.beam_type} k={args.beam_size}] "
        f"[BFGS top-{args.n_trees_to_refine}]"
    )

    feynman_pn_targets = load_feynman_pn_targets()
    if feynman_pn_targets:
        print(f"Loaded {len(feynman_pn_targets)} Feynman GT formulas.", flush=True)

    all_results: list[dict] = []
    out_path = srbench_path(args.results_dir, args.target_noise, "results_feynman_e2e.pkl.gz")

    # Resume: load datasets already scored in a previous (interrupted / OOM-killed)
    # run so we skip them -- e2e inference is slow and GPU-memory-heavy.
    _done_pairs: set = set()
    try:
        _rows = load_results(out_path)[1]
        # Keep only VALID prior rows; drop failed/None (e.g. an OOM that
        # run_inference swallowed) so they are redone, not kept as garbage.
        def _valid(r):
            f = r.get("predicted_formula")
            return f is not None and str(f).strip() not in ("", "None", "nan")
        all_results = [r for r in _rows if _valid(r)]
        _done_pairs = {(r["dataset"], int(r["seed"])) for r in all_results}
        _dropped = len(_rows) - len(all_results)
        if _done_pairs:
            print(f"Resuming: {len(_done_pairs)} valid rows in {out_path}"
                  + (f" (dropped {_dropped} failed/None -> will redo)" if _dropped else ""),
                  flush=True)
    except Exception:
        all_results = []

    for seed in range(args.seed, args.seed + args.n_seeds):
        rng = np.random.RandomState(seed)
        seed_results: list[dict] = []
        print(f"\n--- Seed {seed} ---", flush=True)

        for i, name in enumerate(dataset_names):
            if (name, seed) in _done_pairs:
                print(f"[{i+1}/{len(dataset_names)}] {name} ... skipped (already done)", flush=True)
                continue
            gt_pn      = feynman_pn_targets.get(name, "")
            _ood_entry = ood_dataset.get(name)
            _ood_data  = (_ood_entry["X"], _ood_entry["y"]) if _ood_entry is not None else None

            print(f"[{i+1}/{len(dataset_names)}] {name} ... ", end="", flush=True)

            t0 = time.perf_counter()
            res = evaluate_equation(
                dataset_name=name,
                pmlb_dir=args.pmlb_dir,
                mw=mw,
                rng=rng,
                n_points=args.n_points,
                noise=args.target_noise,
                max_input_points=args.max_input_points,
                max_number_bags=args.max_number_bags,
                n_trees_to_refine=args.n_trees_to_refine,
                r2_threshold=args.r2_threshold,
                seed=seed,
                ood_data=_ood_data,
            )
            elapsed = time.perf_counter() - t0

            _fml = res.get("predicted_formula") or "None"
            print(f"  pred:  {_fml}", flush=True)
            if gt_pn:
                print(f"  GT:    {gt_pn}", flush=True)
            if res.get("ood_r2") is not None:
                print(f"  OOD R^2:{res['ood_r2']:.4f}", flush=True)
            _ood_str = f"  OOD R^2={res['ood_r2']:.4f}" if res.get("ood_r2") is not None else ""
            print(f"R^2={res['r2']:.4f}{_ood_str}  t={elapsed:.1f}s", flush=True)

            seed_results.append(res)
            # Flush incrementally (atomically) so an interrupted run still leaves a
            # usable, resumable artifact.
            save_results(out_path, all_results + seed_results,
                         model_name="e2e", target_noise=args.target_noise)

        acc = [r["accuracy"] for r in seed_results]
        r2s = [r["r2"] for r in seed_results]
        print(
            f"\nSeed {seed}: Accuracy (R^2>={args.r2_threshold}) = "
            f"{sum(acc)}/{len(acc)} = {np.mean(acc):.4f}  |  "
            f"Mean R^2 = {np.mean(r2s):.4f}", flush=True
        )
        all_results.extend(seed_results)

    save_results(out_path, all_results, model_name="e2e", target_noise=args.target_noise)
    results_df = pd.DataFrame(all_results)

    # Aggregate over seeds: per-equation mean, then overall mean
    by_eq = results_df.groupby("dataset")
    mean_r2_per_eq = by_eq["r2"].mean()
    mean_acc_per_eq = by_eq["accuracy"].mean()

    print(f"\n=== Aggregate Results ({args.n_seeds} seed(s), noise={args.target_noise}) ===")
    print(f"  Mean R^2:        {mean_r2_per_eq.mean():.4f} +/- {mean_r2_per_eq.std():.4f}")
    print(
        f"  Accuracy (R^2>={args.r2_threshold}): "
        f"{mean_acc_per_eq.mean():.4f}  ({int(mean_acc_per_eq.sum())}/{len(mean_acc_per_eq)} equations)"
    )
    ood_vals = [r["ood_r2"] for r in all_results if r.get("ood_r2") is not None]
    if ood_vals:
        ood_arr = np.array(ood_vals)
        print(f"  OOD Mean R^2:    {ood_arr.mean():.4f}")
        print(f"  OOD R^2 = 1.0:   {(ood_arr >= 1.0 - 1e-6).sum():3d} / {len(ood_vals)}")
        print(f"  OOD R^2 > 0.99:  {(ood_arr > 0.99).sum():3d} / {len(ood_vals)}")
        print(f"  OOD R^2 > 0.90:  {(ood_arr > 0.90).sum():3d} / {len(ood_vals)}")
    print(f"  Results written to: {out_path}")


# ---------------------------------------------------------------------------
# LLM-SRBench
# ---------------------------------------------------------------------------

def _predict(dstr, X):
    """Predict with a fitted SymbolicTransformerRegressor; NaN on any failure."""
    if dstr is None:
        return np.full(X.shape[0], np.nan)
    try:
        yp = dstr.predict(X)
        if yp is None:
            return np.full(X.shape[0], np.nan)
        return np.asarray(yp, dtype=np.float64).reshape(-1)
    except Exception:
        return np.full(X.shape[0], np.nan)


def run_llmsrbench(args):
    """Writes <results-dir>/noise_<TAU>/llmsrbench/e2e_<split>/results.pkl.gz.

    Scored with the same loader + metrics eval_mymodels.py uses, so the two
    methods are directly comparable (imported lazily: an srbench run should not
    pay for loading the transformer stack).
    """
    from eval_mymodels import load_problems, output_metrics

    print(f"Loading E2E model from {args.weights_path} ...", flush=True)
    raw = load_raw_model(args.weights_path, args.device)
    mw = build_model_wrapper(raw, args.beam_type, args.beam_size,
                             args.beam_temperature, args.max_forward_batch)
    print("Model loaded.\n", flush=True)

    print(f"Loading LLM-SRBench split '{args.split}' ...", flush=True)
    problems = load_problems(args.split)
    if args.problem is not None:
        problems = [q for q in problems if q["name"] == args.problem]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    print(f"  {len(problems)} problems.\n", flush=True)

    # Every run goes in a per-noise subdir (tau=0 -> noise_0/) so the method dir name is
    # identical at every noise level (plot_llmsrbench_ood_vs_gap keys its label off it).
    _method = f"e2e_{args.split}"
    results_path = llmsr_path(args.results_dir, args.target_noise or 0.0, _method)

    # Resume: keep problems already scored (skip-by-name).
    done_ids, existing_rows = set(), []
    for _r in load_results(results_path)[1]:
        _eid = _r.get("equation_id")
        _eq  = _r.get("discovered_equation")
        # Treat a missing/None equation (e.g. an OOM that run_inference
        # swallowed) as NOT done, so a re-run redoes it instead of keeping garbage.
        if _eid in done_ids or _eq in (None, "None", ""):
            continue
        done_ids.add(_eid)
        existing_rows.append(_r)
    if done_ids:
        print(f"Resuming: {len(done_ids)} problems already done in {results_path}", flush=True)

    rows = list(existing_rows)

    # Rewrite the whole artifact after each problem; save_results is atomic, so an
    # interrupted run always leaves a loadable, resumable file.
    def _save():
        save_results(results_path, rows, model_name="e2e", split=args.split,
                     method=_method, target_noise=args.target_noise or 0.0)

    _save()
    for i, q in enumerate(problems):
        if q["name"] in done_ids:
            continue
        train, test = q["train"], q["test"]
        n_vars = train.shape[1] - 1
        # Column 0 is y; columns 1: are X (LLM-SRBench convention).
        X_train, y_train = train[:, 1:], train[:, 0]
        X_test,  y_test  = test[:, 1:],  test[:, 0]

        # Per-seed permutation of the training rows: SymbolicTransformerRegressor's
        # bagging chops the train sequence into consecutive max_input_points chunks
        # and keeps the first max_number_bags, so reordering the rows randomises
        # which points land in each bag -> seed-dependent bags (and thus formula
        # variance) without changing the train set itself.  Without this the bags
        # are the first 100 contiguous 200-point slices, identical across seeds, so
        # e2e showed zero seed-to-seed spread at noise 0.  Mirrors the multi-seed
        # bagging of eval_mymodels.py / eval_e2e_tpsr.py --benchmark llmsrbench; an
        # independent RNG per (seed, problem index) keeps problems decorrelated and
        # the permutation reproducible on resume.
        perm = np.random.default_rng([args.seed, i]).permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        # Additive-RMS target noise on TRAIN only (matches the srbench path above);
        # the held-out test + OOD are scored on clean arrays below.
        if args.target_noise and args.target_noise > 0.0 and len(y_train) > 0:
            _rms = float(np.sqrt(np.mean(np.square(y_train))))
            if _rms > 0.0:
                y_train = y_train + np.random.default_rng(
                    [args.seed, i, 1]).normal(0.0, args.target_noise * _rms, size=len(y_train))

        t0 = time.perf_counter()
        dstr = run_inference(mw, X_train, y_train,
                             args.max_input_points, args.max_number_bags,
                             args.n_trees_to_refine)
        elapsed = time.perf_counter() - t0

        id_m = output_metrics(_predict(dstr, X_test), y_test)
        ood_m = None
        if q["ood_test"] is not None:
            ood_m = output_metrics(
                _predict(dstr, q["ood_test"][:, 1:]), q["ood_test"][:, 0])

        log = {
            "equation_id": q["name"],
            "gt_equation": q["expression"],
            "discovered_equation": get_predicted_formula(dstr),
            "n_vars": n_vars,
            "num_datapoints": int(len(train)),
            "num_eval_datapoints": int(len(test)),
            "search_time": elapsed,
            "seed": args.seed,
            "id_metrics": id_m,
            "ood_metrics": ood_m,
        }
        rows.append(log)
        _save()
        print(f"[{i+1}/{len(problems)}] {q['name']:<20} "
              f"R^2={id_m['r2']:.4f}  t={elapsed:.1f}s", flush=True)

    r2s = np.array([r["id_metrics"]["r2"] for r in rows], dtype=np.float64)
    r2s = r2s[np.isfinite(r2s)]
    print(f"\n{'='*70}")
    print(f"E2E  LLM-SRBench {args.split}  ({len(rows)} problems)")
    if r2s.size:
        print(f"  Acc (R^2 >= 0.99) : {100*np.mean(r2s>=0.99):.1f}%")
        print(f"  Mean R^2         : {np.mean(r2s):.4f}   Median R^2: {np.median(r2s):.4f}")
    print(f"{'='*70}\n[results] {results_path}\n")


if __name__ == "__main__":
    main()
