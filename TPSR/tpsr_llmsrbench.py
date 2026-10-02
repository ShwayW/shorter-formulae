#!/usr/bin/env python3
"""
tpsr_llmsrbench.py — Run the *original-authors'* TPSR (MCTS over the E2E
transformer; Shojaee et al. 2023) on the LLM-SRBench benchmark, writing results
in the SAME results.pkl.gz format as eval_e2e.py / eval_mymodels.py so
compare_llmsrbench.py can include it as one method ("e2e_tpsr").

NOT an entry point: this is the library half of `eval_e2e_tpsr.py --benchmark
llmsrbench`, which owns the CLI and calls run(args).  It lives in TPSR/ because
its imports (tpsr.tpsr_fit, symbolicregression, parsers) must resolve to the
vendored copies here, not to the repo root's same-named modules — the caller puts
TPSR/ first on sys.path before importing this module.

This reuses the TPSR machinery shipped in this directory:
  • tpsr.tpsr_fit            — MCTS (UCT/P-UCT) decoding over the E2E decoder
  • SymbolicTransformerRegressor.refine / predict — BFGS constant refinement
  • symbolicregression env   — tree <-> prefix, rescaling, numexpr evaluation
and the LLM-SRBench data loader + metrics (mirrors eval_mymodels.py --benchmark llmsrbench).

The per-problem flow mirrors evaluate.evaluate_pmlb_mcts: top-k feature select →
StandardScaler whitening → MCTS decode per bag → BFGS refine → rescale back →
predict. The searcher sees TRAIN only; we score the discovered formula on the
held-out TEST (and OOD when present) with the official r2/nmse metrics.

Run it through the entry point, from the repo root:
    ./env/bin/python eval_e2e_tpsr.py --benchmark llmsrbench [--split lsr_transform]
        [--max-problems N] [--problem NAME] [--max-number-bags 10]
        [--width 3] [--rollout 3] [--num-beams 1] [--horizon 200] [--lam 0.1]
"""

import os
import sys
import time
import copy
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# APPEND the repo root (which holds results_io.py), never insert it at 0: the root
# also has its own tpsr.py (the user's MCTS TPSR), and putting it ahead of TPSR/
# shadows the authors' TPSR/tpsr.py that provides tpsr_fit -- so TPSR/ must stay
# first on sys.path, repo root only as a fallback.  eval_e2e_tpsr.py arranges the
# same ordering before importing this module; this block keeps the module
# importable on its own too.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from results_io import llmsr_path, load_results, save_results  # noqa: E402

import symbolicregression  # noqa: F401  (registers env + makes utils.CUDA settable)
from symbolicregression.envs import build_env
from symbolicregression.model.model_wrapper import ModelWrapper
from symbolicregression.model.sklearn_wrapper import (
    SymbolicTransformerRegressor, get_top_k_features,
)
import symbolicregression.model.utils_wrapper as utils_wrapper
from parsers import get_parser
from tpsr import tpsr_fit

REPO_ID = "nnheui/llm-srbench"
_REPLACE_OPS = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}

# Local copy of the benchmark: download once with download_llmsrbench.py (repo
# root) and scp it to the cluster, so evaluation needs no internet / HF cache at
# runtime.  Default lives at <repo-root>/datasets/llmsrbench (repo root is this
# file's grandparent since it sits in TPSR/); override with $LLMSRBENCH_DIR.
LLMSRBENCH_DIR = os.environ.get(
    "LLMSRBENCH_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "datasets", "llmsrbench"),
)


def resolve_llmsrbench_file(rel_path):
    """Prefer the local $LLMSRBENCH_DIR copy; fall back to hf_hub_download."""
    local = os.path.join(LLMSRBENCH_DIR, rel_path)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, rel_path, repo_type="dataset")


# ── Data loading (mirror eval_mymodels.load_problems) ──────────────────────
def load_problems(split):
    import h5py
    import pandas as pd

    hdf5_path = resolve_llmsrbench_file("lsr_bench_data.hdf5")
    parquet_path = resolve_llmsrbench_file(f"data/{split}-00000-of-00001.parquet")
    meta = pd.read_parquet(parquet_path)

    if split == "lsr_transform":
        group_of = lambda name: f"/lsr_transform/{name}"
    elif split.startswith("lsr_synth_"):
        domain = split[len("lsr_synth_"):]
        group_of = lambda name: f"/lsr_synth/{domain}/{name}"
    else:
        raise ValueError(f"Unknown split: {split}")

    problems = []
    with h5py.File(hdf5_path, "r") as f:
        for _, e in meta.iterrows():
            g = f[group_of(e["name"])]
            samples = {k: g[k][...].astype(np.float64) for k in g.keys()}
            problems.append({
                "name":       e["name"],
                "symbols":    list(e["symbols"]),
                "expression": e["expression"],
                "train":      samples.get("train"),
                "test":       samples.get("test"),
                "ood_test":   samples.get("ood_test"),
            })
    return problems


# ── Metrics (mirror eval_mymodels.output_metrics) ──────────────────────────
def output_metrics(y_pred, y):
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_pred) & np.isfinite(y)
    n_valid = int(mask.sum())
    if n_valid == 0:
        return {"mse": float("nan"), "nmse": float("nan"), "r2": float("nan"),
                "kdt": float("nan"), "mape": float("nan"), "num_valid_points": 0}
    yp, yt = y_pred[mask], y[mask]
    var = np.var(yt)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    mse = float(np.mean((yt - yp) ** 2))
    nmse = mse / var if var > 0 else float("nan")
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    kdt = mape = float("nan")
    try:
        from scipy.stats import kendalltau
        kdt = float(kendalltau(yt, yp)[0])
    except Exception:
        pass
    try:
        from sklearn.metrics import mean_absolute_percentage_error
        mape = float(mean_absolute_percentage_error(yt, yp))
    except Exception:
        pass
    return {"mse": mse, "nmse": nmse, "r2": r2, "kdt": kdt, "mape": mape,
            "num_valid_points": n_valid}


def _r2_zero(y_true, y_pred):
    """R² against a zero-prediction baseline (matches TPSR's r2_zero), clipped at 0."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("-inf")
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum(yt ** 2))
    if ss_tot == 0:
        return float("-inf")
    return max(0.0, 1.0 - ss_res / ss_tot)


def _formula_str(predicted_tree):
    if predicted_tree is None:
        return None
    try:
        expr = predicted_tree.infix()
        for op, sym in _REPLACE_OPS.items():
            expr = expr.replace(op, sym)
        return expr
    except Exception:
        return None


# ── Per-problem fit (mirror evaluate.evaluate_pmlb_mcts inner loop) ──────────
def fit_one(X_train, y_train, params, env, mw):
    """Return (fitted_dstr, discovered_formula_str, search_time_seconds).

    fitted_dstr.predict(X_raw, refinement_type='BFGS') maps raw inputs to y.
    """
    y = np.expand_dims(np.asarray(y_train, dtype=np.float64), -1)
    X = [np.asarray(X_train, dtype=np.float64)]
    Y = [y]

    dstr = SymbolicTransformerRegressor(
        model=mw,
        max_input_points=params.max_input_points,
        n_trees_to_refine=params.n_trees_to_refine,
        max_number_bags=params.max_number_bags,
        rescale=params.rescale,
    )
    # Top-k feature selection (E2E handles <= max_input_dimension variables).
    dstr.top_k_features = [
        get_top_k_features(X[0], Y[0], k=mw.env.params.max_input_dimension)]
    X[0] = X[0][:, dstr.top_k_features[0]]

    scaler = utils_wrapper.StandardScaler() if params.rescale else None
    scale_params = {}
    if scaler is not None:
        scaled_X = [scaler.fit_transform(X[0])]
        scale_params[0] = scaler.get_params()
    else:
        scaled_X = X

    bag_number = 1
    done_bagging = False
    bagging_threshold = 0.99
    best_fit_r2 = float("-inf")
    best_tree = None
    total_time = 0.0
    max_bags = min(params.max_number_bags + 1,
                   len(scaled_X[0]) // params.max_input_points + 2)

    while not done_bagging and bag_number < max_bags:
        s, time_elapsed, _ = tpsr_fit(scaled_X, Y, params, env, bag_number)
        total_time += time_elapsed
        generated_tree = list(filter(
            lambda t: t is not None,
            [env.idx_to_infix(s[1:], is_float=False, str_array=False)]))
        if not generated_tree:
            bag_number += 1
            continue

        dstr.start_fit = time.time()
        dstr.tree = {}
        refined = dstr.refine(scaled_X[0], Y[0], generated_tree, verbose=False)
        if not refined:
            bag_number += 1
            continue
        if scaler is not None:
            # Rescale *every* candidate (not just [0]) back to raw-input space, so
            # predict(refinement_type='BFGS') uses a tree in the correct coordinates
            # regardless of which candidate the r2 ordering put first.
            for cand in refined:
                if cand.get("predicted_tree") is not None:
                    cand["predicted_tree"] = scaler.rescale_function(
                        env, cand["predicted_tree"], *scale_params[0])
        dstr.tree[0] = refined

        # Decide bagging on the (raw) training fit, as in evaluate_pmlb_mcts.
        try:
            y_fit = dstr.predict(X_train, refinement_type="BFGS")
            fit_r2 = _r2_zero(y_train, y_fit)
        except Exception:
            fit_r2 = float("-inf")

        # Keep the best bag; always capture the first valid candidate even if its
        # fit R² is non-finite (predict can fail), so a discovered tree is never lost.
        if best_tree is None or fit_r2 > best_fit_r2:
            best_fit_r2 = fit_r2
            best_tree = copy.deepcopy(dstr.tree)

        if fit_r2 > bagging_threshold:
            done_bagging = True
        else:
            bag_number += 1

    if best_tree is None:
        return None, None, total_time
    dstr.tree = best_tree
    info = dstr.retrieve_tree(refinement_type="BFGS", with_infos=True)
    formula = _formula_str(info.get("predicted_tree") if info else None)
    return dstr, formula, total_time


def _predict(dstr, X):
    if dstr is None:
        return np.full(X.shape[0], np.nan)
    try:
        yp = dstr.predict(X, refinement_type="BFGS")
        if yp is None:
            return np.full(X.shape[0], np.nan)
        return np.asarray(yp, dtype=np.float64).reshape(-1)
    except Exception:
        return np.full(X.shape[0], np.nan)


# ── Params setup (mirror run.py) ─────────────────────────────────────────────
def build_params(args):
    params = get_parser().parse_args([])
    params.eval_only = True
    params.cpu = args.device == "cpu"
    params.multi_gpu = False
    params.is_slurm_job = False
    params.device = torch.device(args.device)

    # MCTS / TPSR knobs (defaults mirror TPSR run.sh main config).
    params.backbone_model = "e2e"
    params.width = args.width
    params.num_beams = args.num_beams
    params.rollout = args.rollout
    params.horizon = args.horizon
    params.lam = args.lam
    params.no_seq_cache = args.no_seq_cache
    params.no_prefix_cache = args.no_prefix_cache
    params.sample_only = False
    params.train_value = False
    params.debug = False

    # E2E inference (bagging / refinement) knobs.
    params.max_input_points = args.max_input_points
    params.max_number_bags = args.max_number_bags
    params.n_trees_to_refine = args.n_trees_to_refine
    params.rescale = True

    # cache mode string, as run.py computes it.
    if (not params.no_prefix_cache) and (not params.no_seq_cache):
        params.cache = "b"
    elif params.no_prefix_cache and (not params.no_seq_cache):
        params.cache = "s"
    elif (not params.no_prefix_cache) and params.no_seq_cache:
        params.cache = "k"
    else:
        params.cache = "n"

    symbolicregression.utils.CUDA = not params.cpu
    return params


def _method_dir(args, shard_id=None, num_shards=1):
    """Method dir name for (split, shard). Unsharded runs keep the historical name."""
    base = f"e2e_tpsr_{args.split}"
    if num_shards == 1 or shard_id is None:
        return base
    return f"{base}__shard{shard_id:02d}of{num_shards:02d}"


def merge_shards(args):
    """Fold per-shard artifacts into the canonical results.pkl.gz, then delete them.

    Seeded from the canonical file FIRST so merging is purely additive: a re-run that
    dies early can only fill gaps, never replace an answer already on disk with a
    missing one. Rows are keyed (equation_id, seed) and a row carrying a real equation
    always beats a None/empty one. Same contract as eval_llmsr.merge_shards.
    """
    import shutil

    tau = args.target_noise or 0.0
    canonical = llmsr_path(args.results_dir, tau, _method_dir(args))
    merged = {}
    for r in load_results(canonical)[1]:
        merged[(r.get("equation_id"), r.get("seed"))] = r
    n_prior = len(merged)

    n_found = 0
    for i in range(args.num_shards):
        sp = llmsr_path(args.results_dir, tau, _method_dir(args, i, args.num_shards))
        rows = load_results(sp)[1]
        if rows:
            n_found += 1
        for r in rows:
            key = (r.get("equation_id"), r.get("seed"))
            prev = merged.get(key)
            if prev is None or prev.get("discovered_equation") in (None, "None", ""):
                merged[key] = r

    save_results(canonical, list(merged.values()), model_name="e2e_tpsr",
                 split=args.split, method=_method_dir(args), target_noise=tau)
    print(f"[merge] {n_found}/{args.num_shards} shards + {n_prior} prior "
          f"-> {len(merged)} rows -> {canonical}", flush=True)

    for i in range(args.num_shards):
        d = os.path.dirname(llmsr_path(args.results_dir, tau,
                                       _method_dir(args, i, args.num_shards)))
        if os.path.isdir(d):
            shutil.rmtree(d)
    print(f"[merge] removed {args.num_shards} shard artifacts", flush=True)


def run(args):
    """Evaluate the authors' TPSR on one LLM-SRBench split.

    `args` is the namespace parsed by eval_e2e_tpsr.py (which owns the CLI);
    the fields used here are split / device / max_input_points /
    max_number_bags / n_trees_to_refine / width / num_beams / rollout /
    horizon / lam / no_seq_cache / no_prefix_cache / problem / max_problems /
    results_dir / seed / noise.
    """
    # eval_e2e_tpsr.py spells the shared noise flag --target-noise but stores it
    # as `noise` (the srbench path's long-standing name); accept either.
    args.target_noise = getattr(args, 'target_noise', None)
    if args.target_noise is None:
        args.target_noise = getattr(args, 'noise', 0.0) or 0.0

    # Sharding (optional; absent when another caller builds the namespace).
    args.num_shards = int(getattr(args, 'num_shards', 1) or 1)
    args.shard_id = int(getattr(args, 'shard_id', 0) or 0)
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard-id must be in [0, {args.num_shards}); "
                         f"got {args.shard_id}")
    if getattr(args, 'merge_shards', False):
        return merge_shards(args)

    # Global RNG seeding for reproducibility + any stochastic MCTS/decoder sampling.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    params = build_params(args)

    print("Building symbolicregression env ...", flush=True)
    env = build_env(params)
    env.rng = np.random.RandomState(args.seed)

    print("Loading E2E model (for refine/predict wrapper) ...", flush=True)
    raw = torch.load(os.path.join(_HERE, "symbolicregression", "weights", "model.pt"),
                     map_location=params.device, weights_only=False)
    raw.eval()
    mw = ModelWrapper(
        env=env,
        embedder=raw.embedder,
        encoder=raw.encoder,
        decoder=raw.decoder,
        beam_length_penalty=params.beam_length_penalty,
        beam_size=params.beam_size,
        max_generated_output_len=params.max_generated_output_len,
        beam_early_stopping=params.beam_early_stopping,
        beam_temperature=params.beam_temperature,
        beam_type=params.beam_type,
    )
    print("Model + env ready.\n", flush=True)

    print(f"Loading LLM-SRBench split '{args.split}' ...", flush=True)
    problems = load_problems(args.split)
    if args.problem is not None:
        problems = [q for q in problems if q["name"] == args.problem]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    if args.num_shards > 1:
        # Stride slice, not a contiguous block: problem cost varies with variable count,
        # so striding spreads the expensive ones evenly over the array tasks.
        problems = problems[args.shard_id::args.num_shards]
        print(f"  shard {args.shard_id}/{args.num_shards}", flush=True)
    print(f"  {len(problems)} problems.\n", flush=True)

    # Every run goes in a per-noise subdir (τ=0 → noise_0/) so the method dir name is
    # identical at every noise level (compare/plot scripts key their labels off it).
    _method = _method_dir(args, args.shard_id, args.num_shards)
    results_path = llmsr_path(args.results_dir, args.target_noise or 0.0, _method)

    # Resume: skip problems already written in a previous (interrupted / timed-out)
    # run — TPSR/MCTS is very slow. A missing/None equation (e.g. a swallowed error)
    # is treated as NOT done so it gets retried instead of kept as garbage.
    # A merge deletes the shard artifacts, so a re-submitted array would otherwise
    # redo finished work: fall back to the canonical file, filtered to this shard's
    # own problems so a shard never adopts another's rows.
    _mine = {q["name"] for q in problems}
    _sources = [results_path]
    if args.num_shards > 1:
        _sources.append(llmsr_path(args.results_dir, args.target_noise or 0.0,
                                   _method_dir(args)))
    done_ids, existing_rows = set(), []
    for _src in _sources:
        for _r in load_results(_src)[1]:
            _eid = _r.get("equation_id")
            _eq  = _r.get("discovered_equation")
            if _eid in done_ids or _eq in (None, "None", "") or _eid not in _mine:
                continue
            done_ids.add(_eid)
            existing_rows.append(_r)
    if done_ids:
        print(f"Resuming: {len(done_ids)} problems already done in "
              f"{' + '.join(_sources)}", flush=True)

    rows = list(existing_rows)

    # Rewrite the whole artifact after each problem; save_results is atomic, so an
    # interrupted MCTS run always leaves a loadable, resumable file.
    def _save():
        save_results(results_path, rows, model_name="e2e_tpsr", split=args.split,
                     method=_method, target_noise=args.target_noise or 0.0,
                     num_shards=args.num_shards, shard_id=args.shard_id)

    _save()
    for i, q in enumerate(problems):
        if q["name"] in done_ids:
            continue
        train, test = q["train"], q["test"]
        n_vars = train.shape[1] - 1
        X_train, y_train = train[:, 1:], train[:, 0]
        X_test,  y_test  = test[:, 1:],  test[:, 0]

        # Per-seed permutation of the training rows: the E2E bagging chops the
        # train sequence into consecutive max_input_points chunks, so reordering
        # the rows changes which points land in each bag → seed-dependent bags
        # (and thus formula variance) without altering the train set itself.
        # An independent RNG per (seed, problem index) keeps problems decorrelated.
        perm = np.random.default_rng([args.seed, i]).permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        # Additive-RMS noise on the TRAIN targets only (test/OOD stay clean),
        # same convention as eval_mymodels.py --benchmark llmsrbench / SRBench.
        if args.target_noise and args.target_noise > 0.0:
            _rms = float(np.sqrt(np.mean(np.asarray(y_train, np.float64) ** 2)))
            y_train = y_train + np.random.default_rng(
                [args.seed, i, 1]).normal(0.0, args.target_noise * _rms, size=len(y_train))

        t0 = time.perf_counter()
        try:
            dstr, formula, _ = fit_one(X_train, y_train, params, env, mw)
        except Exception as exc:
            print(f"[{i+1}/{len(problems)}] {q['name']:<20} FAILED: {exc}", flush=True)
            dstr, formula = None, None
        elapsed = time.perf_counter() - t0

        id_m = output_metrics(_predict(dstr, X_test), y_test)
        ood_m = None
        if q["ood_test"] is not None:
            ood_m = output_metrics(
                _predict(dstr, q["ood_test"][:, 1:]), q["ood_test"][:, 0])

        log = {
            "equation_id": q["name"],
            "gt_equation": q["expression"],
            "discovered_equation": formula,
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
              f"R²={id_m['r2']:.4f}  NMSE={id_m['nmse']:.3e}  t={elapsed:.1f}s  "
              f"{formula}", flush=True)

    r2s = np.array([r["id_metrics"]["r2"] for r in rows], dtype=np.float64)
    r2s = r2s[np.isfinite(r2s)]
    print(f"\n{'='*70}")
    print(f"E2E+TPSR  LLM-SRBench {args.split}  ({len(rows)} problems)")
    if r2s.size:
        print(f"  Acc (R² >= 0.99) : {100*np.mean(r2s>=0.99):.1f}%")
        print(f"  Mean R²          : {np.mean(r2s):.4f}   Median R²: {np.median(r2s):.4f}")
    print(f"{'='*70}\n[results] {results_path}\n")

