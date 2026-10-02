#!/usr/bin/env python3
"""
eval_e2e_tpsr.py -- Evaluate E2E+TPSR (MCTS decoding over the E2E transformer) on
either benchmark.  Pick with --benchmark:

  srbench     (default)  119 Feynman formulae from SRBench, via this repo's own
                         E2ETPSR (e2e_tpsr.py).  Reproduces the original TPSR
                         paper's evaluation exactly:
                           - train_test_split shuffle=True, random_state=29910
                           - StandardScaler on X; BFGS on scaled X, rescale back
                           - multi-bag MCTS, early-stop when train R^2 > 0.99
                           - max_bags = min(11, n_train//max_input_points + 2)
                           - R^2 >= 0.99 on test counts as solved
                         Writes <results-dir>/noise_<TAU>/results_e2e_tpsr.pkl.gz.

  llmsrbench             LLM-SRBench, via the ORIGINAL AUTHORS' TPSR vendored in
                         TPSR/ (tpsr.tpsr_fit).  Writes <results-dir>/noise_<TAU>/
                         llmsrbench/e2e_tpsr_<split>/results.pkl.gz.

IMPORTANT -- the two paths use different, conflicting copies of the
`symbolicregression` package (this repo's inner one vs the vendored TPSR/ one),
and the repo root's own tpsr.py shadows the authors' TPSR/tpsr.py.  Neither can be
imported at module scope, so each branch sets sys.path up and imports lazily:
see _setup_srbench_imports() and run_llmsrbench().  One process runs one branch.
The llmsrbench implementation itself lives in TPSR/tpsr_llmsrbench.py, which is a
library module -- this file is the only entry point.
"""

import argparse
import os
import signal

from results_io import load_results, save_results, srbench_path
from bench_cli import bench_defaults, check_bench_flags
import sys
import time
import warnings

warnings.filterwarnings("ignore")


class DatasetTimeout(BaseException):
    pass


def _timeout_handler(signum, frame):
    raise DatasetTimeout("per-dataset timeout exceeded")


# symbolicregression runs its OWN per-call SIGALRM timeout (utils.py's @timeout
# decorator, e.g. @timeout(10) on rescale_function) and raises MyTimeoutError, which
# subclasses BaseException rather than Exception.  So it walks straight through every
# `except Exception` guard below -- including the ones that already wrap the BFGS and
# rescale calls with a sensible fallback -- and kills the whole worker mid-sweep,
# stranding that (seed, noise) cell with however many rows it had written.  Upstream
# catches it explicitly where it cares (envs/simplifiers.py:31); this eval must too.
#
# Imported lazily and cached: symbolicregression is only importable once the sys.path
# juggling in _prepare_inner() has run, which is after module import.
_MY_TIMEOUT = []


def _my_timeout_cls():
    if not _MY_TIMEOUT:
        try:
            from symbolicregression.utils import MyTimeoutError
            _MY_TIMEOUT.append(MyTimeoutError)
        except Exception:
            _MY_TIMEOUT.append(None)
    return _MY_TIMEOUT[0]


def _soft_excs():
    """What a per-candidate step may swallow and fall back from."""
    cls = _my_timeout_cls()
    return (Exception, cls) if cls is not None else (Exception,)


def _dataset_excs():
    """What aborts ONE dataset without taking the worker down with it."""
    cls = _my_timeout_cls()
    return (DatasetTimeout, cls) if cls is not None else (DatasetTimeout,)

_HERE = os.path.dirname(os.path.abspath(__file__))
_INNER = os.path.join(_HERE, "symbolicregression")
_TPSR_DIR = os.path.join(_HERE, "TPSR")


def _purge(*prefixes):
    """Drop already-imported modules so the next import re-resolves on sys.path."""
    for k in [k for k in sys.modules
              if any(k == p or k.startswith(p + ".") for p in prefixes)]:
        del sys.modules[k]


def _setup_srbench_imports():
    """Put this repo's inner symbolicregression/ first, then import E2ETPSR.

    Deferred out of module scope: the llmsrbench branch needs the vendored TPSR/
    copy of the same package instead, and whichever is imported first wins.
    """
    global E2ETPSR
    if _INNER not in sys.path or sys.path[0] != _INNER:
        _purge("symbolicregression")
        sys.path = [_INNER] + [p for p in sys.path if p != _INNER]
    from e2e_tpsr import E2ETPSR


import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from gen_ood_data import load_feynman_pn_targets

E2ETPSR = None   # bound by _setup_srbench_imports()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def list_feynman_datasets(pmlb_dir: str) -> list:
    datasets_dir = os.path.join(pmlb_dir, "datasets")
    return sorted(
        d for d in os.listdir(datasets_dir)
        if d.startswith("feynman_") and os.path.isdir(os.path.join(datasets_dir, d))
    )


def load_pmlb_dataset(pmlb_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(pmlb_dir, "datasets", name, f"{name}.tsv.gz")
    return pd.read_csv(path, sep="\t", compression="gzip")


# ---------------------------------------------------------------------------
# Per-equation evaluation -- matches paper's evaluate_pmlb_mcts exactly
# ---------------------------------------------------------------------------

def evaluate_equation(
    name: str,
    pmlb_dir: str,
    searcher: "E2ETPSR",
    n_points: int,
    noise: float,
    r2_threshold: float,
    max_input_points: int,
    n_bfgs_refine: int,
    split_random_state: int,
    max_bags: int,
    rng: np.random.RandomState,
) -> dict:
    result = {
        "dataset": name,
        "predicted_formula": None,
        "r2": 0.0,
        "accuracy": 0,
        "n_bags_used": 0,
    }

    try:
        df = load_pmlb_dataset(pmlb_dir, name)
    except Exception as e:
        result["error"] = str(e)
        return result

    if len(df) > n_points:
        idx = rng.choice(len(df), n_points, replace=False)
        df  = df.iloc[idx].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values.astype(np.float64)
    y = df["target"].values.astype(np.float64).reshape(-1, 1)

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        return result

    # Paper: train_test_split with shuffle=True, fixed random_state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=split_random_state
    )
    y_train = y_train.ravel()
    y_test  = y_test.ravel()

    if noise > 0.0:
        scale = noise * np.sqrt(np.mean(y_train ** 2))
        y_train = y_train + rng.normal(0.0, scale, size=y_train.shape)

    # StandardScaler fitted on full training set
    from symbolicregression.model.utils_wrapper import StandardScaler, BFGSRefinement
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    scale_a, scale_b = scaler.get_params()

    # Paper: max_bags = min(11, n_train // max_input_points + 2), starting at bag 1
    n_train   = len(X_train)
    n_bags_actual = min(max_bags, n_train // max_input_points + 2)

    best_r2_train = -1.0
    best_tree     = None
    done_bagging  = False

    _bfgs_n = min(1024, n_train)

    for bag_num in range(1, n_bags_actual + 1):
        if done_bagging:
            break

        # Slice bag: rows (bag_num-1)*200 .. bag_num*200
        lo = (bag_num - 1) * max_input_points
        hi = bag_num * max_input_points
        if lo >= n_train:
            break
        X_bag = X_train_s[lo:hi]
        y_bag = y_train[lo:hi]
        if len(X_bag) == 0:
            break

        try:
            candidates = searcher.search(X_bag, y_bag)
        except Exception:
            candidates = []
        finally:
            searcher.decoder.cache = {"slen": 0}
            searcher._src_enc = None
            searcher._src_len = None
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result["n_bags_used"] = bag_num

        if not candidates:
            continue

        # BFGS on scaled X, then rescale tree back to original coords
        for _score, tree in candidates[:max(1, n_bfgs_refine)]:
            try:
                skel, consts = searcher.env.generator.function_to_skeleton(
                    tree, constants_with_idx=True)
                ref_scaled = BFGSRefinement().go(
                    env=searcher.env, tree=skel, coeffs0=consts,
                    X=X_train_s[:_bfgs_n], y=y_train[:_bfgs_n].reshape(-1, 1),
                    downsample=-1, stop_after=searcher.bfgs_stop_after,
                )
                if ref_scaled is None:
                    ref_scaled = tree
            except _soft_excs():
                ref_scaled = tree

            # Rescale tree to original coordinates
            try:
                ref_orig = scaler.rescale_function(searcher.env, ref_scaled,
                                                   scale_a, scale_b)
            except _soft_excs():
                ref_orig = ref_scaled

            # Evaluate on original training set to pick best across bags
            try:
                fn = searcher.env.simplifier.tree_to_numexpr_fn(ref_orig)
                y_pred_train = np.asarray(fn(X_train), dtype=np.float64).ravel()
                if np.all(np.isfinite(y_pred_train)) and len(y_pred_train) == n_train:
                    r2_train = float(max(0.0, r2_score(y_train, y_pred_train)))
                else:
                    r2_train = 0.0
            except Exception:
                r2_train = 0.0

            if r2_train > best_r2_train:
                best_r2_train = r2_train
                best_tree     = ref_orig

            # Paper: early stop when train R^2 > bagging_threshold (0.99)
            if r2_train > r2_threshold:
                done_bagging = True
                break

    if best_tree is None:
        return result

    # Final evaluation on original test set
    try:
        fn     = searcher.env.simplifier.tree_to_numexpr_fn(best_tree)
        y_pred = np.asarray(fn(X_test), dtype=np.float64).ravel()
        if np.all(np.isfinite(y_pred)) and len(y_pred) == len(y_test):
            r2 = float(max(0.0, r2_score(y_test, y_pred)))
        else:
            r2 = 0.0
        result["r2"]               = r2
        result["accuracy"]         = int(r2 >= r2_threshold)
        result["predicted_formula"] = best_tree.infix()
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Defaults that differ per benchmark (parsed as None so "unset" is detectable).
_BENCH_DEFAULTS = {"srbench": {"seed": 42}, "llmsrbench": {"seed": 0}}

# Flags only one benchmark's path reads; passing one to the other benchmark is an
# error rather than a silent no-op.  See bench_cli.
_BENCH_ONLY_FLAGS = {
    "srbench": ["pmlb_dir", "n_points", "r2_threshold", "max_bags", "n_simulations",
                "rollout_beam_width", "top_k", "uct_alg", "c_puct", "ucb_base",
                "prior_temperature", "warm_start", "warm_start_beam_width",
                "bfgs_in_search", "bfgs_stop_after", "n_bfgs_refine", "reward_lam",
                "split_random_state", "start_idx", "end_idx", "dataset_names",
                "timeout_per_dataset", "no_cuda", "n_seeds"],
    "llmsrbench": ["split", "problem", "max_problems", "max_number_bags",
                   "n_trees_to_refine", "width", "num_beams", "rollout", "lam",
                   "no_seq_cache", "no_prefix_cache",
                   "num_shards", "shard_id", "merge_shards"],
}


def main():
    parser = argparse.ArgumentParser(
        description=("Evaluate E2E+TPSR on SRBench (Feynman) or LLM-SRBench. "
                     "Pick with --benchmark."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark", default="srbench", choices=["srbench", "llmsrbench"],
        help="Which benchmark to run. 'srbench' uses this repo's E2ETPSR on the "
             "Feynman PMLB datasets; 'llmsrbench' uses the original authors' TPSR "
             "(vendored in TPSR/) on the LLM-SRBench --split. Default: srbench.",
    )
    parser.add_argument("--weights-path", "--weights_path", dest="weights_path",
                        default="weights/e2e/e2e_model.pt")
    parser.add_argument("--pmlb-dir", "--pmlb_dir", dest="pmlb_dir",
                        default="datasets/pmlb", help="[srbench]")
    parser.add_argument(
        "--results-dir", "--results_dir", "--output", dest="results_dir",
        default="results",
        help="Seed root. srbench writes <results-dir>/noise_<TAU>/"
             "results_e2e_tpsr.pkl.gz; llmsrbench writes <results-dir>/noise_<TAU>/"
             "llmsrbench/e2e_tpsr_<split>/results.pkl.gz (see results_io).",
    )

    # -- LLM-SRBench only -----------------------------------------------------
    parser.add_argument("--split", type=str, default="lsr_transform",
                        help="[llmsrbench] lsr_transform or lsr_synth_<domain>.")
    parser.add_argument("--problem", type=str, default=None,
                        help="[llmsrbench] Evaluate only this problem (by name).")
    parser.add_argument("--max-problems", "--max_problems", dest="max_problems",
                        type=int, default=None,
                        help="[llmsrbench] Cap the number of problems (smoke tests).")
    parser.add_argument("--device", type=str, default=None,
                        help="[llmsrbench] PyTorch device string. Default: cuda when "
                             "available (srbench uses --no_cuda instead).")
    parser.add_argument("--max-number-bags", "--max_number_bags",
                        dest="max_number_bags", type=int, default=10,
                        help="[llmsrbench] Max E2E bags per problem.")
    parser.add_argument("--n-trees-to-refine", "--n_trees_to_refine",
                        dest="n_trees_to_refine", type=int, default=10,
                        help="[llmsrbench] Top-K skeletons passed to BFGS.")
    parser.add_argument("--width", type=int, default=3,
                        help="[llmsrbench] MCTS expansion width.")
    parser.add_argument("--num-beams", "--num_beams", dest="num_beams",
                        type=int, default=1, help="[llmsrbench] Rollout beams.")
    parser.add_argument("--rollout", type=int, default=3,
                        help="[llmsrbench] MCTS rollouts per step.")
    parser.add_argument("--lam", type=float, default=0.1,
                        help="[llmsrbench] Reward lambda (complexity penalty).")
    parser.add_argument("--no-seq-cache", dest="no_seq_cache",
                        action="store_true", default=False,
                        help="[llmsrbench] Disable sequence caching (default: ON, like run.sh).")
    parser.add_argument("--no-prefix-cache", dest="no_prefix_cache",
                        action="store_true", default=True,
                        help="[llmsrbench] Disable top-k/prefix caching (default: ON-disabled, "
                             "like run.sh).")

    # Data
    parser.add_argument("--n_points",           type=int,   default=20000)
    parser.add_argument("--max_input_points",   type=int,   default=200)
    parser.add_argument("--target-noise", "--noise", dest="noise",
                        type=float, default=0.0, metavar="TAU",
                        help="Additive-RMS noise on the TRAIN targets: "
                             "y_train += N(0, TAU*sqrt(mean(y^2))). Held-out test/OOD "
                             "stay clean. Selects the noise_<TAU>/ output subdir.")
    parser.add_argument("--seed",               type=int,   default=None,
                        help="Base random seed. Default: 42 for srbench, 0 for llmsrbench.")
    parser.add_argument("--n-seeds", "--n_seeds", dest="n_seeds",
                        type=int,   default=1)
    parser.add_argument("--r2_threshold",       type=float, default=0.99)
    parser.add_argument("--split_random_state", type=int,   default=29910,
                        help="random_state for train_test_split (paper uses 29910).")
    parser.add_argument("--max_bags",           type=int,   default=11,
                        help="Max MCTS bags per dataset (paper uses min(11, n_train//200+2)).")

    # TPSR hyperparams (paper defaults)
    parser.add_argument("--horizon",              type=int,   default=200)
    parser.add_argument("--n_simulations",        type=int,   default=3)
    parser.add_argument("--rollout_beam_width",   type=int,   default=1)
    parser.add_argument("--top_k",                type=int,   default=3)
    parser.add_argument("--uct_alg",              type=str,   default="uct",
                        choices=["uct", "p_uct", "var_p_uct"])
    parser.add_argument("--c_puct",               type=float, default=1.0)
    parser.add_argument("--ucb_base",             type=float, default=10.0)
    parser.add_argument("--prior_temperature",    type=float, default=1.0)
    parser.add_argument("--warm_start",           action="store_true", default=True)
    parser.add_argument("--no_warm_start",        dest="warm_start", action="store_false")
    parser.add_argument("--warm_start_beam_width",type=int,   default=10)
    parser.add_argument("--bfgs_in_search",       action="store_true", default=False)
    parser.add_argument("--bfgs_in_search_on",    dest="bfgs_in_search", action="store_true")
    parser.add_argument("--bfgs_stop_after",      type=int,   default=5)
    parser.add_argument("--n_bfgs_refine",        type=int,   default=10)
    parser.add_argument("--reward_lam",           type=float, default=0.1)

    # -- Parallelism (llmsrbench): N processes, each --shard-id in [0, N), on disjoint
    # problems (stride slicing), each writing its own artifact so concurrent whole-file
    # saves never clobber; then one --merge-shards pass folds them into the canonical
    # results.pkl.gz. Same contract as eval_llmsr.py. TPSR at the authors' default
    # budget is minutes per problem, so this is what makes a cluster array worthwhile.
    parser.add_argument("--num-shards", "--num_shards", dest="num_shards",
                        type=int, default=1,
                        help="[llmsrbench] Total parallel shards (default 1).")
    parser.add_argument("--shard-id", "--shard_id", dest="shard_id", type=int, default=0,
                        help="[llmsrbench] This process's shard in [0, num_shards).")
    parser.add_argument("--merge-shards", "--merge_shards", dest="merge_shards",
                        action="store_true", default=False,
                        help="[llmsrbench] Merge shard artifacts into the canonical "
                             "results.pkl.gz and exit. Do NOT pass while shards run.")

    parser.add_argument("--verbose",  action="store_true", default=False)
    parser.add_argument("--no_cuda",  action="store_true", default=False)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx",   type=int, default=None)
    parser.add_argument("--dataset_names", type=str, default=None,
                        help="Comma-separated list of dataset names to evaluate. "
                             "Overrides start_idx/end_idx.")
    parser.add_argument("--timeout_per_dataset", type=int, default=7200,
                        help="Max seconds per dataset (default 7200 = 2h). 0 = no timeout.")

    args = parser.parse_args()
    check_bench_flags(args, parser, _BENCH_ONLY_FLAGS)
    bench_defaults(args, _BENCH_DEFAULTS, ("seed",))

    if args.benchmark == "llmsrbench":
        return run_llmsrbench(args)
    return run_srbench(args)


# ---------------------------------------------------------------------------
# LLM-SRBench: the original authors' TPSR, vendored in TPSR/
# ---------------------------------------------------------------------------
def run_llmsrbench(args):
    """Delegate to TPSR/tpsr_llmsrbench.py with the vendored package tree first.

    TPSR/ must precede the repo root on sys.path: both provide `tpsr` and
    `symbolicregression`, and this branch needs the vendored ones (tpsr.tpsr_fit).
    """
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    _purge("symbolicregression", "tpsr")
    sys.path = [_TPSR_DIR] + [p for p in sys.path if p not in (_TPSR_DIR, _INNER)]
    import tpsr_llmsrbench
    return tpsr_llmsrbench.run(args)


# ---------------------------------------------------------------------------
# SRBench (Feynman PMLB): this repo's own E2ETPSR
# ---------------------------------------------------------------------------
def run_srbench(args):
    """Writes <results-dir>/noise_<TAU>/results_e2e_tpsr.pkl.gz."""
    out_path = srbench_path(args.results_dir, args.noise, "results_e2e_tpsr.pkl.gz")

    _cuda_ok = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if _cuda_ok else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {args.weights_path} ...")

    _setup_srbench_imports()

    raw_model = torch.load(args.weights_path, map_location=device, weights_only=False)
    raw_model.eval()

    searcher = E2ETPSR(
        model_wrapper         = raw_model,
        device                = device,
        n_simulations         = args.n_simulations,
        horizon               = args.horizon,
        uct_alg               = args.uct_alg,
        c_puct                = args.c_puct,
        ucb_base              = args.ucb_base,
        top_k                 = args.top_k,
        prior_temperature     = args.prior_temperature,
        rollout_beam_width    = args.rollout_beam_width,
        verbose               = args.verbose,
        reward_lam            = args.reward_lam,
        warm_start            = args.warm_start,
        warm_start_beam_width = args.warm_start_beam_width,
        bfgs_in_search        = args.bfgs_in_search,
        bfgs_stop_after       = args.bfgs_stop_after,
    )

    dataset_names = list_feynman_datasets(args.pmlb_dir)
    if args.dataset_names is not None:
        filter_set = set(n.strip() for n in args.dataset_names.split(","))
        dataset_names = [d for d in dataset_names if d in filter_set]
    elif args.end_idx is not None:
        dataset_names = dataset_names[args.start_idx:args.end_idx]
    else:
        dataset_names = dataset_names[args.start_idx:]

    print(f"Found {len(dataset_names)} Feynman datasets in {args.pmlb_dir}")
    print(
        f"Config -- noise={args.noise} | n_points={args.n_points} | "
        f"split_rs={args.split_random_state} | max_bags={args.max_bags}\n"
        f"  TPSR: uct_alg={args.uct_alg} c_puct={args.c_puct} ucb_base={args.ucb_base}\n"
        f"  horizon={args.horizon} sims={args.n_simulations} "
        f"rollout_bw={args.rollout_beam_width} top_k={args.top_k} "
        f"warm_start={args.warm_start}(bw={args.warm_start_beam_width}) "
        f"bfgs_post={args.n_bfgs_refine}"
    )

    try:
        feynman_pn_targets = load_feynman_pn_targets()
    except FileNotFoundError:
        feynman_pn_targets = {}

    all_results = []

    # Resume: load datasets already scored in a previous (interrupted / timed-out)
    # run and skip them -- TPSR/MCTS inference is very slow, so never redo good rows.
    # Rows with a failed/None formula (e.g. a per-dataset timeout) are dropped so
    # they get retried rather than kept as garbage.
    _done_pairs: set = set()
    try:
        _rows = load_results(out_path)[1]
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
        seed_results = []
        print(f"\n--- Seed {seed} ---", flush=True)

        for i, name in enumerate(dataset_names):
            if (name, seed) in _done_pairs:
                print(f"[{i+1}/{len(dataset_names)}] {name} ... skipped (already done)", flush=True)
                continue
            gt = feynman_pn_targets.get(name, "")
            print(f"[{i+1}/{len(dataset_names)}] {name} ...", end=" ", flush=True)

            t0  = time.perf_counter()
            if args.timeout_per_dataset > 0:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(args.timeout_per_dataset)
            try:
                res = evaluate_equation(
                    name=name,
                    pmlb_dir=args.pmlb_dir,
                    searcher=searcher,
                    n_points=args.n_points,
                    noise=args.noise,
                    r2_threshold=args.r2_threshold,
                    max_input_points=args.max_input_points,
                    n_bfgs_refine=args.n_bfgs_refine,
                    split_random_state=args.split_random_state,
                    max_bags=args.max_bags,
                    rng=rng,
                )
            except _dataset_excs():
                res = {"dataset": name, "predicted_formula": None,
                       "r2": 0.0, "accuracy": 0, "n_bags_used": -1,
                       "error": "timeout"}
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                searcher.decoder.cache = {"slen": 0}
                searcher._src_enc = None
                searcher._src_len = None
            finally:
                if args.timeout_per_dataset > 0:
                    signal.alarm(0)
            elapsed = time.perf_counter() - t0

            res["seed"] = seed
            fml = res.get("predicted_formula") or "None"
            timeout_flag = " TIMEOUT" if res.get("error") == "timeout" else ""
            print(
                f"R^2={res['r2']:.4f}  acc={res['accuracy']}  "
                f"bags={res.get('n_bags_used',1)}  t={elapsed:.1f}s{timeout_flag}",
                flush=True,
            )
            print(f"  pred: {fml}", flush=True)
            if gt:
                print(f"  GT:   {gt}", flush=True)

            seed_results.append(res)
            save_results(out_path, all_results + seed_results,
                         model_name="e2e_tpsr", target_noise=args.noise)

        acc = [r["accuracy"] for r in seed_results]
        r2s = [r["r2"]       for r in seed_results]
        print(
            f"\nSeed {seed}: Accuracy (R^2>={args.r2_threshold}) = "
            f"{sum(acc)}/{len(acc)} = {np.mean(acc):.4f}  |  "
            f"Mean R^2 = {np.mean(r2s):.4f}", flush=True
        )
        all_results.extend(seed_results)

    save_results(out_path, all_results, model_name="e2e_tpsr", target_noise=args.noise)
    results_df = pd.DataFrame(all_results)

    by_eq    = results_df.groupby("dataset")
    mean_r2  = by_eq["r2"].mean()
    mean_acc = by_eq["accuracy"].mean()

    print(f"\n=== Aggregate Results ({args.n_seeds} seed(s), noise={args.noise}) ===")
    print(f"  Mean R^2:        {mean_r2.mean():.4f} +/- {mean_r2.std():.4f}")
    print(
        f"  Accuracy (R^2>={args.r2_threshold}): "
        f"{mean_acc.mean():.4f}  "
        f"({int(mean_acc.sum())}/{len(mean_acc)} equations)"
    )
    print(f"  Results written to: {out_path}")


if __name__ == "__main__":
    main()
