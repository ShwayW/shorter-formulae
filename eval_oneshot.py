#!/usr/bin/env python3
"""
eval_oneshot.py -- Evaluate the ONE-SHOT LLM baseline (llm_methods/oneshot) on the
SRBench Feynman and LLM-SRBench benchmarks, writing the same on-disk artifacts as
eval_llmsr.py / eval_mymodels.py / eval_e2e.py so the shared compare_*/aggregate_seeds
tooling picks it up next to the search-based methods.

What makes it "one-shot": the LLM is shown a table of input-output pairs once and
returns a closed-form formula *with its numeric constants*, which is taken as the
final answer. No evolutionary loop, no BFGS refit -- exactly one LLM call per problem
(plus a capped retry when the reply does not parse). Runtime per problem is one
completion, so a full benchmark takes minutes rather than GPU-days.

Data loading, noise injection and metrics are imported from eval_llmsr.py, so the
one-shot and LLMSR rows are directly comparable: same splits, same 75/25 Feynman
subsample, same TRAIN-only noise convention, same r2/nmse/kdt/mape on TEST and OOD.

Usage:
    # local smoke test against ollama (llama3.1:8b)
    python eval_oneshot.py --searcher_config configs/oneshot_llama31_8b_ollama.yaml \
                           --benchmark srbench --max-problems 5

    # full LLM-SRBench lsr_transform split
    python eval_oneshot.py --searcher_config configs/oneshot_llama31_8b_vllm.yaml \
                           --split lsr_transform --seed 0 --n-seeds 1

Results land in
    <output>/noise_<TAU>/results_<method>.pkl.gz            (feynman / SRBench)
    <output>/noise_<TAU>/llmsrbench/<method>/results.pkl.gz (LLM-SRBench splits)
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "llm_methods"))
sys.path.insert(0, _HERE)

from results_io import load_results, save_results
from bench.dataclasses import SEDTask

# Shared with the LLMSR driver so both methods see byte-identical data and metrics.
from eval_llmsr import (load_problems, load_feynman_problems, output_metrics,
                        safe_predict, add_target_noise, _artifact_path, merge_shards)

warnings.filterwarnings("ignore")


def build_searcher(cfg: dict, seed: int):
    """Construct a OneShotSearcher wired to the backend named in the yaml config."""
    from oneshot.llm import build_llm
    from oneshot.searcher import OneShotSearcher

    return OneShotSearcher(
        name=cfg.get("name", "oneshot"),
        llm=build_llm(cfg),
        mode=cfg.get("mode", "direct"),
        n_prompt_rows=cfg.get("n_prompt_rows", 30),
        max_retries=cfg.get("max_retries", 2),
        num_candidates=cfg.get("num_candidates", 1),
        n_restarts=cfg.get("n_restarts", 4),
        fit_max_rows=cfg.get("fit_max_rows", 5000),
        fit_tol=cfg.get("fit_tol", 1e-14),
        seed=seed,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--searcher_config", type=str, required=True,
                   help="YAML config, e.g. configs/oneshot_llama31_8b_vllm.yaml")
    p.add_argument("--benchmark", choices=["srbench", "llmsrbench"], default=None,
                   help="srbench => --split feynman; llmsrbench => --split lsr_transform.")
    p.add_argument("--split", type=str, default=None,
                   help="feynman, lsr_transform, or lsr_synth_<domain>. "
                        "Default: lsr_transform (feynman with --benchmark srbench).")
    p.add_argument("--n-points", type=int, default=20000,
                   help="feynman only: rows subsampled from the PMLB file before the "
                        "75/25 train/test split (default 20000).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed. Seeds used are seed, ..., seed+n_seeds-1. Seeds the "
                        "prompt's row subsample (and the noise draw).")
    p.add_argument("--n-seeds", "--n_seeds", dest="n_seeds", type=int, default=1)
    p.add_argument("--target-noise", type=float, default=0.0, metavar="TAU",
                   help="Gaussian noise on TRAIN targets: y += N(0, TAU*sqrt(mean(y^2))). "
                        "Test/OOD scored clean.")
    p.add_argument("--mode", choices=["direct", "skeleton"], default=None,
                   help="Override the config's mode. direct: the LLM writes the constants "
                        "and the formula is used as written. skeleton: the LLM writes "
                        "structure with free constants and BFGS fits them to TRAIN "
                        "(LLMSR's optimizer step, run once). A CLI override that "
                        "disagrees with the config tags the method name (_skeleton / "
                        "_direct) so the two arms never share a results dir.")
    p.add_argument("--n-restarts", type=int, default=None,
                   help="skeleton only: BFGS starts per skeleton (first is all-ones).")
    p.add_argument("--fit-max-rows", type=int, default=None,
                   help="skeleton only: cap on TRAIN rows used by the fit (0 = all rows). "
                        "Scoring always uses the full TEST/OOD splits.")
    p.add_argument("--n-prompt-rows", type=int, default=None,
                   help="Override the config's number of observations shown in the prompt.")
    p.add_argument("--num-candidates", type=int, default=None,
                   help="Override the config: >1 draws that many formulas and keeps the "
                        "best by TRAIN MSE. 1 (default) is the pure one-shot protocol.")
    p.add_argument("--problem", type=str, default=None, help="Evaluate only this problem.")
    p.add_argument("--max-problems", type=int, default=None, help="Cap problems (smoke test).")
    p.add_argument("--method-name", type=str, default=None,
                   help="Override the method dir name (default: derived from config 'name').")
    p.add_argument("--output", type=str, default="results", help="Seed root (default results).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total parallel shards (processes) over a stride slice of problems.")
    p.add_argument("--shard-id", type=int, default=0, help="This process's shard in [0, num_shards).")
    p.add_argument("--merge-shards", action="store_true", default=False,
                   help="Fold shard artifacts into the canonical results file, then exit.")
    args = p.parse_args()

    _bench_split = {"srbench": "feynman", "llmsrbench": "lsr_transform"}
    if args.benchmark is not None:
        _implied = _bench_split[args.benchmark]
        if args.split is None:
            args.split = _implied
        elif (args.split == "feynman") != (args.benchmark == "srbench"):
            p.error(f"--benchmark {args.benchmark} and --split {args.split} disagree.")
    if args.split is None:
        args.split = "lsr_transform"
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard-id must be in [0, {args.num_shards}); got {args.shard_id}")

    with open(args.searcher_config) as f:
        cfg = yaml.safe_load(f)
    if args.n_prompt_rows is not None:
        cfg["n_prompt_rows"] = args.n_prompt_rows
    if args.num_candidates is not None:
        cfg["num_candidates"] = args.num_candidates
    if args.n_restarts is not None:
        cfg["n_restarts"] = args.n_restarts
    if args.fit_max_rows is not None:
        cfg["fit_max_rows"] = args.fit_max_rows or None      # 0 -> all rows

    # A --mode that overrides the config gets tagged into the method name: the two modes
    # are different arms of the same experiment, and silently writing skeleton results
    # into the direct arm's results dir would mix them in every downstream plot.
    _cfg_mode = cfg.get("mode", "direct")
    _mode_tag = ""
    if args.mode is not None and args.mode != _cfg_mode:
        cfg["mode"] = args.mode
        _mode_tag = f"_{args.mode}"

    cfg_name = (args.method_name or cfg.get("name", "oneshot")).lower() + _mode_tag
    _method = f"{cfg_name}_{args.split}"
    tau = args.target_noise or 0.0

    if args.merge_shards:
        merge_shards(args.output, tau, args.split, _method, args.num_shards,
                     model_name=cfg_name, api_model=cfg["api_model"])
        return

    shard_tag = ("" if args.num_shards == 1
                 else f"__shard{args.shard_id:02d}of{args.num_shards:02d}")
    results_path = _artifact_path(args.output, tau, args.split, _method,
                                  args.num_shards, args.shard_id)

    print(f"Loading split '{args.split}' ...", flush=True)
    problems = (load_feynman_problems(n_points=args.n_points) if args.split == "feynman"
                else load_problems(args.split))
    if args.problem is not None:
        problems = [q for q in problems if q["name"] == args.problem]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    if args.num_shards > 1:
        problems = problems[args.shard_id::args.num_shards]
        print(f"  shard {args.shard_id}/{args.num_shards}", flush=True)
    print(f"  {len(problems)} problems.\n", flush=True)
    _mode = cfg.get("mode", "direct")
    print(f"Backend: {cfg['api_model']} ({cfg.get('api_type')})  "
          f"mode: {_mode}"
          + (f" (BFGS: {cfg.get('n_restarts', 4)} restarts, "
             f"<= {cfg.get('fit_max_rows', 5000) or 'all'} fit rows)" if _mode == "skeleton" else "")
          + f"  prompt rows: {cfg.get('n_prompt_rows', 30)}  "
          f"candidates/problem: {cfg.get('num_candidates', 1)}  "
          f"retries: {cfg.get('max_retries', 2)}\n", flush=True)

    seeds = list(range(args.seed, args.seed + args.n_seeds))

    # Resume, keyed (equation_id, seed); a missing equation counts as not done. Same
    # shard/canonical fallback as eval_llmsr.py so a chained job never redoes a cell.
    _mine = {q["name"] for q in problems}
    _sources = [results_path]
    if args.num_shards > 1:
        _sources.append(_artifact_path(args.output, tau, args.split, _method))
    done_pairs, existing_rows = set(), []
    for _src in _sources:
        for _r in load_results(_src)[1]:
            _key = (_r.get("equation_id"), _r.get("seed"))
            if _key in done_pairs or _r.get("discovered_equation") in (None, "None", ""):
                continue
            if _r.get("equation_id") not in _mine:
                continue
            done_pairs.add(_key)
            existing_rows.append(_r)
    if done_pairs:
        print(f"Resuming: {len(done_pairs)} (problem, seed) pairs already done", flush=True)

    rows = list(existing_rows)

    def _save():
        save_results(results_path, rows, model_name=cfg_name, split=args.split,
                     method=_method, target_noise=tau, api_model=cfg["api_model"],
                     mode=cfg.get("mode", "direct"),
                     n_prompt_rows=cfg.get("n_prompt_rows", 30),
                     num_candidates=cfg.get("num_candidates", 1),
                     n_restarts=cfg.get("n_restarts", 4),
                     fit_max_rows=cfg.get("fit_max_rows", 5000))

    _save()

    for seed in seeds:
        rng = np.random.default_rng(seed)
        searcher = build_searcher(cfg, seed)

        for i, q in enumerate(problems):
            if (q["name"], seed) in done_pairs:
                continue
            train, test = q["train"], q["test"]
            n_vars = train.shape[1] - 1

            train_seed = add_target_noise(train, args.target_noise, rng)
            task = SEDTask(name=f"{q['name']}_seed{seed}",
                           symbols=q["symbols"], symbol_descs=q["symbol_descs"],
                           symbol_properties=q["symbol_properties"], samples=train_seed)

            t0 = time.perf_counter()
            try:
                result = searcher.discover(task)[0]
                lambda_fn = result.equation.lambda_format
                discovered = result.equation.expression
                aux = dict(result.aux) if result.aux else {}
            except Exception as e:
                print(f"[seed {seed}] {q['name']}: search failed: {e}", flush=True)
                lambda_fn, discovered, aux = None, None, {}
            elapsed = time.perf_counter() - t0

            X_test, y_test = test[:, 1:], test[:, 0]
            id_m = (output_metrics(safe_predict(lambda_fn, X_test), y_test)
                    if lambda_fn is not None
                    else output_metrics(np.full_like(y_test, np.nan), y_test))
            ood_m = None
            if q["ood_test"] is not None and lambda_fn is not None:
                ood = q["ood_test"]
                ood_m = output_metrics(safe_predict(lambda_fn, ood[:, 1:]), ood[:, 0])

            rows.append({
                "equation_id": q["name"],
                "gt_equation": q["expression"],
                "discovered_equation": discovered,
                "n_vars": n_vars,
                "num_datapoints": int(len(train)),
                "num_eval_datapoints": int(len(test)),
                "search_time": elapsed,
                "seed": seed,
                "id_metrics": id_m,
                "ood_metrics": ood_m,
                **aux,
            })
            _save()
            print(f"[seed {seed}] [{i+1}/{len(problems)}] {q['name']:<24} "
                  f"R^2={id_m['r2']:.4f}  t={elapsed:.1f}s  {str(discovered)[:60]}",
                  flush=True)

    # -- Summary --------------------------------------------------------------
    r2s = np.array([r["id_metrics"]["r2"] for r in rows], dtype=np.float64)
    r2s_valid = r2s[np.isfinite(r2s)]
    n_failed = sum(1 for r in rows if r.get("parse_failed"))
    n_fitfail = sum(1 for r in rows if r.get("fit_failed"))
    n_params = [r.get("n_params", 0) for r in rows if not r.get("parse_failed")]
    times = np.array([r["search_time"] for r in rows], dtype=np.float64)
    print(f"\n{'='*70}")
    print(f"{args.split} -- {cfg_name} (one-shot)  ({len(rows)} problems)")
    print(f"{'='*70}")
    if r2s_valid.size:
        for thr in (0.99, 0.999):
            print(f"  Acc (R^2 >= {thr})  : {float(np.mean(r2s_valid >= thr)) * 100:.1f}%")
        print(f"  Mean R^2          : {np.mean(r2s_valid):.4f}")
        print(f"  Median R^2        : {np.median(r2s_valid):.4f}")
        print(f"  Valid / total     : {r2s_valid.size}/{len(rows)}")
    print(f"  Unparseable replies: {n_failed}")
    if cfg.get("mode", "direct") == "skeleton":
        print(f"  Failed BFGS fits   : {n_fitfail}")
        if n_params:
            print(f"  Mean fitted consts : {np.mean(n_params):.2f}")
    if times.size:
        print(f"  Mean time/problem : {np.mean(times):.1f}s")
    print(f"{'='*70}")
    print(f"[results] {results_path}\n")


if __name__ == "__main__":
    main()
