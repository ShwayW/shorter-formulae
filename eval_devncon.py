#!/usr/bin/env python3
"""eval_devncon.py -- evaluate our transformer checkpoints with the DIVIDE-AND-CONQUER
decoder (devncon.py) on SRBench (Feynman/PMLB) and LLM-SRBench.

devncon replaces the single beam search with: train an oracle network, read f's structure
off its Hessian, split the variables into independent groups, beam-search each group's
pseudo-data separately, and recombine.  A plain beam search runs first as a safety net --
if it already solves the problem, the oracle is never trained.

This is the devncon counterpart of eval_mymodels.py / eval_mymodels.py --benchmark llmsrbench, and it deliberately
writes the SAME artifacts under the SAME layout, tagged ``_devncon`` so it can never
overwrite a beam or TPSR run:

    SRBench    <results-dir>/noise_<TAU>/eval_tf_<model>_devncon.pkl.gz
    LLM-SRB    <results-dir>/noise_<TAU>/llmsrbench/<model>_<split>_devncon/results.pkl.gz

Row schemas match eval_mymodels.py and eval_mymodels.py --benchmark llmsrbench exactly, so slurm/collect_results.py,
merge_seed_results.py and every compare_/plot_ script read these with no changes --
point them at the group tree and a devncon curve appears beside the beam curve.

PROTOCOL PARITY (so the curves are comparable)
----------------------------------------------
* the SAME data budget a standalone beam run gets -- ceil(n_bags*sample_size/0.75)
  rows -- then the same outer 75/25 split.  eval_mymodels starves nothing here and neither
  does this; under-feeding that cap is what produced the no-split regressions in the
  retired AIF2 decomposition (_old/aif2.py).
* --target-noise TAU adds N(0, TAU*sqrt(mean(y^2))) to the TRAIN targets ONLY, the
  SRBench/TPSR convention, so TAU here means what it means in the other curves.
* The reported R^2 is on the held-out rows, and devncon selects between its baseline and
  its decompositions on those same held-out rows (``holdout=``) -- the same guard the
  retired AIF2 path used (_old/aif2.py).
* The ORACLE trains on the TRAIN slice only (capped at --oracle-rows), so it can never
  leak the held-out rows, and it never shrinks the transformer's budget.  Its targets
  carry the same noise, or the decomposition would read clean structure the search
  cannot see.

Resume is keyed on (dataset|equation, seed) and the artifact is rewritten atomically
after every problem, so a timed-out Slurm job just needs resubmitting.

USAGE
    python eval_devncon.py --model 89M_40_simp1 --n_seeds 10 --target-noise 0 \
        --results-dir results/mymodels_devncon/seed42
    python eval_devncon.py --model 89M_40_simp1 --bench llmsr --split lsr_transform
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

import devncon
from results_io import llmsr_path, load_results, save_results, srbench_path


# ===========================================================================
def _add_target_noise(y_train, tau, rng):
    """y_train += N(0, tau*sqrt(mean(y^2))) -- SRBench/TPSR convention, TRAIN only."""
    if not tau or tau <= 0.0 or len(y_train) == 0:
        return y_train
    rms = float(np.sqrt(np.mean(np.square(np.asarray(y_train, dtype=np.float64)))))
    if rms <= 0.0:
        return y_train
    return y_train + rng.normal(0.0, tau * rms, size=y_train.shape)


def _prepare(X_all, y_all, rng, n_bags, sample_size, oracle_rows, tau, n_points=None):
    """(X_tr, y_tr, X_te, y_te, X_or, y_or) -- the beam run's data budget, exactly.

    Parity matters more than anything else here -- the point of this script is a curve
    laid beside the beam curve.  So: the SAME data budget a standalone beam run gets
    (ceil(n_bags*SAMPLE_SIZE/0.75) rows), the SAME outer 75/25 split, and noise on the
    train slice only.  Starving this cap is what caused the retired AIF2 decomposition's
    high-dimensional no-split regressions, and it would do the same here.

    The ORACLE trains on the TRAIN slice (capped at ``oracle_rows``) -- never on rows the
    held-out score is computed from, and never at the expense of the transformer's budget.
    Its targets get the same noise, or the decomposition would read structure off cleaner
    data than the search is allowed to see.
    """
    X_all = np.asarray(X_all, dtype=np.float64)
    y_all = np.asarray(y_all, dtype=np.float64).reshape(-1)
    N = X_all.shape[0]

    # 1. The transformer's data budget, taken from the FULL dataset -- never from a
    #    smaller oracle pool.  Capping in the other order starved the baseline beam
    #    (11,250 training rows instead of 20,000) and pushed feynman_I_29_16 to R^2 0.26
    #    against 0.47 for a standalone beam run, which would have shown up as a devncon
    #    curve below the beam curve for a reason having nothing to do with decomposition.
    # `n_points` overrides the derived budget.  On LLM-SRBench it is 20000, matching the
    # cap eval_mymodels.evaluate_xy applies on its own no_split path, so the beam, TPSR
    # and D&C arms all draw from the same pool there.  On SRBench it is left None and the
    # derived ceil(n_bags*sample_size/0.75) stands, which the beam and TPSR jobs now pass
    # explicitly as --n-points so all three arms match there too.
    cap = int(n_points) if n_points else int(np.ceil(n_bags * sample_size / 0.75))
    if N > cap:
        i = rng.choice(N, size=cap, replace=False)
        Xs, ys = X_all[i], y_all[i]
    else:
        Xs, ys = X_all, y_all

    # 2. Outer 75/25, so the reported R^2 is out-of-sample.
    n = Xs.shape[0]
    n_tr = max(1, int(0.75 * n))
    X_tr, X_te = Xs[:n_tr], Xs[n_tr:]
    y_tr, y_te = ys[:n_tr], ys[n_tr:]
    if X_te.shape[0] == 0:
        X_te, y_te = X_tr, y_tr

    # 3. The oracle trains on the TRAIN slice only -- it must never see X_te, or the
    #    held-out score is contaminated through the pseudo-data.  At the default budget
    #    that slice is already ~20k rows, which is what the Hessian wants.
    if X_tr.shape[0] > oracle_rows:
        i = rng.choice(X_tr.shape[0], size=oracle_rows, replace=False)
        X_or, y_or = X_tr[i], y_tr[i]
    else:
        X_or, y_or = X_tr, y_tr

    y_tr = _add_target_noise(y_tr, tau, rng)
    y_or = _add_target_noise(y_or, tau, rng)
    return X_tr, y_tr, X_te, y_te, X_or, y_or


def _solve(X_all, y_all, model, vocab, device, seed, args):
    """One (problem, seed) cell.  Returns (formula, heldout_r2, seconds)."""
    from eval_mymodels import bfgs_refine_mfg, compute_r2, evaluate_xy

    rng = np.random.default_rng(seed)
    X_tr, y_tr, X_te, y_te, X_or, y_or = _prepare(
        X_all, y_all, rng, args.n_bags, args.sample_size, args.oracle_rows,
        args.target_noise, getattr(args, "n_points", None))

    def leaf_solver(Xs, ys, baseline=False):
        # baseline=True -> evaluate_xy does its OWN 75/25 split and selects candidates
        # out-of-sample, exactly as a standalone beam run does.  Sub-problems fit on all
        # of their pseudo-data (no_split=True), which has no held-out concept.
        return evaluate_xy(Xs, ys, model, vocab, device, rng,
                           n_bags=args.n_bags, beam_size=args.beam_size,
                           no_split=not baseline, use_unscaling=args.unscale,
                           max_forward_batch=args.max_forward_batch)

    t0 = time.perf_counter()
    formula, r2 = devncon.devncon_solve(
        X_tr, y_tr, leaf_solver=leaf_solver, r2_fn=compute_r2,
        refine_fn=bfgs_refine_mfg, rng=rng, device=device,
        holdout=(X_te, y_te), oracle_data=(X_or, y_or),
        safety_threshold=args.safety_threshold,
        baseline_draws=args.baseline_draws,
        recon_threshold=args.recon_threshold,
        oracle_points=args.oracle_points,
        verbose=args.verbose)
    return formula, float(r2), time.perf_counter() - t0


# ===========================================================================
def run_srbench(model, vocab, device, model_name, args, solve_fn=None, tag="devncon"):
    """`solve_fn(X, y, model, vocab, device, seed, args) -> (formula, r2, seconds)`.

    Defaults to this module's `_solve` (our decomposition).  eval_phyedc.py passes its
    own so the SAME dataset universe, data budget, noise convention, resume logic and
    results_io layout drive both D&C arms -- `tag` is the only thing that differs in the
    artifact, so nothing can overwrite anything and every compare_/plot_ script reads
    both with no changes.
    """
    from eval_mymodels import (compute_r2, load_dataset, load_feynman_pn_targets,
                         load_feynman_targets)

    solve_fn = solve_fn or _solve

    targets = load_feynman_targets()
    pn_targets = load_feynman_pn_targets()

    dirs = ([os.path.join(args.datasets, args.dataset)] if args.dataset else
            sorted(os.path.join(args.datasets, d) for d in os.listdir(args.datasets)
                   if os.path.isdir(os.path.join(args.datasets, d))))
    out_path = srbench_path(args.results_dir, args.target_noise or 0.0,
                            f"eval_tf_{model_name}_{tag}.pkl.gz")

    _, prior = load_results(out_path)
    rows = [r for r in prior if r.get("predicted_formula")]
    done = {(r["dataset"], r["seed"]) for r in rows}
    if done:
        print(f"Resuming: {len(done)} (dataset, seed) pairs already in {out_path}",
              flush=True)

    def _save():
        save_results(out_path, rows, model_name=f"{model_name}_{tag}",
                     method=tag, target_noise=args.target_noise or 0.0)

    _save()
    seeds = list(range(args.seed, args.seed + args.n_seeds))
    for seed in seeds:
        for i, d in enumerate(dirs, 1):
            name = os.path.basename(d)
            if (name, seed) in done:
                continue
            tsv = os.path.join(d, f"{name}.tsv.gz")
            if not os.path.isfile(tsv):
                continue
            X, y = load_dataset(tsv)
            if X.shape[1] > devncon._NV:
                print(f"[skip] {name}: {X.shape[1]} vars > capacity {devncon._NV}",
                      flush=True)
                continue
            try:
                formula, r2, dt = solve_fn(X, y, model, vocab, device, seed, args)
            except Exception as e:                       # never lose a whole sweep
                print(f"[error] {name} seed {seed}: {e!r}", flush=True)
                continue
            rows.append({"dataset": name, "seed": seed, "round": 1,
                         "r2": r2, "ood_r2": None, "time": dt,
                         "predicted_formula": formula,
                         "target_formula": targets.get(name, ""),
                         "target_pn": pn_targets.get(name, "")})
            done.add((name, seed))
            _save()
            print(f"[seed {seed}] [{i}/{len(dirs)}] {name:<26} R^2={r2:.4f} "
                  f"t={dt:.1f}s", flush=True)

    _summarise_srbench(rows, args, tag)
    print(f"[results] {out_path}\n")


def _summarise_srbench(rows, args, tag="devncon"):
    per = {}
    for r in rows:
        per.setdefault(r["dataset"], []).append(r["r2"])
    m = [float(np.mean(v)) for v in per.values()]
    print(f"\n{'=' * 60}")
    print(f"SRBench + {tag} : {len(per)} datasets, {args.n_seeds} seeds")
    if m:
        print(f"  Mean R^2        : {np.mean(m):.4f}")
        print(f"  Median R^2      : {np.median(m):.4f}")
        for thr in (0.99, 0.999):
            print(f"  R^2 > {thr:<6}    : {sum(1 for v in m if v > thr):3d} / {len(m)}")
    print("=" * 60)


# ===========================================================================
def run_llmsr(model, vocab, device, model_name, args, solve_fn=None, tag="devncon"):
    """LLM-SRBench half of the same loop; see run_srbench for `solve_fn` / `tag`."""
    from eval_mymodels import load_problems, output_metrics, predict

    solve_fn = solve_fn or _solve

    problems = load_problems(args.split)
    if args.problem:
        problems = [q for q in problems if q["name"] == args.problem]
    if args.max_problems:
        problems = problems[:args.max_problems]

    method = f"{model_name}_{args.split}_{tag}"
    out_path = llmsr_path(args.results_dir, args.target_noise or 0.0, method)

    _, prior = load_results(out_path)
    rows = [r for r in prior if r.get("discovered_equation")]
    done = {(r["equation_id"], r["seed"]) for r in rows}
    if done:
        print(f"Resuming: {len(done)} (problem, seed) pairs already in {out_path}",
              flush=True)

    def _save():
        save_results(out_path, rows, model_name=f"{model_name}_{tag}",
                     split=args.split, method=method,
                     target_noise=args.target_noise or 0.0)

    _save()
    seeds = list(range(args.seed, args.seed + args.n_seeds))
    for seed in seeds:
        for i, q in enumerate(problems, 1):
            if (q["name"], seed) in done:
                continue
            train, test = q["train"], q["test"]
            n_vars = train.shape[1] - 1
            if n_vars > devncon._NV:
                print(f"[skip] {q['name']}: {n_vars} vars > capacity {devncon._NV}",
                      flush=True)
                continue
            try:
                formula, _r2, dt = solve_fn(train[:, 1:], train[:, 0], model, vocab,
                                            device, seed, args)
            except Exception as e:
                print(f"[error] {q['name']} seed {seed}: {e!r}", flush=True)
                continue

            # Scored on the benchmark's OWN test split, exactly as eval_mymodels does.
            id_m = output_metrics(predict(formula, test[:, 1:]), test[:, 0])
            ood_m = (output_metrics(predict(formula, q["ood_test"][:, 1:]),
                                    q["ood_test"][:, 0])
                     if q.get("ood_test") is not None else None)
            rows.append({"equation_id": q["name"], "gt_equation": q["expression"],
                         "discovered_equation": formula, "n_vars": n_vars,
                         "num_datapoints": int(len(train)),
                         "num_eval_datapoints": int(len(test)),
                         "search_time": dt, "seed": seed,
                         "id_metrics": id_m, "ood_metrics": ood_m})
            done.add((q["name"], seed))
            _save()
            print(f"[seed {seed}] [{i}/{len(problems)}] {q['name']:<22} "
                  f"R^2={id_m['r2']:.4f}  t={dt:.1f}s", flush=True)

    r2s = np.array([r["id_metrics"]["r2"] for r in rows], dtype=np.float64)
    r2s = r2s[np.isfinite(r2s)]
    print(f"\n{'=' * 70}")
    print(f"LLM-SRBench {args.split} + {tag} -- {model_name}  ({len(rows)} rows)")
    if r2s.size:
        for thr in (0.99, 0.999):
            print(f"  Acc (R^2 >= {thr})  : {float(np.mean(r2s >= thr)) * 100:.1f}%")
        print(f"  Mean R^2          : {np.mean(r2s):.4f}")
        print(f"  Median R^2        : {np.median(r2s):.4f}")
    print("=" * 70)
    print(f"[results] {out_path}\n")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate our checkpoints with the devncon divide-and-conquer decoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --benchmark is the spelling every eval_<method>.py uses; --bench and the
    # old value "llmsr" stay accepted so existing slurm/*.sh invocations keep working.
    ap.add_argument("--benchmark", "--bench", dest="benchmark",
                    choices=["srbench", "llmsrbench", "llmsr"], default="srbench",
                    help="Which benchmark to run. Default: srbench.")
    ap.add_argument("--model", default="89M_40_simp1",
                    help="89M_40_simp1 = 89M, 145M_40_simp1 = 145M, m89float")
    ap.add_argument("--checkpoint", default=None, help="explicit .pth (overrides --model)")
    ap.add_argument("--checkpoints-dir", default="./checkpoints")
    ap.add_argument("--device", default=None)

    ap.add_argument("--datasets", default="./datasets/pmlb/datasets",
                    help="SRBench dataset root")
    ap.add_argument("--dataset", default=None, help="single SRBench dataset")
    ap.add_argument("--split", default="lsr_transform", help="LLM-SRBench split")
    ap.add_argument("--problem", default=None, help="single LLM-SRBench equation")
    ap.add_argument("--max-problems", type=int, default=None)

    ap.add_argument("--seed", type=int, default=42, help="first seed (42..42+n-1)")
    ap.add_argument("--n_seeds", "--n-seeds", dest="n_seeds", type=int, default=1)
    ap.add_argument("--target-noise", type=float, default=0.0, metavar="TAU",
                    help="additive RMS noise on TRAIN targets (SRBench convention)")
    ap.add_argument("--results-dir", default="./results",
                    help="root; the noise subdir is composed from --target-noise")

    ap.add_argument("--n-bags", type=int, default=100,
                    help="IO bags per solve; with --sample-size this sets the data "
                         "budget ceil(n_bags*sample_size/0.75), matching a beam run")
    ap.add_argument("--sample-size", type=int, default=None, metavar="N",
                    help="IO pairs per bag; default = eval_mymodels.SAMPLE_SIZE (200). The "
                         "145M model was trained with up to 400.")
    ap.add_argument("--n-points", type=int, default=None, metavar="N",
                    help="rows subsampled per problem before the 75/25 split. Default: "
                         "SRBench uses the derived ceil(n_bags*sample_size/0.75); "
                         "LLM-SRBench uses 20000, matching evaluate_xy's no_split cap so "
                         "the beam, TPSR and D&C arms share one pool on that benchmark.")
    ap.add_argument("--beam-size", type=int, default=10)
    ap.add_argument("--unscale", action="store_true")
    ap.add_argument("--max-forward-batch", type=int, default=None)

    ap.add_argument("--oracle-rows", type=int, default=20000,
                    help="raw rows for oracle training (its Hessian needs far more "
                         "data than the transformer's context)")
    ap.add_argument("--oracle-points", type=int, default=1000,
                    help="sample points for the Hessian estimate")
    ap.add_argument("--safety-threshold", type=float, default=0.9999,
                    help="baseline R^2 at or above which decomposition is skipped")
    ap.add_argument("--baseline-draws", type=int, default=1,
                    help="full-problem beam draws for the step-0 safety net; the best "
                         "(by the searcher's own held-out split) becomes the floor. "
                         "Re-draws only while short of --safety-threshold, so easy "
                         "problems still cost one draw. Default 1 = the original "
                         "behaviour; 3 matches the retired AIF2 path's baseline draws.")
    ap.add_argument("--recon-threshold", type=float, default=0.999,
                    help="oracle-reconstruction R^2 a decomposition must clear before "
                         "any beam search is spent on it")
    ap.add_argument("--verbose", action="store_true",
                    help="per-problem devncon trace (Hessians, candidates, groups)")
    args = ap.parse_args()

    from eval_mymodels import (DEFAULT_FORWARD_BATCH, SAMPLE_SIZE, load_model,
                         resolve_model_checkpoint)
    if args.max_forward_batch is None:
        args.max_forward_batch = DEFAULT_FORWARD_BATCH
    # evaluate_xy reads the module-level SAMPLE_SIZE (there is no per-call knob), so the
    # bag size is set here once.  The 145M model was trained with up to 400 IO points and
    # its beam curve uses --sample-size 400; matching it is what keeps the devncon 145M_40_simp1
    # curve comparable to the 145M_40_simp1 beam curve.
    if args.sample_size is None:
        args.sample_size = SAMPLE_SIZE
    else:
        import eval_mymodels as _tf
        _tf.SAMPLE_SIZE = int(args.sample_size)

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = args.checkpoint or resolve_model_checkpoint(args.model, args.checkpoints_dir)
    model_name = args.model or os.path.basename(os.path.dirname(ckpt)).replace("_res", "")
    print(f"Loading {ckpt} ...", flush=True)
    model, vocab = load_model(ckpt, device)
    print(f"model {model_name!r} "
          f"({sum(p.numel() for p in model.parameters()):,} weights) on {device}\n",
          flush=True)

    # LLM-SRBench pool: 20000 unless overridden.  eval_mymodels.evaluate_xy hardcodes the
    # same 20000 on its no_split path (which is how beam and TPSR run this benchmark), and
    # it is not reachable by any flag there -- so this default is what keeps the three arms
    # on one data budget.  LSR-Synth ships only 4000 train rows, well under the cap, so it
    # is unaffected either way.
    if args.benchmark != "srbench" and args.n_points is None:
        args.n_points = 20000

    if args.benchmark == "srbench":
        run_srbench(model, vocab, device, model_name, args)
    else:
        run_llmsr(model, vocab, device, model_name, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
