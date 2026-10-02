#!/usr/bin/env python3
"""
eval_nesymres.py -- NeSymReS 100M (Biggio et al. 2021) on SRBench + LLM-SRBench,
swept over noise levels and seeds, restricted to problems the model can actually
express.

WHY THE <=3 VARIABLE RESTRICTION
    The pretrained 100M checkpoint supports at most 3 input variables.  The SRBench
    wrapper handles wider problems by keeping the top-3 features by |correlation|
    with y -- which silently hands the model a truncated problem it cannot possibly
    solve, and scores the miss against it.  This script instead filters the
    benchmark to equations whose GROUND TRUTH has <= --max-vars variables, so every
    reported number is a fair test.  Counts at the default of 3:
        SRBench Feynman   53 / 101 equations
        LLM-SRBench lsr_transform  35 / 111 equations
    Use --max-vars 0 to disable the filter and fall back to top-k truncation
    (comparable to the SRBench leaderboard entry, but not a fair test of the model).

PROTOCOL (matched to eval_mymodels.py so the numbers sit next to ours)
    * 75/25 train/test split; the test targets stay CLEAN.
    * Target noise on the TRAINING targets only, RMS-scaled:
          y_train += N(0, tau * sqrt(mean(y_train^2)))
      byte-for-byte srbench/experiment/evaluate_model.py and TPSR/evaluate.py, so a
      run at tau is comparable to the ground-truth feather rows at target_noise==tau.
    * R^2 on the clean held-out test; solved = R^2 >= --r2-threshold.
    * Inference is the AUTHORS' code: nesymres Model.fitfunc -- beam search
      (--beam-size) over the decoder, then BFGS constant refinement
      (--n-restarts) on each hypothesis.  We do not reimplement any of it.

OUTPUT (results_io layout, so the existing readers pick it up unchanged)
    <root>/nesymres/seed<N>/noise_<tau>/eval_nesymres.pkl.gz
    <root>/nesymres/seed<N>/noise_<tau>/llmsrbench/nesymres_<split>/results.pkl.gz
    Row schemas match eval_mymodels.py / eval_mymodels.py --benchmark llmsrbench respectively, so
    compare_srbench.py, compare_llmsrbench.py and score_sym_acc.py all read them.

Usage:
    # one (seed, noise) cell, both benchmarks
    python eval_nesymres.py --seeds 42 --target-noise 0

    # full sweep locally
    python eval_nesymres.py --seeds 42 43 44 45 46 47 48 49 50 51 \
                            --target-noise 0 0.001 0.01 0.1

    # one shard of the problem list (cluster array task)
    python eval_nesymres.py --seeds 42 --target-noise 0 --shard 3 --n-shards 8
"""
import argparse
import gzip
import hashlib
import os
import pickle
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import results_io                                                    # noqa: E402
from run_nesymres import NeSymReSRunner, DEFAULT_WEIGHTS, _clipped_r2  # noqa: E402

SRBENCH_ARTIFACT = "eval_nesymres.pkl.gz"
METHOD_NAME = "nesymres"


# -- Problem enumeration -------------------------------------------------------

def feynman_problems(pmlb_dir: str, feynman_csv: str, max_vars: int) -> list:
    """[(dataset_name, n_vars, target_formula, target_pn)] for Feynman equations
    with <= max_vars variables that also exist in the PMLB mirror.

    Variable count comes from the equation CSV (the ground truth), NOT from the
    PMLB column count -- they agree, but the CSV is the authority on what the
    equation actually is.
    """
    import csv
    from eval_mymodels import load_feynman_pn_targets

    pn_targets = load_feynman_pn_targets(feynman_csv)
    ddir = os.path.join(pmlb_dir, "datasets")
    have = set(os.listdir(ddir)) if os.path.isdir(ddir) else set()

    out = []
    with open(feynman_csv, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            fname = row.get("Filename", "").strip()
            formula = row.get("Formula", "").strip()
            if not fname or not formula:
                continue
            n_vars = sum(1 for k in range(1, 11)
                         if row.get(f"v{k}_name", "").strip())
            if n_vars == 0 or (max_vars and n_vars > max_vars):
                continue
            name = "feynman_" + fname.replace(".", "_")
            if name not in have:
                continue
            out.append((name, n_vars, formula, pn_targets.get(name, "")))
    return sorted(out)


def llmsrbench_problems(split: str, max_vars: int) -> list:
    """[(name, n_vars, expression, symbols)] for LLM-SRBench problems with
    <= max_vars input variables.  symbols[0] is the output, symbols[1:] the inputs."""
    import pandas as pd
    from eval_mymodels import resolve_llmsrbench_file

    meta = pd.read_parquet(
        resolve_llmsrbench_file(f"data/{split}-00000-of-00001.parquet"))
    out = []
    for _, e in meta.iterrows():
        syms = list(e["symbols"])
        n_vars = len(syms) - 1
        if n_vars <= 0 or (max_vars and n_vars > max_vars):
            continue
        out.append((str(e["name"]), n_vars, str(e["expression"]), syms))
    return sorted(out, key=lambda t: t[0])


def seed_everything(name: str, seed: int) -> "np.random.RandomState":
    """Seed our rng AND the global numpy/torch state for one (problem, seed) cell.

    The global seeding is not belt-and-braces: the authors' BFGS draws its restart
    initialisations from the GLOBAL numpy RNG (bfgs.py:142,
    `x0 = np.random.randn(len(symbols))`), so without this the same seed gives
    different constants -- and sometimes a different equation -- on every run.
    Observed directly: one LLM-SRBench problem scored R^2 0.41 then 0.00 across two
    runs of the same cell. torch is seeded too so the decoder is pinned as well.
    """
    s = cell_seed(name, seed)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except Exception:
        pass
    return np.random.RandomState(s)


def cell_seed(name: str, seed: int) -> int:
    """Stable per-(problem, seed) RNG seed.

    Deliberately NOT built from hash() -- Python salts str/bytes hashing per
    process, so a hash()-derived seed makes runs unreproducible across processes
    and silently confounds paired comparisons (the trap eval_mymodels.py fell into).
    blake2b is stable across processes, machines and Python versions.
    """
    h = hashlib.blake2b(f"{name}|{seed}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2 ** 31)


def shard(items: list, index: int, total: int) -> list:
    """Round-robin slice.  Round-robin rather than contiguous so every shard gets a
    mix of cheap and expensive problems -- contiguous blocks would put all the
    3-variable equations in the same task."""
    if total <= 1:
        return items
    return [x for i, x in enumerate(items) if i % total == index]


# -- Shared per-problem machinery ----------------------------------------------

def add_target_noise(y_train: np.ndarray, tau: float, rng) -> np.ndarray:
    """y_train += N(0, tau * RMS(y_train)); returns y_train unchanged when tau == 0.

    tau == 0 consumes no randomness, so a noise-free run is bit-identical to the
    pipeline with the noise step absent (same invariant eval_mymodels.py relies on).
    """
    if not tau or tau <= 0 or len(y_train) == 0:
        return y_train
    rms = float(np.sqrt(np.mean(np.square(y_train.astype(np.float64)))))
    if rms <= 0:
        return y_train
    return y_train + rng.normal(0.0, tau * rms, size=y_train.shape)


_NSR_VAR = __import__("re").compile(r"(?<![A-Za-z0-9_])x_(\d+)(?![A-Za-z0-9_])")


def to_v_space(formula, top_k_idx):
    """Rewrite NeSymReS's `x_j` into our `v<i+1>` column space.

    Two things have to be undone, and getting either wrong silently breaks every
    symbolic comparison while leaving R^2 untouched (predict() applies the same
    indexing, so the fit still looks right):
      1. NeSymReS numbers variables from x_1, not x_0 as eval_e2e.py does.
      2. `x_j` names the j-th SELECTED feature, not the j-th input column --
         get_top_k_features reorders by |corr(x, y)| whenever the problem is wider
         than the model's 3 slots. top_k_idx maps back.
    With --max-vars 3 the selection is the identity (get_top_k_features short-circuits
    when X has <= k columns), but storing the mapped form keeps the artifact correct
    if the filter is ever relaxed.
    """
    if not formula or top_k_idx is None:
        return formula
    idx = list(top_k_idx)

    def sub(m):
        j = int(m.group(1)) - 1                # x_1 is the first selected feature
        return f"v{idx[j] + 1}" if 0 <= j < len(idx) else m.group(0)

    return _NSR_VAR.sub(sub, str(formula))


def _load_ood(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh).get("datasets", {})


# -- SRBench (Feynman) ---------------------------------------------------------

def run_srbench(runner, problems, seed, tau, args) -> list:
    out_path = os.path.join(
        results_io.noise_dir(
            results_io.seed_dir(os.path.join(args.output, METHOD_NAME), seed), tau),
        SRBENCH_ARTIFACT)

    meta, rows = results_io.load_results(out_path)
    done = {r["dataset"] for r in rows}
    if done:
        print(f"  resuming: {len(done)} datasets already in {out_path}", flush=True)

    ood = _load_ood(args.ood_data)

    for i, (name, n_vars, formula, target_pn) in enumerate(problems, 1):
        if name in done:
            continue
        # Per-(dataset, seed) rng: the split subsample, the noise draw, and the
        # model's own sampling all come from here, so a given cell is reproducible
        # regardless of shard layout or resume point.
        rng = seed_everything(name, seed)
        try:
            import pandas as pd
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
        raw = runner.fit(X_tr, y_tr, rng)
        dt = time.perf_counter() - t0
        pred = to_v_space(raw, runner.top_k_idx_)

        r2 = _clipped_r2(y_te, runner.predict(X_te))
        ood_r2 = None
        if name in ood:
            ood_r2 = _clipped_r2(ood[name]["y"], runner.predict(ood[name]["X"]))

        rows.append({
            "dataset": name, "seed": seed, "round": 1,
            "r2": r2, "ood_r2": ood_r2, "time": dt,
            "predicted_formula": pred, "predicted_formula_raw": raw,
            "feature_idx": list(runner.top_k_idx_ or []),
            "target_formula": formula, "target_pn": target_pn,
            "n_vars": n_vars,
        })
        results_io.save_results(
            out_path, rows, model_name=METHOD_NAME, checkpoint=str(args.weights_path),
            target_noise=tau, seed=seed, max_vars=args.max_vars,
            beam_size=args.beam_size, n_restarts=args.n_restarts)
        print(f"  [{i}/{len(problems)}] {name:<28} R^2={r2:.4f} "
              f"{'solved' if r2 >= args.r2_threshold else '      '} "
              f"t={dt:.1f}s | {pred}", flush=True)
    return rows


# -- LLM-SRBench ---------------------------------------------------------------

def run_llmsrbench(runner, problems, seed, tau, args) -> list:
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
            # Column 0 is y, columns 1: are X (eval_mymodels.py --benchmark llmsrbench's layout).
            X_tr, y_tr = train[:, 1:], train[:, 0]
            X_te, y_te = test[:, 1:], test[:, 0]
            y_tr = add_target_noise(y_tr, tau, rng)

            t0 = time.perf_counter()
            raw = runner.fit(X_tr, y_tr, rng)
            dt = time.perf_counter() - t0
            pred = to_v_space(raw, runner.top_k_idx_)

            r2 = _clipped_r2(y_te, runner.predict(X_te))
            ood = samples.get("ood_test")
            ood_metrics = None
            if ood is not None:
                ood_metrics = {"r2": _clipped_r2(ood[:, 0], runner.predict(ood[:, 1:]))}

            rows.append({
                "equation_id": name, "gt_equation": expression,
                "discovered_equation": pred, "discovered_equation_raw": raw,
                "feature_idx": list(runner.top_k_idx_ or []), "n_vars": n_vars,
                "num_datapoints": int(len(train)), "num_eval_datapoints": int(len(test)),
                "search_time": dt, "seed": seed,
                "id_metrics": {"r2": r2}, "ood_metrics": ood_metrics,
                "symbols": list(map(str, syms)),
            })
            results_io.save_results(
                out_path, rows, model_name=METHOD_NAME, split=args.split,
                checkpoint=str(args.weights_path), target_noise=tau, seed=seed,
                max_vars=args.max_vars, beam_size=args.beam_size,
                n_restarts=args.n_restarts)
            print(f"  [{i}/{len(problems)}] {name:<28} R^2={r2:.4f} "
                  f"{'solved' if r2 >= args.r2_threshold else '      '} "
                  f"t={dt:.1f}s | {pred}", flush=True)
    return rows


# -- Reporting -----------------------------------------------------------------

def summarise(tag: str, rows: list, thr: float, key="r2") -> None:
    vals = []
    for r in rows:
        v = r.get(key)
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
    ap.add_argument("--split", default="lsr_transform", help="LLM-SRBench split.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Seeds to run. The canonical sweep is 42..51.")
    ap.add_argument("--target-noise", type=float, nargs="+", default=[0.0],
                    dest="noises", help="Noise level(s) tau. Canonical: 0 0.001 0.01 0.1")
    ap.add_argument("--max-vars", type=int, default=3,
                    help="Keep only equations with <= this many variables (the "
                         "pretrained model supports 3). 0 disables the filter.")
    ap.add_argument("--output", default="results", help="Results root.")
    ap.add_argument("--pmlb-dir", default="datasets/pmlb")
    ap.add_argument("--feynman-csv", default="datasets/feynman/FeynmanEquations.csv")
    ap.add_argument("--ood-data", default=None,
                    help="feynman_ood_g*.pkl.gz to also score the formula on.")
    ap.add_argument("--n-points", type=int, default=10000,
                    help="Rows subsampled per Feynman equation before the 75/25 "
                         "split. Default 10000 -> a 2500-row clean test set.")
    ap.add_argument("--r2-threshold", type=float, default=0.99)
    # Model knobs -- defaults are the SRBench wrapper's.
    ap.add_argument("--weights-path", default=os.environ.get(
        "NESYMRES_WEIGHTS", str(DEFAULT_WEIGHTS)))
    ap.add_argument("--device", default=None,
                    help="cuda/cpu. Default: cuda when available.")
    ap.add_argument("--beam-size", type=int, default=2,
                    help="Decoder beam width (the authors' beam search). Default 2.")
    ap.add_argument("--n-restarts", type=int, default=10,
                    help="BFGS restarts per hypothesis. Default 10.")
    ap.add_argument("--max-fit-points", type=int, default=500,
                    help="Points fed to the set encoder (its trained regime).")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--list-problems", action="store_true",
                    help="Print the filtered problem list and exit.")
    args = ap.parse_args()

    do_sr = args.benchmark in ("srbench", "both")
    do_lsr = args.benchmark in ("llmsrbench", "both")

    sr_problems = (feynman_problems(args.pmlb_dir, args.feynman_csv, args.max_vars)
                   if do_sr else [])
    lsr_problems = (llmsrbench_problems(args.split, args.max_vars)
                    if do_lsr else [])

    if args.list_problems:
        for n, k, f, _ in sr_problems:
            print(f"srbench    {n:<28} {k}v  {f}")
        for n, k, e, _ in lsr_problems:
            print(f"llmsrbench {n:<28} {k}v  {e}")
        print(f"\nsrbench: {len(sr_problems)}   llmsrbench: {len(lsr_problems)}")
        return

    sr_problems = shard(sr_problems, args.shard, args.n_shards)
    lsr_problems = shard(lsr_problems, args.shard, args.n_shards)

    print(f"NeSymReS 100M | max_vars<={args.max_vars} | "
          f"shard {args.shard}/{args.n_shards} | "
          f"srbench {len(sr_problems)} + llmsrbench {len(lsr_problems)} problems | "
          f"seeds {args.seeds} | noise {args.noises}", flush=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    runner = NeSymReSRunner(
        weights_path=args.weights_path, device=device,
        beam_size=args.beam_size, n_restarts=args.n_restarts,
        max_fit_points=args.max_fit_points)

    for seed in args.seeds:
        for tau in args.noises:
            print(f"\n=== seed {seed} | noise {tau:g} ===", flush=True)
            if sr_problems:
                rows = run_srbench(runner, sr_problems, seed, tau, args)
                summarise("srbench", rows, args.r2_threshold)
            if lsr_problems:
                rows = run_llmsrbench(runner, lsr_problems, seed, tau, args)
                summarise("llmsrbench", rows, args.r2_threshold)


if __name__ == "__main__":
    main()
