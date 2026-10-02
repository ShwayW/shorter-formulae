#!/usr/bin/env python3
"""
score_llmsr_ood.py -- compute OOD R^2 for an LLMSR run's discovered programs, across
noise levels and benchmarks, and cache one CSV per (split, noise) so the comparison
plots can read them instantly.

The llmsr eval saved each discovered program with symbolic params[0..9] but no fitted
constants and no OOD metrics.  To place llmsr on the same OOD-vs-gap axis as the other
methods we re-fit each program's params (BFGS, exactly as llm_methods/llmsr/searcher.py --
no LLM calls), then evaluate the fitted program on each
datasets/<bench>_ood_g<gap>.pkl.gz point cloud.

NOISE.  A run at target-noise tau searched on TRAIN whose targets carried
y += N(0, tau*rms(y)) noise (eval_llmsr.add_target_noise).  To keep the comparison fair
with the transformer baselines -- whose constants ARE the noise-affected model output --
we re-fit constants on the SAME noise level tau (not the clean train), reconstructed
deterministically per (equation, seed) so it is reproducible.  The exact per-problem
noise realisation the run used is not stored, so for tau>0 the re-fit reproduces the
stored id-R^2 only approximately (reported, not gated); tau=0 reproduces it exactly.
Pass --fit-clean to fit on the clean train instead.

Selected by --results-root / --method / --algorithm, so this scores any LLMSR run
(Gemini by default, Llama, ...).  Writes, for every (split, noise):
    <output-dir>/llmsr_<backbone>_ood_raw_<split>_noise<tau>.csv
columns: gap, algorithm, dataset, seed, ood_r2.

Usage:
    # Llama, ALL four noise levels x BOTH benchmarks (the cached-plot workflow):
    python score_llmsr_ood.py \
        --results-root results/llmsr_llama --method llmsr-llama31-8b-vllm \
        --algorithm llmsr-llama
    # narrow it:
    python score_llmsr_ood.py ... --split feynman --noise 0 0.1 --gaps 0 1 2 4
"""
import argparse
import gzip
import os
import pickle
import sys
import zlib

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from eval_llmsr import (load_feynman_problems, load_problems, output_metrics,  # noqa: E402
                        add_target_noise)
import ood_transform  # noqa: E402

MAX_NPARAMS = 10
DEFAULT_SPLITS = ["feynman", "lsr_transform"]
DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]
DEFAULT_GAPS = [0, 1, 2, 4, 8, 16, 32, 64, 128]


def _llmsr_artifact(results_root: str, method: str, split: str, noise: float) -> str:
    """Path to an LLMSR run's discovered-programs artifact at target-noise `noise`.

    Feynman uses the SRBench flat-file layout, the LLM-SRBench splits the per-method-dir
    layout -- exactly as eval_llmsr.py wrote them."""
    nd = f"noise_{float(noise):g}"
    if split == "feynman":
        return os.path.join(results_root, nd, f"results_{method}_feynman.pkl.gz")
    return os.path.join(results_root, nd, "llmsrbench", f"{method}_{split}", "results.pkl.gz")


def _noise_rng(seed: int, eqid) -> np.random.Generator:
    """Deterministic, per-(equation, seed) RNG for reconstructing the tau-noised train."""
    return np.random.default_rng([int(seed), zlib.crc32(str(eqid).encode()) & 0xFFFFFFFF])


def _compile_equation(program: str):
    """exec an LLMSR `def equation(...)` program; return the callable or None.

    The namespace mirrors what the program saw AT SEARCH TIME: llm_methods/llmsr/searcher.py
    prepends `eval_spec`, which defines MAX_NPARAMS and a module-level `params` alongside
    numpy, and llm_methods/llmsr/evaluator.py execs program+preamble into one globals dict.
    Code-specialised backbones (Qwen) routinely reference MAX_NPARAMS inside the equation
    body -- valid during search, a NameError here if we only supply np, which silently
    dropped 44/99 feynman programs from the Qwen run before this was matched up.
    """
    ns = {"np": np, "numpy": np,
          "MAX_NPARAMS": MAX_NPARAMS, "params": [1.0] * MAX_NPARAMS}
    try:
        exec(compile(program, "<llmsr_program>", "exec"), ns)  # noqa: S102
    except Exception:
        return None
    return ns.get("equation")


def _cols(arr_no_y: np.ndarray) -> list:
    return [arr_no_y[:, i] for i in range(arr_no_y.shape[1])]


def _fit_params(equation, X_cols: list, y: np.ndarray):
    """Re-fit params exactly like searcher.py: BFGS, init 1s, MSE."""
    from scipy.optimize import minimize

    def loss(p):
        yp = equation(*X_cols, p)
        return np.mean((yp - y) ** 2)

    try:
        res = minimize(loss, np.ones(MAX_NPARAMS), method="BFGS")
    except Exception:
        return None
    if not np.all(np.isfinite(res.x)):
        return None
    return res.x


def _predict(equation, X_cols: list, params: np.ndarray):
    """Evaluate the program; None on any failure OR a wrong-length output (some LLM
    programs return an array whose length != #rows -- mirrors eval_llmsr.safe_predict,
    so a shape-broken program is skipped instead of crashing output_metrics)."""
    n = len(X_cols[0]) if X_cols else 0
    try:
        yp = np.asarray(equation(*X_cols, params), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if yp.shape[0] != n:
        return None
    return yp


def _load_ood(split: str, gap: int, cache: dict):
    """Load (and cache) the OOD point-cloud dict for (split, gap).  The OOD clouds are
    clean / noise-independent, so they are shared across all noise levels."""
    key = (split, gap)
    if key not in cache:
        path = os.path.join(SCRIPT_DIR, "datasets",
                            ood_transform.ood_basename(split, gap))
        cache[key] = None
        if os.path.exists(path):
            with gzip.open(path) as f:
                cache[key] = pickle.load(f)["datasets"]   # {equation_id: {"X","y"}}
    return cache[key]


def score_one(results_root, method, algorithm, split, noise, gaps,
              prob_by_id, ood_cache, out_dir, fit_clean):
    """Score one (split, noise) and write its CSV.  Returns the path, or None if the
    run's artifact for this (split, noise) is absent."""
    artifact = os.path.join(SCRIPT_DIR, _llmsr_artifact(results_root, method, split, noise))
    tag = f"[{split} tau={float(noise):g}]"
    if not os.path.exists(artifact):
        print(f"{tag} MISSING artifact ({os.path.relpath(artifact, SCRIPT_DIR)}) -- skipped")
        return None
    with gzip.open(artifact) as f:
        recs = pickle.load(f)["results"]

    # Re-fit params (on the tau-noised train unless --fit-clean), keyed (equation, seed)
    # so every seed of a multi-seed run stays distinct.
    fitted, id_check = {}, []
    for r in recs:
        eqid = r["equation_id"]
        seed = int(r.get("seed", 0) or 0)
        prob = prob_by_id.get(eqid)
        eq = _compile_equation(r["discovered_program"])
        if prob is None or eq is None:
            continue
        tr = prob["train"]
        if not fit_clean:
            tr = add_target_noise(tr, float(noise), _noise_rng(seed, eqid))
        te = prob["test"]
        p = _fit_params(eq, _cols(tr[:, 1:]), tr[:, 0])
        if p is None:
            continue
        fitted[(eqid, seed)] = (eq, p)
        yp = _predict(eq, _cols(te[:, 1:]), p)
        if yp is not None:
            r2 = output_metrics(yp, te[:, 0])["r2"]
            id_check.append((r2, (r.get("id_metrics") or {}).get("r2", np.nan)))
    if id_check:
        diffs = np.array([abs(a - b) for a, b in id_check
                          if np.isfinite(a) and np.isfinite(b)])
        exact = (float(noise) == 0.0) or fit_clean
        note = "" if exact else "  (approx: tau>0 re-fits a fresh noise draw)"
        print(f"{tag} fitted {len(fitted)}/{len(recs)}; id-R^2 vs stored: "
              f"median|d|={np.median(diffs):.2e} max|d|={np.max(diffs):.2e}{note}")

    # Score each OOD gap.
    rows = []
    for g in gaps:
        ood = _load_ood(split, g, ood_cache)
        if ood is None:
            print(f"{tag} [gap={g:>3}] missing OOD dataset -- skipped")
            continue
        n = 0
        for (eqid, seed), (eq, p) in fitted.items():
            ds = ood.get(eqid)
            if ds is None:
                continue
            yp = _predict(eq, _cols(np.asarray(ds["X"])), p)
            if yp is None:
                continue
            r2 = output_metrics(yp, np.asarray(ds["y"]))["r2"]
            rows.append({"gap": g, "algorithm": algorithm, "dataset": eqid,
                         "seed": seed, "ood_r2": r2})
            n += 1
        print(f"{tag} [gap={g:>3}] scored {n}")

    backbone = algorithm.replace("llmsr-", "") or algorithm
    out = os.path.join(out_dir,
                       ood_transform.llmsr_backbone_cache_name(backbone, split, noise))
    pd.DataFrame(rows, columns=["gap", "algorithm", "dataset", "seed", "ood_r2"]).to_csv(
        out, index=False)
    print(f"{tag} wrote {len(rows)} rows -> {os.path.relpath(out, SCRIPT_DIR)}\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", default=DEFAULT_SPLITS,
                    choices=["feynman", "lsr_transform"],
                    help="Benchmarks to score (default: both -- SRBench + LLM-SRBench).")
    ap.add_argument("--noise", nargs="+", type=float, default=DEFAULT_NOISES,
                    help="Target-noise levels to score (default: 0 0.001 0.01 0.1).")
    ap.add_argument("--gaps", nargs="+", type=int, default=None,
                    help="Gap values to score. Default: 0 1 2 4 8 16 32 64 128.")
    ap.add_argument("--results-root", default="results/llmsr",
                    help="Root of the LLMSR run to score (default results/llmsr = Gemini; "
                         "e.g. results/llmsr_llama for the self-hosted Llama run).")
    ap.add_argument("--method", default="llmsr-gemini35flash",
                    help="Method-dir/file stem (config 'name' lowercased), e.g. "
                         "llmsr-gemini35flash or llmsr-llama31-8b-vllm.")
    ap.add_argument("--algorithm", default="llmsr-gemini",
                    help="Label written into the 'algorithm' column / output filename "
                         "(e.g. llmsr-gemini, llmsr-llama).")
    ap.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "results"),
                    help="Where the per-(split,noise) CSVs land (default results/).")
    ap.add_argument("--fit-clean", action="store_true",
                    help="Fit constants on the CLEAN train instead of the noise-tau train "
                         "(less fair vs the noise-affected transformer baselines).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.gaps is None:
        args.gaps = ood_transform.default_gaps()

    # Load each split's problems ONCE (feynman reads the ~500MB PMLB tree).
    prob_cache = {}
    for split in args.split:
        probs = load_feynman_problems() if split == "feynman" else load_problems(split)
        prob_cache[split] = {p["name"]: p for p in probs}
        print(f"[{split}] {len(prob_cache[split])} problems loaded")

    ood_cache, written = {}, []
    for split in args.split:
        for noise in args.noise:
            out = score_one(args.results_root, args.method, args.algorithm, split, noise,
                            args.gaps, prob_cache[split], ood_cache, args.output_dir,
                            args.fit_clean)
            if out:
                written.append(out)
    print(f"done: {len(written)} CSV(s) written under "
          f"{os.path.relpath(args.output_dir, SCRIPT_DIR)}/")


if __name__ == "__main__":
    main()
