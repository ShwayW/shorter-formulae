#!/usr/bin/env python3
"""
eval_llmsr.py -- Evaluate LLMSR (Shojaee et al.) with a Gemini backend on the
LLM-SRBench benchmark, in the same on-disk format as eval_mymodels.py --benchmark llmsrbench /
eval_e2e.py / eval_e2e_tpsr.py so results drop into the shared results/ tree and
the compare_*/aggregate_seeds tooling picks them up alongside mymodels/e2e/tpsr.

Unlike the transformer baselines, the "searcher" here is the LLMSR evolutionary
loop (llm_methods/llmsr): an LLM proposes Python `equation(...)` skeletons, each is
scored by fitting its constants to the TRAIN split via BFGS, and the best ones are
fed back into the prompt. We then score the discovered equation on the held-out
TEST (and OOD when present) with the same r2/nmse/mse metrics as the official
harness (bench/pipelines.py compute_output_base_metrics).

Data (same convention as eval_mymodels.py --benchmark llmsrbench): local $LLMSRBENCH_DIR HDF5 + parquet,
column 0 = output y, columns 1: = input variables X in `symbols` order.

Usage:
    python eval_llmsr.py --searcher_config configs/llmsr_gemini35flash.yaml \
                         --split lsr_transform [--seed 0] [--n-seeds 1] \
                         [--target-noise TAU] [--problem NAME] [--max-problems N] \
                         [--global-max-sample-num N] [--output results]

Results land in <output>/noise_<TAU>/llmsrbench/<method>/results.pkl.gz.
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import yaml

# LLMSR (llm_methods/llmsr) imports itself as the top-level package `llmsr` and imports
# the harness as `bench.*`; make both importable regardless of the caller's cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "llm_methods"))
sys.path.insert(0, _HERE)

from results_io import (llmsr_path, llmsr_dir, srbench_path, load_results, save_results)
from bench.dataclasses import SEDTask

warnings.filterwarnings("ignore")

REPO_ID = "nnheui/llm-srbench"

# Local copy of the benchmark (download_llmsrbench.py); override with $LLMSRBENCH_DIR.
LLMSRBENCH_DIR = os.environ.get(
    "LLMSRBENCH_DIR",
    os.path.join(_HERE, "datasets", "llmsrbench"),
)

# SRBench Feynman (PMLB mirror) -- used when --split feynman.
FEYNMAN_DATASETS_DIR = os.path.join(_HERE, "datasets", "pmlb", "datasets")
FEYNMAN_CSV = os.path.join(_HERE, "datasets", "feynman", "FeynmanEquations.csv")


def _artifact_path(output: str, tau: float, split: str, method: str,
                   num_shards: int = 1, shard_id: int = 0) -> str:
    """Path to a (possibly sharded) results artifact.

    Feynman is SRBench, so it uses the flat-file convention (like eval_e2e.py):
    <output>/noise_<tau>/results_<method>.pkl.gz. The LLM-SRBench splits use the
    per-method directory convention: <output>/noise_<tau>/llmsrbench/<method>/results.pkl.gz.
    A shard suffix keeps concurrent writers from clobbering the same file.
    """
    shard = "" if num_shards == 1 else f"__shard{shard_id:02d}of{num_shards:02d}"
    if split == "feynman":
        return srbench_path(output, tau, f"results_{method}{shard}.pkl.gz")
    return llmsr_path(output, tau, f"{method}{shard}")


def resolve_llmsrbench_file(rel_path: str) -> str:
    """Prefer the local $LLMSRBENCH_DIR copy; fall back to hf_hub_download."""
    local = os.path.join(LLMSRBENCH_DIR, rel_path)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, rel_path, repo_type="dataset")


# -- Data loading -------------------------------------------------------------

def load_problems(split: str):
    """Load LLM-SRBench problems for a split.

    Returns dicts with name, symbols, symbol_descs, symbol_properties, expression,
    and train/test/ood_test float64 arrays (col 0 = y, cols 1: = X). LLMSR needs
    symbol_descs/symbol_properties (the transformer loader does not) to write the
    natural-language variable descriptions into its prompt and pick input columns.
    """
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
                "name":              e["name"],
                "symbols":           list(e["symbols"]),
                "symbol_descs":      list(e["symbol_descs"]),
                "symbol_properties": list(e["symbol_properties"]),
                "expression":        e["expression"],
                "train":             samples.get("train"),
                "test":              samples.get("test"),
                "ood_test":          samples.get("ood_test"),
            })
    return problems


# -- Feynman (SRBench / PMLB) loading -----------------------------------------

def _load_feynman_meta(csv_path: str) -> dict:
    """{Filename -> {output, formula, var_names, n_vars}} from FeynmanEquations.csv."""
    import csv as _csv
    meta = {}
    if not os.path.isfile(csv_path):
        return meta
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            key = (row.get("Filename") or "").strip()
            if not key:
                continue
            n = int(row.get("# variables", "0") or 0)
            var_names = [(row.get(f"v{i}_name") or "").strip()
                         for i in range(1, n + 1) if (row.get(f"v{i}_name") or "").strip()]
            meta[key] = {"output": (row.get("Output") or "y").strip() or "y",
                         "formula": (row.get("Formula") or "").strip(),
                         "var_names": var_names, "n_vars": n}
    return meta


def _san(name: str, fallback: str) -> str:
    """Make `name` a valid Python identifier (LLMSR emits it into a def signature)."""
    import re
    s = re.sub(r"\W", "_", (name or "").strip())
    if not s or s[0].isdigit():
        s = "v_" + s
    return s or fallback


def load_feynman_problems(n_points: int = 20000):
    """Load SRBench Feynman problems in the same schema as load_problems().

    Reads PMLB .tsv.gz (col 0..-2 = X, last = y) and FeynmanEquations.csv (variable
    names + ground-truth formula). y is moved to column 0 to match the LLM-SRBench
    convention. Data is subsampled to n_points (fixed seed, so the train/test split
    is identical across search seeds) and split 75/25 like eval_e2e.py.

    Only datasets present in FeynmanEquations.csv are returned (99 of them). The 20
    black-box feynman_test_* dirs are absent from the CSV, so they are dropped here:
    LLMSR needs variable names + a ground-truth equation to build its prompt/score.
    NOTE: Feynman metadata has no natural-language descriptions, so symbol_descs
    fall back to the variable names.
    """
    import gzip
    import csv as _csv
    meta = _load_feynman_meta(FEYNMAN_CSV)
    rng = np.random.default_rng(0)   # fixed data split, independent of the search seed
    problems = []
    dirs = sorted(d for d in os.listdir(FEYNMAN_DATASETS_DIR)
                  if d.startswith("feynman_")
                  and os.path.isdir(os.path.join(FEYNMAN_DATASETS_DIR, d)))
    for label in dirs:
        key = label.replace("feynman_", "").replace("_", ".")
        m = meta.get(key)
        if m is None:
            continue  # no metadata (black-box feynman_test_* or CSV-absent) -> skip

        ddir = os.path.join(FEYNMAN_DATASETS_DIR, label)
        cands = [f for f in os.listdir(ddir) if f.endswith(".tsv.gz")]
        if not cands:
            continue
        with gzip.open(os.path.join(ddir, cands[0]), "rt") as fh:
            reader = _csv.reader(fh, delimiter="\t")
            header = next(reader)
            data = np.array([r for r in reader], dtype=np.float64)

        X, y = data[:, :-1], data[:, -1]
        col_names = header[:-1]
        var_names = m["var_names"] if len(m["var_names"]) == X.shape[1] else col_names
        var_names = [_san(v, f"x{i}") for i, v in enumerate(var_names)]
        out_name = _san(m["output"], "y")

        arr = np.column_stack([y, X])                 # col 0 = output y
        if len(arr) > n_points:
            arr = arr[rng.choice(len(arr), size=n_points, replace=False)]
        n_train = int(0.75 * len(arr))

        problems.append({
            "name":              label,
            "symbols":           [out_name] + var_names,
            "symbol_descs":      [out_name] + var_names,   # CSV has no NL descriptions
            "symbol_properties": ["O"] + ["V"] * len(var_names),
            "expression":        m["formula"],
            "train":             arr[:n_train],
            "test":              arr[n_train:],
            "ood_test":          None,
        })
    return problems


# -- Prediction + metrics (mirror bench/pipelines.py) -------------------------

def safe_predict(lambda_fn, X: np.ndarray) -> np.ndarray:
    """Evaluate the discovered equation on X (N, D); NaN array on any failure.

    The searcher's lambda runs the program in a subprocess sandbox and can return
    None (timeout / exec error) or a wrong-shaped array, so guard every call."""
    n = X.shape[0]
    try:
        y = lambda_fn(X)
        if y is None:
            return np.full(n, np.nan)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if y.shape[0] != n:
            return np.full(n, np.nan)
        return y
    except Exception:
        return np.full(n, np.nan)


def output_metrics(y_pred: np.ndarray, y: np.ndarray) -> dict:
    """Same metrics as the official harness, NaN-robust."""
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


def add_target_noise(train: np.ndarray, tau: float, rng) -> np.ndarray:
    """Return a copy of `train` with Gaussian noise on the target column (col 0):
    y += N(0, tau*sqrt(mean(y^2)))  (SRBench/TPSR convention). Test/OOD stay clean."""
    if not tau:
        return train
    out = train.copy()
    y = out[:, 0]
    scale = tau * np.sqrt(np.mean(y ** 2))
    out[:, 0] = y + rng.normal(0.0, scale, size=y.shape)
    return out


# -- Searcher construction ----------------------------------------------------

def build_searcher(cfg: dict, global_max_sample_num: int, log_path: str):
    """Construct an LLMSRSearcher wired to the Gemini sampler from the yaml config."""
    from llmsr import config as llmsr_config
    from llmsr import sampler as llmsr_sampler
    from llmsr.searcher import LLMSRSearcher

    exp_conf = llmsr_config.ExperienceBufferConfig(num_islands=cfg["num_islands"])
    llmsr_cfg = llmsr_config.Config(
        experience_buffer=exp_conf,
        use_api=True,                       # gpt-style body trimming in _extract_body
        api_model=cfg["api_model"],
        samples_per_prompt=cfg["samples_per_prompt"],
        early_stop_train_r2=cfg.get("early_stop_train_r2", None),
    )

    api_type = cfg.get("api_type", "gemini")
    if api_type == "gemini":
        sampler_class = lambda spp: llmsr_sampler.GeminiLLM(
            samples_per_prompt=spp,
            api_model=cfg["api_model"],
            project=cfg.get("project"),
            location=cfg.get("location", "global"),
            vertexai=cfg.get("vertexai", True),
            thinking_budget=cfg.get("thinking_budget", None),
        )
    elif api_type in ("local", "openai", "vllm"):
        # Self-hosted OpenAI-compatible server (e.g. `vllm serve <model>`). This is
        # NOT the paid OpenAI cloud: the sampler's AsyncOpenAI client is pointed at
        # api_url on the local node and api_key is a placeholder the server ignores,
        # so there is no billing. NB: LocalLLM only takes the OpenAI path when api_url
        # contains "localhost" or "openai" -- keep the vLLM host as localhost.
        # Prefer $LLMSR_API_URL (the sbatch wrapper sets it to this job's unique vLLM
        # port so co-located jobs don't clash on 8000); fall back to the config's
        # api_url. Must contain "localhost"/"openai" for LocalLLM's OpenAI path.
        api_url = os.environ.get("LLMSR_API_URL", cfg["api_url"])
        # api_timeout: the sampler retries a timed-out request forever, so the client
        # timeout must clear the server's slowest completion. 60s (the upstream default)
        # is fine for an 8B model; a 32B under heavy batching needs more.
        sampler_class = lambda spp: llmsr_sampler.LocalLLM(
            samples_per_prompt=spp,
            api_url=api_url,
            api_key=cfg.get("api_key", "EMPTY"),
            timeout=float(cfg.get("api_timeout", 60.0)),
        )
    else:
        raise ValueError(f"eval_llmsr.py wires only the 'gemini' and 'local' (vLLM) "
                         f"backends; got api_type={api_type!r}")

    return LLMSRSearcher(cfg["name"], llmsr_cfg, sampler_class,
                         global_max_sample_num=global_max_sample_num,
                         log_path=log_path)


def merge_shards(output: str, tau: float, split: str, method: str,
                 num_shards: int, **meta) -> None:
    """Fold the per-shard artifacts into the canonical results file for (split, method).

    Rows are de-duplicated on (equation_id, seed); a valid discovered_equation wins
    over a None/empty one. Shard artifacts are removed once merged. Works for both
    the Feynman flat-file layout and the LLM-SRBench per-method-dir layout."""
    import shutil

    canonical = _artifact_path(output, tau, split, method)  # num_shards=1 -> no suffix

    # Seed from the canonical file BEFORE the shards. Merging is destructive twice over
    # otherwise: it deletes every shard artifact when it finishes, so a chained follow-on
    # job (--dependency=afterany fires on COMPLETED too) restarts the cell from zero, and
    # if its shards then die early -- OOM, a dead vLLM server -- run_noise still calls the
    # merge and would overwrite a complete canonical with the few rows that survived, or
    # with nothing at all. Seeding here makes the merge purely additive: the row-preference
    # rule below only replaces a row whose discovered_equation is missing, so a re-run can
    # fill gaps but can never delete an answer we already have.
    merged = {}
    _, prior_rows = load_results(canonical)
    for r in prior_rows:
        merged[(r.get("equation_id"), r.get("seed"))] = r
    n_prior = len(merged)

    n_found = 0
    for i in range(num_shards):
        sp = _artifact_path(output, tau, split, method, num_shards, i)
        _, rows = load_results(sp)
        if rows:
            n_found += 1
        for r in rows:
            key = (r.get("equation_id"), r.get("seed"))
            prev = merged.get(key)
            if prev is None or prev.get("discovered_equation") in (None, "None", ""):
                merged[key] = r

    save_results(canonical, list(merged.values()), method=method, split=split, **meta)
    print(f"[merge] {n_found}/{num_shards} shards + {n_prior} prior "
          f"-> {len(merged)} rows -> {canonical}", flush=True)

    for i in range(num_shards):
        if split == "feynman":
            sp = _artifact_path(output, tau, split, method, num_shards, i)
            if os.path.isfile(sp):
                os.remove(sp)
        else:
            d = llmsr_dir(output, tau, f"{method}__shard{i:02d}of{num_shards:02d}")
            if os.path.isdir(d):
                shutil.rmtree(d)
    print(f"[merge] removed {num_shards} shard artifacts", flush=True)


# -- Main ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--searcher_config", type=str, required=True,
                   help="YAML config, e.g. configs/llmsr_gemini35flash.yaml")
    p.add_argument("--benchmark", choices=["srbench", "llmsrbench"], default=None,
                   help="Which benchmark to run -- the spelling every eval_<method>.py "
                        "uses. Shorthand for --split: srbench => feynman, llmsrbench => "
                        "lsr_transform. Pass --split directly to pick a specific "
                        "LLM-SRBench split. Default: inferred from --split.")
    p.add_argument("--split", type=str, default=None,
                   help="Dataset: LLM-SRBench split (lsr_transform, lsr_synth_<domain>) "
                        "or 'feynman' for the SRBench Feynman (PMLB) benchmark. "
                        "Default: lsr_transform, or feynman with --benchmark srbench.")
    p.add_argument("--n-points", type=int, default=20000,
                   help="feynman only: rows subsampled from the 100k-row PMLB file "
                        "before the 75/25 train/test split (default 20000).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed. Seeds used are seed, seed+1, ..., seed+n_seeds-1.")
    p.add_argument("--n-seeds", "--n_seeds", dest="n_seeds", type=int, default=1,
                   help="Number of seeds scored in one process. Rows keyed "
                        "(equation_id, seed) and resumed per pair.")
    p.add_argument("--target-noise", type=float, default=0.0, metavar="TAU",
                   help="Gaussian noise on TRAIN targets before search: "
                        "y_train += N(0, TAU*sqrt(mean(y^2))). Test/OOD scored clean.")
    p.add_argument("--global-max-sample-num", type=int, default=None,
                   help="Override the config's LLM-call budget per problem "
                        "(lower it for cheap smoke tests).")
    p.add_argument("--early-stop-train-r2", type=float, default=None, metavar="R2",
                   help="Override the config's early_stop_train_r2: stop a problem once "
                        "the best skeleton's TRAIN R^2 exceeds R2. Not passed -> use the "
                        "config value (default configs disable it). Pass a value <= 0 to "
                        "force-disable even when the config enables it.")
    p.add_argument("--problem", type=str, default=None,
                   help="Evaluate only this problem (by name).")
    p.add_argument("--max-problems", type=int, default=None,
                   help="Cap the number of problems (for quick smoke tests).")
    p.add_argument("--method-name", type=str, default=None,
                   help="Override the method dir name (default: derived from config 'name').")
    p.add_argument("--output", type=str, default="results",
                   help="Seed root. Results -> <output>/noise_<TAU>/llmsrbench/<method>/results.pkl.gz")
    # -- Parallelism: run N processes, each --shard-id in [0, N), on disjoint
    # problems (stride slicing). Each writes its own <method>__shardIofN artifact so
    # concurrent whole-file saves never clobber. Then run once with --merge-shards to
    # fold them into the canonical <method>/results.pkl.gz and delete the shard dirs.
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of parallel shards (processes). Default 1 (no sharding).")
    p.add_argument("--shard-id", type=int, default=0,
                   help="This process's shard index in [0, num_shards).")
    p.add_argument("--merge-shards", action="store_true", default=False,
                   help="Merge all shard artifacts for (split, method, noise) into the "
                        "canonical results.pkl.gz, then remove the shard dirs. Do NOT "
                        "pass while shards are still running.")
    args = p.parse_args()
    # --benchmark and --split are two views of one choice; reconcile them here so
    # the rest of the script keeps working purely off args.split.
    _bench_split = {"srbench": "feynman", "llmsrbench": "lsr_transform"}
    if args.benchmark is not None:
        _implied = _bench_split[args.benchmark]
        if args.split is None:
            args.split = _implied
        elif (args.split == "feynman") != (args.benchmark == "srbench"):
            p.error(f"--benchmark {args.benchmark} and --split {args.split} disagree "
                    f"(--benchmark {args.benchmark} means --split {_implied}"
                    + (" or another lsr_* split)." if args.benchmark == "llmsrbench" else ")."))
    if args.split is None:
        args.split = "lsr_transform"
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard-id must be in [0, {args.num_shards}); got {args.shard_id}")

    with open(args.searcher_config) as f:
        cfg = yaml.safe_load(f)

    global_max_sample_num = (args.global_max_sample_num
                             if args.global_max_sample_num is not None
                             else cfg["global_max_sample_num"])

    # Early stop: CLI overrides config; neither -> disabled. A value <= 0 force-disables.
    early_stop = (args.early_stop_train_r2 if args.early_stop_train_r2 is not None
                  else cfg.get("early_stop_train_r2", None))
    if early_stop is not None and early_stop <= 0:
        early_stop = None
    cfg["early_stop_train_r2"] = early_stop   # build_searcher reads this back

    cfg_name = (args.method_name or cfg.get("name", "llmsr")).lower()
    _method = f"{cfg_name}_{args.split}"
    tau = args.target_noise or 0.0

    # --merge-shards: fold shard artifacts into the canonical one and exit.
    if args.merge_shards:
        merge_shards(args.output, tau, args.split, _method, args.num_shards,
                     model_name=cfg_name,
                     api_model=cfg["api_model"], global_max_sample_num=global_max_sample_num)
        return

    # Each shard writes its own artifact; a single (unsharded) run writes the canonical one.
    shard_tag = ("" if args.num_shards == 1
                 else f"__shard{args.shard_id:02d}of{args.num_shards:02d}")
    results_path = _artifact_path(args.output, tau, args.split, _method,
                                  args.num_shards, args.shard_id)

    # Load benchmark.
    print(f"Loading split '{args.split}' ...", flush=True)
    if args.split == "feynman":
        problems = load_feynman_problems(n_points=args.n_points)
    else:
        problems = load_problems(args.split)
    if args.problem is not None:
        problems = [q for q in problems if q["name"] == args.problem]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    if args.num_shards > 1:
        problems = problems[args.shard_id::args.num_shards]  # stride slice balances load
        print(f"  shard {args.shard_id}/{args.num_shards}", flush=True)
    print(f"  {len(problems)} problems.\n", flush=True)
    print(f"Backend: {cfg['api_model']} ({cfg.get('api_type')})  "
          f"budget: {global_max_sample_num} LLM calls/problem  "
          f"samples/prompt: {cfg['samples_per_prompt']}  islands: {cfg['num_islands']}  "
          f"early_stop_train_r2: {early_stop if early_stop is not None else 'off'}\n",
          flush=True)

    seeds = list(range(args.seed, args.seed + args.n_seeds))

    # Resume: keyed on (equation_id, seed); a missing/None equation counts as not done.
    # We read this shard's artifact FIRST, then fall back to the canonical file. The
    # fallback matters for chained jobs: a merge deletes the shard artifacts, so a link
    # that starts after a cell finished would otherwise see no history and re-run the
    # whole cell from scratch. Canonical rows are filtered to this shard's stride slice
    # so a shard never adopts another's rows.
    _mine = {q["name"] for q in problems}
    _sources = [results_path]
    if args.num_shards > 1:
        _sources.append(_artifact_path(args.output, tau, args.split, _method))
    done_pairs, existing_rows = set(), []
    for _src in _sources:
        for _r in load_results(_src)[1]:
            _key = (_r.get("equation_id"), _r.get("seed"))
            _eq = _r.get("discovered_equation")
            if _key in done_pairs or _eq in (None, "None", ""):
                continue
            if _r.get("equation_id") not in _mine:
                continue
            done_pairs.add(_key)
            existing_rows.append(_r)
    if done_pairs:
        print(f"Resuming: {len(done_pairs)} (problem, seed) pairs already done "
              f"in {' + '.join(_sources)}", flush=True)

    rows = list(existing_rows)

    def _save():
        save_results(results_path, rows, model_name=cfg_name, split=args.split,
                     method=_method, target_noise=args.target_noise or 0.0,
                     api_model=cfg["api_model"], global_max_sample_num=global_max_sample_num)

    _save()

    # Per-seed search-log root (tensorboard/json the searcher writes); kept out of
    # results/ so it does not pollute the artifact tree.
    # Include the noise level so parallel per-noise jobs of the same (split, seed) do not
    # write their tensorboard/json search logs into the same directory.
    search_log_root = os.path.join(_HERE, "logs", "llmsr_search",
                                   f"{_method}_noise{tau:g}" + shard_tag)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        seed_log_path = os.path.join(search_log_root, f"seed{seed}")
        os.makedirs(seed_log_path, exist_ok=True)
        searcher = build_searcher(cfg, global_max_sample_num, seed_log_path)

        for i, q in enumerate(problems):
            if (q["name"], seed) in done_pairs:
                continue
            train, test = q["train"], q["test"]
            n_vars = train.shape[1] - 1

            train_seed = add_target_noise(train, args.target_noise, rng)
            task = SEDTask(
                name=f"{q['name']}_seed{seed}",   # unique log dir per (problem, seed)
                symbols=q["symbols"],
                symbol_descs=q["symbol_descs"],
                symbol_properties=q["symbol_properties"],
                samples=train_seed,
            )

            t0 = time.perf_counter()
            try:
                search_results = searcher.discover(task)
                result = search_results[0]
                lambda_fn = result.equation.lambda_format
                discovered = result.equation.program_format
                aux = dict(result.aux) if result.aux else {}
            except Exception as e:
                print(f"[seed {seed}] {q['name']}: search failed: {e}", flush=True)
                lambda_fn, discovered, aux = None, None, {}
            elapsed = time.perf_counter() - t0

            X_test, y_test = test[:, 1:], test[:, 0]
            id_m = (output_metrics(safe_predict(lambda_fn, X_test), y_test)
                    if lambda_fn is not None else output_metrics(np.full_like(y_test, np.nan), y_test))
            ood_m = None
            if q["ood_test"] is not None and lambda_fn is not None:
                ood = q["ood_test"]
                ood_m = output_metrics(safe_predict(lambda_fn, ood[:, 1:]), ood[:, 0])

            log = {
                "equation_id": q["name"],
                "gt_equation": q["expression"],
                "discovered_equation": discovered,
                "discovered_program": discovered,
                "n_vars": n_vars,
                "num_datapoints": int(len(train)),
                "num_eval_datapoints": int(len(test)),
                "search_time": elapsed,
                "seed": seed,
                "id_metrics": id_m,
                "ood_metrics": ood_m,
                **aux,
            }
            rows.append(log)
            _save()
            print(f"[seed {seed}] [{i+1}/{len(problems)}] {q['name']:<24} "
                  f"R^2={id_m['r2']:.4f}  NMSE={id_m['nmse']:.3e}  t={elapsed:.1f}s",
                  flush=True)

    # -- Summary --------------------------------------------------------------
    r2s = np.array([r["id_metrics"]["r2"] for r in rows], dtype=np.float64)
    r2s_valid = r2s[np.isfinite(r2s)]
    print(f"\n{'='*70}")
    print(f"LLM-SRBench {args.split} -- {cfg_name}  ({len(rows)} problems)")
    print(f"{'='*70}")
    if r2s_valid.size:
        for thr in (0.99, 0.999):
            acc = float(np.mean(r2s_valid >= thr)) * 100
            print(f"  Acc (R^2 >= {thr})  : {acc:.1f}%")
        print(f"  Mean R^2          : {np.mean(r2s_valid):.4f}")
        print(f"  Median R^2        : {np.median(r2s_valid):.4f}")
        print(f"  Valid / total    : {r2s_valid.size}/{len(rows)}")
    print(f"{'='*70}")
    print(f"[results] {results_path}\n")


if __name__ == "__main__":
    main()
