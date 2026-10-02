#!/usr/bin/env python3
"""
eval_aifeynman.py --benchmark llmsrbench -- Evaluate AI Feynman 2.0 (Udrescu, Tan, Feng, Neto, Wu
& Tegmark, NeurIPS 2020 -- the ORIGINAL authors' code in ./AI-Feynman/) on the LLM-SRBench
benchmark, writing results in the SAME results.pkl.gz format as eval_mymodels.py --benchmark llmsrbench /
eval_e2e.py --benchmark llmsrbench so compare_llmsrbench.py and the plotting stack can include it as
one more method.  2.0 is also the version SRBench benchmarks, so the "AIFeynman" curve is
the same method on both benchmarks (srbench/README.md lists "AIFeynman 2.0").

NB the 1.0 paper (Udrescu & Tegmark, Science Advances 2020) is the FEYNMAN DATABASE
citation, not this method's -- keep the two apart in the bibliography.

This is the real published AI Feynman, NOT the repo's own aif2.py (which is an
AI-Feynman-2.0-*style* decomposition bolted onto our transformer -- see aif2.py and
`eval_mymodels.py --benchmark llmsrbench --aif2`).  The two are separate methods and land in separate
results trees.

HOW AI FEYNMAN IS DRIVEN
------------------------
Upstream's API is file- and cwd-based, not in-memory:
  * `run_aifeynman(pathdir, filename, ...)` reads a whitespace-separated text file
    whose columns are X0..X{D-1} then y LAST (the opposite of LLM-SRBench's hdf5,
    where column 0 is y);
  * the Fortran brute-force engines are invoked as executables on $PATH
    (`feynman_sr_mdl_mult`, ...) and communicate through ./args.dat, ./mystery.dat,
    ./results.dat in the CURRENT WORKING DIRECTORY;
  * `run_AI_all` unconditionally creates ./results/ in the cwd.

That last point is a live hazard here: run from the repo root it would scribble into
this project's own results/ tree.  So every problem is solved in its OWN scratch
working directory, in a SUBPROCESS (also giving us a hard per-problem wall-clock cap
that upstream does not provide, and isolating segfaults in the Fortran).

BEFORE FIRST USE the Fortran engines must be compiled -- upstream's setup.py cannot
build on this env (it needs numpy.distutils, gone in NumPy >= 1.26):
    ./AI-Feynman/build_aifeynman.sh          # needs gfortran
    python eval_aifeynman.py --benchmark llmsrbench --check-install
Without them AI Feynman still "runs" but silently degrades: S_brute_force.py wraps
its subprocess.call in a bare `except: pass`, so a missing executable is swallowed
and you get polyfit-only results that look plausible and are worthless.  --check-install
(run automatically at startup) turns that silent failure into a loud one.

Usage:
    python eval_aifeynman.py --benchmark llmsrbench --split lsr_transform --seed 42 \
        --output results/aifeynman/seed42 [--time-limit 7200] [--max-problems N]

    # sharded (the normal way -- ~1-2 h/problem x 111 problems), then merged:
    python eval_aifeynman.py --benchmark llmsrbench --num-shards 16 --shard-id $i ...
    python eval_aifeynman.py --benchmark llmsrbench --num-shards 16 --merge-shards ...

Results land in <output>/noise_<TAU>/llmsrbench/aifeynman_<split>/results.pkl.gz.
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

# Headless: aifeynman imports matplotlib.pyplot at module scope.
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
AIFEYNMAN_DIR = os.environ.get("AIFEYNMAN_DIR", os.path.join(REPO_ROOT, "AI-Feynman"))

# The console-script names S_brute_force.py / S_run_bf_polyfit.py shell out to.
# Built by AI-Feynman/build_aifeynman.sh.
FORTRAN_PROGRAMS = [
    "feynman_sr1", "feynman_sr2", "feynman_sr3",
    "feynman_sr_mdl_mult", "feynman_sr_mdl_plus", "feynman_sr_mdl4", "feynman_sr_mdl5",
]

DATA_FILE = "mystery.txt"      # what we hand to run_aifeynman inside the workdir


# -- install check ------------------------------------------------------------

def check_install(verbose: bool = True) -> bool:
    """True iff `aifeynman` imports AND every Fortran engine is on $PATH."""
    ok = True

    if not os.path.isdir(AIFEYNMAN_DIR):
        if verbose:
            print(f"[FAIL] AI-Feynman checkout not found at {AIFEYNMAN_DIR}")
        return False

    sys.path.insert(0, AIFEYNMAN_DIR)
    try:
        import aifeynman  # noqa: F401
        if verbose:
            print(f"[ ok ] `aifeynman` imports from {AIFEYNMAN_DIR}")
    except Exception as e:
        ok = False
        if verbose:
            print(f"[FAIL] cannot import aifeynman: {e}")

    missing = [p for p in FORTRAN_PROGRAMS if shutil.which(p) is None]
    if missing:
        ok = False
        if verbose:
            print(f"[FAIL] Fortran engines missing from $PATH: {', '.join(missing)}")
            print("       Build them with:  ./AI-Feynman/build_aifeynman.sh")
            print("       (needs gfortran: `sudo apt install gfortran`, or `module load gcc` on HPC)")
    elif verbose:
        print(f"[ ok ] all {len(FORTRAN_PROGRAMS)} Fortran engines on $PATH "
              f"({shutil.which(FORTRAN_PROGRAMS[0])})")

    return ok


# -- Pareto-front parsing -----------------------------------------------------

def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def parse_solution_file(path: str) -> list:
    """Parse an AI Feynman `results/solution_*` file into a list of dicts.

    Row layout (S_run_aifeynman.run_aifeynman, np.savetxt(fmt="%s")):
        with an internal test split :  test_error log_err log_err_all complexity error expr
        without one                 :            log_err log_err_all complexity error expr

    The expression is a sympy str and CONTAINS SPACES ("x0 + x1"), so the columns
    cannot simply be split -- we take the maximal leading run of float-parseable
    tokens as the numeric columns and rejoin the remainder as the expression.  A
    constant-only solution ("3.14159") is all-float, hence the last-token fallback.
    """
    if not os.path.exists(path):
        return []

    rows = []
    with open(path) as fh:
        for line in fh:
            toks = line.split()
            if not toks:
                continue
            split_at = next((i for i, t in enumerate(toks) if not _is_float(t)),
                            len(toks) - 1)
            nums = [float(t) for t in toks[:split_at]]
            expr = " ".join(toks[split_at:]).strip()
            if not expr or len(nums) < 2:
                continue
            # complexity + train error are always the last two numeric columns.
            row = {"expression": expr,
                   "complexity": nums[-2],
                   "train_error": nums[-1],
                   "test_error": nums[0] if len(nums) >= 5 else None}
            rows.append(row)
    return rows


def select_solution(rows: list, mode: str = "test_error") -> dict:
    """Pick one point off the Pareto front.

    `test_error` (default) uses AI Feynman's OWN internal held-out split (the
    test_percentage carve-out of the data we gave it, which is LLM-SRBench's TRAIN
    split) -- model selection that never touches LLM-SRBench's held-out test set.
    Falls back to `accuracy` when that column is absent.
    """
    rows = [r for r in rows if r.get("expression")]
    if not rows:
        return {}
    if mode == "test_error" and any(r["test_error"] is not None for r in rows):
        cand = [r for r in rows if r["test_error"] is not None]
        return min(cand, key=lambda r: r["test_error"])
    if mode == "complexity":
        return min(rows, key=lambda r: r["complexity"])
    return min(rows, key=lambda r: r["train_error"])          # "accuracy"


# -- prediction ---------------------------------------------------------------

_VAR_RE = re.compile(r"\bx(\d+)\b")


def normalize_vars(expr: str) -> str:
    """Rewrite AI Feynman's `x0, x1, ...` into the repo's infix convention `x_0, x_1, ...`.

    This is not cosmetic.  The plotting stack routes a discovered formula to one of two
    evaluators (plot_llmsrbench_ood_vs_gap._is_infix): PREFIX formulas use `v1..vN` and
    go to wrapExtEvalPN, INFIX ones use `x_0..x_N` and are eval'd against a numpy
    namespace that binds exactly those names.  AI Feynman's sympy output is infix but
    names its inputs `x0` -- unbound in that namespace, and often paren-free, so the
    infix heuristic would not even fire.  Normalising here means the stored
    discovered_equation is the same string we score, in the format every downstream
    consumer already speaks.  (compare_srbench.py carries the mirror-image mapping,
    "AIFeynman": "x0", for SRBench's published AI Feynman results.)
    """
    return _VAR_RE.sub(r"x_\1", expr or "")


def predict_expr(expr: str, X: np.ndarray) -> np.ndarray:
    """Evaluate a normalized (x_0, x_1, ...) sympy expression on X (N, D).

    Column order is the data file's, i.e. LLM-SRBench's `symbols` order.
    NaN on any failure.
    """
    n = X.shape[0]
    if not expr:
        return np.full(n, np.nan)
    try:
        from sympy import lambdify, symbols
        from sympy.parsing.sympy_parser import parse_expr
        d = X.shape[1]
        var = symbols(" ".join(f"x_{i}" for i in range(d))) if d > 1 else (symbols("x_0"),)
        var = tuple(var) if isinstance(var, (list, tuple)) else (var,)
        f = lambdify(var, parse_expr(expr), modules=["numpy"])
        with np.errstate(all="ignore"):
            out = f(*[X[:, i] for i in range(d)])
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 0:                      # constant expression broadcasts to scalar
            out = np.full(n, float(out))
        if out.shape[0] != n:
            return np.full(n, np.nan)
        return np.where(np.isfinite(out), out, np.nan)
    except Exception:
        return np.full(n, np.nan)


# -- the per-problem worker (runs INSIDE the scratch workdir) -----------------

def _worker_main(argv) -> int:
    """Solve one problem with upstream run_aifeynman. cwd is already the workdir."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args(argv)
    with open(a.config) as fh:
        cfg = json.load(fh)

    sys.path.insert(0, cfg["aifeynman_dir"])
    np.random.seed(cfg["seed"] & 0xFFFFFFFF)
    try:
        import torch
        torch.manual_seed(cfg["seed"] & 0xFFFFFFFF)
    except Exception:
        pass

    from aifeynman import run_aifeynman
    run_aifeynman(
        "./", DATA_FILE,
        cfg["bf_try_time"], cfg["ops_file"],
        polyfit_deg=cfg["polyfit_deg"],
        NN_epochs=cfg["nn_epochs"],
        vars_name=[],                       # no units -> skip dimensional analysis
        test_percentage=cfg["test_percentage"],
    )
    return 0


# -- driver -------------------------------------------------------------------

_LOG_TAIL_BYTES = 64 * 1024      # enough for a traceback + the last search phase


def _save_worker_log_tail(workdir, args, tau, name, status):
    """Copy the tail of a failed worker's aifeynman.log into the results tree.

    Only written when --quiet routed the worker's output to a file; without --quiet
    it already went to the job's stdout.  Best-effort: a diagnostics failure must
    never take down the run that produced the result.
    """
    src = os.path.join(workdir, "aifeynman.log")
    if not os.path.exists(src):
        return
    try:
        from results_io import noise_dir
        dst_dir = os.path.join(noise_dir(args.output, tau), "llmsrbench",
                               "aifeynman_diagnostics")
        os.makedirs(dst_dir, exist_ok=True)
        with open(src, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size > _LOG_TAIL_BYTES:
                fh.seek(size - _LOG_TAIL_BYTES)
            tail = fh.read()
        safe = str(name).replace("/", "_")
        with open(os.path.join(dst_dir, f"{safe}.{status}.log"), "wb") as out:
            out.write(f"# {name}  status={status}  (tail of {size} bytes)\n"
                      .encode())
            out.write(tail)
    except Exception as e:
        print(f"  [warn] could not save worker log for {name}: {e}", flush=True)


def solve_one(X_train, y_train, workdir, args, seed) -> tuple:
    """Run AI Feynman on one problem. Returns (rows, elapsed, status)."""
    os.makedirs(workdir, exist_ok=True)

    # X columns then y LAST -- upstream's convention (data[:, :-1] = X, data[:, -1] = y).
    np.savetxt(os.path.join(workdir, DATA_FILE),
               np.column_stack([X_train, y_train]), fmt="%.12g")

    cfg = {"aifeynman_dir": AIFEYNMAN_DIR, "seed": int(seed),
           "bf_try_time": args.bf_try_time, "ops_file": args.ops_file,
           "polyfit_deg": args.polyfit_deg, "nn_epochs": args.nn_epochs,
           "test_percentage": args.test_percentage}
    cfg_path = os.path.join(workdir, "config.json")
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh)

    # -u: unbuffered, so aifeynman.log is readable WHILE a slow problem is running
    # (Python block-buffers stdout when it is a file rather than a tty) and so a
    # killed-on-timeout worker does not lose its last buffer.
    cmd = [sys.executable, "-u", os.path.abspath(__file__),
           "--_worker", "--config", "config.json"]
    t0 = time.perf_counter()
    status = "ok"
    # --quiet goes to a per-problem log file rather than /dev/null: AI Feynman's
    # chatter would drown a 111-problem job's Slurm .out, but it is the only way to
    # diagnose a crash, and crashed workdirs are kept below.
    log_fh = open(os.path.join(workdir, "aifeynman.log"), "w") if args.quiet else None
    # start_new_session so a timeout can kill the WHOLE process group -- AI Feynman
    # spawns Fortran children that would otherwise be orphaned and keep burning CPU.
    proc = subprocess.Popen(cmd, cwd=workdir, start_new_session=True,
                            stdout=log_fh, stderr=subprocess.STDOUT if log_fh else None)
    try:
        rc = proc.wait(timeout=args.time_limit if args.time_limit > 0 else None)
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

    # Parse whatever made it to disk.  The final solution_ file is written only at
    # the very end, so on a timeout fall back to the two intermediate snapshots.
    res = os.path.join(workdir, "results")
    rows = []
    # Upstream writes the final front as `solution_<filename_orig>` but the two
    # intermediate snapshots as `solution_*_snap_<filename>.txt` -- and <filename>
    # already ends in ".txt", so those really are double-extensioned on disk.
    for stem in (f"solution_{DATA_FILE}",
                 f"solution_first_snap_{DATA_FILE}.txt",
                 f"solution_before_snap_{DATA_FILE}.txt"):
        rows = parse_solution_file(os.path.join(res, stem))
        if rows:
            break
    if not rows and status == "ok":
        status = "no_solution"
    return rows, elapsed, status


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="llmsrbench", choices=["llmsrbench"],
                   help="Which benchmark to run. Only 'llmsrbench' is implemented: "
                        "on SRBench we do not run AI Feynman 2.0 ourselves, we use "
                        "the published SRBench results for it (see compare_srbench.py). "
                        "The flag exists so every eval_<method>.py selects its "
                        "benchmark the same way. Default: llmsrbench.")
    p.add_argument("--check-install", action="store_true",
                   help="Verify the aifeynman import + compiled Fortran engines, then exit.")
    p.add_argument("--split", type=str, default="lsr_transform")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="results/aifeynman",
                   help="Seed root; results go to <output>/noise_<TAU>/llmsrbench/"
                        "aifeynman_<split>/results.pkl.gz (see results_io).")
    p.add_argument("--target-noise", type=float, default=0.0, metavar="TAU",
                   help="Additive-RMS noise on TRAIN targets only "
                        "(SRBench/TPSR convention); selects the noise_<TAU>/ subdir.")
    p.add_argument("--problem", type=str, default=None)
    p.add_argument("--max-problems", type=int, default=None)
    # Sharding, same CLI + on-disk convention as eval_llmsr.py: each shard writes its
    # OWN <method>__shard<i>of<N>/ dir (concurrent whole-file saves would otherwise
    # clobber each other), then one --merge-shards pass folds them into <method>/.
    # AI Feynman is ~1-2 h/problem and the split has 111 of them, so this is the
    # normal way to run it, not an edge case.  check_llmsr.py already globs this shape.
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of parallel shards. Default 1 (no sharding).")
    p.add_argument("--shard-id", type=int, default=0,
                   help="This process's shard index in [0, num_shards).")
    p.add_argument("--retry-nulls", action="store_true", default=False,
                   help="Re-run problems whose stored row has no equation (timeout or "
                        "crash) instead of treating them as done. Use this ONLY when "
                        "something changed that could alter the outcome -- a bigger "
                        "--time-limit, or a fixed bug -- otherwise a re-submission "
                        "just reproduces the same timeouts at full cost.")
    p.add_argument("--merge-shards", action="store_true", default=False,
                   help="Merge every shard artifact for (split, noise) into the canonical "
                        "results.pkl.gz and remove the shard dirs, then exit. Do NOT pass "
                        "while shards are still running.")
    # AI Feynman hyperparameters -- defaults copied from SRBench's tuned config
    # (srbench/experiment/methods/tuned/params/_aifeynman.py) so the numbers are
    # comparable to the published SRBench AIFeynman results.
    p.add_argument("--bf-try-time", type=int, default=60)
    p.add_argument("--ops-file", type=str, default="14ops.txt")
    p.add_argument("--polyfit-deg", type=int, default=4)
    p.add_argument("--nn-epochs", type=int, default=4000)
    p.add_argument("--test-percentage", type=int, default=20)
    p.add_argument("--time-limit", type=float, default=7200.0, metavar="SEC",
                   help="Hard per-problem wall clock (0 = unlimited). Upstream has no "
                        "such cap; SRBench's tuned config uses max_time=7200. Partial "
                        "Pareto fronts on disk are still parsed after a timeout.")
    p.add_argument("--max-train-points", type=int, default=0,
                   help="Subsample TRAIN to at most this many rows (0 = all). The "
                        "Fortran brute force scales with the point count.")
    p.add_argument("--select", type=str, default="test_error",
                   choices=["test_error", "accuracy", "complexity"],
                   help="Which Pareto point to report as THE discovered equation.")
    p.add_argument("--workdir", type=str, default=None,
                   help="Scratch root for the per-problem run dirs "
                        "(default: $SLURM_TMPDIR or a system temp dir).")
    p.add_argument("--keep-workdir", action="store_true",
                   help="Keep each problem's scratch dir (NN checkpoints, Pareto files).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress AI Feynman's very chatty per-problem stdout.")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args, rest = p.parse_known_args()

    if args._worker:
        return _worker_main(rest)

    if args.check_install:
        sys.exit(0 if check_install() else 1)

    if not check_install():
        print("\nAborting: AI Feynman is not fully installed.  A missing Fortran engine "
              "is swallowed by a bare `except: pass` inside S_brute_force.py and would "
              "silently produce polyfit-only garbage.", file=sys.stderr)
        sys.exit(1)

    from results_io import llmsr_dir, llmsr_path, load_results, save_results
    from eval_mymodels import load_problems, output_metrics

    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard-id must be in [0, {args.num_shards}); got {args.shard_id}")

    def _method_name(shard_id=None):
        """aifeynman_<split>[__shard<i>of<N>] -- suffix format matches eval_llmsr.py."""
        base = f"aifeynman_{args.split}"
        if shard_id is None or args.num_shards == 1:
            return base
        return f"{base}__shard{shard_id:02d}of{args.num_shards:02d}"

    tau = args.target_noise or 0.0

    if args.merge_shards:
        merged = {}
        n_found = 0
        for i in range(args.num_shards):
            _, srows = load_results(llmsr_path(args.output, tau, _method_name(i)))
            if srows:
                n_found += 1
            for r in srows:
                key = (r.get("equation_id"), r.get("seed"))
                prev = merged.get(key)
                # A real equation beats a None/empty one from a timed-out shard.
                if prev is None or prev.get("discovered_equation") in (None, "None", ""):
                    merged[key] = r
        canonical = llmsr_path(args.output, tau, _method_name())
        save_results(canonical, list(merged.values()), model_name="aifeynman",
                     split=args.split, method=_method_name(), target_noise=tau)
        print(f"[merge] {n_found}/{args.num_shards} shards -> {len(merged)} rows -> {canonical}")
        for i in range(args.num_shards):
            d = llmsr_dir(args.output, tau, _method_name(i))
            if os.path.isdir(d):
                shutil.rmtree(d)
        print(f"[merge] removed {args.num_shards} shard artifacts")
        return 0

    print(f"Loading LLM-SRBench split '{args.split}' ...", flush=True)
    problems = load_problems(args.split)
    if args.problem is not None:
        problems = [q for q in problems if q["name"] == args.problem]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    if args.num_shards > 1:
        problems = [q for k, q in enumerate(problems) if k % args.num_shards == args.shard_id]
        print(f"  shard {args.shard_id}/{args.num_shards}", flush=True)
    print(f"  {len(problems)} problems.\n", flush=True)

    _method = _method_name(args.shard_id)
    results_path = llmsr_path(args.output, tau, _method)

    # Resume: skip problems already scored (same convention as eval_e2e).
    #
    # A row with no discovered_equation is a REAL outcome -- the method searched for
    # its full budget and came back empty -- so by default it counts as done and is
    # NOT redone.  Re-running a finished cell at the same budget would otherwise burn
    # hours reproducing the identical timeouts.
    #
    # --retry-nulls flips that, and is what you want when the budget CHANGED: after
    # raising --time-limit, the old nulls are the whole point of the re-run.  (That is
    # how 29.7% -> 42.3% happened: 111 nulls from a too-tight 2 h cap, retried at 8 h.)
    done_ids, rows = set(), []
    n_null = 0
    for _r in load_results(results_path)[1]:
        _eid = _r.get("equation_id")
        if _eid in done_ids:
            continue
        _is_null = _r.get("discovered_equation") in (None, "None", "")
        if _is_null:
            n_null += 1
            if args.retry_nulls:
                continue                     # drop it -> the problem is re-run
        done_ids.add(_eid)
        rows.append(_r)
    if done_ids or n_null:
        _fate = "dropped for retry" if args.retry_nulls else "kept as-is (use --retry-nulls to redo)"
        print(f"Resuming: {len(done_ids)} rows kept, {n_null} null (timeout/crash) "
              f"{_fate} -- {results_path}", flush=True)

    scratch_root = args.workdir or os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
    scratch_root = os.path.join(scratch_root, f"aifeynman_{args.split}_seed{args.seed}")
    os.makedirs(scratch_root, exist_ok=True)
    print(f"Scratch: {scratch_root}\n", flush=True)

    def _save():
        save_results(results_path, rows, model_name="aifeynman", split=args.split,
                     method=_method, target_noise=tau,
                     hyperparams={"bf_try_time": args.bf_try_time,
                                  "ops_file": args.ops_file,
                                  "polyfit_deg": args.polyfit_deg,
                                  "nn_epochs": args.nn_epochs,
                                  "test_percentage": args.test_percentage,
                                  "time_limit": args.time_limit,
                                  "select": args.select})

    _save()
    for i, q in enumerate(problems):
        if q["name"] in done_ids:
            continue
        train, test = q["train"], q["test"]
        n_vars = train.shape[1] - 1
        X_train, y_train = train[:, 1:], train[:, 0]     # col 0 is y in the hdf5
        X_test, y_test = test[:, 1:], test[:, 0]

        # Per-seed row permutation, mirroring eval_e2e.py --benchmark llmsrbench: it randomises
        # AI Feynman's internal train/test carve-out and any subsample below, so
        # seeds actually differ.
        rng = np.random.default_rng([args.seed, i])
        perm = rng.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        if args.target_noise and args.target_noise > 0.0 and len(y_train):
            _rms = float(np.sqrt(np.mean(np.square(y_train))))
            if _rms > 0.0:
                y_train = y_train + np.random.default_rng([args.seed, i, 1]).normal(
                    0.0, args.target_noise * _rms, size=len(y_train))

        if args.max_train_points and len(X_train) > args.max_train_points:
            X_train, y_train = X_train[:args.max_train_points], y_train[:args.max_train_points]

        workdir = os.path.join(scratch_root, q["name"].replace("/", "_"))
        pareto, elapsed, status = solve_one(X_train, y_train, workdir, args, args.seed + i)
        best = select_solution(pareto, args.select)
        expr = normalize_vars(best.get("expression")) or None

        id_m = output_metrics(predict_expr(expr, X_test), y_test)
        ood_m = None
        if q["ood_test"] is not None:
            ood_m = output_metrics(predict_expr(expr, q["ood_test"][:, 1:]),
                                   q["ood_test"][:, 0])

        rows.append({
            "equation_id": q["name"],
            "gt_equation": q["expression"],
            "discovered_equation": expr,
            "n_vars": n_vars,
            "num_datapoints": int(len(X_train)),
            "num_eval_datapoints": int(len(test)),
            "search_time": elapsed,
            "seed": args.seed,
            "id_metrics": id_m,
            "ood_metrics": ood_m,
            "status": status,
            "complexity": best.get("complexity"),
            "pareto_size": len(pareto),
        })
        _save()

        # Preserve the tail of the worker log for anything that did not produce an
        # equation, BEFORE the workdir is reclaimed.  On a cluster the workdir lives
        # on node-local scratch that disappears with the allocation, so "keep the
        # workdir for crashes" is not enough -- the first 111-problem run lost all 4
        # of its tracebacks that way.  The tail is capped and lands in the RESULTS
        # tree (not a shard dir, which --merge-shards deletes).
        if status != "ok":
            _save_worker_log_tail(workdir, args, tau, q["name"], status)

        # Keep the scratch dir for genuine crashes (partial Pareto files are the only
        # other diagnosis); a "timeout"/"no_solution" is an ordinary outcome on hard
        # problems, not a bug, so those are reclaimed.
        if not args.keep_workdir and not status.startswith("exit"):
            shutil.rmtree(workdir, ignore_errors=True)

        print(f"[{i+1}/{len(problems)}] {q['name']:<24} R^2={id_m['r2']:.4f}  "
              f"t={elapsed:.0f}s  {status}  {str(expr)[:60]}", flush=True)

    r2s = np.array([r["id_metrics"]["r2"] for r in rows], dtype=np.float64)
    r2s = r2s[np.isfinite(r2s)]
    print(f"\n{'='*70}")
    print(f"AI Feynman  LLM-SRBench {args.split}  ({len(rows)} problems)")
    if r2s.size:
        print(f"  Acc (R^2 >= 0.99) : {100*np.mean(r2s >= 0.99):.1f}%")
        print(f"  Mean R^2         : {np.mean(r2s):.4f}   Median R^2: {np.median(r2s):.4f}")
    print(f"{'='*70}\n[results] {results_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
