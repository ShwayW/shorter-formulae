#!/usr/bin/env python3
"""
score_sym_acc.py -- Run PhysicsRegression's symbolic-accuracy metric over our results.

Computes sym_acc.cal_sym_acc for every (dataset, seed) row of our evaluation
artifacts and reports the per-method rate next to the R^2 solve rate we normally
report, so the two can be read against each other.

Ground truth per benchmark:
  * SRBench/Feynman -- the row's own `target_pn`, which eval_mymodels.py already writes
    in the same v1..vN space as `predicted_formula`.  Rows without it (eval_e2e.py
    artifacts) fall back to eval_mymodels.load_feynman_pn_targets() keyed by dataset,
    with the E2E `x_<i>` naming mapped to `v<i+1>`.
  * LLM-SRBench -- the split parquet's `expression`, with its `symbols` renamed
    into v-space (symbols[0] is the output, symbols[1:] are X columns in order --
    the layout eval_mymodels.py --benchmark llmsrbench loads).

Because sympy `simplify` dominates the runtime, work is spread over a process
pool and every row is cached to CSV; re-running is then free unless --rebuild.

Usage:
    # our transformers on Feynman at noise 0
    python score_sym_acc.py --noise 0

    # a specific artifact, and the LLM-SRBench split
    python score_sym_acc.py --results results/mymodels/noise_0/eval_tf_145M_40.pkl.gz
    python score_sym_acc.py --benchmark llmsrbench --split lsr_transform --noise 0

    # everything we have, all noise levels
    python score_sym_acc.py --noise 0 0.001 0.01 0.1 --groups mymodels e2e

Writes results/sym_acc_<benchmark>_noise<tau>.csv (one row per
method/dataset/seed) and prints the summary table.
"""
import argparse
import glob
import multiprocessing as mp
import os
import re
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import results_io                                        # noqa: E402
import sym_acc                                           # noqa: E402
from sym_acc import sym_acc_pair                         # noqa: E402


# -- Ground truth --------------------------------------------------------------

_FEYNMAN_PN = None


def feynman_pn_targets() -> dict:
    """{dataset_name: ground-truth PN} for Feynman, cached across calls."""
    global _FEYNMAN_PN
    if _FEYNMAN_PN is None:
        from eval_mymodels import load_feynman_pn_targets
        _FEYNMAN_PN = load_feynman_pn_targets()
    return _FEYNMAN_PN


_LSR_META = {}


def llmsrbench_targets(split: str) -> dict:
    """{equation_id: ground-truth expression in v-space} for an LLM-SRBench split.

    The parquet carries the expression in the problem's own symbol names; we
    rename them to v1..vD in the column order eval_mymodels.py --benchmark llmsrbench feeds the model
    (symbols[0] is the output y, symbols[1:] are the X columns).
    """
    if split in _LSR_META:
        return _LSR_META[split]
    from eval_mymodels import resolve_llmsrbench_file
    meta = pd.read_parquet(
        resolve_llmsrbench_file(f"data/{split}-00000-of-00001.parquet"))
    out = {}
    for _, e in meta.iterrows():
        syms = list(e["symbols"])
        expr = str(e["expression"])
        # Longest name first so a rename never clobbers a longer symbol that
        # contains it as a prefix (e.g. m before m_0 would corrupt m_0).
        order = sorted(range(1, len(syms)), key=lambda i: -len(str(syms[i])))
        # Two-phase rename via placeholders, so a fresh v-name can never collide
        # with a not-yet-renamed original symbol.
        for i in order:
            expr = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(str(syms[i]))}(?![A-Za-z0-9_])",
                          f"__V{i}__", expr)
        for i in order:
            expr = expr.replace(f"__V{i}__", f"v{i}")
        out[str(e["name"])] = expr
    _LSR_META[split] = out
    return out


_X_TO_V = re.compile(r"(?<![A-Za-z0-9_])x_(\d+)(?![A-Za-z0-9_])")


def _x_to_v(formula: str) -> str:
    """E2E's `x_0`-based naming -> our `v1`-based naming (same column order)."""
    return _X_TO_V.sub(lambda m: f"v{int(m.group(1)) + 1}", str(formula))


def normalise_pred(formula):
    """Put a prediction in the v-space the ground truth uses.

    eval_e2e.py emits infix in `x_0..` naming while eval_mymodels.py emits PN in `v1..`;
    both index the same columns, so shifting the index by one aligns them.  Keyed
    off the presence of an `x_<i>` token, which our grammar never produces -- so
    this is a no-op for our own artifacts on either benchmark.
    """
    if formula is None:
        return formula
    s = str(formula)
    return _x_to_v(s) if _X_TO_V.search(s) else s


# -- Artifact loading ----------------------------------------------------------

def _rows_of(path: str) -> list:
    _meta, rows = results_io.load_rows_any(path)   # (meta, rows)
    return rows or []


def _method_label(path: str, benchmark: str) -> str:
    """Short, stable name for an artifact, matching how the plots label methods."""
    if benchmark == "llmsrbench":
        return os.path.basename(os.path.dirname(path))
    base = os.path.basename(path)
    for ext in (".pkl.gz", ".pkl", ".csv"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    for pref in ("eval_tf_", "results_"):
        if base.startswith(pref):
            base = base[len(pref):]
    return base


def collect_pairs(path: str, benchmark: str, split: str) -> list:
    """-> [{method, dataset, seed, r2, pred, true}] for one artifact.

    Rows whose ground truth cannot be resolved are dropped here (and counted by
    the caller) rather than being scored 0, which would understate the metric.
    """
    method = _method_label(path, benchmark)
    out, missing_gt, missing_names = [], 0, set()

    if benchmark == "llmsrbench":
        targets = llmsrbench_targets(split)
        for r in _rows_of(path):
            eq = r.get("equation_id")
            true = targets.get(str(eq))
            if true is None:
                # The row's own gt_equation is in the problem's symbol names, not
                # v-space, so it cannot be compared; drop rather than mis-score.
                missing_gt += 1
                missing_names.add(str(eq))
                continue
            idm = r.get("id_metrics") or {}
            out.append({
                "method": method, "dataset": eq, "seed": r.get("seed"),
                "r2": (idm.get("r2") if isinstance(idm, dict) else None),
                "pred": normalise_pred(r.get("discovered_equation")),
                "true": true,
            })
    else:
        fallback = None
        for r in _rows_of(path):
            ds = r.get("dataset")
            true = r.get("target_pn")
            pred = r.get("predicted_formula")
            if not true:
                # No target_pn: an eval_e2e.py artifact.  Fall back to the CSV.
                if fallback is None:
                    fallback = feynman_pn_targets()
                true = fallback.get(str(ds))
            pred = normalise_pred(pred)
            if not true:
                # e.g. the 20 feynman_test_* bonus equations, whose formulas are
                # not in FeynmanEquations.csv and are outside our comparison universe.
                missing_gt += 1
                missing_names.add(str(ds))
                continue
            out.append({
                "method": method, "dataset": ds, "seed": r.get("seed"),
                "r2": r.get("r2"), "pred": pred, "true": true,
            })

    if missing_gt:
        ex = ", ".join(sorted(missing_names)[:3])
        print(f"  [warn] {method}: no ground truth for {missing_gt} row(s) "
              f"({len(missing_names)} dataset(s), e.g. {ex}) -- excluded, not scored 0")
    return out


def discover(results_root: str, benchmark: str, split: str, tau: float,
             groups: list) -> list:
    """Artifact paths for the requested benchmark/noise across result groups."""
    paths = []
    if benchmark == "llmsrbench":
        for g in groups:
            base = os.path.join(
                results_io.group_noise_dir(results_root, g, tau), "llmsrbench")
            for d in sorted(glob.glob(os.path.join(base, f"*_{split}*"))):
                p = os.path.join(d, results_io.RESULTS_NAME)
                if os.path.exists(p):
                    paths.append(p)
    else:
        for g in groups:
            paths += results_io.list_srbench_pkls(results_root, tau, group=g)
            e2e = results_io.e2e_srbench_pkl(results_root, tau, group=g)
            if os.path.exists(e2e):
                paths.append(e2e)
    return paths


# -- Scoring -------------------------------------------------------------------

def _score(rec: dict) -> dict:
    res = sym_acc_pair(rec["pred"], rec["true"])
    return {**{k: rec[k] for k in ("method", "dataset", "seed", "r2")},
            "sym_acc": res.acc, "sym_acc_strict": res.strict,
            "status": res.status, "detail": res.detail}


def score_all(records: list, jobs: int) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["method", "dataset", "seed", "r2",
                                     "sym_acc", "sym_acc_strict", "status", "detail"])
    if jobs <= 1:
        rows = [_score(r) for r in records]
    else:
        # 'fork' keeps the parsed parquet/CSV targets in the children for free;
        # each worker is its own main thread, so sym_acc's SIGALRM guard is live.
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(jobs) as pool:
            rows = []
            for i, row in enumerate(pool.imap_unordered(_score, records, chunksize=4), 1):
                rows.append(row)
                if i % 200 == 0 or i == len(records):
                    print(f"  scored {i}/{len(records)}", flush=True)
    return pd.DataFrame(rows)


# -- Reporting -----------------------------------------------------------------

def summarise(df: pd.DataFrame, r2_thr: float) -> pd.DataFrame:
    """Per-method rates.  A dataset counts as symbolically solved if ANY seed got
    it -- the same "solved in any round/seed" convention compare_srbench uses for
    its solve rate -- alongside the per-row mean for reference."""
    if df.empty:
        return df
    d = df.copy()
    d["solved_r2"] = (pd.to_numeric(d["r2"], errors="coerce") >= r2_thr).astype(float)
    per_ds = (d.groupby(["method", "dataset"])
               .agg(sym_any=("sym_acc", "max"),
                    strict_any=("sym_acc_strict", "max"),
                    r2_any=("solved_r2", "max"))
               .reset_index())
    rows = []
    for m, g in d.groupby("method"):
        gd = per_ds[per_ds["method"] == m]
        rows.append({
            "Method": m,
            "Sym. Acc (any seed)": gd["sym_any"].mean(),
            "Sym. Acc (per row)": g["sym_acc"].mean(),
            "Strict (any seed)": gd["strict_any"].mean(),
            "R^2 Solve (any seed)": gd["r2_any"].mean(),
            "Datasets": int(gd.shape[0]),
            "Rows": int(g.shape[0]),
            "Timeouts": int((g["status"] == "timeout").sum()),
            "Errors": int((g["status"] == "error").sum()),
        })
    return (pd.DataFrame(rows)
              .sort_values("Sym. Acc (any seed)", ascending=False)
              .reset_index(drop=True))


def _print_table(summary: pd.DataFrame, tau: float, benchmark: str) -> None:
    if summary.empty:
        print("No rows scored.")
        return
    disp = summary.copy()
    for c in ("Sym. Acc (any seed)", "Sym. Acc (per row)",
              "Strict (any seed)", "R^2 Solve (any seed)"):
        disp[c] = disp[c].map(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "--")
    print(f"\n=== Symbolic accuracy -- {benchmark}, noise {tau} ===")
    print(disp.to_string(index=False))
    print("\nSym. Acc  = PhysicsRegression cal_sym_acc: equivalent up to a global "
          "additive OR multiplicative constant.")
    print("Strict    = the same pass, restricted to pred - true simplifying to 0.")
    print(f"Timeouts  = rows where sympy exceeded {sym_acc.TIMEOUT_SECONDS}s and were "
          "scored 0 (as upstream does); raise --timeout if this column is large.")


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Symbolic accuracy (PhysicsRegression cal_sym_acc) over our results.")
    ap.add_argument("--results", nargs="+", default=None,
                    help="Explicit artifact paths. Default: auto-discover from "
                         "--results-root/--groups for each --noise.")
    ap.add_argument("--results-root", default="results",
                    help="Results tree root. Default: results")
    ap.add_argument("--groups", nargs="+",
                    default=["mymodels", "mymodels_tpsr", "e2e"],
                    help="Result groups to sweep when auto-discovering.")
    ap.add_argument("--benchmark", choices=["feynman", "llmsrbench"], default="feynman",
                    help="Which benchmark's artifacts to score. Default: feynman")
    ap.add_argument("--split", default="lsr_transform",
                    help="LLM-SRBench split. Default: lsr_transform")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0],
                    help="Noise level(s). Default: 0")
    ap.add_argument("--r2-threshold", type=float, default=0.99, dest="r2_thr",
                    help="R^2 threshold for the side-by-side solve rate. Default: 0.99")
    ap.add_argument("--timeout", type=int, default=sym_acc.TIMEOUT_SECONDS,
                    help="Per-sympy-call budget in seconds. Upstream uses 10; raise it "
                         "when the Timeouts column is large (long predictions), since a "
                         "timed-out row is scored 0. Default: "
                         f"{sym_acc.TIMEOUT_SECONDS}")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Worker processes. Default: cpu_count-1")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N rows per artifact (smoke tests).")
    ap.add_argument("--csv", default=None,
                    help="Per-row output CSV. Default: "
                         "<results-root>/sym_acc_<benchmark>_noise<tau>.csv")
    ap.add_argument("--rebuild", action="store_true",
                    help="Recompute even when the output CSV already exists.")
    args = ap.parse_args()
    # Set before the pool forks, so workers inherit the budget.
    sym_acc.set_timeout(args.timeout)

    for tau in args.noise:
        tag = f"{tau:g}"
        csv_path = args.csv or os.path.join(
            args.results_root, f"sym_acc_{args.benchmark}_noise{tag}.csv")

        if os.path.exists(csv_path) and not args.rebuild:
            print(f"[cache] reading {csv_path}  (--rebuild to recompute)")
            df = pd.read_csv(csv_path)
        else:
            paths = args.results or discover(
                args.results_root, args.benchmark, args.split, tau, args.groups)
            if not paths:
                print(f"[warn] no artifacts found for {args.benchmark} at noise {tag}")
                continue
            records = []
            for p in paths:
                print(f"[load] {p}")
                recs = collect_pairs(p, args.benchmark, args.split)
                if args.limit:
                    recs = recs[: args.limit]
                records += recs
            print(f"Scoring {len(records)} rows on {args.jobs} process(es) "
                  f"({args.timeout}s sympy timeout per call)...")
            df = score_all(records, args.jobs)
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"[write] {csv_path}  ({len(df)} rows)")

        _print_table(summarise(df, args.r2_thr), tag, args.benchmark)


if __name__ == "__main__":
    main()
