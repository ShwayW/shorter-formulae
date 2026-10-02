#!/usr/bin/env python3
"""
merge_seed_results.py -- Combine the per-seed outputs produced by the parallel
Cluster eval jobs (slurm/eval_mymodels.sh, slurm/eval_e2e.sh, slurm/eval_tpsr.sh)
into the single canonical result files that the plotting / comparison scripts
(compare_srbench.py, compare_llmsrbench.py, plot_ood_vs_gap.py, ...) read.

The parallel jobs run one process per seed, each writing to its own directory so
concurrent writers never race on the same file:

    <group>/seed42/noise_<tau>/eval_tf_<model>.pkl.gz            (eval_mymodels.py --results-dir)
    <group>/seed42/noise_<tau>/results_feynman_e2e.pkl.gz        (eval_e2e.py --results-dir)
    <group>/seed42/noise_<tau>/results_e2e_tpsr.pkl.gz           (eval_e2e_tpsr.py --results-dir)
    <group>/seed42/noise_<tau>/llmsrbench/<m>/results.pkl.gz     (eval_mymodels --output)
    <group>/seed43/...                                         (and so on for every seed)

This script walks every <group>/seed*/ directory and merges by artifact type:

  * **/*.pkl.gz      : union of the per-seed "results" record lists, deduped on
                       (dataset|equation_id, seed); metadata from the first file.
                       Relative paths are preserved, so noise_<tau>/ and
                       llmsrbench/<method>/ land in the right place.
  * */results.jsonl  : legacy artifacts, deduped the same way.
  * *.csv            : legacy artifacts, deduped the same way.

The merged files land in the base results dir (default ./results) under their
canonical names, exactly where the plotting scripts expect them.

Usage:
    python merge_seed_results.py                       # base=results, seeds=seed*
    python merge_seed_results.py --results-dir results --seed-glob 'seed*'
    python merge_seed_results.py --delete-seed-dirs    # prune seed dirs (archived sweeps only)

The per-seed dirs are RETAINED by default. Each worker runs --seed N --n_seeds 1, so
seed<N>/ holds exactly that seed's rows: the seed dirs are the source of truth and the
merged files a derived cache, rebuildable at any time. They are also the only thing the
eval scripts resume from. Deleting them makes the cache the sole copy -- see the note on
_sources() for how that silently lost data.
"""
import argparse
import glob
import gzip
import json
import os
import pickle
import shutil
import sys
from collections import OrderedDict


def _seed_dirs(base, seed_glob):
    return sorted(
        d for d in glob.glob(os.path.join(base, seed_glob))
        if os.path.isdir(d)
    )


def _blank(value):
    return value is None or str(value).strip() in ("", "None", "nan", "NaN")


def _row_key(rec):
    """(dataset, seed) identity of a result row, or None if it has neither."""
    ds = rec.get("dataset", rec.get("equation_id"))
    seed = rec.get("seed")
    if ds is None or seed is None:
        return None
    return (str(ds), int(float(seed)))


def _has_formula(rec):
    for field in ("predicted_formula", "discovered_equation"):
        if field in rec:
            return not _blank(rec[field])
    return True


def _fold(merged, rows):
    """Fold `rows` into the `merged` {key: row} map, later sources winning -- except
    that a failed/None row never displaces one carrying a formula (those cells get
    retried, so a real result always beats a recorded failure)."""
    for rec in rows:
        key = _row_key(rec)
        if key is None:
            continue
        prev = merged.get(key)
        if prev is not None and _has_formula(prev) and not _has_formula(rec):
            continue
        merged[key] = rec


def _sources(out_path, seed_paths):
    """Merge inputs, lowest priority first. The EXISTING merged file leads: a job
    that finished earlier deletes its seed dirs, so that file is often the only
    copy of the seeds it completed. Rebuilding from seed dirs alone would drop them."""
    lead = [out_path] if os.path.isfile(out_path) else []
    return lead + sorted(seed_paths)


def merge_pkl_gz(seed_dirs, base):
    """Merge <base>/seed*/**/*.pkl.gz -> <base>/**/*.pkl.gz, preserving relative paths.

    Every artifact is now a .pkl.gz of {**meta, "results": [rows]} (results_io), so one
    glob covers all of them: noise_<tau>/eval_tf_<model>.pkl.gz, noise_<tau>/results_*.pkl.gz
    and noise_<tau>/llmsrbench/<method>/results.pkl.gz.
    """
    by_rel = OrderedDict()
    for d in seed_dirs:
        for path in glob.glob(os.path.join(d, "**", "*.pkl.gz"), recursive=True):
            rel = os.path.relpath(path, d)          # e.g. noise_0/eval_tf_89M_40.pkl.gz
            by_rel.setdefault(rel, []).append(path)

    for rel, paths in by_rel.items():
        out_path = os.path.join(base, rel)
        meta, records = {}, {}
        sources = _sources(out_path, paths)
        for p in sources:
            with gzip.open(p, "rb") as fh:
                obj = pickle.load(fh)
            if not meta:
                meta = {k: v for k, v in obj.items() if k != "results"}
            _fold(records, obj.get("results", []))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with gzip.open(out_path, "wb") as fh:
            pickle.dump({**meta, "results": [records[k] for k in sorted(records)]}, fh)
        print(f"[pkl.gz] {out_path}  <-  {len(sources)} source file(s), "
              f"{len(records)} unique (dataset,seed) records")


def merge_jsonl(seed_dirs, base):
    """Merge results/seed*/**/results.jsonl -> results/**/results.jsonl (concatenate rows)."""
    by_rel = OrderedDict()
    for d in seed_dirs:
        for path in glob.glob(os.path.join(d, "**", "results.jsonl"), recursive=True):
            rel = os.path.relpath(path, d)          # e.g. llmsrbench/m89_lsr_transform/results.jsonl
            by_rel.setdefault(rel, []).append(path)

    for rel, paths in by_rel.items():
        out_path = os.path.join(base, rel)
        sources = _sources(out_path, paths)
        rows = {}
        for p in sources:                       # read every source before truncating out_path
            with open(p) as fh:
                _fold(rows, [json.loads(line) for line in fh if line.strip()])
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as out:
            for key in sorted(rows):
                out.write(json.dumps(rows[key]) + "\n")
        print(f"[jsonl ] {out_path}  <-  {len(sources)} source file(s), {len(rows)} rows")


def merge_csv(seed_dirs, base):
    """Merge results/seed*/**/*.csv -> results/**/*.csv (row concat, dedupe on dataset+seed).

    Recursive + relpath-preserving so per-noise CSVs written under seed<N>/noise_<tau>/
    merge into results/noise_<tau>/... alongside the transformer noise outputs.
    """
    try:
        import pandas as pd
    except ImportError:
        print("[csv   ] pandas unavailable; skipping CSV merge.", file=sys.stderr)
        return
    by_rel = OrderedDict()
    for d in seed_dirs:
        for path in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True):
            rel = os.path.relpath(path, d)
            by_rel.setdefault(rel, []).append(path)

    for rel, paths in by_rel.items():
        out_path = os.path.join(base, rel)
        sources = _sources(out_path, paths)
        df = pd.concat([pd.read_csv(p) for p in sources], ignore_index=True)
        subset = [c for c in ("dataset", "seed") if c in df.columns]
        if subset:
            if "predicted_formula" in df.columns:
                # Stable sort puts failed rows first, so keep="last" prefers a row
                # with a formula and, among those, the latest source.
                df = df.assign(_ok=~df["predicted_formula"].map(_blank))
                df = df.sort_values("_ok", kind="stable").drop(columns="_ok")
            df = df.drop_duplicates(subset=subset, keep="last")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[csv   ] {out_path}  <-  {len(sources)} source file(s), {len(df)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results",
                    help="Base results directory containing the per-seed subdirs (default: results).")
    ap.add_argument("--seed-glob", default="seed*",
                    help="Glob (relative to --results-dir) matching the per-seed subdirs (default: seed*).")
    ap.add_argument("--keep-seed-dirs", action="store_true",
                    help="No-op, kept for compatibility -- seed dirs are retained by default.")
    ap.add_argument("--delete-seed-dirs", action="store_true",
                    help="Delete the per-seed subdirectories after merging. Only do this once "
                         "the sweep is finished AND archived: they are the source of truth and "
                         "the only thing the eval scripts resume from.")
    args = ap.parse_args()

    seed_dirs = _seed_dirs(args.results_dir, args.seed_glob)
    if not seed_dirs:
        sys.exit(f"No per-seed directories matching '{args.seed_glob}' under {args.results_dir}.")
    print(f"Merging {len(seed_dirs)} seed dir(s): {', '.join(os.path.basename(d) for d in seed_dirs)}\n")

    merge_pkl_gz(seed_dirs, args.results_dir)
    merge_jsonl(seed_dirs, args.results_dir)
    merge_csv(seed_dirs, args.results_dir)

    if args.delete_seed_dirs:
        for d in seed_dirs:
            shutil.rmtree(d)
        print(f"\n--delete-seed-dirs: removed {len(seed_dirs)} per-seed dir(s). "
              f"The merged files are now the only copy.")
    else:
        print(f"\n{len(seed_dirs)} per-seed dir(s) retained (source of truth; resume state). "
              f"Pass --delete-seed-dirs to prune once archived.")


if __name__ == "__main__":
    main()
