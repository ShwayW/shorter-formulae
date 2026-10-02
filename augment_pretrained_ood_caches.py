#!/usr/bin/env python3
"""
augment_pretrained_ood_caches.py -- the NeSymReS / tf4sr / SymFormer twin of
augment_phye2e_caches.py.  Compute the OOD R^2 of the three pretrained-transformer
baselines and append them to the plotting caches.

WHY THIS IS NEEDED (and not optional)
-------------------------------------
plot_ood_vs_gap.py --reuse-ood backfills missing *gaps*, never missing *ALGORITHMS*
(plot_ood_vs_gap.py:267-274).  So without this script a --reuse-ood figure silently
drops these three curves rather than failing -- the worst kind of wrong.  Exactly the
hazard augment_base_ood_caches.py documents for the 89M/145M rows.

The four SRBench caches currently carry 22 algorithms and none of these three.

OOD DOMAIN: this reads whatever datasets/{feynman,llmsrbench}_ood_g*.pkl.gz hold and
does NOT regenerate them.  As of 2026-09-07 those are all `method: 'freeze'` (verified
in every one of the 9 gaps x 2 benchmarks), which is the domain these rows are scored
against.  Regenerating them with a different --method would silently invalidate every
row already in the cache, not just these -- so this script never writes them.

FORMULA FORMAT
--------------
All three emit INFIX over v1..vn:
  * NeSymReS   -- eval_nesymres.to_v_space() rewrites the model's x_1.. into v1..
  * tf4sr      -- eval_tf4sr._formula_string() emits v1.. directly
  * SymFormer  -- eval_symformer._formula_string() maps its x/y onto v1..
so the SRBench half evaluates with compare_srbench._ood_eval_infix style "v1".  Getting
this wrong does not crash; it scores every formula 0.0, so the style is passed
explicitly rather than sniffed (the trap augment_phye2e_caches.py calls out for "x_0").

COVERAGE.  These three do not cover the whole benchmark, and that is a property of the
checkpoints, not of this script:
    nesymres    52 / 99 SRBench,  35 / 111 LSR-T   (<= 3 variables)
    tf4sr       97 / 99 SRBench, 105 / 111 LSR-T   (<= 6 vars, positive inputs)
    symformer   16 / 99 SRBench,   5 / 111 LSR-T   (<= 2 variables)
Rows are emitted only for the problems each method actually attempted, so any figure
mixing them with the 119-problem arms needs a common subset -- the cache cannot
express that and will not warn.

Idempotent: rerunning replaces this method's rows rather than duplicating them.

USAGE
    python augment_pretrained_ood_caches.py                     # all 3, all 4 noises
    python augment_pretrained_ood_caches.py --methods tf4sr     # one method
    python augment_pretrained_ood_caches.py --noises 0          # one noise level
    python augment_pretrained_ood_caches.py --bench srbench     # one half
"""
import argparse
import gzip
import os
import pickle
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import ood_transform                                              # noqa: E402
import results_io                                                 # noqa: E402
from augment_tpsr_caches import _append_idempotent, _llmsr_ood_rows  # noqa: E402
from compare_srbench import _ood_eval_infix, _ood_load_e2e_all    # noqa: E402
from plot_llmsrbench_ood_vs_gap import load_method_data           # noqa: E402

import re

_V_VAR_RE = re.compile(r"(?<![A-Za-z0-9_])v(\d+)(?![A-Za-z0-9_])")


def _v_to_x0(formula):
    """v1..vn  ->  x_0..x_(n-1), the notation the LLM-SRBench OOD evaluator expects."""
    if not formula:
        return formula
    return _V_VAR_RE.sub(lambda m: f"x_{int(m.group(1)) - 1}", str(formula))


DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]
DEFAULT_GAPS = [0, 1, 2, 4, 8, 16, 32, 64, 128]

# label -> (results group, SRBench artifact, LLM-SRBench method dir stem)
METHODS = {
    "nesymres":  ("nesymres",  "eval_nesymres.pkl.gz",  "nesymres"),
    "tf4sr":     ("tf4sr",     "eval_tf4sr.pkl.gz",     "tf4sr"),
    "symformer": ("symformer", "eval_symformer.pkl.gz", "symformer"),
}
FORMULA_STYLE = "v1"          # all three emit infix over v1..vn


# -- SRBench -------------------------------------------------------------------

def _srbench_ood_rows(pkl_paths_by_label, gaps):
    """Per-(gap, algorithm, dataset, seed) OOD R^2, one row per attempted problem.

    The infix branch of compare_srbench.compute_ood_r2 with no baselines / e2e / GT,
    so only the methods asked for are evaluated.  Seeds live in separate per-seed
    artifacts here (unlike PhyE2E's single merged pkl), so they are unioned per label.
    """
    rows = []
    for g in gaps:
        ood_path = os.path.join(SCRIPT_DIR, ood_transform.feynman_ood_path(g))
        if not os.path.exists(ood_path):
            print(f"  [warn] missing {ood_path}")
            continue
        with gzip.open(ood_path, "rb") as fh:
            ood_datasets = pickle.load(fh)["datasets"]
        for label, paths in pkl_paths_by_label.items():
            n = 0
            for p in paths:
                for ds, seed_fmls in _ood_load_e2e_all(p).items():
                    if ds not in ood_datasets:
                        continue
                    X, y = ood_datasets[ds]["X"], ood_datasets[ds]["y"]
                    for seed, formula in seed_fmls:
                        rows.append({
                            "algorithm": label, "dataset": ds,
                            "ood_r2": _ood_eval_infix(formula, FORMULA_STYLE, X, y),
                            "seed": int(seed), "gap": g})
                        n += 1
            print(f"  [gap={g:>3}] {label:<10} {n:>5} rows", flush=True)
    return pd.DataFrame(rows, columns=["algorithm", "dataset", "ood_r2", "seed", "gap"])


def augment_srbench(root, methods, noises, gaps):
    for tau in noises:
        by_label = {}
        for label in methods:
            group, art, _ = METHODS[label]
            paths = []
            for seed in range(42, 52):
                p = os.path.join(
                    results_io.noise_dir(
                        results_io.seed_dir(os.path.join(root, group), seed), tau), art)
                if os.path.exists(p):
                    paths.append(p)
            if paths:
                by_label[label] = paths
            else:
                print(f"  [skip] {label} tau={tau:g}: no SRBench artifacts")
        if not by_label:
            continue
        print(f"[srbench] tau={tau:g}: "
              + ", ".join(f"{k}({len(v)} seeds)" for k, v in by_label.items()), flush=True)
        df = _srbench_ood_rows(by_label, gaps)
        if df.empty:
            print(f"  [warn] no rows produced for tau={tau:g}")
            continue
        _append_idempotent(
            os.path.join(root, ood_transform.srbench_cache_name(tau)), df)


# -- LLM-SRBench ---------------------------------------------------------------

def augment_llmsr(root, methods, noises, gaps, split):
    for tau in noises:
        methods_data = {}
        for label in methods:
            group, _, stem = METHODS[label]
            for seed in range(42, 52):
                d = results_io.llmsr_dir(
                    results_io.seed_dir(os.path.join(root, group), seed),
                    tau, f"{stem}_{split}")
                p = os.path.join(d, results_io.RESULTS_NAME)
                if os.path.exists(p):
                    # load_method_data keys by label; keep one entry per label by
                    # letting it merge the per-seed artifacts under that label.
                    methods_data.setdefault(label, []).append(p)
        if not methods_data:
            print(f"  [skip] llmsrbench tau={tau:g}: no artifacts")
            continue
        print(f"[llmsrbench] tau={tau:g}: "
              + ", ".join(f"{k}({len(v)} seeds)" for k, v in methods_data.items()),
              flush=True)
        # load_method_data returns {seed: {equation_id: (formula, time)}} for ONE
        # artifact; each per-seed file contributes its own seed key, so the per-label
        # merge is a dict update, not a list concat.  eval_ood_raw_for_gap wants
        # {label: {seed: {equation_id: ...}}}.
        # The LLM-SRBench OOD evaluator routes on notation: plot_llmsrbench_ood_vs_gap
        # ._is_infix looks for "x_<digit>" and, failing that, hands the string to
        # wrapExtEvalPN as PREFIX over v1..vN.  Our three write INFIX over v1..vn --
        # neither branch -- so every row silently scored 0.0.  Rewrite v<i> to
        # x_<i-1> so the infix branch takes them.  (The SRBench half needs no such
        # fixup: it passes style="v1" to _ood_eval_infix explicitly.)
        loaded = {}
        for label, paths in methods_data.items():
            merged = {}
            for p in paths:
                try:
                    merged.update(load_method_data(p))
                except Exception as exc:
                    print(f"  [warn] {label} {os.path.basename(p)}: {exc}")
            if merged:
                merged = {seed: {eq: (_v_to_x0(f), t) for eq, (f, t) in pm.items()}
                          for seed, pm in merged.items()}
                loaded[label] = merged
                print(f"    {label}: seeds {sorted(merged)}", flush=True)
        if not loaded:
            continue
        df = _llmsr_ood_rows(loaded, gaps)
        if df.empty:
            print(f"  [warn] no llmsrbench rows for tau={tau:g}")
            continue
        _append_idempotent(
            os.path.join(root, ood_transform.llmsr_cache_name(tau)), df)


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=sorted(METHODS),
                    choices=sorted(METHODS))
    ap.add_argument("--noises", type=float, nargs="+", default=DEFAULT_NOISES)
    ap.add_argument("--gaps", type=int, nargs="+", default=DEFAULT_GAPS)
    ap.add_argument("--bench", choices=["srbench", "llmsrbench", "both"], default="both")
    ap.add_argument("--split", default="lsr_transform")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--check-domain", action="store_true",
                    help="Print the OOD domain method recorded in each dataset and exit.")
    args = ap.parse_args()

    if args.check_domain:
        for g in args.gaps:
            for name, fn in (("feynman", ood_transform.feynman_ood_path),
                             ("llmsrbench", ood_transform.llmsr_ood_path)):
                p = os.path.join(SCRIPT_DIR, fn(g))
                if not os.path.exists(p):
                    print(f"  {name:<11} g={g:<4} MISSING"); continue
                with gzip.open(p, "rb") as fh:
                    d = pickle.load(fh)
                print(f"  {name:<11} g={g:<4} method={d.get('method')} "
                      f"n={len(d.get('datasets', {}))}")
        return

    if args.bench in ("srbench", "both"):
        augment_srbench(args.results_root, args.methods, args.noises, args.gaps)
    if args.bench in ("llmsrbench", "both"):
        augment_llmsr(args.results_root, args.methods, args.noises, args.gaps,
                      args.split)


if __name__ == "__main__":
    main()
