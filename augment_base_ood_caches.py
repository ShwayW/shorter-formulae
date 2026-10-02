#!/usr/bin/env python3
"""
augment_base_ood_caches.py -- compute the OOD R^2 of the BASE transformer models
(89M / 145M, and optionally the whitened + float variants) and append them to the
SRBench OOD caches.

Why this exists: results/ood_raw_noise<tau>.csv is built up by the per-method augment
scripts (augment_{tpsr,devncon}_caches.py).  Each of those builds its tf_file_labels
list from its OWN method's pkls and picks up the SRBench published baselines + e2e for
free inside compute_ood_r2 (feather_path=SRBENCH_GT).  None of them contributes the base
models -- so a cache assembled purely from those augments contains every method EXCEPT
89M / 145M, which is exactly the state the four caches are in.

plot_ood_vs_gap.py --reuse-ood cannot repair that: its reuse branch backfills missing
*gaps* only, never missing *algorithms* (plot_ood_vs_gap.py:267-274).  Dropping
--reuse-ood does fix it, but recomputes all ~21 methods (~11.5 min/gap x 36) to obtain
these two.  This script computes just the base rows and appends them idempotently,
leaving every other method's rows untouched.

Both halves are covered: the SRBench caches (OOD + complexity) and, since the LLM-SRBench
half used to be missing here entirely, the llmsr caches too.  Before that, our models'
llmsr rows could only enter the cache via a full (non---reuse-ood) run of
plot_llmsrbench_ood_vs_gap.py, which recomputes every method to obtain them.

Usage:
    python augment_base_ood_caches.py                    # m145 + m89 (145M + 89M)
    python augment_base_ood_caches.py --all-labels       # + _unscale / m89float variants
    python augment_base_ood_caches.py --noises 0         # one noise level
    python augment_base_ood_caches.py --bench srbench    # one half

A model evaluated under a sibling protocol lives in its own results group and is not in
FIGURE_LABELS, so it needs both flags or it is filtered out silently:

    python augment_base_ood_caches.py --group mymodels_400 --labels simplipy_145M_80l
"""
import argparse
import gzip
import os
import pickle
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import results_io
import ood_transform
from compare_srbench import _ood_load_tf_all, _ood_pn_evaluator, _ood_eval_pn
from plot_ood_vs_gap import _label_for_path, _remap_scale_labels
from augment_tpsr_caches import _append_idempotent, _llmsr_ood_rows
from plot_complexity_vs_noise import _load_complexity
from plot_llmsr_time_complexity_vs_noise import _method_complexity
from plot_llmsrbench_ood_vs_gap import discover_methods_from_dirs, load_method_data

DEFAULT_NOISES = [0.0, 0.001, 0.01, 0.1]
DEFAULT_GAPS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
# The two labels the six figures actually draw (SRBENCH_KEEP in make_requested_plots.sh:36):
# m145 -> 145M, m89 -> 89M.  The whitened (_unscale) and float (m89float) variants
# are dropped by SRBENCH_DROP / absent from SRBENCH_KEEP, so they are opt-in here.
# Both spellings: artifacts predating the 2026-08-17 checkpoint rename are labelled
# m145/m89, later ones 145M_40/89M_40.
# Both sides of the simp-rename: the same weights write different artifact labels
# depending on whether the run predates checkpoints/<name>_simp{0,1}_res.
FIGURE_LABELS = {"m145", "m89",
                 "145M_40", "89M_40", "145M_80",
                 "145M_40_simp1", "89M_40_simp1", "145M_80_simp0"}


def _base_ood_rows(pkls, gaps):
    """Per-(gap, raw_label, dataset, seed) OOD R^2 -- the transformer-only branch of
    compare_srbench.compute_ood_r2 (no baselines / e2e / GT), so only these pkls are
    evaluated.  Mirrors augment_tpsr_caches._srbench_ood_rows."""
    pn_eval = _ood_pn_evaluator()
    rows = []
    for g in gaps:
        ood_path = os.path.join(SCRIPT_DIR, ood_transform.feynman_ood_path(g))
        if not os.path.exists(ood_path):
            print(f"  [warn] missing {ood_path}")
            continue
        with gzip.open(ood_path, "rb") as fh:
            ood_datasets = pickle.load(fh)["datasets"]
        for path in pkls:
            label = _label_for_path(path)                 # m145 / m89 / ...
            for ds, seed_fmls in _ood_load_tf_all(path).items():
                if ds not in ood_datasets:
                    continue
                X, y = ood_datasets[ds]["X"], ood_datasets[ds]["y"]
                for seed, formula in seed_fmls:
                    rows.append({"algorithm": label, "dataset": ds,
                                 "ood_r2": _ood_eval_pn(formula, X, y, pn_eval),
                                 "seed": int(seed), "gap": g})
        print(f"  [gap={g:>3}] SRBench OOD done", flush=True)
    return pd.DataFrame(rows, columns=["algorithm", "dataset", "ood_r2", "seed", "gap"])


def _srbench_complexity_rows(pkls, tau, r2_thr):
    """The REPO's complexity metric (sympy node count) for each base-model pkl.

    _load_complexity also returns every SRBench published baseline, so the rows we want
    are picked out by label rather than relabelling everything it hands back -- the same
    trap augment_pysr_caches._srbench_complexity_rows guards against.

    is_ours=True: these ARE our checkpoints, which is what the complexity figures use to
    style them apart from the external methods.
    """
    cplx, _mine, _common = _load_complexity(pkls, None, None, r2_thr, tau)
    # DISPLAY labels, not raw: unlike the OOD cache (raw, remapped at plot time), the
    # complexity cache stores '145M'/'89M', and _load_complexity keys its table the same
    # way.  Filtering on the raw label silently matched nothing.
    wanted = set(_remap_scale_labels([_label_for_path(p) for p in pkls]).values())
    recs = [{"algorithm": k, "complexity": c, "std": s, "is_ours": True}
            for k, (c, s) in cplx.items() if k in wanted]
    if not recs:
        print(f"  [warn] no complexity rows for {sorted(wanted)} "
              f"(got {sorted(cplx)[:4]}...) -- skipping")
    return pd.DataFrame(recs, columns=["algorithm", "complexity", "std", "is_ours"])


def augment_llmsr(root, group, noises, gaps, split, r2_thr, labels):
    """The LLM-SRBench half, which this script historically lacked entirely.

    Our models' llmsr rows used to enter the cache only via a full (non---reuse-ood) run
    of plot_llmsrbench_ood_vs_gap.py, which recomputes every method to obtain them.  The
    artifacts are prefix PN, which _llmsr_ood_rows already handles -- it is what draws
    the '<model>+TPSR' arms, which are these same checkpoints.
    """
    for tau in noises:
        print(f"[LLM-SRBench] noise {tau:g}", flush=True)
        dirs = results_io.list_llmsr_method_dirs(root, tau, split, groups=(group,))
        methods = discover_methods_from_dirs(dirs, split)
        if labels is not None:
            # Match EITHER label space: discover_methods_from_dirs returns the DISPLAY
            # label, while --labels is naturally given as the raw artifact/model name
            # (the same string the SRBench half filters on).  Comparing only against the
            # display label silently discovered nothing.
            def _keep(label, art):
                raw = os.path.basename(os.path.dirname(art)).replace(f"_{split}", "")
                return label in labels or raw in labels
            methods = {k: v for k, v in methods.items() if _keep(k, v)}
        if not methods:
            print(f"  no {group} llmsrbench dirs at this noise -- skipping")
            continue
        print(f"  labels: {sorted(methods)}", flush=True)
        ood = _llmsr_ood_rows({k: load_method_data(v) for k, v in methods.items()}, gaps)
        if not ood.empty:
            _append_idempotent(
                os.path.join(root, ood_transform.llmsr_cache_name(tau)), ood)
        recs = []
        for label, art in methods.items():
            st = _method_complexity(art, r2_thr)
            if st is not None:
                recs.append({"algorithm": label, "complexity": st[0], "std": st[1],
                             "is_ours": True})
        if recs:
            _append_idempotent(
                os.path.join(root, ood_transform.llmsr_complexity_cache_name(tau)),
                pd.DataFrame(recs, columns=["algorithm", "complexity", "std", "is_ours"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noises", type=float, nargs="+", default=DEFAULT_NOISES)
    ap.add_argument("--gaps", type=int, nargs="+", default=DEFAULT_GAPS)
    ap.add_argument("--all-labels", action="store_true",
                    help="Also compute the whitened (_unscale) and 89M-float (m89float) "
                         "variants. Off by default: no figure in make_requested_plots.sh "
                         "draws them, and they cost ~2.5x the runtime.")
    ap.add_argument("--group", default="mymodels", metavar="DIR",
                    help="Results group to read the artifacts from (default: mymodels). "
                         "Point it at a sibling tree -- e.g. mymodels_400, the 400-point "
                         "IO-bag protocol the big models must be fed -- to augment the "
                         "models evaluated there.")
    ap.add_argument("--labels", nargs="+", default=None, metavar="L",
                    help="Only these labels, instead of the FIGURE_LABELS whitelist. "
                         "Needed for a model whose name is not yet in that set, which "
                         "would otherwise be filtered out silently.")
    ap.add_argument("--bench", choices=["all", "srbench", "llmsr"], default="all",
                    help="Which half to augment. Default: all.")
    ap.add_argument("--r2-threshold", type=float, default=0.99, dest="r2_thr")
    ap.add_argument("--split", default="lsr_transform")
    ap.add_argument("--results-root", default=os.path.join(SCRIPT_DIR, "results"))
    args = ap.parse_args()

    # GUARD: _remap_scale_labels only inserts "_unscaled" when a label's PLAIN twin is in
    # the same batch (it tests `paired`).  Ask for the --unscale arms alone and it strips
    # the suffix instead, so their rows are written under the PLAIN arms' names and
    # silently REPLACE them -- observed 2026-09-23 on the t120M ablation, where the four
    # plain complexity rows were overwritten with the unscaled values (12.16 -> 23.27)
    # before anyone could notice.  Refuse rather than corrupt: naming both is also what
    # makes the pairing produce the right display label in the first place.
    if args.labels:
        lonely = [l for l in args.labels
                  if l.endswith("_unscale") and l[:-len("_unscale")] not in args.labels]
        if lonely:
            sys.exit("[error] --labels lists these --unscale arms without their plain "
                     "counterparts:\n         " + "\n         ".join(lonely) +
                     "\n       Their rows would be written under the PLAIN names and "
                     "overwrite them.\n       Add the plain labels to --labels and re-run.")

    labels = set(args.labels) if args.labels else None

    if args.bench in ("all", "srbench"):
        for tau in args.noises:
            print(f"[SRBench] noise {tau:g}", flush=True)
            pkls = list(results_io.list_srbench_pkls(args.results_root, tau, args.group))
            if labels is not None:
                pkls = [p for p in pkls if _label_for_path(p) in labels]
            elif not args.all_labels:
                pkls = [p for p in pkls if _label_for_path(p) in FIGURE_LABELS]
            if not pkls:
                print(f"  no base {args.group} pkls -- skipping")
                continue
            print(f"  labels: {sorted(_label_for_path(p) for p in pkls)}", flush=True)
            ood = _base_ood_rows(pkls, args.gaps)
            if ood.empty:
                print("  no rows produced -- skipping")
                continue
            _append_idempotent(os.path.join(args.results_root,
                                            ood_transform.srbench_cache_name(tau)), ood)
            cplx = _srbench_complexity_rows(pkls, tau, args.r2_thr)
            if not cplx.empty:
                _append_idempotent(
                    os.path.join(args.results_root,
                                 ood_transform.srbench_complexity_cache_name(tau)), cplx)

    if args.bench in ("all", "llmsr"):
        augment_llmsr(args.results_root, args.group, args.noises, args.gaps,
                      args.split, args.r2_thr, labels)
    print("Base-model OOD augmentation complete.")


if __name__ == "__main__":
    main()
