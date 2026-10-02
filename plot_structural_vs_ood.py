#!/usr/bin/env python3
"""
plot_structural_vs_ood.py -- Formula edit distance to GT vs OOD R^2 for the 5 models.

For every predicted formula (best formula per dataset, per model) this computes two
independent quantities and scatters one against the other:

  * OOD R^2  (x): the predicted formula re-evaluated on the pre-generated OOD set
    datasets/feynman_ood_g<gap>.pkl.gz (same PN evaluator + clip-to-[0,1] as
    compare_srbench).  --gap selects the distribution-shift gap (default 128).

  * Edit distance to GT (y): a CONTINUOUS structural-distance measure (0 = identical
    structure, 1 = maximally different) -- the graded replacement for the binary
    SRBench symbolic_solution.  It is the Levenshtein (token edit) distance between
    the predicted and ground-truth formulas after putting both in a canonical,
    constant-agnostic skeleton form, normalised by the longer formula's length:
        1. parse each mantissa-free-grammar PN into an expression tree;
        2. skeletonise -- every numeric literal (floats, 0/1/2/3, pi) -> a single "C"
           token, so the distance measures STRUCTURE (operators, variables, nesting),
           not the exact fitted constants;
        3. canonicalise the order of commutative operands (+, *) so a+b and b+a match;
        4. serialise back to a prefix token stream and take the normalised
           Levenshtein distance.
    Like symbolic_solution this is a PURE-FORM check (independent of the OOD data),
    so the scatter shows how formula-structure distance tracks OOD extrapolation --
    but as a gradient rather than all-or-nothing.  It is gap-independent, so
    --reuse-metric skips the recomputation when sweeping multiple gaps.

Usage:
    python plot_structural_vs_ood.py [--gap 128]
        [--results results/eval_tf_145M_40.pkl.gz ...]
        [--output plots/structural_vs_ood_g128.png] [--csv ...] [--show]
"""
import argparse
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from plot_io import save_fig

import results_io  # noqa: E402
from compare_srbench import (  # noqa: E402
    _SP_BINARY, _SP_UNARY, _BASELINE_COLOR,
    _ood_pn_evaluator, _ood_load_tf_best, _ood_eval_pn,
)
from plot_ood_vs_gap import _tf_color, _remap_scale_labels, _label_for_path  # noqa: E402

# Operator arities (single source of truth = the sympy converter's op sets).
_ARITY = {**{t: 2 for t in _SP_BINARY}, **{t: 1 for t in _SP_UNARY}}
_COMMUTATIVE = {"+", "*"}          # operands may be reordered freely


# -- Structural edit distance -------------------------------------------------
def _is_num(tok: str) -> bool:
    if tok == "pi":
        return True
    try:
        float(tok)
        return True
    except (TypeError, ValueError):
        return False


def _pn_to_tree(tokens):
    """Parse a prefix token list into a (label, [children]) tree, skeletonising
    every numeric leaf to "C".  Returns (tree, n_tokens_consumed)."""
    pos = 0

    def rec():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        arity = _ARITY.get(tok, 0)
        kids = [rec() for _ in range(arity)]
        label = "C" if (arity == 0 and _is_num(tok)) else tok   # variables kept as-is
        return (label, kids)

    tree = rec()
    return tree, pos


def _canon(node):
    """Recursively sort the operands of commutative ops so a+b == b+a."""
    label, kids = node
    kids = [_canon(k) for k in kids]
    if label in _COMMUTATIVE:
        kids = sorted(kids, key=_serialize)
    return (label, kids)


def _serialize(node):
    """Flatten a tree back to a prefix token list."""
    label, kids = node
    out = [label]
    for k in kids:
        out.extend(_serialize(k))
    return out


def _levenshtein(a, b) -> int:
    """Token-level edit distance between two sequences (unit ins/del/sub cost)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def formula_edit_distance(pred_pn: str, gt_pn: str):
    """(normalised, raw) structural edit distance between two PN formulas.

    normalised in [0, 1] = Levenshtein(canon skeleton) / max(len).  An unparseable /
    empty prediction is treated as maximally distant (1.0).
    """
    if not pred_pn or not gt_pn or not str(pred_pn).strip() or not str(gt_pn).strip():
        return 1.0, None
    try:
        a = _serialize(_canon(_pn_to_tree(pred_pn.split())[0]))
        b = _serialize(_canon(_pn_to_tree(gt_pn.split())[0]))
    except Exception:
        return 1.0, None
    d = _levenshtein(a, b)
    n = max(len(a), len(b), 1)
    return d / n, d


# -- Data assembly ------------------------------------------------------------
def collect(tf_files, gap: int, metric_cache: dict | None = None) -> pd.DataFrame:
    """Return DataFrame[model, dataset, ood_r2, edit_dist, edit_dist_raw].

    Edit distance is gap-independent (formula-vs-GT), so `metric_cache` -- a
    {(model, dataset): normalised_edit_dist} map from a prior run -- lets a multi-gap
    sweep skip its recomputation and only redo the cheap OOD R^2 per gap.
    """
    ood_path = os.path.join(SCRIPT_DIR, f"datasets/feynman_ood_g{gap}.pkl.gz")
    if not os.path.exists(ood_path):
        raise SystemExit(f"OOD dataset not found: {ood_path}")
    with gzip.open(ood_path, "rb") as fh:
        ood = pickle.load(fh)["datasets"]          # {dataset: {"X","y"}}
    pn_eval = _ood_pn_evaluator()

    records = []
    for label, path in tf_files:
        best = _ood_load_tf_best(path)             # {dataset: (seed, best_formula)}
        with gzip.open(path, "rb") as fh:
            gt_of = {r["dataset"]: r.get("target_pn", "") for r in pickle.load(fh)["results"]}
        n_proc = 0
        dsum = 0.0
        for ds, (_seed, formula) in best.items():
            if ds not in ood:          # only datasets present in this gap's OOD set
                continue
            r2 = _ood_eval_pn(formula, ood[ds]["X"], ood[ds]["y"], pn_eval)
            cached = metric_cache.get((label, ds)) if metric_cache is not None else None
            if cached is not None:
                nd, rd = float(cached), None
            else:
                nd, rd = formula_edit_distance(formula, gt_of.get(ds, ""))
            n_proc += 1
            dsum += nd
            records.append({"model": label, "dataset": ds, "ood_r2": float(r2),
                            "edit_dist": float(nd), "edit_dist_raw": rd})
        _src = "cached" if metric_cache is not None else "computed"
        print(f"  [{label:14s}] {n_proc:3d} formulas (in OOD set)  |  mean edit dist "
              f"({_src}): {dsum / max(n_proc, 1):.3f}", flush=True)
    return pd.DataFrame(records)


# -- Plot ---------------------------------------------------------------------
def _color(label):
    return _tf_color(label) or _BASELINE_COLOR


def make_plot(df: pd.DataFrame, gap: int, out_path: str, r2_thr: float, show: bool):
    models = sorted(df["model"].unique(),
                    key=lambda m: df[df.model == m]["edit_dist"].mean())
    rng = np.random.default_rng(0)

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(f"Formula edit distance to GT  vs  OOD R^2   (gap = {gap})",
                 fontsize=15, fontweight="bold", y=0.98)

    # -- Left: per-prediction scatter (y = edit distance) + trend --
    for m in models:
        sub = df[df.model == m]
        ax.scatter(sub["ood_r2"], sub["edit_dist"], s=42, color=_color(m), alpha=0.6,
                   edgecolor="white", linewidth=0.4,
                   label=f"{m}  (mean {sub['edit_dist'].mean():.2f}, n={len(sub)})")

    # Mean edit distance per OOD-R^2 bin. OOD R^2 is bimodal (piles at 0 and 1), so we
    # bin coarsely and draw only well-populated bins (n>=8) -- fine bins would be noisy.
    edges = [0.0, 0.5, 0.9, 0.99, 1.0001]
    ctr, mean_d, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (df["ood_r2"] >= lo) & (df["ood_r2"] < hi)
        if m.sum() >= 8:
            ctr.append(df.loc[m, "ood_r2"].mean())
            mean_d.append(df.loc[m, "edit_dist"].mean())
            cnt.append(int(m.sum()))
    ax.plot(ctr, mean_d, color="black", lw=2.2, marker="s", markersize=9,
            markeredgecolor="white", zorder=5, label="mean edit dist | OOD R^2 bin")
    for x, d, c in zip(ctr, mean_d, cnt):
        ax.annotate(f"{d:.2f}\n(n={c})", (x, d), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="0.25")

    r_p = (np.corrcoef(df["edit_dist"], df["ood_r2"])[0, 1]
           if len(df) > 1 else float("nan"))
    ax.text(0.5, 0.92, f"Pearson corr(edit dist, OOD R^2) = {r_p:.2f}",
            transform=ax.transAxes, fontsize=10, color="0.25", ha="center",
            bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    ax.axvline(r2_thr, color="0.6", ls=":", lw=1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(f"OOD R^2  (feynman_ood_g{gap}, clipped to [0,1])", fontsize=12)
    ax.set_ylabel("Normalized formula edit distance to GT\n(0 = identical structure, 1 = maximally different)",
                  fontsize=12)
    ax.set_title("Per-prediction: structural distance vs OOD extrapolation", fontsize=12)
    ax.grid(ls="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # -- Right: edit-distance distribution split by whether OOD R^2 clears threshold --
    solved = df["ood_r2"] >= r2_thr
    groups = [(f"OOD R^2>={r2_thr}\n(extrapolates)", df[solved]["edit_dist"].values, "#15803d"),
              (f"OOD R^2<{r2_thr}\n(fails OOD)",     df[~solved]["edit_dist"].values, "#b91c1c")]
    for i, (name, vals, col) in enumerate(groups):
        if len(vals) == 0:
            continue
        axb.scatter(np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals)),
                    vals, s=20, color=col, alpha=0.5, edgecolor="none")
        bp = axb.boxplot([vals], positions=[i], widths=0.5, patch_artist=True,
                         showfliers=False, medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set(facecolor=col, alpha=0.25)
        axb.text(i, 1.02, f"med {np.median(vals):.2f}\nn={len(vals)}",
                 ha="center", fontsize=8, color=col)
    axb.set_xticks([0, 1]); axb.set_xticklabels([g[0] for g in groups])
    axb.set_ylim(-0.03, 1.10)
    axb.set_ylabel("Edit distance to GT", fontsize=11)
    axb.set_title("Edit distance by OOD outcome", fontsize=12)
    axb.grid(axis="y", ls="--", alpha=0.4)
    axb.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.join(SCRIPT_DIR, out_path)), exist_ok=True)
    save_fig(fig, os.path.join(SCRIPT_DIR, out_path))
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gap", type=int, default=128,
                    help="OOD distribution-shift gap (datasets/feynman_ood_g<gap>.pkl.gz). Default: 128.")
    ap.add_argument("--results", nargs="+", default=None,
                    help="Transformer eval_mymodels .pkl.gz files. Default: auto-discover the "
                         "merged noise-free artifacts under results/mymodels/noise_0/.")
    ap.add_argument("--r2-threshold", type=float, default=0.99, dest="r2_thr")
    ap.add_argument("--output", default=None,
                    help="Output PNG. Default: plots/structural_vs_ood_g<gap>.png")
    ap.add_argument("--csv", default=None,
                    help="Optional path to also write the per-prediction table as CSV.")
    ap.add_argument("--from-csv", default=None,
                    help="Skip recomputation and re-plot from a previously written --csv "
                         "table (columns: model, dataset, ood_r2, edit_dist).")
    ap.add_argument("--reuse-metric", default=None,
                    help="Load the (gap-independent) edit_dist column from a prior --csv "
                         "and recompute only OOD R^2 for --gap -- fast path for a multi-gap sweep.")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    if args.output is None:
        args.output = f"plots/structural_vs_ood_g{args.gap}.png"

    # This plot is noise-free only (no --noise flag); it reads the merged transformer
    # artifacts under results/mymodels/noise_0/.
    if args.results is None:
        result_paths = results_io.list_srbench_pkls(
            os.path.join(SCRIPT_DIR, "results"), 0.0, "mymodels")
    else:
        result_paths = [os.path.join(SCRIPT_DIR, p) for p in args.results]
    tf_files = []
    for ap_ in result_paths:
        if not os.path.exists(ap_):
            print(f"[warn] missing, skipping: {ap_}")
            continue
        tf_files.append((_label_for_path(ap_), ap_))
    if not tf_files:
        raise SystemExit("No transformer result files found.")

    # Canonicalise labels (89M / 89M_unscaled / 145M / 145M_unscaled / 89M-float).
    remap = _remap_scale_labels([lbl for lbl, _ in tf_files])
    tf_files = [(remap.get(lbl, lbl), path) for lbl, path in tf_files]

    if args.from_csv:
        df = pd.read_csv(os.path.join(SCRIPT_DIR, args.from_csv))
        print(f"Re-plotting from {args.from_csv} ({len(df)} rows).")
    else:
        metric_cache = None
        if args.reuse_metric:
            _mc = pd.read_csv(os.path.join(SCRIPT_DIR, args.reuse_metric))
            metric_cache = {(r["model"], r["dataset"]): float(r["edit_dist"])
                            for _, r in _mc.iterrows()}
            print(f"Reusing edit_dist from {args.reuse_metric} ({len(metric_cache)} entries); "
                  f"recomputing only OOD R^2 (gap {args.gap}).", flush=True)
        else:
            print(f"Computing OOD R^2 (gap {args.gap}) + formula edit distance for "
                  f"{len(tf_files)} models ...", flush=True)
        df = collect(tf_files, args.gap, metric_cache=metric_cache)
    if df.empty:
        raise SystemExit("No predictions to plot.")

    # Summary.
    print(f"\n{'='*68}\nFormula edit distance to GT  vs  OOD R^2  (gap {args.gap})\n{'='*68}")
    for m, g in df.groupby("model"):
        sol = g["ood_r2"] >= args.r2_thr
        print(f"  {m:14s} n={len(g):3d}  mean edit dist={g['edit_dist'].mean():.3f}  "
              f"| by OOD: solved={g[sol]['edit_dist'].mean():.3f} / "
              f"failed={g[~sol]['edit_dist'].mean():.3f}")
    if len(df) > 1:
        r = np.corrcoef(df["edit_dist"], df["ood_r2"])[0, 1]
        print(f"  Pearson corr(edit dist, OOD R^2) = {r:.3f}  (expect negative)")
    print("=" * 68)

    if args.csv:
        df.to_csv(os.path.join(SCRIPT_DIR, args.csv), index=False)
        print(f"[csv]  Saved -> {args.csv}")
    make_plot(df, args.gap, args.output, args.r2_thr, args.show)


if __name__ == "__main__":
    main()
