#!/usr/bin/env python3
"""
proximity_vs_ood.py -- Does a sampler draw that LOOKS like a Feynman equation also
BEHAVE like one out of distribution?

Unlike plot_structural_vs_ood.py, which scatters *model predictions* against their own
ground truth, this script asks the prior-side question: draw N formulae straight from
the online training generator (data_process.sample_formula -- the exact distribution
the transformer is trained on, no model involved), snap each one to its structurally
nearest Feynman equation, and see whether structural proximity buys any numerical
agreement on that equation's OOD region.

Pipeline
--------
  1. SAMPLE   N formulae from data_process.sample_formula under a fixed seed (42).
  2. MATCH    each draw against all 99 Feynman ground-truth formulae, converted to the
              grammar's canonical PN by eval_mymodels.load_feynman_pn_targets, and keep the
              argmin of the structural edit distance (see below).  Ties break on the
              dataset name, so the match is deterministic.
  3. SCORE    the draw's OOD R^2 against that equation's pre-generated OOD set
              datasets/feynman_ood_g<gap>.pkl.gz (default gap 2), evaluated with the
              same C PN evaluator the rest of the repo uses.
  4. PLOT     x = edit distance to the nearest GT, y = OOD R^2.

Edit distance (x)
-----------------
Reuses plot_structural_vs_ood.formula_edit_distance: both formulae are parsed into
trees, every numeric leaf is skeletonised to "C" (so the distance measures STRUCTURE,
not fitted constants), commutative operands are sorted, and the token-level
Levenshtein distance is normalised by the longer skeleton.  0 = identical structure,
1 = maximally different.  --distance raw switches to the unnormalised token count for
both the matching and the axis; normalised is the default because raw distance makes
long ground truths artificially "far" from every short draw.

OOD R^2 (y) -- read this before interpreting the plot
-----------------------------------------------------
A sampler draw is a bare skeleton with no fitted constants, so it carries none of the
target's units or scale.  Evaluated as-is it scores R^2 = 0 against essentially every
Feynman equation (measured: 0/300 draws above zero), which makes the raw metric a flat
line at zero rather than a signal.  Three variants are therefore computed for every
row and any one can be plotted with --r2:

  scaled  (default) -- R^2 after the best affine rescale a*f(x)+b on the OOD set.
                       Identical to corr(f(X_ood), y_ood)^2, i.e. how well the draw's
                       SHAPE tracks the target where units are quotiented out.  This
                       is the variant that carries signal; it is a similarity measure,
                       NOT a generalisation claim -- a,b are read off the OOD set.
  id-fit            -- a,b fitted on the in-distribution set (gap 0), then applied at
                       --gap.  The honest extrapolation number, and near-zero almost
                       everywhere: an affine fix-up learned in-distribution does not
                       survive the shift.  Needs datasets/feynman_ood_g0.pkl.gz.
  direct            -- no rescaling at all (compare_srbench._ood_eval_pn, the
                       convention used for real model predictions).  Zero everywhere.

All three are clipped to [0, 1] exactly as compare_srbench does, so they sit on the
same scale as every other OOD number in the repo.

Variable-count mismatch
-----------------------
A draw may use more variables than the matched dataset has columns.  Following
compare_srbench._ood_eval_pn, the missing columns are zero-filled and the row is kept
with var_overflow=True (about a third of draws at the default settings).  Pass
--require-var-fit to instead restrict each draw's match to datasets with at least as
many columns, or --drop-var-overflow to keep the unrestricted match but exclude those
rows from the plot.

Usage
-----
    python proximity_vs_ood.py                                  # N=1000, seed 42, gap 2
    python proximity_vs_ood.py --n 5000 --gap 4 --csv out/prox_g4.csv
    python proximity_vs_ood.py --r2 id-fit --output plots/prox_idfit.png
    PLOT_FORMAT=pdf python proximity_vs_ood.py                  # vector output
"""
import argparse
import gzip
import os
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from plot_io import save_fig                                             # noqa: E402
from grammar import V, set_operator_weights                              # noqa: E402
from data_process import sample_formula                                  # noqa: E402
from eval_mymodels import load_feynman_pn_targets                              # noqa: E402
from compare_srbench import _ood_pn_evaluator, _ood_r2_score             # noqa: E402
from plot_structural_vs_ood import (                                     # noqa: E402
    _pn_to_tree, _canon, _serialize, _levenshtein,
)

_N_VARS = len(V)                 # the PN evaluator always wants len(V) input columns

# Single-hue scatter (the repo's steel blue) + the status pair used by
# plot_structural_vs_ood for the solved/failed split.
_SCATTER_COLOR = "#457b9d"
_GOOD_COLOR    = "#15803d"
_BAD_COLOR     = "#b91c1c"

# Generator defaults.  These mirror the live training configuration in
# train_transformer.py's __main__ block, so "the sampler" here is the same sampler the
# checkpoints were trained on -- with one deliberate exception: simplify_targets stays
# True (the data_process default).  Training currently draws RAW trees, but an
# unsimplified draw would be compared against a CANONICAL ground truth, inflating the
# edit distance for algebraic reasons that have nothing to do with structure.  Pass
# --no-simplify to reproduce the raw-target training distribution instead.
_GEN_DEFAULTS = dict(
    fml_len_range=(1, 80),
    complexity_decay=1.0,
    length_decay=0.99,
    operator_weights="tiered",
    simplify_targets=True,
    use_prefactors=False,
    restrict_consts=True,
)


# -- Sampling ------------------------------------------------------------------
def draw_formulas(n: int, seed: int, gen: dict, verbose: bool = True) -> list:
    """[(formula_pn, n_vars)] -- n accepted draws from the online generator.

    Rejected draws (constant-valued, or past the length cap) are re-drawn, exactly as
    online_batch_iterater does, so the returned set is the generator's formula
    distribution conditioned on acceptance.
    """
    set_operator_weights(gen["operator_weights"])       # per-process grammar setting
    np.random.seed(seed)
    random.seed(seed)

    out, attempts, t0 = [], 0, time.time()
    while len(out) < n:
        attempts += 1
        drawn = sample_formula(
            gen["fml_len_range"],
            complexity_decay=gen["complexity_decay"],
            length_decay=gen["length_decay"],
            simplify_targets=gen["simplify_targets"],
            use_prefactors=gen["use_prefactors"],
            restrict_consts=gen["restrict_consts"],
        )
        if drawn is not None:
            out.append(drawn)
    if verbose:
        lens = np.array([len(f.split()) for f, _ in out])
        print(f"  drew {n} formulae in {time.time() - t0:.1f}s "
              f"({attempts} attempts, {100 * (1 - n / attempts):.0f}% rejected)  |  "
              f"length: median {np.median(lens):.0f}, mean {lens.mean():.1f}, max {lens.max()}  |  "
              f"vars: mean {np.mean([k for _, k in out]):.1f}", flush=True)
    return out


# -- Structural distance -------------------------------------------------------
def skeleton(pn: str):
    """Canonical, constant-agnostic prefix token list for a PN formula (None on
    failure).  Factored out of plot_structural_vs_ood.formula_edit_distance so the 99
    ground-truth skeletons are built once instead of once per sampled formula."""
    if not pn or not str(pn).strip():
        return None
    try:
        return _serialize(_canon(_pn_to_tree(str(pn).split())[0]))
    except Exception:
        return None


def nearest_gt(sk, gt_skeletons: dict, normalise: bool, min_len: int = 0):
    """(dataset, normalised_distance, raw_distance) for the closest ground truth.

    gt_skeletons: {dataset: (skeleton_tokens, n_cols)}.  Iterated in sorted order and
    compared with a strict <, so ties break deterministically on the dataset name.
    min_len > 0 restricts the search to datasets with at least that many columns
    (--require-var-fit).  Returns None when nothing is eligible.
    """
    best = None
    for ds in sorted(gt_skeletons):
        gt_sk, n_cols = gt_skeletons[ds]
        if n_cols < min_len:
            continue
        raw = _levenshtein(sk, gt_sk)
        norm = raw / max(len(sk), len(gt_sk), 1)
        key = norm if normalise else raw
        if best is None or key < best[0]:
            best = (key, ds, norm, raw)
    if best is None:
        return None
    _key, ds, norm, raw = best
    return ds, norm, raw


# -- OOD scoring ---------------------------------------------------------------
def predict_pn(formula: str, X: np.ndarray, pn_eval):
    """f(X) for a PN formula, or None if it does not evaluate cleanly.

    Mirrors compare_srbench._ood_eval_pn: the input is zero-padded to len(V) columns
    (a formula may reference more variables than the dataset has), and a non-zero
    stack-left or any non-finite output rejects the whole prediction.
    """
    try:
        X64 = np.asarray(X, np.float64)
        if X64.shape[1] < _N_VARS:
            X64 = np.concatenate(
                [X64, np.zeros((X64.shape[0], _N_VARS - X64.shape[1]), np.float64)], axis=1)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            preds, stacklefts = pn_eval(formula, X64)
        preds = np.asarray(preds, np.float64)
        if np.any(stacklefts != 0) or not np.isfinite(preds).all():
            return None
        return preds
    except Exception:
        return None


def score_row(formula: str, ood_ds: dict, id_ds, pn_eval) -> dict:
    """{direct, scaled, id_fit} OOD R^2 for one formula on one dataset.

    All three are clipped to [0, 1] by _ood_r2_score, matching every other OOD number
    in the repo.  A formula that fails to evaluate scores 0 on all three.
    """
    out = {"r2_direct": 0.0, "r2_scaled": 0.0, "r2_id_fit": 0.0}
    y = np.asarray(ood_ds["y"], np.float64)
    p = predict_pn(formula, ood_ds["X"], pn_eval)
    if p is None:
        return out

    out["r2_direct"] = _ood_r2_score(y, p)

    # scaled: best affine rescale on the OOD set.  max_{a,b} R^2(y, a*p+b) is exactly
    # the squared Pearson correlation, computed directly to avoid the ill-conditioned
    # least-squares solve when p is near-constant.
    sp, sy = float(np.std(p)), float(np.std(y))
    if sp > 0 and sy > 0:
        r = float(np.corrcoef(p, y)[0, 1])
        out["r2_scaled"] = float(min(max(r * r, 0.0), 1.0)) if np.isfinite(r) else 0.0

    # id-fit: a, b from the in-distribution set, applied at this gap.
    if id_ds is not None:
        p_id = predict_pn(formula, id_ds["X"], pn_eval)
        if p_id is not None and np.std(p_id) > 0:
            A = np.column_stack([p_id, np.ones_like(p_id)])
            try:
                coef, *_ = np.linalg.lstsq(A, np.asarray(id_ds["y"], np.float64), rcond=None)
                fitted = coef[0] * p + coef[1]
                if np.isfinite(fitted).all():
                    out["r2_id_fit"] = _ood_r2_score(y, fitted)
            except np.linalg.LinAlgError:
                pass
    return out


# -- Data assembly -------------------------------------------------------------
def collect(n: int, seed: int, gap: int, gen: dict, normalise: bool,
            require_var_fit: bool, feynman_csv: str) -> pd.DataFrame:
    """One row per sampled formula: its nearest GT, the distance, and the three R^2s."""
    ood_path = os.path.join(SCRIPT_DIR, f"datasets/feynman_ood_g{gap}.pkl.gz")
    if not os.path.exists(ood_path):
        raise SystemExit(f"OOD dataset not found: {ood_path}  (generate it with gen_ood_data.py)")
    with gzip.open(ood_path, "rb") as fh:
        ood = pickle.load(fh)["datasets"]

    # In-distribution reference for the id-fit variant.  gap 0 is the same synthetic
    # generator at zero shift (see the OOD-vs-gap convention), so it is the matched
    # in-distribution counterpart of every gap file.
    id_path = os.path.join(SCRIPT_DIR, "datasets/feynman_ood_g0.pkl.gz")
    id_sets = None
    if os.path.exists(id_path):
        with gzip.open(id_path, "rb") as fh:
            id_sets = pickle.load(fh)["datasets"]
    else:
        print(f"[warn] {id_path} missing -- the id-fit R^2 column will be all zeros.")

    # Ground truth: canonical PN per Feynman equation, restricted to those that also
    # have an OOD set (the 99 non-black-box equations).
    pn_targets = load_feynman_pn_targets(feynman_csv)
    gt_skeletons = {}
    for ds in sorted(set(pn_targets) & set(ood)):
        sk = skeleton(pn_targets[ds])
        if sk is not None:
            gt_skeletons[ds] = (sk, int(np.asarray(ood[ds]["X"]).shape[1]))
    if not gt_skeletons:
        raise SystemExit("No Feynman ground-truth formulas could be converted to PN.")
    print(f"  {len(gt_skeletons)} ground-truth Feynman formulae in canonical PN "
          f"(of {len(pn_targets)} in the CSV, {len(ood)} in the OOD set).", flush=True)

    pn_eval = _ood_pn_evaluator()
    if pn_eval is None:
        raise SystemExit("The C PN evaluator is unavailable; OOD R^2 cannot be computed.")

    formulas = draw_formulas(n, seed, gen)

    t0 = time.time()
    records = []
    n_unmatched = 0
    for i, (fml, n_vars) in enumerate(formulas, 1):
        sk = skeleton(fml)
        if sk is None:
            continue  # unparseable draw (should not happen -- the grammar emits valid PN)
        match = nearest_gt(sk, gt_skeletons, normalise,
                           min_len=n_vars if require_var_fit else 0)
        if match is None:
            # --require-var-fit only: the draw uses more variables than ANY Feynman
            # dataset has columns, so no eligible ground truth exists.
            n_unmatched += 1
            continue
        ds, norm_d, raw_d = match
        r2 = score_row(fml, ood[ds], (id_sets or {}).get(ds), pn_eval)
        records.append({
            "formula": fml,
            "n_vars": n_vars,
            "fml_len": len(fml.split()),
            "dataset": ds,
            "dataset_n_vars": gt_skeletons[ds][1],
            "var_overflow": n_vars > gt_skeletons[ds][1],
            "edit_dist": norm_d,
            "edit_dist_raw": raw_d,
            **r2,
        })
        if i % 250 == 0:
            print(f"    {i}/{len(formulas)} scored ({time.time() - t0:.0f}s)", flush=True)
    print(f"  matched + scored {len(records)} formulae in {time.time() - t0:.1f}s", flush=True)
    if n_unmatched:
        print(f"  [--require-var-fit] {n_unmatched} draw(s) dropped: more variables than "
              f"any Feynman dataset has columns.")
    return pd.DataFrame(records)


# -- Plot ----------------------------------------------------------------------
_R2_LABEL = {
    "scaled": "OOD $R^2$ after best affine rescale\n(= corr$(f, y)^2$; units quotiented out)",
    "id-fit": "OOD $R^2$, affine fitted in-distribution\n(gap 0 fit, applied at this gap)",
    "direct": "OOD $R^2$, no rescaling\n(formula evaluated as-is)",
}
_R2_COL = {"scaled": "r2_scaled", "id-fit": "r2_id_fit", "direct": "r2_direct"}


def make_plot(df: pd.DataFrame, gap: int, r2_kind: str, distance: str, r2_thr: float,
              out_path: str, show: bool):
    col = _R2_COL[r2_kind]
    xcol = "edit_dist" if distance == "normalized" else "edit_dist_raw"
    x, y = df[xcol].to_numpy(float), df[col].to_numpy(float)
    rng = np.random.default_rng(0)

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(f"Sampler draws: structural proximity to the nearest Feynman equation "
                 f"vs OOD $R^2$   (gap = {gap}, n = {len(df)})",
                 fontsize=15, fontweight="bold", y=0.98)

    # -- Left: the scatter --
    ax.scatter(x, y, s=26, color=_SCATTER_COLOR, alpha=0.45,
               edgecolor="white", linewidth=0.3, zorder=2)

    # Mean R^2 per edit-distance bin.  Equal-count bins (deciles of the observed
    # distance) rather than equal-width, so every marker rests on the same n.
    n_bins = min(10, max(2, len(df) // 40))
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    ctr, mean_r2, cnt = [], [], []
    for lo, hi, last in zip(edges[:-1], edges[1:], range(len(edges) - 1)):
        m = (x >= lo) & (x <= hi) if last == len(edges) - 2 else (x >= lo) & (x < hi)
        if m.sum() >= 5:
            ctr.append(x[m].mean())
            mean_r2.append(y[m].mean())
            cnt.append(int(m.sum()))
    if ctr:
        ax.plot(ctr, mean_r2, color="black", lw=2, marker="s", markersize=8,
                markeredgecolor="white", zorder=5,
                label=f"mean $R^2$ per distance decile (n$\\approx${int(np.median(cnt))} each)")
        # Stagger the value labels: the high-distance deciles bunch together on x and
        # their means bunch together on y, so a single offset would overlap them.
        for j, (cx, cy) in enumerate(zip(ctr, mean_r2)):
            ax.annotate(f"{cy:.2f}", (cx, cy), textcoords="offset points",
                        xytext=(0, 10 if j % 2 == 0 else 22), ha="center",
                        fontsize=8, color="0.25")

    if len(df) > 1:
        pear = float(np.corrcoef(x, y)[0, 1])
        spear = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        ax.text(0.98, 0.95, f"Pearson  r = {pear:+.2f}\nSpearman $\\rho$ = {spear:+.2f}",
                transform=ax.transAxes, fontsize=10, color="0.2", ha="right", va="top",
                bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    ax.axhline(r2_thr, color="0.6", ls=":", lw=1, zorder=1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(("Normalized" if distance == "normalized" else "Raw token")
                  + " edit distance to the nearest Feynman formula"
                  + ("  (0 = identical structure, 1 = maximally different)"
                     if distance == "normalized" else "  (tokens)"), fontsize=11)
    ax.set_ylabel(_R2_LABEL[r2_kind], fontsize=11)
    ax.set_title("Per sampled formula", fontsize=12)
    ax.grid(ls="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    if ctr:
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.83), fontsize=9, framealpha=0.9)

    # The y=0 pile-up is the story on the right-hand side of the axis, and overplotting
    # hides its size -- state it.
    n_zero = int((y <= 1e-9).sum())
    ax.text(0.01, 0.02, f"{n_zero}/{len(y)} draws score exactly 0",
            transform=ax.transAxes, fontsize=9, color="0.35", ha="left", va="bottom")

    # -- Right: the same relation read the other way round --
    hit = y >= r2_thr
    groups = [(f"$R^2\\geq${r2_thr}\n(tracks the target)", x[hit], _GOOD_COLOR),
              (f"$R^2<${r2_thr}\n(does not)",              x[~hit], _BAD_COLOR)]
    for i, (name, vals, colr) in enumerate(groups):
        if len(vals) == 0:
            continue
        axb.scatter(np.full(len(vals), i) + rng.uniform(-0.13, 0.13, len(vals)),
                    vals, s=14, color=colr, alpha=0.4, edgecolor="none", zorder=2)
        bp = axb.boxplot([vals], positions=[i], widths=0.5, patch_artist=True,
                         showfliers=False, medianprops=dict(color="black", lw=2), zorder=3)
        bp["boxes"][0].set(facecolor=colr, alpha=0.25)
        axb.text(i, 1.01, f"med {np.median(vals):.2f}\nn={len(vals)}", transform=
                 axb.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color=colr)
    axb.set_xticks([0, 1])
    axb.set_xticklabels([g[0] for g in groups], fontsize=9)
    axb.set_xlim(-0.6, 1.6)
    axb.set_ylabel(("Normalized" if distance == "normalized" else "Raw") + " edit distance",
                   fontsize=10)
    # pad clears the per-group med/n captions drawn just above the axes.
    axb.set_title("Edit distance by OOD outcome", fontsize=12, pad=30)
    axb.grid(axis="y", ls="--", alpha=0.4)
    axb.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, os.path.join(SCRIPT_DIR, out_path))
    if show:
        plt.show()
    plt.close(fig)


# -- Entry point ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="Number of formulae to sample. Default: 1000.")
    ap.add_argument("--seed", type=int, default=42, help="Generator RNG seed. Default: 42.")
    ap.add_argument("--gap", type=int, default=2,
                    help="OOD shift gap -- datasets/feynman_ood_g<gap>.pkl.gz. Default: 2.")
    ap.add_argument("--r2", choices=list(_R2_COL), default="scaled",
                    help="Which R^2 variant to plot (all three are always written to --csv). "
                         "Default: scaled -- see the module docstring on why the direct "
                         "metric is identically zero for unfitted sampler draws.")
    ap.add_argument("--distance", choices=["normalized", "raw"], default="normalized",
                    help="Edit-distance flavour used BOTH to pick the nearest GT and as the "
                         "x-axis. Default: normalized.")
    ap.add_argument("--r2-threshold", type=float, default=0.5, dest="r2_thr",
                    help="R^2 cut for the right-hand outcome split. Default: 0.5.")
    ap.add_argument("--require-var-fit", action="store_true",
                    help="Match each draw only against datasets with at least as many "
                         "columns as it has variables (no zero-padded inputs).")
    ap.add_argument("--drop-var-overflow", action="store_true",
                    help="Keep the unrestricted match but drop rows whose formula uses more "
                         "variables than the matched dataset has columns.")
    ap.add_argument("--feynman-csv", default="./datasets/feynman/FeynmanEquations.csv")
    # Generator knobs -- defaults mirror the live training config (see _GEN_DEFAULTS).
    ap.add_argument("--max-fml-len", type=int, default=_GEN_DEFAULTS["fml_len_range"][1],
                    help="Symbolic-token cap on a draw. Default: 80.")
    ap.add_argument("--length-decay", type=float, default=_GEN_DEFAULTS["length_decay"],
                    help="Target-length-first size sampling. Default: 0.99. Pass 0 to disable.")
    ap.add_argument("--complexity-decay", type=float, default=_GEN_DEFAULTS["complexity_decay"],
                    help="Variable-count decay. Default: 1.0 (uniform). Pass 0 to disable.")
    ap.add_argument("--operator-weights", choices=["uniform", "tiered"],
                    default=_GEN_DEFAULTS["operator_weights"], help="Default: tiered.")
    ap.add_argument("--no-simplify", action="store_true",
                    help="Sample RAW (unsimplified) trees, as training currently does. Off by "
                         "default: the ground truth is canonical, so raw draws inflate the "
                         "edit distance for algebraic rather than structural reasons.")
    ap.add_argument("--output", default=None,
                    help="Output image. Default: plots/proximity_vs_ood_g<gap>.png")
    ap.add_argument("--csv", default=None, help="Also write the per-formula table here.")
    ap.add_argument("--from-csv", default=None,
                    help="Skip sampling/scoring and re-plot from a previously written --csv.")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    if args.output is None:
        args.output = f"plots/proximity_vs_ood_g{args.gap}.png"

    if args.from_csv:
        df = pd.read_csv(os.path.join(SCRIPT_DIR, args.from_csv))
        print(f"Re-plotting from {args.from_csv} ({len(df)} rows).")
    else:
        gen = dict(
            fml_len_range=(_GEN_DEFAULTS["fml_len_range"][0], args.max_fml_len),
            complexity_decay=args.complexity_decay or None,
            length_decay=args.length_decay or None,
            operator_weights=args.operator_weights,
            simplify_targets=not args.no_simplify,
            use_prefactors=False,
            restrict_consts=True,
        )
        print(f"Sampling {args.n} formulae (seed {args.seed}) and scoring them on "
              f"feynman_ood_g{args.gap} ...")
        print(f"  generator: {gen}", flush=True)
        df = collect(args.n, args.seed, args.gap, gen, args.distance == "normalized",
                     args.require_var_fit, args.feynman_csv)

    if df.empty:
        raise SystemExit("No formulae survived sampling/matching -- nothing to plot.")

    if args.csv:
        csv_path = os.path.join(SCRIPT_DIR, args.csv)
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[csv]  Saved -> {args.csv}")

    n_all = len(df)
    if args.drop_var_overflow:
        df = df[~df["var_overflow"].astype(bool)].reset_index(drop=True)
        print(f"  dropped {n_all - len(df)} var-overflow rows ({len(df)} left).")

    # -- Summary --
    xcol = "edit_dist" if args.distance == "normalized" else "edit_dist_raw"
    x = df[xcol].to_numpy(float)
    print(f"\n{'=' * 72}\nProximity to nearest Feynman formula vs OOD R^2  (gap {args.gap}, "
          f"n={len(df)})\n{'=' * 72}")
    print(f"  edit distance ({args.distance}): min {x.min():.3f}  median {np.median(x):.3f}  "
          f"mean {x.mean():.3f}  max {x.max():.3f}")
    print(f"  var overflow (formula vars > dataset cols): {int(df['var_overflow'].sum())}/{len(df)}")
    print(f"  nearest-GT equations hit: {df['dataset'].nunique()} distinct; most common "
          f"{', '.join(f'{d} ({c})' for d, c in df['dataset'].value_counts().head(3).items())}")
    for kind, col in _R2_COL.items():
        v = df[col].to_numpy(float)
        r = float(np.corrcoef(x, v)[0, 1]) if len(df) > 1 and v.std() > 0 else float("nan")
        rho = (float(pd.Series(x).corr(pd.Series(v), method="spearman"))
               if len(df) > 1 and v.std() > 0 else float("nan"))
        mark = " <- plotted" if kind == args.r2 else ""
        print(f"  R^2 [{kind:6s}]  mean {v.mean():.3f}  |  >0: {int((v > 1e-9).sum()):4d}  "
              f">=0.5: {int((v >= 0.5).sum()):4d}  >=0.9: {int((v >= 0.9).sum()):4d}  |  "
              f"corr(dist, R^2) r={r:+.3f} rho={rho:+.3f}{mark}")
    print("=" * 72)

    make_plot(df, args.gap, args.r2, args.distance, args.r2_thr, args.output, args.show)


if __name__ == "__main__":
    main()
