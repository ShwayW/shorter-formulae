#!/usr/bin/env python3
"""
feynman_target_table.py -- one row per SRBench/Feynman problem, joining what the
GENERATOR can reach to what the MODELS actually achieve out of distribution.

Columns (in this order; the first five are the table proper, the rest are the
supporting detail behind them):

  target                      PMLB dataset name (feynman_I_12_1, ...).
  min_edit_dist_to_draws      Over N formulae drawn from the online training generator,
                              the SMALLEST structural edit distance from any draw to
                              this target.  0 = the sampler reproduced the target's
                              structure exactly.  This is the per-target transpose of
                              proximity_vs_ood.py, which takes the min the other way
                              round (nearest target per draw).
  expressible_in_G            Can the grammar write this formula at all -- i.e. does its
                              canonical PN use only V + C + U + B tokens?  See below.
  <model>_ood_r2_k<gap>       Our model's OOD R^2 on this problem at gap k, meaned over
                              the evaluation seeds.
  e2e_ood_r2_k<gap>           Same for the end-to-end baseline.

  ... then, per headline column: the raw (unnormalised) distance and the actual closest
  draw; the sampler-budget check and the reason string behind expressible_in_G; the
  target's PN, source formula, variable count and PN length; and the seed std / best /
  count behind each OOD R^2 mean.

Universe
--------
The 99 Feynman problems that have BOTH a ground-truth formula convertible to canonical
PN (eval_mymodels.load_feynman_pn_targets) and a pre-generated OOD point cloud
(datasets/feynman_ood_g0.pkl.gz) -- the same 99-problem universe the OOD-vs-gap plots
report solve rates out of.

expressible_in_G vs within_sampler_budget
-----------------------------------------
Two different questions, both answered:
  * expressible_in_G      -- pure grammar membership: every token of the canonical PN is
                             in grammar.FlatGram (V + C + U + B).  Note that the
                             mantissa-free grammar composes constants out of {0,1,2,3,pi}
                             ('/ 1 2' = 0.5), so any rational is expressible.
  * within_sampler_budget -- whether the ONLINE GENERATOR could actually draw it under
                             the configured limits: PN length <= the fml_len cap, binary
                             ops <= grammar._DIST_MAX_OPS, unary ops <=
                             grammar._MAX_UNARY_OPS, variables <=
                             len(V).  A formula can be expressible but unreachable.
Both are computed per row; expressible_in_G is the requested column.

OOD R^2 source
--------------
Read-only from the shared OOD R^2 cache results/ood_raw_noise<tau>.csv (the per-
(algorithm, dataset, seed, gap) table that plot_ood_vs_gap.py builds and every OOD
figure reuses).  This script NEVER writes it -- rebuilding that cache loses the model
rows.  Cache algorithm names are mapped to display labels with the same
plot_ood_vs_gap._remap_scale_labels the figures use, so "89M" here is the same model as
"89M" there.  A problem with no cache row at the requested gap (feynman_I_26_2 has no
OOD set beyond gap 2) gets an EMPTY cell, not a zero.

Usage
-----
    python feynman_target_table.py                       # 1000 draws, seed 42, gap 128
    python feynman_target_table.py --n 5000 --gap 64
    python feynman_target_table.py --model 145M --out results/table_145M.csv
"""
import argparse
import gzip
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import ood_transform                                                     # noqa: E402
from grammar import B, U, V, FlatGram, _DIST_MAX_OPS, _MAX_UNARY_OPS    # noqa: E402
from eval_mymodels import load_feynman_targets, load_feynman_pn_targets        # noqa: E402
from plot_ood_vs_gap import _remap_scale_labels                          # noqa: E402
from plot_structural_vs_ood import _levenshtein                          # noqa: E402
from proximity_vs_ood import _GEN_DEFAULTS, draw_formulas, skeleton      # noqa: E402

_FLATGRAM = set(FlatGram)
_B, _U, _V = set(B), set(U), set(V)
_MAX_UNARY = _MAX_UNARY_OPS   # the generator's cap on unary ops per draw (grammar.py)


# -- Grammar checks ------------------------------------------------------------
def grammar_check(pn: str, max_fml_len: int) -> dict:
    """Expressibility + sampler-reachability of one ground-truth PN formula."""
    toks = str(pn).split()
    bad = sorted({t for t in toks if t not in _FLATGRAM})
    n_bin = sum(1 for t in toks if t in _B)
    n_un = sum(1 for t in toks if t in _U)
    n_vars = len({t for t in toks if t in _V})

    if not toks:
        return dict(expressible_in_G=False, within_sampler_budget=False,
                    expressible_note="no PN conversion", pn_len=0, n_vars=0,
                    n_binary_ops=0, n_unary_ops=0)

    # A token outside V+C+U+B means the converter had to fall back to a literal the
    # grammar cannot write (a non-rational float) or emitted an ERR_ marker.
    expressible = not bad
    reasons = []
    if bad:
        reasons.append("non-grammar tokens: " + ", ".join(bad[:4]))
    if len(toks) > max_fml_len:
        reasons.append(f"PN length {len(toks)} > cap {max_fml_len}")
    if n_bin > _DIST_MAX_OPS:
        reasons.append(f"{n_bin} binary ops > _DIST_MAX_OPS {_DIST_MAX_OPS}")
    if n_un > _MAX_UNARY:
        reasons.append(f"{n_un} unary ops > cap {_MAX_UNARY}")
    if n_vars > len(_V):
        reasons.append(f"{n_vars} variables > len(V) {len(_V)}")
    return dict(
        expressible_in_G=expressible,
        within_sampler_budget=not reasons,
        expressible_note="; ".join(reasons) if reasons else "ok",
        pn_len=len(toks), n_vars=n_vars, n_binary_ops=n_bin, n_unary_ops=n_un,
    )


# -- Nearest training draw -----------------------------------------------------
def min_distance_to_draws(gt_pn: dict, draws: list, verbose: bool = True) -> dict:
    """{dataset: (norm_dist, raw_dist, closest_draw_formula)} over all draws.

    Distances are the same constant-agnostic structural Levenshtein used by
    proximity_vs_ood.py / plot_structural_vs_ood.py: numeric leaves collapse to "C" and
    commutative operands are sorted, so the number measures structure, not constants.
    The argmin is taken on the NORMALISED distance (raw distance would call every long
    target far from every short draw); ties keep the first draw in sample order.
    """
    draw_sk = []
    for fml, _n_vars in draws:
        sk = skeleton(fml)
        if sk is not None:
            draw_sk.append((fml, sk))

    out, t0 = {}, time.time()
    for i, (ds, pn) in enumerate(sorted(gt_pn.items()), 1):
        gt_sk = skeleton(pn)
        if gt_sk is None:
            out[ds] = (float("nan"), -1, "")
            continue
        best = None
        for fml, sk in draw_sk:
            raw = _levenshtein(sk, gt_sk)
            norm = raw / max(len(sk), len(gt_sk), 1)
            if best is None or norm < best[0]:
                best = (norm, raw, fml)
        out[ds] = best
        if verbose and i % 25 == 0:
            print(f"    {i}/{len(gt_pn)} targets matched ({time.time() - t0:.0f}s)", flush=True)
    if verbose:
        print(f"  min-distance over {len(draw_sk)} draws computed in "
              f"{time.time() - t0:.1f}s", flush=True)
    return out


# -- OOD R^2 cache -------------------------------------------------------------
def load_ood_cache(results_root: str, noise: float, gap: int, labels: list) -> dict:
    """{display_label: {dataset: (mean, std, best, n_seeds)}} at `gap`.

    Read-only.  Raises SystemExit when the cache or a requested label is absent rather
    than silently producing an empty column -- an empty OOD column would read as "the
    model failed everywhere" instead of "the data was not loaded".
    """
    path = os.path.join(results_root, ood_transform.srbench_cache_name(noise))
    if not os.path.exists(path):
        raise SystemExit(
            f"OOD R^2 cache not found: {path}\n"
            f"Build it with:  python plot_ood_vs_gap.py --noise {noise:g}")
    full = pd.read_csv(path)
    df = full[full["gap"] == gap]
    if df.empty:
        gaps = ", ".join(str(int(g)) for g in sorted(full["gap"].unique()))
        raise SystemExit(f"{path} has no rows at gap {gap} (available: {gaps}).")

    # Cache rows carry the raw eval_mymodels labels (m145 / m89 pre-rename,
    # 145M_40 / 89M_40 after); map them to
    # the parameter-count names the figures use so --model 89M means the same model here.
    remap = _remap_scale_labels(sorted(df["algorithm"].unique()))
    df = df.assign(label=df["algorithm"].map(remap))

    available = sorted(df["label"].unique())
    out = {}
    for lbl in labels:
        sub = df[df["label"] == lbl]
        if sub.empty:
            raise SystemExit(f"No rows for '{lbl}' at gap {gap} in {path}.\n"
                             f"Available labels: {', '.join(available)}")
        g = sub.groupby("dataset")["ood_r2"]
        out[lbl] = {ds: (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                         float(v.max()), int(len(v)))
                    for ds, v in g}
        print(f"  [{lbl:8s}] {len(out[lbl])} datasets, "
              f"{int(sub.groupby('dataset').size().median())} seeds each  "
              f"(from {os.path.basename(path)})", flush=True)
    return out


# -- Table ---------------------------------------------------------------------
def build_table(args) -> pd.DataFrame:
    # Universe: targets that have BOTH a canonical PN and an OOD point cloud.
    pn_targets = load_feynman_pn_targets(args.feynman_csv)
    src_formulas = load_feynman_targets(args.feynman_csv)
    base_path = os.path.join(SCRIPT_DIR, ood_transform.feynman_ood_path(0))
    if not os.path.exists(base_path):
        raise SystemExit(f"Baseline OOD set not found: {base_path}")
    with gzip.open(base_path, "rb") as fh:
        base_datasets = set(pickle.load(fh)["datasets"])
    universe = sorted(set(pn_targets) & base_datasets)
    print(f"  {len(universe)} Feynman targets "
          f"({len(pn_targets)} with a canonical PN, {len(base_datasets)} with OOD data).",
          flush=True)

    gen = dict(_GEN_DEFAULTS, fml_len_range=(1, args.max_fml_len),
               operator_weights=args.operator_weights,
               simplify_targets=not args.no_simplify)
    print(f"Sampling {args.n} training draws (seed {args.seed}) ...")
    print(f"  generator: {gen}", flush=True)
    draws = draw_formulas(args.n, args.seed, gen)

    gt_pn = {ds: pn_targets[ds] for ds in universe}
    dists = min_distance_to_draws(gt_pn, draws)

    print(f"Loading OOD R^2 at gap {args.gap} (noise {args.noise:g}) ...", flush=True)
    ood = load_ood_cache(os.path.join(SCRIPT_DIR, args.results_root),
                         args.noise, args.gap, [args.model, args.e2e_label])

    mcol = f"{args.model}_ood_r2_k{args.gap}"
    ecol = f"{args.e2e_label}_ood_r2_k{args.gap}"
    rows = []
    for ds in universe:
        norm_d, raw_d, closest = dists[ds]
        chk = grammar_check(pn_targets[ds], args.max_fml_len)
        row = {
            "target": ds,
            "min_edit_dist_to_draws": norm_d,
            "expressible_in_G": chk["expressible_in_G"],
            mcol: np.nan,
            ecol: np.nan,
            # -- supporting detail --
            "min_edit_dist_to_draws_raw": raw_d,
            "closest_draw": closest,
            "within_sampler_budget": chk["within_sampler_budget"],
            "expressible_note": chk["expressible_note"],
            "target_pn": pn_targets[ds],
            "target_formula": src_formulas.get(ds, ""),
            "n_vars": chk["n_vars"],
            "pn_len": chk["pn_len"],
        }
        for lbl, col in ((args.model, mcol), (args.e2e_label, ecol)):
            stats = ood[lbl].get(ds)
            if stats is None:
                # No OOD point cloud at this gap -> no cache row.  Left empty on
                # purpose: a 0 here would be indistinguishable from a real failure.
                row[f"{col}_std"] = np.nan
                row[f"{col}_best"] = np.nan
                row[f"{col}_n_seeds"] = 0
                continue
            mean, std, best, n = stats
            row[col] = mean
            row[f"{col}_std"] = std
            row[f"{col}_best"] = best
            row[f"{col}_n_seeds"] = n
        rows.append(row)

    df = pd.DataFrame(rows)
    # Requested five first, everything else after, in the order built above.
    head = ["target", "min_edit_dist_to_draws", "expressible_in_G", mcol, ecol]
    return df[head + [c for c in df.columns if c not in head]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000,
                    help="Training draws to measure the min edit distance over. Default: 1000.")
    ap.add_argument("--seed", type=int, default=42, help="Generator RNG seed. Default: 42.")
    ap.add_argument("--gap", type=int, default=128, help="OOD gap k. Default: 128.")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Target-noise level of the OOD cache to read. Default: 0.")
    ap.add_argument("--model", default="89M",
                    help="Display label of our model in the OOD cache. Default: 89M.")
    ap.add_argument("--e2e-label", default="e2e", help="Baseline label. Default: e2e.")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--feynman-csv", default="./datasets/feynman/FeynmanEquations.csv")
    ap.add_argument("--max-fml-len", type=int, default=_GEN_DEFAULTS["fml_len_range"][1],
                    help="Generator symbolic-token cap, also the length bound used by "
                         "within_sampler_budget. Default: 80.")
    ap.add_argument("--operator-weights", choices=["uniform", "tiered"],
                    default=_GEN_DEFAULTS["operator_weights"], help="Default: tiered.")
    ap.add_argument("--no-simplify", action="store_true",
                    help="Draw RAW (unsimplified) trees; see proximity_vs_ood.py.")
    ap.add_argument("--out", default=None,
                    help="Output CSV. Default: results/feynman_target_table_k<gap>.csv")
    args = ap.parse_args()

    if args.out is None:
        args.out = f"results/feynman_target_table_k{args.gap}.csv"

    df = build_table(args)

    out_path = os.path.join(SCRIPT_DIR, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)

    # -- Summary --
    mcol = f"{args.model}_ood_r2_k{args.gap}"
    ecol = f"{args.e2e_label}_ood_r2_k{args.gap}"
    d = df["min_edit_dist_to_draws"]
    print(f"\n{'=' * 72}\n{len(df)} Feynman targets  |  {args.n} draws  |  gap {args.gap}, "
          f"noise {args.noise:g}\n{'=' * 72}")
    print(f"  min edit distance to a draw: min {d.min():.3f}  median {d.median():.3f}  "
          f"max {d.max():.3f}  |  exactly 0 (structure reproduced): "
          f"{int((d <= 1e-12).sum())}")
    print(f"  expressible in G:      {int(df['expressible_in_G'].sum())}/{len(df)}")
    print(f"  within sampler budget: {int(df['within_sampler_budget'].sum())}/{len(df)}")
    for col in (mcol, ecol):
        v = df[col]
        miss = int(v.isna().sum())
        print(f"  {col}: mean {v.mean():.3f}  |  >=0.99: {int((v >= 0.99).sum())}  "
              f">=0.5: {int((v >= 0.5).sum())}  |  no data: {miss}"
              + (f" ({', '.join(df.loc[v.isna(), 'target'])})" if 0 < miss <= 3 else ""))
    both = df[[mcol, ecol]].dropna()
    if len(both) > 1:
        print(f"  corr({mcol}, {ecol}) = {both.corr().iloc[0, 1]:+.3f}")
        sub = df.dropna(subset=[mcol])
        if sub["min_edit_dist_to_draws"].std() > 0 and sub[mcol].std() > 0:
            print(f"  corr(min edit distance, {mcol}) = "
                  f"{sub['min_edit_dist_to_draws'].corr(sub[mcol]):+.3f}")
    print("=" * 72)
    print(f"[csv]  Saved -> {args.out}  ({len(df)} rows x {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
