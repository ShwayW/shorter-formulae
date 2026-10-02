#!/usr/bin/env python3
r"""
make_ablation_tables.py -- LaTeX tables for the 2x2 data-generation ablation.

The ablation's curve figures (plot_ablation_ood_vs_gap.py) sit next to sets 1 and 3 in
the paper and read as the same figure again.  These tables carry the same measurements
in the form the ablation actually has -- a 2x2 FACTORIAL -- and add the two quantities a
curve cannot show: the main effect of each factor and their interaction.

TABLE 1 (factorial).  Rows are the four arms as the 2x2 they are (prefactors x
simplification); column groups are benchmark x target noise; each group reports

  k0   the solve rate at gap 0 (in-domain recovery), mean over the 10 seeds
  ret  retention = solve rate at gap 128 / solve rate at gap 0

Two numbers per curve instead of nine: everything between the two gaps is monotone
interpolation (ood_transform.monotone_gap enforces it), so the pair fixes the shape.
Under them sits the marginal block:

  D pref   mean(prefactors on) - mean(prefactors off)
  D simp   mean(simplify on)   - mean(simplify off)
  inter    (on,on - on,off) - (off,on - off,off)

CAVEAT, PRINTED IN THE CAPTION: where |inter| exceeds either main effect, the marginal
means are not interpretable alone -- a reader quoting "prefactors cost 0.23" on
LSR-Transform would be misreading a number that is mostly interaction.  The three rows
are therefore always emitted together, never the main effects by themselves.

TABLE 2 (cost).  Per-formula inference time and formula complexity for the same arms
plus the 145M reference, at every noise level -- the content of the set-2-style dot plot
(plot_ablation_complexity.py), which is five rows of two metrics and reads better as a
table.

Both tables are computed from the SAME caches and the SAME thresholding helpers the
figures use, so the numbers cannot drift from them:
  * solve rates -- results/ood_ablation_t120M[_llmsr]_noise<tau>.csv, thresholded by
    plot_ood_vs_gap.accuracy_from_raw / plot_ablation_ood_vs_gap._llmsr_accuracy_df
  * time and complexity -- plot_ablation_complexity._tables_for_noise, i.e.
    compare_srbench.load_transformer_results

Output is booktabs (already loaded by papers/iclr2027/iclr2027.tex); no other package is
required -- no siunitx, no multirow.

Usage:
    python make_ablation_tables.py                        # both tables to stdout
    python make_ablation_tables.py --tables factorial
    python make_ablation_tables.py --noises 0 0.01 0.1    # more column groups
    python make_ablation_tables.py --std inline           # +/- seed std inside k0
    python make_ablation_tables.py --output tables.tex
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from plot_ood_vs_gap import accuracy_from_raw
from plot_ablation_ood_vs_gap import _llmsr_accuracy_df, DISPLAY
from plot_ablation_complexity import _tables_for_noise

# (prefactors, simplify) -> the raw cache label of that cell.  Row order below is the
# reading order of the 2x2: both on, then one factor off at a time, then both off.
CELLS = {
    (1, 1): "prefac_on_t120M",
    (1, 0): "prefac_on_nosimp_t120M",
    (0, 1): "prefac_off_t120M",
    (0, 0): "simp_off_t120M",
}
ROW_ORDER = [(1, 1), (1, 0), (0, 1), (0, 0)]
REFERENCE = "145M"           # the yardstick row in the cost table, not a cell of the 2x2

BENCH_NAME = {"srbench": "Feynman", "llmsr": "LSR-Transform"}
DEFAULT_NOISES = [0.0, 0.1]
DEFAULT_BENCHES = ["srbench", "llmsr"]


def _cache(bench, tau):
    tag = f"{float(tau):g}"
    name = (f"ood_ablation_t120M_llmsr_noise{tag}.csv" if bench == "llmsr"
            else f"ood_ablation_t120M_noise{tag}.csv")
    return os.path.join(SCRIPT_DIR, "results", name)


def solve_rates(bench, tau, r2_thr, gaps):
    """{label: {gap: (mean over seeds, std over seeds)}} for the four arms."""
    path = _cache(bench, tau)
    if not os.path.exists(path):
        sys.exit(f"[error] cache not found: {path}\n"
                 "        build it with augment_base_ood_caches.py against "
                 "results/ood_t120M_root (see plot_ablation_ood_vs_gap.py's docstring).")
    df = pd.read_csv(path)
    df = df[df["algorithm"].isin(CELLS.values()) & df["gap"].isin(gaps)]
    if df.empty:
        sys.exit(f"[error] {path} holds no rows for the ablation arms at gaps {gaps}")
    # The figures' own thresholders, so a table cell and a plotted point are the same
    # number: both restrict to the common problem set and apply the monotone-gap rule.
    acc = _llmsr_accuracy_df(df, r2_thr) if bench == "llmsr" else accuracy_from_raw(df, r2_thr)
    out = {}
    for (alg, gap), grp in acc.groupby(["algorithm", "gap"]):
        rates = grp["solve_rate"].dropna().to_numpy()
        if rates.size:
            out.setdefault(alg, {})[int(gap)] = (float(rates.mean()), float(rates.std()))
    return out


def factorial_block(rates, k0, kr):
    """Per-cell (k0 rate, k0 std, retention) plus the two main effects and interaction."""
    cells, out = {}, {}
    for key in ROW_ORDER:
        label = CELLS[key]
        m0, s0 = rates.get(label, {}).get(k0, (float("nan"), float("nan")))
        m1, _ = rates.get(label, {}).get(kr, (float("nan"), float("nan")))
        # Retention is undefined, not zero, when nothing was solved in-domain.
        ret = (m1 / m0) if (np.isfinite(m0) and m0 > 0) else float("nan")
        cells[key] = m0
        out[key] = (m0, s0, ret)
    eff_pref = (cells[(1, 1)] + cells[(1, 0)]) / 2 - (cells[(0, 1)] + cells[(0, 0)]) / 2
    eff_simp = (cells[(1, 1)] + cells[(0, 1)]) / 2 - (cells[(1, 0)] + cells[(0, 0)]) / 2
    inter = (cells[(1, 1)] - cells[(1, 0)]) - (cells[(0, 1)] - cells[(0, 0)])
    return out, eff_pref, eff_simp, inter


def _num(v, nd=3):
    return "--" if not np.isfinite(v) else f"{v:.{nd}f}"


def _signed(v, nd=3):
    return "--" if not np.isfinite(v) else f"${v:+.{nd}f}$"


def table_factorial(args):
    groups = [(b, t) for b in args.benches for t in args.noises]
    data, dominated = {}, []
    for b, t in groups:
        rates = solve_rates(b, t, args.r2_thr, [args.gap0, args.ret_gap])
        block = factorial_block(rates, args.gap0, args.ret_gap)
        data[(b, t)] = block
        _, ep, es, it = block
        if abs(it) > max(abs(ep), abs(es)):
            dominated.append(f"{BENCH_NAME[b]} $\\tau={t:g}$")

    max_std = max((c[1] for blk in data.values() for c in blk[0].values()
                   if np.isfinite(c[1])), default=float("nan"))

    L = []
    L.append("% " + "-" * 74)
    L.append("% 2x2 data-generation ablation: solve rate and retention, with the")
    L.append("% factorial main effects and interaction.  Regenerate with:")
    L.append("%   python make_ablation_tables.py --tables factorial \\")
    L.append("%       --benches " + " ".join(args.benches)
             + " --noises " + " ".join(f"{t:g}" for t in args.noises))
    L.append("% " + "-" * 74)
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    if args.font != "normal":
        L.append("\\" + args.font)
    L.append(r"\caption{%")
    L.append(r"  Data-generation ablation on the $2\times2$ of affine subtree "
             r"transformations (\emph{affine}) and target standardization "
             r"(\emph{standard}), the two generation choices of "
             r"Section~\ref{subsec:dataGeneration}.")
    L.append(r"  \emph{affine}~=~on means the generator emits the affine "
             r"transformations that Section~\ref{sec:discussion} removes; "
             r"\emph{standard}~=~on means targets are standardized.  (In the code "
             r"these are \texttt{use\_prefactors} and \texttt{simplify\_targets}.)")
    L.append(r"  \textbf{k0} is the fraction of equations recovered at "
             f"extrapolation gap $k={args.gap0}$ "
             r"(mean over 10 seeds); \textbf{ret} is retention, "
             f"the solve rate at $k={args.ret_gap}$ divided by that at $k={args.gap0}$.")
    if np.isfinite(max_std):
        L.append(r"  The seed standard deviation of every k0 entry is at most "
                 f"${max_std:.3f}$."
                 if args.std != "inline" else
                 r"  k0 entries carry $\pm$ one standard deviation over the 10 seeds.")
    L.append(r"  The lower block gives the factorial decomposition of k0: each factor's "
             r"main effect and their interaction.")
    if dominated:
        L.append(r"  \emph{The interaction exceeds both main effects for "
                 + ", ".join(dominated) +
                 r"; for those columns the main effects are not interpretable in "
                 r"isolation and must be read together with the interaction term.}")
    L.append(r"}")
    L.append(r"\label{tab:ablation-factorial}")

    ncols = 2 + 2 * len(groups)
    L.append(r"\begin{tabular}{ll" + "rr" * len(groups) + "}")
    L.append(r"\toprule")
    head = ["affine", "standard"]
    for b, t in groups:
        head.append(r"\multicolumn{2}{c}{" + f"{BENCH_NAME[b]}, $\\tau={t:g}$" + "}")
    L.append(" & ".join(head) + r" \\")
    L.append(" ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}"
                      for i in range(len(groups))))
    L.append(" & ".join(["", ""] + ["k0", "ret"] * len(groups)) + r" \\")
    L.append(r"\midrule")

    # Bold the best k0 in each column group -- best cell, not best row overall.
    best = {}
    for g in groups:
        cellmap = data[g][0]
        vals = {k: cellmap[k][0] for k in ROW_ORDER if np.isfinite(cellmap[k][0])}
        best[g] = max(vals, key=vals.get) if vals else None

    for key in ROW_ORDER:
        p, s = key
        cells = ["on" if p else "off", "on" if s else "off"]
        for g in groups:
            m0, s0, ret = data[g][0][key]
            txt = _num(m0)
            if args.std == "inline" and np.isfinite(s0):
                txt += f"\\,$\\pm$\\,{s0:.3f}"
            if not args.no_bold and best[g] == key and np.isfinite(m0):
                txt = r"\textbf{" + txt + "}"
            cells += [txt, _num(ret, 2)]
        L.append(" & ".join(cells) + r" \\")

    L.append(r"\midrule")
    for name, idx in ((r"$\Delta$ affine", 1), (r"$\Delta$ standard", 2),
                      ("interaction", 3)):
        cells = [r"\multicolumn{2}{l}{" + name + "}"]
        for g in groups:
            cells += [_signed(data[g][idx]), ""]
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L), ncols


def table_cost(args):
    """Per-formula time and complexity, one column group per noise level."""
    noises = args.cost_noises
    series, universe, gt_note = {}, None, None
    for t in noises:
        tab, datasets = _tables_for_noise(t, args.r2_thr, verbose=False)
        if tab is None:
            print(f"% [warn] no time/complexity results at noise {t:g}", file=sys.stderr)
            continue
        universe = datasets if universe is None else (universe & datasets)
        for _, r in tab.iterrows():
            series.setdefault(r["method"], {})[t] = (r["time"], r["complexity"],
                                                     int(r["n_solved"]))
    if not series:
        sys.exit("[error] no time/complexity results found")
    drawn = [t for t in noises if any(t in v for v in series.values())]

    # Same row order as the figure: slowest first, by the largest time over the levels.
    def _rep(m):
        vals = [series[m][t][0] for t in drawn
                if t in series[m] and np.isfinite(series[m][t][0])]
        return max(vals) if vals else 0.0
    arms = [DISPLAY.get(CELLS[k], CELLS[k]) for k in ROW_ORDER]
    rows = sorted([m for m in series if m in arms], key=_rep, reverse=True)
    if REFERENCE in series:
        rows.append(REFERENCE)              # the yardstick sits last, after a rule

    L = []
    L.append("% " + "-" * 74)
    L.append("% Ablation cost: per-formula inference time and formula complexity.")
    L.append("%   python make_ablation_tables.py --tables cost")
    L.append("% " + "-" * 74)
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    if args.font != "normal":
        L.append("\\" + args.font)
    L.append(r"\caption{%")
    L.append(r"  Cost of the ablation arms: mean inference time per formula (s) and mean "
             r"complexity of the recovered formulae (sympy node count, over the equations "
             r"a seed solved), at each target noise $\tau$.")
    L.append(f"  Computed on the {len(universe)} equations shared by every arm and noise "
             f"level; $n$ is the number solved at $R^2 \\geq {args.r2_thr}$.")
    L.append(r"  The 145M reference is listed for scale and is not a cell of the "
             r"$2\times2$.")
    L.append(r"}")
    L.append(r"\label{tab:ablation-cost}")
    L.append(r"\begin{tabular}{l" + "rrr" * len(drawn) + "}")
    L.append(r"\toprule")
    head = ["arm"] + [r"\multicolumn{3}{c}{" + f"$\\tau={t:g}$" + "}" for t in drawn]
    L.append(" & ".join(head) + r" \\")
    L.append(" ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                      for i in range(len(drawn))))
    L.append(" & ".join(["arm"] + ["time", "cplx", "$n$"] * len(drawn))
             .replace("arm &", " &", 1) + r" \\")
    L.append(r"\midrule")
    for m in rows:
        if m == REFERENCE:
            L.append(r"\midrule")
        cells = [m.replace("=", "$=$")]
        for t in drawn:
            if t in series.get(m, {}):
                tm, cx, n = series[m][t]
                cells += [_num(tm, 1), _num(cx, 1), str(n)]
            else:
                cells += ["--", "--", "--"]
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L), 1 + 3 * len(drawn)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", choices=["all", "factorial", "cost"], default="all")
    ap.add_argument("--benches", nargs="+", choices=["srbench", "llmsr"],
                    default=DEFAULT_BENCHES,
                    help="benchmarks to column-group the factorial table by")
    ap.add_argument("--noises", nargs="+", type=float, default=DEFAULT_NOISES,
                    metavar="TAU",
                    help="noise levels in the factorial table (default 0 0.1: the clean "
                         "run and the level where the factors' roles invert)")
    ap.add_argument("--cost-noises", nargs="+", type=float,
                    default=[0.0, 0.001, 0.01, 0.1], metavar="TAU",
                    help="noise levels in the cost table (default: all four)")
    ap.add_argument("--gap0", type=int, default=0, help="the in-domain gap (default 0)")
    ap.add_argument("--ret-gap", type=int, default=128,
                    help="the far gap retention is measured at (default 128)")
    ap.add_argument("--r2-thr", type=float, default=0.99)
    ap.add_argument("--std", choices=["none", "inline"], default="none",
                    help="'inline' prints +/- the seed std inside every k0 cell; the "
                         "default states the maximum in the caption instead, which keeps "
                         "the table inside a two-column textwidth")
    ap.add_argument("--font", choices=["normal", "small", "footnotesize"],
                    default="small",
                    help="size command emitted inside the table environment.  Both "
                         "tables are wide; 'small' is the default because the defaults "
                         "here already run to 10 and 13 columns.")
    ap.add_argument("--no-bold", action="store_true",
                    help="do not bold the best k0 of each column group")
    ap.add_argument("--output", default=None,
                    help="write the LaTeX here instead of stdout")
    args = ap.parse_args()

    parts = []
    if args.tables in ("all", "factorial"):
        tex, ncols = table_factorial(args)
        parts.append(tex)
        if ncols > 10:
            print(f"% [note] the factorial table has {ncols} columns -- consider fewer "
                  "--noises or --benches to keep it inside the textwidth", file=sys.stderr)
    if args.tables in ("all", "cost"):
        tex, ncols = table_cost(args)
        parts.append(tex)
        if ncols > 10:
            print(f"% [note] the cost table has {ncols} columns -- at ICLR's 5.5 in "
                  "textwidth consider --cost-noises 0 0.1, or wrap it in "
                  "\\resizebox{\\textwidth}{!}{...}", file=sys.stderr)
    out = "\n\n".join(parts) + "\n"

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(out)
        print(f"[tables] wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
