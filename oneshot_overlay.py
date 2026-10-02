"""oneshot_overlay.py -- draw the one-shot LLM baselines (a SINGLE LLM call per problem,
eval_oneshot.py) on the SRBench (sets 1/2) and LLM-SRBench (sets 3/4) figures, straight
from the score_oneshot_ood.py caches
(results/oneshot_<backbone>_ood_raw_<split>_noise<tau>.csv).

This is llmsr_overlay.py's sibling: same cache schema, same live time/complexity
computation from the raw artifacts.  Three differences:

  * THREE backbones, all complete at all four noises and both splits -- unlike the LLMSR
    side, where only Llama ran past noise 0 (which is why llmsr_overlay draws Llama
    alone).  Gemini keeps the blue and Llama the amber that plot_ood_common.py /
    make_oneshot_plots.sh already give the one-shot arms; Llama-3.3-70B takes a darker
    amber to stay in the Llama family.
  * The seed story is NO LONGER uniform, which is why the spread is computed per arm
    rather than assumed away.  Gemini and Llama-3.1-8B ran seed 0 only, so their curves
    are a single line with zero-length error bars, exactly as before.  Llama-3.3-70B ran
    the canonical 10 seeds, so it carries +/-1 seed-std error bars -- the same spread,
    computed the same way, as every multi-seed curve beside it (plot_ood_vs_gap's
    band_lo/band_hi).  Its time/complexity points likewise average each problem across
    seeds first, so their std is across PROBLEMS, matching the convention the dot plots
    state for our own models rather than mixing in seed variance.
  * Complexity is counted with compare_srbench.formula_complexity, not
    llmsr_complexity: a one-shot artifact stores a CONCRETE infix formula (the model
    wrote the constants, or BFGS fitted them at search time), not an LLMSR-style
    `def equation` program with symbolic params -- the same reason AI Feynman, PhyE2E
    and the standalone TPSR arm are counted that way.
  * The caches need a denominator repair (load_cache below).  score_oneshot_ood.py DROPS
    a problem whose reply carried no equation, or one sympy could not parse -- 7 of
    LLM-SRBench's 111 for Llama, 3 for Gemini, 5 of Feynman's 99 for Llama.  Scoring the
    solve rate over the surviving rows alone would divide by 104 where every other arm
    divides by 111, quietly inflating the one-shot curves by several points.  A reply we
    cannot parse is a problem the method did not solve, so load_cache restores those
    problems as UNSOLVED rows at every gap, from the artifact's own problem list.

Every curve is drawn DASHED, so the "one LLM call" family reads as a family against the
solid search-based curves (the convention plot_ood_common.py already uses).

One caveat these arms inherit from llmsr_overlay, unchanged: on the SRBench grid the
other curves are scored on the COMMON SUBSET of Feynman problems every method in the
figure attempted (96 at present, held down by AI Feynman), while an overlay is scored on
its own run's 99.  The 96 are a subset of the 99, so the arms are close but not exactly
like-for-like -- read a small difference on that figure with that in mind.  On
LLM-SRBench there is no such gap: every method attempts all 111 equations.
"""
import os

import numpy as np
import pandas as pd

from plot_markers import method_marker, HOLLOW, CURVE_MS, CURVE_LW
import ood_transform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LINESTYLE = "--"

# Display label -> attributes:
#   backbone : the cache's backbone tag (ood_transform.oneshot_backbone_cache_name)
#   src      : the 'algorithm' label inside that cache (mapped to the display label by
#              the best-seed figure, which reads the caches directly)
#   color    : Gemini dark grey (pairing with LLMSR-Gemini's black, so one backbone reads
#              as one colour family), Llama amber -- as in plot_ood_common.LLMSR_COLORS.
#              Neither collides with a curve already on these figures: amber's nearest
#              neighbour is the 89M family's light orange (#fb923c), which none of the
#              five carries, and the dashed line + own marker separate them anyway.
#              Gemini was blue (#2563eb) to start with, which sat right on top of
#              AIFeynman's steel-blue dashed curve on the SRBench grid and next to
#              LLMSR-Llama's azure on the LLM-SRBench one.
#   root     : the run tree under results/
#   method   : the eval_oneshot.py method name, which is what its filenames carry.  The
#              "-fit" arm (skeleton + BFGS) is the one the OOD caches were scored from;
#              results/oneshot_llama/noise_0 also holds an older "direct" arm, which is
#              deliberately NOT picked up here.
BACKBONES = {
    "OneShot-Gemini": {
        "backbone": "gemini",
        "src":      "oneshot-gemini",
        "color":    "#525252",
        "root":     "results/oneshot_gemini",
        "method":   "oneshot-fit-gemini35flash",
    },
    "OneShot-Llama": {
        "backbone": "llama",
        "src":      "oneshot-llama",
        "color":    "#f59e0b",
        "root":     "results/oneshot_llama",
        "method":   "oneshot-fit-llama31-8b",
    },
    # Llama-3.3-70B, self-hosted on cluster (slurm/eval_oneshot_vllm.sh, TP=4).  The only
    # one-shot arm run at the CANONICAL 10 seeds rather than seed 0, so it is also the
    # only one whose curves carry a spread -- see solve_rate_stats_by_gap.  Darker amber
    # keeps it in the Llama colour family without colliding with the 8B arm's #f59e0b.
    "OneShot-Llama70B": {
        "backbone": "llama70b",
        "src":      "oneshot-llama70b",
        "color":    "#b45309",
        "root":     "results/oneshot_llama70b",
        "method":   "oneshot-fit-llama33-70b-vllm",
    },
}
LABELS = tuple(BACKBONES)


def is_oneshot(label):
    return label in BACKBONES


def visible_labels(exclude=None):
    """LABELS minus anything named in `exclude` (the figures' --exclude list).

    The one-shot arms are appended AFTER the dataframe-level --only/--exclude filtering
    (they never enter the shared caches), so a figure that wants to drop one has to ask
    for it here.  Names are matched on the INTERNAL label ("OneShot-Llama"), the same
    key --exclude uses for every other arm, not on the printed one ("OneShot-Llama8B").
    """
    drop = set(exclude or ())
    return [l for l in LABELS if l not in drop]


def color(label):
    return BACKBONES[label]["color"]


def add_include_arg(ap, extra=""):
    """Register the shared --include-oneshot flag on an argument parser."""
    ap.add_argument("--include-oneshot", action="store_true",
                    help="Also draw the one-shot LLM baselines (OneShot-Gemini, "
                         "OneShot-Llama, OneShot-Llama70B; one LLM call per problem, "
                         "eval_oneshot.py) from the score_oneshot_ood.py caches." + extra)


def cache_path(label, split, noise):
    """Path to the score_oneshot_ood.py OOD-R^2 cache for (label, split, noise)."""
    spec = BACKBONES[label]
    return os.path.join(SCRIPT_DIR, "results",
                        ood_transform.oneshot_backbone_cache_name(
                            spec["backbone"], split, noise))


def artifact(label, split, noise):
    """Path to a one-shot run's discovered-formula artifact (eval_oneshot.py layout,
    which is eval_llmsr.py's: flat file for Feynman, per-method dir for LLM-SRBench)."""
    spec = BACKBONES[label]
    nd = f"noise_{float(noise):g}"
    if split == "feynman":
        return os.path.join(SCRIPT_DIR, spec["root"], nd,
                            f"results_{spec['method']}_feynman.pkl.gz")
    return os.path.join(SCRIPT_DIR, spec["root"], nd, "llmsrbench",
                        f"{spec['method']}_{split}", "results.pkl.gz")


def attempted_ids(label, split, noise):
    """The problems this arm actually attempted, from its artifact -- the universe its
    solve rate must be divided by.  None when the artifact is missing."""
    from results_io import load_rows_any

    path = artifact(label, split, noise)
    if not os.path.exists(path):
        return None
    _meta, rows = load_rows_any(path)
    ids = {str(r["equation_id"]) for r in rows if r.get("equation_id") is not None}
    return ids or None


def load_cache(label, split, noise):
    """The arm's OOD-R^2 rows (gap, algorithm, equation_id, seed, ood_r2), with the
    problems score_oneshot_ood.py dropped -- no equation, or one it could not parse --
    restored as UNSOLVED rows (ood_r2 = -inf) at every gap and seed the cache carries.
    See the module docstring: without them the denominator is the parsed subset, not the
    benchmark.  None when the cache is missing."""
    path = cache_path(label, split, noise)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "equation_id" not in df.columns and "dataset" in df.columns:
        df = df.rename(columns={"dataset": "equation_id"})
    df["equation_id"] = df["equation_id"].astype(str)
    df = df[["gap", "algorithm", "equation_id", "seed", "ood_r2"]]
    if df.empty:
        return df

    missing = (attempted_ids(label, split, noise) or set()) - set(df["equation_id"])
    if missing:
        alg = df["algorithm"].iloc[0]
        pad = [{"gap": g, "algorithm": alg, "equation_id": e, "seed": s,
                "ood_r2": -np.inf}
               for g in df["gap"].unique()
               for s in df["seed"].unique()
               for e in sorted(missing)]
        df = pd.concat([df, pd.DataFrame(pad)], ignore_index=True)
    return df


def solve_rate_stats_by_gap(label, split, noise, thr, monotone=True):
    """{gap: (mean, std)} for one arm: the fraction of the problems it attempted whose
    OOD R^2 >= thr, computed per seed, then the mean and population std ACROSS SEEDS.

    The std is the same quantity plot_ood_vs_gap shades as band_lo/band_hi for every
    other multi-seed curve -- never a Wilson/binomial interval.  A single-seed arm
    (Gemini, Llama-3.1-8B) has one rate, so its std is 0 and its error bars have zero
    length, leaving those curves exactly as they were drawn before.  Empty dict if the
    cache is missing."""
    df = load_cache(label, split, noise)
    if df is None or df.empty:
        return {}
    if monotone:
        df = ood_transform.monotone_gap(ood_transform.restrict_to_common(df), thr)
    out = {}
    for g, gd in df.groupby("gap"):
        seed_rates = []
        for _s, sd in gd.groupby("seed"):
            n_total = sd["equation_id"].nunique()
            if n_total == 0:
                continue
            n_correct = int((sd.groupby("equation_id")["ood_r2"].max() >= thr).sum())
            seed_rates.append(n_correct / n_total)
        if seed_rates:
            out[int(g)] = (float(np.mean(seed_rates)), float(np.std(seed_rates)))
    return out


def solve_rate_by_gap(label, split, noise, thr, monotone=True):
    """{gap: solve_rate} -- solve_rate_stats_by_gap without the spread."""
    return {g: m for g, (m, _s) in
            solve_rate_stats_by_gap(label, split, noise, thr, monotone=monotone).items()}


def time_complexity_stat(label, split, noise, thr):
    """((time_mean, time_std), (cplx_mean, cplx_std)) for one one-shot arm at
    (split, noise); (None, None) if the artifact is missing.  Mirrors
    llmsr_overlay.time_complexity_stat: time = search_time over every problem;
    complexity = the node count over the SOLVED problems (id-R^2 >= thr).

    Each problem is first averaged ACROSS SEEDS, so the reported std is across PROBLEMS
    -- the convention plot_time_complexity_vs_noise states for our own multi-seed models
    ("std is across datasets, each dataset's mean-across-seeds time").  Pooling the raw
    (problem, seed) rows instead would fold seed variance into a bar that every other row
    on the same panel means as problem spread, and would do it over 10x the samples.  A
    single-seed arm has one row per problem, so this is a no-op for Gemini / Llama-3.1-8B.
    """
    from collections import defaultdict

    from results_io import load_rows_any
    from compare_srbench import formula_complexity

    path = artifact(label, split, noise)
    if not os.path.exists(path):
        return None, None
    _meta, rows = load_rows_any(path)
    # Grouped by SEED, not by problem: each seed contributes one mean and the reported
    # std is the spread of those, the quantity every other row of sets 2/4 carries.
    # The two small backbones ran seed 0 only, so their bars are 0.0; the 70B ran ten.
    times, cplx = defaultdict(list), defaultdict(list)
    for r in rows:
        seed = int(r.get("seed", 0) or 0)
        t = r.get("search_time")
        if t is not None and np.isfinite(t):
            times[seed].append(float(t))
        r2 = (r.get("id_metrics") or {}).get("r2")
        if r2 is not None and np.isfinite(r2) and r2 >= thr:
            c = formula_complexity(r.get("discovered_equation"))
            if np.isfinite(c):
                cplx[seed].append(float(c))
    _stat = lambda d: ((lambda p: (float(np.mean(p)),
                                   float(np.std(p)) if len(p) >= 2 else 0.0))(
                           [float(np.mean(v)) for v in d.values() if v])
                       if any(d.values()) else None)
    return _stat(times), _stat(cplx)


def add_time_complexity_rows(st, sc, split, noises, thr, labels=None):
    """Append the one-shot rows to a time / complexity dot plot's series dicts.

    Shared by sets 2 and 4, which both build {label: {noise: (mean, std)}} tables.  Rows
    are added for whichever (arm, noise) artifacts exist; a missing one is simply absent
    from that panel rather than plotted as a hole."""
    for label in (labels or LABELS):
        for tau in noises:
            t_stat, c_stat = time_complexity_stat(label, split, tau, thr)
            if t_stat is not None:
                st.setdefault(label, {})[tau] = t_stat
            if c_stat is not None:
                sc.setdefault(label, {})[tau] = c_stat


def draw_overlay(ax, split, noise, gaps, thr, *, as_percent, labels=None, monotone=True):
    """Draw the one-shot curves on `ax` for (split, noise), each with +/-1 seed-std
    error bars where the arm ran more than one seed.  Returns the line handles (an arm
    whose cache is missing / empty is skipped).  `as_percent`: set1 plots fractions
    [0,1], set3 plots percent [0,100]."""
    gap_pos = {g: i for i, g in enumerate(gaps)}
    top = 100.0 if as_percent else 1.0
    lines = []
    for label in (labels or LABELS):
        rates = solve_rate_stats_by_gap(label, split, noise, thr, monotone=monotone)
        if not rates:
            continue
        xs, ys, es = [], [], []
        for g in gaps:
            v = rates.get(int(g))
            if v is not None and np.isfinite(v[0]):
                mean, std = v
                xs.append(gap_pos[g])
                ys.append(top * mean)
                es.append(top * (std if np.isfinite(std) else 0.0))
        if not xs:
            continue
        col = color(label)
        (line,) = ax.plot(xs, ys, color=col, lw=CURVE_LW, linestyle=LINESTYLE,
                          marker=method_marker(label), markersize=CURVE_MS, **HOLLOW,
                          label=label, zorder=6)
        # +/-1 seed-std, clipped to the axis range so a wide bar near 0 or the ceiling
        # does not run off the panel.  A single-seed arm has std 0 here, so this adds
        # nothing visible to the Gemini / Llama-3.1-8B curves.  Same style as
        # plot_ood_vs_gap._draw_curves, so the two spreads read as the same quantity.
        if any(e > 0 for e in es):
            ys_a, es_a = np.asarray(ys, float), np.asarray(es, float)
            yerr = np.vstack([np.clip(ys_a - np.clip(ys_a - es_a, 0, None), 0, None),
                              np.clip(np.clip(ys_a + es_a, None, top) - ys_a, 0, None)])
            ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor=col,
                        elinewidth=1.0, capsize=2.5, capthick=1.0, alpha=0.85,
                        zorder=5.8)
        lines.append(line)
    return lines
