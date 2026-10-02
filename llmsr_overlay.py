"""llmsr_overlay.py -- draw the single-seed LLMSR-Llama OOD-accuracy-vs-gap curve on the
SRBench (set1) and LLM-SRBench (set3) grids, straight from the score_llmsr_ood.py caches
(results/llmsr_llama_ood_raw_<split>_noise<tau>.csv).

This is a sanity-check overlay: it reuses the same curve styling as the other lines
(same lw / marker-edge), and the LLMSR-Llama colour (azure) + marker ("<") it already
has in sets 5/6.  The run is a SINGLE seed, so it is one line with NO error bars.
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from plot_markers import method_marker, HOLLOW, CURVE_MS, CURVE_LW
import ood_transform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# One entry per self-hosted LLM-SR backbone.  Adding a backbone here is enough to draw
# it: the artifact tree and the score_llmsr_ood cache names are identical across
# backbones apart from the run directory and the method string, which is why this is a
# table rather than a second module.  Colours match plot_ood_common.LLMSR_COLORS.
BACKBONES = {
    "LLMSR-Llama": {
        "backbone": "llama",
        "root":     "results/llmsr_llama",
        "method":   "llmsr-llama31-8b-vllm",
        # azure -- matches plot_llmsrbench_curves_best_seed._LLMSR_BACKBONES
        # (was rose #e11d48; red is now 145M+D&C's, see devncon_addon)
        "color":    "#0369a1",
    },
    "LLMSR-Qwen": {
        "backbone": "qwen",
        "root":     "results/llmsr_qwen",
        "method":   "llmsr-qwen25-coder-32b-vllm",
        "color":    "#7c3aed",          # violet -- plot_ood_common.LLMSR_COLORS
    },
}
LLMSR_LABELS = tuple(BACKBONES)
DEFAULT_LABEL = "LLMSR-Llama"

# Kept as module constants because call sites outside this file read them by name
# (plot_time_complexity_vs_noise stores rows under LLMSR_LABEL).  They name the DEFAULT
# backbone, so every pre-existing call site behaves exactly as it did before.
LLMSR_LABEL = DEFAULT_LABEL
LLMSR_COLOR = BACKBONES[DEFAULT_LABEL]["color"]


def _artifact(split, noise, label=DEFAULT_LABEL):
    """Path to `label`'s discovered-programs artifact at (split, noise)."""
    spec = BACKBONES[label]
    nd = f"noise_{float(noise):g}"
    if split == "feynman":
        return os.path.join(SCRIPT_DIR, spec["root"], nd,
                            f"results_{spec['method']}_feynman.pkl.gz")
    return os.path.join(SCRIPT_DIR, spec["root"], nd, "llmsrbench",
                        f"{spec['method']}_{split}", "results.pkl.gz")


def time_complexity_stat(split, noise, thr, label=DEFAULT_LABEL):
    """((time_mean, time_std), (cplx_mean, cplx_std)) for LLMSR-Llama at (split, noise);
    (None, None) if the artifact is missing.  Mirrors set6's per-seed computation over
    the single-seed run: time = search_time over every problem; complexity = the
    llmsr_complexity node count over the SOLVED problems (id-R^2 >= thr)."""
    from results_io import load_rows_any
    from llmsr_complexity import llmsr_complexity

    path = _artifact(split, noise, label)
    if not os.path.exists(path):
        return None, None
    _meta, rows = load_rows_any(path)
    # Both stats are per-SEED: each seed contributes its own mean, and the reported std
    # is the spread of those means (0.0 for a single-seed run), so this row carries the
    # same quantity as every other row of sets 2 and 4.
    times, cplx = defaultdict(list), defaultdict(list)
    for r in rows:
        seed = int(r.get("seed", 0) or 0)
        t = r.get("search_time")
        if t is not None and np.isfinite(t):
            times[seed].append(float(t))
        r2 = (r.get("id_metrics") or {}).get("r2")
        if r2 is not None and np.isfinite(r2) and r2 >= thr:
            c = llmsr_complexity(r.get("discovered_equation"))
            if np.isfinite(c):
                cplx[seed].append(float(c))
    return _seed_stat(times), _seed_stat(cplx)


def _seed_stat(by_seed):
    """(mean, std) over per-seed means of {seed: [values]}; None when empty.

    The std is the seed-to-seed spread -- 0.0 for a single-seed run -- which is what
    every dot-panel bar in sets 2/4 reports (compare_srbench._seed_std_of_mean)."""
    per_seed = [float(np.mean(v)) for v in by_seed.values() if v]
    if not per_seed:
        return None
    return float(np.mean(per_seed)), (float(np.std(per_seed)) if len(per_seed) >= 2 else 0.0)


def _cache_path(split, noise, label=DEFAULT_LABEL):
    return os.path.join(SCRIPT_DIR, "results",
                        ood_transform.llmsr_backbone_cache_name(
                            BACKBONES[label]["backbone"], split, noise))


def solve_rate_by_gap(csv_path, thr, monotone=True):
    """{gap: solve_rate} from a score_llmsr_ood cache.  Solve rate per gap = the fraction
    of equations whose OOD R^2 >= thr, computed per seed then averaged over seeds (a
    single-seed run -> that seed's rate).  Empty dict if the cache is missing."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if monotone:
        df = ood_transform.monotone_gap(ood_transform.restrict_to_common(df), thr)
    key = "dataset" if "dataset" in df.columns else "equation_id"
    out = {}
    for g, gd in df.groupby("gap"):
        seed_rates = []
        for _s, sd in gd.groupby("seed"):
            n_total = sd[key].nunique()
            if n_total == 0:
                continue
            n_correct = int((sd.groupby(key)["ood_r2"].max() >= thr).sum())
            seed_rates.append(n_correct / n_total)
        if seed_rates:
            out[int(g)] = float(np.mean(seed_rates))
    return out


def draw_overlay(ax, split, noise, gaps, thr, *, as_percent, monotone=True,
                 label=DEFAULT_LABEL):
    """Draw `label`'s curve on `ax` for (split, noise).  Returns the line handle,
    or None if the cache is missing / empty.  `as_percent`: set1 plots fractions [0,1],
    set3 plots percent [0,100]."""
    rates = solve_rate_by_gap(_cache_path(split, noise, label), thr, monotone=monotone)
    if not rates:
        return None
    gap_pos = {g: i for i, g in enumerate(gaps)}
    xs, ys = [], []
    for g in gaps:
        v = rates.get(int(g))
        if v is not None and np.isfinite(v):
            xs.append(gap_pos[g])
            ys.append(100.0 * v if as_percent else v)
    if not xs:
        return None
    (line,) = ax.plot(xs, ys, color=BACKBONES[label]["color"], lw=CURVE_LW,
                      linestyle="-", marker=method_marker(label),
                      markersize=CURVE_MS, **HOLLOW, label=label, zorder=6)
    return line
