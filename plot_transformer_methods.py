#!/usr/bin/env python3
"""
plot_transformer_methods.py -- the transformer-only cut of sets 1 and 2.

Draws TWO figures, built by the very scripts that build sets 1 and 2, so they are
style-identical to the paper's by construction -- each is one canvas with SRBench and
LSR-Transform on it, not a separate figure per benchmark:

    plots/transformers_curves_grid.pdf      set-1 style: OOD solve rate vs gap, four
                                            noise columns, SRBench over LSR-Transform
                                            (plot_combined_curves_grid.py)
    plots/transformers_time_complexity.pdf  set-2 style: search time | formula size for
                                            each benchmark, one row of four panels
                                            (plot_combined_time_complexity.py)

(Until 2026-09-22 this emitted FOUR figures -- a curve grid and a dot plot per
benchmark, transformers_{srbench,llmsrbench}_*.pdf -- drawn by the single-row scripts.
They are now merged the same way sets 1/3 and 2/4 were merged into sets 1 and 2: one
legend strip instead of two, and the two benchmarks read side by side.)

Restricted to the TRANSFORMER-BASED methods and nothing else -- no GP baselines
(Operon, FEAT, AFP, ...), no LLM search methods, no AI Feynman:

    ours (fine tuned, 48h)   our noise-augmented finetune, beam search
    + D&C                    the same checkpoint under divide-and-conquer decoding
    e2e                      end-to-end transformer     TPSR   e2e + MCTS decoding
    PhyE2E (E2E)             PhysicsRegression
    NeSymReS                 Biggio et al. 2021
    tf4sr                    Lalande et al. 2023
    SymFormer                Vastl et al. 2022

The two "ours" arms are the FINE-TUNED checkpoint (145M-len80-float_ftnoise_48h),
matching what sets 1 and 2 print as "ours" -- the plain 145M arms this figure used to
draw were retired from the paper's figures on 2026-09-22.

WHY A SEPARATE SCRIPT
---------------------
NeSymReS / tf4sr / SymFormer are deliberately NOT added to sets 1-3: those figures are
the full-benchmark comparison, and these three cannot answer it (see COVERAGE).  This
script keeps them in their own pair of figures rather than changing what set*.pdf means.
It holds no plotting logic at all: both figures are produced by shelling out to the
combined scripts with a --only whitelist, and the three baselines reach the dot plots
through pretrained_addon (--include-pretrained), which reads their per-seed artifacts
live because no cache knows those trees.

!! COVERAGE -- READ BEFORE QUOTING ANY NUMBER FROM THESE FIGURES !!
The methods are NOT scored on the same problems.  Each is evaluated on every problem it
can represent, and the three additions can represent very few:

    ours / +D&C / e2e / TPSR / PhyE2E           119 SRBench  |  111 LSR-Transform
    tf4sr        97  (<= 6 variables, strictly positive inputs)  |  105
    NeSymReS     52  (<= 3 variables)                            |   35
    SymFormer    16  (<= 2 variables)                            |    5

So SymFormer's high solve rate is "78% of the 16 easiest problems", not "78% of the
benchmark".  --common-subset restricts the three baselines' time/complexity rows to the
problems all three attempted; it does NOT restrict the full-coverage arms, whose rows
come from caches holding one aggregate row each.

OOD DOMAIN: reads the cached OOD tables, which are scored against
datasets/feynman_ood_g*.pkl.gz -- all `method: 'freeze'` as of 2026-09-07.  Run
augment_pretrained_ood_caches.py first if the three additions are missing from the
cache; --reuse-ood backfills missing GAPS but never missing ALGORITHMS, so their curves
would silently vanish rather than error.

USAGE
    python plot_transformer_methods.py                    # both figures
    python plot_transformer_methods.py --figure curves    # set-1 style only
    python plot_transformer_methods.py --figure timecplx  # set-2 style only
    python plot_transformer_methods.py --common-subset    # see COVERAGE
"""
import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DEFAULT_NOISES = ["0", "0.001", "0.01", "0.1"]

# Our fine-tuned checkpoint, spelled as the caches and results trees key it.  The paper's
# figures print it (and its two decodes) through plot_io.LEGEND_DISPLAY.
OURS = "145M-len80-float_ftnoise_48h"

# The methods to draw, as the labels --only matches -- POST-remap names
# (plot_ood_vs_gap._MODEL_RENAME), which is also what the time/complexity series are
# keyed by.  The three lowercase ones are pretrained_addon's.
# No f"{OURS}+TPSR" (dropped 2026-09-23): MCTS decoding of our checkpoint is not drawn
# here.  "TPSR" below is the STANDALONE e2e+TPSR arm, which stays.
TRANSFORMERS = [OURS, f"{OURS}+D&C", "e2e", "TPSR", "PhyE2E (E2E)",
                "nesymres", "tf4sr", "symformer"]


def _run(cmd):
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    return subprocess.call(cmd, cwd=SCRIPT_DIR)


# -- figure 1: set-1 style two-row curve grid -------------------------------------

def make_curves(args):
    """plot_combined_curves_grid.py, i.e. exactly the set-1 command, with --only.

    Both inner command lines filter with --only here; set 1 whitelists its SRBench row
    and blacklists its LSR-Transform one, but this figure wants the same eight arms on
    both rows, so the whitelist is the simpler spelling for each.  The SRBench row needs
    no --include-pretrained: its three baselines come out of the OOD cache, which
    augment_pretrained_ood_caches.py has already written them into.
    """
    inner = ["--grid", "--reuse-ood", "--include-tpsr-combo", "--include-devncon",
             "--include-phye2e", "--only", *TRANSFORMERS]
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "plot_combined_curves_grid.py"),
           "--output", args.curves_output,
           "--", *inner,
           "--", *inner, "--include-e2e-tpsr", "--include-pretrained"]
    return _run(cmd)


# -- figure 2: set-2 style row of four time / complexity panels --------------------

def make_timecplx(args):
    """plot_combined_time_complexity.py, i.e. exactly the set-2 command, with --only.

    --include-pretrained is passed on BOTH rows here (unlike the curve figure): no
    time or complexity cache knows results/{nesymres,tf4sr,symformer}/, so those rows
    are read live from the per-seed artifacts by pretrained_addon.
    """
    inner = ["--noises", *args.noises, "--reuse-cache",
             "--include-tpsr-combo", "--include-devncon", "--include-phye2e",
             "--include-pretrained", "--only", *TRANSFORMERS]
    if args.common_subset:
        inner += ["--pretrained-common-subset"]
    cmd = [sys.executable,
           os.path.join(SCRIPT_DIR, "plot_combined_time_complexity.py"),
           "--output", args.timecplx_output,
           "--", *inner,
           "--", *inner, "--include-e2e-tpsr"]
    return _run(cmd)


# -- Main ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=["curves", "timecplx", "both"], default="both")
    ap.add_argument("--noises", nargs="+", default=DEFAULT_NOISES,
                    help="Noise levels for the time/complexity figure (the curve grid "
                         "draws its own four columns).")
    ap.add_argument("--common-subset", action="store_true",
                    help="Restrict the three baselines' time/complexity rows to the "
                         "problems ALL THREE attempted (the SymFormer-sized set). See "
                         "the COVERAGE note in this file's docstring.")
    ap.add_argument("--curves-output", default="plots/transformers_curves_grid.png")
    ap.add_argument("--timecplx-output",
                    default="plots/transformers_time_complexity.png")
    args = ap.parse_args()

    rc = 0
    if args.figure in ("curves", "both"):
        print("### figure 1: OOD solve rate vs gap, both benchmarks (set-1 style) ###",
              flush=True)
        rc |= make_curves(args)
    if args.figure in ("timecplx", "both"):
        print("### figure 2: time + formula size, both benchmarks (set-2 style) ###",
              flush=True)
        rc |= make_timecplx(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
