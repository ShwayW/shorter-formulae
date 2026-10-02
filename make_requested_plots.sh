#!/bin/bash
# make_requested_plots.sh -- regenerate the three requested benchmark figures.
#
# 2026-09-21: what used to be five figures is now three.  The two benchmarks draw nearly
# the same arms, so each pair was merged onto one canvas under one legend:
#
#   1 OOD accuracy vs extrapolation gap, 4 noise columns -- SRBench on the TOP row,
#       LSR-Transform on the BOTTOM row  (the old sets 1 and 3).
#       plots/set1_curves_grid.pdf, via plot_combined_curves_grid.py
#   2 Search time + formula size -- SRBench on the TOP row, LSR-Transform on the BOTTOM
#       row  (the old sets 2 and 4).
#       plots/set2_time_complexity.pdf, via plot_combined_time_complexity.py
#   3 LSR-Transform 3-panel at noise 0, best seed: OOD-accuracy-vs-gap curves + search
#       time + formula size, incl. LLM-SR  (was set 5).
#       plots/set3_llmsrbench_bestseed.pdf
#
# The four standalone scripts behind sets 1 and 2 still run on their own (that is how
# the appendix figures are made -- see make_llm_appendix_plots.sh); they are simply not
# invoked here any more.
#
# All three figures carry the standalone "e2e+TPSR" arm -- the authors' TPSR; see
# INCLUDE_E2E_TPSR below.  On SRBench it is their published single run, so no band; on
# LLM-SRBench it is our own 3-seed sweep, so that row shows its spread.
#
# All three also carry the one-shot LLM arms -- OneShot-Gemini / OneShot-Llama /
# OneShot-Llama70B, one LLM call per problem; see INCLUDE_ONESHOT below.  Every figure
# drops the two Llama arms and keeps Gemini alone -- see CURVE_DROP.
#
# All three show the "<model> + TPSR" combined runs from results/mymodels_tpsr/
# (89M+TPSR, 145M+TPSR): sets 1 and 2 opt them in with --include-tpsr-combo; set 3 picks
# them up via its default method list.  The OOD-R^2 and complexity for sets 1-2 are
# precomputed once by augment_tpsr_caches.py and appended to the same caches the plots
# reuse (the guard below runs that step only if it hasn't run yet); set 3 computes its
# combo complexity live.
#
# The curve rows (sets 1 and 3) draw one distinct marker per method (plot_markers.py) so
# the lines are separable for colour-blind viewers and in greyscale, not by colour alone.
# All markers -- the per-method ones on the curves and the per-noise ones on the dot
# plots of sets 2/3 -- are drawn HOLLOW (outline only, no fill: plot_markers.HOLLOW), so
# points that land on top of each other stay individually readable.
#
# Uses the cached OOD-R^2 / complexity tables (fast).  For a from-scratch run drop the
# --reuse-ood / --reuse-cache flags below; set 3 additionally needs the LLMSR OOD
# cache, produced by:
#   python score_llmsr_ood.py --split lsr_transform
#
# Usage:   ./make_requested_plots.sh            # PDF (the only output format)
#          FORCE_AUGMENT=1 ./make_requested_plots.sh   # recompute the tpsr cache rows
set -eu
source env/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# The tpsr-combo display labels ("89M+TPSR", "145M+TPSR") are passed straight through
# --only; kept as a quoted array for clarity (and robustness if a label ever gains a space).
# The base-model combos "145M+TPSR" / "145M+D&C" were dropped 2026-09-22: only the
# fine-tuned TPSR and D&C curves are kept, so each decode appears once rather than
# twice (base and fine tuned).  They are named in DROP_ARMS below for the
# LLM-SRBench figures, which filter by --exclude rather than by this whitelist.
# Naming: the prefix is the BENCHMARK ROW a list filters, not a family of methods --
# SRBENCH_* is the SRBench row's whitelist/blacklist, LSRT_* the LSR-Transform row's.
# Both hold our own transformer arms as well as the baselines (renamed from SR_ONLY /
# SR_DROP / LLMSR_DROP* on 2026-09-22, which read as excluding ours).
SRBENCH_KEEP=(e2e TPSR AIFeynman
         # The noise-augmented finetune of that arm.  Only the 48 h checkpoint is drawn
         # (2026-09-22): the 7.6 h (e27) and 23 h (e80) ones were dropped in its favour,
         # exactly as e80 once replaced e27.  Their results and cache rows are untouched --
         # re-add the labels here (and remove them from DROP_ARMS) to bring them back.
         #
         # Four arms, all on the 48 h checkpoint: plain beam, its --unscale decode (which
         # is +13..16 points on SRBench, hence drawn rather than dropped as the other
         # _unscaled arms are), and the D&C / TPSR decodes.
         "145M-len80-float_ftnoise_48h"
         "145M-len80-float_ftnoise_48h_unscaled"
         "145M-len80-float_ftnoise_48h+D&C"
         "145M-len80-float_ftnoise_48h+TPSR"
         "PhyE2E (E2E)"                               # base + combined (the -units arm is dropped)
         "PhyE2E (E2E)-faithful"                      # + the three faithful-defaults arms
         "PhyE2E (E2E)-faithful_dnc"
         "PhyE2E (E2E)-faithful_dnc_mcts")
SRBENCH_DROP="145M_unscaled 89M_unscaled"                 # whitened variants, always dropped
LSRT_DROP="145M_unscaled 89M_unscaled 89M-float"    # set 3 & 4: keep e2e/145M(+TPSR)

# Arms dropped from EVERY figure: the three 89M arms, the three 145M-len80 arms
# (including the SimpliPy-canonicalised one), the units-hint PhyE2E arm and the LLM-SR
# Llama-8B arm -- the last two dropped 2026-09-12.  The
# SRBench figures filter with a --only whitelist, so they are simply absent from SRBENCH_KEEP
# above; the LLM-SRBench figures filter with --exclude, so they are named here.  Set 5
# takes them through its own --exclude, which thins its DEFAULT_METHODS list.
DROP_ARMS=(89M "89M+TPSR" "89M+D&C" 145M-len80 "145M-len80+D&C" "145M-len80-simplipy"
           # The plain 145M arm: superseded in sets 1-5 by the len-80 float arm, which
           # now prints as "ours" (2026-09-17).  Exact match, so the "145M+TPSR" and
           # "145M+D&C" combined arms are untouched.
           145M
           "PhyE2E (E2E)-units"          # the units-hint PhyE2E arm, dropped 2026-09-12
           # The base-model TPSR / D&C combos, dropped 2026-09-22 in favour of the
           # fine-tuned ones.  Exact isin() match, so "145M-len80-float_ftnoise_48h+TPSR"
           # and "...+D&C" are untouched.
           "145M+TPSR" "145M+D&C"
           # (LLMSR-Llama was dropped here until 2026-09-23; it is now drawn in sets 1-2
           #  via LLMSR_ARGS, so excluding it would cancel that out.)
           # The 7.6 h finetune and its TPSR / D&C arms, dropped 2026-09-21 in favour of
           # the 23 h checkpoint.  --exclude is an exact isin() test, so all three are named.
           "145M-len80-float_ftnoise_e27" "145M-len80-float_ftnoise_e27+TPSR" "145M-len80-float_ftnoise_e27+D&C"
           # The 23 h checkpoint, dropped 2026-09-22 in favour of the 48 h one.
           "145M-len80-float_ftnoise_e80"
           # The UN-finetuned len-80 float arm, dropped 2026-09-23: the 48 h finetune now
           # prints as plain "ours", so keeping this one would put two curves called
           # "ours" in the same legend.  Its results and cache rows are untouched.
           "145M-len80-float"
           145M_100l)                    # fml_len<=100 arm: mid-pack accuracy at ~4x the
                                         # inference cost.  It reaches the LLM-SRBench
                                         # figures only by ARTIFACT discovery (GROUPS
                                         # gained mymodels_400), hence set 4 alone.
LSRT_DROP="$LSRT_DROP ${DROP_ARMS[*]}"

# Labels that must be dropped through a QUOTED array rather than $LSRT_DROP: that is a
# space-separated string expanded unquoted, so "PhyE2E (E2E)-units" would word-split into
# two bogus arguments and silently fail to match.  Applied to BOTH LLM-SRBench figures so
# the pair stays matched -- only set 4 was actually drawing it (2026-09-14).
LSRT_DROP_ARR=("PhyE2E (E2E)-units")

# Arms dropped from ALL FIVE figures: the two Llama one-shot arms.  The three one-shot
# arms tell one story, so only the Gemini one is drawn.  Applied to sets 2/4/5 as well
# as the grids (2026-09-14) so each dot plot lists exactly the methods its companion
# grid draws -- set 2 matches set 1, set 4 matches set 3.  These names are the INTERNAL
# labels (oneshot_overlay.BACKBONES keys), not the printed "OneShot-Llama8B".
# PySR-10 is named here as well as omitted from --pysr-groups: that flag only governs
# LIVE discovery, so under --reuse-ood / --reuse-cache (sets 1 and 2) its cached rows
# were still drawn.  Set 3 computes live and so respected the flag alone -- which is
# exactly how an arm ends up in two figures out of three.
CURVE_DROP=("OneShot-Llama" "OneShot-Llama70B" "PySR-10")

# --- LLM-SR overlays (results/llmsr_llama, results/llmsr_qwen) -----------------------
# The two self-hosted backbones: Llama-3.1-8B and Qwen2.5-Coder-32B, both run through
# LLM-SR's search loop (thousands of LLM calls per problem, hence their own curves
# rather than a one-shot arm).  --llmsr-backbones is REQUIRED to get more than one: its
# default is LLMSR-Llama alone, which is what every figure drew before the flag existed,
# so naming only --include-llmsr would silently leave Qwen out.
# The Gemini LLM-SR arm is NOT here -- set 3 picks it up through its own method list.
LLMSR_ARGS=(--include-llmsr --llmsr-backbones LLMSR-Llama LLMSR-Qwen)

# --- one-time: append the tpsr-combo OOD-R^2 + complexity to the plotting caches ----
# (idempotent; skipped when the rows are already present unless FORCE_AUGMENT=1).
# An augment is skipped only when EVERY cache it writes already carries its rows, at
# EVERY noise level.  The old guards tested a single file (results/ood_raw_noise0.csv),
# so an augment that had only got as far as the OOD side -- or that was interrupted
# partway through the noise sweep -- looked complete and never re-ran.  That is how the
# D&C rows went missing from all eight complexity caches while the OOD caches had them.
# Each augment is guarded on the cache families it actually writes:
#   tpsr        -> SRBench + LLM-SRBench, OOD + complexity  (all four)
#   devncon     -> the SRBench OOD cache only; augment_devncon_caches.py writes nothing
#                  else (its docstring's mention of complexity_noise*.csv is aspirational).
#                  D&C complexity comes from the plot scripts' own non---reuse-cache path.
needs_augment() {                 # $1 = marker, $2.. = cache families to check
  [ "${FORCE_AUGMENT:-0}" = "1" ] && return 0
  local marker="$1"; shift
  local tau kind f
  for tau in 0 0.001 0.01 0.1; do
    for kind in "$@"; do
      case "$kind" in
        ood)        f="results/ood_raw_noise${tau}.csv" ;;
        cplx)       f="results/complexity_noise${tau}.csv" ;;
        llmsr_ood)  f="results/llmsr_ood_raw_noise${tau}.csv" ;;
        llmsr_cplx) f="results/llmsr_complexity_noise${tau}.csv" ;;
      esac
      grep -qiF -- "$marker" "$f" 2>/dev/null || return 0
    done
  done
  return 1
}

# Exact-algorithm variant of needs_augment.  needs_augment greps for a SUBSTRING, which
# is fine for markers like "PhyE2E (E2E)" but wrong for "TPSR": every cache already holds
# "89M+TPSR"/"145M+TPSR" rows, so a substring test always reports the standalone arm as
# present and the augment would never run.  This checks the algorithm column itself.
needs_augment_alg() {            # $1 = exact algorithm, $2.. = cache families
  [ "${FORCE_AUGMENT:-0}" = "1" ] && return 0
  local alg="$1"; shift
  python - "$alg" "$@" <<'EOF'
import sys, os
import pandas as pd
alg, kinds = sys.argv[1], sys.argv[2:]
name = {"ood": "results/ood_raw_noise{t}.csv",
        "cplx": "results/complexity_noise{t}.csv",
        "llmsr_ood": "results/llmsr_ood_raw_noise{t}.csv",
        "llmsr_cplx": "results/llmsr_complexity_noise{t}.csv"}
for tau in ("0", "0.001", "0.01", "0.1"):
    for k in kinds:
        f = name[k].format(t=tau)
        try:
            if alg not in set(pd.read_csv(f)["algorithm"].astype(str)):
                raise SystemExit(0)          # missing -> needs augment
        except FileNotFoundError:
            raise SystemExit(0)
raise SystemExit(1)                          # every cache already has it
EOF
}

if needs_augment tpsr ood cplx llmsr_ood llmsr_cplx; then
  echo "### augmenting caches with the 89M/145M+TPSR results (one-time, slow) ###"
  python augment_tpsr_caches.py
fi
# --- standalone TPSR (the authors' MCTS decoding over the E2E transformer) ----------
# Two different runs under ONE label: LLM-SRBench from our own 3-seed sweep
# (results/tpsr/, eval_e2e_tpsr.py --benchmark llmsrbench), SRBench from the authors'
# published Feynman csv.  Both land in the caches under the algorithm "TPSR" and draw as
# "e2e+TPSR" in the legends -- the identity the SRBench figures already used.
#     INCLUDE_E2E_TPSR=0 ./make_requested_plots.sh     # drop the curve entirely
E2E_TPSR_ARGS=()
if [ "${INCLUDE_E2E_TPSR:-1}" = "1" ]; then
  E2E_TPSR_ARGS=(--include-e2e-tpsr)
  if needs_augment_alg TPSR ood llmsr_ood llmsr_cplx; then
    echo "### augmenting caches with the standalone TPSR results (one-time, slow) ###"
    python augment_e2e_tpsr_caches.py
  fi
fi

if needs_augment "D&C" ood; then
  echo "### augmenting caches with the 89M/145M+D&C (DEVNCON) results (one-time, slow) ###"
  python augment_devncon_caches.py
fi

# --- AI Feynman 2.0 baseline (the ORIGINAL authors' code, results/aifeynman/) --------
# ON by default since 2026-09-21: the run has landed and been merged
# (results/aifeynman/seed42/noise_0/llmsrbench/, all 111 LSR-Transform equations), so the
# figures carry an "AIFeynman" curve on the LSR-Transform side too -- the SRBench side
# already had one, from SRBench's published results.
#
# It is still ONE SEED AT NOISE 0 only (the run costs ~1-2 h/problem; see
# slurm/eval_aifeynman.sh), so on the LSR-Transform side the arm appears in the noise-0
# column alone, with no error bars.  That is a property of the run, not of the plotting.
#     INCLUDE_AIFEYNMAN=0 ./make_requested_plots.sh    # drop it again
# --- PhyE2E baselines (Ying et al., Nature MI 2025) ----------------------------------
# ON by default: unlike AI Feynman these runs are COMPLETE (10 seeds x 4 noises, both
# benchmarks), so every figure carries the curves.
#
# TWO ARMS are drawn, one per results tree, each with its own label/colour/marker:
#     phye2e        -> "PhyE2E (E2E)"          fuchsia,        no units hint
#     phye2e_units  -> "PhyE2E (E2E)-units"    darker fuchsia, physical units supplied
# Three MORE arms run PhyE2E at the AUTHORS' OWN parser defaults (PHYE2E_FAITHFUL=1 in
# slurm/eval_common.sh: greedy decode / beam_size 1, ONE 200-point bag, 200-row fit budget)
# rather than this repo's beam-search protocol.  Added 2026-08-30, complete on both
# benchmarks x 4 noises x 10 seeds:
#     phye2e_faithful           -> "PhyE2E (E2E)-faithful"           plain
#     phye2e_faithful_dnc       -> "PhyE2E (E2E)-faithful_dnc"       + Divide-and-Conquer
#     phye2e_faithful_dnc_mcts  -> "PhyE2E (E2E)-faithful_dnc_mcts"  + D&C + MCTS
# Their OOD caches are already built (results/{,llmsr_}ood_raw_noise*.csv), so the
# augment loop below is a no-op for them.
#
# A further arm (phye2e_full, the --use-divide/--use-mcts/--use-gp published pipeline)
# is still running on cluster; when it lands, add it to PHYE2E_GROUPS and to SRBENCH_KEEP
# and it labels itself "PhyE2E (E2E)-full" automatically -- no code change needed.
#
#     INCLUDE_PHYE2E=0 ./make_requested_plots.sh              # drop them entirely
#     PHYE2E_GROUPS="phye2e" ./make_requested_plots.sh        # just the plain arm
PHYE2E_GROUPS="${PHYE2E_GROUPS:-phye2e phye2e_faithful phye2e_faithful_dnc phye2e_faithful_dnc_mcts}"
PHYE2E_ARGS=()
if [ "${INCLUDE_PHYE2E:-1}" = "1" ]; then
  PHYE2E_ARGS=(--include-phye2e --phye2e-groups $PHYE2E_GROUPS)
  # Each tree writes its own rows into all four cache families (SRBench + LLM-SRBench,
  # OOD + complexity) under its own label, so this runs once PER ARM.
  for _g in $PHYE2E_GROUPS; do
    _marker="PhyE2E (E2E)"
    [ "$_g" = "phye2e" ] || _marker="PhyE2E (E2E)-${_g#phye2e_}"
    if needs_augment "$_marker" ood cplx llmsr_ood llmsr_cplx; then
      echo "### augmenting caches with the $_g results (one-time, slow) ###"
      python augment_phye2e_caches.py --phye2e-group "$_g"
    fi
  done
fi

# --- one-shot LLM baselines (eval_oneshot.py: ONE LLM call per problem) --------------
# ON by default, because these runs are COMPLETE: all three backbones -- Gemini 3.5
# Flash, Llama-3.1-8B and Llama-3.3-70B, each the skeleton+BFGS ("fit") arm -- ran all
# four noises on both benchmarks, and score_oneshot_ood.py has already cached their OOD
# R^2 (results/oneshot_<backbone>_ood_raw_<split>_noise<tau>.csv).
#
# The two small backbones ran seed 0 only; the 70B ran the canonical 10 seeds
# (slurm/eval_oneshot_vllm.sh on cluster), so it is the one one-shot curve that carries
# +/-1 seed-std error bars.  Nothing else about the wiring differs -- oneshot_overlay's
# BACKBONES registry is the only place a backbone is declared, and all five figures pick
# a new entry up from it.
#
# No augment step, unlike every other arm above: the one-shot rows never enter the shared
# caches.  The curve figures read the per-backbone CSVs directly and the dot plots compute
# search time + formula complexity live from results/oneshot_<backbone>/ -- so there is
# nothing to keep in sync and nothing to re-run when the figures are regenerated.
#
# They draw DASHED (grey = Gemini, amber = Llama-3.1-8B, dark amber = Llama-3.3-70B,
# thin "tri" markers), so the "one LLM call" family reads as a family against the solid
# search-based curves.
#
#     INCLUDE_ONESHOT=0 ./make_requested_plots.sh    # drop them entirely
ONESHOT_ARGS=()
if [ "${INCLUDE_ONESHOT:-1}" = "1" ]; then
  ONESHOT_ARGS=(--include-oneshot)
fi

AIFEYNMAN_ARGS=()
if [ "${INCLUDE_AIFEYNMAN:-1}" = "1" ]; then
  AIFEYNMAN_ARGS=(--include-aifeynman)
  [ -n "${AIFEYNMAN_GROUP+x}" ] && AIFEYNMAN_ARGS+=(--aifeynman-group "$AIFEYNMAN_GROUP")
  # Only the LLM-SRBench caches gain rows -- the SRBench "AIFeynman" curve stays
  # SRBench's published one (see augment_aifeynman_caches.py).
  if [ "${FORCE_AUGMENT:-0}" = "1" ] || ! grep -q "AIFeynman" results/llmsr_ood_raw_noise0.csv 2>/dev/null; then
    echo "### augmenting LLM-SRBench caches with the AI Feynman results (one-time, slow) ###"
    python augment_aifeynman_caches.py "${AIFEYNMAN_ARGS[@]:1}"
  fi
fi

# --- PySR (results/pysr/, eval_pysr.py) ---------------------------------------------
# ON by default now that the cluster sweep has banked seeds 42-47 (24 of its 40 cells);
# the remaining seeds 48-51 only tighten the error bars, they add no new arm.
#
# PySR is ABSENT from srbench's published ground-truth feather (14 algorithms, no PySR),
# so unlike AIFeynman there is no published curve to fall back on -- every PySR point in
# these figures is our own run, which is why pysr_addon draws it SOLID (ours-run) rather
# than dashed (published).
#
# Its rows enter all four cache families via augment_pysr_caches.py, which must run
# LOCALLY: the complexity half needs srbench/results/ground-truth_results.feather, which
# the cluster checkout does not carry.
#
#     INCLUDE_PYSR=0 ./make_requested_plots.sh    # drop it
PYSR_ARGS=()
if [ "${INCLUDE_PYSR:-1}" = "1" ]; then
  # TWO budget arms, both short: the 10 s/fit sweep (results/pysr_10/, drawn as
  # "PySR (10s)") and the 60 s one (results/pysr_60/, "PySR (1m)").  The original
  # 1 h/fit run (results/pysr/) and the 1800 s one are on disk and in the caches but
  # are NOT drawn -- see the blacklist below.  --pysr-groups is required, not
  # optional: every figure except the pure --reuse-ood curves identifies PySR by
  # ARTIFACT discovery, and the default group is the only one discovered without it --
  # so the PySR-10 cache rows would sit unused and the curve would simply be absent.
  # 2026-09-22: the 10 s arm was dropped from sets 1-3, so only pysr_60 is loaded.
  # This governs LIVE discovery only; the cached OOD/complexity rows are dropped via
  # CURVE_DROP above.  BOTH are needed -- omitting pysr_10 here alone still left the
  # curve in sets 1 and 2, which reuse the caches.  Results and cache rows are
  # untouched: restore the curve by re-adding pysr_10 here AND removing it there.
  # 2026-09-23: the 1800 s arm is drawn as well ("PySR (30m)"), so the figures show
  # what a search-based method reaches with ~100x our synthesis time.  The 1 h arm
  # stays dropped: it is within 0.02 of the 1800 s one at every point.
  PYSR_ARGS=(--include-pysr --pysr-groups pysr_60 pysr_1800)
  # Sets 1 and 2 filter with the SRBENCH_KEEP whitelist, so --include-pysr is NOT enough on
  # its own there: an un-whitelisted label is dropped after the data loads, silently.
  # Sets 3/4/5 filter with --exclude instead, so they need nothing extra.
  SRBENCH_KEEP+=("PySR-10" "PySR-60" "PySR-1800")
  # The 1 h and 1800 s trees are also on disk and their rows are in the caches, so the
  # cache-row figure (set 4) draws them unless told not to -- which is how it came to
  # show four PySR arms while every other figure showed one.  Blacklist them so all five
  # figures agree on the same two arms; drop these two entries to see the full sweep.
  LSRT_DROP="$LSRT_DROP PySR"
  SRBENCH_DROP="$SRBENCH_DROP PySR"
  if needs_augment "PySR" ood cplx llmsr_ood llmsr_cplx; then
    echo "### augmenting caches with the PySR results (one-time, slow) ###"
    python augment_pysr_caches.py
  fi
fi

# 1. SRBench curves (top row) over LSR-Transform curves (bottom row), 4 noise columns.
#    One canvas, one shared legend: the two benchmarks draw nearly the same arms, so the
#    merged figure spends one legend strip instead of two (2026-09-21; sets 1 and 3 used
#    to be two separate single-row figures, and set 3 below still emits its own).
#    plot_combined_curves_grid.py holds no plotting logic -- it hands each row to the
#    standalone script's row-drawer, so each row is filtered by THAT script's own CLI
#    (SRBench --only whitelist, LLM-SRBench --exclude blacklist).  Hence the two inner
#    command lines below, separated by `--`: they are verbatim the set-1 and set-3 flags.
python plot_combined_curves_grid.py --output plots/set1_curves_grid.png \
  -- --grid --reuse-ood --include-tpsr-combo --include-devncon \
    ${PHYE2E_ARGS[@]+"${PHYE2E_ARGS[@]}"} ${ONESHOT_ARGS[@]+"${ONESHOT_ARGS[@]}"} \
    ${PYSR_ARGS[@]+"${PYSR_ARGS[@]}"} \
    ${LLMSR_ARGS[@]+"${LLMSR_ARGS[@]}"} \
    --exclude $SRBENCH_DROP "${CURVE_DROP[@]}" --only "${SRBENCH_KEEP[@]}" \
  -- --grid --reuse-ood --include-tpsr-combo ${AIFEYNMAN_ARGS[@]+"${AIFEYNMAN_ARGS[@]}"} \
    ${PHYE2E_ARGS[@]+"${PHYE2E_ARGS[@]}"} ${E2E_TPSR_ARGS[@]+"${E2E_TPSR_ARGS[@]}"} --include-devncon \
    ${ONESHOT_ARGS[@]+"${ONESHOT_ARGS[@]}"} ${PYSR_ARGS[@]+"${PYSR_ARGS[@]}"} \
    ${LLMSR_ARGS[@]+"${LLMSR_ARGS[@]}"} \
    --exclude $LSRT_DROP "${CURVE_DROP[@]}" "${LSRT_DROP_ARR[@]}"

# 2. SRBench time + formula size (top row) over LSR-Transform (bottom row).
#    Merged from the old sets 2 and 4 on 2026-09-21, the same way set 1 merged the old
#    sets 1 and 3: one canvas, one shared Target-Noise key instead of two.  As there,
#    plot_combined_time_complexity.py holds no plotting logic -- it hands each row to the
#    standalone script's draw_row, so each row keeps its own CLI (SRBench --only
#    whitelist, LLM-SRBench --exclude blacklist).  The two inner command lines below,
#    separated by `--`, are verbatim the old set-2 and set-4 flags.
python plot_combined_time_complexity.py --output plots/set2_time_complexity.png \
  -- --noises 0 0.001 0.01 0.1 --reuse-cache \
    --include-tpsr-combo --include-devncon ${PHYE2E_ARGS[@]+"${PHYE2E_ARGS[@]}"} \
    ${ONESHOT_ARGS[@]+"${ONESHOT_ARGS[@]}"} ${PYSR_ARGS[@]+"${PYSR_ARGS[@]}"} \
    ${LLMSR_ARGS[@]+"${LLMSR_ARGS[@]}"} \
    --exclude $SRBENCH_DROP "${CURVE_DROP[@]}" --only "${SRBENCH_KEEP[@]}" \
  -- --noises 0 0.001 0.01 0.1 --reuse-cache \
    --include-tpsr-combo ${AIFEYNMAN_ARGS[@]+"${AIFEYNMAN_ARGS[@]}"} ${PHYE2E_ARGS[@]+"${PHYE2E_ARGS[@]}"} \
    ${E2E_TPSR_ARGS[@]+"${E2E_TPSR_ARGS[@]}"} --include-devncon \
    ${ONESHOT_ARGS[@]+"${ONESHOT_ARGS[@]}"} ${PYSR_ARGS[@]+"${PYSR_ARGS[@]}"} \
    ${LLMSR_ARGS[@]+"${LLMSR_ARGS[@]}"} \
    --exclude $LSRT_DROP "${CURVE_DROP[@]}" "${LSRT_DROP_ARR[@]}"

# 3. LLM-SRBench curves + time + formula size, noise 0, best seed, + LLMSR.
#    (Was set 5; renumbered 2026-09-21 when sets 1/3 and 2/4 were merged into two
#    figures.  The standalone LLM-SRBench curve grid and dot plot -- the old sets 3 and 4
#    -- are no longer emitted: they are the bottom rows of sets 1 and 2.  Both scripts
#    still run on their own if either is ever wanted back.)
python plot_llmsrbench_time_complexity_best_seed.py ${AIFEYNMAN_ARGS[@]+"${AIFEYNMAN_ARGS[@]}"} ${PHYE2E_ARGS[@]+"${PHYE2E_ARGS[@]}"} \
    ${E2E_TPSR_ARGS[@]+"${E2E_TPSR_ARGS[@]}"} ${ONESHOT_ARGS[@]+"${ONESHOT_ARGS[@]}"} \
    ${PYSR_ARGS[@]+"${PYSR_ARGS[@]}"} \
    --exclude "${DROP_ARMS[@]}" "${CURVE_DROP[@]}" \
    --output plots/set3_llmsrbench_bestseed.png

echo "All three figures written to plots/."
