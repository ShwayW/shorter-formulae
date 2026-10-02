#!/bin/bash
# Appendix figures: the language-trained arms the BODY figures deliberately hide.
#
#   bash make_llm_appendix_plots.sh
#
# WHY THIS EXISTS: make_requested_plots.sh drops every Llama arm from all five body
# figures -- the two one-shot arms via CURVE_DROP (2026-09-14, "the three one-shot arms
# tell one story, so only the Gemini one is drawn") and the LLM-SR Llama-8B arm via
# DROP_ARMS (2026-09-12).  It also never passes --include-llmsr, so no LLM-SR overlay is
# drawn on the curve grids at all.  The paper's Section 5 nevertheless compares those
# arms, so the claims were unverifiable from any figure.  This appendix figure shows
# them, in two figures mirroring the body's sets 1 and 2 -- appx_curves_llm.pdf (a
# two-row canvas, SRBench over LSR-Transform) and appx_time_complexity_llm.pdf (one row
# of four panels).  The body figures are NOT regenerated here -- different --output names.
#
# WHAT DIFFERS FROM make_requested_plots.sh:
#   * CURVE_DROP emptied              -> the two one-shot Llama arms come back
#   * "LLMSR-Llama" out of DROP_ARMS  -> the LLM-SR Llama arm is no longer excluded
#   * SRBENCH_KEEP gains the three labels  -> sets 1/2 filter with a WHITELIST, so removing
#                                        them from the blacklist is not enough; an
#                                        un-whitelisted label is dropped after its data
#                                        loads, silently.  This is the asymmetry that
#                                        hid the SRBench one-shot arms.
#   * --include-llmsr --llmsr-backbones LLMSR-Llama LLMSR-Qwen
#                                     -> draws BOTH self-hosted LLM-SR backbones.  Qwen
#                                        was never plotted anywhere despite having full
#                                        coverage (both benchmarks x 4 noise levels).
#
# The plot scripts always write PDF regardless of the extension passed (plot_io.save_fig
# rewrites it), so --output names .pdf here to match what actually lands on disk.
set -u
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-plots}"
LOG="logs/plotgen/llm_appendix_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs/plotgen "$OUT_DIR"

# Do NOT pipe this into head: SIGPIPE killed a previous replot mid-figure while the shell
# still reported exit 0 (see final_replot_all_sets.sh).
exec > >(tee -a "$LOG") 2>&1
echo "=== LLM appendix replot started $(date -Is) ==="

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# The repo venv, not a bare `python`: make_requested_plots.sh assumes an ACTIVATED
# venv and dies with "python: command not found" under a non-interactive ssh, which
# is how this script is normally driven.  PY is overridable for other hosts.
PY="${PY:-./env/bin/python}"
[ -x "$PY" ] || { echo "no python at $PY -- set PY=<interpreter>" >&2; exit 1; }
"$PY" -c "import matplotlib, pandas" || { echo "$PY lacks matplotlib/pandas" >&2; exit 1; }

# ---- arms -----------------------------------------------------------------------
# Naming (renamed from SR_ONLY / SR_DROP / LLMSR_DROP* on 2026-09-22): the prefix is the
# BENCHMARK ROW the list filters, not a family of methods -- SRBENCH_* is the top row's
# whitelist/blacklist, LSRT_* the bottom row's.  Both hold our own transformer arms as
# well as the baselines, which the old "SR_ONLY" spelling read as excluding.
#
# These are INTERNAL labels (oneshot_overlay.BACKBONES / llmsr_overlay.BACKBONES keys),
# not printed ones.
# Trimmed 2026-09-22 to the LANGUAGE arms plus our own curves: the figure exists to
# compare the language-trained models, and the non-language baselines it used to carry
# (E2E, TPSR (E2E), PhyE2E, AI Feynman, PySR (1m)) plus the superseded "ours" decodes
# (D&C (ours), TPSR (ours), the 7.6 h and 23 h finetunes) only crowded the panels.  The
# opt-in flags for those arms are emptied below as well -- a label absent from BOTH the
# whitelist and the loader is dropped twice over, which is the intent.
# The four fine-tuned arms were added 2026-09-22 so the TWO ROWS MATCH: the
# LSR-Transform row draws them by default (it filters with a blacklist), so while they
# were missing here the merged figure's legend named arms that only ever appeared in
# its bottom half.
# NOT "145M-len80-float" (the un-finetuned base): plot_io prints the 48 h finetune as
# the plain "ours", exactly as the body figures do, so drawing the base arm beside it put
# TWO rows/curves called "ours" in every panel.  Dropped 2026-09-24 -- the appendix
# follows the body's convention: "ours" IS the fine-tuned arm.
SRBENCH_KEEP=("145M-len80-float_ftnoise_48h"
              "145M-len80-float_ftnoise_48h_unscaled"
              "145M-len80-float_ftnoise_48h+D&C"
              "PySR-10"
              "OneShot-Llama" "OneShot-Llama70B")
SRBENCH_DROP="145M_unscaled 89M_unscaled PySR PySR-1800"

# DROP_ARMS minus "LLMSR-Llama" -- everything else stays dropped exactly as the body,
# plus (2026-09-22) the non-language baselines and superseded "ours" decodes named in
# the SRBENCH_KEEP note above.  The LSR-Transform row filters with a BLACKLIST, so removing
# them from the whitelist is not enough -- they have to be named here too.
DROP_ARMS=(89M "89M+TPSR" "89M+D&C" 145M-len80 "145M-len80+D&C" "145M-len80-simplipy"
           145M "PhyE2E (E2E)-units" 145M_100l
           e2e TPSR "e2e+TPSR" AIFeynman "PySR-60"
           "145M+TPSR" "145M+D&C"
           "145M-len80-float_ftnoise_e27" "145M-len80-float_ftnoise_e27+TPSR"
           "145M-len80-float_ftnoise_e27+D&C" "145M-len80-float_ftnoise_e80"
           # MCTS decoding of the 48 h checkpoint, dropped 2026-09-23.  Named here for
           # the LSR-Transform row AND absent from SRBENCH_KEEP above for the SRBench
           # one -- the two rows filter oppositely, so either alone leaves it drawn.
           "145M-len80-float_ftnoise_48h+TPSR"
           # The un-finetuned base arm, dropped 2026-09-24 for the LSR-Transform row the
           # same way it left SRBENCH_KEEP above: it printed as a second "ours".  Exact
           # isin() match, so every "145M-len80-float_ftnoise_48h*" arm is untouched.
           "145M-len80-float")
LSRT_DROP="145M_unscaled 89M_unscaled 89M-float PySR PySR-1800 ${DROP_ARMS[*]}"
# Labels carrying a SPACE must be dropped through this quoted array: $LSRT_DROP is a
# space-separated string expanded unquoted, so "PhyE2E (E2E)" would word-split into two
# bogus arguments and match nothing.
LSRT_DROP_ARR=("PhyE2E (E2E)-units" "PhyE2E (E2E)" "PhyE2E (E2E)-faithful"
                "PhyE2E (E2E)-faithful_dnc" "PhyE2E (E2E)-faithful_dnc_mcts")

# Emptied: this is the whole point of the appendix figures.
CURVE_DROP=()

ONESHOT_ARGS=(--include-oneshot)
# PySR (10s) only: PySR (1m) is one of the arms dropped 2026-09-22.
PYSR_ARGS=(--include-pysr --pysr-groups pysr_10)
# AI Feynman, PhyE2E and the standalone e2e+TPSR arm are no longer drawn, so their
# loaders are simply not asked for -- cheaper than loading the trees and filtering them
# out, and it keeps the arrays in place for whoever wants them back (put the flags
# back here AND take the labels out of DROP_ARMS / SRBENCH_KEEP above; either one alone
# drops the arm silently).
AIFEYNMAN_ARGS=()
PHYE2E_ARGS=()
E2E_TPSR_ARGS=()
LLMSR_ARGS=(--include-llmsr --llmsr-backbones LLMSR-Llama LLMSR-Qwen)

# ---- SRBench over LSR-Transform, all language-trained arms ------------------------
# ONE figure (2026-09-22; was appx1_srbench_curves_llm.pdf + appx2_llmsrbench_curves_llm.pdf),
# merged the same way sets 1 and 3 were merged into the body's set 1: the two benchmarks
# draw nearly the same arms, so one canvas spends one legend strip instead of two -- and
# with this many language arms that strip is the tallest thing in either old figure.
# plot_combined_curves_grid.py holds no plotting logic; it hands each row to the
# standalone script's row-drawer, so the two inner command lines below, separated by
# `--`, are verbatim the old A and B commands minus their own --output.
"$PY" plot_combined_curves_grid.py --output "$OUT_DIR/appx_curves_llm.pdf" \
  -- --grid --reuse-ood --include-tpsr-combo --include-devncon \
    ${PHYE2E_ARGS[@]+"${PHYE2E_ARGS[@]}"} "${ONESHOT_ARGS[@]}" "${PYSR_ARGS[@]}" "${LLMSR_ARGS[@]}" \
    --exclude $SRBENCH_DROP ${CURVE_DROP[@]+"${CURVE_DROP[@]}"} --only "${SRBENCH_KEEP[@]}" \
  -- --grid --reuse-ood --include-tpsr-combo \
    ${AIFEYNMAN_ARGS[@]+"${AIFEYNMAN_ARGS[@]}"} ${PHYE2E_ARGS[@]+"${PHYE2E_ARGS[@]}"} \
    ${E2E_TPSR_ARGS[@]+"${E2E_TPSR_ARGS[@]}"} --include-devncon \
    "${ONESHOT_ARGS[@]}" "${PYSR_ARGS[@]}" "${LLMSR_ARGS[@]}" \
    --exclude $LSRT_DROP ${CURVE_DROP[@]+"${CURVE_DROP[@]}"} "${LSRT_DROP_ARR[@]}"

# ---- search time + formula size, same arms -----------------------------------------
# The set-2-style twin of the figure above, added 2026-09-23: the curve figure says how
# accurate the language arms are, this one says what they cost -- and that cost is the
# whole point of the LLM-SR rows (orders of magnitude slower synthesis).  Same arm
# filters, same combined script the body's set 2 uses.
# --layout stacked: SRBench row ABOVE the LSR-Transform row (2026-09-24).  The body's
# set 2 keeps the default one-row form -- this flag is the only difference between them.
"$PY" plot_combined_time_complexity.py --output "$OUT_DIR/appx_time_complexity_llm.pdf" \
  --layout stacked \
  -- --noises 0 0.001 0.01 0.1 --reuse-cache --include-tpsr-combo --include-devncon \
    ${PHYE2E_ARGS[@]+"${PHYE2E_ARGS[@]}"} "${ONESHOT_ARGS[@]}" "${PYSR_ARGS[@]}" "${LLMSR_ARGS[@]}" \
    --exclude $SRBENCH_DROP ${CURVE_DROP[@]+"${CURVE_DROP[@]}"} --only "${SRBENCH_KEEP[@]}" \
  -- --noises 0 0.001 0.01 0.1 --reuse-cache --include-tpsr-combo \
    ${AIFEYNMAN_ARGS[@]+"${AIFEYNMAN_ARGS[@]}"} ${PHYE2E_ARGS[@]+"${PHYE2E_ARGS[@]}"} \
    ${E2E_TPSR_ARGS[@]+"${E2E_TPSR_ARGS[@]}"} --include-devncon \
    "${ONESHOT_ARGS[@]}" "${PYSR_ARGS[@]}" "${LLMSR_ARGS[@]}" \
    --exclude $LSRT_DROP ${CURVE_DROP[@]+"${CURVE_DROP[@]}"} "${LSRT_DROP_ARR[@]}"

# ---- verify: the flags are not evidence, the legend is ----------------------------
# One legend now serves both rows, so an arm drawn on EITHER benchmark shows up here.
# Match the PRINTED names (plot_io.LEGEND_DISPLAY), not the internal labels: the
# legend says "one-shot Llama-8B" / "LLM-SR (Qwen-32B)", so the old
# "(OneShot|LLMSR)-<x>" pattern matched nothing and the check silently passed.
echo "=== arms drawn per appendix figure ==="
for f in appx_curves_llm appx_time_complexity_llm; do
  printf "%-28s " "$f"
  pdftotext "$OUT_DIR/$f.pdf" - 2>/dev/null | tr -s '\n' ' ' \
    | grep -oiE "one-shot [A-Za-z0-9.-]+|LLM-SR \([A-Za-z0-9.-]+\)" | sort -u | tr '\n' ' '
  echo
done
echo "=== LLM appendix replot done $(date -Is) ==="
