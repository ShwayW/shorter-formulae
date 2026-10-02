#!/bin/bash
# run_all_evals.sh — Beam-search evaluation of our 5 canonical methods on the
# SRBench (Feynman) and LLM-SRBench benchmarks, saving predicted formulae so the
# plotting scripts (compare_srbench.py, plot_ood_vs_gap.py,
# plot_llmsrbench_ood_vs_gap.py) can read them.
#
# The 5 methods come from 3 checkpoints in checkpoints/:
#   89M            = 89M_40_simp1               (89M weights, 4-constant grammar; raw inputs, default)
#   145M           = 145M_40_simp1  (--sample-size 400 --n-points 534; trained on up to 400 IO pts)
#
# Outputs:
#   SRBench    → results/eval_tf_<model>[_unscale].pkl.gz  (via eval_mymodels.py)
#   LLM-SRBench→ results/llmsrbench/<model>_<split>[_unscale]/results.jsonl (via eval_mymodels.py --benchmark llmsrbench)
#
# Usage:
#   ./run_all_evals.sh [all|srbench|llmsr]     # which benchmark(s); default: all
#
# Env knobs (with defaults):
#   DEVICE=cuda   NBAGS=100   BEAM=10   SPLIT=lsr_transform   M145_SAMPLE=400
#
# Notes:
#   * OMP_NUM_THREADS=1 — extEvalPN.so has an uncapped OpenMP loop that otherwise
#     spawns one thread per core inside every worker (see train_transformer.sh).
#   * On small GPUs, lower NBAGS (e.g. NBAGS=25) and/or BEAM if 145M_40_simp1 OOMs at 400-pt bags.
set -u

WHICH="${1:-all}"
DEVICE="${DEVICE:-cuda}"
NBAGS="${NBAGS:-100}"
BEAM="${BEAM:-10}"
SPLIT="${SPLIT:-lsr_transform}"
M145_SAMPLE="${M145_SAMPLE:-400}"
M145_POINTS=$(python3 -c "import math; print(math.ceil(${M145_SAMPLE}/0.75))")

export OMP_NUM_THREADS=1
LOGDIR="out/eval_logs"
mkdir -p "$LOGDIR" results

# method label → eval_mymodels.py extra args (beyond --model/--device/--n-bags/--beam-size)
run_srbench () {
  local name="$1"; shift
  echo "=========================================================================="
  echo "[SRBench] $name  ::  eval_mymodels.py $*"
  echo "=========================================================================="
  OMP_NUM_THREADS=1 python eval_mymodels.py --device "$DEVICE" --n-bags "$NBAGS" \
      --beam-size "$BEAM" --results-dir results/mymodels "$@" 2>&1 \
      | tee "$LOGDIR/srbench_${name}.log"
}

run_llmsr () {
  local name="$1"; shift
  echo "=========================================================================="
  echo "[LLM-SRBench] $name  ::  eval_mymodels.py --benchmark llmsrbench --split $SPLIT $*"
  echo "=========================================================================="
  OMP_NUM_THREADS=1 python eval_mymodels.py --benchmark llmsrbench --device "$DEVICE" --split "$SPLIT" \
      --n-bags "$NBAGS" --beam-size "$BEAM" --output results/mymodels "$@" 2>&1 \
      | tee "$LOGDIR/llmsr_${name}.log"
}

if [[ "$WHICH" == "all" || "$WHICH" == "srbench" ]]; then
  run_srbench "89M"           --model 89M_40_simp1
  run_srbench "145M"          --model 145M_40_simp1 --sample-size "$M145_SAMPLE" --n-points "$M145_POINTS"
fi

if [[ "$WHICH" == "all" || "$WHICH" == "llmsr" ]]; then
  run_llmsr "89M"           --model 89M_40_simp1
  run_llmsr "145M"          --model 145M_40_simp1
fi

# The eval scripts write all seeds into results/mymodels/noise_<τ>/… directly,
# which the plotting scripts (compare_srbench.py, plot_ood_vs_gap.py, …) read as-is.
echo "All requested evaluations finished. Logs in $LOGDIR/."
