#!/bin/bash
# run_noise_evals.sh — Noisy-target evaluation of our 5 canonical transformer
# methods on SRBench (Feynman) and LLM-SRBench, at the SRBench noise levels
# {0.001, 0.01, 0.1}.  This is the noise sweep companion to run_all_evals.sh
# (which is the noise-free τ=0 point).
#
# Noise convention (matches srbench/experiment/evaluate_model.py, TPSR/evaluate.py,
# and symbolicregression's environment):  the TRAINING targets get
#     y_train += N(0, τ · sqrt(mean(y_train²)))
# added before fitting; the held-out test (and every OOD set) stays CLEAN.  So a
# run at τ is directly comparable to the ground-truth feather rows at target_noise==τ
# and to the TPSR *_allnoise.csv rows at that τ.
#
# The 5 methods / 3 checkpoints are identical to run_all_evals.sh:
#   89M   145M
#
# Outputs (per noise level τ; file/dir labels are IDENTICAL to the noise-free run
# so every plotting-script label/colour remap works unchanged):
#   SRBench    → results/noise_<τ>/eval_tf_<model>[_unscale].pkl.gz
#   LLM-SRBench→ results/llmsrbench/noise_<τ>/<model>_<split>[_unscale]/results.jsonl
#
# Usage:
#   ./run_noise_evals.sh [all|srbench|llmsr]          # which benchmark(s); default: all
#
# Env knobs (with defaults — DEFAULTS MATCH the noise-free canonical run):
#   DEVICE=cuda   NBAGS=100   BEAM=10   SPLIT=lsr_transform   M145_SAMPLE=400
#   NOISES="0.001 0.01 0.1"                            # target-noise levels to sweep
#
# Notes:
#   * OMP_NUM_THREADS=1 — extEvalPN.so has an uncapped OpenMP loop (see run_all_evals.sh).
#   * Resumable: each (dataset,seed) pair already in results/noise_<τ>/… is skipped,
#     so re-running after an interruption continues where it stopped.  Do NOT change
#     NBAGS between resumes of the same τ (mixing bag counts corrupts the sweep).
#   * ~n·(per-model time): at NBAGS=100 each model is ~1.2–1.5 h/benchmark, so the
#     full 5×3×2 matrix is ~30–40 h.  Lower NBAGS (e.g. 25) for a faster, lower-power run.
set -u

WHICH="${1:-all}"
DEVICE="${DEVICE:-cuda}"
NBAGS="${NBAGS:-100}"
BEAM="${BEAM:-10}"
SPLIT="${SPLIT:-lsr_transform}"
M145_SAMPLE="${M145_SAMPLE:-400}"
M145_POINTS=$(python3 -c "import math; print(math.ceil(${M145_SAMPLE}/0.75))")
NOISES="${NOISES:-0.001 0.01 0.1}"

export OMP_NUM_THREADS=1
LOGDIR="out/eval_logs"
mkdir -p "$LOGDIR" results

run_srbench () {
  local name="$1"; local tau="$2"; shift 2
  echo "=========================================================================="
  echo "[SRBench τ=$tau] $name  ::  eval_mymodels.py --target-noise $tau $*"
  echo "=========================================================================="
  OMP_NUM_THREADS=1 python eval_mymodels.py --device "$DEVICE" --n-bags "$NBAGS" \
      --beam-size "$BEAM" --target-noise "$tau" --results-dir results/mymodels "$@" 2>&1 \
      | tee "$LOGDIR/srbench_${name}_noise${tau}.log"
}

run_llmsr () {
  local name="$1"; local tau="$2"; shift 2
  echo "=========================================================================="
  echo "[LLM-SRBench τ=$tau] $name  ::  eval_mymodels.py --benchmark llmsrbench --split $SPLIT --target-noise $tau $*"
  echo "=========================================================================="
  OMP_NUM_THREADS=1 python eval_mymodels.py --benchmark llmsrbench --device "$DEVICE" --split "$SPLIT" \
      --n-bags "$NBAGS" --beam-size "$BEAM" --target-noise "$tau" \
      --output results/mymodels "$@" 2>&1 \
      | tee "$LOGDIR/llmsr_${name}_noise${tau}.log"
}

for TAU in $NOISES; do
  if [[ "$WHICH" == "all" || "$WHICH" == "srbench" ]]; then
    run_srbench "89M"           "$TAU" --model 89M_40_simp1
    run_srbench "145M"          "$TAU" --model 145M_40_simp1 --sample-size "$M145_SAMPLE" --n-points "$M145_POINTS"
  fi

  if [[ "$WHICH" == "all" || "$WHICH" == "llmsr" ]]; then
    run_llmsr "89M"           "$TAU" --model 89M_40_simp1
    run_llmsr "145M"          "$TAU" --model 145M_40_simp1
  fi
done

# The eval scripts write all seeds into results/mymodels/noise_<τ>/… directly,
# which the plotting scripts read as-is.
echo "All requested noisy evaluations finished. Logs in $LOGDIR/."
