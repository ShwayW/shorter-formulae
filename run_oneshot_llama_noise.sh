#!/usr/bin/env bash
# One-shot LLM baseline (Llama-3.1-8B via local ollama) on both benchmarks at the three
# non-zero noise levels. Seed 0, one seed -- same protocol as the noise-0 run now in
# results/oneshot_llama/noise_0/, and the same sweep run for Gemini in
# run_oneshot_gemini_noise.sh. Nothing is billed; ollama serves one request at a time,
# so the cells run sequentially anyway.
#
# ARM selects which config: "fit" = skeleton + BFGS (default, mirrors the Gemini sweep),
# "direct" = the reply's constants are the answer. Both write distinct method names into
# the same output root, so the two arms never collide.
set -u
cd "$(dirname "$0")"
ARM="${ARM:-fit}"
PY=env/bin/python
case "$ARM" in
  fit)    CFG=configs/oneshot_fit_llama31_8b_ollama.yaml ;;
  direct) CFG=configs/oneshot_llama31_8b_ollama.yaml ;;
  *) echo "ARM must be fit|direct" >&2; exit 2 ;;
esac
OUT=results/oneshot_llama
LOGS=logs/oneshot_llama

for tau in 0.001 0.01 0.1; do
  for bench in srbench llmsrbench; do
    log="$LOGS/${ARM}_${bench}_noise${tau}.log"
    echo "=== $(date +%H:%M:%S) [$ARM] $bench tau=$tau -> $log"
    $PY eval_oneshot.py --searcher_config "$CFG" --benchmark "$bench" \
        --seed 0 --n-seeds 1 --target-noise "$tau" --output "$OUT" > "$log" 2>&1
    echo "    exit=$? $(grep -c 'R\^2=' "$log") problems"
  done
done
echo "=== $(date +%H:%M:%S) all done [$ARM]"
