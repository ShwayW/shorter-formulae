#!/usr/bin/env bash
# One-shot LLM baseline (Gemini 3.5 Flash, skeleton+BFGS arm) on both benchmarks at the
# three non-zero noise levels. Seed 0, one seed -- the same protocol as the existing
# noise-0 run in results/oneshot_gemini/noise_0/. ~1 LLM call per problem, so a cell is
# ~110 billed calls and a few minutes; run sequentially to stay well under rate limits.
set -u
cd "$(dirname "$0")"
PY=env/bin/python
CFG=configs/oneshot_fit_gemini35flash.yaml
OUT=results/oneshot_gemini
LOGS=logs/oneshot_gemini

for tau in 0.001 0.01 0.1; do
  for bench in srbench llmsrbench; do
    log="$LOGS/${bench}_noise${tau}.log"
    echo "=== $(date +%H:%M:%S) $bench tau=$tau -> $log"
    $PY eval_oneshot.py --searcher_config "$CFG" --benchmark "$bench" \
        --seed 0 --n-seeds 1 --target-noise "$tau" --output "$OUT" > "$log" 2>&1
    echo "    exit=$? $(grep -c 'R\^2=' "$log") problems"
  done
done
echo "=== $(date +%H:%M:%S) all done"
