#!/usr/bin/env bash
#
# run_llmsr.sh -- shard-parallel LLMSR+Gemini evaluation, then merge, per split.
# Supports splitting one job across multiple machines via --shard-start/--shard-count.
#
# For each split it launches a range of shard processes (each on a disjoint slice of
# the problems, writing its own artifact), waits for them, and -- if this machine ran
# ALL shards -- merges them into the canonical results file. When the shards are split
# across machines, each machine runs its own range and does NOT merge; you rsync every
# machine's results/ onto one host and run --merge-only there.
#
# Usage:
#   run_llmsr.sh --config <yaml> --num-shards <N> [options]
#
# Options:
#   --config PATH     searcher config yaml (required)
#   --num-shards N    TOTAL shards across ALL machines (required)
#   --split NAME      split to run; repeatable. Default: lsr_transform feynman
#   --shard-start K   first shard-id to launch on THIS machine (default 0)
#   --shard-count C   how many shard-ids to launch here (default: N - shard-start)
#   --merge-only      do not launch; just merge shard artifacts already on local disk
#   --extra "ARGS"    extra flags forwarded to eval_llmsr.py
#                     (e.g. "--global-max-sample-num 1000 --target-noise 0.0")
#   -h, --help        show this help
#
# Single machine (launch all shards, then auto-merge):
#   ./run_llmsr.sh --config configs/llmsr_gemini35flash.yaml --num-shards 16
#
# Two machines, both splits balanced across 32 shards:
#   # Machine A:
#   ./run_llmsr.sh --config configs/llmsr_gemini35flash.yaml --num-shards 32 --shard-start 0  --shard-count 16
#   # Machine B:
#   ./run_llmsr.sh --config configs/llmsr_gemini35flash.yaml --num-shards 32 --shard-start 16 --shard-count 16
#   # then rsync BOTH machines' results/ onto one host and merge (pass the SAME --extra):
#   ./run_llmsr.sh --config configs/llmsr_gemini35flash.yaml --num-shards 32 --merge-only
#
# Resume is automatic and per-(equation_id, seed): re-run the same command to continue.
set -eo pipefail

usage() { sed -n '2,/^set -eo/p' "$0" | sed 's/^# \{0,1\}//; s/^#//'; }

CONFIG=""; N=""; SHARD_START=0; SHARD_COUNT=""; MERGE_ONLY=0; EXTRA_STR=""
SPLITS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --config)      CONFIG="$2"; shift 2;;
        --num-shards)  N="$2"; shift 2;;
        --split)       SPLITS+=("$2"); shift 2;;
        --shard-start) SHARD_START="$2"; shift 2;;
        --shard-count) SHARD_COUNT="$2"; shift 2;;
        --merge-only)  MERGE_ONLY=1; shift;;
        --extra)       EXTRA_STR="$2"; shift 2;;
        -h|--help)     usage; exit 0;;
        *) echo "unknown arg: $1" >&2; echo "run with --help for usage" >&2; exit 1;;
    esac
done

[ -n "$CONFIG" ] || { echo "ERROR: --config required (see --help)" >&2; exit 1; }
[ -n "$N" ]      || { echo "ERROR: --num-shards required (see --help)" >&2; exit 1; }
if [ "${#SPLITS[@]}" -eq 0 ]; then SPLITS=(lsr_transform feynman); fi
if [ -z "$SHARD_COUNT" ]; then SHARD_COUNT=$((N - SHARD_START)); fi
read -r -a EXTRA <<< "$EXTRA_STR"

SHARD_END=$((SHARD_START + SHARD_COUNT))   # exclusive
if [ "$SHARD_START" -lt 0 ] || [ "$SHARD_END" -gt "$N" ] || [ "$SHARD_COUNT" -le 0 ]; then
    echo "ERROR: shard range [$SHARD_START,$SHARD_END) invalid for --num-shards $N" >&2
    exit 1
fi
HAS_ALL=0
if [ "$SHARD_START" -eq 0 ] && [ "$SHARD_COUNT" -eq "$N" ]; then HAS_ALL=1; fi

merge_split() {
    local SPLIT="$1"
    echo "=== $SPLIT: merging $N shards ==="
    python eval_llmsr.py --searcher_config "$CONFIG" --split "$SPLIT" \
        --num-shards "$N" --merge-shards "${EXTRA[@]}"
}

if [ "$MERGE_ONLY" -eq 1 ]; then
    for SPLIT in "${SPLITS[@]}"; do merge_split "$SPLIT"; done
    echo "Merge-only complete."
    exit 0
fi

# Raise the open-file limit before launching. Heavy sharding (N shards x
# samples_per_prompt async HTTPS calls) plus per-evaluation sandbox subprocesses and
# genai/oauth sockets exhaust the default soft limit (~1024), triggering "Too many
# open files" retry storms that silently 10x the wall time. Raise the SOFT limit
# toward the hard cap (as high as allowed without root); children inherit it.
WANT_NOFILE="${LLMSR_NOFILE:-1048576}"
HARD_NOFILE="$(ulimit -Hn 2>/dev/null || echo unlimited)"
if [ "$HARD_NOFILE" = "unlimited" ]; then
    TARGET_NOFILE="$WANT_NOFILE"
elif [ "$WANT_NOFILE" -gt "$HARD_NOFILE" ] 2>/dev/null; then
    TARGET_NOFILE="$HARD_NOFILE"      # can't exceed the hard cap without root
else
    TARGET_NOFILE="$WANT_NOFILE"
fi
ulimit -n "$TARGET_NOFILE" 2>/dev/null || true
CUR_NOFILE="$(ulimit -n)"
echo "open-file limit (ulimit -n): $CUR_NOFILE  (hard cap: $HARD_NOFILE)"
if [ "$CUR_NOFILE" != "unlimited" ] && [ "$CUR_NOFILE" -lt 8192 ] 2>/dev/null; then
    echo "WARNING: open-file limit is still low ($CUR_NOFILE). If you see 'Too many open"
    echo "         files' in the shard logs, raise the hard cap (e.g. edit /etc/security/"
    echo "         limits.conf or run as a user allowed a higher 'ulimit -Hn')."
fi

echo "config=$CONFIG  num_shards=$N  this-machine shard-ids [$SHARD_START,$SHARD_END)  splits=${SPLITS[*]}  extra='$EXTRA_STR'"

for SPLIT in "${SPLITS[@]}"; do
    LOGDIR="logs/run_llmsr/${SPLIT}"
    mkdir -p "$LOGDIR"
    echo "=== $SPLIT: launching shards $SHARD_START..$((SHARD_END - 1)) of $N ==="
    pids=()
    for i in $(seq "$SHARD_START" $((SHARD_END - 1))); do
        python eval_llmsr.py --searcher_config "$CONFIG" --split "$SPLIT" \
            --num-shards "$N" --shard-id "$i" "${EXTRA[@]}" \
            > "$LOGDIR/shard_${i}.log" 2>&1 &
        pids+=($!)
        echo "  shard $i/$N -> PID ${pids[-1]}  (log: $LOGDIR/shard_${i}.log)"
    done

    fail=0
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
    if [ "$fail" -ne 0 ]; then
        echo "!! $SPLIT: a shard exited non-zero; NOT merging."
        echo "   Inspect $LOGDIR/*.log and re-run the same command (resume is automatic)."
        exit 1
    fi

    if [ "$HAS_ALL" -eq 1 ]; then
        merge_split "$SPLIT"
    else
        echo "=== $SPLIT: shards [$SHARD_START,$SHARD_END) done on this machine (a SUBSET of $N)."
        echo "    NOT merging. When every machine's range is done, rsync all results/ onto one"
        echo "    host and run (with the SAME --extra):"
        echo "      ./run_llmsr.sh --config $CONFIG --num-shards $N --merge-only --split $SPLIT"
    fi
done
echo "Done."
