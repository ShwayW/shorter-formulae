#!/bin/bash
# Pretraining job for train_transformer.py.
#
# USAGE
#   sbatch --job-name=<name> train_transformer.sh <RUN_NAME> [USE_PREFACTORS] [RESTRICT_CONSTS]
#
#   e.g. the two arms of the prefactor ablation, submitted from the SAME checkout:
#     sbatch --job-name=prefac_on  train_transformer.sh prefac_on  1
#     sbatch --job-name=prefac_off train_transformer.sh prefac_off 0
#
# ISOLATION -- what keeps two concurrent jobs out of each other's way:
#   * RUN_NAME sends every artifact train_transformer.py writes (per-epoch
#     checkpoint, final fixed_weights.pth, training-curve pickle) to
#     checkpoints/<RUN_NAME>_res/.  That is the "<name>_res" layout
#     eval_mymodels.py --model expects, so each arm is evaluable by name.
#   * --output below carries %j (and %x), so the two jobs do not share a log.
#   * MPLCONFIGDIR is per-job, so the two jobs do not race on matplotlib's cache.
#   * Nothing else in the training path writes to the checkout: the .so files and
#     any BPE/supernet checkpoints are read-only, and __pycache__ writes are
#     atomic (tmp file + rename) so concurrent imports are safe.
#   Each job still needs its OWN GPU; both arms request --gres=gpu:1 separately.
#
#   Set RESUME=1 to continue an arm from its own checkpoint after a timeout:
#     sbatch --job-name=prefac_on --export=ALL,RESUME=1 train_transformer.sh prefac_on 1
#
#SBATCH --account=<your-slurm-account>
#SBATCH --time=1-00:00          # Runtime (DD-HH:MM) -- one CHUNK, not the whole run.
                                # Shorter walltimes usually queue much faster; chain
                                # 1-day chunks with RESUME=1 instead of one long job.
# NO --mem DIRECTIVE HERE: clusters differ (some need an explicit value, others hand out
# the node's per-GPU share automatically). Add e.g. `--mem=32G` on the sbatch line if needed.
# Conservative fallback ONLY -- a command-line --cpus-per-task beats this directive.
# train_transformer.py sizes num_data_workers off SLURM_CPUS_PER_TASK, so whatever lands
# here also sets the worker count.
#SBATCH --cpus-per-task=3
# NO GPU DIRECTIVE HERE ON PURPOSE: clusters disagree on the form (some require a typed
# `--gres=gpu:h100:1`, others reject --gres and want `--gpus-per-node=1`), and a directive
# here would be additive with the command line. A direct sbatch MUST pass one itself, e.g.
#   sbatch --gres=gpu:h100:1 ... train_transformer.sh <RUN_NAME>
#   sbatch --gpus-per-node=1 ... train_transformer.sh <RUN_NAME>
#SBATCH --job-name=train_tf
#SBATCH --output=out/%x_%j.out  # %x for job name, %j for jobID -- never shared between jobs

# ---- run identity ---------------------------------------------------------
# Positional args are the PRIMARY form, not a convenience: some clusters wrap sbatch in a
# shell function that forces `--export=NONE --get-user-env`, so environment variables do
# not reliably reach the job there. Args always arrive, and this script re-exports them
# inside the job before python runs. The matching env vars still win when set, so an
# --export-style launch keeps working on clusters where that is reliable.
RUN_NAME="${RUN_NAME:-$1}"
USE_PREFACTORS="${USE_PREFACTORS:-${2:-0}}"
# Default 0 = UNRESTRICTED constants; see the note in submit_train_chain.sh.
RESTRICT_CONSTS="${RESTRICT_CONSTS:-${3:-0}}"
RESUME="${RESUME:-${4:-0}}"

if [ -z "$RUN_NAME" ]; then
    echo "ERROR: no RUN_NAME. Usage: sbatch --job-name=<name> $0 <RUN_NAME> [USE_PREFACTORS] [RESTRICT_CONSTS]" >&2
    exit 1
fi
case "$RUN_NAME" in
    */*|*' '*) echo "ERROR: RUN_NAME '$RUN_NAME' must be a plain directory name" >&2; exit 1 ;;
esac

RUN_DIR="checkpoints/${RUN_NAME}_res"
mkdir -p out "$RUN_DIR"

# Refuse to silently clobber a previous run of the same name.  RESUME=1 continues
# it on purpose; FORCE=1 overwrites on purpose.
CKPT="$RUN_DIR/model_fixed_Van_checkpoint.pth"
if [ -e "$CKPT" ] && [ "$RESUME" != "1" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: $CKPT already exists -- a run named '$RUN_NAME' has trained here." >&2
    echo "       Pass RESUME=1 to continue it, FORCE=1 to overwrite, or pick another RUN_NAME." >&2
    exit 1
fi

# A finished run writes fixed_weights.pth.  Later chunks of a chain must not restart
# training after an early stop, so bail out instead of burning the allocation.
if [ "$RESUME" = "1" ] && [ -e "$RUN_DIR/fixed_weights.pth" ]; then
    echo "run '$RUN_NAME' already finished ($RUN_DIR/fixed_weights.pth exists) -- nothing to do"
    exit 0
fi

export RUN_NAME USE_PREFACTORS RESTRICT_CONSTS RESUME

# Python module and venv are variables, not hardcoded, because they differ per cluster.
#   PYTHON_MODULE -- this used to be pinned to python/3.13.3, which does NOT exist on
#     every cluster (some list only 3.13.2).  `module load` of a missing
#     version does not abort the script, so the job silently proceeded on a broken
#     interpreter stack instead of failing at submission.  Verify with
#     `module -t avail python/3.13` on the target cluster before overriding.
#   VENV -- every cluster currently uses `env`; the variable exists so a cluster can point
#     at an alternate venv without editing this file:
#       sbatch --export=ALL,VENV=some_other_env train_transformer.sh <RUN_NAME>
PYTHON_MODULE="${PYTHON_MODULE:-python/3.13.2}"
VENV="${VENV:-env}"

module load StdEnv/2023
module load "$PYTHON_MODULE" scipy-stack
module load gcc

if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: venv '$VENV' not found in $(pwd) -- set VENV or run slurm/setup.sh first" >&2
    exit 1
fi
source "$VENV/bin/activate"
python -c 'import sys, torch; print(f"python {sys.version.split()[0]}, torch {torch.__version__}")' \
    || { echo "ERROR: venv '$VENV' cannot import torch" >&2; exit 1; }

# Pin OpenMP to a single thread. extEvalPN.so (the C++ PN evaluator) has a
# `#pragma omp parallel for` that, left uncapped, spawns one thread per node core
# (~80 on the compute nodes) inside EVERY DataLoader worker — hundreds of threads
# fighting over the 3 allocated CPUs, plus their stacks. The eval is over <=200
# rows per call, so per-call OpenMP is pure overhead anyway; real parallelism
# comes from num_data_workers. Training compute is on the GPU.
export OMP_NUM_THREADS=1

# Reduce allocator fragmentation. The prefactor arm OOMed on an 80GB H100 with
# 3.5 GiB "reserved but unallocated" -- expandable_segments lets the caching
# allocator grow a segment instead of stranding it. Allocator behaviour only; it
# does not change any computation, so it is safe to differ between chunks.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Per-job matplotlib cache: the default ~/.config/matplotlib is shared, and two
# jobs building it at the same time can race.
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mplconfig-${SLURM_JOB_ID:-$$}"
mkdir -p "$MPLCONFIGDIR"

# ---- preflight: is this GPU actually usable? -------------------------------
# On a shared node a leaked process from someone else's job can be sitting on most
# of the card. The symptom is not a slow run, it is an instant
# "CUDA error: out of memory" as the model moves to the device -- and because the
# chain is wired with --dependency=afterany, the NEXT chunk then starts and can land
# on the same bad node, so one sick GPU can eat an entire chain. Requeue instead of burning
# the chunk: the job keeps its id and dependencies and gets another shot at a node.
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$free_mib" ] && [ "$free_mib" -lt "${MIN_FREE_GPU_MIB:-40000}" ]; then
    echo "ERROR: only ${free_mib} MiB free on $(hostname) -- the GPU is already occupied." >&2
    nvidia-smi >&2
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "${NO_REQUEUE:-0}" != "1" ]; then
        echo "Requeueing job $SLURM_JOB_ID to try a different node." >&2
        scontrol requeue "$SLURM_JOB_ID" && sleep 30
    fi
    exit 1
fi
echo "preflight: ${free_mib:-unknown} MiB free on $(hostname)"

echo "=== run '$RUN_NAME' -> $RUN_DIR | use_prefactors=$USE_PREFACTORS restrict_consts=$RESTRICT_CONSTS resume=$RESUME ==="
nvidia-smi
python train_transformer.py
