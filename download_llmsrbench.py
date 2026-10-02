#!/usr/bin/env python3
"""
download_llmsrbench.py -- Download the LLM-SRBench benchmark to a LOCAL directory
so evaluation needs no internet / Hugging Face cache at runtime.

Run this once on a machine with internet, then `scp`/`rsync` the resulting
`datasets/llmsrbench/` directory to the compute cluster. The evaluators
(`eval_mymodels.py --benchmark llmsrbench`, `eval_e2e.py --benchmark llmsrbench`, `eval_e2e_tpsr.py --benchmark llmsrbench`)
read from `$LLMSRBENCH_DIR` (default `datasets/llmsrbench/`) first and only fall
back to a live Hugging Face download if the local copy is missing.

Downloads just the two artifacts the loaders use:
  * lsr_bench_data.hdf5                    -- the actual samples for every problem
  * data/<split>-00000-of-00001.parquet   -- per-split metadata (name/symbols/expr)

Usage:
    python download_llmsrbench.py                        # -> datasets/llmsrbench/
    python download_llmsrbench.py --output-dir /path/to/llmsrbench
    python download_llmsrbench.py --all-files            # whole dataset repo

Requires `huggingface_hub` (`pip install huggingface_hub`). If the dataset is
gated, run `huggingface-cli login` first.
"""
import argparse
import os

REPO_ID = "nnheui/llm-srbench"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "datasets", "llmsrbench")


def _dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                    help=f"Where to put the benchmark (default: {DEFAULT_OUTPUT}).")
    ap.add_argument("--repo-id", default=REPO_ID,
                    help=f"Hugging Face dataset repo id (default: {REPO_ID}).")
    ap.add_argument("--all-files", action="store_true",
                    help="Download the entire dataset repo, not just the hdf5 + parquet "
                         "files the evaluators need.")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    os.makedirs(args.output_dir, exist_ok=True)
    allow = None if args.all_files else ["lsr_bench_data.hdf5", "data/*.parquet"]

    print(f"Downloading {args.repo_id} -> {args.output_dir}")
    print(f"  patterns: {'ALL files' if allow is None else allow}", flush=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.output_dir,
        allow_patterns=allow,
    )

    print(f"\nDone. {_dir_size_mb(args.output_dir):.1f} MB in {args.output_dir}")
    hdf5 = os.path.join(args.output_dir, "lsr_bench_data.hdf5")
    if not os.path.exists(hdf5):
        print(f"[warn] Expected {hdf5} not found -- did the pattern match? "
              f"Try --all-files.")
    print("\nNext steps:")
    print(f"  1. scp/rsync '{args.output_dir}' to each cluster's repo at datasets/llmsrbench/")
    print("  2. The evaluators pick it up automatically (or set $LLMSRBENCH_DIR).")


if __name__ == "__main__":
    main()
