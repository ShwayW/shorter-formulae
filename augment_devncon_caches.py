#!/usr/bin/env python3
"""augment_devncon_caches.py -- Append devncon OOD-R^2 and complexity to plotting caches.

One-time step (idempotent): compute OOD R^2 for 89M+D&C and 145M+D&C on all
distribution-shift gaps, and refit complexity; write the rows to the existing
ood_raw_noise*.csv and complexity_noise*.csv caches (create if absent).

The cache rows enable plot_ood_vs_gap.py / plot_time_complexity_vs_noise.py to
include devncon methods whenever --include-devncon is passed.

Usage:  python augment_devncon_caches.py
"""

import os
import sys
import glob
import gzip
import pickle
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import results_io
import devncon_addon
from compare_srbench import compute_ood_r2, SRBENCH_GT
NOISES = [0.0, 0.001, 0.01, 0.1]
GAPS = [0, 1, 2, 4, 8, 16, 32, 64, 128]


def augment_ood_caches():
    """Append devncon OOD rows to ood_raw_noise*.csv."""
    root = os.path.join(SCRIPT_DIR, "results")

    for noise in NOISES:
        devncon_pkls = devncon_addon.srbench_devncon_pkls(root, noise)
        if not devncon_pkls:
            continue

        # Create tf_file_labels for compute_ood_r2
        tf_file_labels = [(devncon_addon.display_label(
            os.path.basename(p).replace("eval_tf_", "").replace(".pkl.gz", "")), p)
                          for p in devncon_pkls]

        print(f"[SRBench] noise {noise}")
        ood_raw = []
        for gap in GAPS:
            ood_path = os.path.join(SCRIPT_DIR, f"datasets/feynman_ood_g{gap}.pkl.gz")
            if not os.path.exists(ood_path):
                continue

            ood_df, _ = compute_ood_r2(
                ood_path, tf_file_labels, SRBENCH_GT, include_gt=False, noise=noise,
                all_seeds=True
            )
            if not ood_df.empty:
                ood_df["gap"] = gap
                ood_raw.append(ood_df)
            print(f"  [gap={gap:3d}] SRBench OOD done")

        if ood_raw:
            raw_df = pd.concat(ood_raw, ignore_index=True)
            cache_path = os.path.join(root, f"ood_raw_noise{noise:g}.csv")
            if os.path.exists(cache_path):
                existing = pd.read_csv(cache_path)
                # REPLACE, don't append.  A plain concat stacks a second copy of the
                # D&C rows on every re-run (e.g. re-scoring a fresh devncon group),
                # and the plots then average the old holed run with the new one.  Only
                # the algorithms we just recomputed are dropped -- every other method's
                # rows are untouched, since those are expensive and not regenerated here.
                labels = set(raw_df["algorithm"].unique())
                stale = existing["algorithm"].isin(labels).sum()
                if stale:
                    print(f"  dropping {stale} stale row(s) for {sorted(labels)}")
                    existing = existing[~existing["algorithm"].isin(labels)]
                raw_df = pd.concat([existing, raw_df], ignore_index=True)
            raw_df.to_csv(cache_path, index=False)
            print(f"  wrote {os.path.basename(cache_path)}: +{len([r for r in ood_raw if not r.empty])} gap sections")


if __name__ == "__main__":
    augment_ood_caches()
    print("Devncon OOD augmentation complete.")
