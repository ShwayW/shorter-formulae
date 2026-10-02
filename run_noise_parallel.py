#!/usr/bin/env python3
"""
run_noise_parallel.py -- Parallel driver for the target-noise sweep.

The single-process eval is CPU-bound (BFGS + PN formula eval) with the GPU nearly
idle (~950 MiB / 8 GB, ~0% util for an 89M model; 145M peaks ~1400 MiB).  So we run
several independent (model, noise, benchmark) jobs concurrently: each writes to its
OWN output file (no races), the transformer jobs self-resume (already-done
(dataset,seed) pairs are skipped), and per-job .done markers make the whole sweep
resumable.  Concurrency is capped by a GPU-memory budget AND a process count.

Covers the 5 transformer models + the (now additive-RMS) e2e baseline, on both
SRBench and LLM-SRBench, at tau in {0.001, 0.01, 0.1}.  When every job of a given tau
finishes it auto-runs make_noise_plots.sh for that tau if that script is present
(plotting scripts are not part of this release, so this is a no-op here).

Usage:   python run_noise_parallel.py [--max-procs 6] [--mem-budget 7000]
                                       [--noises 0.001 0.01 0.1] [--which all|srbench|llmsr]
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "out", "eval_logs")
MARKDIR = os.path.join(LOGDIR, "parallel_markers")
MASTER = os.path.join(LOGDIR, "noise_parallel_master.log")

NBAGS = 100
BEAM = 10
M145_SAMPLE = 400
M145_POINTS = 534         # ceil(400 / 0.75)
SPLIT = "lsr_transform"
DEVICE = "cuda"

# GPU-memory estimates (MiB) from OBSERVED peaks under load.  An OOM cascade on
# 2026-07-02 proved the old estimates were far too low: e2e SAMPLING actually uses
# ~3.24 GiB (not ~1.5), and the GPU's usable capacity is only ~7.6 GiB (display /
# system holds the rest).  mem_budget below must stay well under 7.6 GiB.
MEM = {"89M": 1100, "145M_srbench": 1700, "145M_llmsr": 1450, "e2e": 3400}

# Real GPU free-memory gate -- the true OOM backstop over the (fallible) estimates.
# Before launching, query nvidia-smi and require measured free >= task_mem + MARGIN;
# launch ONE job per cycle then wait SETTLE so the allocation registers before the
# next decision (prevents the ramp-up races that caused the 2026-07-02 OOM cascade).
GPU_SAFETY_MARGIN = 900   # MiB of measured free headroom to always keep spare
LAUNCH_SETTLE     = 9     # s to wait after a launch for its memory to allocate


def _gpu_free_mib() -> int:
    """Measured free GPU memory (MiB) via nvidia-smi. Returns -1 on failure, which
    the caller treats as 'skip the real-memory gate this cycle' -- the estimate
    budget still governs, so a transient query hiccup can't stall the whole sweep."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def tf_srbench(tau):
    base = ["python", "eval_mymodels.py", "--device", DEVICE, "--n-bags", str(NBAGS),
            "--beam-size", str(BEAM), "--target-noise", str(tau),
            "--results-dir", "results/mymodels"]
    nv = ["--sample-size", str(M145_SAMPLE), "--n-points", str(M145_POINTS)]
    return [
        ("89M",           base + ["--model", "89M_40_simp1"], MEM["89M"]),
        ("145M",          base + ["--model", "145M_40_simp1"] + nv, MEM["145M_srbench"]),
        ("e2e",           ["python", "eval_e2e.py", "--noise", str(tau),
                           "--beam_type", "sampling",
                           "--results-dir", "results/e2e"],
                          MEM["e2e"]),
    ]


def tf_llmsr(tau):
    base = ["python", "eval_mymodels.py", "--benchmark", "llmsrbench", "--device", DEVICE, "--split", SPLIT,
            "--n-bags", str(NBAGS), "--beam-size", str(BEAM), "--target-noise", str(tau),
            "--output", "results/mymodels"]
    return [
        ("89M",           base + ["--model", "89M_40_simp1"], MEM["89M"]),
        ("145M",          base + ["--model", "145M_40_simp1"], MEM["145M_llmsr"]),
        ("e2e",           ["python", "eval_e2e.py", "--benchmark", "llmsrbench", "--device", DEVICE,
                           "--split", SPLIT, "--beam-type", "sampling",
                           "--output", "results/e2e",
                           "--target-noise", str(tau)], MEM["e2e"]),
    ]


def build_tasks(noises, which):
    tasks = []
    for tau in noises:
        groups = []
        if which in ("all", "srbench"):
            groups.append(("srbench", tf_srbench(tau)))
        if which in ("all", "llmsr"):
            groups.append(("llmsr", tf_llmsr(tau)))
        for bench, items in groups:
            for label, cmd, mem in items:
                tasks.append({
                    "tau": tau, "bench": bench, "label": label, "cmd": cmd, "mem": mem,
                    "log": os.path.join(LOGDIR, f"{bench}_{label}_noise{tau:g}.log"),
                    "mark": os.path.join(MARKDIR, f"{bench}_{label}_noise{tau:g}.done"),
                })
    return tasks


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(MASTER, "a") as fh:
        fh.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-procs", type=int, default=5)
    ap.add_argument("--mem-budget", type=int, default=6800,
                    help="Estimate-based concurrent GPU MiB cap. The real backstop is "
                         "the measured-free gate (task_mem + GPU_SAFETY_MARGIN), so this "
                         "stays under the ~7.6 GiB usable while allowing e2e + 3 light "
                         "(3400 + 3*1100 = 6700). OOM-safe via the live nvidia-smi check.")
    ap.add_argument("--noises", nargs="+", type=float, default=[0.001, 0.01, 0.1])
    ap.add_argument("--which", choices=["all", "srbench", "llmsr"], default="all")
    ap.add_argument("--no-e2e", action="store_true",
                    help="Skip the e2e baseline jobs. Noisy e2e is ~7.5x slower than "
                         "noise-free (~15h/job), so it dominates the sweep; dropping it "
                         "leaves the 5 transformer models + noise-matched SRBench/TPSR.")
    ap.add_argument("--no-plot", action="store_true", help="Skip auto per-tau plotting.")
    args = ap.parse_args()

    os.makedirs(MARKDIR, exist_ok=True)
    # expandable_segments:True lets PyTorch grow/shrink its allocator arena so
    # transient spikes across concurrent processes fragment less -> fewer OOMs
    # (the CUDA OOM error explicitly recommends it).
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

    tasks = build_tasks(args.noises, args.which)
    if args.no_e2e:
        tasks = [t for t in tasks if t["label"] != "e2e"]
    pending = [t for t in tasks if not os.path.exists(t["mark"])]
    done = [t for t in tasks if os.path.exists(t["mark"])]
    log(f"=== parallel noise sweep: {len(tasks)} jobs "
        f"({len(done)} already done) | max_procs={args.max_procs} "
        f"mem_budget={args.mem_budget} MiB ===")

    # Per-tau eval-job accounting so we can trigger plotting when a tau fully finishes.
    tau_total, tau_done = {}, {}
    for t in tasks:
        tau_total[t["tau"]] = tau_total.get(t["tau"], 0) + 1
    for t in done:
        tau_done[t["tau"]] = tau_done.get(t["tau"], 0) + 1
    plotted = set()
    plot_procs = []

    def maybe_plot(tau):
        if args.no_plot or tau in plotted or not os.path.exists(os.path.join(HERE, "make_noise_plots.sh")):
            return
        if tau_done.get(tau, 0) >= tau_total.get(tau, 0):
            plotted.add(tau)
            plog = os.path.join(LOGDIR, f"noise_plots_{tau:g}.log")
            log(f"tau={tau:g} complete -> generating plots (-> {os.path.basename(plog)})")
            # The eval scripts write all seeds into results/{mymodels,e2e}/noise_<tau>/
            # directly, which the plot scripts read as-is -- no staging step needed.
            pf = open(plog, "w")
            p = subprocess.Popen(["bash", "make_noise_plots.sh"], cwd=HERE,
                                 env=dict(env, NOISES=f"{tau:g}"),
                                 stdout=pf, stderr=subprocess.STDOUT)
            plot_procs.append((p, pf))

    # Trigger plots for tau levels already fully done before this (re)start.
    for tau in args.noises:
        maybe_plot(tau)

    running = []   # list of (task, Popen, logfile_handle)
    used_mem = 0

    _stall_note = 0.0
    while pending or running:
        # 1) Reap finished jobs FIRST so their memory frees before we launch more.
        still = []
        for t, p, fh in running:
            rc = p.poll()
            if rc is None:
                still.append((t, p, fh))
                continue
            fh.close()
            used_mem -= t["mem"]
            if rc == 0:
                open(t["mark"], "w").close()
                tau_done[t["tau"]] = tau_done.get(t["tau"], 0) + 1
                log(f"DONE    {t['bench']:8s} {t['label']:14s} tau={t['tau']:g}  (rc=0)")
                maybe_plot(t["tau"])
            else:
                t["tries"] = t.get("tries", 0) + 1
                if t["tries"] < 3:
                    pending.append(t)   # re-queue at the end (memory frees first);
                                        # all job types now resume, so a retry is cheap.
                    log(f"FAIL    {t['bench']:8s} {t['label']:14s} tau={t['tau']:g}  "
                        f"(rc={rc}, try {t['tries']}/3 -> re-queued). See {os.path.basename(t['log'])}")
                else:
                    log(f"GIVEUP  {t['bench']:8s} {t['label']:14s} tau={t['tau']:g}  "
                        f"(rc={rc} after {t['tries']} tries). See {os.path.basename(t['log'])}")
        running = still

        # 2) Launch at most ONE job, gated by BOTH the estimate budget AND the
        #    measured free GPU memory (task_mem + MARGIN), then settle so the
        #    allocation registers before the next launch decision.
        launched = False
        if pending and len(running) < args.max_procs:
            free = _gpu_free_mib()
            for idx, t in enumerate(pending):
                est_ok  = used_mem + t["mem"] <= args.mem_budget
                real_ok = (free < 0) or (free - GPU_SAFETY_MARGIN >= t["mem"])
                if est_ok and real_ok:
                    fh = open(t["log"], "a")
                    fh.write(f"\n===== launch {t['bench']} {t['label']} tau={t['tau']:g} "
                             f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                    fh.flush()
                    p = subprocess.Popen(t["cmd"], cwd=HERE, env=env,
                                         stdout=fh, stderr=subprocess.STDOUT)
                    running.append((t, p, fh))
                    used_mem += t["mem"]
                    pending.pop(idx)
                    log(f"launch  {t['bench']:8s} {t['label']:14s} tau={t['tau']:g}  "
                        f"(running={len(running)}, est~{used_mem} MiB, free={free} MiB, "
                        f"pending={len(pending)})")
                    launched = True
                    break
            # If nothing fits with the GPU otherwise idle, note it (rate-limited) so a
            # genuine wait (external GPU use / job bigger than free) is visible, not silent.
            if not launched and not running and time.time() - _stall_note > 120:
                log(f"[wait] no job fits: free={free} MiB, next="
                    f"{pending[0]['label']}({pending[0]['mem']} MiB)+{GPU_SAFETY_MARGIN} margin")
                _stall_note = time.time()

        time.sleep(LAUNCH_SETTLE if launched else 5)

    for p, pf in plot_procs:
        p.wait(); pf.close()
    log("=== ALL noise jobs finished. ===")


if __name__ == "__main__":
    main()
