"""
results_io.py -- the canonical on-disk layout for evaluation results, plus the
.pkl.gz reader/writer every eval script shares.

Layout (one tree per cluster job group, created by slurm/eval_*.sh):

    results/<group>/                       group in {mymodels, e2e, tpsr}
        run_info.json                      Slurm job id, seeds, noises, git rev
        seed42/
            noise_0/                       tau=0 is a real directory, not the root
                eval_tf_89M_40.pkl.gz              (eval_mymodels.py)
                results_feynman_e2e.pkl.gz         (eval_e2e.py)
                results_e2e_tpsr.pkl.gz            (eval_e2e_tpsr.py)
                llmsrbench/
                    m89_lsr_transform/results.pkl.gz
                    e2e_lsr_transform/results.pkl.gz
            noise_0.001/  noise_0.01/  noise_0.1/   ... same shape
        seed43/ ... seed51/

Every artifact is a gzipped pickle of {**metadata, "results": [row, ...]}, where a
row is a dict carrying at least a dataset key (`dataset` for SRBench, `equation_id`
for LLM-SRBench) and `seed`.  That uniformity is what lets merge_seed_results.py
and check_results.py treat all three groups identically.

Seeds are fixed (42..51) so runs are reproducible, comparable across the three
clusters, and resumable -- a resubmit finds its own seed dir and skips finished work.
The Slurm job id is recorded as metadata, never baked into a path.
"""
import glob
import gzip
import json
import os
import pickle

__all__ = [
    "noise_dir", "seed_dir", "srbench_path", "llmsr_dir", "llmsr_path",
    "save_results", "load_results", "load_rows_any", "write_run_info", "job_id",
    "GROUPS", "BASE_GROUPS", "group_noise_dir", "list_srbench_pkls",
    "list_base_srbench_pkls", "e2e_srbench_pkl",
    "list_llmsr_method_dirs", "find_llmsr_method_dir",
]

RESULTS_NAME = "results.pkl.gz"        # LLM-SRBench per-method artifact

# Job groups, one per cluster download.  A group holds all seeds for the models it
# ran: mymodels = the eval_mymodels transformer(s), e2e = the E2E sampler, tpsr = TPSR.
# mymodels_400 is a second transformer group, not a different kind of thing: the big
# models must be fed --sample-size 400 to match the IO bags they were trained on (the
# 89M models saw 200), so their eval writes there.  See slurm/eval_common.sh:57-61.
GROUPS = ("mymodels", "e2e", "tpsr", "mymodels_400", "mymodels_ftnoise", "mymodels_ftnoise_e80",
          "mymodels_ftnoise_48h")

# The groups holding OUR transformer checkpoints, as opposed to the baselines each
# other group carries.  list_base_srbench_pkls pools them so a figure's base-model
# discovery does not have to know which tree a checkpoint was evaluated in.
BASE_GROUPS = ("mymodels", "mymodels_400", "mymodels_ftnoise", "mymodels_ftnoise_e80",
               "mymodels_ftnoise_48h")


# -- path composition ---------------------------------------------------------

def noise_dir(root: str, tau: float) -> str:
    """<root>/noise_<tau>.  tau=0 gets a real 'noise_0' directory (not the bare root),
    so every seed dir holds exactly one subdirectory per noise level."""
    return os.path.join(root, f"noise_{float(tau):g}")


def seed_dir(root: str, seed: int) -> str:
    return os.path.join(root, f"seed{int(seed)}")


def srbench_path(root: str, tau: float, filename: str) -> str:
    """<root>/noise_<tau>/<filename> -- filename ends in .pkl.gz."""
    return os.path.join(noise_dir(root, tau), filename)


def llmsr_dir(root: str, tau: float, method: str) -> str:
    """<root>/noise_<tau>/llmsrbench/<method>."""
    return os.path.join(noise_dir(root, tau), "llmsrbench", method)


def llmsr_path(root: str, tau: float, method: str) -> str:
    return os.path.join(llmsr_dir(root, tau, method), RESULTS_NAME)


# -- plotting: read the group tree the eval scripts write ---------------------
# The cluster jobs write results/<group>/seed<N>/noise_<tau>/..., which
# merge_seed_results.py folds into results/<group>/noise_<tau>/... (all seeds in
# one artifact); a local eval_mymodels.py run writes that merged artifact directly. The
# plot scripts predate this layout -- they used to read a flat results/ view. The
# helpers below let them read the merged group tree directly, no staging step needed.

def group_noise_dir(results_root: str, group: str, tau: float) -> str:
    """results/<group>/noise_<tau> -- where a group's merged artifacts live."""
    return noise_dir(os.path.join(results_root, group), tau)


def list_srbench_pkls(results_root: str, tau: float, group: str = "mymodels",
                      prefix: str = "eval_tf", ext: str = ".pkl.gz") -> list:
    """The merged SRBench transformer artifacts (eval_tf_*.pkl.gz) for a group at
    noise tau.  Empty list if the group/noise dir does not exist."""
    d = group_noise_dir(results_root, group, tau)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.startswith(prefix) and f.endswith(ext))


def list_base_srbench_pkls(results_root: str, tau: float,
                           groups=BASE_GROUPS, prefix: str = "eval_tf") -> list:
    """The merged SRBench transformer artifacts for OUR checkpoints at noise tau,
    pooled across every base group.

    Figures used to read group "mymodels" alone, which silently omitted any checkpoint
    evaluated under the 400-point protocol (results/mymodels_400/) -- the artifact was
    simply never discovered, so its curve was absent rather than wrong.
    """
    pkls = []
    for g in groups:
        pkls += list_srbench_pkls(results_root, tau, g, prefix=prefix)
    return sorted(pkls)


def e2e_srbench_pkl(results_root: str, tau: float, group: str = "e2e",
                    filename: str = "results_feynman_e2e.pkl.gz") -> str:
    """The merged E2E Feynman SRBench artifact for noise tau."""
    return os.path.join(group_noise_dir(results_root, group, tau), filename)


def list_llmsr_method_dirs(results_root: str, tau: float, split: str,
                           groups=GROUPS) -> list:
    """Every LLM-SRBench <method>_<split>* dir across groups at noise tau.

    Methods are split across groups (the transformers under mymodels, e2e under
    e2e), so the discovery that plot scripts run over one flat llmsrbench/ dir now
    sweeps each group's llmsrbench/ dir and pools the results."""
    dirs = []
    for g in groups:
        base = os.path.join(group_noise_dir(results_root, g, tau), "llmsrbench")
        if os.path.isdir(base):
            dirs += glob.glob(os.path.join(base, f"*_{split}*"))
    return sorted(dirs)


def find_llmsr_method_dir(results_root: str, tau: float, dirname: str,
                          groups=GROUPS) -> str:
    """The path to one named LLM-SRBench method dir (e.g. 'm145_lsr_transform'),
    searching each group.  Returns the mymodels-group path when none exists, so the
    caller's os.path.exists() guard still decides whether the method is present."""
    for g in groups:
        p = os.path.join(group_noise_dir(results_root, g, tau), "llmsrbench", dirname)
        if os.path.isdir(p):
            return p
    return os.path.join(group_noise_dir(results_root, groups[0], tau), "llmsrbench", dirname)


# -- provenance ---------------------------------------------------------------

def job_id() -> str:
    """Slurm job id if we're inside an allocation, else 'local'."""
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or "local"


def write_run_info(root: str, **fields) -> str:
    """Record job id / seeds / noises next to a group's results. Merges with any
    existing file so a resubmit appends its job id rather than erasing the first."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "run_info.json")
    info = {}
    if os.path.exists(path):
        try:
            with open(path) as fh:
                info = json.load(fh)
        except Exception:
            info = {}
    runs = info.get("runs", [])
    runs.append({"job_id": job_id(), **fields})
    info["runs"] = runs
    with open(path, "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    return path


# -- artifact IO --------------------------------------------------------------

def save_results(path: str, rows: list, **meta) -> None:
    """Write {**meta, 'results': rows} to `path` atomically.

    The temp-file + os.replace dance matters: these files are rewritten after every
    dataset so an interrupted run can resume, and a reader (or the next resume) must
    never observe a half-written pickle.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = os.path.join(os.path.dirname(path) or ".",
                       f".{os.path.basename(path)}.part{os.getpid()}")
    payload = {**meta, "job_id": meta.get("job_id", job_id()), "results": list(rows)}
    with gzip.open(tmp, "wb") as fh:
        pickle.dump(payload, fh)
    os.replace(tmp, path)


def load_results(path: str):
    """Return (meta, rows) from a .pkl.gz artifact; ({}, []) when it does not exist."""
    if not os.path.exists(path):
        return {}, []
    with gzip.open(path, "rb") as fh:
        obj = pickle.load(fh)
    return {k: v for k, v in obj.items() if k != "results"}, list(obj.get("results", []))


def load_rows_any(path: str):
    """(meta, rows) from .pkl.gz, or from a legacy .csv / .jsonl artifact.

    Kept so archived trees (results/_old, pre-migration downloads) still load in the
    merge/collect/plot tooling without a conversion pass.
    """
    if path.endswith(".pkl.gz") or path.endswith(".pkl"):
        return load_results(path)
    if not os.path.exists(path):
        return {}, []
    if path.endswith(".csv"):
        import pandas as pd
        return {}, pd.read_csv(path).to_dict("records")
    with open(path) as fh:
        return {}, [json.loads(line) for line in fh if line.strip()]
