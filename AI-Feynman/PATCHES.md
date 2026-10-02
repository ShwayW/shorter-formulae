# Local patches to the upstream AI-Feynman checkout

This is the original authors' AI Feynman 2.0 (github.com/SJ001/AI-Feynman, MIT).
It is vendored **unmodified except for the changes listed here**, all of which are
compatibility/robustness fixes — none of them changes the search algorithm, its
hyperparameters, or which formulas it can find, so results stay comparable to the
published AI Feynman / SRBench numbers.

Every patch is marked in-source with a `# PATCH (ours):` comment, and this
checkout keeps its own `.git`, so `git -C AI-Feynman diff` shows the exact delta.

Driven from the repo root by `eval_aifeynman.py --benchmark llmsrbench` (LLM-SRBench).

---

## 1. Build: `build_aifeynman.sh` instead of `pip install .`  (new file)

Upstream `setup.py` builds the seven `.f90` engines as f2py extension modules via
`numpy.distutils`, **removed in NumPy 1.26**; this repo's `env/` runs NumPy 2.x on
Python 3.12, so `pip install .` fails immediately:

    ModuleNotFoundError: No module named 'numpy.distutils'

We don't need the f2py wrappers. Each `.f90` is already a standalone Fortran
*program* (`program symbolic_regress / call go / end`), and `aifeynman/S_brute_force.py`
invokes them as **executables on `$PATH`** — `subprocess.call(["feynman_sr_mdl_mult"])` —
passing arguments through `./args.dat` in the cwd, never through Python. So compiling
each program to a plain binary named after its `console_scripts` entry point is a
faithful build with no numpy/meson/f2py involvement.

    ./AI-Feynman/build_aifeynman.sh          # -> env/bin/feynman_sr*
    python eval_aifeynman.py --benchmark llmsrbench --check-install

Requires `gfortran` (`sudo apt install gfortran`, or `module load gcc` on an
Lmod-based HPC cluster). Flags used: `-ffree-line-length-none` (two lines in
`symbolic_regress_mdl3.f90` exceed the 132-col free-form limit), `-fno-range-check`,
`-std=legacy` (tabs, and the GNU `system()` / `lnblnk()` extensions).

## 2. CUDA tensors reaching `np.array()` — `S_symmetry.py`, `S_separability.py`

6 sites (`S_symmetry.py` ×4, `S_separability.py` ×2), all the same line:

```python
-        return min_error, best_i, best_j, best_mu, best_sigma
+        return float(min_error), best_i, best_j, float(best_mu), float(best_sigma)
```

`min_error`/`best_mu`/`best_sigma` come from `torch.median` / `torch.mean` / `torch.std`
and are GPU tensors on a CUDA build. `run_AI_all` feeds them straight into
`np.array([...])`, which on modern torch raises:

    TypeError: can't convert cuda:0 device type tensor to numpy.
               Use Tensor.cpu() to copy the tensor to host memory first.

This kills every problem on a GPU node. Every downstream use of these three values
is scalar arithmetic, so returning Python floats is equivalent (and matches what the
code did on the CPU path all along).

## 3. Unguarded `np.loadtxt` after a brute-force timeout — `S_run_aifeynman.py`

2 sites, in the compositionality and generalized-symmetry blocks of `run_AI_all`:

```python
 brute_force_comp("results/", ..., 600, "14ops.txt")
+if not os.path.exists("results_comp.dat"):
+    break
 bf_all_output = np.loadtxt("results_comp.dat", dtype="str")
```

(and the same for `results_gen_sym.dat`.)

`S_brute_force_comp.py` **deletes** `results_comp.dat` before starting and runs the
Fortran engine under `subprocess.call(..., timeout=600)`. Whenever that engine is
killed before its first write — common on hard problems — the file does not exist and
the unguarded `loadtxt` raises `FileNotFoundError`, aborting the entire problem and
discarding the Pareto front already found. Upstream clearly intended this to be
recoverable: the surrounding `try:` / `except: idx_comp = 0` is still there, commented
out (`#try:` … `#except:`). The patch skips the optional block instead of crashing.

## 4. Python-2 true division reaching `range()` — `S_run_aifeynman.py`

1 site, in `run_AI_all`:

```python
-model_feynman = NN_train(pathdir, filename, NN_epochs/2,  lrs=1e-3, ...)
+model_feynman = NN_train(pathdir, filename, NN_epochs//2, lrs=1e-3, ...)
```

Under Python 3 `NN_epochs/2` is a float, so `NN_train` reaches `for epoch in range(epochs)`
and raises:

    TypeError: 'float' object cannot be interpreted as an integer

This is on the **recursive** branch — the one `run_AI_all` takes after it finds a
symmetry or separability and re-enters itself on the reduced problem — so it fires on
essentially every non-trivial problem, i.e. AI Feynman's whole decomposition path was
dead on Python 3. Floor division restores the intended "half as many epochs on the
pretrained sub-problem".

## 5. `LinAlgError` in the graph-modularity stage — `S_gradient_decomposition.py`

1 site, in `score_consistency`:

```python
-norms = [np.linalg.norm(grads_tensor[i, :].numpy()) for i in range(n_pts)]
-normalized_grads = [grads_tensor[i,:].numpy()/norms[i] for i in range(n_pts)]
-A = np.array(normalized_grads)
+_g = grads_tensor.numpy(); _norms = np.linalg.norm(_g, axis=1)
+_keep = np.isfinite(_norms) & (_norms > 0) & np.all(np.isfinite(_g), axis=1)
+if not _keep.any(): return 0.0, np.zeros(_g.shape[1])
+A = _g[_keep] / _norms[_keep][:, None]; n_pts = int(_keep.sum())
```

Each gradient row is divided by its own norm with no check. A sample point where the
surrogate net has a **zero** gradient (or a non-finite one) makes that row inf/NaN, so
`D = A^T A` is non-finite and the following `np.linalg.eig(D)` raises:

    numpy.linalg.LinAlgError: Array must not contain infs or NaNs

Nothing catches it. It propagates out of `score_consistency` →
`filter_decompositions_relative_scoring` → `identify_decompositions` → `run_AI_all`'s
generalized-symmetry block (`S_run_aifeynman.py:167`, where only the *inner* loop has a
`try/except`), destroying the entire problem — **after hours of search**.

Measured impact: **49 of 111** LLM-SRBench problems (44%) died this way in the first
full 8 h run, at 1.3–7.6 h in.

Dropping the degenerate rows is the conservative fix: a zero or undefined gradient
carries no directional information and cannot contribute to a consistency score, and
dropping it keeps the *other* candidate decompositions scorable instead of losing 2.0's
modularity stage for the whole problem. When every row is finite and non-zero the
computation is bit-for-bit the original (verified: identical to 1e-15, `A` unchanged,
`n_kept == n_pts`).

## 6. Runtime environment (no source change)

`eval_aifeynman.py --benchmark llmsrbench` sets `MPLBACKEND=Agg` (the package imports
`matplotlib.pyplot` at module scope, which fails on a headless node) and runs every
problem in its own scratch cwd, because `run_AI_all` unconditionally creates
`./results/` and the Fortran engines read/write `./args.dat`, `./mystery.dat`,
`./results.dat` in the current directory — from the repo root that would collide with
this project's own `results/` tree, and it makes concurrent problems in one directory
unsafe.

## 7. Missing dependency

`sortedcontainers` (imported by `aifeynman/get_pareto.py`) was not in `env/`;
installed with pip. Note upstream's `install_requires` also lists `sklearn`, the
dead stub package that now errors on install — another reason not to `pip install .`.
