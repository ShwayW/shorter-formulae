# Patches to the vendored PhysicsRegression (PhyE2E)

Upstream: [ZhangLab-DeepNeuroCogLab / PhysicsRegression](https://www.nature.com/articles/s42256-025-01126-3)
— Ying et al., *A Neural Symbolic Model for Space Physics*, Nature Machine
Intelligence 2025.

Seven changes were needed to run the authors' pretrained model from this repo. Four are
environment-compat fixes (their `environment.yml` pins torch 2.0.1 / NumPy 1.x / a
conda layout and a GPU we do not always have), one enables the beam-search decode path
this repo evaluates with, and one adds a memory dial. Nothing about the model or its
search behaviour was otherwise altered.

The pretrained checkpoint lives at `weights/phye2e/phye2e_model.pt` (374 MB), downloaded from
the authors' Google Drive folder linked in their README:

```bash
mkdir -p weights/phye2e
python -c "import gdown; gdown.download(id='1szeTQmqSkl8DPBq2aXoHzEOusg-R1sx0', output='weights/phye2e/phye2e_model.pt')"
```

Extra pip deps this required: `timeout_decorator`, `openpyxl`, `gdown`.

---

## 1. `symbolicregression/model/utils_wrapper.py` + 4 others — `np.infty` → `np.inf`

**This one silently changed results, so read it before comparing any numbers.**

`np.infty` was removed in NumPy 2.0. It appears 12 times across 5 files; the fatal one
is `TimedFun.__init__`, which every BFGS refinement call constructs:

```python
class TimedFun:
    def __init__(self, fun, stop_after=10):
        self.best_fun_value = np.infty      # AttributeError on NumPy >= 2.0
```

`SymbolicTransformerRegressor._refine()` wraps the whole refinement in a bare
`except:` that returns the *unrefined* tree, so on NumPy 2 **every constant-refinement
pass failed silently** and the "BFGS" candidates came back byte-identical to the
unrefined ones. The model's symbolic constants were never fitted to the data.

Measured effect on `feynman_I_12_1` (y = x₀·x₁), beam search, 5 bags:

| | predicted | R² |
|---|---|---|
| before | `3 * (x_0 * (x_1 * 1/4))` | 0.7426 |
| after | `3.0369 * (x_0 * (x_1 * 0.3293))` | **1.0000** |

## 2. `symbolicregression/model/model_wrapper.py` — implement `beam_type="search"`

Upstream raises `NotImplementedError` for the beam-search branch. The dead code beneath
that raise could not have worked: it decoded hypotheses with `env.idx_to_infix()`, which
predates the units-aware decoder and cannot parse a `single-seq` sequence (unit tokens
interleaved with equation tokens), and it never extended `outputs`, so the function
would have returned nothing.

`decoder.generate_beam()` itself is complete. The patch decodes each hypothesis through
the *same* `env.equation_encoder.decode()` call the greedy and sampling branches use,
with `units=None` — correct for `single-seq`, where the units ride inside the token
sequence — and extends all three output lists. The beam score stands in for
`word_perplexity` to keep them aligned.

This is what `eval_phye2e.py --decode search` runs. `--decode sampling` and
`--decode greedy` still take the untouched upstream paths.

## 3. `PhysicsRegression.py` — `torch.load` under torch ≥ 2.6

`torch.load(path)` now defaults to `weights_only=True`, which refuses this checkpoint
(it stores the training `argparse.Namespace` and NumPy scalars). Allowlisting the
globals cascades, so the load is explicit instead, and `map_location` was added so a
`cuda:0`-saved checkpoint still loads on a CPU-only machine.

## 4. `symbolicregression/envs/generators.py` — CWD-relative data paths

`build_env()` reads `./data/FeynmanEquations.xlsx` and `./data/units.csv`, which only
resolve when the process runs from inside `PhysicsRegression/`. Our entry point
(`eval_phye2e.py`) lives at the repo root, so both are now anchored to the package.

## 5. `symbolicregression/model/model_wrapper.py` — `max_forward_batch` knob

The forward chunk size is derived as
`min(10000/T, 100000/beam_size/max_generated_output_len)`, which works out to ~50 bags
per pass. Under beam search each bag expands by `beam_size`, so a default SRBench run
(100 bags) decodes ~500 sequences at once and OOMs an 8 GB GPU. `ModelWrapper` now
accepts `max_forward_batch` (`None` = upstream formula) to cap it — a pure
speed↔memory dial with identical results, mirroring the knob e2e's own `ModelWrapper`
has. `eval_phye2e.py --max-forward-batch` defaults to 10.

Note that `eval_phye2e.py` deliberately lets a CUDA OOM crash the run rather than
recording it as a failed row: a swallowed OOM once turned an entire e2e sweep into
all-`None` results that still exited 0.

## 6. `PhysicsRegression.py` — allow `device="cpu"`

`PhyReg.__init__` asserted `"cuda" in device`, so a CPU-only machine could not construct
the model at all — including a cluster **login node**, which is where the smoke test
before an sbatch has to run. `build_modules()` only moves modules onto the GPU when
`params.cpu` is False, so the patch sets `params.device` and `params.cpu` together.

## 7. `symbolicregression/model/transformer.py` — `.byte()` → `.bool()` masks

Two sites in the decoder append `<EOS>` to unfinished sequences with

```python
generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)
```

A uint8 mask was a deprecation *warning* on the torch 2.0 their `environment.yml` pins
— they silence exactly that message in five files — and is a hard error from ~2.1 on:
`RuntimeError: masked_fill only supports boolean masks, but got dtype Byte`.

It only triggers when a decode reaches `max_len`, so it failed **~1% of cells at
random**, and each failure loses the whole formula (the row is recorded with no
prediction, i.e. counted as unsolved). Upstream had already written the fix on the next
line and left it commented out; this uncomments it.

Measured on the first 13.5 h of the 2×2 cluster run: 35 of 39 total failures were this.
The other four were 2 × `NameError: name 'E' is not defined` and 2 ×
`AttributeError: 'NoneType' object has no attribute 'infix'` — genuine upstream edge
cases in their formula evaluation, ~0.1% of cells, left alone.

---

## Not patched, but worth knowing

* **`use_const_optimization` is a no-op for integer constants.** Their extra constant
  pass finds constants with the regex `[+-]?\b\d+\.\d+\b`, which only matches floats
  with a decimal point — a skeleton like `3 * x_0 * x_1 / 4` has nothing for it to
  optimise. Irrelevant once patch #1 restored the real BFGS refinement, but do not read
  `--use-const-optimization` as "constants get fitted".
* **`rescale` is forced to `False`** by their own `PhyReg.__init__`, so the E2E scaling
  trick is off. Left as the authors have it.
* **Physical units are a supported hint we do not supply.** PhyE2E can take per-variable
  units, and Feynman has them, but no other method in this repo gets that information.
  Passing them would be a different (and more favourable) configuration — see the
  `eval_phye2e.py` docstring.
