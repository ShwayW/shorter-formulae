# Local patches to the vendored tf4sr

Upstream: [omron-sinicx/transformer4sr](https://github.com/omron-sinicx/transformer4sr)
(Lalande et al., NeurIPS 2023 AI4Science workshop). No licence is declared upstream
— see `notes/VENDORED.md`.

All three patches below are **GPU-correctness fixes only**; none changes a computed
value on CPU, and none touches the model weights or architecture. Upstream never hit
them because `evaluate_model.py` loads with `map_location=torch.device('cpu')` and
runs inference entirely on CPU.

Driver: `eval_tf4sr.py` (repo root), which runs the model on `cuda` when available.

## 1. `model/positional_encodings.py` — table allocated on CPU

`PositionalEncodings.forward` built `pe = torch.zeros(seq_length, d_model)` with no
device, then returned `self.dropout(x + pe)`. With `x` on GPU this raises:

    RuntimeError: Expected all tensors to be on the same device,
    but found at least two devices, cuda:0 and cpu!

Patched to allocate `pe`, `numerator` and `denominator` on `x.device`. Values are
identical; only the device changes.

## 2. `model/multihead_attention.py` — mask fill value allocated on CPU

`scaled_dot_product_attention` did

    scores = torch.where(mask, torch.Tensor([-1e9]), scores)

`torch.Tensor([...])` is always a CPU tensor, so the same RuntimeError fires on the
first masked attention. Patched to build the fill as
`torch.tensor(-1e9, device=scores.device, dtype=scores.dtype)`.

Note this was reached only *after* fixing patch 1 — the two failures are sequential
on the same forward pass, so fixing one alone still looks broken.

## 3. `model/transformer_model.py` — causal mask allocated on CPU

`forward()` built `future_mask` via `torch.ones(...)` with no device — the same class
of bug. Patched to pass `device=target_seq.device`.

`eval_tf4sr.py` drives `model.encoder` / `model.decoder` directly and builds its own
causal mask, so it never reaches this line; it is fixed so the module is not left
half-working for anyone calling `model(...)`.

## Extra runtime dependencies

`model/_utils.py` imports `zss` and `Levenshtein` at module scope (for the tree-edit
and Levenshtein distances that upstream reports). They are needed for the import to
succeed even though `eval_tf4sr.py` scores R² and never calls them:

    pip install zss python-Levenshtein

## Not patched (deliberately)

- `model/_utils.py:163,190` (beam search) and `:260,282` (distance metrics) also
  allocate CPU tensors without a device. `eval_tf4sr.py` uses greedy decoding and
  R², so none is on its path. Left as upstream.
- `evaluate_model.py` itself is untouched: it is upstream's SRSD/zss harness and is
  not used by our evaluation. `eval_tf4sr.py` replaces it wholesale.
