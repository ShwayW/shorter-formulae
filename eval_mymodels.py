#!/usr/bin/env python3
"""
eval_mymodels.py -- Evaluate our pretrained VanillaTransformer on either benchmark.

Pick the benchmark with --benchmark:

  srbench     (default)  Feynman PMLB datasets under --datasets.  Bagging trick:
                         for each dataset draw n_bags independent 200-pair samples,
                         synthesise a formula from each (beam search + BFGS
                         refinement), evaluate every candidate on all available
                         points, keep the best R^2.  Writes
                         <results-dir>/noise_<TAU>/eval_tf_<model>.pkl.gz.

  llmsrbench             LLM-SRBench (Shojaee et al., ICML 2025), --split
                         lsr_transform or lsr_synth_<domain>.  The searcher sees
                         only the TRAIN split; the formula is then scored on the
                         held-out TEST (and OOD when present) with the official
                         metrics.  Writes <results-dir>/noise_<TAU>/llmsrbench/
                         <model>_<split>[_tag]/results.pkl.gz.

Flags that apply to one benchmark only are grouped as such in --help.  Several
defaults are benchmark-dependent (--seed, --n-bags, --beam-size and the TPSR
knobs); see _apply_bench_defaults.  Both paths share one inference core
(evaluate_xy), so a change to the search affects both.

Usage
-----
# full SRBench benchmark
python eval_mymodels.py --model 89M_40_simp1

# single SRBench dataset, on CPU
python eval_mymodels.py --model 89M_40_simp1 --dataset feynman_I_6_2 --device cpu

# LLM-SRBench
python eval_mymodels.py --benchmark llmsrbench --model 89M_40_simp1 --split lsr_transform
"""

import os
import sys
import glob
import gzip
import csv
import pickle
import argparse
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from results_io import noise_dir, llmsr_path, load_results, save_results
from bench_cli import bench_defaults, check_bench_flags
from grammar import V, C, B, U, is_num_sign, is_num_mantissa, is_num_exp
from feynman_pn_analysis import formula_to_pn as _feynman_to_pn
from tfs.van import VanillaTransformer
from utils import idx_to_toks, _reassemble_floats
from bfgs_optimizer import BFGSOptimizer
from funcWrappers import wrapExtEvalPN
from tpsr import TPSR
from subtree_bpe import SubtreeBPE

import threading as _threading
_GPU_BFGS_LOCK = _threading.Lock()

# When --gpu-bfgs is enabled this is set to a callable
#   refine_batch(formulas: list[str], X, y) -> list[str]
# that refines a whole batch of skeletons in ONE gpu_bfgs.batch_refine call
# (one lock acquisition, one X/y upload). The top-N refine loop uses it instead
# of the per-formula threadpool so the per-call overhead is paid once, not N*.
_GPU_BFGS_REFINE_BATCH = None


def enable_gpu_bfgs(device, restarts=8, iters=40, dtype="float64"):
    """Route constant refinement through the batched GPU Levenberg-Marquardt +
    Variable-Projection optimizer (gpu_bfgs.batch_refine) instead of SciPy.

    Patches BFGSOptimizer.opt_consts (used by the final polish and any stray
    callers) AND installs a batch refiner (_GPU_BFGS_REFINE_BATCH) that the
    top-N refine loop calls once for all candidates.  Calls are serialized by a
    lock (each batch_refine already batches its restarts on the GPU)."""
    import torch as _torch
    import gpu_bfgs
    global _GPU_BFGS_REFINE_BATCH
    _dtype = _torch.float64 if dtype == "float64" else _torch.float32

    def _opt_consts(self, X, y, stop_after=10, n_restarts=8):
        if not self.has_constants:
            return " ".join(str(t) for t in self.tokens)
        cmap = {self.param_names[i]: repr(self.initial_guess[i])
                for i in range(len(self.param_names))}
        formula = " ".join(cmap.get(t, t) for t in self.tokens)
        with _GPU_BFGS_LOCK:
            try:
                return gpu_bfgs.batch_refine([formula], X, y, device,
                                             n_restarts=restarts, max_iter=iters,
                                             dtype=_dtype, seed=0)[0]
            except Exception:
                return formula

    BFGSOptimizer.opt_consts = _opt_consts

    def _refine_batch(formulas, X, y):
        if not formulas:
            return []
        with _GPU_BFGS_LOCK:
            try:
                return gpu_bfgs.batch_refine(list(formulas), X, y, device,
                                             n_restarts=restarts, max_iter=iters,
                                             dtype=_dtype, seed=0)
            except Exception:
                return list(formulas)

    _GPU_BFGS_REFINE_BATCH = _refine_batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_SIZE      = 200  # IO pairs drawn per bag
#SAMPLE_SIZE      = 400  # IO pairs drawn per bag
N_BAGS           = 100  # default number of bags (single source of truth; see --n-bags)
DEFAULT_FORWARD_BATCH = 10  # default bags processed per GPU call (override with --max-forward-batch)
BEAM_SIZE        = 10    # beam width -- ALL beams are now kept and pooled across bags
N_REFINE         = 10    # unique skeletons to BFGS-refine after pooling
BFGS_DOWNSAMPLE  = 1024  # max points passed to BFGS (matches eval_e2e.py downsample=1024)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(tsv_gz_path: str):
    """Load a .tsv.gz dataset.

    Returns
    -------
    X : np.ndarray, shape (N, D), float32   -- input variables
    y : np.ndarray, shape (N,),  float32   -- target output
    """
    with gzip.open(tsv_gz_path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header row
        rows = list(reader)

    data = np.array(rows, dtype=np.float64)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


# ---------------------------------------------------------------------------
# Whitening helper
# ---------------------------------------------------------------------------
def _whiten(X: np.ndarray) -> np.ndarray:
    """Column-wise z-score whitening, matching data_process.py line 125.

    inputs = (inputs_raw - mean(inputs_raw)) / (std(inputs_raw) + 1e-8)
    """
    mean = np.mean(X, axis=0)
    std  = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)


# ---------------------------------------------------------------------------
# Input unscaling
# ---------------------------------------------------------------------------
def apply_input_unscaling(formula: str, mean: np.ndarray, std: np.ndarray,
                           n_active: int) -> str:
    """Substitute v_i -> (v_i - mu_i) / sigma_i for the first n_active variable slots.

    The transformer was trained on whitened inputs.  This converts the predicted
    formula from whitened-variable space to original-variable space, giving
    BFGS a well-initialised starting point when refining against raw data.

    Prefix-notation substitution:
        vi  ->  / + vi [-mu_i] [sigma_i]   (= (vi - mu_i) / sigma_i)
    """
    tokens = formula.split()
    if not tokens:
        return formula
    result = []
    for tok in tokens:
        if tok in V:
            idx = V.index(tok)
            if idx < n_active:
                result.extend([
                    '/', '+', tok,
                    f"{-float(mean[idx]):.8g}",
                    f"{float(std[idx]):.8g}",
                ])
                continue
        result.append(tok)
    return ' '.join(result)


# ---------------------------------------------------------------------------
# BPE expansion
# ---------------------------------------------------------------------------
def expand_bpe_tokens(formula: str, bpe) -> str:
    """Expand all <BPE_N> compound tokens to their primitive PN strings.

    Iterates until no compound tokens remain, handling nested BPE tokens
    (a token whose expansion itself contains other BPE tokens).
    """
    tokens = formula.split()
    changed = True
    while changed:
        changed = False
        new_tokens = []
        for tok in tokens:
            if tok in bpe.token_to_pn:
                new_tokens.extend(bpe.token_to_pn[tok].split())
                changed = True
            else:
                new_tokens.append(tok)
        tokens = new_tokens
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Formula sanitization
# ---------------------------------------------------------------------------
_BINARY_SET   = set(B)
_UNARY_SET    = set(U)
_GRAMMAR_TOKS = set(V + list(C) + list(U) + list(B) + ["<bes>", "<pad>"])
_CONST_TOKS   = set(C)  # vocab numeric constants that BFGS will fit -> normalise in skeleton


# extEvalPN.cpp uses a fixed stack of 128 doubles (s_fast[128]).
# It evaluates tokens in REVERSE order, so terminals at the end of the
# formula are pushed first in the reversed pass.  Maximum safe stack depth
# is 127 (writing to s_fast[127] is the last safe write; s_fast[128] would
# overflow). We stay one below that limit to be safe.
_EVAL_STACK_LIMIT = 127


def _max_eval_depth(tokens: list) -> int:
    """Compute the peak stack depth extEvalPN would reach for these tokens.

    extEvalPN processes tokens in REVERSE order (right-to-left):
      terminal  -> push  (+1)
      binary op -> pop 2, push 1  (net -1)
      unary  op -> pop 1, push 1  (net  0)
    """
    depth = 0
    peak  = 0
    for tok in reversed(tokens):
        if tok in _BINARY_SET:
            depth -= 1
        elif tok not in _UNARY_SET:
            depth += 1      # terminal
        if depth > peak:
            peak = depth
    return peak


def sanitize_formula(formula: str) -> str:
    """Make a beam-search output valid and safe for BFGS and evaluation.

    Steps:
    1. Reassemble any leftover float encoding tokens (NUM+/-, Mxxxx, E+/-xx)
       that idx_to_toks did not collapse -> concrete float literals.  Drop
       any encoding tokens that still remain after reassembly.
    2. Walk the token list with a needs-counter:
         binary op -> needs += 1 | unary op -> needs += 0 | terminal -> needs -= 1
       Stop at the first complete expression (needs == 0), discarding any
       trailing tokens.  If the token list is exhausted while needs > 0,
       pad with '1' for each remaining open slot.
    3. Stack-overflow guard: extEvalPN.cpp uses a fixed 128-element stack.
       Check the peak evaluation depth after truncation/padding; return ''
       if it would overflow (depth >= 128).
    4. Small wrapExtEvalPN validation on dummy data.
    """
    tokens = formula.split()
    if not tokens:
        return ""

    # --- 1. float token reassembly & cleanup ---
    tokens = _reassemble_floats(tokens)
    tokens = [t for t in tokens
              if not (is_num_sign(t) or is_num_mantissa(t) or is_num_exp(t))]
    if not tokens:
        return ""

    # --- 2. truncate / complete to a valid prefix expression ---
    needs = 1
    valid = []
    for tok in tokens:
        valid.append(tok)
        if tok in _BINARY_SET:
            needs += 1
        elif tok not in _UNARY_SET:
            needs -= 1          # terminal
        if needs == 0:
            break

    # Pad incomplete expressions -- but only if the padding itself won't
    # overflow the evaluator stack.  Padding with 'needs' terminals at the
    # end means they appear FIRST in the reversed evaluation pass, pushing
    # depth straight to 'needs'.
    if needs > 0:
        if needs > _EVAL_STACK_LIMIT:
            return ""           # formula is almost entirely operators; discard
        valid.extend(["1"] * needs)

    # --- 3. stack-overflow guard ---
    # Even without padding, a pathologically left-deep formula can overflow.
    if _max_eval_depth(valid) >= 128:
        return ""

    formula = " ".join(valid)

    # --- 4. structural validation via wrapExtEvalPN on dummy data ---
    dummy_X = np.ones((3, len(V)), dtype=np.float64)
    try:
        _, stacklefts = wrapExtEvalPN(formula, dummy_X)
        if np.any(stacklefts != 0):
            return ""
    except Exception:
        return ""

    return formula


def formula_is_complete(formula: str) -> bool:
    """True iff `formula` is already a complete prefix expression.

    Mirrors sanitize_formula's float reassembly + needs-counter walk, but reports
    whether the token stream forms exactly one complete expression with no
    missing operands -- i.e. needs reaches 0 exactly at the final token.  Used to
    skip recording partial sequences that sanitize_formula would otherwise pad
    with constants (producing degenerate candidates).
    """
    tokens = _reassemble_floats(formula.split())
    tokens = [t for t in tokens
              if not (is_num_sign(t) or is_num_mantissa(t) or is_num_exp(t))]
    if not tokens:
        return False
    needs = 1
    for i, tok in enumerate(tokens):
        if tok in _BINARY_SET:
            needs += 1
        elif tok not in _UNARY_SET:
            needs -= 1          # terminal
        if needs == 0:
            return i == len(tokens) - 1   # complete, with no trailing tokens
    return False                          # needs > 0 -> incomplete (would be padded)


# ---------------------------------------------------------------------------
# Tensor construction
# ---------------------------------------------------------------------------
def build_ios_tensor(X_sample: np.ndarray, y_sample: np.ndarray,
                     n_vars: int = None) -> torch.Tensor:
    """Pack (N, D) inputs and (N,) outputs into a (1, N*(n_vars+1)) float tensor.

    Real variable columns are whitened (zero mean, unit std) exactly as during
    training (data_process.py).  Unused slots are filled with NaN -- matching
    training, where online_batch_iterater NaN-pads unused variable columns so the
    LPE maps them to its learnable INPUT_PAD embedding.
    """
    N, D = X_sample.shape
    num_vars = n_vars if n_vars is not None else len(V)

    # whiten the real variable columns
    X_whitened = _whiten(X_sample).astype(np.float32)

    if D < num_vars:
        # NaN-pad unused slots: matches training (online_batch_iterater line ~255)
        pad_nan = np.full((N, num_vars - D), np.nan, dtype=np.float32)
        X_padded = np.concatenate([X_whitened, pad_nan], axis=1)
    elif D > num_vars:
        warnings.warn(
            f"Dataset has {D} input variables but model supports only {num_vars}; "
            f"truncating the last {D - num_vars} column(s).",
            stacklevel=2,
        )
        X_padded = X_whitened[:, :num_vars]
    else:
        X_padded = X_whitened[:, :num_vars]

    # each row: [v1, v2, ..., v{num_vars}, output]
    ios = np.concatenate([X_padded, y_sample.reshape(-1, 1)], axis=1)  # (N, num_vars+1)
    return torch.from_numpy(ios.reshape(1, -1))  # (1, N*(num_vars+1))


# ---------------------------------------------------------------------------
# R^2 evaluation
# ---------------------------------------------------------------------------
def compute_r2(formula: str, X_all: np.ndarray, y_all: np.ndarray,
               _x64: np.ndarray = None, _y64: np.ndarray = None,
               _ss_tot: float = None) -> float:
    """Evaluate a prefix-notation formula on (X_all, y_all) and return R^2.

    X_all is used raw (unwhitened): BFGS already fitted the formula's constants
    to raw data, so evaluation must also use raw inputs.

    Pass _x64/_y64 (pre-padded float64) and _ss_tot (cached denominator) to
    skip repeated conversion/computation in hot loops.

    Returns 0.0 on any error or malformed formula.
    """
    if not formula.strip():
        return 0.0

    try:
        if _x64 is not None:
            X_np = _x64
            y    = _y64 if _y64 is not None else y_all.astype(np.float64)
        else:
            num_vars = len(V)
            X_np = X_all.astype(np.float64)
            if X_np.shape[1] < num_vars:
                pad_cols = np.zeros((X_np.shape[0], num_vars - X_np.shape[1]), dtype=np.float64)
                X_np = np.concatenate([X_np, pad_cols], axis=1)
            y = y_all.astype(np.float64)

        preds, stacklefts = wrapExtEvalPN(formula, X_np)

        if np.any(stacklefts != 0):
            return 0.0
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            return 0.0

        with np.errstate(over='ignore', invalid='ignore'):
            ss_res = np.sum((y - preds) ** 2)
        if not np.isfinite(ss_res):
            return 0.0
        ss_tot = _ss_tot if _ss_tot is not None else float(np.sum((y - np.mean(y)) ** 2))

        if np.isclose(ss_tot, 0.0):
            return 1.0 if np.isclose(ss_res, 0.0) else 0.0

        r2 = 1.0 - min(ss_res / ss_tot, 1.0)
        return float(max(r2, 0.0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device):
    """Load the VanillaTransformer from a checkpoint.

    Returns (model, vocab).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["model_config"]
    state_dict = checkpoint["Van_state_dict"]

    # Derive architecture params from the checkpoint weights so that checkpoints
    # trained with different grammar sizes or sequence lengths load correctly.
    ckpt_vocab_size   = state_dict["fc.weight"].shape[0]
    ckpt_max_seq_len  = state_dict["dec_positional_encoding.pe"].shape[0]

    # Prefer the vocab stored in the checkpoint (saved by train_transformer.py).
    # Fall back to reconstruction for legacy checkpoints that predate vocab storage.
    if "vocab" in checkpoint:
        vocab  = checkpoint["vocab"]
        n_vars = sum(1 for t in vocab if t in set(V))
    else:
        _special = ["<bes>", "<pad>"]

        # Derive n_vars from the LPE weight shape -- robust to U/B list changes.
        # flat_dim = patch_dim * tps * d_emb  ->  n_vars = flat_dim/(tps*d_emb) - 1
        # NOTE: lpe_mantissa_len is the INPUT float-encoding width (the LinearPointEmbedder
        # tokenises the IO pairs); it is independent of the mantissa-free OUTPUT grammar.
        if config.get("use_lpe") and "lpe.mlp.0.weight" in state_dict:
            lpe_d_emb        = config["lpe_d_emb"]
            lpe_mantissa_len = config["lpe_mantissa_len"]
            tps              = 2 + lpe_mantissa_len
            flat_dim         = state_dict["lpe.mlp.0.weight"].shape[1]
            n_vars           = flat_dim // (tps * lpe_d_emb) - 1
        else:
            _fixed = len(C) + len(U) + len(B) + len(_special)
            n_vars = ckpt_vocab_size - _fixed

        # Recover the U and B prefix that was present when the checkpoint was saved.
        # New operators are always appended at the end of U and B, so the checkpoint's
        # operators are a leading prefix of the current lists.
        _fixed_excl_ub = n_vars + len(C) + len(_special)
        _n_ub    = ckpt_vocab_size - _fixed_excl_ub   # how many U+B ops in checkpoint
        _delta   = len(U) + len(B) - _n_ub            # how many ops were added since
        _delta_B = min(_delta, len(B) - 1)             # binary ops added (at most len(B)-1)
        _delta_U = _delta - _delta_B
        _ckpt_U  = U[:len(U) - _delta_U] if _delta_U > 0 else list(U)
        _ckpt_B  = B[:len(B) - _delta_B] if _delta_B > 0 else list(B)
        # Mantissa-free grammar: vocab = V + C + U + B + special (matches FlatGram + special).
        vocab    = V[:n_vars] + C + _ckpt_U + _ckpt_B + _special

    assert len(vocab) == ckpt_vocab_size, \
        f"vocab size mismatch: reconstructed {len(vocab)} != checkpoint {ckpt_vocab_size}"

    pad_idx    = vocab.index("<pad>")
    patch_dim  = n_vars + 1   # input vars + 1 output column

    model = VanillaTransformer(
        config=config,
        patch_dim=patch_dim,
        vocab_size=ckpt_vocab_size,
        max_seq_length=ckpt_max_seq_len,
        pad_idx=pad_idx,
    )
    has_mid_layer = "enc_mid_layer.weight" in state_dict
    if not has_mid_layer:
        model_keys = set(model.state_dict().keys())
        ckpt_keys  = set(state_dict.keys())
        expected_missing = {"enc_mid_layer.weight", "enc_mid_layer.bias"}
        unexpected = ckpt_keys - model_keys
        missing    = (model_keys - ckpt_keys) - expected_missing
        if unexpected:
            print(f"[load_model] Ignoring unexpected checkpoint keys: {sorted(unexpected)}", flush=True)
        if missing:
            print(f"[load_model] Missing keys (will use random init): {sorted(missing)}", flush=True)
    model.load_state_dict(state_dict, strict=has_mid_layer)
    if not has_mid_layer and model.enc_mid_layer is not None:
        # old checkpoint: enc_mid_layer didn't exist, so initialise it as
        # identity to preserve the original LPE -> encoder pass-through behaviour
        torch.nn.init.eye_(model.enc_mid_layer.weight)
        torch.nn.init.zeros_(model.enc_mid_layer.bias)
    model.set_sample_config(config)
    model.eval()
    model.to(device)

    return model, vocab


# ---------------------------------------------------------------------------
# Multi-beam decoding helpers
# ---------------------------------------------------------------------------
def beam_search_all_beams(
    model, ios_tensor: torch.Tensor, vocab: list,
    max_fml_len: int, device: torch.device, beam_size: int,
) -> torch.Tensor:
    """Beam search with KV caching returning ALL beams (best-first).

    Accepts batch input: ios_tensor shape (B, io_dim).
    Returns shape (B, beam_size, seq_len).

    Key optimisations vs the original:
    - KV cache: decoder self-attention K/V are accumulated incrementally
      (O(T) per step) rather than recomputed over the full growing sequence
      (O(T^2) per step).
    - Cross-attention K/V are precomputed once from the fixed encoder output.
    - torch.inference_mode() instead of no_grad for lower overhead.
    - bfloat16 autocast on CUDA.
    """
    bes_index = vocab.index("<bes>")

    inputs = ios_tensor.float().to(device)   # (batch_size, io_dim)
    batch_size = inputs.shape[0]

    model.eval()
    use_amp = (device.type == "cuda")
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast("cpu", enabled=False)

    with torch.inference_mode(), amp_ctx:
        enc_output, enc_mask = model.encode(inputs)                       # (batch_size, S, d)
        enc_output = enc_output.repeat_interleave(beam_size, dim=0)       # (batch_size*beam, S, d)
        enc_mask   = enc_mask.repeat_interleave(beam_size, dim=0)

        # Precompute cross-attention K/V once for all decoder layers
        cross_kv = model.precompute_cross_kv(enc_output)

        # Initialise empty self-KV caches (grow by 1 each step)
        self_kv = model.init_self_kv_cache(batch_size * beam_size, device)

        seqs   = torch.full((batch_size * beam_size, 1), bes_index, dtype=torch.int64, device=device)
        scores = torch.full((batch_size, beam_size), float('-inf'), device=device)
        scores[:, 0] = 0.0
        done   = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

        for step in range(max_fml_len):
            last_tok = seqs[:, -1:]                                       # (batch_size*beam, 1)

            logits, new_kv = model.decode_step(
                last_tok, step, enc_mask, self_kv, cross_kv)              # (batch_size*beam, V_sz)

            log_probs = torch.log_softmax(logits.float(), dim=-1)         # cast back for score accumulation
            V_sz      = log_probs.shape[-1]
            log_probs = log_probs.reshape(batch_size, beam_size, V_sz)

            log_probs.masked_fill_(done.unsqueeze(-1), float('-inf'))
            if done.any():
                di, dj = done.nonzero(as_tuple=True)
                log_probs[di, dj, bes_index] = 0.0

            cand      = scores.unsqueeze(-1) + log_probs                  # (batch_size, beam, V_sz)
            cand_flat = cand.reshape(batch_size, beam_size * V_sz)
            top_scores, top_idx = cand_flat.topk(beam_size, dim=-1)       # (batch_size, beam)

            beam_idx = top_idx // V_sz                                    # (batch_size, beam) -- old beam
            tok_idx  = top_idx % V_sz                                     # (batch_size, beam) -- new token

            seq_len      = seqs.shape[1]
            seqs_3d      = seqs.reshape(batch_size, beam_size, seq_len)
            beam_idx_exp = beam_idx.unsqueeze(-1).expand(-1, -1, seq_len)
            new_seqs     = seqs_3d.gather(1, beam_idx_exp)
            new_seqs     = torch.cat([new_seqs, tok_idx.unsqueeze(-1)], dim=-1)

            # Reorder KV caches to match the new beam ordering
            batch_offset   = torch.arange(batch_size, device=device).unsqueeze(1) * beam_size
            flat_beam_idx  = (batch_offset + beam_idx).reshape(-1)        # global old-beam indices
            self_kv = model.reorder_kv_cache(new_kv, flat_beam_idx)

            scores = top_scores
            seqs   = new_seqs.reshape(batch_size * beam_size, seq_len + 1)
            done   = (tok_idx == bes_index)
            if done.all():
                break

        # Sort each example's beams best-first
        order   = scores.argsort(dim=-1, descending=True)                 # (batch_size, beam)
        seqs_3d = seqs.reshape(batch_size, beam_size, -1)
        all_seqs = seqs_3d[
            torch.arange(batch_size, device=device).unsqueeze(1), order
        ].cpu()                                                            # (batch_size, beam, final_len)

    return all_seqs


def sample_sequences(
    model, ios_tensor: torch.Tensor, vocab: list,
    max_fml_len: int, device: torch.device,
    n_samples: int = 10, temperature: float = 1.0, top_p: float = 1.0,
) -> torch.Tensor:
    """Sample n_samples formula sequences per bag from the model distribution.

    Batched: accepts B bags at once (matching beam_search_all_beams).
    Each bag produces n_samples independent sequences via ancestral sampling
    with optional temperature scaling and nucleus (top-p) filtering.
    KV-cached decode -- O(B * n_samples * T) ops, same structure as beam search.

    Parameters
    ----------
    ios_tensor : (B, io_dim) -- batch of B IO bags.

    Returns
    -------
    seqs : (B, n_samples, seq_len) long tensor on CPU
    """
    bes_idx = vocab.index("<bes>")
    inputs = ios_tensor.float().to(device)               # (B, io_dim)
    B = inputs.shape[0]

    use_amp = device.type == "cuda"
    amp_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
               if use_amp else torch.amp.autocast("cpu", enabled=False))

    with torch.inference_mode(), amp_ctx:
        enc_output, enc_mask = model.encode(inputs)                         # (B, S, d)
        enc_output = enc_output.repeat_interleave(n_samples, dim=0)        # (B*n_samples, S, d)
        enc_mask   = enc_mask.repeat_interleave(n_samples, dim=0)
        cross_kv   = model.precompute_cross_kv(enc_output)
        self_kv    = model.init_self_kv_cache(B * n_samples, device)

        seqs = torch.full((B * n_samples, 1), bes_idx, dtype=torch.long, device=device)
        done = torch.zeros(B * n_samples, dtype=torch.bool, device=device)

        for step in range(max_fml_len):
            last_tok = seqs[:, -1:]                                         # (B*n_samples, 1)
            logits, self_kv = model.decode_step(
                last_tok, step, enc_mask, self_kv, cross_kv)               # (B*n_samples, V)
            logits = logits.float()

            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)                          # (B*n_samples, V)

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                # Zero tokens whose cumulative mass (exc. themselves) > top_p;
                # always keep at least the top-1 token per row.
                remove = (cumulative - sorted_probs) > top_p
                sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

            next_tok = torch.multinomial(probs, num_samples=1)             # (B*n_samples, 1)
            next_tok = next_tok.masked_fill(done.unsqueeze(1), bes_idx)

            seqs = torch.cat([seqs, next_tok], dim=1)
            done = done | (next_tok.squeeze(-1) == bes_idx)
            if done.all():
                break

    return seqs.reshape(B, n_samples, -1).cpu()                            # (B, n_samples, seq_len)


def formula_skeleton(formula: str) -> str:
    """Replace all numeric constants with '1' so that two formulas differing
    only in constant values share the same skeleton.

    Normalises both float literals (from unscaling/BFGS) and the vocab integer
    constants {0, 2, 3, pi} -- BFGS fits the latter too, so pre- and post-BFGS
    formulas must map to the same skeleton for raw_map lookup to work.

    Literals are detected by float() alone, NOT by looking for a '.'.  Unscaling
    formats mu and sigma with '%.8g', which drops the point whenever the value is
    integral or takes an exponent: a constant column (sigma floored to std+1e-8)
    emits "1e-08", and an integral mean emits "-5".  Those tokens have no '.', so
    the old test left them in the skeleton while BFGS -- which decides fittability
    with is_number(), not this rule -- turned them into dotted floats.  Pre- and
    post-BFGS skeletons then differed and the raw_map lookup missed, but only on
    the --unscale arms, since nothing else introduces these literals.

    Non-finite tokens are deliberately NOT normalised: float() accepts "inf" and
    "nan", and a degenerate constant is worth keeping distinct rather than
    silently merging into the skeleton of a well-formed formula.
    """
    result = []
    for tok in formula.split():
        if tok in _CONST_TOKS:
            result.append('1')
            continue
        if tok not in _GRAMMAR_TOKS:
            try:
                if math.isfinite(float(tok)):
                    result.append('1')
                    continue
            except ValueError:
                pass
        result.append(tok)
    return ' '.join(result)


def bfgs_refine_mfg(formula: str, X: np.ndarray, y: np.ndarray,
                    stop_after: float = 10, n_restarts: int = 8) -> str:
    """BFGS refinement for mantissa-free-grammar (MFG) formulas.

    The MFG uses symbolic constant tokens (0, 1, 2, 3, pi) instead of CONST
    placeholders.  BFGSOptimizer already treats 0/1/2/3 as free constants
    (they pass is_number()), but 'pi' does not.  This wrapper:

      1. Replaces 'pi' with its numeric value so BFGS treats it as a free
         constant alongside 0, 1, 2, 3.
      2. If the formula has no fittable constants at all (pure variable
         expression like '* v1 v2'), prepends a free scalar '1.0' so BFGS
         can recover a missing multiplicative factor (e.g. 1/(2*pi)).

    Falls back gracefully to the original formula on any error.
    Works transparently for old CONST-token formulas too.
    """
    try:
        # Step 1: make 'pi' fittable by replacing it with its numeric value
        numeric = " ".join(str(np.pi) if t == "pi" else t
                           for t in formula.split())

        opt = BFGSOptimizer(numeric)

        # Step 2: no fittable constants -> prepend free scalar
        if not opt.has_constants:
            numeric = f"* 1.0 {numeric}"
            opt = BFGSOptimizer(numeric)

        refined = opt.opt_consts(X, y, stop_after=stop_after, n_restarts=n_restarts)
        return refined if refined else formula
    except Exception:
        return formula


def _bfgs_refine(args: tuple) -> tuple:
    """BFGS constant-refinement worker, compatible with ThreadPoolExecutor."""
    formula, X_bfgs, y_bfgs = args
    try:
        refined = bfgs_refine_mfg(formula, X_bfgs, y_bfgs)
        return formula, refined if refined else None
    except Exception:
        return formula, None


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------
def evaluate_dataset(dataset_dir: str, model, vocab, device: torch.device,
                     rng: np.random.Generator,
                     n_bags: int = N_BAGS,
                     use_tpsr: bool = False, tpsr_sims: int = 3,
                     tpsr_c_puct: float = 1.0, tpsr_top_k: int = 20,
                     tpsr_n_bags: int = 3, tpsr_prior_temperature: float = 1.0,
                     tpsr_horizon: int = 40, tpsr_rollout_beam_width: int = 1,
                     tpsr_no_search_bfgs: bool = True,
                     tpsr_warm_start: bool = True,
                     tpsr_warm_start_beam_width: int = 10,
                     use_sampling: bool = False, n_samples: int = 10,
                     temperature: float = 1.0, top_p: float = 1.0,
                     no_split: bool = False,
                     use_bfgs: bool = True,
                     beam_size: int = BEAM_SIZE,
                     bpe_model=None,
                     use_unscaling: bool = True,
                     gt_pn: str = "",
                     ood_data: tuple | None = None,
                     use_offset_repair: bool = False,
                     n_points_override: int | None = None,
                     sample_size_override: int | None = None,
                     target_noise: float = 0.0,
                     max_forward_batch: int = DEFAULT_FORWARD_BATCH):
    """Bagging evaluation for one dataset directory.

    Parameters
    ----------
    n_points : subsample this many rows before splitting (matches eval_e2e.py).
    ood_data : pre-generated (X_ood, y_ood) arrays from gen_ood_data.py, or None.

    Returns
    -------
    (best_formula, best_r2) normally; (best_formula, best_r2, ood_r2) when ood_data is set.
    """
    # get data files
    tsv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".tsv.gz")]

    # the base case
    if not tsv_files:
        return ("", 0.0, None) if ood_data is not None else ("", 0.0)

    # get the dataset: the IO pairs
    tsv_gz_path = os.path.join(dataset_dir, tsv_files[0])
    X_all, y_all = load_dataset(tsv_gz_path)
    return evaluate_xy(
        X_all, y_all, model, vocab, device, rng,
        n_bags=n_bags, use_tpsr=use_tpsr, tpsr_sims=tpsr_sims,
        tpsr_c_puct=tpsr_c_puct, tpsr_top_k=tpsr_top_k, tpsr_n_bags=tpsr_n_bags,
        tpsr_prior_temperature=tpsr_prior_temperature, tpsr_horizon=tpsr_horizon,
        tpsr_rollout_beam_width=tpsr_rollout_beam_width,
        tpsr_no_search_bfgs=tpsr_no_search_bfgs,
        tpsr_warm_start=tpsr_warm_start,
        tpsr_warm_start_beam_width=tpsr_warm_start_beam_width,
        use_sampling=use_sampling, n_samples=n_samples,
        temperature=temperature, top_p=top_p, no_split=no_split,
        use_bfgs=use_bfgs, beam_size=beam_size, bpe_model=bpe_model,
        use_unscaling=use_unscaling, gt_pn=gt_pn, ood_data=ood_data,
        use_offset_repair=use_offset_repair,
        n_points_override=n_points_override,
        sample_size_override=sample_size_override, target_noise=target_noise,
        max_forward_batch=max_forward_batch,
    )


def evaluate_xy(X_all: np.ndarray, y_all: np.ndarray, model, vocab,
                device: torch.device, rng: np.random.Generator,
                n_bags: int = N_BAGS,
                use_tpsr: bool = False, tpsr_sims: int = 3,
                tpsr_c_puct: float = 1.0, tpsr_top_k: int = 20,
                tpsr_n_bags: int = 3, tpsr_prior_temperature: float = 1.0,
                tpsr_horizon: int = 40, tpsr_rollout_beam_width: int = 1,
                tpsr_no_search_bfgs: bool = True,
                tpsr_warm_start: bool = True,
                tpsr_warm_start_beam_width: int = 10,
                use_sampling: bool = False, n_samples: int = 10,
                temperature: float = 1.0, top_p: float = 1.0,
                no_split: bool = False,
                use_bfgs: bool = True,
                beam_size: int = BEAM_SIZE,
                bpe_model=None,
                return_candidates: bool = False,
                use_unscaling: bool = True,
                gt_pn: str = "",
                ood_data: tuple | None = None,
                use_offset_repair: bool = False,
                n_points_override: int | None = None,
                sample_size_override: int | None = None,
                target_noise: float = 0.0,
                max_forward_batch: int = DEFAULT_FORWARD_BATCH):
    """Bagging evaluation given in-memory (X_all, y_all) arrays.

    Identical logic to evaluate_dataset but decoupled from the .tsv.gz loader,
    so it can be reused by other harnesses (e.g. eval_mymodels.py --benchmark llmsrbench). X_all is
    (N, D) raw inputs, y_all is (N,). Returns (best_formula, best_r2).

    If return_candidates=True, returns (best_formula, best_r2, candidates) where
    candidates is the full post-BFGS refined pool of formula strings (raw input
    space), for diagnostic re-ranking under alternative criteria.
    """
    N = X_all.shape[0]
    # Cap how many rows to load.  A larger pool means more distinct rows for
    # bootstrap sampling (more diverse bags), so we load up to n_bags *
    # SAMPLE_SIZE / 0.75 rows.  When no_split=True all data is training data
    # (matches e2e: no held-out test split, BFGS and evaluation use the same set).
    if n_points_override is not None:
        n_points = n_points_override
    elif no_split:
        # Load a large fixed budget for BFGS/evaluation; the model input bag is
        # still only SAMPLE_SIZE points (drawn by bootstrap from the full pool).
        n_points = 20_000
    else:
        n_points = int(np.ceil(n_bags * SAMPLE_SIZE / 0.75))
    if N > n_points:
        idx = rng.choice(N, size=n_points, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]
        N = n_points

    if no_split:
        # Use all data for both fitting and evaluation (e2e convention).
        n_train = N
        X_train = X_test = X_all
        y_train = y_test = y_all
    else:
        # 75/25 train/test split -- bags and BFGS fit on train, R^2 on test.
        n_train = int(0.75 * N)
        X_train, X_test = X_all[:n_train], X_all[n_train:]
        y_train, y_test = y_all[:n_train], y_all[n_train:]

    # -- Target-noise injection (SRBench / TPSR convention) ------------------
    # Add Gaussian noise to the TRAINING targets only, std proportional to the RMS
    # of the (clean) training target:  y_train += N(0, target_noise * sqrt(mean(y^2))).
    # This is byte-for-byte the convention of srbench/experiment/evaluate_model.py
    # (y_train_scaled += normal(0, target_noise*sqrt(mean(y^2)))) and TPSR/evaluate.py,
    # so a run at target_noise=tau is directly comparable to the SRBench feather rows
    # with target_noise==tau (and to the E2E/TPSR baselines evaluated at the same tau).
    #   * 75/25 split: the held-out y_test stays CLEAN, so the reported R^2 (and the
    #     OOD R^2, scored downstream on separate clean data) measure recovery of the
    #     true signal from noisy training data -- exactly SRBench's r2_test at noise tau.
    #   * no_split (e2e / LLM-SRBench): all data is training data, so the fit AND the
    #     in-process candidate selection must see the same noisy targets; the real
    #     held-out test/OOD is scored on clean arrays by the caller (eval_mymodels).
    # The draw uses the per-dataset `rng`, so runs are reproducible; target_noise==0
    # consumes no randomness and is bit-identical to the noise-free pipeline.
    if target_noise and target_noise > 0.0 and len(y_train) > 0:
        _y_rms = float(np.sqrt(np.mean(np.square(y_train.astype(np.float64)))))
        if _y_rms > 0.0:
            y_train = y_train + rng.normal(0.0, target_noise * _y_rms, size=y_train.shape)
            if no_split:
                y_test = y_train   # e2e convention: fit set == select set, both noisy

    bfgs_idx = rng.choice(n_train, size=min(BFGS_DOWNSAMPLE, n_train), replace=False)
    X_train_bfgs = X_train[bfgs_idx].astype(np.float64)
    if X_train_bfgs.shape[1] < len(V):
        X_train_bfgs = np.concatenate(
            [X_train_bfgs, np.zeros((len(bfgs_idx), len(V) - X_train_bfgs.shape[1]), dtype=np.float64)],
            axis=1,
        )
    y_train_bfgs = y_train[bfgs_idx].astype(np.float64)

    _nv = len(V)
    X_train64 = X_train.astype(np.float64)
    if X_train64.shape[1] < _nv:
        X_train64 = np.concatenate(
            [X_train64, np.zeros((X_train64.shape[0], _nv - X_train64.shape[1]), dtype=np.float64)], axis=1)
    X_test64 = X_test.astype(np.float64)
    if X_test64.shape[1] < _nv:
        X_test64 = np.concatenate(
            [X_test64, np.zeros((X_test64.shape[0], _nv - X_test64.shape[1]), dtype=np.float64)], axis=1)
    y_train64     = y_train.astype(np.float64)
    y_test64      = y_test.astype(np.float64)
    y_test_ss_tot = float(np.sum((y_test64 - np.mean(y_test64)) ** 2))

    # --sample-size reaches this path via sample_size_override.  It used to not: the
    # override existed only on _multiseed_batched / _multidataset_batched, so every
    # caller routed through here (ALL of LLM-SRBench, and SRBench whenever
    # _use_batched is False -- i.e. --n_seeds 1, --tpsr, or --no-split) silently ran
    # 200-point bags no matter what --sample-size said.  That under-fed the models
    # trained with num_io_pairs_range=[50, 400].
    sample_size  = min(sample_size_override or SAMPLE_SIZE, n_train)
    n_vars       = model.patch_dim - 1
    ckpt_max_fml_len = model.max_seq_length - 1
    D = X_all.shape[1]

    # Training target is always the raw formula (data_process.py generates y from
    # raw inputs, even though the transformer sees whitened X as context).  So
    # whether or not we do unscaling, scoring/BFGS arrays stay in raw space.

    # ----------------------------------------------------------------
    # Phase 1: collect ALL candidates from every bag, then pool,
    # deduplicate by skeleton, pre-score, and BFGS the top-N_REFINE.
    # Two modes:
    #   beam (default) -- batched beam search across all bags (one GPU call
    #                    per max_forward_batch bags; fast).
    #   tpsr           -- per-bag MCTS with P-UCT, transformer as policy +
    #                    rollout agent; sequential but deeper exploration.
    # ----------------------------------------------------------------
    # Each entry: (formula_str, X_sample_float64, y_sample_float64)
    all_candidates: list = []
    # TPSR only: skeleton -> "bag<i>:<phase>" naming the search phase that first
    # emitted it (warm_start / rollout@step / commit@step).  Used to attribute
    # the winning formula to where the search actually found it.
    tpsr_provenance: dict = {}
    # Maps formula_skeleton(candidate) -> (raw_formula, unscaled_formula)
    # for the first occurrence of each skeleton -- used to print the winner's lineage.
    raw_map: dict = {}

    # Collect per-bag IO tensors and metadata first.
    bag_ios: list  = []
    bag_meta: list = []   # (X_s64, y_s64, sample_mean, sample_std)

    # Bootstrap bagging: each bag is an independent random sample (with
    # replacement) of size SAMPLE_SIZE from X_train.  This gives full
    # diversity across all n_bags even when the dataset is small, unlike
    # consecutive slicing which exhausts after n_train/SAMPLE_SIZE tiles.
    _effective_n_bags = tpsr_n_bags if use_tpsr else n_bags

    for _ in range(_effective_n_bags):
        idx = rng.integers(0, n_train, size=sample_size)
        X_sample = X_train[idx]
        y_sample = y_train[idx]

        sample_mean = np.mean(X_sample, axis=0)
        sample_std  = np.std(X_sample, axis=0) + 1e-8

        bag_ios.append(build_ios_tensor(X_sample, y_sample, n_vars=n_vars))

        X_s64 = X_sample.astype(np.float64)
        if X_s64.shape[1] < len(V):
            X_s64 = np.concatenate(
                [X_s64, np.zeros((X_s64.shape[0], len(V) - X_s64.shape[1]), dtype=np.float64)],
                axis=1,
            )
        y_s64 = y_sample.astype(np.float64)
        bag_meta.append((X_s64, y_s64, sample_mean, sample_std))

    if use_tpsr:
        # ---- TPSR path: per-bag MCTS ----
        tpsr_searcher = TPSR(
            model, vocab, device,
            n_simulations=tpsr_sims,
            horizon=tpsr_horizon,
            c_puct=tpsr_c_puct,
            top_k=tpsr_top_k,
            prior_temperature=tpsr_prior_temperature,
            rollout_beam_width=tpsr_rollout_beam_width,
            bpe_model=bpe_model,
            verbose=True,
            use_unscaling=use_unscaling,
            use_bfgs=use_bfgs,
            bfgs_in_search=not tpsr_no_search_bfgs,
            warm_start=tpsr_warm_start,
            warm_start_beam_width=tpsr_warm_start_beam_width,
        )
        # MCTS explores deeply on a single IO context; using many bags
        # gives diminishing returns but multiplies cost linearly.
        tpsr_bags_ios  = bag_ios[:tpsr_n_bags]
        tpsr_bags_meta = bag_meta[:tpsr_n_bags]
        for _bag_i, (bag_tensor, (X_s64, y_s64, smean, sstd)) in enumerate(
                zip(tpsr_bags_ios, tpsr_bags_meta)):
            formulas = tpsr_searcher.search(
                bag_tensor, X_s64, y_s64, smean, sstd, D,
                X_reward=X_train_bfgs, y_reward=y_train_bfgs,
            )
            for _skel, _src in tpsr_searcher.last_skel_source.items():
                tpsr_provenance.setdefault(_skel, f"bag{_bag_i}:{_src}")
            for formula in formulas:
                if bpe_model is not None:
                    formula = expand_bpe_tokens(formula, bpe_model)
                all_candidates.append((formula, X_s64, y_s64))
    elif use_sampling:
        # ---- Stochastic sampling path + greedy baseline (matches e2e) ----
        # Greedy (beam_size=1) is always included as a deterministic baseline,
        # then n_samples stochastic draws are added -- matching ModelWrapper.forward()
        # which runs greedy first and merges it with beam/sampling candidates.
        pad_idx = vocab.index("<pad>")

        greedy_list = []
        sample_list = []
        for chunk_start in range(0, len(bag_ios), max_forward_batch):
            chunk = bag_ios[chunk_start:chunk_start + max_forward_batch]
            ios_batch = torch.cat(chunk, dim=0)
            greedy_list.append(beam_search_all_beams(
                model, ios_batch, vocab,
                max_fml_len=ckpt_max_fml_len,
                device=device,
                beam_size=1,
            ).cpu())   # (chunk_size, 1, seq_len)
            sample_list.append(sample_sequences(
                model, ios_batch, vocab,
                max_fml_len=ckpt_max_fml_len,
                device=device,
                n_samples=n_samples,
                temperature=temperature,
                top_p=top_p,
            ).cpu())   # (chunk_size, n_samples, seq_len)

        # Pad and concatenate greedy + samples along the beam dimension.
        max_gl = max(s.shape[2] for s in greedy_list)
        max_sl = max(s.shape[2] for s in sample_list)
        max_seq_len = max(max_gl, max_sl)
        greedy_batch = torch.cat([
            torch.nn.functional.pad(s, (0, max_seq_len - s.shape[2]), value=pad_idx)
            for s in greedy_list], dim=0)   # (n_bags, 1, seq_len)
        sample_batch = torch.cat([
            torch.nn.functional.pad(s, (0, max_seq_len - s.shape[2]), value=pad_idx)
            for s in sample_list], dim=0)   # (n_bags, n_samples, seq_len)
        all_seqs_batch = torch.cat([greedy_batch, sample_batch], dim=1)  # (n_bags, 1+n_samples, seq_len)

        for bag_i, (X_s64, y_s64, sample_mean, sample_std) in enumerate(bag_meta):
            _, pred_fmls = idx_to_toks(all_seqs_batch[bag_i], vocab)
            for formula in pred_fmls:
                if bpe_model is not None:
                    formula = expand_bpe_tokens(formula, bpe_model)
                raw = sanitize_formula(formula)
                if raw:
                    candidate = apply_input_unscaling(raw, sample_mean, sample_std, D) if use_unscaling else raw
                    skel = formula_skeleton(candidate)
                    if skel not in raw_map:
                        raw_map[skel] = (raw, candidate)
                    all_candidates.append((candidate, X_s64, y_s64))
    else:
        # ---- Beam search path (default): batched across bags ----
        all_seqs_list = []
        pad_idx = vocab.index("<pad>")
        for chunk_start in range(0, len(bag_ios), max_forward_batch):
            chunk = bag_ios[chunk_start:chunk_start + max_forward_batch]
            ios_batch = torch.cat(chunk, dim=0)
            seqs = beam_search_all_beams(
                model, ios_batch, vocab,
                max_fml_len=ckpt_max_fml_len,
                device=device,
                beam_size=beam_size,
            )  # (len(chunk), beam_size, seq_len)
            all_seqs_list.append(seqs.cpu())

        # Pad all chunks to the same seq_len before concatenating.
        max_seq_len = max(s.shape[2] for s in all_seqs_list)
        padded_seqs = [
            torch.nn.functional.pad(s, (0, max_seq_len - s.shape[2]), value=pad_idx)
            for s in all_seqs_list
        ]
        all_seqs_batch = torch.cat(padded_seqs, dim=0)  # (n_bags, BEAM_SIZE, seq_len)

        for bag_i, (X_s64, y_s64, sample_mean, sample_std) in enumerate(bag_meta):
            _, pred_fmls = idx_to_toks(all_seqs_batch[bag_i], vocab)
            for formula in pred_fmls:
                if bpe_model is not None:
                    formula = expand_bpe_tokens(formula, bpe_model)
                raw = sanitize_formula(formula)
                if not raw:
                    continue
                candidate = apply_input_unscaling(raw, sample_mean, sample_std, D) if use_unscaling else raw
                skel = formula_skeleton(candidate)
                if skel not in raw_map:
                    raw_map[skel] = (raw, candidate)
                all_candidates.append((candidate, X_s64, y_s64))

    if not all_candidates:
        return ("", 0.0, []) if return_candidates else ("", 0.0)

    # Deduplicate by structural skeleton (float literals -> "1").
    seen_skeletons: dict = {}
    unique_candidates: list = []
    for formula, X_s, y_s in all_candidates:
        skel = formula_skeleton(formula)
        if skel not in seen_skeletons:
            seen_skeletons[skel] = True
            unique_candidates.append((formula, X_s, y_s))

    # Pre-score by actual R^2 on train: formulas already close to the right
    # constants rank above those that merely correlate directionally.
    y_train_bfgs_ss_tot = float(np.sum((y_train_bfgs - np.mean(y_train_bfgs)) ** 2))
    scored = [
        (compute_r2(f, None, None, _x64=X_train_bfgs, _y64=y_train_bfgs, _ss_tot=y_train_bfgs_ss_tot), f, X_s, y_s)
        for f, X_s, y_s in unique_candidates
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    # BFGS-refine the top-N_REFINE unique skeletons; collect both original and
    # refined candidates, then re-rank all by R^2 on held-out test (matching
    # eval_e2e.py order_candidates with metric="r2").
    best_formula = ""
    best_r2      = -1.0

    refined_candidates: list[str] = []
    top_formulas = [formula for _, formula, _, _ in scored[:N_REFINE]]
    if not use_bfgs:
        # BFGS constant-refinement disabled (--no-bfgs): rank the model's emitted
        # formulas as-is.  The mantissa-free grammar produces explicit constant
        # subexpressions, so there are no CONST placeholders for BFGS to fit.
        refined_candidates = list(top_formulas)
    elif _GPU_BFGS_REFINE_BATCH is not None:
        # GPU VarPro: refine all top-N skeletons in a single batched call so the
        # per-formula lock + X/y-upload overhead is paid once instead of N times.
        refined = _GPU_BFGS_REFINE_BATCH(top_formulas, X_train_bfgs, y_train_bfgs)
        for orig, ref in zip(top_formulas, refined):
            refined_candidates.append(orig)
            if ref and ref != orig:
                refined_candidates.append(ref)
    else:
        bfgs_inputs = [(formula, X_train_bfgs, y_train_bfgs) for formula in top_formulas]
        with ThreadPoolExecutor(max_workers=min(len(bfgs_inputs), os.cpu_count() or 4)) as pool:
            for orig, refined in pool.map(_bfgs_refine, bfgs_inputs):
                refined_candidates.append(orig)
                if refined:
                    refined_candidates.append(refined)

    for candidate in refined_candidates:
        r2 = compute_r2(candidate, X_test, y_test, _x64=X_test64, _y64=y_test64, _ss_tot=y_test_ss_tot)
        if r2 > best_r2:
            best_r2      = r2
            best_formula = candidate

    # Polish the winner: re-run BFGS on the best formula with 4* more data
    # and a longer budget. Only adopted if the polished version scores higher.
    if use_bfgs and best_formula:
        polish_n = min(4096, n_train)
        polish_idx = rng.choice(n_train, size=polish_n, replace=False)
        X_polish = X_train64[polish_idx]
        y_polish = y_train64[polish_idx]
        try:
            polished = bfgs_refine_mfg(
                best_formula, X_polish, y_polish, stop_after=30, n_restarts=16)
            r2_polished = compute_r2(
                polished, X_test, y_test,
                _x64=X_test64, _y64=y_test64, _ss_tot=y_test_ss_tot)
            if r2_polished > best_r2:
                best_r2      = r2_polished
                best_formula = polished
        except Exception:
            pass

    # ---- offset repair: recover a missing +1/-1 that BFGS structurally cannot ----
    # Fires only on the near-miss band.  Unlike the candidate re-ranking above (which
    # scores on the held-out TEST slice), this decides on a validation split carved out
    # of TRAIN, so it never consults the reported metric.  See offset_repair.py.
    if use_offset_repair and best_formula:
        try:
            import offset_repair as _orep

            def _r2_tr(f, X, y):
                return compute_r2(f, None, None,
                                  _x64=X.astype(np.float64), _y64=y.astype(np.float64))

            def _ref_full(f, X, y):
                return bfgs_refine_mfg(f, X.astype(np.float64), y.astype(np.float64))

            def _ref_fast(f, X, y):
                return bfgs_refine_mfg(f, X.astype(np.float64), y.astype(np.float64),
                                       stop_after=1, n_restarts=1)

            _rep, _rinfo = _orep.repair(
                best_formula, X_train64, y_train64, _r2_tr, _ref_full,
                screen_refine_fn=_ref_fast, max_len=ckpt_max_fml_len, verbose=True)
            if _rinfo.get("improved") and _rep and _rep != best_formula:
                best_formula = _rep
                best_r2 = compute_r2(best_formula, X_test, y_test,
                                     _x64=X_test64, _y64=y_test64, _ss_tot=y_test_ss_tot)
        except Exception as _e:
            print(f"  [offset-repair] skipped ({_e!r})", flush=True)

    # OOD evaluation: score best_formula on pre-generated OOD data.
    ood_r2 = None
    if ood_data is not None and best_formula:
        X_ood, y_ood = ood_data
        _nv = len(V)
        X_ood64 = X_ood.astype(np.float64)
        if X_ood64.shape[1] < _nv:
            X_ood64 = np.concatenate(
                [X_ood64, np.zeros((X_ood64.shape[0], _nv - X_ood64.shape[1]),
                                   dtype=np.float64)], axis=1)
        ood_r2 = compute_r2(best_formula, None, None, _x64=X_ood64, _y64=y_ood)

    best_skel = formula_skeleton(best_formula)
    _raw_pred = ""
    if best_skel in raw_map:
        _raw_pred = raw_map[best_skel][0]
    else:
        # bfgs_refine_mfg may have prepended '* NUMBER' to a constant-free formula.
        # Try the tail (everything after '* NUMBER') as a fallback lookup.
        _toks = best_formula.split()
        if len(_toks) >= 3 and _toks[0] == '*':
            try:
                float(_toks[1])
                _tail_skel = formula_skeleton(' '.join(_toks[2:]))
                if _tail_skel in raw_map:
                    _raw_pred = raw_map[_tail_skel][0]
            except ValueError:
                pass
    print(f"  pred:  {_raw_pred or best_formula}", flush=True)
    print(f"  +BFGS: {best_formula}", flush=True)
    if use_tpsr and best_formula and tpsr_provenance:
        _cats = {"warm_start": 0, "rollout": 0, "commit": 0, "init": 0}
        for _src in tpsr_provenance.values():
            for _c in _cats:
                if _c in _src:
                    _cats[_c] += 1
                    break
        _win_src = tpsr_provenance.get(formula_skeleton(best_formula))
        if _win_src is None:
            # bfgs_refine_mfg may have prepended '* NUMBER' to a constant-free
            # formula; retry the tail (everything after '* NUMBER').
            _toks = best_formula.split()
            if len(_toks) >= 3 and _toks[0] == '*':
                try:
                    float(_toks[1])
                    _win_src = tpsr_provenance.get(formula_skeleton(' '.join(_toks[2:])))
                except ValueError:
                    pass
        print(f"  TPSR provenance: warm_start={_cats['warm_start']} "
              f"rollout={_cats['rollout']} commit={_cats['commit']} "
              f"(unique skeletons={len(tpsr_provenance)})", flush=True)
        print(f"  winner source: {_win_src or 'post-search BFGS/polish (not in search pool)'}",
              flush=True)
    if gt_pn:
        print(f"  GT:    {gt_pn}", flush=True)
    if ood_r2 is not None:
        print(f"  OOD R^2:{ood_r2:.4f}", flush=True)

    if return_candidates:
        return best_formula, best_r2, refined_candidates
    if ood_data is not None:
        return best_formula, best_r2, ood_r2
    return best_formula, best_r2


# ---------------------------------------------------------------------------
# Multi-seed batched evaluation
# ---------------------------------------------------------------------------
def _multiseed_batched(
    X_full: np.ndarray, y_full: np.ndarray, model, vocab,
    device: torch.device, seeds: list,
    n_points_override: "int | None" = None,
    sample_size_override: "int | None" = None,
    n_bags: int = 1,
    use_sampling: bool = True, n_samples: int = 10,
    temperature: float = 1.0, top_p: float = 1.0,
    use_bfgs: bool = True, beam_size: int = BEAM_SIZE,
    bpe_model=None, use_unscaling: bool = True,
    ood_data: "tuple | None" = None,
) -> dict:
    """Evaluate one dataset for *multiple seeds* in a **single** batched GPU call.

    All seeds' IO bags are stacked into (n_seeds, io_dim) so the encoder and
    decoder run once instead of n_seeds times.  BFGS refinement then runs for
    all seeds concurrently via a flat ThreadPoolExecutor across 16 CPU cores.

    Speedup over the sequential loop: ~n_seeds * (gpu_time / total_time) ~ 10-20*.

    Each (seed, dataset) pair gets an independent RNG derived from both the seed
    and a hash of the dataset contents, so results are reproducible and ordering-
    independent (unlike the shared-RNG sequential loop).

    Returns
    -------
    dict: seed -> (formula, r2) or (formula, r2, ood_r2) when ood_data is not None.
    """
    N_full, D = X_full.shape
    _nv      = len(V)
    n_vars          = model.patch_dim - 1
    ckpt_max_fml_len = model.max_seq_length - 1
    pad_idx         = vocab.index("<pad>")

    # ---------- Phase 1: per-seed data preparation (CPU, fast) ----------
    per_seed_state: list = []
    bag_ios: list        = []

    for seed in seeds:
        # Independent RNG per (seed, dataset-hash) so order doesn't matter.
        rng = np.random.default_rng([seed, int(abs(hash(X_full.tobytes()[:64])) % (2**31))])

        # Subsample n_points from the full dataset.
        n_pts = n_points_override if n_points_override is not None else int(np.ceil(n_bags * SAMPLE_SIZE / 0.75))
        if N_full > n_pts:
            idx = rng.choice(N_full, size=n_pts, replace=False)
            X_all, y_all = X_full[idx], y_full[idx]
        else:
            X_all, y_all = X_full.copy(), y_full.copy()
        N = len(y_all)

        # 75/25 train/test split.
        n_train     = int(0.75 * N)
        X_train, X_test = X_all[:n_train], X_all[n_train:]
        y_train, y_test = y_all[:n_train], y_all[n_train:]

        _ss_cap = sample_size_override if sample_size_override is not None else SAMPLE_SIZE
        sample_size = min(_ss_cap, n_train)

        # Bootstrap bag (1 bag).
        idx_bag  = rng.integers(0, n_train, size=sample_size)
        X_sample = X_train[idx_bag]
        y_sample = y_train[idx_bag]
        sample_mean = np.mean(X_sample, axis=0)
        sample_std  = np.std(X_sample,  axis=0) + 1e-8

        bag_ios.append(build_ios_tensor(X_sample, y_sample, n_vars=n_vars))

        # Pre-pad float64 arrays for compute_r2 / BFGS.
        def _pad64(arr):
            a = arr.astype(np.float64)
            if a.shape[1] < _nv:
                a = np.concatenate([a, np.zeros((a.shape[0], _nv - a.shape[1]), dtype=np.float64)], axis=1)
            return a

        X_s64  = _pad64(X_sample)
        y_s64  = y_sample.astype(np.float64)

        bfgs_idx = rng.choice(n_train, size=min(BFGS_DOWNSAMPLE, n_train), replace=False)
        X_bfgs   = _pad64(X_train[bfgs_idx])
        y_bfgs   = y_train[bfgs_idx].astype(np.float64)

        X_test64       = _pad64(X_test)
        y_test64       = y_test.astype(np.float64)
        y_test_ss_tot  = float(np.sum((y_test64 - np.mean(y_test64)) ** 2))

        X_train64 = _pad64(X_train)
        y_train64 = y_train.astype(np.float64)

        per_seed_state.append({
            "seed": seed, "rng": rng, "D": D,
            "sample_mean": sample_mean, "sample_std": sample_std,
            "X_s64": X_s64, "y_s64": y_s64,
            "X_bfgs": X_bfgs, "y_bfgs": y_bfgs,
            "X_test64": X_test64, "y_test64": y_test64,
            "y_test_ss_tot": y_test_ss_tot,
            "n_train": n_train, "X_train64": X_train64, "y_train64": y_train64,
        })

    # ---------- Phase 2: batched GPU inference (all seeds in one call) ----------
    ios_batch = torch.cat(bag_ios, dim=0)          # (total_items, io_dim)

    if use_sampling:
        greedy_seqs = beam_search_all_beams(
            model, ios_batch, vocab, ckpt_max_fml_len, device, beam_size=1)   # (n_seeds,1,L)
        sample_seqs = sample_sequences(
            model, ios_batch, vocab, ckpt_max_fml_len, device,
            n_samples=n_samples, temperature=temperature, top_p=top_p)         # (n_seeds,S,L)
        L = max(greedy_seqs.shape[2], sample_seqs.shape[2])
        greedy_seqs = torch.nn.functional.pad(greedy_seqs, (0, L - greedy_seqs.shape[2]), value=pad_idx)
        sample_seqs = torch.nn.functional.pad(sample_seqs, (0, L - sample_seqs.shape[2]), value=pad_idx)
        all_seqs = torch.cat([greedy_seqs, sample_seqs], dim=1)                # (n_seeds,1+S,L)
    else:
        all_seqs = beam_search_all_beams(
            model, ios_batch, vocab, ckpt_max_fml_len, device, beam_size=beam_size)  # (n_seeds,B,L)

    # ---------- Phase 3: decode formulas per seed (CPU) ----------
    per_seed_candidates: list = []
    for si, state in enumerate(per_seed_state):
        _, pred_fmls = idx_to_toks(all_seqs[si], vocab)
        candidates = []
        for formula in pred_fmls:
            if bpe_model is not None:
                formula = expand_bpe_tokens(formula, bpe_model)
            raw = sanitize_formula(formula)
            if not raw:
                continue
            cand = (apply_input_unscaling(raw, state["sample_mean"], state["sample_std"], state["D"])
                    if use_unscaling else raw)
            candidates.append((cand, state["X_s64"], state["y_s64"]))
        per_seed_candidates.append((candidates, state))

    # ---------- Phase 4: BFGS refinement -- all seeds in parallel ----------
    def _refine_one_seed(packed):
        candidates, state = packed
        X_bfgs        = state["X_bfgs"]
        y_bfgs        = state["y_bfgs"]
        X_test64      = state["X_test64"]
        y_test64      = state["y_test64"]
        y_test_ss_tot = state["y_test_ss_tot"]
        n_train       = state["n_train"]
        rng           = state["rng"]
        X_train64     = state["X_train64"]
        y_train64     = state["y_train64"]

        if not candidates:
            return "", 0.0

        # Deduplicate by skeleton.
        seen: dict = {}
        unique = [(f, Xs, ys) for f, Xs, ys in candidates
                  if formula_skeleton(f) not in seen and not seen.update({formula_skeleton(f): True})]  # type: ignore[func-returns-value]

        # Pre-score on BFGS data.
        y_ss = float(np.sum((y_bfgs - np.mean(y_bfgs)) ** 2))
        scored = sorted(
            [(compute_r2(f, None, None, _x64=X_bfgs, _y64=y_bfgs, _ss_tot=y_ss), f, Xs, ys)
             for f, Xs, ys in unique],
            key=lambda x: x[0], reverse=True)

        top_formulas = [f for _, f, _, _ in scored[:N_REFINE]]
        refined: list[str] = []
        if use_bfgs and top_formulas:
            bfgs_in = [(f, X_bfgs, y_bfgs) for f in top_formulas]
            with ThreadPoolExecutor(max_workers=min(len(bfgs_in), 4)) as p:
                for orig, ref in p.map(_bfgs_refine, bfgs_in):
                    refined.append(orig)
                    if ref:
                        refined.append(ref)
        else:
            refined = list(top_formulas)

        best_formula, best_r2 = "", -1.0
        for c in refined:
            r2 = compute_r2(c, None, None, _x64=X_test64, _y64=y_test64, _ss_tot=y_test_ss_tot)
            if r2 > best_r2:
                best_r2, best_formula = r2, c

        # Polish with more data.
        if use_bfgs and best_formula:
            polish_n   = min(4096, n_train)
            polish_idx = rng.choice(n_train, size=polish_n, replace=False)
            try:
                polished = bfgs_refine_mfg(
                    best_formula, X_train64[polish_idx], y_train64[polish_idx],
                    stop_after=30, n_restarts=16)
                r2p = compute_r2(polished, None, None, _x64=X_test64, _y64=y_test64, _ss_tot=y_test_ss_tot)
                if r2p > best_r2:
                    best_r2, best_formula = r2p, polished
            except Exception:
                pass

        return best_formula, max(best_r2, 0.0)

    # Run outer-seed workers; each spawns at most 4 inner BFGS threads.
    # Total threads <= n_seeds * 4; on 16 cores scipy's C code runs truly parallel.
    outer_workers = min(len(per_seed_candidates), max(1, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=outer_workers) as pool:
        bfgs_results = list(pool.map(_refine_one_seed, per_seed_candidates))

    # ---------- Phase 5: OOD scoring and result assembly ----------
    result_dict: dict = {}
    for si, state in enumerate(per_seed_state):
        formula, r2 = bfgs_results[si]
        seed = state["seed"]
        if ood_data is not None and formula:
            X_ood, y_ood = ood_data
            X_ood64 = X_ood.astype(np.float64)
            if X_ood64.shape[1] < _nv:
                X_ood64 = np.concatenate(
                    [X_ood64, np.zeros((X_ood64.shape[0], _nv - X_ood64.shape[1]), dtype=np.float64)], axis=1)
            ood_r2 = compute_r2(formula, None, None, _x64=X_ood64, _y64=y_ood)
            result_dict[seed] = (formula, r2, ood_r2)
        else:
            result_dict[seed] = (formula, r2)
    return result_dict


# Datasets batched into one GPU call.  cross-KV scales as
# DATASET_CHUNK * n_seeds * n_samples * enc_seq_len * n_layers so keep this
# at 1 to avoid OOM (900 sequences = 7.5 GB; 100 sequences = 2.7 GB).
# Increase only if you have a larger GPU.
DATASET_CHUNK = 1


def _multidataset_batched(
    dataset_items: list,          # list of (X_full, y_full, seeds_needed, ood_data)
    model, vocab,
    device: torch.device,
    n_points_override: "int | None" = None,
    sample_size_override: "int | None" = None,
    n_bags: int = 1,
    use_sampling: bool = True, n_samples: int = 10,
    temperature: float = 1.0, top_p: float = 1.0,
    use_bfgs: bool = True, beam_size: int = BEAM_SIZE,
    bpe_model=None, use_unscaling: bool = True,
    target_noise: float = 0.0,
) -> list:
    """One GPU call for multiple datasets * multiple seeds.

    Flattens every (dataset, seed) pair into a single batched inference call,
    then runs BFGS for all pairs concurrently in one ThreadPoolExecutor.

    Parameters
    ----------
    dataset_items : list of (X_full, y_full, seeds_needed, ood_data)
        ood_data may be None for datasets without OOD test points.

    Returns
    -------
    list[dict] : one dict per input dataset_item, each mapping
        seed -> (formula, r2) or (formula, r2, ood_r2).
    """
    _nv             = len(V)
    n_vars          = model.patch_dim - 1
    ckpt_max_fml_len = model.max_seq_length - 1
    pad_idx         = vocab.index("<pad>")

    # ---------- Phase 1: per-(dataset, seed) data preparation ----------
    flat_bag_ios: list = []   # io tensors in flat order
    flat_state:   list = []   # per-item state dicts
    flat_key:     list = []   # (ds_idx, seed) for each flat entry

    for ds_idx, (X_full, y_full, seeds_needed, _ood_data) in enumerate(dataset_items):
        N_full = X_full.shape[0]

        def _pad64(arr, nv=_nv):
            a = arr.astype(np.float64)
            if a.shape[1] < nv:
                a = np.concatenate([a, np.zeros((a.shape[0], nv - a.shape[1]), dtype=np.float64)], axis=1)
            return a

        for seed in seeds_needed:
            rng = np.random.default_rng([seed, int(abs(hash(X_full.tobytes()[:64])) % (2**31))])

            n_pts = n_points_override if n_points_override is not None else int(np.ceil(n_bags * SAMPLE_SIZE / 0.75))
            if N_full > n_pts:
                idx = rng.choice(N_full, size=n_pts, replace=False)
                X_all, y_all = X_full[idx], y_full[idx]
            else:
                X_all, y_all = X_full.copy(), y_full.copy()
            N = len(y_all)

            n_train     = int(0.75 * N)
            X_train, X_test = X_all[:n_train], X_all[n_train:]
            y_train, y_test = y_all[:n_train], y_all[n_train:]

            # Target-noise on the training target only (SRBench/TPSR convention);
            # held-out y_test stays clean.  See evaluate_xy for the rationale.
            if target_noise and target_noise > 0.0 and len(y_train) > 0:
                _y_rms = float(np.sqrt(np.mean(np.square(y_train.astype(np.float64)))))
                if _y_rms > 0.0:
                    y_train = y_train + rng.normal(0.0, target_noise * _y_rms, size=y_train.shape)

            _ss_cap = sample_size_override if sample_size_override is not None else SAMPLE_SIZE
            sample_size = min(_ss_cap, n_train)
            idx_bag     = rng.integers(0, n_train, size=sample_size)
            X_sample    = X_train[idx_bag]
            y_sample    = y_train[idx_bag]
            sample_mean = np.mean(X_sample, axis=0)
            sample_std  = np.std(X_sample,  axis=0) + 1e-8

            flat_bag_ios.append(build_ios_tensor(X_sample, y_sample, n_vars=n_vars))

            bfgs_idx = rng.choice(n_train, size=min(BFGS_DOWNSAMPLE, n_train), replace=False)

            flat_state.append({
                "seed": seed, "ds_idx": ds_idx, "rng": rng, "D": X_full.shape[1],
                "sample_mean": sample_mean, "sample_std": sample_std,
                "X_s64": _pad64(X_sample), "y_s64": y_sample.astype(np.float64),
                "X_bfgs": _pad64(X_train[bfgs_idx]),
                "y_bfgs": y_train[bfgs_idx].astype(np.float64),
                "X_test64": _pad64(X_test), "y_test64": y_test.astype(np.float64),
                "y_test_ss_tot": float(np.sum((y_test.astype(np.float64) - y_test.mean()) ** 2)),
                "n_train": n_train,
                "X_train64": _pad64(X_train), "y_train64": y_train.astype(np.float64),
            })
            flat_key.append((ds_idx, seed))

    if not flat_bag_ios:
        return [{} for _ in dataset_items]

    # ---------- Phase 2: batched GPU inference (all items in one call) ----------
    ios_batch = torch.cat(flat_bag_ios, dim=0)   # (total_items, io_dim)
    n_items   = len(flat_bag_ios)

    if use_sampling:
        greedy_seqs = beam_search_all_beams(
            model, ios_batch, vocab, ckpt_max_fml_len, device, beam_size=1)   # (n_items,1,L)
        sample_seqs = sample_sequences(
            model, ios_batch, vocab, ckpt_max_fml_len, device,
            n_samples=n_samples, temperature=temperature, top_p=top_p)         # (n_items,S,L)
        L = max(greedy_seqs.shape[2], sample_seqs.shape[2])
        greedy_seqs = torch.nn.functional.pad(greedy_seqs, (0, L - greedy_seqs.shape[2]), value=pad_idx)
        sample_seqs = torch.nn.functional.pad(sample_seqs, (0, L - sample_seqs.shape[2]), value=pad_idx)
        all_seqs = torch.cat([greedy_seqs, sample_seqs], dim=1)                # (n_items,1+S,L)
    else:
        all_seqs = beam_search_all_beams(
            model, ios_batch, vocab, ckpt_max_fml_len, device, beam_size=beam_size)

    # ---------- Phase 3: decode formulas for each flat item ----------
    flat_candidates: list = []
    for fi, state in enumerate(flat_state):
        _, pred_fmls = idx_to_toks(all_seqs[fi], vocab)
        candidates = []
        for formula in pred_fmls:
            if bpe_model is not None:
                formula = expand_bpe_tokens(formula, bpe_model)
            raw = sanitize_formula(formula)
            if not raw:
                continue
            cand = (apply_input_unscaling(raw, state["sample_mean"], state["sample_std"], state["D"])
                    if use_unscaling else raw)
            candidates.append((cand, state["X_s64"], state["y_s64"]))
        flat_candidates.append((candidates, state))

    # ---------- Phase 4: BFGS -- all (dataset, seed) pairs in one pool ----------
    def _refine(packed):
        candidates, state = packed
        X_bfgs        = state["X_bfgs"]
        y_bfgs        = state["y_bfgs"]
        X_test64      = state["X_test64"]
        y_test64      = state["y_test64"]
        y_test_ss_tot = state["y_test_ss_tot"]
        n_train       = state["n_train"]
        rng           = state["rng"]
        X_train64     = state["X_train64"]
        y_train64     = state["y_train64"]

        if not candidates:
            return "", 0.0

        seen: dict = {}
        unique = [(f, Xs, ys) for f, Xs, ys in candidates
                  if formula_skeleton(f) not in seen and not seen.update({formula_skeleton(f): True})]  # type: ignore[func-returns-value]

        y_ss = float(np.sum((y_bfgs - np.mean(y_bfgs)) ** 2))
        scored = sorted(
            [(compute_r2(f, None, None, _x64=X_bfgs, _y64=y_bfgs, _ss_tot=y_ss), f, Xs, ys)
             for f, Xs, ys in unique],
            key=lambda x: x[0], reverse=True)

        top_formulas = [f for _, f, _, _ in scored[:N_REFINE]]
        refined: list[str] = []
        if use_bfgs and top_formulas:
            bfgs_in = [(f, X_bfgs, y_bfgs) for f in top_formulas]
            with ThreadPoolExecutor(max_workers=min(len(bfgs_in), 4)) as p:
                for orig, ref in p.map(_bfgs_refine, bfgs_in):
                    refined.append(orig)
                    if ref:
                        refined.append(ref)
        else:
            refined = list(top_formulas)

        best_formula, best_r2 = "", -1.0
        for c in refined:
            r2 = compute_r2(c, None, None, _x64=X_test64, _y64=y_test64, _ss_tot=y_test_ss_tot)
            if r2 > best_r2:
                best_r2, best_formula = r2, c

        if use_bfgs and best_formula:
            polish_n   = min(4096, n_train)
            polish_idx = rng.choice(n_train, size=polish_n, replace=False)
            try:
                polished = bfgs_refine_mfg(
                    best_formula, X_train64[polish_idx], y_train64[polish_idx],
                    stop_after=30, n_restarts=16)
                r2p = compute_r2(polished, None, None, _x64=X_test64, _y64=y_test64, _ss_tot=y_test_ss_tot)
                if r2p > best_r2:
                    best_r2, best_formula = r2p, polished
            except Exception:
                pass

        return best_formula, max(best_r2, 0.0)

    # One flat pool for ALL (dataset * seed) pairs -- fills all 16 cores.
    flat_workers = min(n_items, max(1, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=flat_workers) as pool:
        flat_bfgs = list(pool.map(_refine, flat_candidates))

    # ---------- Phase 5: OOD scoring and assemble per-dataset dicts ----------
    result_dicts: list = [{} for _ in dataset_items]
    for fi, (ds_idx, seed) in enumerate(flat_key):
        formula, r2 = flat_bfgs[fi]
        _ood_data = dataset_items[ds_idx][3]
        if _ood_data is not None and formula:
            X_ood, y_ood = _ood_data
            X_ood64 = X_ood.astype(np.float64)
            if X_ood64.shape[1] < _nv:
                X_ood64 = np.concatenate(
                    [X_ood64, np.zeros((X_ood64.shape[0], _nv - X_ood64.shape[1]), dtype=np.float64)], axis=1)
            ood_r2 = compute_r2(formula, None, None, _x64=X_ood64, _y64=y_ood)
            result_dicts[ds_idx][seed] = (formula, r2, ood_r2)
        else:
            result_dicts[ds_idx][seed] = (formula, r2)
    return result_dicts


# ---------------------------------------------------------------------------
# Model name -> checkpoint resolution
# ---------------------------------------------------------------------------

def resolve_model_checkpoint(model_name: str, checkpoints_dir: str = "./checkpoints") -> str:
    """Return the checkpoint path for a named model.

    Looks inside <checkpoints_dir>/<model_name>_res/ in this order:
      1. fixed_weights.pth        (exact)
      2. model_*_new.pth          (glob, sorted; first match)
      3. model_*.pth              (glob, sorted; first match)
    Raises FileNotFoundError if none of the above is found.
    """
    res_dir = os.path.join(checkpoints_dir, f"{model_name}_res")
    fixed = os.path.join(res_dir, "fixed_weights.pth")
    if os.path.isfile(fixed):
        return fixed
    for pattern in ("model_*_new.pth", "model_*.pth"):
        matches = sorted(glob.glob(os.path.join(res_dir, pattern)))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No checkpoint found for model '{model_name}' in {res_dir}. "
        f"Expected fixed_weights.pth, model_*_new.pth, or model_*.pth."
    )


def _available_models(checkpoints_dir: str = "./checkpoints") -> list:
    """Return model names that have at least one recognised checkpoint file."""
    models = []
    if not os.path.isdir(checkpoints_dir):
        return models
    for entry in sorted(os.listdir(checkpoints_dir)):
        if entry.endswith("_res") and os.path.isdir(os.path.join(checkpoints_dir, entry)):
            name = entry[:-4]  # strip "_res"
            try:
                resolve_model_checkpoint(name, checkpoints_dir)
                models.append(name)
            except FileNotFoundError:
                pass
    return models


# ---------------------------------------------------------------------------
# Result persistence helpers
# ---------------------------------------------------------------------------

def load_feynman_targets(
    feynman_csv: str = "./datasets/feynman/FeynmanEquations.csv",
) -> dict:
    """Return {dataset_name: formula_str} from the Feynman equations CSV.

    The CSV 'Filename' column (e.g. 'I.6.2') maps to the PMLB dataset name
    'feynman_I_6_2' by replacing dots with underscores and prepending 'feynman_'.
    """
    targets: dict = {}
    if not os.path.isfile(feynman_csv):
        return targets
    with open(feynman_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname   = row.get("Filename", "").strip()
            formula = row.get("Formula", "").strip()
            if fname and formula:
                dataset_name = "feynman_" + fname.replace(".", "_")
                targets[dataset_name] = formula
    return targets


def load_feynman_pn_targets(
    feynman_csv: str = "./datasets/feynman/FeynmanEquations.csv",
) -> dict:
    """Return {dataset_name: pn_str} -- ground-truth Feynman formulas in canonical PN.

    Converts each CSV formula via feynman_pn_analysis.formula_to_pn + simplify().
    Entries that fail conversion are omitted.
    """
    pn_targets: dict = {}
    if not os.path.isfile(feynman_csv):
        return pn_targets
    with open(feynman_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("Filename", "").strip()
            fml   = row.get("Formula",  "").strip()
            if not fname or not fml:
                continue
            var_names = [row.get(f"v{k}_name", "").strip()
                         for k in range(1, 11)
                         if row.get(f"v{k}_name", "").strip()]
            if not var_names:
                continue
            pn, status = _feynman_to_pn(fml, var_names)
            if pn and "ERR" not in pn:
                dataset_name = "feynman_" + fname.replace(".", "_")
                pn_targets[dataset_name] = pn
    return pn_targets


def _model_name_from_checkpoint(checkpoint_path: str) -> str:
    """Derive a short model name from a checkpoint path.

    Looks for the '<name>_res' directory component first
    (e.g. 'checkpoints/m89_res/...' -> '89M_40_simp1').
    Falls back to the checkpoint file stem.
    """
    parts = checkpoint_path.replace("\\", "/").split("/")
    for part in parts:
        if part.endswith("_res"):
            return part[:-4]
    return os.path.splitext(os.path.basename(checkpoint_path))[0]


# ---------------------------------------------------------------------------
# LLM-SRBench: data loading, prediction and metrics
#
# LLM-SRBench (HF repo `nnheui/llm-srbench`) ships two things:
#   * parquet metadata    -- symbols / descriptions / ground-truth expression
#                            (what `datasets.load_dataset` downloads), and
#   * lsr_bench_data.hdf5 -- the actual (X, y) sample arrays (239 MB), which the
#                            official harness pulls via `snapshot_download` /
#                            `hf_hub_download`, NOT `load_dataset`.
#
# Data layout in the hdf5 (verified):
#   /lsr_transform/<name>       -> {'train': (Ntr, 1+D), 'test': (Nte, 1+D)}
#   /lsr_synth/<domain>/<name>  -> {'train', 'test', 'ood_test'}
#   Column 0 is the OUTPUT y; columns 1: are the input variables X, in the order
#   of `symbols` (symbol_properties[0] == 'O').  NOTE this is the opposite of the
#   SRBench .tsv.gz convention above (y last).
#
# Scoring replicates the official bench/pipelines.py compute_output_base_metrics.
# ---------------------------------------------------------------------------

REPO_ID = "nnheui/llm-srbench"
_NV = len(V)

# Local copy of the benchmark: download once with download_llmsrbench.py and scp
# it to the cluster, so evaluation needs no internet / HF cache at runtime.
# Override the location with $LLMSRBENCH_DIR.
LLMSRBENCH_DIR = os.environ.get(
    "LLMSRBENCH_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "llmsrbench"),
)


def resolve_llmsrbench_file(rel_path: str) -> str:
    """Return a path to a benchmark file, preferring the local $LLMSRBENCH_DIR copy;
    fall back to hf_hub_download (needs internet / a warm HF cache) if absent."""
    local = os.path.join(LLMSRBENCH_DIR, rel_path)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, rel_path, repo_type="dataset")


def load_problems(split: str):
    """Load LLM-SRBench problems for a split.

    Returns a list of dicts: {name, symbols, expression, train, test, ood_test}.
    `train`/`test`/`ood_test` are float64 arrays with column 0 = y, cols 1: = X.
    """
    import h5py
    import pandas as pd

    hdf5_path = resolve_llmsrbench_file("lsr_bench_data.hdf5")
    # Read the split's metadata (name/symbols/expression) straight from the
    # parquet.  This avoids the optional `datasets` dependency -- which is not
    # installed here and, when it is, gets shadowed by the repo's local ./datasets/
    # directory when running from the repo root.
    parquet_path = resolve_llmsrbench_file(f"data/{split}-00000-of-00001.parquet")
    meta = pd.read_parquet(parquet_path)

    # hdf5 group prefix: lsr_transform/<name>  or  lsr_synth/<domain>/<name>
    if split == "lsr_transform":
        group_of = lambda name: f"/lsr_transform/{name}"
    elif split.startswith("lsr_synth_"):
        domain = split[len("lsr_synth_"):]
        group_of = lambda name: f"/lsr_synth/{domain}/{name}"
    else:
        raise ValueError(f"Unknown split: {split}")

    problems = []
    with h5py.File(hdf5_path, "r") as f:
        for _, e in meta.iterrows():
            g = f[group_of(e["name"])]
            samples = {k: g[k][...].astype(np.float64) for k in g.keys()}
            problems.append({
                "name":       e["name"],
                "symbols":    list(e["symbols"]),
                "expression": e["expression"],
                "train":      samples.get("train"),
                "test":       samples.get("test"),
                "ood_test":   samples.get("ood_test"),
            })
    return problems


def predict(formula: str, X: np.ndarray) -> np.ndarray:
    """Evaluate a prefix-notation formula on raw inputs X (N, D).

    Pads X to len(V) columns (the model's variable slots) and returns y_pred,
    with NaN wherever evaluation is invalid (non-empty stack / overflow)."""
    if not formula or not formula.strip():
        return np.full(X.shape[0], np.nan)
    Xp = X.astype(np.float64)
    if Xp.shape[1] < _NV:
        Xp = np.concatenate(
            [Xp, np.zeros((Xp.shape[0], _NV - Xp.shape[1]), dtype=np.float64)], axis=1)
    try:
        preds, stacklefts = wrapExtEvalPN(formula, Xp)
    except Exception:
        return np.full(X.shape[0], np.nan)
    preds = np.asarray(preds, dtype=np.float64)
    if np.any(stacklefts != 0):
        return np.full(X.shape[0], np.nan)
    return preds


def output_metrics(y_pred: np.ndarray, y: np.ndarray) -> dict:
    """Same metrics as the official harness, NaN-robust."""
    mask = np.isfinite(y_pred) & np.isfinite(y)
    n_valid = int(mask.sum())
    if n_valid == 0:
        return {"mse": float("nan"), "nmse": float("nan"), "r2": float("nan"),
                "kdt": float("nan"), "mape": float("nan"), "num_valid_points": 0}
    yp, yt = y_pred[mask], y[mask]
    var = np.var(yt)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    mse = float(np.mean((yt - yp) ** 2))
    nmse = mse / var if var > 0 else float("nan")
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    kdt = mape = float("nan")
    try:
        from scipy.stats import kendalltau
        kdt = float(kendalltau(yt, yp)[0])
    except Exception:
        pass
    try:
        from sklearn.metrics import mean_absolute_percentage_error
        mape = float(mean_absolute_percentage_error(yt, yp))
    except Exception:
        pass
    return {"mse": mse, "nmse": nmse, "r2": r2, "kdt": kdt, "mape": mape,
            "num_valid_points": n_valid}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Defaults that differ per benchmark.  Each flag below parses with default=None
# so "not given" is distinguishable from "given the srbench value"; whichever
# benchmark runs then fills in its own historical default, keeping both runs
# bit-for-bit what they were as two separate scripts.
_BENCH_DEFAULTS = {
    "srbench": {
        "seed": 42, "n_bags": N_BAGS, "beam_size": BEAM_SIZE,
        "tpsr_sims": 3, "tpsr_top_k": 10, "tpsr_rollout_beam_width": 1,
        "tpsr_horizon_primitive": 40,
    },
    "llmsrbench": {
        "seed": 0, "n_bags": 10, "beam_size": 5,
        "tpsr_sims": 10, "tpsr_top_k": 3, "tpsr_rollout_beam_width": 3,
        "tpsr_horizon_primitive": 50,
    },
}


def _apply_bench_defaults(args):
    """Fill in the benchmark-dependent defaults for flags left unset."""
    d = bench_defaults(args, _BENCH_DEFAULTS,
                       ("seed", "n_bags", "beam_size",
                        "tpsr_sims", "tpsr_top_k", "tpsr_rollout_beam_width"))
    args._tpsr_horizon_primitive = d["tpsr_horizon_primitive"]


# Flags only one benchmark's run_* function reads (verified against the run_*
# bodies).  Passing one to the other benchmark is an error -- see bench_cli.
_BENCH_ONLY_FLAGS = {
    "srbench": [
        "dataset", "datasets", "n_points", "no_split", "no_bfgs",
        "no_resume", "failed_only", "ood_data", "offset_repair",
        "sample_n", "temperature", "top_p",
        "gpu_bfgs", "gpu_bfgs_restarts", "gpu_bfgs_iters", "gpu_bfgs_dtype",
        "tpsr_search_bfgs", "tpsr_no_warm_start", "tpsr_warm_start_beam_width",
    ],
    "llmsrbench": ["split", "problem", "max_problems"],
}


def main():
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--checkpoints-dir", default="./checkpoints")
    _pre_args, _ = _pre.parse_known_args()
    _models = _available_models(_pre_args.checkpoints_dir)

    parser = argparse.ArgumentParser(
        description=("Evaluate our VanillaTransformer on SRBench (Feynman PMLB) or "
                     "LLM-SRBench. Pick with --benchmark."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        default="srbench",
        choices=["srbench", "llmsrbench"],
        help="Which benchmark to run. 'srbench' = Feynman PMLB datasets under "
             "--datasets; 'llmsrbench' = LLM-SRBench --split. Default: srbench.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="./checkpoints",
        help="Directory containing <model>_res/ subdirectories. Used with --model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=_models if _models else None,
        help=(
            "Named model to evaluate. Looks for fixed_weights.pth, then "
            "model_*_new.pth, then model_*.pth inside checkpoints/<model>_res/. "
            "Available: " + (", ".join(_models) if _models else "(none found)")
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit path to a model checkpoint (.pth). Ignored when --model is set. "
             "Defaults to the first available model in --checkpoints-dir.",
    )
    parser.add_argument(
        "--datasets",
        default="./datasets/pmlb/datasets",
        help="Root directory containing one sub-folder per dataset",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device string, e.g. 'cpu' or 'cuda:0'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed (seeds used are seed, seed+1, ..., seed+n_seeds-1). "
             "Default: 42 for srbench, 0 for llmsrbench.",
    )
    parser.add_argument(
        "--n-seeds", "--n_seeds",
        dest="n_seeds",
        type=int,
        default=1,
        help="Number of random seeds to score in ONE process, i.e. one checkpoint "
             "load (matches eval_e2e.py default). Rows are keyed (dataset|equation, "
             "seed) and resume per pair.",
    )
    parser.add_argument(
        "--target-noise",
        type=float,
        default=0.0,
        metavar="TAU",
        help="Add Gaussian noise to the TRAINING targets before fitting: "
             "y_train += N(0, TAU*sqrt(mean(y^2))).  Same convention as SRBench "
             "(evaluate_model.py) and TPSR (evaluate.py), so TAUin{0.001,0.01,0.1} "
             "is directly comparable to the ground-truth feather rows at that "
             "target_noise.  The held-out test (and OOD) targets stay clean.  "
             "TAU>0 tags the output file as eval_tf_<model>_noise<TAU>.pkl.gz. "
             "Default: 0.0 (noise-free).",
    )
    parser.add_argument(
        "--results-dir", "--output",
        dest="results_dir",
        default="./results",
        metavar="DIR",
        help="Base directory for result .pkl.gz files. Default: ./results. "
             "Override to give each parallel per-seed run its own directory "
             "(e.g. results/seed7) so concurrent processes don't race on the "
             "same file; merge afterwards with merge_seed_results.py. "
             "srbench writes <DIR>/noise_<TAU>/eval_tf_<model>.pkl.gz; llmsrbench "
             "writes <DIR>/noise_<TAU>/llmsrbench/<model>_<split>[_tag]/results.pkl.gz.",
    )
    parser.add_argument(
        "--max-forward-batch",
        type=int,
        default=DEFAULT_FORWARD_BATCH,
        metavar="N",
        help="Number of IO bags pushed through the GPU per forward pass during "
             "beam search / sampling (the per-dataset n_seeds=1 path). Pure "
             "speed<->memory dial -- results are identical. Higher = faster / more "
             f"VRAM; lower = slower / less VRAM. Default: {DEFAULT_FORWARD_BATCH}.",
    )
    parser.add_argument(
        "--n-bags",
        type=int,
        default=None,
        help="Number of IO bags per dataset. On srbench this also controls how much "
             "data is loaded: ceil(n_bags * SAMPLE_SIZE / 0.75) rows are subsampled so "
             f"the training split contains exactly n_bags bags of {SAMPLE_SIZE} points "
             f"each. Default: {N_BAGS} for srbench, 10 for llmsrbench.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=None,
        metavar="N",
        help="Rows to subsample from each PMLB dataset per seed (before the 75/25 "
             "train/test split). Different seeds draw different subsets, creating "
             "formula variance across seeds. e2e paper uses 200 with --n-bags 1. "
             "Default: None (uses ceil(n_bags * SAMPLE_SIZE / 0.75)).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help="IO bag size: how many (x,y) pairs are fed to the encoder per inference "
             "call. Defaults to SAMPLE_SIZE (200). Set to 400 for the 145M model "
             "(145M_40_simp1/m145_unscale) which was trained with up to 400 IO points. "
             "--n-points must be at least ceil(N/0.75) so the training split can fill "
             "the bag, but that floor (534 at N=400) is a minimum, not a target: the "
             "cluster scripts pass --n-points 20000 to match eval_e2e.py/eval_phye2e.py, "
             "since the pool only sizes BFGS and the held-out R^2, not the bag.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help=f"Beam width for beam search (default: {BEAM_SIZE} for srbench, 5 for "
             "llmsrbench). Total candidates = n_bags * beam_size before deduplication.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="[srbench] Evaluate a single dataset by folder name (e.g. feynman_I_6_2)",
    )
    parser.add_argument(
        "--split",
        default="lsr_transform",
        help="[llmsrbench] Which split to evaluate: lsr_transform or "
             "lsr_synth_<domain>. Default: lsr_transform.",
    )
    parser.add_argument(
        "--problem",
        default=None,
        help="[llmsrbench] Evaluate only this problem (by name).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="[llmsrbench] Cap the number of problems (for quick smoke tests).",
    )
    parser.add_argument(
        "--tpsr",
        action="store_true",
        default=False,
        help="Use TPSR (MCTS + P-UCT) instead of beam search for formula synthesis.",
    )
    parser.add_argument(
        "--tpsr-sims",
        type=int,
        default=None,
        metavar="N",
        help="Number of MCTS simulations per outer-loop step (TPSR mode only). "
             "Paper uses 3 (the fast default). Raise (e.g. 10-25) for more search "
             "at higher cost; each sim rolls the decoder out to completion. "
             "Default: 3 for srbench, 10 for llmsrbench.",
    )
    parser.add_argument(
        "--tpsr-c-puct",
        type=float,
        default=1.0,
        metavar="C",
        help="P-UCT exploration constant (TPSR mode only). Default: 1.0.",
    )
    parser.add_argument(
        "--tpsr-top-k",
        type=int,
        default=None,
        metavar="K",
        help="Max children expanded per MCTS node (TPSR mode only). "
             "Default: 10 for srbench, 3 for llmsrbench.",
    )
    parser.add_argument(
        "--tpsr-n-bags",
        type=int,
        default=3,
        metavar="B",
        help="Number of IO bags to run MCTS on (TPSR mode only). "
             "Beam search benefits from many bags; MCTS explores deeply on few. Default: 3.",
    )
    parser.add_argument(
        "--tpsr-horizon",
        type=int,
        default=None,
        metavar="H",
        help="Max tokens committed by the outer MCTS loop (TPSR mode only). "
             "Auto-detected if omitted: 20 for BPE models (each token encodes a "
             "subtree), else 40 for srbench / 50 for llmsrbench.",
    )
    parser.add_argument(
        "--tpsr-rollout-beam-width",
        type=int,
        default=None,
        metavar="W",
        help="Beam width for rollout completions (TPSR mode only). "
             "1 = greedy (fastest); 3 = paper setting. "
             "Default: 1 for srbench, 3 for llmsrbench.",
    )
    parser.add_argument(
        "--tpsr-search-bfgs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run BFGS inside the TPSR rollout reward. OFF by default (fast): the "
             "post-search BFGS pass still refines the top candidates, so accuracy is "
             "largely preserved while avoiding a BFGS fit per unique skeleton "
             "(~10-50x slower). Pass --tpsr-search-bfgs to re-enable for a stronger "
             "(slower) reward signal. Default: off.",
    )
    parser.add_argument(
        "--tpsr-prior-temperature",
        type=float,
        #default=3.0,
        default=1.0,
        metavar="T",
        help="Softmax temperature applied to model logits before computing P-UCT priors "
             "(TPSR mode only). T > 1 flattens the prior so low-probability tokens "
             "get meaningful exploration. Default: 3.0.",
    )
    parser.add_argument(
        "--tpsr-no-warm-start",
        action="store_true",
        default=False,
        help="Disable warm-starting TPSR from beam search (TPSR mode only). "
             "By default TPSR runs a beam search from the root first, records "
             "every completion (so its candidate set >= beam search's) and seeds "
             "the MCTS tree along each beam path. Pass this for the un-warm-started "
             "ablation.",
    )
    parser.add_argument(
        "--tpsr-warm-start-beam-width",
        type=int,
        default=10,
        metavar="W",
        help="Beam width for the warm-start beam search (TPSR mode only). "
             "Default: 10 (matches --beam-size).",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=0,
        metavar="N",
        help="Stochastic sampling mode: draw N independent sequences per bag instead of "
             "beam search. 0 = disabled (use beam search). e2e paper used N=10. Default: 0.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="T",
        help="Sampling temperature (only with --sample-n). "
             "< 1.0 = more peaked / conservative, > 1.0 = more random. Default: 1.0.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        metavar="P",
        help="Nucleus sampling: keep only the top-P probability mass at each step "
             "(only with --sample-n). 1.0 = disabled. Default: 1.0.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        default=False,
        help="Use all data for BFGS fitting and R^2 evaluation with no held-out test split. "
             "Matches the e2e/symbolicregression convention. Combine with --sample-n and "
             "--n-bags 1 to fully replicate the e2e inference pipeline.",
    )
    parser.add_argument(
        "--bpe-model",
        default=None,
        metavar="PATH",
        help="Path to a SubtreeBPE .pkl file. BPE compound tokens in beam output "
             "are expanded to PN primitives before sanitization. Omit for a model "
             "trained on the bare grammar (no BPE).",
    )
    parser.add_argument(
        "--failed-only",
        nargs="?",
        const="",
        default=None,
        metavar="RESULTS_PKL_GZ",
        help="Only evaluate datasets where a previous run scored R^2 <= 0.99. "
             "Provide a path to a .pkl.gz results file, or omit the path to "
             "auto-detect ./results/eval_tf_<model>.pkl.gz.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Re-run every (dataset, seed) pair in this run from scratch instead "
             "of skipping pairs already present in ./results/eval_tf_<model>.pkl.gz. "
             "Rows for pairs OUTSIDE this run are preserved (so combining with "
             "--dataset re-runs only that dataset and keeps the rest of the file).",
    )
    parser.add_argument(
        "--no-bfgs",
        action="store_true",
        default=False,
        help="Disable BFGS constant-refinement and the final polish; rank the "
             "model's emitted formulas as-is. Intended for mantissa-free-grammar "
             "models whose constants are explicit subexpressions (no CONST "
             "placeholders for BFGS to fit).",
    )
    parser.add_argument(
        "--unscale",
        action="store_true",
        default=False,
        help="Apply input unscaling: substitute vi -> (vi-mu)/sigma to express the "
             "predicted formula in raw-variable space. Off by default (formula "
             "kept in whitened space); tags the output as eval_tf_<model>_unscale.pkl.gz.",
    )
    parser.add_argument(
        "--gpu-bfgs",
        action="store_true",
        default=False,
        help="Refine constants with batched GPU Levenberg-Marquardt + Variable "
             "Projection (gpu_bfgs.batch_refine) instead of SciPy BFGS. ~2.5x "
             "(fp64) / ~3.9x (fp32) faster end-to-end at equal-or-better solve rate.",
    )
    parser.add_argument(
        "--gpu-bfgs-restarts", type=int, default=8, metavar="R",
        help="Random restarts for --gpu-bfgs (batched in parallel). Default: 8.",
    )
    parser.add_argument(
        "--gpu-bfgs-iters", type=int, default=40, metavar="N",
        help="Levenberg-Marquardt iterations for --gpu-bfgs. Default: 40.",
    )
    parser.add_argument(
        "--gpu-bfgs-dtype", default="float64", choices=["float64", "float32"],
        help="Precision for --gpu-bfgs. float32 is faster; float64 matches SciPy "
             "accuracy (recommended on datacenter GPUs). Default: float64.",
    )
    parser.add_argument(
        "--ood-data",
        default=None,
        metavar="PATH",
        help="Path to a pre-generated OOD dataset file (.pkl.gz) produced by "
             "gen_ood_data.py.  When provided, each dataset's best formula is also "
             "scored on the OOD test set and OOD R^2 is reported alongside in-distribution R^2.",
    )
    parser.add_argument(
        "--offset-repair", action="store_true", default=False,
        help="After BFGS, try inserting the grammar's ++/-- (x+1 / x-1) unary at each "
             "site of the winning formula and keep a variant that beats it on a "
             "validation split of TRAIN.  Targets the one failure BFGS cannot fix: an "
             "additive offset inside a nonlinearity (exp(u)-1, 1/(g-1)).  Fires only "
             "when validation R^2 is in [0.90, 1), so cost is bounded.",
    )
    args = parser.parse_args()
    check_bench_flags(args, parser, _BENCH_ONLY_FLAGS)
    _apply_bench_defaults(args)

    if args.benchmark == "llmsrbench":
        return run_llmsrbench(args, _models)
    return run_srbench(args, _models)


# ---------------------------------------------------------------------------
# SRBench (Feynman PMLB)
# ---------------------------------------------------------------------------
def run_srbench(args, _models):
    """Evaluate on the Feynman PMLB datasets.

    Writes <args.results_dir>/noise_<TAU>/eval_tf_<model>.pkl.gz, one row per
    (dataset, seed), resuming on that pair.
    """
    # Resolve to a single checkpoint path from whichever source was given.
    if args.model is not None:
        checkpoint_path = resolve_model_checkpoint(args.model, args.checkpoints_dir)
    elif args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        if not _models:
            raise FileNotFoundError(
                f"No models found in {args.checkpoints_dir} and no --checkpoint provided."
            )
        checkpoint_path = resolve_model_checkpoint(_models[0], args.checkpoints_dir)
        print(f"No --model or --checkpoint specified; defaulting to '{_models[0]}'.", flush=True)

    model_name = args.model or _model_name_from_checkpoint(checkpoint_path)
    if args.tpsr:
        model_name = model_name + "_tpsr"
    if args.unscale:
        model_name = model_name + "_unscale"

    # Every run is written to a per-noise subdirectory (tau=0 -> noise_0/) rather than
    # tagged into model_name, so the file label stays "89M_40_simp1"/"145M_40_simp1"/... and every
    # downstream label/colour remap in the plotting scripts works unchanged -- the
    # noise level is conveyed by the directory + plot filename.  See results_io.
    results_dir = noise_dir(args.results_dir, args.target_noise or 0.0)

    device = torch.device(args.device)

    bpe_model = None
    if args.bpe_model is not None:
        if not os.path.exists(args.bpe_model):
            raise FileNotFoundError(f"BPE model not found: {args.bpe_model}")
        bpe_model = SubtreeBPE.load(args.bpe_model)
        print(f"Loaded BPE model: {args.bpe_model} ({len(bpe_model.merges)} merges)", flush=True)

    if args.tpsr_horizon is None:
        args.tpsr_horizon = 20 if bpe_model is not None else args._tpsr_horizon_primitive
        print(f"TPSR horizon auto-set to {args.tpsr_horizon} "
              f"({'BPE' if bpe_model is not None else 'primitive-token'} model).", flush=True)

    print(f"Loading model from {checkpoint_path} ...", flush=True)
    model, vocab = load_model(checkpoint_path, device)
    print("Model loaded.\n", flush=True)

    if args.no_bfgs:
        print("[no-bfgs] constant refinement disabled; ranking emitted formulas as-is.", flush=True)
        if args.gpu_bfgs:
            print("[no-bfgs] --gpu-bfgs ignored (no constants to refine).", flush=True)
    elif args.gpu_bfgs:
        enable_gpu_bfgs(device, restarts=args.gpu_bfgs_restarts,
                        iters=args.gpu_bfgs_iters, dtype=args.gpu_bfgs_dtype)
        print(f"[gpu-bfgs] constant refinement -> GPU VarPro LM "
              f"(restarts={args.gpu_bfgs_restarts}, iters={args.gpu_bfgs_iters}, "
              f"{args.gpu_bfgs_dtype})", flush=True)

    # Load ground-truth Feynman formulas (best-effort; missing entries are "").
    feynman_targets = load_feynman_targets()
    feynman_pn_targets = load_feynman_pn_targets()
    if feynman_targets:
        print(f"Loaded {len(feynman_targets)} Feynman target formulas "
              f"({len(feynman_pn_targets)} converted to canonical PN).", flush=True)

    # Load pre-generated OOD datasets (produced by gen_ood_data.py), if provided.
    ood_dataset: dict = {}
    if args.ood_data is not None:
        with gzip.open(args.ood_data, "rb") as _fh:
            _ood_file = pickle.load(_fh)
        ood_dataset = _ood_file["datasets"]
        print(f"Loaded OOD data for {len(ood_dataset)} datasets from {args.ood_data} "
              f"(gap={_ood_file.get('gap', '?')}, "
              f"n_points={_ood_file.get('n_points', '?')}).", flush=True)

    # collect dataset directories
    if args.dataset:
        dataset_dirs = [os.path.join(args.datasets, args.dataset)]
    else:
        dataset_dirs = sorted([
            os.path.join(args.datasets, d)
            for d in os.listdir(args.datasets)
            if os.path.isdir(os.path.join(args.datasets, d))
        ])

    # Filter to previously-failed datasets if --failed-only is set.
    current_round = 1
    if args.failed_only is not None:
        prev_path = args.failed_only or os.path.join(results_dir, f"eval_tf_{model_name}.pkl.gz")
        with gzip.open(prev_path, "rb") as fh:
            prev = pickle.load(fh)
        # Backfill missing "round" fields and rewrite the file if any were absent.
        if any("round" not in r for r in prev["results"]):
            for r in prev["results"]:
                r.setdefault("round", 1)
            with gzip.open(prev_path, "wb") as fh:
                pickle.dump(prev, fh)
            print(f"Backfilled round=1 into {prev_path}", flush=True)
        prev_r2s: dict[str, list[float]] = {}
        for row in prev["results"]:
            prev_r2s.setdefault(row["dataset"], []).append(row["r2"])
        failed_ds = {ds for ds, vals in prev_r2s.items() if np.mean(vals) <= 0.99}
        dataset_dirs = [d for d in dataset_dirs if os.path.basename(d) in failed_ds]
        current_round = max((r.get("round", 1) for r in prev["results"]), default=1) + 1
        print(f"--failed-only: {len(dataset_dirs)} datasets with R^2 <= 0.99 "
              f"from {prev_path} (round {current_round})", flush=True)

    out_path = os.path.join(results_dir, f"eval_tf_{model_name}.pkl.gz")

    # Resume: load any partial results written by a previous interrupted run.
    # Skip (dataset, seed) pairs that are already present.  With --no-resume,
    # re-run this run's pairs from scratch but preserve all other rows.
    all_results: list[dict] = []
    _done_pairs: set = set()
    if args.failed_only is None and os.path.isfile(out_path):
        try:
            with gzip.open(out_path, "rb") as _fh:
                _prev = pickle.load(_fh)
            _prior = list(_prev.get("results", []))
            if args.no_resume:
                # Drop rows for pairs this run will recompute; keep the rest so the
                # incremental save never clobbers untouched datasets.  _done_pairs
                # stays empty so nothing is skipped.
                _run_pairs = {(os.path.basename(d), s)
                              for d in dataset_dirs
                              for s in range(args.seed, args.seed + args.n_seeds)}
                all_results = [r for r in _prior
                               if (r["dataset"], r["seed"]) not in _run_pairs]
                _dropped = len(_prior) - len(all_results)
                print(f"--no-resume: re-running from scratch "
                      f"(dropped {_dropped} stale row(s), preserved {len(all_results)}) "
                      f"in {out_path}", flush=True)
            else:
                all_results = _prior
                _done_pairs = {(r["dataset"], r["seed"]) for r in all_results}
                if _done_pairs:
                    print(f"Resuming: {len(_done_pairs)} (dataset, seed) pairs already done "
                          f"from {out_path}", flush=True)
        except Exception:
            pass   # corrupt/missing -- start fresh

    # Per-dataset R^2 accumulated across seeds, then averaged.
    dataset_r2s: dict[str, list[float]] = {os.path.basename(d): [] for d in dataset_dirs}
    for r in all_results:
        if r["dataset"] in dataset_r2s:
            dataset_r2s[r["dataset"]].append(r["r2"])
    total_runs   = len(all_results)

    # Batched multi-seed path: all seeds for one dataset in one GPU call.
    # Active when: n_seeds > 1, stochastic sampling, no TPSR/no-split.
    _use_batched = (
        args.n_seeds > 1
        and args.sample_n > 0
        and not args.tpsr
        and not args.no_split
    )

    def _save_incremental():
        os.makedirs(results_dir, exist_ok=True)
        with gzip.open(out_path, "wb") as _fh:
            pickle.dump({"model_name": model_name, "checkpoint": checkpoint_path,
                         "results": all_results}, _fh)

    if _use_batched:
        # -- Chunked multi-dataset loop: DATASET_CHUNK datasets * n_seeds per GPU call --
        all_seeds = list(range(args.seed, args.seed + args.n_seeds))
        print(f"\n[batched multi-dataset] {len(all_seeds)} seeds * {len(dataset_dirs)} datasets "
              f"-- {DATASET_CHUNK} datasets per GPU call, BFGS parallel across all pairs.", flush=True)

        # Pre-scan: separate already-done from pending.
        pending: list = []   # (global_idx, dataset_dir, seeds_needed)
        for i, dataset_dir in enumerate(dataset_dirs):
            name = os.path.basename(dataset_dir)
            seeds_needed = [s for s in all_seeds if (name, s) not in _done_pairs]
            if seeds_needed:
                pending.append((i, dataset_dir, seeds_needed))
            else:
                print(f"[{i+1}/{len(dataset_dirs)}] {name} ... all {len(all_seeds)} seeds done",
                      flush=True)

        # Process in chunks of DATASET_CHUNK datasets.
        for chunk_start in range(0, len(pending), DATASET_CHUNK):
            chunk = pending[chunk_start:chunk_start + DATASET_CHUNK]

            # Load data for every dataset in this chunk.
            chunk_items: list = []   # (X_full, y_full, seeds_needed, ood_data)
            chunk_meta:  list = []   # (global_idx, name, seeds_needed)
            for gi, dataset_dir, seeds_needed in chunk:
                name = os.path.basename(dataset_dir)
                tsv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".tsv.gz")]
                if not tsv_files:
                    continue
                X_full, y_full = load_dataset(os.path.join(dataset_dir, tsv_files[0]))
                _ood_entry = ood_dataset.get(name)
                _ood_data  = (_ood_entry["X"], _ood_entry["y"]) if _ood_entry is not None else None
                chunk_items.append((X_full, y_full, seeds_needed, _ood_data))
                chunk_meta.append((gi, name, seeds_needed))
                print(f"  [{gi+1}/{len(dataset_dirs)}] {name} ({len(seeds_needed)} seeds)",
                      flush=True)

            if not chunk_items:
                continue

            t0 = time.perf_counter()
            results_list = _multidataset_batched(
                chunk_items, model, vocab, device,
                n_points_override=args.n_points,
                sample_size_override=args.sample_size,
                n_bags=args.n_bags,
                use_sampling=True,
                n_samples=args.sample_n,
                temperature=args.temperature,
                top_p=args.top_p,
                use_bfgs=not args.no_bfgs,
                beam_size=args.beam_size,
                bpe_model=bpe_model,
                use_unscaling=args.unscale,
                target_noise=args.target_noise,
            )
            chunk_elapsed = time.perf_counter() - t0
            total_seeds_in_chunk = sum(len(s) for _, _, s in chunk_meta)
            per_seed_time = chunk_elapsed / max(total_seeds_in_chunk, 1)

            # Save results for each dataset in the chunk.
            for seed_results, (gi, name, seeds_needed) in zip(results_list, chunk_meta):
                for seed, result in sorted(seed_results.items()):
                    if len(result) == 3:
                        formula, r2, ood_r2 = result
                    else:
                        formula, r2 = result
                        ood_r2 = None
                    dataset_r2s[name].append(r2)
                    total_runs += 1
                    _ood_str = f"  OOD={ood_r2:.4f}" if ood_r2 is not None else ""
                    print(f"  [{name}] seed={seed}: R^2={r2:.4f}{_ood_str}", flush=True)
                    all_results.append({
                        "dataset": name, "seed": seed, "round": current_round,
                        "r2": r2, "ood_r2": ood_r2,
                        "time": per_seed_time,
                        "predicted_formula": formula,
                        "target_formula": feynman_targets.get(name, ""),
                        "target_pn": feynman_pn_targets.get(name, ""),
                    })
                    _done_pairs.add((name, seed))

            chunk_num = chunk_start // DATASET_CHUNK + 1
            n_chunks  = (len(pending) + DATASET_CHUNK - 1) // DATASET_CHUNK
            print(f"  -> chunk {chunk_num}/{n_chunks}: {len(chunk_items)} datasets, "
                  f"{total_seeds_in_chunk} seeds in {chunk_elapsed:.1f}s "
                  f"({per_seed_time:.1f}s/seed)", flush=True)
            _save_incremental()

    else:
        # -- Original seed-outer loop (single seed / TPSR) --
        for seed in range(args.seed, args.seed + args.n_seeds):
            rng = np.random.default_rng(seed)
            print(f"\n--- Seed {seed} ---", flush=True)

            for i, dataset_dir in enumerate(dataset_dirs):
                name = os.path.basename(dataset_dir)
                if (name, seed) in _done_pairs:
                    print(f"[{i+1}/{len(dataset_dirs)}] {name} ... skipped (already done)", flush=True)
                    continue
                print(f"[{i+1}/{len(dataset_dirs)}] {name} ... ", end="", flush=True)

                _ood_entry = ood_dataset.get(name)
                _ood_data  = (_ood_entry["X"], _ood_entry["y"]) if _ood_entry is not None else None

                t0 = time.perf_counter()
                _eval_result = evaluate_dataset(
                    dataset_dir, model, vocab, device, rng,
                    n_bags=args.n_bags,
                    use_tpsr=args.tpsr,
                    tpsr_sims=args.tpsr_sims,
                    tpsr_c_puct=args.tpsr_c_puct,
                    tpsr_top_k=args.tpsr_top_k,
                    tpsr_n_bags=args.tpsr_n_bags,
                    tpsr_prior_temperature=args.tpsr_prior_temperature,
                    tpsr_horizon=args.tpsr_horizon,
                    tpsr_rollout_beam_width=args.tpsr_rollout_beam_width,
                    tpsr_no_search_bfgs=not args.tpsr_search_bfgs,
                    tpsr_warm_start=not args.tpsr_no_warm_start,
                    tpsr_warm_start_beam_width=args.tpsr_warm_start_beam_width,
                    use_sampling=args.sample_n > 0,
                    n_samples=args.sample_n,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    no_split=args.no_split,
                    use_bfgs=not args.no_bfgs,
                    beam_size=args.beam_size,
                    bpe_model=bpe_model,
                    use_unscaling=args.unscale,
                    gt_pn=feynman_pn_targets.get(name, ""),
                    ood_data=_ood_data,
                    use_offset_repair=args.offset_repair,
                    n_points_override=args.n_points,
                    sample_size_override=args.sample_size,
                    target_noise=args.target_noise,
                    max_forward_batch=args.max_forward_batch,
                )
                if _ood_data is not None:
                    formula, r2, ood_r2 = _eval_result
                else:
                    formula, r2 = _eval_result
                    ood_r2 = None
                elapsed = time.perf_counter() - t0
                dataset_r2s[name].append(r2)
                total_runs += 1
                _ood_str = f"  OOD R^2={ood_r2:.4f}" if ood_r2 is not None else ""
                print(f"R^2={r2:.4f}{_ood_str}  t={elapsed:.1f}s", flush=True)

                all_results.append({
                    "dataset":           name,
                    "seed":              seed,
                    "round":             current_round,
                    "r2":                r2,
                    "ood_r2":            ood_r2,
                    "time":              elapsed,
                    "predicted_formula": formula,
                    "target_formula":    feynman_targets.get(name, ""),
                    "target_pn":         feynman_pn_targets.get(name, ""),
                })
                _done_pairs.add((name, seed))
                _save_incremental()

    # Summary: average R^2 across seeds per dataset.
    mean_r2s = [np.mean(v) for v in dataset_r2s.values()]
    n = len(mean_r2s)
    print(f"\n{'='*60}")
    print(f"Datasets evaluated : {n}  (averaged over {args.n_seeds} seeds)")
    if n:
        print(f"Mean R^2            : {np.mean(mean_r2s):.4f}")
        print(f"Median R^2          : {np.median(mean_r2s):.4f}")
        print(f"R^2 = 1.0           : {sum(1 for v in mean_r2s if v >= 1.0 - 1e-6):3d} / {n}")
        print(f"R^2 > 0.99          : {sum(1 for v in mean_r2s if v > 0.99):3d} / {n}")
        print(f"R^2 > 0.90          : {sum(1 for v in mean_r2s if v > 0.90):3d} / {n}")
    ood_vals = [r["ood_r2"] for r in all_results if r.get("ood_r2") is not None]
    if ood_vals:
        print(f"OOD Mean R^2        : {np.mean(ood_vals):.4f}")
        print(f"OOD R^2 = 1.0       : {sum(1 for v in ood_vals if v >= 1.0 - 1e-6):3d} / {len(ood_vals)}")
        print(f"OOD R^2 > 0.99      : {sum(1 for v in ood_vals if v > 0.99):3d} / {len(ood_vals)}")
        print(f"OOD R^2 > 0.90      : {sum(1 for v in ood_vals if v > 0.90):3d} / {len(ood_vals)}")
    print("=" * 60)

    # When --failed-only is active, merge new results into the previous file:
    # keep passing rows from before, replace failed rows with the new ones.
    if args.failed_only is not None:
        re_evaluated = {(r["dataset"], r["seed"]) for r in all_results}
        retained = [r for r in prev["results"] if (r["dataset"], r["seed"]) not in re_evaluated]
        all_results = retained + all_results

        # Combined summary across all rounds.
        # Use max R^2 per dataset: a dataset counts as solved if it was ever solved,
        # regardless of earlier failed attempts pulling the average down.
        combined_means: dict[str, float] = {}
        dataset_round: dict[str, int] = {}
        for r in all_results:
            ds = r["dataset"]
            combined_means[ds] = max(combined_means.get(ds, 0.0), r["r2"])
            dataset_round[ds] = r.get("round", 1)
        n_all = len(combined_means)
        all_vals = list(combined_means.values())
        max_round = max(dataset_round.values(), default=1)

        print(f"\n{'='*60}")
        print(f"Combined ({n_all} datasets total)")
        print(f"Mean R^2   : {np.mean(all_vals):.4f}")
        print(f"Median R^2 : {np.median(all_vals):.4f}")
        print(f"R^2 = 1.0  : {sum(1 for v in all_vals if v >= 1.0 - 1e-6):3d} / {n_all}")
        print(f"R^2 > 0.99 : {sum(1 for v in all_vals if v > 0.99):3d} / {n_all}")
        print(f"R^2 > 0.90 : {sum(1 for v in all_vals if v > 0.90):3d} / {n_all}")
        for rnd in range(1, max_round + 1):
            in_rnd = [ds for ds, r in dataset_round.items() if r == rnd]
            passed = sum(1 for ds in in_rnd if combined_means[ds] > 0.99)
            print(f"  Round {rnd}: {len(in_rnd):3d} evaluated -> {passed:3d} passed R^2 > 0.99")
        print("=" * 60)

    # Final save (out_path already defined and incrementally written above).
    os.makedirs(results_dir, exist_ok=True)
    with gzip.open(out_path, "wb") as f:
        pickle.dump({
            "model_name": model_name,
            "checkpoint": checkpoint_path,
            "results":    all_results,
        }, f)
    print(f"Results saved to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# LLM-SRBench
# ---------------------------------------------------------------------------
def run_llmsrbench(args, _models):
    """Evaluate on an LLM-SRBench split.

    Writes <args.results_dir>/noise_<TAU>/llmsrbench/<model>_<split>[_tag]/
    results.pkl.gz, one row per (equation, seed), resuming on that pair.
    """
    device = torch.device(args.device)

    # Resolve + load model (same precedence as the srbench path).
    if args.model is not None:
        checkpoint_path = resolve_model_checkpoint(args.model, args.checkpoints_dir)
    elif args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        if not _models:
            raise FileNotFoundError(
                f"No models found in {args.checkpoints_dir} and no --checkpoint provided."
            )
        checkpoint_path = resolve_model_checkpoint(_models[0], args.checkpoints_dir)
        print(f"No --model or --checkpoint specified; defaulting to '{_models[0]}'.", flush=True)
    model_name = args.model or _model_name_from_checkpoint(checkpoint_path)

    bpe_model = None
    if args.bpe_model is not None:
        bpe_model = SubtreeBPE.load(args.bpe_model)
        print(f"Loaded BPE model: {args.bpe_model} ({len(bpe_model.merges)} merges)", flush=True)

    if args.tpsr and args.tpsr_horizon is None:
        args.tpsr_horizon = 20 if bpe_model is not None else args._tpsr_horizon_primitive
        print(f"TPSR horizon auto-set to {args.tpsr_horizon} "
              f"({'BPE' if bpe_model is not None else 'primitive-token'} model).", flush=True)

    print(f"Loading model from {checkpoint_path} ...", flush=True)
    model, vocab = load_model(checkpoint_path, device)
    print("Model loaded.\n", flush=True)
    print(f"Search: {'TPSR' if args.tpsr else 'beam'}", flush=True)

    # Load benchmark.
    print(f"Loading LLM-SRBench split '{args.split}' ...", flush=True)
    problems = load_problems(args.split)
    if args.problem is not None:
        problems = [q for q in problems if q["name"] == args.problem]
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    print(f"  {len(problems)} problems.\n", flush=True)

    # Config-aware output dir so with/without unscaling and with/without TPSR runs
    # of the same model do not overwrite each other.  With-unscaling keeps the
    # original (untagged) name for backward compatibility.
    _tag = (("_unscale" if args.unscale else "")
            + ("_tpsr" if args.tpsr else "")
            )
    # Every run goes in a per-noise subdir (tau=0 -> noise_0/) so the model dir name --
    # which the compare/plot scripts key their labels off -- is identical at every
    # noise level.  See results_io for the layout.
    _method = f"{model_name}_{args.split}{_tag}"
    results_path = llmsr_path(args.results_dir, args.target_noise or 0.0, _method)

    seeds = list(range(args.seed, args.seed + args.n_seeds))

    # Resume: keyed on (equation_id, seed) so a multi-seed run resumes mid-seed.
    # A missing/None equation counts as NOT done, so it is retried, not kept.
    done_pairs, existing_rows = set(), []
    for _r in load_results(results_path)[1]:
        _key = (_r.get("equation_id"), _r.get("seed"))
        _eq = _r.get("discovered_equation")
        if _key in done_pairs or _eq in (None, "None", ""):
            continue
        done_pairs.add(_key)
        existing_rows.append(_r)
    if done_pairs:
        print(f"Resuming: {len(done_pairs)} (problem, seed) pairs already done "
              f"in {results_path}", flush=True)

    rows = list(existing_rows)
    # Rewrite the whole artifact after each problem. save_results is atomic
    # (temp + os.replace), so an interrupted run always leaves a loadable file --
    # the pkl.gz equivalent of the old append-and-flush.
    def _save():
        save_results(results_path, rows, model_name=model_name, split=args.split,
                     method=_method, target_noise=args.target_noise or 0.0)

    _save()
    # Seeds loop OUTSIDE the model load: one process scores every (problem, seed) for
    # this model, so a formula-level array task pays 1 checkpoint load, not n_seeds.
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for i, q in enumerate(problems):
            if (q["name"], seed) in done_pairs:
                continue
            train, test = q["train"], q["test"]
            n_vars = train.shape[1] - 1
            if n_vars > _NV:
                print(f"[skip] {q['name']}: {n_vars} vars > model capacity {_NV}", flush=True)
                continue

            X_train, y_train = train[:, 1:], train[:, 0]
            X_test,  y_test  = test[:, 1:],  test[:, 0]

            t0 = time.perf_counter()
            # Searcher sees TRAIN only -> no_split fits + selects on train.
            formula, _train_r2 = evaluate_xy(
                X_train, y_train, model, vocab, device, rng,
                n_bags=args.n_bags, beam_size=args.beam_size,
                no_split=True, bpe_model=bpe_model,
                sample_size_override=args.sample_size,
                use_unscaling=args.unscale,
                use_tpsr=args.tpsr, tpsr_sims=args.tpsr_sims,
                tpsr_c_puct=args.tpsr_c_puct, tpsr_top_k=args.tpsr_top_k,
                tpsr_n_bags=args.tpsr_n_bags, tpsr_horizon=args.tpsr_horizon,
                tpsr_rollout_beam_width=args.tpsr_rollout_beam_width,
                tpsr_prior_temperature=args.tpsr_prior_temperature,
                target_noise=args.target_noise,
                max_forward_batch=args.max_forward_batch,
            )
            elapsed = time.perf_counter() - t0

            id_m = output_metrics(predict(formula, X_test), y_test)
            ood_m = None
            if q["ood_test"] is not None:
                ood_m = output_metrics(
                    predict(formula, q["ood_test"][:, 1:]), q["ood_test"][:, 0])

            log = {
                "equation_id": q["name"],
                "gt_equation": q["expression"],
                "discovered_equation": formula,
                "n_vars": n_vars,
                "num_datapoints": int(len(train)),
                "num_eval_datapoints": int(len(test)),
                "search_time": elapsed,
                "seed": seed,
                "id_metrics": id_m,
                "ood_metrics": ood_m,
            }
            rows.append(log)
            _save()
            print(f"[seed {seed}] [{i+1}/{len(problems)}] {q['name']:<20} "
                  f"R^2={id_m['r2']:.4f}  NMSE={id_m['nmse']:.3e}  t={elapsed:.1f}s  "
                  f"{formula}", flush=True)

    # -- Summary --------------------------------------------------------------
    r2s = np.array([r["id_metrics"]["r2"] for r in rows], dtype=np.float64)
    r2s_valid = r2s[np.isfinite(r2s)]
    print(f"\n{'='*70}")
    print(f"LLM-SRBench {args.split} -- {model_name}  ({len(rows)} problems)")
    print(f"{'='*70}")
    if r2s_valid.size:
        for thr in (0.99, 0.999):
            acc = float(np.mean(r2s_valid >= thr)) * 100
            print(f"  Acc (R^2 >= {thr})  : {acc:.1f}%")
        print(f"  Mean R^2          : {np.mean(r2s_valid):.4f}")
        print(f"  Median R^2        : {np.median(r2s_valid):.4f}")
        print(f"  Valid / total    : {r2s_valid.size}/{len(rows)}")
    print(f"{'='*70}")
    print(f"[results] {results_path}\n")


if __name__ == "__main__":
    main()
