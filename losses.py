"""
losses.py - Custom loss functions for symbolic regression transformer.

Three components, combinable via SymbolicRegressionLoss:

  WeightedCELoss   -- token-level CE reweighted by semantic role
  EquivLoss        -- soft distance between predicted/target formula outputs on X
  ComplexityLoss   -- differentiable expected-length penalty

To reproduce the original plain cross-entropy loss, set:
  LOSS_W_CE          = 1.0
  LOSS_W_EQUIV       = 0.0
  LOSS_W_COMPLEXITY  = 0.0
  LOSS_UNIFORM_WEIGHTS = True
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from grammar import V, C, U, B, NUM_SIGN, NUM_MANTISSA, NUM_EXP, detokenize_floats
from funcWrappers import wrapExtEvalPN

# ---------------------------------------------------------------------------
# Top-level configuration constants
# Set LOSS_W_EQUIV=0, LOSS_W_COMPLEXITY=0, LOSS_UNIFORM_WEIGHTS=True
# to reproduce the original nn.CrossEntropyLoss behaviour.
# ---------------------------------------------------------------------------
LOSS_UNIFORM_WEIGHTS = True  # True -> all tokens weight 1.0  (plain CE)
LOSS_LABEL_SMOOTHING = 0.1
LOSS_W_CE            = 1.0
LOSS_W_EQUIV         = 0.0   # 0.0 to disable
LOSS_W_COMPLEXITY    = 0.0  # 0.0 to disable
LOSS_EQUIV_DISTANCE  = 'l2'  # 'l2' or 'cosine'

_N_VARS = len(V)   # 10
_IO_DIM = _N_VARS + 1  # 11 (10 vars + 1 target y)

# ---------------------------------------------------------------------------
# Token weight table
# ---------------------------------------------------------------------------

def build_token_weights(vocab: list, uniform: bool = False, dtype=torch.float32) -> torch.Tensor:
    """
    Per-token weight vector aligned to vocab.

    If uniform=True (or LOSS_UNIFORM_WEIGHTS=True), every token gets weight 1.0,
    reproducing plain cross-entropy.  Otherwise:
      Operators (U + B):      2.0  -- structural skeleton
      Variables (V):          2.0  -- structural skeleton
      Symbolic constants (C): 1.0  -- medium importance
      Float tokens:           0.3  -- fine-tunable post-hoc via BFGS
      Everything else:        1.0  -- <bes>, <pad>, CONST, ...
    """
    if uniform or LOSS_UNIFORM_WEIGHTS:
        return torch.ones(len(vocab), dtype=dtype)

    op_set    = set(U + B)
    var_set   = set(V)
    sym_c_set = set(C)
    float_set = set(NUM_SIGN + NUM_MANTISSA + NUM_EXP)

    w = []
    for tok in vocab:
        if tok in op_set or tok in var_set:
            w.append(2.0)
        elif tok in sym_c_set:
            w.append(1.0)
        elif tok in float_set:
            w.append(0.3)
        else:
            w.append(1.0)
    return torch.tensor(w, dtype=dtype)


# ---------------------------------------------------------------------------
# Component 1 -- Weighted Cross-Entropy
# ---------------------------------------------------------------------------

class WeightedCELoss(nn.Module):
    """
    Cross-entropy with per-token role weights and label smoothing.

    Args:
        vocab:           full vocabulary (FlatGram + special tokens)
        pad_index:       index of <pad> in vocab
        label_smoothing: applied uniformly across classes
    """

    def __init__(self, vocab: list, pad_index: int, label_smoothing: float = 0.1):
        super().__init__()
        self.pad_index = pad_index
        self.label_smoothing = label_smoothing
        self.register_buffer('weights', build_token_weights(vocab))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, T, V) -- raw model output
            targets: (B, T)    -- ground-truth token indices
        Returns:
            scalar loss
        """
        return F.cross_entropy(
            logits.contiguous().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            weight=self.weights,
            ignore_index=self.pad_index,
            label_smoothing=self.label_smoothing,
            reduction='mean',
        )


# ---------------------------------------------------------------------------
# Component 2 -- Equivalence Loss
# ---------------------------------------------------------------------------

class EquivLoss(nn.Module):
    """
    Evaluates the predicted formula on the IO set X and measures soft distance
    to the ground-truth output y drawn directly from ios_batch.

    L_equiv = D(eval(f_hat, X), eval(f*, X))

    where D is L2 MSE or cosine distance and eval(f*, X) = y from ios_batch.

    Note: formula evaluation is non-differentiable; this term acts as a
    reward-shaping signal whose magnitude informs the combined loss.
    Gradients flow back only as far as the .detach()-free parts of total loss.
    """

    def __init__(
        self,
        vocab: list,
        pad_index: int,
        distance: str = 'l2',
    ):
        super().__init__()
        self.vocab = vocab
        self.pad_index = pad_index
        self.distance = distance
        self._eos_set = {'<pad>', '<bes>'}

    @torch.no_grad()
    def _decode(self, token_ids: torch.Tensor) -> str:
        """1-D token-index tensor -> Polish-notation formula string."""
        toks = []
        for idx in token_ids.tolist():
            tok = self.vocab[idx]
            if tok in self._eos_set:
                break
            toks.append(tok)
        return ' '.join(detokenize_floats(toks))

    @staticmethod
    def _eval(fml_str: str, X: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate PN formula on X; return None on any error or NaN/Inf."""
        if not fml_str.strip():
            return None
        try:
            out, errs = wrapExtEvalPN(fml_str, X)
            if np.any(errs != 0) or not np.all(np.isfinite(out)):
                return None
            return out.astype(np.float32)
        except Exception:
            return None

    def forward(
        self,
        logits: torch.Tensor,     # (B, T, V)
        ios_batch: torch.Tensor,  # (B, max_ios * _IO_DIM)
    ) -> torch.Tensor:
        device = logits.device
        B = logits.size(0)
        pred_ids = logits.argmax(dim=-1)  # (B, T) -- detached from graph

        ios_np = ios_batch.cpu().float().numpy()
        ios_3d = ios_np.reshape(B, -1, _IO_DIM)  # (B, max_ios, 11)

        total = torch.tensor(0.0, device=device)
        n_valid = 0

        for i in range(B):
            rows = ios_3d[i]
            valid = np.isfinite(rows[:, -1])  # output column is NaN for padded rows
            rows = rows[valid]
            if len(rows) == 0:
                continue

            X = rows[:, :_N_VARS]   # (n_ios, 10)
            y = rows[:, _N_VARS]    # (n_ios,)

            fml_str = self._decode(pred_ids[i])
            y_hat = self._eval(fml_str, X)
            if y_hat is None:
                continue

            y_t  = torch.tensor(y,     dtype=torch.float32, device=device)
            yh_t = torch.tensor(y_hat, dtype=torch.float32, device=device)

            if self.distance == 'cosine':
                dist = 1.0 - F.cosine_similarity(yh_t.unsqueeze(0), y_t.unsqueeze(0))
            else:  # l2
                dist = F.mse_loss(yh_t, y_t)

            total = total + dist
            n_valid += 1

        return total / n_valid if n_valid > 0 else total


# ---------------------------------------------------------------------------
# Component 3 -- Complexity Loss
# ---------------------------------------------------------------------------

class ComplexityLoss(nn.Module):
    """
    Differentiable complexity penalty based on expected sequence length.

    For each position t, P(token != pad) = 1 - softmax(logits)[pad_index].
    L_complexity = mean_batch( sum_t P(not_pad_t) / T )

    This is fully differentiable and encourages the model to prefer shorter,
    more parsimonious expressions.
    """

    def __init__(self, pad_index: int):
        super().__init__()
        self.pad_index = pad_index

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (B, T, V) -> scalar"""
        probs    = torch.softmax(logits, dim=-1)        # (B, T, V)
        p_pad    = probs[:, :, self.pad_index]          # (B, T)
        p_active = 1.0 - p_pad                         # (B, T)
        T = logits.size(1)
        return p_active.sum(dim=1).mean() / T           # normalise by max seq len


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class SymbolicRegressionLoss(nn.Module):
    """
    L = w_ce * L_ce  +  w_equiv * L_equiv  +  w_complexity * L_complexity

    Args:
        vocab:            full vocabulary (FlatGram + special tokens)
        pad_index:        index of <pad> in vocab
        w_ce:             weight for CE component
        w_equiv:          weight for equiv component (0 to disable)
        w_complexity:     weight for complexity component (0 to disable)
        label_smoothing:  label smoothing for CE
        equiv_distance:   'l2' or 'cosine' for equiv loss
    """

    def __init__(
        self,
        vocab: list,
        pad_index: int,
        w_ce: float = LOSS_W_CE,
        w_equiv: float = LOSS_W_EQUIV,
        w_complexity: float = LOSS_W_COMPLEXITY,
        label_smoothing: float = LOSS_LABEL_SMOOTHING,
        equiv_distance: str = LOSS_EQUIV_DISTANCE,
    ):
        super().__init__()
        self.w_ce         = w_ce
        self.w_equiv      = w_equiv
        self.w_complexity = w_complexity

        self.ce_loss         = WeightedCELoss(vocab, pad_index, label_smoothing)
        self.equiv_loss      = EquivLoss(vocab, pad_index, distance=equiv_distance)
        self.complexity_loss = ComplexityLoss(pad_index)

    def forward(
        self,
        logits: torch.Tensor,                      # (B, T, V)
        targets: torch.Tensor,                     # (B, T)
        ios_batch: Optional[torch.Tensor] = None,  # (B, max_ios * 11)
    ) -> dict:
        """
        Returns:
            dict with keys 'loss', 'ce', 'equiv', 'complexity'.
            Call .backward() on result['loss'].
        """
        l_ce  = self.ce_loss(logits, targets)
        total = self.w_ce * l_ce

        l_equiv = torch.zeros(1, device=logits.device).squeeze()
        if self.w_equiv > 0 and ios_batch is not None:
            l_equiv = self.equiv_loss(logits, ios_batch)
            total   = total + self.w_equiv * l_equiv

        l_complexity = torch.zeros(1, device=logits.device).squeeze()
        if self.w_complexity > 0:
            l_complexity = self.complexity_loss(logits)
            total        = total + self.w_complexity * l_complexity

        return {
            'loss':       total,
            'ce':         l_ce.detach(),
            'equiv':      l_equiv.detach(),
            'complexity': l_complexity.detach(),
        }
