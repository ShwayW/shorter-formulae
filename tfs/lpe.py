"""
LinearPointEmbedder

Encodes a batch of IO pairs by representing each scalar value as
(2 + mantissa_len) float tokens [sign, M_0, ..., M_{k-1}, exp], embedding
them, and compressing the full per-IO-pair embedding with an MLP.

This replaces the raw-float FNN projection (emb_layers + enc_mid_layer) in
VanillaTransformer with a richer, float-aware encoding -- analogous to the
LinearPointEmbedder in Kamienny et al. 2022, adapted for our fixed-patch format.

Speed: the float encoding is fully vectorised in PyTorch and stays on the GPU
throughout.  No Python loops over IO pairs or variables, no CPU round-trips.

Float vocabulary layout for a given mantissa_len (float_precision fixed at 3):
  base       = (float_precision + 1) // mantissa_len   digits per mantissa token
  max_tok    = 10 ** base                               mantissa vocab size

  [0]                   NUM+   (positive / zero sign)
  [1]                   NUM-   (negative sign)
  [2 .. 2+max_tok-1]   M0 .. M(max_tok-1)  (shared mantissa vocab)
  [2+max_tok .. 2+max_tok+200]   E-100 .. E+100
  [2+max_tok+201]       <INPUT_PAD>   (absent input-variable slots -- learnable)
  [2+max_tok+202]       <OUTPUT_PAD>  (absent output-variable slots -- learnable)

mantissa_len=2 (default, grammar-compatible):
  base=2, max_tok=100 -> vocab layout matches grammar.py NUM_SIGN/NUM_MANTISSA/NUM_EXP
mantissa_len=1: base=4, max_tok=10000  (single 4-digit mantissa token per scalar)
mantissa_len=4: base=1, max_tok=10     (four 1-digit mantissa tokens per scalar)

Unlike a zero-fixed padding_idx, both pad embeddings are fully learnable so the
model can distinguish "this input variable slot is absent" from "this output
variable slot is absent".
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Invariant float vocabulary constants (independent of mantissa_len)
# ---------------------------------------------------------------------------
FLOAT_PRECISION  = 3        # total significant digits -- must match grammar.py
_MANTISSA_OFFSET = 2        # mantissa tokens start here
_EXP_MIN         = -100
_EXP_MAX         = 100

# Defaults for mantissa_len=2 (grammar-compatible).
FLOAT_VOCAB_SIZE = 303      # 2 + 100 + 201
INPUT_PAD_ID     = 303
OUTPUT_PAD_ID    = 304
FULL_VOCAB_SIZE  = 305
FLOAT_PAD_ID     = INPUT_PAD_ID   # backward-compat alias


# ---------------------------------------------------------------------------
# Vocabulary helpers (mantissa_len-aware)
# ---------------------------------------------------------------------------
def vocab_params(mantissa_len: int, float_precision: int = FLOAT_PRECISION):
    """Return derived vocabulary constants for the given mantissa_len.

    Parameters
    ----------
    mantissa_len   : number of mantissa tokens per scalar (1, 2, or 4 for
                     float_precision=3; must divide float_precision+1 evenly)
    float_precision: total significant digits (default 3, must match grammar.py)

    Returns
    -------
    base           : digits per mantissa token
    max_tok        : mantissa vocabulary size (= 10 ** base)
    exp_offset     : first exponent token ID (= _MANTISSA_OFFSET + max_tok)
    float_vocab    : total float token count (excl. pad tokens)
    inp_pad        : INPUT_PAD_ID for this mantissa_len
    out_pad        : OUTPUT_PAD_ID for this mantissa_len
    full_vocab     : FULL_VOCAB_SIZE for this mantissa_len
    """
    total_digits = float_precision + 1          # 4 for float_precision=3
    assert total_digits % mantissa_len == 0, (
        f"float_precision+1={total_digits} must be divisible by mantissa_len={mantissa_len}"
    )
    base        = total_digits // mantissa_len
    max_tok     = 10 ** base
    exp_offset  = _MANTISSA_OFFSET + max_tok
    float_vocab = exp_offset + (_EXP_MAX - _EXP_MIN + 1)   # +201
    inp_pad     = float_vocab
    out_pad     = float_vocab + 1
    full_vocab  = float_vocab + 2
    return base, max_tok, exp_offset, float_vocab, inp_pad, out_pad, full_vocab


def make_id_to_str(mantissa_len: int = 1,
                   float_precision: int = FLOAT_PRECISION) -> dict[int, str]:
    """Build the full ID->string map for the given mantissa_len.

    For mantissa_len=2 this is identical to the module-level FLOAT_ID_TO_STR.
    """
    base, max_tok, exp_offset, _, inp_pad, out_pad, _ = vocab_params(
        mantissa_len, float_precision
    )
    d: dict[int, str] = {0: "NUM+", 1: "NUM-"}
    for m in range(max_tok):
        d[_MANTISSA_OFFSET + m] = f"M{m:0{base}d}"
    for e in range(_EXP_MIN, _EXP_MAX + 1):
        d[exp_offset + (e - _EXP_MIN)] = f"E{e:+d}"
    d[inp_pad] = "<INPUT_PAD>"
    d[out_pad] = "<OUTPUT_PAD>"
    return d


# Module-level default (mantissa_len=2, grammar-compatible)
FLOAT_ID_TO_STR: dict[int, str] = make_id_to_str(mantissa_len=2)

# Cache of ID->string maps, keyed by (mantissa_len, float_precision).
_ID_TO_STR_CACHE: dict[tuple[int, int], dict[int, str]] = {
    (2, FLOAT_PRECISION): FLOAT_ID_TO_STR,
}


def ids_to_str(tok_ids, mantissa_len: int = 1,
               float_precision: int = FLOAT_PRECISION) -> list:
    """Convert token IDs to human-readable strings.

    Parameters
    ----------
    tok_ids      : int, nested list/tuple of ints, or a torch.Tensor.
                   Tensors are converted via .tolist() first.
    mantissa_len : must match the mantissa_len used when encoding (default 1).
    float_precision: must match the float_precision used when encoding (default 3).

    Returns
    -------
    Nested list of strings with the same shape as tok_ids.

    Examples
    --------
    >>> ids_to_str(encode_floats_torch(torch.tensor([3.14]))[0].tolist())
    ['NUM+', 'M3140', 'E+0']
    >>> ids_to_str(encode_floats_torch(torch.tensor([3.14]), mantissa_len=2)[0].tolist(), mantissa_len=2)
    ['NUM+', 'M31', 'M40', 'E+0']
    """
    key = (mantissa_len, float_precision)
    lut = _ID_TO_STR_CACHE.get(key)
    if lut is None:
        lut = _ID_TO_STR_CACHE[key] = make_id_to_str(mantissa_len, float_precision)
    if hasattr(tok_ids, "tolist"):
        tok_ids = tok_ids.tolist()

    def _recurse(x):
        if isinstance(x, int):
            return lut.get(x, f"<UNKNOWN:{x}>")
        return [_recurse(item) for item in x]

    return _recurse(tok_ids)

# ---------------------------------------------------------------------------
# Vectorised float encoder (GPU-resident)
# ---------------------------------------------------------------------------
def encode_floats_torch(values: torch.Tensor,
                        mantissa_len: int = 1,
                        float_precision: int = FLOAT_PRECISION) -> torch.Tensor:
    """Encode every scalar in *values* to (2 + mantissa_len) token IDs.

    Token layout per scalar: [sign, M_0, M_1, ..., M_{k-1}, exp]
    where k = mantissa_len and the M_i tokens share a vocabulary of size
    10^(float_precision+1 / mantissa_len).

    Parameters
    ----------
    values        : (...) float32 tensor on any device (including MPS)
    mantissa_len  : number of mantissa tokens per scalar (default 1)
    float_precision: significant digits (default 3, must match grammar.py)

    Returns
    -------
    (..., 2 + mantissa_len) int64 tensor on the same device.
    NaN and Inf entries -> all INPUT_PAD_ID for this mantissa_len.
    LinearPointEmbedder.forward remaps output-column invalids to OUTPUT_PAD_ID.

    NOTE: uses float32 throughout to stay GPU-resident on MPS, which does not
    support float64.  Float32 precision is sufficient for the 4-digit mantissa
    encoding.

    Matches grammar.encode_float() exactly (for mantissa_len=1):
      - sign: NUM+ (0) if x >= 0, NUM- (1) if x < 0
      - exponent: floor(log10(|x|)) clamped to [-100, 100]; 0 for x == 0
      - mantissa: round(|x| * 10^(float_precision-exp)) clamped to [0, 9999]
    """
    base, max_tok, exp_offset, _, inp_pad, _, _ = vocab_params(
        mantissa_len, float_precision
    )

    shape    = values.shape
    flat     = values.reshape(-1).float()          # float32, stays on device
    invalid_mask = ~torch.isfinite(flat)

    safe      = flat.masked_fill(invalid_mask, 0.0)
    abs_vals  = safe.abs()
    zero_mask = abs_vals == 0.0

    # ---- exponent ----
    safe_log = abs_vals.clamp(min=1e-38)
    exp      = torch.floor(torch.log10(safe_log)).long()
    exp      = exp.clamp(_EXP_MIN, _EXP_MAX)
    exp      = exp.masked_fill(zero_mask, 0)

    # Underflow guard: scale 10^(fp-exp) overflows float32 when fp-exp > 38.
    underflow_mask = (float_precision - exp) > 38
    exp      = exp.masked_fill(underflow_mask, 0)

    # ---- full mantissa integer M in [0, 10^(float_precision+1) - 1] ----
    scale      = torch.pow(10.0, (float_precision - exp).float())
    mantissa_f = abs_vals.masked_fill(underflow_mask, 0.0) * scale
    full_max   = float(10 ** (float_precision + 1) - 1 + 1000)  # safe clamp ceiling
    mantissa   = torch.round(mantissa_f.clamp(max=full_max)).long()

    # Overflow correction: floor(log10) can undershoot by 1 near power-of-10
    # boundaries, making mantissa round up to 10^(fp+1). Bump exp and recompute.
    m_max    = 10 ** (float_precision + 1)          # 10000 for fp=3
    over     = (mantissa >= m_max).long()
    exp_c    = (exp + over).clamp(_EXP_MIN, _EXP_MAX)
    scale_c  = torch.pow(10.0, (float_precision - exp_c).float())
    mantissa = torch.where(
        over.bool(),
        torch.round((abs_vals * scale_c).clamp(max=full_max)).long(),
        mantissa,
    ).clamp(0, m_max - 1)
    exp = exp_c

    # ---- split M into mantissa_len chunks, each in [0, max_tok) ----
    # divisors[i] = max_tok^(mantissa_len-1-i), so chunk[i] = (M // div[i]) % max_tok
    divisors = torch.tensor(
        [max_tok ** (mantissa_len - 1 - i) for i in range(mantissa_len)],
        dtype=torch.long, device=flat.device,
    )                                               # (mantissa_len,)
    m_exp    = mantissa.unsqueeze(-1)               # (N, 1)
    chunks   = (m_exp // divisors) % max_tok        # (N, mantissa_len)
    mantissa_ids = _MANTISSA_OFFSET + chunks        # (N, mantissa_len)

    # ---- sign and exponent ----
    sign_ids = (safe < 0).long()                    # (N,)
    exp_ids  = exp_offset + (exp - _EXP_MIN)        # (N,)

    # ---- overwrite invalids with INPUT_PAD_ID ----
    pad_val      = torch.full_like(sign_ids, inp_pad)
    sign_ids     = torch.where(invalid_mask, pad_val, sign_ids)
    exp_ids      = torch.where(invalid_mask, pad_val, exp_ids)
    mantissa_ids = torch.where(
        invalid_mask.unsqueeze(-1).expand_as(mantissa_ids),
        pad_val.unsqueeze(-1).expand_as(mantissa_ids),
        mantissa_ids,
    )

    # ---- assemble: [sign, M_0, ..., M_{k-1}, exp] ----
    out = torch.cat(
        [sign_ids.unsqueeze(-1), mantissa_ids, exp_ids.unsqueeze(-1)],
        dim=-1,
    )                                               # (N, 2+mantissa_len)
    return out.reshape(shape + (2 + mantissa_len,))


# ---------------------------------------------------------------------------
# LinearPointEmbedder
# ---------------------------------------------------------------------------
class LinearPointEmbedder(nn.Module):
    """Encodes IO pairs via float-token embeddings + MLP compression.

    Drop-in replacement for the (emb_layers + enc_mid_layer) block in
    VanillaTransformer.  Call forward() where encode() currently runs the
    FNN stack; the encoder-attention layers are unchanged.

    Architecture (per IO pair, length-N sequence processed in parallel):
      1. Each of the `patch_dim` scalars -> (2 + mantissa_len) token IDs
      2. NaN slots in the first `n_input_dim` columns -> INPUT_PAD_ID (learnable)
         NaN slots in the remaining output columns    -> OUTPUT_PAD_ID (learnable)
      3. nn.Embedding lookup: (patch_dim * tps) tokens -> (patch_dim * tps, d_emb)
         where tps = 2 + mantissa_len (tokens per scalar)
      4. Flatten to (patch_dim * tps * d_emb,)
      5. MLP: flat_dim -> [hidden_dim * n_hidden_layers] -> d_enc

    Parameters
    ----------
    patch_dim        : scalars per IO pair (n_input_dim + n_output_dim)
    n_input_dim      : how many of the patch_dim columns are input variables;
                       the remaining (patch_dim - n_input_dim) are outputs.
                       NaN slots in input columns -> INPUT_PAD_ID,
                       NaN slots in output columns -> OUTPUT_PAD_ID.
    d_emb            : embedding dim per float token
    d_enc            : output dim = encoder in_dim_enc_embed
    mantissa_len     : mantissa tokens per scalar (default 1 = grammar-compatible).
                       Must divide (float_precision + 1) evenly.
                       mantissa_len=1 -> 3 tokens/scalar, vocab 10203+2
                       mantissa_len=2 -> 4 tokens/scalar, vocab 103+2
                       mantissa_len=4 -> 6 tokens/scalar, vocab 13+2
    float_precision  : significant digits (default 3, must match grammar.py)
    n_mlp_layers     : total MLP depth (>=1); first layer is always flat_dim->hidden
    expansion_factor : hidden_dim = flat_dim * expansion_factor
    """

    def __init__(
        self,
        patch_dim: int,
        n_input_dim: int,
        d_emb: int,
        d_enc: int,
        mantissa_len: int = 1,
        float_precision: int = FLOAT_PRECISION,
        n_mlp_layers: int = 2,
        expansion_factor: float = 1.0,
    ):
        super().__init__()
        assert 0 < n_input_dim < patch_dim, (
            f"n_input_dim={n_input_dim} must be in (0, patch_dim={patch_dim})"
        )
        _, _, _, _, inp_pad, out_pad, full_vocab = vocab_params(
            mantissa_len, float_precision
        )

        self.patch_dim       = patch_dim
        self.n_input_dim     = n_input_dim
        self.d_emb           = d_emb
        self.d_enc           = d_enc
        self.mantissa_len    = mantissa_len
        self.float_precision = float_precision
        self.tps             = 2 + mantissa_len      # tokens per scalar
        self.tokens_per_patch = patch_dim * self.tps
        self._inp_pad        = inp_pad
        self._out_pad        = out_pad
        self._full_vocab     = full_vocab

        # Shared embedding table -- INPUT_PAD and OUTPUT_PAD are both fully
        # learnable so the model can distinguish absent input from absent output.
        self.embedding = nn.Embedding(full_vocab, d_emb)

        flat_dim   = self.tokens_per_patch * d_emb
        hidden_dim = max(1, int(flat_dim * expansion_factor))

        layers: list[nn.Module] = [nn.Linear(flat_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_mlp_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, d_enc))
        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, ios: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ios : (B, N, patch_dim) float tensor.
              NaN marks absent/padded slots.  The first n_input_dim columns are
              input variables; the rest are outputs.  NaN in an input column ->
              INPUT_PAD_ID embedding; NaN in an output column -> OUTPUT_PAD_ID
              embedding.  Both are learnable, so the model can distinguish the
              two cases.

        Returns
        -------
        (B, N, d_enc) float tensor -- ready to feed into the encoder layers.
        """
        B, N, D = ios.shape
        tps = self.tps # tokens per scalar
        assert D == self.patch_dim, (
            f"LinearPointEmbedder expects patch_dim={self.patch_dim}, got {D}"
        )

        # Step 1 -- float encoding: (B, N, D, tps) token IDs, all on device.
        # NaN/Inf positions come out as INPUT_PAD_ID; we fix up output columns below.
        tok_ids = encode_floats_torch(
            ios, self.mantissa_len, self.float_precision
        )                                           # (B, N, D, tps) int64
        tok_ids = tok_ids.reshape(B, N, D * tps)   # (B, N, D*tps)

        # Step 2 -- remap output-column invalids to OUTPUT_PAD_ID.
        invalid = ~torch.isfinite(ios)              # (B, N, D)
        invalid_tok = (
            invalid.unsqueeze(-1)
                   .expand(B, N, D, tps)
                   .reshape(B, N, D * tps)
        )
        is_output_col = torch.zeros(D * tps, dtype=torch.bool, device=ios.device)
        is_output_col[self.n_input_dim * tps:] = True

        tok_ids = torch.where(
            invalid_tok & is_output_col,
            tok_ids.new_full((), self._out_pad),
            tok_ids,
        )

        # Step 3 -- embed
        tok_ids  = tok_ids.clamp(0, self._full_vocab - 1)
        embedded = self.embedding(tok_ids)           # (B, N, D*tps, d_emb)

        # Step 4 -- flatten + MLP
        return self.mlp(embedded.reshape(B, N, -1))
