## Decoder-only Transformer supernet
#
# A GPT-style decoder-only transformer for symbolic regression.  Unlike the
# encoder-decoder VanillaTransformer (tfs/van.py), there is no separate encoder
# and no cross-attention: the IO pairs and the formula live in ONE causal stream.
#
# Sequence layout (per example, before padding):
#
#     <bes>  in_1 out_1  in_2 out_2  ...  in_N out_N  <sep>  f_1 f_2 ... f_k  <bes>
#
#   * <bes>            : begin-of-sequence marker (reused from the data target).
#   * in_i / out_i     : the input-variable group and the output of IO pair i,
#                        each embedded into ONE vector by the float-token IO
#                        embedder (same float -> token scheme as van.py's LPE).
#   * <sep>            : separates the IO context from the formula.
#   * f_1..f_k <bes>   : the formula tokens followed by the closing <bes>.
#
# Positional encoding (this is the key design choice requested):
#
#     token : <bes>  in  out  in  out ...  <sep>   f_1  f_2 ...
#     pos   :   0     0   1    0   1  ...     1      2    3  ...
#
#   - every input group  -> position 0   (a "role" position, shared across pairs)
#   - every output       -> position 1   (a "role" position, shared across pairs)
#   - <bes> shares position 0 (it leads the context), <sep> shares position 1
#     (it closes the context), so the formula naturally begins at position 2 and
#     increments 2, 3, 4, ..., L for the rest of the stream.  Because the IO pairs
#     are an unordered set, they do not consume sequential positions; the
#     positional table therefore only needs to span the formula length.
#
# Targets (teacher forcing): every IO / context position is set to <pad> (ignored
# by the optimizer) and the formula is shifted right by one, exactly like the
# encoder-decoder transformer.  See build_targets().
#
# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core.linear import Linear
from .core.decoderonlylayer import DecoderOnlyLayer
from .core.embedding import Embedding
from .core.positional import PositionalEncoding
from .core.layernorm import LayerNorm
from .lpe import encode_floats_torch, vocab_params


# ---------------------------------------------------------------------------
# Float-token IO embedder
# ---------------------------------------------------------------------------
class IOEmbedder(nn.Module):
    """Embed IO pairs with the SAME float-token scheme as van.py's LinearPointEmbedder.

    Each scalar is converted to (2 + mantissa_len) token ids by
    lpe.encode_floats_torch (sign / mantissa / exponent), embedded through a
    shared table, flattened and compressed by an MLP -- identical in spirit to
    LinearPointEmbedder.

    The ONLY difference from LPE is the output granularity: LPE fuses all
    `patch_dim` columns of a pair into a single vector, whereas the decoder-only
    model needs the input-variable group and the output to occupy SEPARATE
    sequence positions (input -> position 0, output -> position 1).  This module
    therefore returns two vectors per IO pair, one for the inputs and one for the
    output, using two MLP heads over the shared float-token embedding table.

    NaN handling matches LPE: NaN in an input column -> learnable INPUT_PAD
    embedding; NaN in the output column -> learnable OUTPUT_PAD embedding.

    Parameters
    ----------
    patch_dim        : scalars per IO pair (= n_input_dim + 1; last column is the output)
    d_emb            : embedding dim per float token
    d_out            : output dim of each MLP head (= encoder/model embed dim)
    mantissa_len     : mantissa tokens per scalar (default 2 = grammar-compatible)
    n_mlp_layers     : total MLP depth (>= 1)
    expansion_factor : hidden_dim = flat_dim * expansion_factor
    """

    def __init__(self, patch_dim, d_emb, d_out, mantissa_len=2,
                 n_mlp_layers=1, expansion_factor=1.0):
        super().__init__()
        assert patch_dim >= 2, f"patch_dim={patch_dim} must be >= 2 (inputs + output)"
        _, _, _, _, inp_pad, out_pad, full_vocab = vocab_params(mantissa_len)

        self.patch_dim    = patch_dim
        self.n_input_dim  = patch_dim - 1          # last column is the output
        self.d_emb        = d_emb
        self.d_out        = d_out
        self.mantissa_len = mantissa_len
        self.tps          = 2 + mantissa_len       # tokens per scalar
        self._inp_pad     = inp_pad
        self._out_pad     = out_pad
        self._full_vocab  = full_vocab

        # Shared float-token embedding table (INPUT_PAD and OUTPUT_PAD learnable).
        self.embedding = nn.Embedding(full_vocab, d_emb)

        # Two MLP heads: one for the input-variable group, one for the output.
        in_flat  = self.n_input_dim * self.tps * d_emb
        out_flat = 1 * self.tps * d_emb
        self.input_mlp  = self._build_mlp(in_flat,  d_out, n_mlp_layers, expansion_factor)
        self.output_mlp = self._build_mlp(out_flat, d_out, n_mlp_layers, expansion_factor)

    @staticmethod
    def _build_mlp(flat_dim, d_out, n_mlp_layers, expansion_factor):
        hidden = max(1, int(flat_dim * expansion_factor))
        layers = [nn.Linear(flat_dim, hidden), nn.ReLU()]
        for _ in range(n_mlp_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, d_out))
        return nn.Sequential(*layers)

    def forward(self, ios):
        """ios: (B, N, patch_dim) float tensor.  NaN marks absent slots.

        Returns (input_emb, output_emb), each (B, N, d_out)."""
        B, N, D = ios.shape
        assert D == self.patch_dim, f"expected patch_dim={self.patch_dim}, got {D}"

        # float -> token ids (NaN/Inf -> INPUT_PAD for every column)
        tok_ids = encode_floats_torch(ios, self.mantissa_len)   # (B, N, D, tps)
        invalid = ~torch.isfinite(ios)                          # (B, N, D)

        in_tok  = tok_ids[:, :, :self.n_input_dim, :]           # (B, N, n_in, tps)
        out_tok = tok_ids[:, :, self.n_input_dim:, :]           # (B, N, 1, tps)

        # remap NaN in the output column INPUT_PAD -> OUTPUT_PAD (learnable, distinct)
        out_inv = invalid[:, :, self.n_input_dim:].unsqueeze(-1)  # (B, N, 1, 1)
        out_tok = torch.where(out_inv, out_tok.new_full((), self._out_pad), out_tok)

        in_tok  = in_tok.reshape(B, N, -1).clamp(0, self._full_vocab - 1)
        out_tok = out_tok.reshape(B, N, -1).clamp(0, self._full_vocab - 1)

        in_emb  = self.embedding(in_tok).reshape(B, N, -1)      # (B, N, n_in*tps*d_emb)
        out_emb = self.embedding(out_tok).reshape(B, N, -1)     # (B, N, tps*d_emb)
        return self.input_mlp(in_emb), self.output_mlp(out_emb)


# ---------------------------------------------------------------------------
# Decoder-only Transformer
# ---------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config, patch_dim, vocab_size, max_seq_length, pad_idx,
                 dropout=0.00, sep_idx=None):
        """
        config         : architecture config (see deco_config.py).
        patch_dim      : scalars per IO pair (len(V) + 1).
        vocab_size     : full vocabulary size (FlatGram + special symbols).
        max_seq_length : capacity of the positional table.  Must be >= the maximum
                         formula-token sequence length + 1 (the IO set does NOT
                         consume sequential positions, so N is irrelevant here).
        pad_idx        : index of <pad> (ignored by the loss; embeds to zero).
        sep_idx        : index of <sep> (separates IO context from the formula).
                         Falls back to config['sep_idx'] if not given.
        """
        super(DecoderOnlyTransformer, self).__init__()
        # stats
        self.pad_idx = pad_idx
        self.patch_dim = patch_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        sep_idx = sep_idx if sep_idx is not None else config.get('sep_idx', None)
        if sep_idx is None:
            raise ValueError("DecoderOnlyTransformer requires sep_idx (the <sep> token index).")
        self.sep_idx = sep_idx

        # Positional table only needs to span the formula length (+ the 2 role
        # positions for the IO set).  Size generously and assert at forward time.
        self.pos_table_size = max_seq_length + 2

        # decoder-stack hyperparameters (single stack -- no encoder)
        self.activation = config.get('activation', 'gelu')

        # LayerNorm placement: 'pre' (default) or 'post'.  Ablation knob -- see
        # study_norm_position_init_grad.py.
        self.norm_position = config.get('norm_position', 'pre')
        if self.norm_position not in ('pre', 'post'):
            raise ValueError(f"norm_position must be 'pre' or 'post', got {self.norm_position!r}")
        self.super_num_dec_layers   = config['num_dec_layers']
        self.super_num_dec_heads    = config['num_dec_heads']
        self.super_in_dim_dec_embed = config['in_dim_dec_embed']
        self.super_qk_dim_dec_embed = config['qk_dim_dec_embed']
        self.super_v_dim_dec_embed  = config['v_dim_dec_embed']
        self.super_dim_dec_mlp      = config['dim_dec_mlp']

        # currently-sampled dims
        self.sample_num_dec_layers   = None
        self.sample_num_dec_heads    = None
        self.sample_in_dim_dec_embed = None
        self.sample_qk_dim_dec_embed = None
        self.sample_v_dim_dec_embed  = None
        self.sample_dim_dec_mlp      = None

        # IO embedder (fixed dims; the io_mid_layer below projects to the sampled
        # dim, mirroring van.py's lpe + enc_mid_layer split so the supernet can
        # slice the model width without touching the embedder).
        self.io_embedder = IOEmbedder(
            patch_dim=patch_dim,
            d_emb=config.get('lpe_d_emb', 64),
            d_out=self.super_in_dim_dec_embed,
            mantissa_len=config.get('lpe_mantissa_len', 2),
            n_mlp_layers=config.get('lpe_n_mlp_layers', 1),
            expansion_factor=config.get('lpe_expansion_factor', 1.0),
        )
        self.io_mid_layer = Linear(self.super_in_dim_dec_embed, self.super_in_dim_dec_embed)

        # token embedding (shared by <bes>, <sep>, formula tokens) + positions
        self.token_embedding = Embedding(self.vocab_size, self.super_in_dim_dec_embed, padding_idx=self.pad_idx)
        self.positional_encoding = PositionalEncoding(
            self.super_in_dim_dec_embed, self.pos_table_size,
            mode=config.get('dec_pos_enc', 'sinusoidal'),
        )

        # the decoder-only (causal) layers
        self.decoder_layers = nn.ModuleList([
            DecoderOnlyLayer(self.super_in_dim_dec_embed,
                             self.super_qk_dim_dec_embed[layerI],
                             self.super_v_dim_dec_embed[layerI],
                             self.super_num_dec_heads[layerI],
                             self.super_dim_dec_mlp[layerI],
                             self.dropout,
                             activation=self.activation,
                             norm_position=self.norm_position)
            for layerI in range(self.super_num_dec_layers)
        ])
        # Pre-LN only -- under post-LN each sublayer already ends in a LayerNorm.
        self.decoder_norm = LayerNorm(self.super_in_dim_dec_embed) if self.norm_position == 'pre' else None

        # final unembedding layer
        self.fc = Linear(self.super_in_dim_dec_embed, self.vocab_size)

    # ------------------------------------------------------------------
    def set_sample_config(self, config):
        self.sample_num_dec_layers   = config['num_dec_layers']
        self.sample_num_dec_heads    = config['num_dec_heads']
        self.sample_in_dim_dec_embed = config['in_dim_dec_embed']
        self.sample_qk_dim_dec_embed = config['qk_dim_dec_embed']
        self.sample_v_dim_dec_embed  = config['v_dim_dec_embed']
        self.sample_dim_dec_mlp      = config['dim_dec_mlp']

        # IO embedder output is always super_in_dim_dec_embed; project to sampled dim
        self.io_mid_layer.set_sample_config(
            sample_in_dim=self.super_in_dim_dec_embed,
            sample_out_dim=self.sample_in_dim_dec_embed,
            is_identity_layer=False,
        )

        # token embedding + positional encoding follow the sampled width
        self.token_embedding.set_sample_config(self.sample_in_dim_dec_embed)
        self.positional_encoding.set_sample_config(self.sample_in_dim_dec_embed)
        self.sample_dropout = self.compute_dropout(self.sample_in_dim_dec_embed, self.super_in_dim_dec_embed)

        # decoder layers
        for declayI, layer in enumerate(self.decoder_layers):
            if (declayI < self.sample_num_dec_layers):
                layer.set_sample_config(is_identity_layer=False,
                                        sample_in_embed_dim=self.sample_in_dim_dec_embed,
                                        sample_qk_embed_dim=self.sample_qk_dim_dec_embed[declayI],
                                        sample_v_embed_dim=self.sample_v_dim_dec_embed[declayI],
                                        sample_d_ff=self.sample_dim_dec_mlp[declayI],
                                        sample_num_heads=self.sample_num_dec_heads[declayI],
                                        sample_dropout=self.compute_dropout(self.sample_in_dim_dec_embed, self.super_in_dim_dec_embed))
            else:
                layer.set_sample_config(is_identity_layer=True)

        if self.decoder_norm is not None:
            self.decoder_norm.set_sample_config(self.sample_in_dim_dec_embed)
        self.fc.set_sample_config(self.sample_in_dim_dec_embed, self.vocab_size)

    def compute_dropout(self, sample_embed_dim, super_embed_dim):
        return self.dropout * sample_embed_dim / super_embed_dim

    # ------------------------------------------------------------------
    # Sequence-construction helpers
    # ------------------------------------------------------------------
    def _num_io_pairs(self, ios):
        """Infer N (number of IO pairs) from the flat (B, N*patch_dim) tensor."""
        return ios.shape[1] // self.patch_dim if ios.dim() == 2 else ios.shape[1]

    def build_position_ids(self, num_pairs, fml_len, device):
        """Position ids for the full stream (length 1 + 2N + 1 + (L-1) = 2N+L+1).

        layout: [0] (<bes>) + [0,1]*N (IO) + [1] (<sep>) + [2,3,...,L] (formula).
        """
        N, L = num_pairs, fml_len
        io_pos  = torch.tensor([0, 1], device=device).repeat(N)            # (2N,)
        fml_pos = torch.arange(2, 2 + (L - 1), device=device)             # (L-1,) -> 2..L
        position_ids = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),  # <bes>
            io_pos,                                            # IO set
            torch.ones(1, dtype=torch.long, device=device),   # <sep>
            fml_pos,                                           # formula
        ])
        return position_ids

    def build_targets(self, ios, fml):
        """Teacher-forcing targets aligned to forward()'s output stream.

        Every <bes>/IO/<sep> context position -> <pad> (ignored); the formula is
        shifted right by one.  Shape: (B, 1 + 2N + L).
        """
        B = fml.shape[0]
        N = self._num_io_pairs(ios)
        device = fml.device
        pad_block = torch.full((B, 1 + 2 * N), self.pad_idx, dtype=torch.long, device=device)
        last_pad  = torch.full((B, 1), self.pad_idx, dtype=torch.long, device=device)
        # formula-region target = fml[:, 1:] (drop the leading <bes>) then one <pad>
        return torch.cat([pad_block, fml[:, 1:], last_pad], dim=1)        # (B, 2N+L+1)

    def build_padding_mask(self, ios_3d, fml):
        """Key-padding mask (B, S): True marks positions that must NOT be attended.

        Padded IO pairs (all-NaN rows, identified by a NaN output column, exactly
        like van.py) and <pad> formula tokens are masked; <bes>/<sep>/real tokens
        are not.
        """
        B, N, _ = ios_3d.shape
        device = ios_3d.device
        io_pair_pad = ~torch.isfinite(ios_3d[..., -1])                    # (B, N) True=absent pair
        io_tok_pad  = io_pair_pad.repeat_interleave(2, dim=1)             # (B, 2N) in & out share the pair flag
        fml_pad     = (fml[:, 1:] == self.pad_idx)                        # (B, L-1)
        false_col   = torch.zeros(B, 1, dtype=torch.bool, device=device)
        return torch.cat([false_col, io_tok_pad, false_col, fml_pad], dim=1)  # (B, 2N+L+1)

    # ------------------------------------------------------------------
    def forward(self, ios, fml):
        """
        ios : (B, N*patch_dim) flat IO tensor (same format as van.py).
        fml : (B, L) long tensor "<bes> f_1 ... f_k <bes> <pad>...", i.e. the
              data target.  fml[:, 0] is reused as the stream-leading <bes>; a
              <sep> is inserted before the formula body fml[:, 1:].

        Returns logits (B, 1 + 2N + L, vocab_size) over the full causal stream.
        """
        B = ios.shape[0]
        device = ios.device
        ios_3d = ios.view(B, -1, self.patch_dim)                         # (B, N, patch_dim)
        N = ios_3d.shape[1]
        L = fml.shape[1]

        # positional capacity check (max position used is L)
        assert L <= self.pos_table_size - 1, (
            f"formula length {L} exceeds positional capacity {self.pos_table_size}; "
            f"increase max_seq_length"
        )

        # --- IO embeddings: inputs and outputs as separate positions ---
        in_emb, out_emb = self.io_embedder(ios_3d)                       # (B,N,super), (B,N,super)
        io_emb = torch.stack([in_emb, out_emb], dim=2).reshape(B, 2 * N, -1)  # interleave: in,out,in,out,...
        io_emb = self.io_mid_layer(io_emb)                              # (B, 2N, sample)

        # --- token embeddings: <bes>, <sep>, formula body ---
        bes_emb = self.token_embedding(fml[:, 0:1])                      # (B, 1, sample)  (leading <bes>)
        sep_col = torch.full((B, 1), self.sep_idx, dtype=torch.long, device=device)
        sep_emb = self.token_embedding(sep_col)                         # (B, 1, sample)
        fml_emb = self.token_embedding(fml[:, 1:])                       # (B, L-1, sample)

        # --- assemble the stream and add positional encodings ---
        seq = torch.cat([bes_emb, io_emb, sep_emb, fml_emb], dim=1)      # (B, S, sample), S = 2N+L+1
        position_ids = self.build_position_ids(N, L, device)            # (S,)
        seq = seq + self.positional_encoding.samples['pe'][position_ids].to(device).unsqueeze(0)
        seq = F.dropout(seq, p=self.sample_dropout, training=self.training)

        # --- causal decoder stack ---
        key_padding_mask = self.build_padding_mask(ios_3d, fml)         # (B, S)
        for layer in self.decoder_layers:
            seq = layer(seq, key_padding_mask=key_padding_mask)
        if self.decoder_norm is not None:
            seq = self.decoder_norm(seq)

        return self.fc(seq)                                             # (B, S, vocab_size)

    def get_dims_str(self, indent=0):
        indent_str = "\t" * indent
        dims_str  = f"{indent_str}io embed -> mid: {self.io_mid_layer.get_dims_str()}\n"
        dims_str += f"{indent_str}tok embed: {self.token_embedding.get_dims_str()}\n"
        dims_str += f"{indent_str}pos embed: {self.positional_encoding.get_dims_str()}\n"
        for layerI, layer in enumerate(self.decoder_layers):
            dims_str += f"{indent_str}deco layer {layerI + 1}:\n{layer.get_dims_str(indent)}\n"
        dims_str += f"{indent_str}final fnn: {self.fc.get_dims_str(indent)}"
        return dims_str
