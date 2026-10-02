## Vanilla Transformer supernet
# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core.linear import Linear
from .core.encoderlayer import EncoderLayer
from .core.decoderlayer import DecoderLayer
from .core.embedding import Embedding
from .core.positional import PositionalEncoding
from .core.layernorm import LayerNorm
from .lpe import LinearPointEmbedder

class VanillaTransformer(nn.Module):
    def __init__(self, config, patch_dim, vocab_size, max_seq_length, pad_idx, dropout = 0.00):
        super(VanillaTransformer, self).__init__()
        # stats
        self.pad_idx = pad_idx
        self.patch_dim = patch_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        ## The largest model possible
        # fnns (skipped when use_lpe=True -- LPE replaces the FNN stack)
        self.use_lpe = config.get('use_lpe', False)
        if not self.use_lpe:
            self.super_num_emb_layers = 1 + config['num_emb_layers']
            self.super_emb_dims = [self.patch_dim] + config['emb_dims']

        # encoder part hyperparameters
        self.activation = config.get('activation', 'gelu')

        # LayerNorm placement for every encoder/decoder layer: 'pre' (default) or
        # 'post'.  This is an ablation knob -- see study_norm_position_init_grad.py.
        self.norm_position = config.get('norm_position', 'pre')
        if self.norm_position not in ('pre', 'post'):
            raise ValueError(f"norm_position must be 'pre' or 'post', got {self.norm_position!r}")

        self.super_num_enc_layers = config['num_enc_layers'] # the L_enc in the paper
        self.super_num_enc_heads = config['num_enc_heads'] # number of head in the encoder's mhatn layer, the H in the paper
        self.super_in_dim_enc_embed = config['in_dim_enc_embed'] # d_e in the paper
        self.super_qk_dim_enc_embed = config['qk_dim_enc_embed'] # d_qk in the paper
        self.super_v_dim_enc_embed = config['v_dim_enc_embed'] # d_v in the paper
        self.super_dim_enc_mlp = config['dim_enc_mlp'] # d_mlp in the paper

        # decoder part hyperparameters
        self.super_num_dec_layers = config['num_dec_layers'] # the L_dec in the paper
        self.super_num_dec_heads = config['num_dec_heads'] # number of head in the decoder's mhatn layer, the H in the paper
        self.super_in_dim_dec_embed = config['in_dim_dec_embed'] # d_e in the paper
        self.super_qk_dim_dec_embed = config['qk_dim_dec_embed'] # d_qk in the paper
        self.super_v_dim_dec_embed = config['v_dim_dec_embed'] # d_v in the paper
        self.super_dim_dec_mlp = config['dim_dec_mlp'] # d_mlp in the paper

        ## The current sampled model dimensions
        # fnns
        self.sample_num_emb_layers = None
        self.sample_emb_dims = None

        # encoder part hyperparameters
        self.sample_num_enc_layers = None
        self.sample_num_enc_heads = None
        self.sample_in_dim_enc_embed = None
        self.sample_qk_dim_enc_embed = None
        self.sample_v_dim_enc_embed = None
        self.sample_dim_enc_mlp = None

        # decoder part hyperparameters
        self.sample_num_dec_layers = None
        self.sample_num_dec_heads = None
        self.sample_in_dim_dec_embed = None
        self.sample_qk_dim_dec_embed = None
        self.sample_v_dim_dec_embed = None
        self.sample_dim_dec_mlp = None

        ## initialize the various ViT components
        # encoder embeddings -- either LinearPointEmbedder or the raw-float FNN stack
        if self.use_lpe:
            # n_input_dim: how many of the patch_dim columns are input variables.
            # Defaults to patch_dim - 1 (one output variable), matching our grammar
            # where patch_dim = len(V) + 1.
            n_input_dim = config.get('lpe_n_input_dim', self.patch_dim - 1)
            self.lpe = LinearPointEmbedder(
                patch_dim=self.patch_dim,
                n_input_dim=n_input_dim,
                d_emb=config['lpe_d_emb'],
                d_enc=self.super_in_dim_enc_embed,
                mantissa_len=config.get('lpe_mantissa_len', 1),
                n_mlp_layers=config.get('lpe_n_mlp_layers', 2),
                expansion_factor=config.get('lpe_expansion_factor', 1.0),
            )
            self.emb_layers = nn.ModuleList()
            # Projects LPE's fixed d_enc output to the (sampled) encoder input
            # dim, mirroring enc_mid_layer in the FNN path so supernet slicing
            # works for both embedder types.
            self.enc_mid_layer = Linear(self.super_in_dim_enc_embed,
                                        self.super_in_dim_enc_embed)
        else:
            self.lpe = None
            self.emb_layers = nn.ModuleList([Linear(self.super_emb_dims[layerI],
                                                         self.super_emb_dims[layerI + 1]) for layerI in range(self.super_num_emb_layers - 1)])
            self.enc_mid_layer = Linear(self.super_emb_dims[-1], self.super_in_dim_enc_embed)

        # encder layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.super_in_dim_enc_embed,
                                                               self.super_qk_dim_enc_embed[layerI],
                                                               self.super_v_dim_enc_embed[layerI],
                                                               self.super_num_enc_heads[layerI],
                                                               self.super_dim_enc_mlp[layerI],
                                                               self.dropout,
                                                               activation=self.activation,
                                                               norm_position=self.norm_position) for layerI in range(self.super_num_enc_layers)])
        # The trailing norm belongs to the pre-LN formulation only: under post-LN each
        # sublayer already ends in a LayerNorm, so an extra one here would be an
        # un-paired normalisation and would not be the post-LN baseline.
        self.encoder_norm = LayerNorm(self.super_in_dim_enc_embed) if self.norm_position == 'pre' else None

        # decoding embeddings
        #dec_pos_enc_mode = config.get('dec_pos_enc', 'learnable')  # 'learnable' or 'sinusoidal'
        dec_pos_enc_mode = config.get('dec_pos_enc', 'sinusoidal')  # 'learnable' or 'sinusoidal'
        self.decoder_embedding = Embedding(self.vocab_size, self.super_in_dim_dec_embed, padding_idx = self.pad_idx)
        self.dec_positional_encoding = PositionalEncoding(self.super_in_dim_dec_embed, self.max_seq_length, mode=dec_pos_enc_mode)

        # decoder layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.super_in_dim_dec_embed,
                                                               self.super_in_dim_enc_embed,
                                                               self.super_qk_dim_dec_embed[layerI],
                                                               self.super_v_dim_dec_embed[layerI],
                                                               self.super_num_dec_heads[layerI],
                                                               self.super_dim_dec_mlp[layerI],
                                                               self.dropout,
                                                               activation=self.activation,
                                                               norm_position=self.norm_position) for layerI in range(self.super_num_dec_layers)])
        self.decoder_norm = LayerNorm(self.super_in_dim_dec_embed) if self.norm_position == 'pre' else None

        # final layer
        self.fc = Linear(self.super_in_dim_dec_embed, self.vocab_size)


    def set_sample_config(self, config):
        ## FNNs (only meaningful for the FNN embedder path)
        if not self.use_lpe:
            self.sample_num_emb_layers = 1 + config['num_emb_layers']
            self.sample_emb_dims = [self.patch_dim] + config['emb_dims']

        ## Encoder configuration settings
        self.sample_num_enc_layers = config['num_enc_layers']
        self.sample_num_enc_heads = config['num_enc_heads']
        self.sample_in_dim_enc_embed = config['in_dim_enc_embed']
        self.sample_qk_dim_enc_embed = config['qk_dim_enc_embed']
        self.sample_v_dim_enc_embed = config['v_dim_enc_embed']
        self.sample_dim_enc_mlp = config['dim_enc_mlp']

        # set the embedder configurations (FNN stack only; LPE has no supernet slicing)
        if not self.use_lpe:
            for layI, layer in enumerate(self.emb_layers):
                if (layI < self.sample_num_emb_layers - 1):
                    layer.set_sample_config(sample_in_dim = self.sample_emb_dims[layI],
                                            sample_out_dim = self.sample_emb_dims[layI + 1],
                                            is_identity_layer = False)
                else:
                    layer.set_sample_config(is_identity_layer = True)

        # set the encoderlayers configurations
        for enclayI, layer in enumerate(self.encoder_layers):
            if (enclayI < self.sample_num_enc_layers):
                layer.set_sample_config(is_identity_layer = False,
                                        sample_in_embed_dim = self.sample_in_dim_enc_embed,
                                        sample_qk_embed_dim = self.sample_qk_dim_enc_embed[enclayI],
                                        sample_v_embed_dim = self.sample_v_dim_enc_embed[enclayI],
                                        sample_d_ff = self.sample_dim_enc_mlp[enclayI],
                                        sample_num_heads = self.sample_num_enc_heads[enclayI],
                                        sample_dropout = self.compute_dropout(self.sample_in_dim_enc_embed, self.super_in_dim_enc_embed))
            else:
                layer.set_sample_config(is_identity_layer = True)

        ## Decoder configuration settings
        self.sample_num_dec_layers = config['num_dec_layers']
        self.sample_num_dec_heads = config['num_dec_heads']
        self.sample_in_dim_dec_embed = config['in_dim_dec_embed']
        self.sample_qk_dim_dec_embed = config['qk_dim_dec_embed']
        self.sample_v_dim_dec_embed = config['v_dim_dec_embed']
        self.sample_dim_dec_mlp = config['dim_dec_mlp']

        # configure enc_mid_layer for both paths
        if self.use_lpe:
            # LPE output is always super_in_dim_enc_embed; project to sampled dim
            self.enc_mid_layer.set_sample_config(
                sample_in_dim=self.super_in_dim_enc_embed,
                sample_out_dim=self.sample_in_dim_enc_embed,
                is_identity_layer=False,
            )
        else:
            self.enc_mid_layer.set_sample_config(
                sample_in_dim=self.sample_emb_dims[self.sample_num_emb_layers - 1],
                sample_out_dim=self.sample_in_dim_enc_embed,
                is_identity_layer=False,
            )


        # final encoder/decoder norms (pre-LN only -- absent under post-LN)
        if self.encoder_norm is not None:
            self.encoder_norm.set_sample_config(self.sample_in_dim_enc_embed)
        if self.decoder_norm is not None:
            self.decoder_norm.set_sample_config(self.sample_in_dim_dec_embed)

        # embedding dropout rates (scaled for supernet subsampling)
        self.enc_sample_dropout = self.compute_dropout(self.sample_in_dim_enc_embed, self.super_in_dim_enc_embed)

        # set the decoder configurations
        self.decoder_embedding.set_sample_config(self.sample_in_dim_dec_embed)
        self.dec_positional_encoding.set_sample_config(self.sample_in_dim_dec_embed)
        self.dec_sample_dropout = self.compute_dropout(self.sample_in_dim_dec_embed, self.super_in_dim_dec_embed)

        # set the encoderlayers configurations
        for declayI, layer in enumerate(self.decoder_layers):
            if (declayI < self.sample_num_dec_layers):
                layer.set_sample_config(is_identity_layer = False,
                                        sample_in_dec_dim = self.sample_in_dim_dec_embed,
                                        sample_in_enc_dim = self.sample_in_dim_enc_embed,
                                        sample_qk_embed_dim = self.sample_qk_dim_dec_embed[declayI],
                                        sample_v_embed_dim = self.sample_v_dim_dec_embed[declayI],
                                        sample_d_ff = self.sample_dim_dec_mlp[declayI],
                                        sample_num_heads = self.sample_num_dec_heads[declayI],
                                        sample_dropout = self.compute_dropout(self.sample_in_dim_dec_embed, self.super_in_dim_dec_embed))
            else:
                layer.set_sample_config(is_identity_layer = True)
        
        # the final layer
        self.fc.set_sample_config(self.sample_in_dim_dec_embed, self.vocab_size)


    def compute_dropout(self, sample_embed_dim, super_embed_dim):
        return self.dropout * sample_embed_dim / super_embed_dim


    def encode(self, ios):
        """Encode a batch of IO pairs.

        Returns (enc_output, ios_padding_mask) so beam search can cache the
        encoder output and avoid re-running it at every decoding step.
        """
        ios = ios.view(ios.shape[0], -1, self.patch_dim)      # (B, N, patch_dim)
        # Mask only truly absent IO rows (beyond num_ios padding): those have NaN
        # in the output column (last column).  Valid rows may have NaN in unused
        # variable columns, but always have a finite output -- do not mask them.
        ios_padding_mask = ~torch.isfinite(ios[..., -1])  # (B, N) -- True = padded row

        if self.use_lpe:
            # LPE (linear point embedder) encodes each scalar as float tokens; NaN -> PAD embedding (zero)
            enc_output = self.lpe(ios)                      # (B, N, super_in_dim_enc_embed)
            enc_output = self.enc_mid_layer(enc_output)     # (B, N, sample_in_dim_enc_embed)
            enc_output = F.dropout(enc_output, p=self.enc_sample_dropout, training=self.training)
        else:
            valid_mask = (~ios_padding_mask).unsqueeze(-1).float()
            ios = torch.nan_to_num(ios, nan=0.0)
            ios = torch.clamp(ios, min=-1e3, max=1e3)
            ios = ios * valid_mask
            for layer in self.emb_layers:
                ios = torch.relu(layer(ios))
            enc_output = torch.relu(self.enc_mid_layer(ios))

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_key_padding_mask=ios_padding_mask)
        if self.encoder_norm is not None:
            enc_output = self.encoder_norm(enc_output)
        return enc_output, ios_padding_mask

    def decode(self, tgt, enc_output, ios_padding_mask):
        """Run one decoder pass given a cached encoder output."""
        fml_padding_mask = (tgt == self.pad_idx)
        tgt = self.decoder_embedding(tgt)
        tgt = self.dec_positional_encoding(tgt)
        tgt_embedded = F.dropout(tgt, p=self.dec_sample_dropout, training=self.training)
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(
                dec_output,
                enc_output,
                tgt_key_padding_mask=fml_padding_mask,
                memory_key_padding_mask=ios_padding_mask,
            )
        if self.decoder_norm is not None:
            dec_output = self.decoder_norm(dec_output)
        return self.fc(dec_output)

    # ------------------------------------------------------------------
    # Cached / incremental decoding helpers
    # ------------------------------------------------------------------

    def precompute_cross_kv(self, enc_output):
        """Run W_k / W_v on enc_output once per decoder layer.  Returns a list
        of (K, V) tuples (None, None for identity layers) that can be reused
        across all decode steps instead of being recomputed each time."""
        cross_kvs = []
        for dec_layer in self.decoder_layers:
            if dec_layer.is_identity_layer:
                cross_kvs.append((None, None))
            else:
                K, V = dec_layer.cross_attn.precompute_kv(enc_output)
                cross_kvs.append((K, V))
        return cross_kvs

    def init_self_kv_cache(self, B: int, device):
        """Return a list of empty (K, V) tensors -- one per decoder layer --
        that will be grown token-by-token during incremental decoding."""
        caches = []
        for i, dec_layer in enumerate(self.decoder_layers):
            if dec_layer.is_identity_layer:
                caches.append((None, None))
            else:
                d_k = self.sample_qk_dim_dec_embed[i]
                d_v = self.sample_v_dim_dec_embed[i]
                H   = self.sample_num_dec_heads[i]
                caches.append((
                    torch.zeros(B, H, 0, d_k, device=device),
                    torch.zeros(B, H, 0, d_v, device=device),
                ))
        return caches

    @staticmethod
    def reorder_kv_cache(kv_caches, flat_beam_idx):
        """Reindex KV caches to follow beam reordering.

        flat_beam_idx: 1-D LongTensor of length B*beam_size mapping new beam
                       positions to old beam positions (global indices).
        """
        return [
            (K[flat_beam_idx], V[flat_beam_idx]) if K is not None else (None, None)
            for K, V in kv_caches
        ]

    def decode_step(self, new_tok, step_idx: int, ios_padding_mask,
                    self_kv_caches, cross_kv_cache):
        """One incremental decoder step.

        new_tok:        (B, 1)  -- the single token at the current position
        step_idx:       int     -- 0-based position index (for positional encoding)
        self_kv_caches: list of (K, V) per layer  -- updated in-place (returned)
        cross_kv_cache: list of precomputed (K, V) per layer

        Returns: logits (B, vocab_size), updated self_kv_caches
        """
        x = self.decoder_embedding(new_tok)           # (B, 1, d_dec)
        # Add positional encoding for just this position
        pe = self.dec_positional_encoding.samples['pe'][step_idx]   # (d_dec,)
        x = x + pe.to(x.device)                       # broadcast -> (B, 1, d_dec)

        new_kv_caches = []
        for i, dec_layer in enumerate(self.decoder_layers):
            cross_K, cross_V = cross_kv_cache[i]
            self_K, self_V   = self_kv_caches[i]
            x, new_K, new_V  = dec_layer.forward_step(
                x, self_K, self_V, cross_K, cross_V,
                memory_padding_mask=ios_padding_mask,
            )
            new_kv_caches.append((new_K, new_V))

        if self.decoder_norm is not None:
            x = self.decoder_norm(x)     # (B, 1, d_dec)
        logits = self.fc(x[:, 0, :])     # (B, vocab_size)
        return logits, new_kv_caches


    def forward(self, ios, tgt):
        enc_output, ios_padding_mask = self.encode(ios)
        return self.decode(tgt, enc_output, ios_padding_mask)


    def get_dims_str(self, indent = 0):
        # prints out dimensions of all components, for debug
        # encoder embeddings
        indent_str = "\t" * indent
        dims_str = f"{indent_str}enc fnn: {self.enc_fc.get_dims_str()}\n"

        # encder layers
        for layerI, layer in enumerate(self.encoder_layers):
            layer_dims_str = layer.get_dims_str(indent)
            dims_str += f"{indent_str}enc layer {layerI + 1}:\n{layer_dims_str}\n"

        # decoding embeddings
        dims_str += f"{indent_str}dec tok embed: {self.decoder_embedding.get_dims_str()}\n"
        dims_str += f"{indent_str}dec pos embed: {self.dec_positional_encoding.get_dims_str()}\n"

        # decoder layers
        for layerI, layer in enumerate(self.decoder_layers):
            dims_str += f"{indent_str}dec layer {layerI + 1}:\n{layer.get_dims_str(indent)}\n"

        # final layer
        dims_str += f"{indent_str}final fnn: {self.fc.get_dims_str(indent)}"
        return dims_str



