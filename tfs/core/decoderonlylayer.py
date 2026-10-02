# imports
import torch.nn as nn
import torch.nn.functional as F
from .multihead_attn import MultiHeadAttention
from .linear import Linear
from .layernorm import LayerNorm

# Decoder-only (GPT-style) layer: pre-LN causal self-attention + feed-forward.
#
# This is the EncoderLayer block (self-attention + FFN, no cross-attention) but
# with the self-attention masked causally so a position can only attend to itself
# and earlier positions.  It is the building block of the decoder-only transformer
# in tfs/deco.py -- there is no encoder and therefore no cross-attention sublayer.
class DecoderOnlyLayer(nn.Module):
    def __init__(self, super_in_embed_dim, super_qk_embed_dim, super_v_embed_dim, super_num_heads, super_d_ff, dropout = 0., use_bias = True, activation = 'gelu', norm_position = 'pre'):
        super(DecoderOnlyLayer, self).__init__()

        # 'pre' -- x + sublayer(norm(x));  'post' -- norm(x + sublayer(x)).
        # See EncoderLayer for the rationale; the parameter set is identical either way.
        if norm_position not in ('pre', 'post'):
            raise ValueError(f"norm_position must be 'pre' or 'post', got {norm_position!r}")
        self.norm_position = norm_position

        # the largest possible model
        self.super_in_embed_dim = super_in_embed_dim
        self.super_qk_embed_dim = super_qk_embed_dim
        self.super_v_embed_dim = super_v_embed_dim
        self.super_num_heads = super_num_heads
        self.super_d_ff = super_d_ff
        self.super_dropout = dropout

        # current sampled dimension
        self.sample_in_embed_dim = None
        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_d_ff = None
        self.sample_dropout = None

        # initialize components (self-attention only -- no cross-attention)
        self.self_attn = MultiHeadAttention(super_in_embed_dim, super_in_embed_dim, super_qk_embed_dim, super_v_embed_dim, super_num_heads, super_in_embed_dim, use_bias)
        self.fc1 = Linear(super_in_embed_dim, super_d_ff)
        self.fc2 = Linear(super_d_ff, super_in_embed_dim)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.norm1 = LayerNorm(super_in_embed_dim)
        self.norm2 = LayerNorm(super_in_embed_dim)

    def set_sample_config(self, is_identity_layer, sample_in_embed_dim = None, sample_qk_embed_dim = None, sample_v_embed_dim = None, sample_d_ff = None, sample_num_heads = None, sample_dropout = None):
        if (is_identity_layer):
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_qk_embed_dim = sample_qk_embed_dim
        self.sample_v_embed_dim = sample_v_embed_dim
        self.sample_num_heads = sample_num_heads
        self.sample_d_ff = sample_d_ff
        self.sample_dropout = sample_dropout

        # set config for the components
        self.self_attn.set_sample_config(sample_in_embed_dim, sample_in_embed_dim, sample_qk_embed_dim, sample_v_embed_dim, sample_num_heads, sample_in_embed_dim)
        self.fc1.set_sample_config(sample_in_embed_dim, sample_d_ff)
        self.fc2.set_sample_config(sample_d_ff, sample_in_embed_dim)
        self.norm1.set_sample_config(sample_in_embed_dim)
        self.norm2.set_sample_config(sample_in_embed_dim)

    def forward(self, x, key_padding_mask = None):
        # if this layer is skipped, do nothing
        if (self.is_identity_layer): return x

        # is_causal=True restricts each position to attend to itself and earlier ones.
        if (self.norm_position == 'pre'):
            # Pre-LN causal self-attention: normalise before the sublayer, add to residual.
            normed = self.norm1(x)
            attn_output = self.self_attn(normed, normed, is_causal = True, padding_mask = key_padding_mask)
            x = x + F.dropout(attn_output, p = self.sample_dropout, training = self.training)

            # Pre-LN feed-forward
            ff_output = self.fc2(F.dropout(self.activation(self.fc1(self.norm2(x))), p = self.sample_dropout, training = self.training))
            x = x + F.dropout(ff_output, p = self.sample_dropout, training = self.training)
        else:
            # Post-LN causal self-attention: sublayer on the raw stream, normalise the sum
            attn_output = self.self_attn(x, x, is_causal = True, padding_mask = key_padding_mask)
            x = self.norm1(x + F.dropout(attn_output, p = self.sample_dropout, training = self.training))

            # Post-LN feed-forward
            ff_output = self.fc2(F.dropout(self.activation(self.fc1(x)), p = self.sample_dropout, training = self.training))
            x = self.norm2(x + F.dropout(ff_output, p = self.sample_dropout, training = self.training))

        return x

    def get_dims_str(self, indent = 0):
        # prints out dimensions of all components, for debug
        if (self.is_identity_layer): return ""
        indent_str = "\t" * indent
        dims_str = f"{indent_str}deco:\n"
        dims_str += f"{indent_str}{self.self_attn.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}{self.norm1.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}{self.fc1.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}{self.fc2.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}{self.norm2.get_dims_str(indent + 1)}\n"
        return dims_str
