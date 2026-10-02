# imports
import torch.nn as nn
import torch.nn.functional as F
from .multihead_attn import MultiHeadAttention
from .linear import Linear
from .layernorm import LayerNorm

# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, super_in_dec_dim, super_in_enc_dim, super_qk_embed_dim, super_v_embed_dim, super_num_heads, super_d_ff, dropout, use_bias = True, activation = 'gelu', norm_position = 'pre'):
        super(DecoderLayer, self).__init__()

        # 'pre' -- x + sublayer(norm(x));  'post' -- norm(x + sublayer(x)).
        # See EncoderLayer for the rationale; the parameter set is identical either way.
        if norm_position not in ('pre', 'post'):
            raise ValueError(f"norm_position must be 'pre' or 'post', got {norm_position!r}")
        self.norm_position = norm_position

        # the largest possible model
        self.super_in_dec_dim = super_in_dec_dim
        self.super_in_enc_dim = super_in_enc_dim
        self.super_qk_embed_dim = super_qk_embed_dim
        self.super_v_embed_dim = super_v_embed_dim
        self.super_num_heads = super_num_heads
        self.super_d_ff = super_d_ff
        self.super_dropout = dropout

        # current sampled dimension
        self.sample_in_dec_dim = None
        self.sample_in_enc_dim = None
        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_d_ff = None
        self.sample_dropout = None

        # initialize the components
        self.self_attn = MultiHeadAttention(super_in_dec_dim, super_in_dec_dim, super_qk_embed_dim, super_v_embed_dim, super_num_heads, super_in_dec_dim, use_bias)
        self.cross_attn = MultiHeadAttention(super_in_dec_dim, super_in_enc_dim, super_qk_embed_dim, super_v_embed_dim, super_num_heads, super_in_dec_dim, use_bias)
        self.fc1 = Linear(super_in_dec_dim, super_d_ff)
        self.fc2 = Linear(super_d_ff, super_in_dec_dim)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.norm1 = LayerNorm(super_in_dec_dim)
        self.norm2 = LayerNorm(super_in_dec_dim)
        self.norm3 = LayerNorm(super_in_dec_dim)

    def set_sample_config(self, is_identity_layer, sample_in_dec_dim = None, sample_in_enc_dim = None, sample_qk_embed_dim = None, sample_v_embed_dim = None, sample_d_ff = None, sample_num_heads = None, sample_dropout = None):
        if (is_identity_layer):
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_in_dec_dim = sample_in_dec_dim
        self.sample_in_enc_dim = sample_in_enc_dim
        self.sample_qk_embed_dim = sample_qk_embed_dim
        self.sample_v_embed_dim = sample_v_embed_dim
        self.sample_num_heads = sample_num_heads
        self.sample_d_ff = sample_d_ff
        self.sample_dropout = sample_dropout
        
        # set config for the components
        self.self_attn.set_sample_config(sample_in_dec_dim, sample_in_dec_dim, sample_qk_embed_dim, sample_v_embed_dim, sample_num_heads, sample_in_dec_dim)
        self.cross_attn.set_sample_config(sample_in_dec_dim, sample_in_enc_dim, sample_qk_embed_dim, sample_v_embed_dim, sample_num_heads, sample_in_dec_dim)
        self.fc1.set_sample_config(sample_in_dec_dim, sample_d_ff)
        self.fc2.set_sample_config(sample_d_ff, sample_in_dec_dim)
        self.norm1.set_sample_config(sample_in_dec_dim)
        self.norm2.set_sample_config(sample_in_dec_dim)
        self.norm3.set_sample_config(sample_in_dec_dim)
        
    def forward(self, x, enc_output, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if (self.is_identity_layer): return x

        if (self.norm_position == 'pre'):
            # 1. Pre-LN autoregressive self-attention
            normed = self.norm1(x)
            attn_output = self.self_attn(
                X=normed,
                Z=normed,
                is_causal=True,
                padding_mask=tgt_key_padding_mask
            )
            x = x + F.dropout(attn_output, p=self.sample_dropout, training=self.training)

            # 2. Pre-LN cross-attention (normalise decoder stream; enc_output is already processed)
            attn_output = self.cross_attn(
                X=self.norm2(x),
                Z=enc_output,
                is_causal=False,
                padding_mask=memory_key_padding_mask
            )
            x = x + F.dropout(attn_output, p=self.sample_dropout, training=self.training)

            # 3. Pre-LN feed-forward
            ff_output = self.fc2(F.dropout(self.activation(self.fc1(self.norm3(x))), p=self.sample_dropout, training=self.training))
            x = x + F.dropout(ff_output, p=self.sample_dropout, training=self.training)
        else:
            # 1. Post-LN autoregressive self-attention
            attn_output = self.self_attn(
                X=x,
                Z=x,
                is_causal=True,
                padding_mask=tgt_key_padding_mask
            )
            x = self.norm1(x + F.dropout(attn_output, p=self.sample_dropout, training=self.training))

            # 2. Post-LN cross-attention
            attn_output = self.cross_attn(
                X=x,
                Z=enc_output,
                is_causal=False,
                padding_mask=memory_key_padding_mask
            )
            x = self.norm2(x + F.dropout(attn_output, p=self.sample_dropout, training=self.training))

            # 3. Post-LN feed-forward
            ff_output = self.fc2(F.dropout(self.activation(self.fc1(x)), p=self.sample_dropout, training=self.training))
            x = self.norm3(x + F.dropout(ff_output, p=self.sample_dropout, training=self.training))

        return x

    def forward_step(self, x_new, self_K_cache, self_V_cache, cross_K, cross_V, memory_padding_mask=None):
        """Incremental single-token decode step with KV cache.

        x_new:         (B, 1, d_dec)
        self_K_cache:  (B, H, T, d_k)  -- past self-attention keys
        self_V_cache:  (B, H, T, d_v)  -- past self-attention values
        cross_K/V:     precomputed encoder cross-attention keys/values
        Returns: x_new updated, new_self_K (B,H,T+1,d_k), new_self_V
        """
        if self.is_identity_layer:
            return x_new, self_K_cache, self_V_cache

        if (self.norm_position == 'pre'):
            # 1. Self-attention with KV cache
            attn_out, new_K, new_V = self.self_attn.forward_step(self.norm1(x_new), self_K_cache, self_V_cache)
            x_new = x_new + attn_out

            # 2. Cross-attention with precomputed K, V (no dropout during inference)
            attn_out = self.cross_attn.forward_step_cross(self.norm2(x_new), cross_K, cross_V, padding_mask=memory_padding_mask)
            x_new = x_new + attn_out

            # 3. Feed-forward
            x_new = x_new + self.fc2(self.activation(self.fc1(self.norm3(x_new))))
        else:
            # 1. Self-attention with KV cache
            attn_out, new_K, new_V = self.self_attn.forward_step(x_new, self_K_cache, self_V_cache)
            x_new = self.norm1(x_new + attn_out)

            # 2. Cross-attention with precomputed K, V (no dropout during inference)
            attn_out = self.cross_attn.forward_step_cross(x_new, cross_K, cross_V, padding_mask=memory_padding_mask)
            x_new = self.norm2(x_new + attn_out)

            # 3. Feed-forward
            x_new = self.norm3(x_new + self.fc2(self.activation(self.fc1(x_new))))

        return x_new, new_K, new_V

    def get_dims_str(self, indent = 0):
        # prints out dimensions of all components, for debug
        if (self.is_identity_layer): return ""
        indent_str = "\t" * indent
        dims_str = f"{indent_str}dec:\n"
        dims_str += f"{indent_str}{self.self_attn.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}{self.norm1.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}{self.cross_attn.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}{self.norm2.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}{self.fc1.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}{self.fc2.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}{self.norm3.get_dims_str(indent + 1)}\n"
        return dims_str



