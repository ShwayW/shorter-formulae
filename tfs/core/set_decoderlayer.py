# imports
import torch.nn as nn
from .linear import Linear
from .pooling_multihead_attn import PoolingMultiheadAttention
from .set_attention import SetAttention

# Encoder layer
class SetDecoderLayer(nn.Module):
    def __init__(self, super_in_embed_dim, super_num_outputs, super_dim_output, super_num_heads, num_induced_inds_super, dropout = 0., use_bias = True):
        super(SetDecoderLayer, self).__init__()

        # the largest possible model
        self.super_in_embed_dim = super_in_embed_dim
        self.super_num_outputs = super_num_outputs
        self.super_dim_output = super_dim_output
        self.super_num_heads = super_num_heads
        self.num_induced_inds_super = num_induced_inds_super
        self.super_dropout = dropout

        # current sampled dimension
        self.sample_in_embed_dim = None
        self.sample_num_outputs = None
        self.sample_dim_output = None
        self.sample_num_heads = None
        self.num_induced_inds_sample = None
        self.sample_dropout = None

        # initialize components
        self.pma = PoolingMultiheadAttention(super_in_embed_dim, super_num_heads, super_num_outputs, use_bias)
        self.sab = SetAttention(super_in_embed_dim, super_in_embed_dim, super_num_heads, use_bias)
        self.ffn = Linear(super_in_embed_dim, super_dim_output)

    def set_sample_config(self, is_identity_layer, sample_in_embed_dim, sample_num_outputs, sample_dim_output, sample_num_heads, num_induced_inds_sample, sample_dropout):
        if (is_identity_layer):
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_outputs = sample_num_outputs
        self.sample_dim_output = sample_dim_output
        self.sample_num_heads = sample_num_heads
        self.num_induced_inds_sample = num_induced_inds_sample
        self.sample_dropout = sample_dropout
        
        # set config for the components
        self.pma.set_sample_config(sample_in_embed_dim, sample_num_heads, sample_num_outputs)
        self.sab.set_sample_config(sample_in_embed_dim, sample_in_embed_dim, sample_num_heads)
        self.ffn.set_sample_config(sample_in_embed_dim, sample_dim_output)
        
    def forward(self, x):
        if (self.is_identity_layer): return x
        return self.ffn(self.sab(self.pma(x)))



