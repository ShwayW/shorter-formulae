# imports
import torch.nn as nn
from .induced_set_attention import InducedSetAttention

# Encoder layer
class SetEncoderLayer(nn.Module):
    def __init__(self, super_in_embed_dim, super_num_heads, num_induced_inds_super, dropout = 0., use_bias = True):
        super(SetEncoderLayer, self).__init__()

        # the largest possible model
        self.super_in_embed_dim = super_in_embed_dim
        self.super_num_heads = super_num_heads
        self.num_induced_inds_super = num_induced_inds_super
        self.super_dropout = dropout

        # current sampled dimension
        self.sample_in_embed_dim = None
        self.sample_num_heads = None
        self.sample_induced_inds_super = None
        self.sample_dropout = None

        # initialize components
        self.isab = InducedSetAttention(super_in_embed_dim, super_in_embed_dim, super_num_heads, num_induced_inds_super, use_bias)


    def set_sample_config(self, is_identity_layer, sample_in_embed_dim = None, sample_num_heads = None, num_induced_inds_sample = None, sample_dropout = None):
        if (is_identity_layer):
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads
        self.num_induced_inds_sample = num_induced_inds_sample
        self.sample_dropout = sample_dropout
        
        # set config for the components
        self.isab.set_sample_config(sample_in_embed_dim, sample_in_embed_dim, sample_num_heads, num_induced_inds_sample)
        

    def forward(self, x):
        if (self.is_identity_layer): return x
        return self.isab(x)


    def get_dims_str(self, indent = 0):
        # get the dimension describing string of all components
        if (self.is_identity_layer): return ""
        indent_str = "\t" * indent
        dims_str = f"{indent_str}{self.isab.get_dims_str(indent + 1)}"
        return dims_str



