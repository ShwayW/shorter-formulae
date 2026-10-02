# imports
import torch.nn as nn
from .multihead_attn_block import MultiheadAttentionBlock

# class definition for multihead attention super
class SetAttention(nn.Module):
    def __init__(self, dim_in_super, dim_out_super, num_heads_super, use_bias = True):
        super(SetAttention, self).__init__()

        # stats for super
        self.dim_in_super = dim_in_super
        self.dim_out_super = dim_out_super
        self.num_heads_super = num_heads_super

        # stats for sample
        self.dim_in_sample = None
        self.dim_out_sample = None
        self.num_heads_sample = None

        # initialize the internal multihead attention block (MAB)
        self.mab = MultiheadAttentionBlock(dim_in_super, dim_in_super, dim_out_super, num_heads_super, use_bias)
        

    def set_sample_config(self, dim_in_sample, dim_out_sample, num_heads_sample):
        # stats
        self.dim_in_sample = dim_in_sample
        self.dim_out_sample = dim_out_sample
        self.num_heads_sample = num_heads_sample

        # invoke the set_sample_config function of the internal MAB
        self.mab.set_sample_config(dim_in_sample, dim_in_sample, dim_out_sample, num_heads_sample)
        

    # multi-head attention function
    def forward(self, X):
        return self.mab(X, X)


    def get_dims_str(self, indent = 0):
        indent_str = "\t" * indent
        dims_str = f"{indent_str}{self.mab.get_dims_str(indent + 1)}"
        return dims_str


