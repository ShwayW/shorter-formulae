# imports
import torch
import torch.nn as nn
from .multihead_attn_block import MultiheadAttentionBlock

# class definition for multihead attention super
class InducedSetAttention(nn.Module):
    def __init__(self, dim_in_super, dim_out_super, num_heads_super, num_induced_inds_super, use_bias = True):
        super(InducedSetAttention, self).__init__()

        # stats for super
        self.dim_in_super = dim_in_super
        self.dim_out_super = dim_out_super
        self.num_heads_super = num_heads_super
        self.num_induced_inds_super = num_induced_inds_super

        # stats for sample
        self.dim_in_sample = None
        self.dim_out_sample = None
        self.num_heads_sample = None
        self.num_induced_inds_sample = None

        # initialize the internal multihead attention blocks (MAB)
        self.sampleI = None
        self.I = nn.Parameter(torch.Tensor(1, num_induced_inds_super, dim_out_super))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MultiheadAttentionBlock(dim_out_super, dim_in_super, dim_out_super, num_heads_super, use_bias)
        self.mab1 = MultiheadAttentionBlock(dim_in_super, dim_out_super, dim_out_super, num_heads_super, use_bias)
        

    def set_sample_config(self, dim_in_sample, dim_out_sample, num_heads_sample, num_induced_inds_sample):
        # stats
        self.dim_in_sample = dim_in_sample
        self.dim_out_sample = dim_out_sample
        self.num_heads_sample = num_heads_sample
        self.num_induced_inds_sample = num_induced_inds_sample

        # invoke the set_sample_config function of the internal MAB
        self.sampleI = self.I[0, :num_induced_inds_sample, :dim_out_sample]
        self.mab0.set_sample_config(dim_out_sample, dim_in_sample, dim_out_sample, num_heads_sample)
        self.mab1.set_sample_config(dim_in_sample, dim_out_sample, dim_out_sample, num_heads_sample)


    # multi-head attention function
    def forward(self, X):
        H = self.mab0(self.sampleI.repeat(X.shape[0], 1, 1), X)
        return self.mab1(X, H)


    def get_dims_str(self, indent = 0):
        # prints out dimensions of all components, for debug
        indent_str = "\t" * indent
        dims_str = f"{indent_str}ISAB I: {list(self.I.shape)}\n"
        dims_str += f"{indent_str}ISAB MAB_0:\n{self.mab0.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}ISAB MAB_1:\n{self.mab1.get_dims_str(indent + 1)}"
        return dims_str



