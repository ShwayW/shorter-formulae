# imports
import torch
import torch.nn as nn
from .multihead_attn_block import MultiheadAttentionBlock

# class definition for multihead attention super
class PoolingMultiheadAttention(nn.Module):
    def __init__(self, seed_dim_super, num_heads_super, num_seeds_super, use_bias = True):
        super(PoolingMultiheadAttention, self).__init__()

        # stats for super
        self.seed_dim_super = seed_dim_super
        self.num_heads_super = num_heads_super
        self.num_seeds_super = num_seeds_super

        # stats for sample
        self.seed_dim_sample = None
        self.num_heads_sample = None
        self.num_seeds_sample = None

        # initialize the internal multihead attention blocks (MAB)
        self.sampleS = None
        self.S = nn.Parameter(torch.Tensor(1, num_seeds_super, seed_dim_super))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiheadAttentionBlock(seed_dim_super, seed_dim_super, seed_dim_super, num_heads_super, use_bias)
        

    def set_sample_config(self, seed_dim_sample, num_heads_sample, num_seeds_sample):
        # stats
        self.seed_dim_sample = seed_dim_sample
        self.num_heads_sample = num_heads_sample
        self.num_seeds_sample = num_seeds_sample

        # invoke the set_sample_config function of the internal MAB
        self.sampleS = self.S[0, :num_seeds_sample, :seed_dim_sample]
        self.mab.set_sample_config(seed_dim_sample, seed_dim_sample, seed_dim_sample, num_heads_sample)
        

    # multi-head attention function
    def forward(self, X):
        return self.mab(self.sampleS.repeat(X.size(0), 1, 1), X)


    def get_dims_str(self, indent = 0):
        # get the dimension describing string
        indent_str = "\t" * indent
        return f"{indent_str}PMA S: {list(self.S.shape)}\n{indent_str}PMA MAB:\n{self.mab.get_dims_str(indent + 1)}"



