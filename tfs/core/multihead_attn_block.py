# imports
import torch.nn as nn
from .linear import Linear
from .layernorm import LayerNorm
from .multihead_attn import MultiHeadAttention

# class definition for multihead attention super
class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dim_Q_super, dim_K_super, dim_V_super, num_heads_super, use_bias = True):
        super(MultiheadAttentionBlock, self).__init__()

        # stats
        self.dim_Q_super = dim_Q_super
        self.dim_K_super = dim_K_super
        self.dim_V_super = dim_V_super
        self.num_heads_super = num_heads_super

        # the sample dims
        self.dim_Q_sample = None
        self.dim_K_sample = None
        self.dim_V_sample = None
        self.num_heads_sample = None

        # initialize the internal multihead attention block (MAB)
        self.layer_norm0 = LayerNorm(dim_V_super)
        self.layer_norm1 = LayerNorm(dim_V_super)
        self.fc = Linear(dim_V_super, dim_V_super)
        self.attn = MultiHeadAttention(dim_Q_super, dim_K_super, dim_V_super, dim_V_super, num_heads_super, dim_Q_super, use_bias)
        

    def set_sample_config(self, dim_Q_sample, dim_K_sample, dim_V_sample, num_heads_sample):
        # stats
        self.dim_Q_sample = dim_Q_sample
        self.dim_K_sample = dim_K_sample
        self.dim_V_sample = dim_V_sample
        self.num_heads_sample = num_heads_sample

        # invoke the set_sample_config function of the internal MAB
        self.layer_norm0.set_sample_config(dim_V_sample)
        self.layer_norm1.set_sample_config(dim_V_sample)
        self.fc.set_sample_config(dim_V_sample, dim_V_sample)
        self.attn.set_sample_config(dim_Q_sample, dim_K_sample, dim_V_sample, dim_V_sample, num_heads_sample, dim_Q_sample)
        

    # multi-head attention function
    def forward(self, X, Y):
        H = self.layer_norm0(X + self.attn(X, Y))
        return self.layer_norm1(H + self.fc(H))


    def get_dims_str(self, indent = 0):
        # get the dimension describing string
        indent_str = "\t" * indent
        dims_str = f"{indent_str}MAB:\n"
        dims_str += f"{self.attn.get_dims_str(indent + 1)}\n"
        dims_str += f"{indent_str}\tlayer norm 0: {self.layer_norm0.get_dims_str()}\n"
        dims_str += f"{indent_str}\tfnn: {self.fc.get_dims_str()}\n"
        dims_str += f"{indent_str}\tlayer norm 1: {self.layer_norm1.get_dims_str()}"
        return dims_str



