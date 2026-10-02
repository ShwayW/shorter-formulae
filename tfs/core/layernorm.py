# imports
import torch.nn as nn
import torch.nn.functional as F

# class definition for layernorm super
class LayerNorm(nn.LayerNorm):
    def __init__(self, super_embed_dim):
        super().__init__(super_embed_dim)

        # the largest possible model
        self.super_embed_dim = super_embed_dim

        # the current configuration
        self.sample_embed_dim = None

        # initialize the parameters
        self.samples = {}

        
    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.samples['w'] = self.weight[:sample_embed_dim]
        self.samples['b'] = self.bias[:sample_embed_dim]
        return self.samples


    def forward(self, x):
        return F.layer_norm(x, (self.sample_embed_dim,), weight = self.weight[:self.sample_embed_dim], bias = self.bias[:self.sample_embed_dim], eps = self.eps)


    def get_dims_str(self, indent = 0):
        # prints out dimensions of all components, for debug
        indent_str = "\t" * indent
        b_list = list(self.samples['b'].shape)
        return f"{indent_str}[W: {list(self.samples['w'].shape)}, b: {b_list[0] if len(b_list) == 1 else b_list}]"



