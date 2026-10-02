# imports
import torch.nn as nn
import torch.nn.functional as F

# class definition for Embedding
class Embedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx = None):
        # initialize the weights
        super().__init__(num_embeddings, embedding_dim, padding_idx = padding_idx)

        # largest possible dimensions
        self.super_embed_dim = embedding_dim

        # current config
        self.sample_embed_dim = None
        self.padding_idx = padding_idx

        # init params
        self.samples = {}
        #nn.init.kaiming_uniform_(self.weight)

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.samples['w'] = self.weight[:, :self.sample_embed_dim]

    def forward(self, x):
        res = F.embedding(x, self.weight[:, :self.sample_embed_dim], padding_idx = self.padding_idx)
        return res

    def get_dims_str(self, indent = 0):
        # prints out dimensions of all components, for debug
        indent_str = "\t" * indent
        return f"{indent_str}[W: {list(self.samples['w'].shape)}]"



