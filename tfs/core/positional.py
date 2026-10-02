# imports
import math
import torch
import torch.nn as nn

# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length, mode='learnable'):
        super(PositionalEncoding, self).__init__()

        # largest model possible
        self.super_embed_dim = embed_dim
        self.mode = mode

        # current config
        self.sample_embed_dim = None

        # initializations
        self.max_seq_length = max_seq_length

        if mode == 'sinusoidal':
            pe = self._make_sinusoidal(max_seq_length, embed_dim)
        else:
            pe = torch.zeros(self.max_seq_length, self.super_embed_dim)
            nn.init.kaiming_uniform_(pe)

        self.register_buffer('pe', pe)
        self.samples = {}

    @staticmethod
    def _make_sinusoidal(max_seq_length, embed_dim):
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:embed_dim // 2])
        return pe

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        if self.mode == 'sinusoidal':
            # sinusoidal encodings must be recomputed for the sampled dim
            # because the frequencies depend on the full embedding dimension
            self.samples['pe'] = self._make_sinusoidal(self.max_seq_length, sample_embed_dim).to(self.pe.device)
        else:
            self.samples['pe'] = self.pe[:, :sample_embed_dim]

    def forward(self, x):
        return x + self.samples['pe'][:x.shape[1]].to(x.device)

    def get_dims_str(self, indent = 0):
        # prints out dimensions of all components, for debug
        indent_str = "\t" * indent
        return f"{indent_str}[P: {self.samples['pe'].shape[0]}, mode: {self.mode}]"
