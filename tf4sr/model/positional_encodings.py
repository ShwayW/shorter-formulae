import torch

class PositionalEncodings(torch.nn.Module):
    def __init__(self, seq_length, d_model, dropout):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout)
    
    def forward(self, x):
        # PATCHED: build the table on x's device. Upstream allocates on CPU
        # unconditionally, so `x + pe` raises on GPU ("two devices, cuda:0 and
        # cpu"). Their own evaluate_model.py loads with map_location='cpu', so
        # the bug never surfaces upstream. Values are unchanged.
        dev = x.device
        pe = torch.zeros(self.seq_length, self.d_model, device=dev)
        numerator = torch.arange(0, self.seq_length, device=dev).unsqueeze(1)
        denominator = torch.pow(10e4, torch.arange(0, self.d_model, 2, device=dev) / self.d_model).unsqueeze(0)
        pe[:, 0::2] = torch.sin(numerator / denominator)
        pe[:, 1::2] = torch.cos(numerator / denominator)
        pe.requires_grad_(False)
        return self.dropout(x + pe)