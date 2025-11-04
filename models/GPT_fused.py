import math, torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]

class GPTFusedDynamics(nn.Module):
    """
    Generalized Transformer Dynamics for single- or multi-modal latent sequences.
    Supports gated residual connection (for temporal stability).
    """
    def __init__(self, d_model=64, n_layer=4, n_head=4, d_ff=256, dropout=0.1, gated=True):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.pos = PositionalEncoding(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.gated = gated
        if gated:
            self.gate = nn.Linear(d_model, d_model)

    def _causal_mask(self, L, device):
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        return mask

    def forward(self, x):
        # x: (B, L, D)
        y = self.pos(x)
        y = self.encoder(y, mask=self._causal_mask(y.size(1), y.device))
        y = self.proj(y)
        if self.gated:
            g = torch.sigmoid(self.gate(y))
            y = g * y + (1.0 - g) * x
        return y
