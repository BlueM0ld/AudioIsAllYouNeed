import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        
        initial_pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        initial_pe[:, 0::2] = torch.sin(position * div_term)
        initial_pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not parameter)
        self.register_buffer('pe', initial_pe.unsqueeze(0))  # [1, max_seq_len, dim]
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, dim]
        return x + self.pe[:, :x.size(1), :]

