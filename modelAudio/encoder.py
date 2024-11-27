import torch
import torch.nn as nn
from modelAudio.positional_encoding import PositionalEncoding
from modelAudio.attention import SingleHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, dim, dim_ff):
        super().__init__()
        self.self_attention = SingleHeadAttention(dim)
        self.norm_attn = nn.LayerNorm(dim)
        # Fully connected feed forward network
        self.fc1 = nn.Linear(dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        # Dropout for regularization of 
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        x2, _ = self.self_attention(x, x, x, mask)
        x = self.norm_attn(x + self.dropout(x2))
        # Feed-forward network and residual connection
        x2 = self.fc2(torch.relu(self.fc1(x)))
        x = self.norm_ffn(x + self.dropout(x2))
        return x    
        

class Encoder(nn.Module):
    def __init__(self, dim, num_layers, dim_ff, max_seq_len, input_dim=200):
        super().__init__()
        self.projection = nn.Linear(200, dim)
        self.positional_encoding = PositionalEncoding(dim, max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(dim, dim_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, features, mask=None):
        # Project features to model dimension
        # Project err was here!
        x = self.projection(features)
                
        # Add positional encoding to the embeddings
        x = self.positional_encoding(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)
