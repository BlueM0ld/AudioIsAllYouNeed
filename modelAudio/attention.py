import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
    
    # q, k, v are the queries, keys, and value
    def forward(self, q,k,v,mask=None):
        
        k = k.transpose(-2, -1)
        
        # Compute attention scores
        # This is q.k^T/sqrt(d_k)
        attention_scores = (q @ k) * self.scale
        
        if mask is not None:
            # Mask out the padded tokens 
            # Can't use INF we get NaN in WANDB
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_scores = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention scores to the values
        attention_output = attention_scores @ v
        
        return attention_output, attention_scores