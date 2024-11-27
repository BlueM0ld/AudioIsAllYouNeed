import torch

def create_padding_mask(seq):
    """Creates a mask for padding tokens (zeros)"""
    # seq shape: (batch_size, seq_len)
    device = seq.device
    mask = (seq != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask.to(device)

def create_look_ahead_mask(size, device):
    """Creates a triangular mask to prevent attending to future tokens"""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask.unsqueeze(0).to(device)  # (1, size, size)

def create_combined_mask(tgt):
    """Combines padding and look-ahead masks"""
    device = tgt.device
    size = tgt.size(1)
    padding_mask = create_padding_mask(tgt)  # (batch_size, 1, 1, seq_len)
    look_ahead_mask = create_look_ahead_mask(size, device)  # (1, size, size)
    
    # Combine the masks
    combined_mask = torch.max(
        look_ahead_mask.expand(padding_mask.size(0), -1, -1),
        padding_mask.squeeze(1)
    )
    return combined_mask.to(device) 