import torch 
import torch.nn as nn
from modelAudio.encoder import Encoder
from modelAudio.decoder import Decoder
from modelAudio.positional_encoding import PositionalEncoding

import torch
import torch.nn as nn
from modelAudio.encoder import Encoder
from modelAudio.decoder import Decoder
from modelAudio.positional_encoding import PositionalEncoding

class ConvG(nn.Module):
    def __init__(self, chan_num):
        super().__init__()
        self.conv1 = nn.Conv1d(chan_num, chan_num, 3, padding=1)
        self.g1 = nn.GELU()
        self.conv2 = nn.Conv1d(chan_num, chan_num, 3, padding=1)
        self.g2 = nn.GELU()
        self.norm = nn.BatchNorm1d(chan_num)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.g1(x)
        x = self.conv2(x)
        x = self.g2(x)
        x = self.norm(x)
        return x


class Whisper(nn.Module):
    
    def __init__(self, 
                 vocab_size,
                 dim=512,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_ff=2048,
                 max_seq_len=128,
                 pad_idx=0):
        
        super().__init__()
        
        self.convG = ConvG(chan_num=128)
        
        self.encoder = Encoder(
            dim=dim,
            num_layers=num_encoder_layers,
            dim_ff=dim_ff,
            max_seq_len=max_seq_len,
            input_dim=128
        )
        
        self.decoder = Decoder(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_decoder_layers,
            dim_ff=dim_ff,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx
        )
            
        self.output_projection = nn.Linear(dim, vocab_size)  # Corrected to match output
        
    def forward(self, mel_spectrogram):
        # Ensure the input is in the correct shape
        # Change from [batch_size, channels, n_frames, n_mels] to [batch_size, n_frames, n_mels]
        mel_spectrogram = mel_spectrogram.squeeze(1)  # Remove the channel dimension
        mel_spectrogram = mel_spectrogram.permute(0, 2, 1)  # Change to [batch_size, n_frames, n_mels]
        print("this is the tensor for melspec", mel_spectrogram.shape)
        # Pass the mel spectrogram through the ConvG
        processed_spectrogram = self.convG(mel_spectrogram)

        # Pass processed spectrogram through the encoder
        encoder_output = self.encoder(processed_spectrogram)
        # decoder_output = self.decoder(encoder_output)

        # Use encoder output directly for classification
        pooled_output = torch.mean(encoder_output, dim=1)  # Average pooling over the sequence length

        # Use pooled output for classification
        output = self.output_projection(pooled_output)
        
        return output
