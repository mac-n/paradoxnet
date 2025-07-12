import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=48, n_heads=1, d_ff=96, n_layers=4, output_dim=1, is_text=False, vocab_size=None):
        super(TransformerModel, self).__init__()
        self.is_text = is_text
        if is_text:
            assert vocab_size is not None, "vocab_size must be provided for text data"
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.linear_in = nn.Linear(1, d_model)

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.linear_out = nn.Linear(d_model, output_dim)

        self.d_model = d_model

    def forward(self, src):
        if self.is_text:
            src = self.embedding(src) * math.sqrt(self.d_model)
        else:
            src = src.unsqueeze(-1) # Add feature dimension
            src = self.linear_in(src)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1) # Aggregate over sequence length
        output = self.linear_out(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x
