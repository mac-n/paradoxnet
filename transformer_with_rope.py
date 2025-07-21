import torch
import torch.nn as nn
import math

# RoPE implementation (from ParadoxNet Complex)
def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embedding to the input tensor."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(2)
    return x_out.type_as(x)

class RoPEPositionalEncoding(nn.Module):
    """Generates rotary positional embeddings (RoPE)."""
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        theta = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.freqs_cis[:seq_len, :]

class TransformerWithRoPE(nn.Module):
    """Transformer model with RoPE instead of standard positional encoding"""
    def __init__(self, vocab_size, d_model=48, n_heads=3, d_ff=96, n_layers=3):
        super(TransformerWithRoPE, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rope_encoder = RoPEPositionalEncoding(d_model)
        
        # Standard transformer encoder layers (but we'll apply RoPE manually)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.linear_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        # Embedding
        src = self.embedding(src) * math.sqrt(self.d_model)
        batch_size, seq_len = src.shape[0], src.shape[1]
        
        # Apply RoPE
        freqs_cis = self.rope_encoder(seq_len)
        
        # Need to ensure d_model is even for RoPE
        if self.d_model % 2 == 1:
            # Pad with zeros if odd dimension
            src = torch.cat([src, torch.zeros_like(src[:, :, :1])], dim=-1)
            src_with_rope = apply_rotary_pos_emb(src, freqs_cis)
            src_with_rope = src_with_rope[:, :, :-1]  # Remove padding
        else:
            src_with_rope = apply_rotary_pos_emb(src, freqs_cis)
        
        # Transformer processing
        output = self.transformer_encoder(src_with_rope)
        output = output.mean(dim=1)  # Aggregate over sequence length
        output = self.linear_out(output)
        return output

class StandardTransformer(nn.Module):
    """Standard transformer with sin/cos positional encoding (baseline)"""
    def __init__(self, vocab_size, d_model=48, n_heads=3, d_ff=96, n_layers=3):
        super(StandardTransformer, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.linear_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.linear_out(output)
        return output

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1).unsqueeze(0)
        return x