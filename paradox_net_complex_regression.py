import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Note: This version is adapted for regression tasks like Lorenz.

# --- Helper function for Rotary Positional Encoding ---
def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embedding to the input tensor."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(2)
    return x_out.type_as(x)

class PositionalEncoding(nn.Module):
    """Generates rotary positional embeddings (RoPE)."""
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        theta = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.freqs_cis[t, :]

class ComplexLinear(nn.Module):
    """A linear layer that operates on complex-valued tensors, handling sequences."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_re = nn.Parameter(torch.randn(in_features // 2, out_features // 2) * 0.02)
        self.weight_im = nn.Parameter(torch.randn(in_features // 2, out_features // 2) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_re, x_im = x.real, x.imag
        if x.dim() == 3:
            out_re = torch.einsum('bsi,io->bso', x_re, self.weight_re) - torch.einsum('bsi,io->bso', x_im, self.weight_im)
            out_im = torch.einsum('bsi,io->bso', x_re, self.weight_im) + torch.einsum('bsi,io->bso', x_im, self.weight_re)
        else:
            out_re = x_re @ self.weight_re - x_im @ self.weight_im
            out_im = x_re @ self.weight_im + x_im @ self.weight_re
        return torch.complex(out_re, out_im)

class DiscretePatternLayer(nn.Module):
    """Hidden layer for complex numbers and sequences."""
    def __init__(self, input_dim, hidden_dim, n_patterns=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_patterns = n_patterns
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02)
        self.to_penultimate = ComplexLinear(hidden_dim, hidden_dim)

    def apply_self_processing(self, x: torch.Tensor) -> torch.Tensor:
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        return hidden

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.apply_self_processing(x)
        penultimate_contribution = self.to_penultimate(hidden)
        return hidden, penultimate_contribution

class PenultimatePatternLayer(nn.Module):
    """Penultimate layer adapted for complex numbers and regression."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_patterns=8):
        super().__init__()
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02)
        # REGRESSION CHANGE: Final predictor is a standard Linear layer on the real part
        self.output_predictor = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        # REGRESSION CHANGE: Use the real part of the hidden state for prediction
        predicted_output = self.output_predictor(hidden.real)
        return predicted_output

class ParadoxNetComplexRegression(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dims, output_dim, n_patterns=8):
        super().__init__()
        # REGRESSION CHANGE: The input is not a vocabulary, but a single feature
        self.linear_in = nn.Linear(input_dim, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        self.hidden_layers = nn.ModuleList()
        current_dim = embedding_dim
        for h_dim in hidden_dims:
            self.hidden_layers.append(DiscretePatternLayer(current_dim, h_dim, n_patterns))
            current_dim = h_dim
            
        self.penultimate_layer = PenultimatePatternLayer(hidden_dims[-1], hidden_dims[-1], output_dim, n_patterns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is shape [batch, seq_len]
        batch_size, seq_len = x.shape
        # REGRESSION CHANGE: Unsqueeze to add a feature dimension, then embed
        embedded = self.linear_in(x.unsqueeze(-1))
        
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

        penultimate_contributions = []
        for layer in self.hidden_layers:
            current_seq, penultimate = layer(current_seq)
            penultimate_contributions.append(penultimate.mean(dim=1))
            
        consensus_view = torch.sum(torch.stack(penultimate_contributions), dim=0)
        recursive_residual = current_seq.mean(dim=1)
        
        penultimate_input = consensus_view + recursive_residual
        
        final_output = self.penultimate_layer(penultimate_input)
        
        return final_output
