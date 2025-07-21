import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

# --- Correct Helper Functions ---
def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embedding to the input tensor."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0)
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

class ComplexLinear(nn.Module):
    """A fully functional linear layer for complex-valued tensors."""
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

@dataclass
class LayerStats:
    """Track statistics for a single layer during forward pass"""
    prediction_errors: torch.Tensor
    confidence_values: torch.Tensor
    penultimate_magnitude: torch.Tensor
    continue_magnitude: torch.Tensor
    layer_idx: int
    pattern_usage: torch.Tensor
    pattern_entropy: float = 0.0
    self_paradox_magnitude: float = 0.0

class HybridComplexLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        self.last_entropy = 0.0
        self.last_paradox_magnitude = 0.0

        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        self.to_penultimate = ComplexLinear(hidden_dim, penultimate_dim)

        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02)
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        self.next_pattern_dict = nn.Parameter(torch.randn(n_patterns, next_dim // 2, dtype=torch.cfloat) * 0.02)
        self.next_pattern_attention = nn.Linear(hidden_dim, n_patterns)
        self.last_stats: Optional[LayerStats] = None

    def compress_activity(self, x: torch.Tensor, is_next_layer: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2: x = x.unsqueeze(1)
        attention_layer = self.next_pattern_attention if is_next_layer else self.pattern_attention
        patterns = self.next_pattern_dict if is_next_layer else self.pattern_dict
        x_real_interleaved = torch.view_as_real(x).flatten(start_dim=2)
        attn_scores = attention_layer(x_real_interleaved)
        pattern_weights = F.softmax(attn_scores, dim=-1)
        with torch.no_grad():
            entropy = -torch.sum(pattern_weights * torch.log(pattern_weights + 1e-10), dim=-1)
            self.last_entropy = entropy.mean().item()
        compressed = torch.einsum('bsp,pd->bsd', pattern_weights.cfloat(), patterns)
        return compressed.squeeze(1), pattern_weights.squeeze(1)

    def apply_self_paradox_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-prediction paradox using magnitude gating."""
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        
        with torch.no_grad():
            self.last_paradox_magnitude = torch.mean(paradox.abs()).item()
            
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        return hidden

    def forward(self, x: torch.Tensor, next_layer: Optional['HybridComplexLayer'], 
                layer_idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        
        hidden = self.apply_self_paradox_nonlinearity(x)
        
        if next_layer is not None:
            # --- FINAL LOGIC FOR TRUE PREDICTION ---
            
            # 1. PREDICT: Use this layer's predictive dictionary (`next_pattern_dict`)
            # to predict the next layer's compressed state.
            predicted_next, _ = self.compress_activity(hidden, is_next_layer=True)

            # 2. TARGET: Get the next layer's ACTUAL compressed state by having it
            # use its own dictionary (`pattern_dict`).
            with torch.no_grad():
                actual_next_hidden = next_layer.apply_self_paradox_nonlinearity(hidden)
                compressed_next, _ = next_layer.compress_activity(actual_next_hidden, is_next_layer=False)
            
            # --- END OF CHANGE ---

            pred_error = torch.mean((compressed_next - predicted_next).abs()**2, dim=1, keepdim=True)
            
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            penultimate_features = self.to_penultimate(hidden)
            penultimate_contribution = penultimate_features * confidence.cfloat()
            continue_up = hidden * (1 - confidence).cfloat()

            _, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(penultimate_contribution.detach().abs(), dim=-1),
                continue_magnitude=torch.mean(continue_up.detach().abs(), dim=-1),
                layer_idx=layer_idx,
                pattern_usage=my_patterns.detach().mean(0),
                pattern_entropy=self.last_entropy,
                self_paradox_magnitude=self.last_paradox_magnitude
            )
            return continue_up, penultimate_contribution, pred_error
            
        else: # This is the last layer
            penultimate_contribution = self.to_penultimate(hidden)
            _, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            self.last_stats = LayerStats(
                prediction_errors=torch.zeros(1, 1, device=x.device),
                confidence_values=torch.ones(1, 1, device=x.device),
                penultimate_magnitude=torch.mean(penultimate_contribution.detach().abs(), dim=-1),
                continue_magnitude=torch.tensor(0.0, device=x.device),
                layer_idx=layer_idx,
                pattern_usage=my_patterns.detach().mean(0),
                pattern_entropy=self.last_entropy,
                self_paradox_magnitude=self.last_paradox_magnitude
            )
            return None, penultimate_contribution, None

class ParadoxNetHybridComplex(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dims, penultimate_dim, n_patterns=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        self.layers = nn.ModuleList()
        current_dim = embedding_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else penultimate_dim
            layer = HybridComplexLayer(
                input_dim=current_dim, 
                hidden_dim=hidden_dim, 
                next_dim=next_dim, 
                penultimate_dim=penultimate_dim,
                n_patterns=n_patterns
            )
            self.layers.append(layer)
            current_dim = hidden_dim
        
        self.final = nn.Linear(penultimate_dim // 2, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

        all_errors = []
        final_penultimate_sum = torch.zeros(batch_size, self.layers[0].to_penultimate.out_features // 2, dtype=torch.cfloat, device=x.device)

        for i in range(seq_len):
            current_slice = current_seq[:, i, :]
            penultimate_slice_contributions = []
            for j, layer in enumerate(self.layers):
                next_layer = self.layers[j+1] if j < len(self.layers)-1 else None
                current_slice, penultimate, error = layer(current_slice, next_layer, j)
                if error is not None and i == seq_len -1:
                    all_errors.append(error)
                penultimate_slice_contributions.append(penultimate)
                if current_slice is None:
                    break
            
            penultimate_sum_slice = torch.sum(torch.stack(penultimate_slice_contributions), dim=0)
            final_penultimate_sum += penultimate_sum_slice

        final_penultimate_sum /= seq_len
        output = self.final(final_penultimate_sum.real)
        
        return output, torch.cat(all_errors, dim=1) if all_errors else None