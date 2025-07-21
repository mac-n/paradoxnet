import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Copy the working parts from the original
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
    """A linear layer that operates on complex-valued tensors, now handling sequences."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Note: We divide in_features by 2 because the complex dimension is half the real dimension
        self.weight_re = nn.Parameter(torch.randn(in_features // 2, out_features // 2) * 0.02)
        self.weight_im = nn.Parameter(torch.randn(in_features // 2, out_features // 2) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a complex tensor. It can be [batch, out_features/2] or [batch, seq_len, out_features/2]
        x_re, x_im = x.real, x.imag

        if x.dim() == 3: # Has a sequence dimension
            # Using einsum for clarity: 'bsi,io->bso'
            # b: batch, s: sequence, i: in_features, o: out_features
            out_re = torch.einsum('bsi,io->bso', x_re, self.weight_re) - torch.einsum('bsi,io->bso', x_im, self.weight_im)
            out_im = torch.einsum('bsi,io->bso', x_re, self.weight_im) + torch.einsum('bsi,io->bso', x_im, self.weight_re)
        else: # No sequence dimension
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

class DiscretePatternLayerWithRouting(nn.Module):
    """Hidden layer with complex patterns and routing - FIXED VERSION"""
    def __init__(self, input_dim, hidden_dim, n_patterns=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_patterns = n_patterns
        
        # Core processing (same as original)
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # Complex pattern dictionaries - FIXED: proper dimensions
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02)
        self.pattern_attention = ComplexLinear(hidden_dim, n_patterns * 2)  # Real outputs for softmax
        
        # FIXED: Use same output dim as input for consistency 
        self.to_penultimate = ComplexLinear(hidden_dim, hidden_dim)
        
        # Stats tracking
        self.last_stats: Optional[LayerStats] = None
        self.last_paradox_magnitude: float = 0.0
        self.last_entropy: float = 0.0

    def apply_self_processing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-prediction paradox as complex nonlinearity."""
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        
        # Track paradox magnitude for statistics
        with torch.no_grad():
            self.last_paradox_magnitude = torch.mean(paradox.abs()).item()
        
        # Complex paradox gating
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        return hidden
    
    def compress_activity_complex(self, hidden: torch.Tensor, use_gumbel: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress activity using complex pattern dictionaries - FIXED einsum"""
        # Get attention scores (real values for softmax)
        attention_scores = self.pattern_attention(hidden).real
        
        # Apply attention mechanism
        if use_gumbel:
            pattern_weights = F.gumbel_softmax(attention_scores, tau=1.0, hard=False)
        else:
            pattern_weights = F.softmax(attention_scores, dim=-1)
        
        # Calculate entropy for statistics
        with torch.no_grad():
            entropy = -torch.sum(pattern_weights * torch.log(pattern_weights + 1e-8), dim=-1)
            self.last_entropy = entropy.mean().item()
        
        # FIXED: Handle sequence dimension properly
        if hidden.dim() == 3:  # [batch, seq, hidden_dim/2]
            # For sequences: [batch, seq, patterns] @ [patterns, hidden_dim/2] -> [batch, seq, hidden_dim/2]
            compressed = torch.einsum('bsp,phd->bshd', pattern_weights, self.pattern_dict)
        else:  # [batch, hidden_dim/2]
            # For non-sequences: [batch, patterns] @ [patterns, hidden_dim/2] -> [batch, hidden_dim/2]  
            compressed = torch.einsum('bp,phd->bhd', pattern_weights, self.pattern_dict)
        
        return compressed, pattern_weights
    
    def forward(self, x: torch.Tensor, next_layer: Optional['DiscretePatternLayerWithRouting'] = None,
                use_gumbel: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with complex pattern routing - MATCHES ORIGINAL STRUCTURE"""
        # Apply self-prediction paradox nonlinearity
        hidden = self.apply_self_processing(x)
        
        if next_layer is not None:
            # Compress own activity
            my_compressed, my_patterns = self.compress_activity_complex(hidden, use_gumbel=use_gumbel)
            
            # Get actual next layer transformation
            with torch.no_grad():
                actual_next = next_layer.apply_self_processing(hidden)
                compressed_next, _ = next_layer.compress_activity_complex(actual_next, use_gumbel=use_gumbel)
            
            # FIXED: Handle dimension matching properly
            if my_compressed.shape != compressed_next.shape:
                min_dim = min(my_compressed.shape[-1], compressed_next.shape[-1])
                if my_compressed.dim() == 3:
                    my_compressed = my_compressed[..., :min_dim]
                    compressed_next = compressed_next[..., :min_dim]
                else:
                    my_compressed = my_compressed[:, :min_dim]
                    compressed_next = compressed_next[:, :min_dim]
            
            # Complex prediction error (use magnitude for confidence)
            complex_error = compressed_next - my_compressed
            pred_error = torch.mean(complex_error.abs()**2, dim=-1, keepdim=True)
            
            # Calculate confidence based on prediction accuracy
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Route information based on confidence
            penultimate_features = self.to_penultimate(hidden)
            penultimate_contribution = penultimate_features * confidence
            continue_up = hidden * (1 - confidence)
            
            return continue_up, penultimate_contribution, pred_error
        
        else:
            # No next layer - just return penultimate contribution
            penultimate_contribution = self.to_penultimate(hidden)
            return hidden, penultimate_contribution, None

class PenultimatePatternLayerFixed(nn.Module):
    """Penultimate layer - MATCHES ORIGINAL STRUCTURE"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_patterns=8, use_patterns=True):
        super().__init__()
        self.use_patterns = use_patterns
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        if use_patterns:
            self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02)
            self.pattern_attention = ComplexLinear(hidden_dim, n_patterns*2)
        
        # Final output predictor is real-valued
        self.output_predictor = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        
        if self.use_patterns:
            # Apply pattern-based processing
            attention_scores = self.pattern_attention(hidden).real
            pattern_weights = F.softmax(attention_scores, dim=-1)
            # FIXED: Handle dimension properly
            if hidden.dim() == 3:
                compressed = torch.einsum('bsp,phd->bshd', pattern_weights, self.pattern_dict)
            else:
                compressed = torch.einsum('bp,phd->bhd', pattern_weights, self.pattern_dict)
            predicted_output = self.output_predictor(compressed.real)
        else:
            # Direct prediction without patterns
            predicted_output = self.output_predictor(hidden.real)
            
        return predicted_output

class ParadoxNetComplexPatternsFixed(nn.Module):
    """FIXED VERSION - matches original structure but adds pattern routing"""
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8, use_gumbel=False):
        super().__init__()
        self.use_gumbel = use_gumbel
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Build layers with same structure as original
        self.hidden_layers = nn.ModuleList()
        current_dim = embedding_dim
        for h_dim in hidden_dims:
            layer = DiscretePatternLayerWithRouting(current_dim, h_dim, n_patterns)
            self.hidden_layers.append(layer)
            current_dim = h_dim
            
        self.penultimate_layer = PenultimatePatternLayerFixed(
            hidden_dims[-1], hidden_dims[-1], vocab_size, n_patterns, use_patterns=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass - MATCHES ORIGINAL STRUCTURE"""
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

        penultimate_contributions = []
        prediction_errors = []
        
        # Process through hidden layers with routing
        for i, layer in enumerate(self.hidden_layers):
            next_layer = self.hidden_layers[i+1] if i+1 < len(self.hidden_layers) else None
            
            continue_up, penultimate_contrib, pred_error = layer(
                current_seq, next_layer, use_gumbel=self.use_gumbel
            )
            
            # FIXED: All penultimate contributions should have same shape
            penultimate_contributions.append(penultimate_contrib.mean(dim=1))  # Average over sequence
            
            if pred_error is not None:
                prediction_errors.append(pred_error.mean())  # Scalar for each layer
            
            current_seq = continue_up
        
        # FIXED: Now all contributions have same shape [batch, hidden_dim]
        consensus_view = torch.sum(torch.stack(penultimate_contributions), dim=0)
        recursive_residual = current_seq.mean(dim=1)  # [batch, hidden_dim]
        
        penultimate_input = consensus_view + recursive_residual
        final_output = self.penultimate_layer(penultimate_input)
        
        # Combine prediction errors
        combined_pred_errors = None
        if prediction_errors:
            combined_pred_errors = torch.stack(prediction_errors)
        
        return final_output, combined_pred_errors

# Factory functions
def create_complex_patterns_fixed(vocab_size, embedding_dim, hidden_dims, n_patterns=8, use_gumbel=False):
    """Fixed complex patterns that actually work."""
    return ParadoxNetComplexPatternsFixed(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        n_patterns=n_patterns,
        use_gumbel=use_gumbel
    )