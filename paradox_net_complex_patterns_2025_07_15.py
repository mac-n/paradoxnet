import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

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

# Complex Paradox Net with Pattern Dictionaries and Routing - 2025_07_15
# Integrates complex-valued pattern compression with confidence-based routing

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

class DiscretePatternLayer(nn.Module):
    """Hidden layer with complex patterns and routing."""
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8, gumbel_temp=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        self.gumbel_temp = gumbel_temp
        
        # Core processing
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # Complex pattern dictionaries
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02)
        self.pattern_attention = ComplexLinear(hidden_dim, n_patterns * 2)  # Real outputs for softmax
        
        # Next layer prediction patterns
        self.next_pattern_dict = nn.Parameter(torch.randn(n_patterns, next_dim // 2, dtype=torch.cfloat) * 0.02)
        self.next_pattern_attention = ComplexLinear(hidden_dim, n_patterns * 2)
        
        # Output pathway
        self.to_penultimate = ComplexLinear(hidden_dim, penultimate_dim)
        
        # Stats tracking
        self.last_stats: Optional[LayerStats] = None
        self.last_paradox_magnitude: float = 0.0

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
    
    def compress_activity_complex(self, hidden: torch.Tensor, is_next_layer: bool = False, use_gumbel: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress activity using complex pattern dictionaries."""
        if is_next_layer:
            attention_scores = self.next_pattern_attention(hidden).real
            patterns = self.next_pattern_dict
        else:
            attention_scores = self.pattern_attention(hidden).real
            patterns = self.pattern_dict
        
        # Apply attention mechanism
        if use_gumbel:
            pattern_weights = F.gumbel_softmax(attention_scores, tau=self.gumbel_temp, hard=False)
        else:
            pattern_weights = F.softmax(attention_scores, dim=-1)
        
        # Calculate entropy for statistics
        with torch.no_grad():
            entropy = -torch.sum(pattern_weights * torch.log(pattern_weights + 1e-8), dim=-1)
            self.last_entropy = entropy.mean().item()
        
        # Complex pattern compression: [batch, patterns] @ [patterns, hidden_dim/2] -> [batch, hidden_dim/2]
        compressed = torch.einsum('bp,phd->bhd', pattern_weights, patterns)
        return compressed, pattern_weights
    
    def forward(self, x: torch.Tensor, next_layer: Optional['DiscretePatternLayer'] = None, 
                layer_idx: int = 0, use_gumbel: bool = False) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with complex pattern routing."""
        # Ensure x is at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Apply self-prediction paradox nonlinearity
        hidden = self.apply_self_processing(x)
        
        if next_layer is not None:
            # Compress own activity and predict next layer
            my_compressed, my_patterns = self.compress_activity_complex(hidden, is_next_layer=False, use_gumbel=use_gumbel)
            predicted_next = my_compressed
            
            # Get actual next layer transformation
            with torch.no_grad():
                actual_next = next_layer.apply_self_processing(hidden)
                compressed_next, _ = next_layer.compress_activity_complex(actual_next, is_next_layer=True, use_gumbel=use_gumbel)
            
            # Match dimensions for prediction error
            min_dim = min(predicted_next.size(-1), compressed_next.size(-1))
            if predicted_next.dim() == 3:  # sequence dimension
                predicted_next = predicted_next[..., :min_dim]
                compressed_next = compressed_next[..., :min_dim]
            else:
                predicted_next = predicted_next[:, :min_dim]
                compressed_next = compressed_next[:, :min_dim]
            
            # Complex prediction error (use magnitude for confidence)
            complex_error = compressed_next - predicted_next
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
            
            # Track statistics
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach().abs(), dim=-1)),
                continue_magnitude=torch.mean(torch.norm(continue_up.detach().abs(), dim=-1)),
                layer_idx=layer_idx,
                pattern_usage=my_patterns.detach().mean(0),
                pattern_entropy=getattr(self, 'last_entropy', 0.0),
                self_paradox_magnitude=self.last_paradox_magnitude
            )
            
            return continue_up, penultimate_contribution, pred_error
        
        else:
            # No next layer - just return penultimate contribution
            penultimate_contribution = self.to_penultimate(hidden)
            return None, penultimate_contribution, None

class PenultimatePatternLayer(nn.Module):
    """Penultimate layer with optional consensus view mechanism."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_patterns=8, use_consensus=True):
        super().__init__()
        self.use_consensus = use_consensus
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        if use_consensus:
            self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02)
            self.pattern_attention = ComplexLinear(hidden_dim, n_patterns*2)
        
        # Final output predictor is real-valued
        self.output_predictor = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        
        if self.use_consensus:
            # Apply pattern-based processing
            attention_scores = self.pattern_attention(hidden).real
            pattern_weights = F.softmax(attention_scores, dim=-1)
            compressed = torch.einsum('bp,phd->bhd', pattern_weights, self.pattern_dict)
            predicted_output = self.output_predictor(compressed.real)
        else:
            # Direct prediction without consensus view
            predicted_output = self.output_predictor(hidden.real)
            
        return predicted_output

class ParadoxNetComplexPatterns(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dims, penultimate_dim, n_patterns=8, 
                 use_consensus=True, use_gumbel=False, gumbel_temp=1.0):
        super().__init__()
        self.use_consensus = use_consensus
        self.use_gumbel = use_gumbel
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Build layers with proper dimensions for routing
        self.hidden_layers = nn.ModuleList()
        current_dim = embedding_dim
        
        for i, h_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i+1] if i+1 < len(hidden_dims) else h_dim
            layer = DiscretePatternLayer(
                input_dim=current_dim, 
                hidden_dim=h_dim, 
                next_dim=next_dim,
                penultimate_dim=penultimate_dim,
                n_patterns=n_patterns,
                gumbel_temp=gumbel_temp
            )
            self.hidden_layers.append(layer)
            current_dim = h_dim
            
        self.penultimate_layer = PenultimatePatternLayer(
            input_dim=penultimate_dim, 
            hidden_dim=hidden_dims[-1], 
            output_dim=vocab_size, 
            n_patterns=n_patterns,
            use_consensus=use_consensus
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with pattern routing and prediction error collection."""
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        # Apply rotary positional encoding and convert to complex
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

        penultimate_contributions = []
        prediction_errors = []
        
        # Process through hidden layers with routing
        for i, layer in enumerate(self.hidden_layers):
            next_layer = self.hidden_layers[i+1] if i+1 < len(self.hidden_layers) else None
            
            continue_up, penultimate_contrib, pred_error = layer(
                current_seq, next_layer, layer_idx=i, use_gumbel=self.use_gumbel
            )
            
            if penultimate_contrib is not None:
                # Average over sequence dimension for penultimate contribution
                penultimate_contributions.append(penultimate_contrib.mean(dim=1))
            
            if pred_error is not None:
                prediction_errors.append(pred_error)
            
            # Update current sequence for next layer
            if continue_up is not None:
                current_seq = continue_up
            else:
                # If no continue_up, just use the processed hidden state
                current_seq = layer.apply_self_processing(current_seq)
        
        # Combine contributions
        if penultimate_contributions:
            if self.use_consensus:
                consensus_view = torch.sum(torch.stack(penultimate_contributions), dim=0)
                final_hidden = consensus_view
            else:
                # Just use the last layer's contribution
                final_hidden = penultimate_contributions[-1]
        else:
            # Fallback: use final sequence state
            final_hidden = current_seq.mean(dim=1)
        
        # Final output
        final_output = self.penultimate_layer(final_hidden)
        
        # Combine prediction errors for loss
        combined_pred_errors = None
        if prediction_errors:
            combined_pred_errors = torch.cat(prediction_errors, dim=0)
        
        return final_output, combined_pred_errors


# Factory functions for different configurations
def create_complex_patterns_softmax(vocab_size, embedding_dim, hidden_dims, penultimate_dim, n_patterns=8, use_consensus=True):
    """Standard complex patterns with softmax attention."""
    return ParadoxNetComplexPatterns(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim, 
        hidden_dims=hidden_dims,
        penultimate_dim=penultimate_dim,
        n_patterns=n_patterns,
        use_consensus=use_consensus,
        use_gumbel=False
    )

def create_complex_patterns_gumbel(vocab_size, embedding_dim, hidden_dims, penultimate_dim, n_patterns=8, gumbel_temp=1.0, use_consensus=True):
    """Complex patterns with Gumbel softmax for sharper attention."""
    return ParadoxNetComplexPatterns(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims, 
        penultimate_dim=penultimate_dim,
        n_patterns=n_patterns,
        use_consensus=use_consensus,
        use_gumbel=True,
        gumbel_temp=gumbel_temp
    )

def create_complex_patterns_no_consensus(vocab_size, embedding_dim, hidden_dims, penultimate_dim, n_patterns=8):
    """Complex patterns without consensus view in penultimate layer."""
    return ParadoxNetComplexPatterns(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        penultimate_dim=penultimate_dim, 
        n_patterns=n_patterns,
        use_consensus=False,
        use_gumbel=False
    )
