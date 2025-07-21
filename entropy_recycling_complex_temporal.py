import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

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
    """A linear layer that operates on complex-valued tensors."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Complex dimension is half the real dimension
        self.weight_re = nn.Parameter(torch.randn(in_features // 2, out_features // 2) * 0.02)
        self.weight_im = nn.Parameter(torch.randn(in_features // 2, out_features // 2) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a complex tensor
        x_re, x_im = x.real, x.imag

        if x.dim() == 3: # Has a sequence dimension
            # Using einsum for clarity: 'bsi,io->bso'
            out_re = torch.einsum('bsi,io->bso', x_re, self.weight_re) - torch.einsum('bsi,io->bso', x_im, self.weight_im)
            out_im = torch.einsum('bsi,io->bso', x_re, self.weight_im) + torch.einsum('bsi,io->bso', x_im, self.weight_re)
        else: # No sequence dimension
            out_re = x_re @ self.weight_re - x_im @ self.weight_im
            out_im = x_re @ self.weight_im + x_im @ self.weight_re
            
        return torch.complex(out_re, out_im)

@dataclass
class LayerStats:
    """Track statistics for complex temporal entropy recycling layers"""
    prediction_errors: torch.Tensor
    confidence_values: torch.Tensor
    penultimate_magnitude: torch.Tensor
    continue_magnitude: torch.Tensor
    layer_idx: int
    pattern_usage: torch.Tensor
    pattern_entropy: float = 0.0
    self_paradox_magnitude: float = 0.0
    composition_alpha: float = 0.0
    entropy_magnitude: float = 0.0
    temporal_entropy_magnitude: float = 0.0

class ComplexTemporalEntropyRecyclingLayer(nn.Module):
    """Complex temporal entropy recycling layer with RoPE"""
    
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8,
                 composition_from_prev=True, prev_layer=None, is_bottom=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        self.composition_from_prev = composition_from_prev
        self.is_bottom = is_bottom
        
        # Complex processing pathway
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # COMPOSITIONAL PATTERN MECHANISM (complex)
        if composition_from_prev and prev_layer is not None:
            self.composition_weights = nn.Parameter(
                torch.randn(n_patterns, prev_layer.n_patterns, dtype=torch.cfloat) / (prev_layer.n_patterns ** 0.5)
            )
            self.prev_layer = prev_layer
            self.pattern_dict = None
        else:
            self.pattern_dict = nn.Parameter(
                torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02
            )
            self.composition_weights = None
            self.prev_layer = None
        
        # Next layer prediction patterns (complex)
        self.next_pattern_dict = nn.Parameter(
            torch.randn(n_patterns, next_dim // 2, dtype=torch.cfloat) * 0.02
        )
        self.next_pattern_attention = ComplexLinear(hidden_dim, n_patterns * 2)  # Real output for softmax
        
        # Pattern attention (complex)
        self.pattern_attention = ComplexLinear(hidden_dim, n_patterns * 2)  # Real output for softmax
        
        # Output pathway (complex)
        self.to_penultimate = ComplexLinear(hidden_dim, penultimate_dim)
        
        # TEMPORAL ENTROPY PROCESSING (special for bottom layer)
        if is_bottom:
            self.temporal_entropy_processor = ComplexLinear(hidden_dim, hidden_dim)
            self.temporal_entropy_predictor = ComplexLinear(hidden_dim, hidden_dim)  # Predict optimal entropy processing
            self.temporal_entropy_gate = nn.Linear(hidden_dim // 2, 1)  # Real gate for temporal entropy influence
        
        # Stats tracking
        self.last_stats: Optional[LayerStats] = None
    
    def get_pattern_dict(self):
        """Get compositional pattern dictionary"""
        if self.composition_weights is not None and self.prev_layer is not None:
            prev_patterns = self.prev_layer.get_pattern_dict()
            # Complex matrix multiplication for composition
            composed_patterns = torch.einsum('ij,jk->ik', self.composition_weights, prev_patterns)
            return composed_patterns
        else:
            return self.pattern_dict
    
    def extract_patterns_and_entropy(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract patterns and compute entropy as residual in complex space"""
        patterns = self.get_pattern_dict()
        
        # Complex attention over patterns
        attn = self.pattern_attention(hidden)  # Real output for softmax
        pattern_weights = F.softmax(attn, dim=-1)
        
        # Pattern reconstruction (what patterns can explain) - complex
        if hidden.dim() == 2:  # [batch, complex_dim]
            pattern_reconstruction = torch.einsum('bp,pk->bk', pattern_weights, patterns)
        else:  # Sequence case [batch, seq, complex_dim]
            pattern_reconstruction = torch.einsum('bsp,pk->bsk', pattern_weights, patterns)
        
        # Entropy = what patterns CANNOT explain (complex)
        entropy = hidden - pattern_reconstruction
        
        return pattern_reconstruction, entropy, pattern_weights
    
    def apply_self_paradox_nonlinearity(self, x: torch.Tensor, temporal_entropy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Complex paradox mechanism with temporal entropy integration"""
        # Process input (complex)
        hidden_linear = self.process(x)
        
        # Temporal entropy integration for bottom layer
        if self.is_bottom and temporal_entropy is not None:
            # Handle batch size mismatch
            batch_size = hidden_linear.size(0)
            temp_batch_size = temporal_entropy.size(0)
            
            if temp_batch_size != batch_size:
                if temp_batch_size < batch_size:
                    repeats = (batch_size + temp_batch_size - 1) // temp_batch_size
                    temporal_entropy = temporal_entropy.repeat(repeats, 1)[:batch_size]
                else:
                    temporal_entropy = temporal_entropy[:batch_size]
            
            # Process temporal entropy (complex) and gate its influence (real)
            processed_temporal = self.temporal_entropy_processor(temporal_entropy)
            temporal_gate = torch.sigmoid(self.temporal_entropy_gate(hidden_linear.real))
            
            # Apply temporal influence using complex multiplication
            temporal_influence = processed_temporal * temporal_gate.unsqueeze(-1)
            hidden_linear = hidden_linear + temporal_influence
        
        # Self-prediction paradox mechanism (complex)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        paradox_magnitude = torch.norm(paradox, dim=-1, keepdim=True)
        
        # "I'm confused about myself → let more through" (complex gating)
        hidden = hidden_linear * torch.sigmoid(paradox_magnitude)
        return hidden
    
    def forward(self, x: torch.Tensor, next_layer: Optional['ComplexTemporalEntropyRecyclingLayer'], 
                layer_idx: int, temporal_entropy: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Forward pass with complex temporal entropy extraction"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Apply complex paradox mechanism with temporal entropy
        hidden = self.apply_self_paradox_nonlinearity(x, temporal_entropy)
        
        # Extract patterns and entropy (complex)
        pattern_reconstruction, entropy, pattern_weights = self.extract_patterns_and_entropy(hidden)
        
        # Track temporal entropy magnitude
        temporal_entropy_magnitude = 0.0
        if temporal_entropy is not None:
            temporal_entropy_magnitude = torch.mean(torch.norm(temporal_entropy, dim=-1)).item()
        
        if next_layer is not None:
            # Predict next layer (complex)
            predicted_next = pattern_reconstruction
            
            # Get actual next layer behavior
            with torch.no_grad():
                actual_next = next_layer.apply_self_paradox_nonlinearity(hidden)
                actual_patterns, _, _ = next_layer.extract_patterns_and_entropy(actual_next)
            
            # Complex prediction error
            pred_error = torch.mean(torch.norm(actual_patterns - predicted_next, dim=-1)**2, dim=-1, keepdim=True)
            
            # Routing confidence
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Route information (complex -> real for penultimate)
            penultimate_features = self.to_penultimate(pattern_reconstruction)
            if penultimate_features.dim() > 2:  # Sequence case
                penultimate_features = penultimate_features.mean(dim=1)  # Average over sequence
            penultimate_contribution = penultimate_features.real * confidence  # Use real part
            continue_up = hidden * (1 - confidence.unsqueeze(-1))
            
            # Track composition statistics
            composition_alpha = 0.0
            if self.composition_weights is not None:
                with torch.no_grad():
                    comp_weights_norm = F.softmax(torch.abs(self.composition_weights), dim=-1)
                    composition_alpha = 1.0 - torch.mean(torch.max(comp_weights_norm, dim=-1)[0]).item()
            
            # Enhanced statistics
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.mean(torch.norm(continue_up.detach(), dim=-1)),
                layer_idx=layer_idx,
                pattern_usage=pattern_weights.detach().mean(0),
                pattern_entropy=0.0,
                self_paradox_magnitude=torch.mean(torch.norm(hidden.detach(), dim=-1)).item(),
                composition_alpha=composition_alpha,
                entropy_magnitude=torch.mean(torch.norm(entropy, dim=-1)).item(),
                temporal_entropy_magnitude=temporal_entropy_magnitude
            )
            
            return continue_up, penultimate_contribution, pred_error, entropy
            
        else:
            # Last layer processing (complex -> real)
            penultimate_features = self.to_penultimate(pattern_reconstruction)
            if penultimate_features.dim() > 2:  # Sequence case
                penultimate_features = penultimate_features.mean(dim=1)
            penultimate_contribution = penultimate_features.real  # Use real part
            
            composition_alpha = 0.0
            if self.composition_weights is not None:
                with torch.no_grad():
                    comp_weights_norm = F.softmax(torch.abs(self.composition_weights), dim=-1)
                    composition_alpha = 1.0 - torch.mean(torch.max(comp_weights_norm, dim=-1)[0]).item()
            
            self.last_stats = LayerStats(
                prediction_errors=torch.zeros(1, 1, device=x.device),
                confidence_values=torch.ones(1, 1, device=x.device),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.tensor(0.0, device=x.device),
                layer_idx=layer_idx,
                pattern_usage=pattern_weights.detach().mean(0),
                pattern_entropy=0.0,
                self_paradox_magnitude=torch.mean(torch.norm(hidden.detach(), dim=-1)).item(),
                composition_alpha=composition_alpha,
                entropy_magnitude=torch.mean(torch.norm(entropy, dim=-1)).item(),
                temporal_entropy_magnitude=temporal_entropy_magnitude
            )
            
            return None, penultimate_contribution, None, entropy

class ComplexTemporalEntropyRecyclingNet(nn.Module):
    """Complete complex temporal entropy recycling network with RoPE"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Store entropy from previous epoch for temporal recycling
        self.previous_entropy = None
        
        # Create complex temporal entropy recycling layers
        self.layers = nn.ModuleList()
        current_dim = embedding_dim
        
        prev_layer = None
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else hidden_dim
            
            layer = ComplexTemporalEntropyRecyclingLayer(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                next_dim=next_dim,
                penultimate_dim=hidden_dim,
                n_patterns=n_patterns,
                composition_from_prev=(i > 0),
                prev_layer=prev_layer,
                is_bottom=(i == 0)
            )
            self.layers.append(layer)
            prev_layer = layer
            current_dim = hidden_dim
        
        # Final output (real)
        self.final = nn.Linear(hidden_dims[-1] // 2, vocab_size)  # Convert from complex to real
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass with complex temporal entropy recycling + RoPE"""
        batch_size, seq_len = x.shape
        
        # Embedding + RoPE positional encoding
        embedded = self.embedding(x)  # [batch, seq, embed_dim]
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        # Apply RoPE and convert to complex
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_complex = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))
        
        # Process sequence (mean pooling for now)
        penultimate_contributions = []
        current = current_complex.mean(dim=1)  # [batch, complex_dim]
        all_errors = []
        all_entropy = []
        
        # Track temporal prediction error for Layer 0
        temporal_prediction_error = None
        
        # Single pass: process all layers with Layer 0 using previous epoch's entropy
        for i, layer in enumerate(self.layers):
            next_layer = self.layers[i+1] if i < len(self.layers)-1 else None
            
            if i == 0:
                # Layer 0: Use entropy from previous epoch (temporal recycling)
                current, penultimate, error, entropy = layer(current, next_layer, i, temporal_entropy=self.previous_entropy)
            else:
                current, penultimate, error, entropy = layer(current, next_layer, i)
            
            if error is not None:
                all_errors.append(error)
            penultimate_contributions.append(penultimate)
            
            # Collect entropy for NEXT epoch (except from layer 0)
            if i > 0:
                all_entropy.append(entropy)
        
        # Prepare entropy for next epoch AND compute temporal prediction error
        if all_entropy:
            # Sum all entropy for next epoch (complex space)
            total_entropy = torch.stack(all_entropy).sum(dim=0)
            
            # TEMPORAL INPUT PREDICTION: Compare Layer 0's prediction to actual entropy
            if self.previous_entropy is not None:
                layer_0 = self.layers[0]
                if hasattr(layer_0, 'temporal_entropy_predictor'):
                    # Layer 0 predicts what optimal entropy processing should look like
                    temporal_prediction = layer_0.temporal_entropy_predictor(current_complex.mean(dim=1))
                    # Prediction error: predicted optimal processing vs actual accumulated entropy
                    temporal_prediction_error = F.mse_loss(
                        torch.view_as_real(temporal_prediction).flatten(1),
                        torch.view_as_real(total_entropy).flatten(1)
                    )
            
            # Store for next epoch (detach to avoid gradient accumulation)
            self.previous_entropy = total_entropy.detach()
        else:
            self.previous_entropy = None
            temporal_prediction_error = None
        
        # Combine penultimate contributions (all are real now)
        penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
        
        # Final output
        output = self.final(penultimate)
        
        return output, torch.cat(all_errors, dim=1) if all_errors else None, temporal_prediction_error

# Factory function
def create_complex_temporal_entropy_recycling_net(sequence_length=20, hidden_dims=[64, 64, 64], n_patterns=8):
    """Create complex temporal entropy recycling version with RoPE"""
    return ComplexTemporalEntropyRecyclingNet(
        vocab_size=128,  # Will be set by experiment
        embedding_dim=64,
        hidden_dims=hidden_dims,
        n_patterns=n_patterns
    )

if __name__ == "__main__":
    print("🔮 TESTING COMPLEX TEMPORAL ENTROPY RECYCLING + ROPE 🔮")
    
    # Create network
    net = create_complex_temporal_entropy_recycling_net()
    
    # Test data
    x = torch.randint(0, 57, (5, 10))  # Token sequences
    
    # Forward pass
    output, errors, temporal_error = net(x)
    
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Errors: {errors.shape if errors is not None else None}")
    print(f"Temporal error: {temporal_error.item() if temporal_error is not None else None}")
    
    # Check entropy statistics
    print(f"\n=== COMPLEX TEMPORAL ENTROPY STATISTICS ===")
    for i, layer in enumerate(net.layers):
        if layer.last_stats:
            stats = layer.last_stats
            print(f"Layer {i}:")
            print(f"  Entropy magnitude: {stats.entropy_magnitude:.3f}")
            print(f"  Temporal entropy magnitude: {stats.temporal_entropy_magnitude:.3f}")
            print(f"  Composition alpha: {stats.composition_alpha:.3f}")
            print(f"  Paradox magnitude: {stats.self_paradox_magnitude:.3f}")
    
    print(f"\n✅ Complex temporal entropy recycling + RoPE working!")