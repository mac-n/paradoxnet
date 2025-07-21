import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# RoPE implementation
def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embedding to complex tensors."""
    # x is already complex, freqs_cis is complex
    if x.dim() == 3:  # [batch, seq, features]
        # Expand freqs_cis to match batch dimension
        freqs_cis = freqs_cis.unsqueeze(0).expand(x.size(0), -1, -1)
    x_rotated = x * freqs_cis
    return x_rotated

class RoPEPositionalEncoding(nn.Module):
    """Generates rotary positional embeddings for complex space."""
    def __init__(self, complex_dim: int, max_len: int = 5000):
        super().__init__()
        # For complex_dim complex numbers, we need complex_dim frequency components
        theta = 1.0 / (10000.0 ** (torch.arange(0, complex_dim).float() / complex_dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.freqs_cis[:seq_len, :]

class ComplexLinear(nn.Module):
    """Linear layer for complex-valued tensors."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Complex weights
        self.weight = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.cfloat) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be [batch, features] or [batch, seq, features] - both complex
        if x.dim() == 3:  # sequence dimension
            return torch.einsum('bsi,io->bso', x, self.weight)
        else:
            return x @ self.weight

@dataclass
class LayerStats:
    """Track statistics for entropy recycling complex layers"""
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
    phase_diversity: float = 0.0  # Track phase relationships

class EntropyRecyclingComplexLayer(nn.Module):
    """Entropy recycling layer in complex space with RoPE"""
    
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8,
                 composition_from_prev=True, prev_layer=None, is_bottom=False):
        super().__init__()
        
        self.input_dim = input_dim  # Complex dimension
        self.hidden_dim = hidden_dim  # Complex dimension
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        self.composition_from_prev = composition_from_prev
        self.is_bottom = is_bottom
        
        # Complex processing pathway
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # COMPOSITIONAL PATTERN MECHANISM (in complex space)
        if composition_from_prev and prev_layer is not None:
            self.composition_weights = nn.Parameter(
                torch.randn(n_patterns, prev_layer.n_patterns, dtype=torch.cfloat) / (prev_layer.n_patterns ** 0.5)
            )
            self.prev_layer = prev_layer
            self.pattern_dict = None
        else:
            self.pattern_dict = nn.Parameter(
                torch.randn(n_patterns, hidden_dim, dtype=torch.cfloat) / hidden_dim**0.5
            )
            self.composition_weights = None
            self.prev_layer = None
        
        # Next layer prediction patterns (complex)
        self.next_pattern_dict = nn.Parameter(
            torch.randn(n_patterns, next_dim, dtype=torch.cfloat) / next_dim**0.5
        )
        self.next_pattern_attention = ComplexLinear(hidden_dim, n_patterns)
        
        # Pattern attention (complex -> real for softmax)
        self.pattern_attention = ComplexLinear(hidden_dim, n_patterns)
        
        # Output pathway
        self.to_penultimate = ComplexLinear(hidden_dim, penultimate_dim)
        
        # ENTROPY PROCESSING (special for bottom layer)
        if is_bottom:
            self.entropy_processor = ComplexLinear(hidden_dim, hidden_dim)
        
        # Stats tracking
        self.last_stats: Optional[LayerStats] = None
    
    def get_pattern_dict(self):
        """Get compositional pattern dictionary"""
        if self.composition_weights is not None and self.prev_layer is not None:
            prev_patterns = self.prev_layer.get_pattern_dict()
            composed_patterns = self.composition_weights @ prev_patterns
            return composed_patterns
        else:
            return self.pattern_dict
    
    def extract_patterns_and_entropy(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract patterns and compute entropy in complex space"""
        patterns = self.get_pattern_dict()
        
        # Complex attention -> real for softmax
        attn_complex = self.pattern_attention(hidden)
        attn_real = attn_complex.abs()  # Use magnitude for attention
        pattern_weights = F.softmax(attn_real, dim=-1)
        
        # Pattern reconstruction (complex)
        pattern_reconstruction = pattern_weights.unsqueeze(-1) * patterns.unsqueeze(0)
        pattern_reconstruction = pattern_reconstruction.sum(dim=-2)  # Sum over patterns
        
        # Entropy = what patterns CANNOT explain (complex residual)
        entropy = hidden - pattern_reconstruction
        
        return pattern_reconstruction, entropy, pattern_weights
    
    def apply_self_paradox_nonlinearity(self, x: torch.Tensor, accumulated_entropy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Complex paradox mechanism with entropy integration"""
        # Process input
        hidden_linear = self.process(x)
        
        if self.is_bottom and accumulated_entropy is not None:
            # Bottom layer: add processed entropy
            processed_entropy = self.entropy_processor(accumulated_entropy)
            hidden_linear = hidden_linear + processed_entropy
        
        # Self-prediction paradox (complex)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        paradox_magnitude = paradox.abs()
        
        # "I'm confused about myself → let more through" (complex gating)
        hidden = hidden_linear * torch.sigmoid(paradox_magnitude)
        return hidden
    
    def forward(self, x: torch.Tensor, next_layer: Optional['EntropyRecyclingComplexLayer'], 
                layer_idx: int, accumulated_entropy: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Forward pass with complex entropy extraction"""
        if x.dim() == 2:  # Add sequence dimension if missing
            x = x.unsqueeze(1)
        
        # Apply complex paradox mechanism
        hidden = self.apply_self_paradox_nonlinearity(x, accumulated_entropy)
        
        # Extract patterns and entropy (complex)
        pattern_reconstruction, entropy, pattern_weights = self.extract_patterns_and_entropy(hidden)
        
        # Track phase diversity
        with torch.no_grad():
            phase_angles = torch.angle(hidden)
            phase_diversity = torch.std(phase_angles).item() if hidden.numel() > 0 else 0.0
        
        if next_layer is not None:
            # Predict next layer (complex)
            predicted_next = pattern_reconstruction
            
            # Get actual next layer behavior
            with torch.no_grad():
                actual_next = next_layer.apply_self_paradox_nonlinearity(hidden)
                actual_patterns, _, _ = next_layer.extract_patterns_and_entropy(actual_next)
            
            # Match dimensions for prediction error
            min_dim = min(predicted_next.size(-1), actual_patterns.size(-1))
            predicted_next = predicted_next[..., :min_dim]
            actual_patterns = actual_patterns[..., :min_dim]
            
            # Complex prediction error
            pred_error = torch.mean((actual_patterns - predicted_next).abs()**2, dim=-1, keepdim=True)
            
            # Routing confidence (same logic, but with complex magnitudes)
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Route information (complex)
            penultimate_features = self.to_penultimate(pattern_reconstruction)
            penultimate_contribution = penultimate_features * confidence.unsqueeze(-1)
            continue_up = hidden * (1 - confidence).unsqueeze(-1)
            
            # Track composition statistics
            composition_alpha = 0.0
            if self.composition_weights is not None:
                with torch.no_grad():
                    comp_weights_norm = F.softmax(self.composition_weights.abs(), dim=-1)
                    composition_alpha = 1.0 - torch.mean(torch.max(comp_weights_norm, dim=-1)[0]).item()
            
            # Enhanced statistics
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(penultimate_contribution.abs()),
                continue_magnitude=torch.mean(continue_up.abs()),
                layer_idx=layer_idx,
                pattern_usage=pattern_weights.detach().mean(0),
                pattern_entropy=0.0,  # TODO: Implement complex entropy
                self_paradox_magnitude=torch.mean(hidden.abs()).item(),
                composition_alpha=composition_alpha,
                entropy_magnitude=torch.mean(entropy.abs()).item(),
                phase_diversity=phase_diversity
            )
            
            return continue_up, penultimate_contribution, pred_error, entropy
            
        else:
            # Last layer processing
            penultimate_contribution = self.to_penultimate(pattern_reconstruction)
            
            composition_alpha = 0.0
            if self.composition_weights is not None:
                with torch.no_grad():
                    comp_weights_norm = F.softmax(self.composition_weights.abs(), dim=-1)
                    composition_alpha = 1.0 - torch.mean(torch.max(comp_weights_norm, dim=-1)[0]).item()
            
            self.last_stats = LayerStats(
                prediction_errors=torch.zeros(1, 1, device=x.device),
                confidence_values=torch.ones(1, 1, device=x.device),
                penultimate_magnitude=torch.mean(penultimate_contribution.abs()),
                continue_magnitude=torch.tensor(0.0, device=x.device),
                layer_idx=layer_idx,
                pattern_usage=pattern_weights.detach().mean(0),
                pattern_entropy=0.0,
                self_paradox_magnitude=torch.mean(hidden.abs()).item(),
                composition_alpha=composition_alpha,
                entropy_magnitude=torch.mean(entropy.abs()).item(),
                phase_diversity=phase_diversity
            )
            
            return None, penultimate_contribution, None, entropy

class EntropyRecyclingComplexNet(nn.Module):
    """Complete entropy recycling network in complex space with RoPE"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.complex_dim = embedding_dim // 2  # Half the embedding dim for complex
        self.rope_encoder = RoPEPositionalEncoding(self.complex_dim)  # Match complex dimension
        
        # Store entropy from previous epoch for temporal recycling
        self.previous_entropy = None
        
        # Convert real embeddings to complex
        self.embed_to_complex = nn.Linear(embedding_dim, self.complex_dim * 2)  # Real -> Complex preparation
        
        # Create entropy recycling layers (complex)
        self.layers = nn.ModuleList()
        current_dim = embedding_dim // 2  # Complex dimension
        
        prev_layer = None
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else hidden_dim
            
            layer = EntropyRecyclingComplexLayer(
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
        
        # Final output (complex -> real)
        self.final = nn.Linear(hidden_dims[-1] * 2, vocab_size)  # *2 for complex->real
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with complex entropy recycling + RoPE"""
        # Embedding
        embedded = self.embedding(x)  # [batch, seq, embed_dim]
        batch_size, seq_len, embed_dim = embedded.shape
        
        # Prepare for complex space
        embedded_prep = self.embed_to_complex(embedded)
        embedded_complex = torch.view_as_complex(
            embedded_prep.reshape(batch_size, seq_len, self.complex_dim, 2)
        )
        
        # Apply RoPE
        freqs_cis = self.rope_encoder(seq_len)
        embedded_with_rope = apply_rotary_pos_emb(embedded_complex, freqs_cis)
        
        # Process sequence (mean pooling to single vector per batch)
        penultimate_contributions = []
        current = embedded_with_rope.mean(dim=1)  # [batch, complex_dim]
        all_errors = []
        all_entropy = []
        
        # Single pass: process all layers with Layer 0 using previous epoch's entropy
        for i, layer in enumerate(self.layers):
            next_layer = self.layers[i+1] if i < len(self.layers)-1 else None
            
            if i == 0:
                # Layer 0: Use entropy from previous epoch (temporal recycling)
                current, penultimate, error, entropy = layer(current, next_layer, i, accumulated_entropy=self.previous_entropy)
            else:
                current, penultimate, error, entropy = layer(current, next_layer, i)
            
            if error is not None:
                all_errors.append(error)
            penultimate_contributions.append(penultimate)
            
            # Collect entropy for NEXT epoch (except from layer 0)
            if i > 0:
                all_entropy.append(entropy)
        
        # Prepare entropy for next epoch
        if all_entropy:
            # Flatten and sum all entropy for next epoch
            flattened_entropy = []
            target_shape = None
            
            for i, entropy in enumerate(all_entropy):
                # Flatten to [batch, features] consistently
                while entropy.dim() > 2:
                    entropy = entropy.mean(dim=1)  # Average out extra dimensions
                
                if target_shape is None:
                    target_shape = entropy.shape
                
                # Ensure all have same shape
                if entropy.shape != target_shape:
                    # Pad or truncate to match
                    if entropy.size(-1) != target_shape[-1]:
                        min_feat = min(entropy.size(-1), target_shape[-1])
                        entropy = entropy[..., :min_feat]
                        if entropy.size(-1) < target_shape[-1]:
                            padding = torch.zeros(*entropy.shape[:-1], target_shape[-1] - entropy.size(-1), 
                                                dtype=entropy.dtype, device=entropy.device)
                            entropy = torch.cat([entropy, padding], dim=-1)
                
                flattened_entropy.append(entropy)
            
            # Store for next epoch (detach to avoid gradient accumulation)
            self.previous_entropy = torch.stack(flattened_entropy).sum(dim=0).detach()
        else:
            # No entropy to recycle
            self.previous_entropy = None
        
        # Combine penultimate contributions (complex) - handle different shapes
        flattened_penultimate = []
        for i, contrib in enumerate(penultimate_contributions):
            # Flatten to [batch, features] consistently
            while contrib.dim() > 2:
                contrib = contrib.mean(dim=1)  # Average out extra dimensions
            flattened_penultimate.append(contrib)
        
        penultimate_complex = torch.stack(flattened_penultimate).sum(dim=0)
        
        # Convert complex to real for final output
        penultimate_real = torch.cat([
            penultimate_complex.real, 
            penultimate_complex.imag
        ], dim=-1)
        
        # Final prediction
        output = self.final(penultimate_real)
        
        return output

# Factory function
def create_entropy_recycling_complex_net(sequence_length=20, hidden_dims=[32, 32, 32], n_patterns=8):
    """Create complex entropy recycling version with RoPE"""
    return EntropyRecyclingComplexNet(
        vocab_size=128,  # Will be set by experiment
        embedding_dim=64,  # Will create 32 complex dimensions
        hidden_dims=hidden_dims,  # Complex dimensions
        n_patterns=n_patterns
    )

if __name__ == "__main__":
    print("🌀 TESTING COMPLEX ENTROPY RECYCLING 🌀")
    
    # Create network
    net = create_entropy_recycling_complex_net()
    
    # Test data
    x = torch.randint(0, 57, (50, 20))  # Token sequences
    
    # Forward pass
    output = net(x)
    
    print(f"Input: {x.shape}, Output: {output.shape}")
    
    # Check entropy statistics
    print(f"\n=== COMPLEX ENTROPY STATISTICS ===")
    for i, layer in enumerate(net.layers):
        if layer.last_stats:
            stats = layer.last_stats
            print(f"Layer {i}:")
            print(f"  Entropy magnitude: {stats.entropy_magnitude:.3f}")
            print(f"  Phase diversity: {stats.phase_diversity:.3f}")
            print(f"  Composition alpha: {stats.composition_alpha:.3f}")
            print(f"  Paradox magnitude: {stats.self_paradox_magnitude:.3f}")
    
    print(f"\n✅ Complex entropy recycling working!")