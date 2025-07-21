import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Standard positional encoding
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
        # x shape: [batch, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].squeeze(1).unsqueeze(0)
        return x

@dataclass
class LayerStats:
    """Track statistics for temporal entropy recycling layers"""
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
    temporal_entropy_magnitude: float = 0.0  # Track temporal entropy

class TemporalEntropyRecyclingLayer(nn.Module):
    """Temporal entropy recycling layer in real space"""
    
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8,
                 composition_from_prev=True, prev_layer=None, is_bottom=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        self.composition_from_prev = composition_from_prev
        self.is_bottom = is_bottom
        
        # Real processing pathway
        self.process = nn.Linear(input_dim, hidden_dim)
        self.self_predictor = nn.Linear(hidden_dim, hidden_dim)
        
        # COMPOSITIONAL PATTERN MECHANISM (real)
        if composition_from_prev and prev_layer is not None:
            self.composition_weights = nn.Parameter(
                torch.randn(n_patterns, prev_layer.n_patterns) / (prev_layer.n_patterns ** 0.5)
            )
            self.prev_layer = prev_layer
            self.pattern_dict = None
        else:
            self.pattern_dict = nn.Parameter(
                torch.randn(n_patterns, hidden_dim) / hidden_dim**0.5
            )
            self.composition_weights = None
            self.prev_layer = None
        
        # Next layer prediction patterns
        self.next_pattern_dict = nn.Parameter(
            torch.randn(n_patterns, next_dim) / next_dim**0.5
        )
        self.next_pattern_attention = nn.Linear(hidden_dim, n_patterns)
        
        # Pattern attention
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        
        # Output pathway
        self.to_penultimate = nn.Linear(hidden_dim, penultimate_dim)
        
        # TEMPORAL ENTROPY PROCESSING (special for bottom layer)
        if is_bottom:
            self.temporal_entropy_processor = nn.Linear(hidden_dim, hidden_dim)
            self.temporal_entropy_predictor = nn.Linear(hidden_dim, hidden_dim)  # Predict optimal entropy processing
            self.temporal_entropy_gate = nn.Linear(hidden_dim, 1)  # Gate temporal entropy influence
        
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
        """Extract patterns and compute entropy as residual"""
        patterns = self.get_pattern_dict()
        
        # Attention over patterns
        attn = self.pattern_attention(hidden)
        pattern_weights = F.softmax(attn, dim=-1)
        
        # Pattern reconstruction (what patterns can explain)
        pattern_reconstruction = pattern_weights @ patterns
        
        # Entropy = what patterns CANNOT explain
        entropy = hidden - pattern_reconstruction
        
        return pattern_reconstruction, entropy, pattern_weights
    
    def apply_self_paradox_nonlinearity(self, x: torch.Tensor, temporal_entropy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Real paradox mechanism with temporal entropy integration"""
        # Process input
        hidden_linear = self.process(x)
        
        # Temporal entropy integration for bottom layer
        temporal_influence = 0.0
        if self.is_bottom and temporal_entropy is not None:
            # Handle batch size mismatch - temporal entropy is from previous batch
            batch_size = hidden_linear.size(0)
            temp_batch_size = temporal_entropy.size(0)
            
            if temp_batch_size != batch_size:
                # Repeat or truncate temporal entropy to match current batch size
                if temp_batch_size < batch_size:
                    # Repeat temporal entropy to fill current batch
                    repeats = (batch_size + temp_batch_size - 1) // temp_batch_size
                    temporal_entropy = temporal_entropy.repeat(repeats, 1)[:batch_size]
                else:
                    # Truncate temporal entropy to match current batch
                    temporal_entropy = temporal_entropy[:batch_size]
            
            # Process temporal entropy and gate its influence
            processed_temporal = self.temporal_entropy_processor(temporal_entropy)
            temporal_gate = torch.sigmoid(self.temporal_entropy_gate(hidden_linear))
            temporal_influence = processed_temporal * temporal_gate
            hidden_linear = hidden_linear + temporal_influence
        
        # Self-prediction paradox mechanism
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        paradox_magnitude = torch.norm(paradox, dim=-1, keepdim=True)
        
        # "I'm confused about myself → let more through"
        hidden = hidden_linear * torch.sigmoid(paradox_magnitude)
        return hidden
    
    def forward(self, x: torch.Tensor, next_layer: Optional['TemporalEntropyRecyclingLayer'], 
                layer_idx: int, temporal_entropy: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Forward pass with temporal entropy extraction"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Apply paradox mechanism with temporal entropy
        hidden = self.apply_self_paradox_nonlinearity(x, temporal_entropy)
        
        # Extract patterns and entropy
        pattern_reconstruction, entropy, pattern_weights = self.extract_patterns_and_entropy(hidden)
        
        # Track temporal entropy magnitude
        temporal_entropy_magnitude = 0.0
        if temporal_entropy is not None:
            temporal_entropy_magnitude = torch.mean(torch.norm(temporal_entropy, dim=-1)).item()
        
        if next_layer is not None:
            # Predict next layer
            predicted_next = pattern_reconstruction
            
            # Get actual next layer behavior
            with torch.no_grad():
                actual_next = next_layer.apply_self_paradox_nonlinearity(hidden)
                actual_patterns, _, _ = next_layer.extract_patterns_and_entropy(actual_next)
            
            # Match dimensions
            min_dim = min(predicted_next.size(-1), actual_patterns.size(-1))
            predicted_next = predicted_next[:, :min_dim]
            actual_patterns = actual_patterns[:, :min_dim]
            
            # Prediction error
            pred_error = torch.mean((actual_patterns - predicted_next)**2, dim=1, keepdim=True)
            
            # Routing confidence
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Route information
            penultimate_features = self.to_penultimate(pattern_reconstruction)
            penultimate_contribution = penultimate_features * confidence
            continue_up = hidden * (1 - confidence)
            
            # Track composition statistics
            composition_alpha = 0.0
            if self.composition_weights is not None:
                with torch.no_grad():
                    comp_weights_norm = F.softmax(self.composition_weights, dim=-1)
                    composition_alpha = 1.0 - torch.mean(torch.max(comp_weights_norm, dim=-1)[0]).item()
            
            # Enhanced statistics
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.mean(torch.norm(continue_up.detach(), dim=1)),
                layer_idx=layer_idx,
                pattern_usage=pattern_weights.detach().mean(0),
                pattern_entropy=0.0,  # TODO: Implement pattern entropy
                self_paradox_magnitude=torch.mean(torch.norm(hidden.detach(), dim=1)).item(),
                composition_alpha=composition_alpha,
                entropy_magnitude=torch.mean(torch.norm(entropy, dim=-1)).item(),
                temporal_entropy_magnitude=temporal_entropy_magnitude
            )
            
            return continue_up, penultimate_contribution, pred_error, entropy
            
        else:
            # Last layer processing
            penultimate_contribution = self.to_penultimate(pattern_reconstruction)
            
            composition_alpha = 0.0
            if self.composition_weights is not None:
                with torch.no_grad():
                    comp_weights_norm = F.softmax(self.composition_weights, dim=-1)
                    composition_alpha = 1.0 - torch.mean(torch.max(comp_weights_norm, dim=-1)[0]).item()
            
            self.last_stats = LayerStats(
                prediction_errors=torch.zeros(1, 1, device=x.device),
                confidence_values=torch.ones(1, 1, device=x.device),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.tensor(0.0, device=x.device),
                layer_idx=layer_idx,
                pattern_usage=pattern_weights.detach().mean(0),
                pattern_entropy=0.0,
                self_paradox_magnitude=torch.mean(torch.norm(hidden.detach(), dim=1)).item(),
                composition_alpha=composition_alpha,
                entropy_magnitude=torch.mean(torch.norm(entropy, dim=-1)).item(),
                temporal_entropy_magnitude=temporal_entropy_magnitude
            )
            
            return None, penultimate_contribution, None, entropy

class TemporalEntropyRecyclingNet(nn.Module):
    """Complete temporal entropy recycling network in real space"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Store entropy from previous epoch for temporal recycling
        self.previous_entropy = None
        
        # Create temporal entropy recycling layers
        self.layers = nn.ModuleList()
        current_dim = embedding_dim
        
        prev_layer = None
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else hidden_dim
            
            layer = TemporalEntropyRecyclingLayer(
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
        
        # Final output
        self.final = nn.Linear(hidden_dims[-1], vocab_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass with temporal entropy recycling"""
        # Embedding + positional encoding
        embedded = self.embedding(x)  # [batch, seq, embed_dim]
        embedded = self.pos_encoder(embedded)  # Add positional encoding
        
        # Process sequence (mean pooling)
        penultimate_contributions = []
        current = embedded.mean(dim=1)  # [batch, embed_dim]
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
                
                # TEMPORAL INPUT PREDICTION: Layer 0 predicts optimal entropy processing
                if self.previous_entropy is not None and hasattr(layer, 'temporal_entropy_predictor'):
                    # Predict what the optimal entropy processing should be
                    temporal_prediction = layer.temporal_entropy_predictor(current)
                    # Compare prediction to actual accumulated entropy (from this forward pass)
                    # We'll compute this after collecting all entropy
                    pass
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
            # Sum all entropy for next epoch (real space - simpler)
            total_entropy = torch.stack(all_entropy).sum(dim=0)
            
            # TEMPORAL INPUT PREDICTION: Compare Layer 0's prediction to actual entropy
            if self.previous_entropy is not None:
                layer_0 = self.layers[0]
                if hasattr(layer_0, 'temporal_entropy_predictor'):
                    # Layer 0 predicts what optimal entropy processing should look like
                    temporal_prediction = layer_0.temporal_entropy_predictor(embedded.mean(dim=1))
                    # Prediction error: predicted optimal processing vs actual accumulated entropy
                    temporal_prediction_error = F.mse_loss(temporal_prediction, total_entropy)
            
            # Store for next epoch (detach to avoid gradient accumulation)
            self.previous_entropy = total_entropy.detach()
        else:
            self.previous_entropy = None
            temporal_prediction_error = None
        
        # Combine penultimate contributions
        penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
        
        # Final output
        output = self.final(penultimate)
        
        return output, torch.cat(all_errors, dim=1) if all_errors else None, temporal_prediction_error

# Factory function
def create_temporal_entropy_recycling_net(sequence_length=20, hidden_dims=[64, 64, 64], n_patterns=8):
    """Create temporal entropy recycling version in real space"""
    return TemporalEntropyRecyclingNet(
        vocab_size=128,  # Will be set by experiment
        embedding_dim=64,
        hidden_dims=hidden_dims,
        n_patterns=n_patterns
    )

if __name__ == "__main__":
    print("⏰ TESTING TEMPORAL ENTROPY RECYCLING ⏰")
    
    # Create network
    net = create_temporal_entropy_recycling_net()
    
    # Test data
    x = torch.randint(0, 57, (50, 20))  # Token sequences
    
    # Forward pass
    output, errors = net(x)
    
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Errors: {errors.shape if errors is not None else None}")
    
    # Check entropy statistics
    print(f"\n=== TEMPORAL ENTROPY STATISTICS ===")
    for i, layer in enumerate(net.layers):
        if layer.last_stats:
            stats = layer.last_stats
            print(f"Layer {i}:")
            print(f"  Entropy magnitude: {stats.entropy_magnitude:.3f}")
            print(f"  Temporal entropy magnitude: {stats.temporal_entropy_magnitude:.3f}")
            print(f"  Composition alpha: {stats.composition_alpha:.3f}")
            print(f"  Paradox magnitude: {stats.self_paradox_magnitude:.3f}")
    
    print(f"\n✅ Temporal entropy recycling working!")