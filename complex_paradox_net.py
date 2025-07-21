import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
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
    composition_alpha: float = 0.0  # Track composition vs independence ratio

class CompositionalDiscretePatternLayer(nn.Module):
    """
    Enhanced DiscretePatternLayer with hierarchical compositional patterns.
    
    Key innovation: Layer N's patterns are built from Layer N-1's patterns.
    This creates a natural hierarchy: phonemes → morphemes → words → phrases
    """
    
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8,
                 initial_temp=1.0, min_temp=0.1, temp_decay=0.99, 
                 composition_from_prev=True, prev_layer=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        self.composition_from_prev = composition_from_prev
        self.last_entropy = 0.0 
        
        # Temperature parameters (unchanged)
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        
        # Main processing pathway (unchanged from stable ParadoxNet)
        self.process = nn.Linear(input_dim, hidden_dim)  # NO ReLU - pure linear
        self.self_predictor = nn.Linear(hidden_dim, hidden_dim)  # Paradox mechanism
        
        # COMPOSITIONAL PATTERN MECHANISM
        if composition_from_prev and prev_layer is not None:
            # Composition weights: how to build from previous layer's patterns
            self.composition_weights = nn.Parameter(
                torch.randn(n_patterns, prev_layer.n_patterns) / (prev_layer.n_patterns ** 0.5)
            )
            self.prev_layer = prev_layer
            # Pattern dict will be computed dynamically
            self.pattern_dict = None  # Property computed from previous layer
        else:
            # Base layer: independent patterns
            self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim) / hidden_dim**0.5)
            self.composition_weights = None
            self.prev_layer = None
        
        # Next layer prediction patterns (unchanged)
        self.next_pattern_dict = nn.Parameter(torch.randn(n_patterns, next_dim) / next_dim**0.5)
        self.next_pattern_attention = nn.Linear(hidden_dim, n_patterns)
        
        # Pattern attention (works with computed patterns)
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        
        # Output pathway (unchanged)
        self.to_penultimate = nn.Linear(hidden_dim, penultimate_dim)
        
        # Stats tracking
        self.last_stats: Optional[LayerStats] = None
        self.last_entropy: float = 0.0
        self.last_paradox_magnitude: float = 0.0
    
    def get_pattern_dict(self):
        """
        Get the current pattern dictionary - computed hierarchically or base patterns.
        """
        if self.composition_weights is not None and self.prev_layer is not None:
            # Hierarchical: patterns = weighted combination of previous layer's patterns
            prev_patterns = self.prev_layer.get_pattern_dict()
            composed_patterns = self.composition_weights @ prev_patterns
            return composed_patterns
        else:
            # Base layer: fixed independent patterns
            return self.pattern_dict
    
    def update_temperature(self):
        """Anneal temperature for more discrete selections"""
        self.current_temp = max(
            self.min_temp,
            self.current_temp * self.temp_decay
        )
    
    def compress_activity(self, x: torch.Tensor, is_next_layer: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress activity using compositional pattern dictionary"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if is_next_layer:
            attention = self.next_pattern_attention
            patterns = self.next_pattern_dict
        else:
            attention = self.pattern_attention
            patterns = self.get_pattern_dict()  # Use compositional patterns!
        
        # Compute attention weights
        attn = attention(x)
        pattern_weights = F.softmax(attn, dim=-1)
        
        # Calculate entropy
        with torch.no_grad():
            entropy = -torch.sum(pattern_weights * torch.log(pattern_weights + 1e-10), dim=-1)
            self.last_entropy = entropy.mean().item()
        
        # Compress using weighted combination of patterns
        compressed = pattern_weights @ patterns
        return compressed, pattern_weights
    
    def apply_self_paradox_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-prediction paradox as nonlinearity (unchanged from stable version)"""
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        paradox_magnitude = torch.norm(paradox, dim=-1, keepdim=True)
        
        with torch.no_grad():
            self.last_paradox_magnitude = torch.mean(paradox_magnitude).item()
        
        # The paradox creates the nonlinearity instead of ReLU
        hidden = hidden_linear * torch.sigmoid(paradox_magnitude)
        return hidden
    
    def forward(self, x: torch.Tensor, next_layer: Optional['CompositionalDiscretePatternLayer'], 
                layer_idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with compositional patterns"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Apply self-paradox nonlinearity (unchanged)
        hidden = self.apply_self_paradox_nonlinearity(x)
        
        if next_layer is not None:
            # Compress and predict next layer (now uses compositional patterns)
            my_compressed, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            predicted_next = my_compressed
            
            # Get actual next layer transformation
            with torch.no_grad():
                actual_next = next_layer.apply_self_paradox_nonlinearity(hidden)
                compressed_next, _ = next_layer.compress_activity(actual_next, is_next_layer=True)
            
            # Match dimensions
            min_dim = min(predicted_next.size(1), compressed_next.size(1))
            predicted_next = predicted_next[:, :min_dim]
            compressed_next = compressed_next[:, :min_dim]
            
            # Prediction error (unchanged)
            pred_error = torch.mean((compressed_next - predicted_next)**2, dim=1, keepdim=True)
            
            # Route based on prediction accuracy (unchanged)
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Add routing cost (unchanged)
            routing_balance = confidence * (1 - confidence)
            routing_cost = 0.1 * torch.mean(routing_balance)
            pred_error = pred_error + routing_cost
            
            # Route information (unchanged)
            penultimate_features = self.to_penultimate(hidden)
            penultimate_contribution = penultimate_features * confidence
            continue_up = hidden * (1 - confidence)
            
            # Track composition statistics
            composition_alpha = 0.0
            if self.composition_weights is not None:
                with torch.no_grad():
                    # Measure how much composition vs independence (diversity of weights)
                    comp_weights_norm = F.softmax(self.composition_weights, dim=-1)
                    composition_alpha = 1.0 - torch.mean(torch.max(comp_weights_norm, dim=-1)[0])
            
            # Enhanced statistics with composition info
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.mean(torch.norm(continue_up.detach(), dim=1)),
                layer_idx=layer_idx,
                pattern_usage=my_patterns.detach().mean(0),
                pattern_entropy=self.last_entropy,
                self_paradox_magnitude=self.last_paradox_magnitude,
                composition_alpha=composition_alpha
            )
            
            return continue_up, penultimate_contribution, pred_error
            
        else:
            # Last layer processing (unchanged)
            penultimate_contribution = self.to_penultimate(hidden)
            _, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            
            composition_alpha = 0.0
            if self.composition_weights is not None:
                with torch.no_grad():
                    comp_weights_norm = F.softmax(self.composition_weights, dim=-1)
                    composition_alpha = 1.0 - torch.mean(torch.max(comp_weights_norm, dim=-1)[0])
            
            self.last_stats = LayerStats(
                prediction_errors=torch.zeros(1, 1, device=x.device),
                confidence_values=torch.ones(1, 1, device=x.device),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.tensor(0.0, device=x.device),
                layer_idx=layer_idx,
                pattern_usage=my_patterns.detach().mean(0),
                pattern_entropy=self.last_entropy,
                self_paradox_magnitude=self.last_paradox_magnitude,
                composition_alpha=composition_alpha
            )
            
            return None, penultimate_contribution, None


class CompositionalDiscretePatternPredictiveNet(nn.Module):
    """
    Enhanced DiscretePatternPredictiveNet with hierarchical compositional patterns.
    
    Now Layer N's patterns are built from Layer N-1's patterns, creating true hierarchy!
    """
    
    def __init__(self, input_dim, hidden_dims, penultimate_dim, output_dim, n_patterns=8,
                 initial_temp=1.0, min_temp=0.1, temp_decay=0.99):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        # Create layers with hierarchical composition
        prev_layer = None
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else penultimate_dim
            
            layer = CompositionalDiscretePatternLayer(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                next_dim=next_dim,
                penultimate_dim=penultimate_dim,
                n_patterns=n_patterns,
                initial_temp=initial_temp,
                min_temp=min_temp,
                temp_decay=temp_decay,
                composition_from_prev=(i > 0),  # First layer is base, rest compose
                prev_layer=prev_layer
            )
            self.layers.append(layer)
            prev_layer = layer  # This layer becomes previous for next layer
            current_dim = hidden_dim
        
        self.final = nn.Linear(penultimate_dim, output_dim)
    
    def update_temperatures(self):
        """Update temperature for all layers"""
        for layer in self.layers:
            layer.update_temperature()
    
    def get_layer_stats(self) -> List[LayerStats]:
        """Get statistics including composition information"""
        return [layer.last_stats for layer in self.layers if layer.last_stats is not None]
    
    def get_composition_hierarchy(self) -> List[str]:
        """Get a description of the compositional hierarchy"""
        descriptions = []
        for i, layer in enumerate(self.layers):
            if layer.composition_weights is not None:
                descriptions.append(f"Layer {i}: Composed from Layer {i-1}")
            else:
                descriptions.append(f"Layer {i}: Base patterns")
        return descriptions
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with hierarchical compositional patterns"""
        penultimate_contributions = []
        current = x
        all_errors = []
        
        # Process through layers with composition
        for i, layer in enumerate(self.layers):
            next_layer = self.layers[i+1] if i < len(self.layers)-1 else None
            current, penultimate, error = layer(current, next_layer, i)
            
            if error is not None:
                all_errors.append(error)
            penultimate_contributions.append(penultimate)
        
        # Combine penultimate contributions
        penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
        
        # Final output
        output = self.final(penultimate)
        
        return output, torch.cat(all_errors, dim=1) if all_errors else None


# Factory function for testing
def create_compositional_paradox_net(sequence_length=20, hidden_dims=[64, 32, 16], n_patterns=8):
    """Create compositional version of stable ParadoxNet"""
    return CompositionalDiscretePatternPredictiveNet(
        input_dim=sequence_length,
        hidden_dims=hidden_dims,
        penultimate_dim=32,
        output_dim=1,  # Will be set by experiment harness
        n_patterns=n_patterns
    )


# Test compositional structure
def test_compositional_hierarchy():
    """Test that compositional hierarchy is working correctly"""
    
    print("🔗 TESTING COMPOSITIONAL HIERARCHY 🔗")
    
    # Create network
    net = create_compositional_paradox_net()
    
    # Generate test data
    x = torch.randn(50, 20)
    y = torch.randn(50, 1)
    
    # Forward pass
    output, errors = net(x)
    
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Errors: {errors.shape if errors is not None else None}")
    
    # Check compositional hierarchy
    print("\n=== COMPOSITIONAL HIERARCHY ===")
    hierarchy = net.get_composition_hierarchy()
    for desc in hierarchy:
        print(desc)
    
    # Check composition statistics
    print(f"\n=== COMPOSITION STATISTICS ===")
    stats = net.get_layer_stats()
    for i, stat in enumerate(stats):
        print(f"Layer {i}:")
        print(f"  Pattern entropy: {stat.pattern_entropy:.3f}")
        print(f"  Composition alpha: {stat.composition_alpha:.3f}")
        print(f"  Self-paradox magnitude: {stat.self_paradox_magnitude:.3f}")
    
    print(f"\n✅ Compositional hierarchy working!")
    return net

if __name__ == "__main__":
    test_compositional_hierarchy()