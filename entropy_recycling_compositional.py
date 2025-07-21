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
    composition_alpha: float = 0.0
    entropy_magnitude: float = 0.0  # Track entropy flowing to Layer 0

class EntropyRecyclingLayer(nn.Module):
    """
    Layer with entropy recycling - patterns go up, entropy accumulates at bottom
    """
    
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8,
                 initial_temp=1.0, min_temp=0.1, temp_decay=0.99, 
                 composition_from_prev=True, prev_layer=None, is_bottom=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        self.composition_from_prev = composition_from_prev
        self.is_bottom = is_bottom  # Special bottom layer handling
        self.last_entropy = 0.0 
        
        # Temperature parameters
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        
        # Main processing pathway
        self.process = nn.Linear(input_dim, hidden_dim)
        self.self_predictor = nn.Linear(hidden_dim, hidden_dim)
        
        # COMPOSITIONAL PATTERN MECHANISM
        if composition_from_prev and prev_layer is not None:
            self.composition_weights = nn.Parameter(
                torch.randn(n_patterns, prev_layer.n_patterns) / (prev_layer.n_patterns ** 0.5)
            )
            self.prev_layer = prev_layer
            self.pattern_dict = None
        else:
            self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim) / hidden_dim**0.5)
            self.composition_weights = None
            self.prev_layer = None
        
        # Next layer prediction patterns
        self.next_pattern_dict = nn.Parameter(torch.randn(n_patterns, next_dim) / next_dim**0.5)
        self.next_pattern_attention = nn.Linear(hidden_dim, n_patterns)
        
        # Pattern attention
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        
        # Output pathway
        self.to_penultimate = nn.Linear(hidden_dim, penultimate_dim)
        
        # ENTROPY PROCESSING (special for bottom layer)
        if is_bottom:
            self.entropy_processor = nn.Linear(hidden_dim, hidden_dim)  # Process accumulated entropy
        
        # Stats tracking
        self.last_stats: Optional[LayerStats] = None
        self.last_entropy_magnitude: float = 0.0
    
    def get_pattern_dict(self):
        """Get pattern dictionary - hierarchical or base"""
        if self.composition_weights is not None and self.prev_layer is not None:
            prev_patterns = self.prev_layer.get_pattern_dict()
            composed_patterns = self.composition_weights @ prev_patterns
            return composed_patterns
        else:
            return self.pattern_dict
    
    def extract_patterns_and_entropy(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        KEY INNOVATION: Extract patterns, compute entropy as residual
        """
        # Get pattern dictionary and compute attention
        patterns = self.get_pattern_dict()
        attn = self.pattern_attention(hidden)
        pattern_weights = F.softmax(attn, dim=-1)
        
        # Pattern reconstruction (what patterns can explain)
        pattern_reconstruction = pattern_weights @ patterns
        
        # Match dimensions for entropy calculation
        if pattern_reconstruction.size(-1) != hidden.size(-1):
            # Project pattern reconstruction to hidden dimension
            if not hasattr(self, 'pattern_projector'):
                self.pattern_projector = nn.Linear(pattern_reconstruction.size(-1), hidden.size(-1)).to(hidden.device)
            pattern_reconstruction = self.pattern_projector(pattern_reconstruction)
        
        # Entropy = what patterns CANNOT explain
        entropy = hidden - pattern_reconstruction
        
        return pattern_reconstruction, entropy, pattern_weights
    
    def apply_self_paradox_nonlinearity(self, x: torch.Tensor, accumulated_entropy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply paradox mechanism, with entropy integration for bottom layer
        """
        # Process input first
        hidden_linear = self.process(x)
        
        if self.is_bottom and accumulated_entropy is not None:
            # Bottom layer: add processed entropy to the processed input (not raw input)
            processed_entropy = self.entropy_processor(accumulated_entropy)
            hidden_linear = hidden_linear + processed_entropy
        
        # Self-prediction paradox mechanism
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        paradox_magnitude = torch.norm(paradox, dim=-1, keepdim=True)
        
        # "I'm confused about myself → let more through"
        hidden = hidden_linear * torch.sigmoid(paradox_magnitude)
        return hidden
    
    def forward(self, x: torch.Tensor, next_layer: Optional['EntropyRecyclingLayer'], 
                layer_idx: int, accumulated_entropy: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass with entropy extraction
        Returns: (continue_up, penultimate_contribution, pred_error, entropy)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Apply paradox mechanism (with entropy for bottom layer)
        hidden = self.apply_self_paradox_nonlinearity(x, accumulated_entropy)
        
        # Extract patterns and entropy
        pattern_reconstruction, entropy, pattern_weights = self.extract_patterns_and_entropy(hidden)
        
        with torch.no_grad():
            self.last_entropy_magnitude = torch.mean(torch.norm(entropy, dim=-1)).item()
        
        if next_layer is not None:
            # Predict next layer behavior
            predicted_next = pattern_reconstruction  # Predict based on our pattern extraction
            
            # Get actual next layer behavior
            with torch.no_grad():
                actual_next = next_layer.apply_self_paradox_nonlinearity(hidden)
                actual_patterns, _, _ = next_layer.extract_patterns_and_entropy(actual_next)
            
            # Match dimensions
            min_dim = min(predicted_next.size(1), actual_patterns.size(1))
            predicted_next = predicted_next[:, :min_dim]
            actual_patterns = actual_patterns[:, :min_dim]
            
            # Prediction error
            pred_error = torch.mean((actual_patterns - predicted_next)**2, dim=1, keepdim=True)
            
            # "Don't tell me boring stuff" - route based on prediction accuracy
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Routing: confident patterns go UP, entropy stays for recycling
            penultimate_features = self.to_penultimate(pattern_reconstruction)
            penultimate_contribution = penultimate_features * confidence
            continue_up = hidden * (1 - confidence)
            
            # Track stats
            composition_alpha = 0.0
            if self.composition_weights is not None:
                with torch.no_grad():
                    comp_weights_norm = F.softmax(self.composition_weights, dim=-1)
                    composition_alpha = 1.0 - torch.mean(torch.max(comp_weights_norm, dim=-1)[0])
            
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.mean(torch.norm(continue_up.detach(), dim=1)),
                layer_idx=layer_idx,
                pattern_usage=pattern_weights.detach().mean(0),
                pattern_entropy=self.last_entropy,
                self_paradox_magnitude=torch.mean(torch.norm(hidden.detach(), dim=1)).item(),
                composition_alpha=composition_alpha,
                entropy_magnitude=self.last_entropy_magnitude
            )
            
            return continue_up, penultimate_contribution, pred_error, entropy
            
        else:
            # Last layer processing
            penultimate_contribution = self.to_penultimate(pattern_reconstruction)
            
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
                pattern_usage=pattern_weights.detach().mean(0),
                pattern_entropy=self.last_entropy,
                self_paradox_magnitude=torch.mean(torch.norm(hidden.detach(), dim=1)).item(),
                composition_alpha=composition_alpha,
                entropy_magnitude=self.last_entropy_magnitude
            )
            
            return None, penultimate_contribution, None, entropy

class EntropyRecyclingNet(nn.Module):
    """
    Network with entropy recycling - all entropy flows to Layer 0
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
            
            layer = EntropyRecyclingLayer(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                next_dim=next_dim,
                penultimate_dim=penultimate_dim,
                n_patterns=n_patterns,
                initial_temp=initial_temp,
                min_temp=min_temp,
                temp_decay=temp_decay,
                composition_from_prev=(i > 0),
                prev_layer=prev_layer,
                is_bottom=(i == 0)  # First layer is bottom
            )
            self.layers.append(layer)
            prev_layer = layer
            current_dim = hidden_dim
        
        self.final = nn.Linear(penultimate_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with entropy recycling to Layer 0
        """
        penultimate_contributions = []
        current = x
        all_errors = []
        all_entropy = []
        
        # First pass: collect entropy from all layers
        for i, layer in enumerate(self.layers):
            next_layer = self.layers[i+1] if i < len(self.layers)-1 else None
            
            if i == 0:
                # Layer 0: no accumulated entropy yet
                current, penultimate, error, entropy = layer(current, next_layer, i, accumulated_entropy=None)
            else:
                # Other layers: normal processing
                current, penultimate, error, entropy = layer(current, next_layer, i)
            
            if error is not None:
                all_errors.append(error)
            penultimate_contributions.append(penultimate)
            
            # Collect entropy for recycling (except from layer 0)
            if i > 0:
                all_entropy.append(entropy)
        
        # Second pass: Layer 0 processes accumulated entropy
        if all_entropy:
            # Project all entropy to Layer 0's dimension and sum
            layer_0_dim = self.layers[0].hidden_dim
            projected_entropy = []
            
            for i, entropy in enumerate(all_entropy):
                if entropy.size(-1) != layer_0_dim:
                    # Project to Layer 0's dimension
                    layer_idx = i + 1  # entropy from layer i+1
                    projector_name = f'entropy_projector_to_0_from_{layer_idx}'
                    if not hasattr(self, projector_name):
                        projector = nn.Linear(entropy.size(-1), layer_0_dim).to(entropy.device)
                        setattr(self, projector_name, projector)
                    else:
                        projector = getattr(self, projector_name)
                    entropy = projector(entropy)
                projected_entropy.append(entropy)
            
            total_entropy = torch.sum(torch.stack(projected_entropy), dim=0)
            
            # Re-process Layer 0 with accumulated entropy
            layer_0 = self.layers[0]
            next_layer_0 = self.layers[1] if len(self.layers) > 1 else None
            _, penultimate_0_enhanced, error_0_enhanced, _ = layer_0(
                x, next_layer_0, 0, accumulated_entropy=total_entropy
            )
            
            # Replace Layer 0's contribution with enhanced version
            penultimate_contributions[0] = penultimate_0_enhanced
            if error_0_enhanced is not None and len(all_errors) > 0:
                all_errors[0] = error_0_enhanced
        
        # Combine penultimate contributions
        penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
        
        # Final output
        output = self.final(penultimate)
        
        return output, torch.cat(all_errors, dim=1) if all_errors else None

# Factory function
def create_entropy_recycling_net(sequence_length=20, hidden_dims=[64, 64, 64], n_patterns=8):
    """Create entropy recycling version with consistent dimensions"""
    return EntropyRecyclingNet(
        input_dim=sequence_length,
        hidden_dims=hidden_dims,
        penultimate_dim=64,  # Match hidden dims
        output_dim=1,  # Will be set by experiment harness
        n_patterns=n_patterns
    )

if __name__ == "__main__":
    print("🔄 TESTING ENTROPY RECYCLING 🔄")
    
    # Create network
    net = create_entropy_recycling_net()
    
    # Test data
    x = torch.randn(50, 20)
    
    # Forward pass
    output, errors = net(x)
    
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Errors: {errors.shape if errors is not None else None}")
    
    # Check entropy statistics
    print(f"\n=== ENTROPY STATISTICS ===")
    for i, layer in enumerate(net.layers):
        if layer.last_stats:
            stats = layer.last_stats
            print(f"Layer {i}:")
            print(f"  Entropy magnitude: {stats.entropy_magnitude:.3f}")
            print(f"  Pattern entropy: {stats.pattern_entropy:.3f}")
            print(f"  Composition alpha: {stats.composition_alpha:.3f}")
    
    print(f"\n✅ Entropy recycling working!")