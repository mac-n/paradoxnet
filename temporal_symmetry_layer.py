import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TemporalSymmetryLayer(nn.Module):
    """Layer with temporal symmetry: past entropy (bottom) + future pattern prediction (top)"""
    
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8,
                 composition_from_prev=True, prev_layer=None, is_bottom=False, is_top=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_patterns = n_patterns
        self.is_bottom = is_bottom
        self.is_top = is_top
        
        # Core components
        self.process = nn.Linear(input_dim, hidden_dim)
        self.self_predictor = nn.Linear(hidden_dim, hidden_dim)
        
        # Pattern dictionary
        self.pattern_dict = nn.Parameter(
            torch.randn(n_patterns, hidden_dim) / hidden_dim**0.5
        )
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        self.to_penultimate = nn.Linear(hidden_dim, penultimate_dim)
        
        # BOTTOM LAYER: Temporal entropy recycling (existing)
        if is_bottom:
            self.temporal_entropy_processor = nn.Linear(hidden_dim, hidden_dim)
            self.temporal_entropy_predictor = nn.Linear(hidden_dim, hidden_dim)
            self.temporal_entropy_gate = nn.Linear(hidden_dim, 1)
        
        # TOP LAYER: Temporal pattern evolution prediction (NEW!)
        if is_top:
            self.temporal_pattern_predictor = nn.Linear(hidden_dim, n_patterns * hidden_dim)
            self.pattern_evolution_guide = nn.Linear(n_patterns * hidden_dim, hidden_dim)
        
        # Store temporal memories
        self.previous_future_prediction = None
        self.pattern_evolution_momentum = None
        
    def apply_temporal_acceleration(self, current_patterns: torch.Tensor) -> torch.Tensor:
        """Apply learning acceleration guidance from future predictions"""
        if self.is_top and self.pattern_evolution_momentum is not None:
            # Acceleration = current change + momentum from future prediction
            pattern_acceleration = 0.1 * self.pattern_evolution_momentum
            accelerated_patterns = current_patterns + pattern_acceleration
            return accelerated_patterns
        return current_patterns
    
    def predict_future_patterns(self, current_state: torch.Tensor) -> torch.Tensor:
        """Top layer: predict optimal future pattern organization"""
        if not self.is_top:
            return None
            
        # Predict future pattern evolution
        future_prediction = self.temporal_pattern_predictor(current_state)
        future_patterns = future_prediction.view(self.n_patterns, self.hidden_dim)
        
        # Generate evolution guidance
        evolution_guide = self.pattern_evolution_guide(future_prediction)
        
        # Store for next epoch's acceleration
        self.pattern_evolution_momentum = evolution_guide.detach()
        
        return future_patterns
    
    def apply_self_paradox_nonlinearity(self, x: torch.Tensor, temporal_entropy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Paradox mechanism with temporal acceleration"""
        hidden_linear = self.process(x)
        
        # BOTTOM: Apply temporal entropy (past failures)
        if self.is_bottom and temporal_entropy is not None:
            batch_size = hidden_linear.size(0)
            temp_batch_size = temporal_entropy.size(0)
            
            if temp_batch_size != batch_size:
                if temp_batch_size < batch_size:
                    repeats = (batch_size + temp_batch_size - 1) // temp_batch_size
                    temporal_entropy = temporal_entropy.repeat(repeats, 1)[:batch_size]
                else:
                    temporal_entropy = temporal_entropy[:batch_size]
            
            processed_temporal = self.temporal_entropy_processor(temporal_entropy)
            temporal_gate = torch.sigmoid(self.temporal_entropy_gate(hidden_linear))
            temporal_influence = processed_temporal * temporal_gate
            hidden_linear = hidden_linear + temporal_influence
        
        # Self-prediction (first derivative)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        paradox_magnitude = torch.norm(paradox, dim=-1, keepdim=True)
        
        # Apply paradox nonlinearity
        hidden = hidden_linear * torch.sigmoid(paradox_magnitude)
        
        # TOP: Apply temporal acceleration (second derivative!)
        if self.is_top:
            hidden = self.apply_temporal_acceleration(hidden)
        
        return hidden
    
    def forward(self, x: torch.Tensor, next_layer: Optional['TemporalSymmetryLayer'] = None, 
                layer_idx: int = 0, temporal_entropy: Optional[torch.Tensor] = None):
        """Forward with temporal symmetry"""
        
        # Apply paradox with temporal effects
        hidden = self.apply_self_paradox_nonlinearity(x, temporal_entropy)
        
        # Extract patterns
        attn = self.pattern_attention(hidden)
        pattern_weights = F.softmax(attn, dim=-1)
        pattern_reconstruction = pattern_weights @ self.pattern_dict
        entropy = hidden - pattern_reconstruction
        
        # TOP LAYER: Predict future pattern evolution
        future_prediction = None
        temporal_consistency_loss = None
        
        if self.is_top:
            future_prediction = self.predict_future_patterns(hidden.mean(dim=0))
            
            # Temporal consistency: current patterns should evolve toward previous predictions
            if self.previous_future_prediction is not None:
                temporal_consistency_loss = F.mse_loss(
                    self.pattern_dict, 
                    self.previous_future_prediction
                )
            
            # Store prediction for next epoch
            self.previous_future_prediction = future_prediction.detach()
        
        # Standard processing
        penultimate_contribution = self.to_penultimate(pattern_reconstruction)
        
        return hidden, penultimate_contribution, entropy, temporal_consistency_loss

# Test the temporal symmetry concept
if __name__ == "__main__":
    print("🌪️ TESTING TEMPORAL SYMMETRY LAYERS ⚡")
    
    # Create bottom layer (entropy recycling)
    bottom_layer = TemporalSymmetryLayer(
        input_dim=64, hidden_dim=64, next_dim=64, penultimate_dim=64,
        n_patterns=8, is_bottom=True
    )
    
    # Create top layer (pattern evolution prediction)
    top_layer = TemporalSymmetryLayer(
        input_dim=64, hidden_dim=64, next_dim=64, penultimate_dim=64,
        n_patterns=8, is_top=True
    )
    
    # Test data
    x = torch.randn(32, 64)
    temporal_entropy = torch.randn(32, 64)
    
    print("Testing bottom layer (past → present)...")
    bottom_out, bottom_pen, bottom_entropy, _ = bottom_layer(x, temporal_entropy=temporal_entropy)
    print(f"✅ Bottom layer output: {bottom_out.shape}")
    
    print("Testing top layer (present → future)...")
    top_out, top_pen, top_entropy, consistency_loss = top_layer(bottom_out)
    print(f"✅ Top layer output: {top_out.shape}")
    print(f"✅ Temporal consistency loss: {consistency_loss}")
    
    print("🎯 TEMPORAL DERIVATIVE HIERARCHY:")
    print("   Interlayer prediction (spatial) → Self prediction (1st derivative) → Temporal prediction (2nd derivative)")
    print("   Learning acceleration through temporal symmetry! 🚀⚡🔮")