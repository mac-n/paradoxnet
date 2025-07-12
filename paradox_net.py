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
    pattern_attention_entropy: float = 0.0  # New field for pattern coordination
    adaptive_temperature: float = 0.0  # New field for self-modulated temperature
    
class DiscretePatternLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8,
                 initial_temp=1.0, min_temp=0.1, temp_decay=0.99, temp_lr=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        self.last_entropy = 0.0 
        self.temp_lr = temp_lr  # Learning rate for temperature updates
        
        # Temperature parameters
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        
        # Main processing pathway - WITHOUT ReLU!
        self.process = nn.Linear(input_dim, hidden_dim)
        self.self_predictor = nn.Linear(hidden_dim, hidden_dim)  # Single predictor for both uses
        
        # Pattern dictionaries (unchanged)
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim) / hidden_dim**0.5)
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        self.next_pattern_dict = nn.Parameter(torch.randn(n_patterns, next_dim) / next_dim**0.5)
        self.next_pattern_attention = nn.Linear(hidden_dim, n_patterns)
        
        # NEW: Self-attention between patterns for coordination
        self.pattern_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,  # Keep simple for transparency
            batch_first=True,
            dropout=0.0
        )
        
        # Output pathway (unchanged)
        self.to_penultimate = nn.Linear(hidden_dim, penultimate_dim)
        
        # Stats tracking
        self.last_stats: Optional[LayerStats] = None
        self.last_entropy: float = 0.0
        self.last_paradox_magnitude: float = 0.0
        self.last_pattern_attention_entropy: float = 0.0
    
    def update_temperature(self):
        """Anneal base temperature"""
        self.current_temp = max(
            self.min_temp,
            self.current_temp * self.temp_decay
        )
    
    def compress_activity(self, x: torch.Tensor, is_next_layer: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress activity using pattern dictionary"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Choose appropriate attention and patterns
        if is_next_layer:
            attention = self.next_pattern_attention
            patterns = self.next_pattern_dict
        else:
            attention = self.pattern_attention
            patterns = self.pattern_dict
        
        # Compute attention weights with adaptive temperature
        attn = attention(x)
        pattern_weights = F.softmax(attn / self.current_temp, dim=-1)
        
        # Calculate entropy
        with torch.no_grad():
            entropy = -torch.sum(pattern_weights * torch.log(pattern_weights + 1e-10), dim=-1)
            self.last_entropy = entropy.mean().item()
        
        # Compress using weighted combination of patterns
        compressed = pattern_weights @ patterns
        return compressed, pattern_weights
    
    def apply_unified_self_processing(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Apply pattern coordination + self-paradox nonlinearity + temperature modulation"""
        # Pure linear transformation first
        hidden_linear = self.process(x)
        
        # Get pattern attention weights
        attn = self.pattern_attention(hidden_linear)
        pattern_weights = F.softmax(attn / self.current_temp, dim=-1)
        
        # Calculate entropy for pattern selection
        with torch.no_grad():
            entropy = -torch.sum(pattern_weights * torch.log(pattern_weights + 1e-10), dim=-1)
            self.last_entropy = entropy.mean().item()
        
        # Self-attention between the actual pattern vectors
        # Expand patterns to batch size: [batch, n_patterns, hidden_dim]
        patterns = self.pattern_dict.unsqueeze(0).expand(hidden_linear.size(0), -1, -1)
        
        # Let patterns coordinate with each other
        if patterns.size(0) > 0:
            attended_patterns, attention_weights = self.pattern_self_attention(
                patterns,  # Q: Each pattern asks what to focus on
                patterns,  # K: Each pattern provides its information  
                patterns   # V: Each pattern contributes its updated state
            )
            
            # Track pattern coordination entropy
            with torch.no_grad():
                # attention_weights is [batch, n_patterns, n_patterns]
                attn_flat = attention_weights.view(-1, attention_weights.size(-1))
                attn_entropy = -torch.sum(attn_flat * torch.log(attn_flat + 1e-10), dim=-1)
                self.last_pattern_attention_entropy = attn_entropy.mean().item()
        else:
            attended_patterns = patterns
            self.last_pattern_attention_entropy = 0.0
        
        # Combine attended patterns using pattern weights
        # attended_patterns: [batch, n_patterns, hidden_dim]
        # pattern_weights: [batch, n_patterns]
        coordinated_contribution = (pattern_weights.unsqueeze(-1) * attended_patterns).sum(dim=1)
        hidden_enhanced = hidden_linear + coordinated_contribution
        
        # UNIFIED self-prediction for both nonlinearity AND temperature
        self_prediction = self.self_predictor(hidden_enhanced)
        paradox = self_prediction - hidden_enhanced
        
        # Track paradox magnitude
        with torch.no_grad():
            self.last_paradox_magnitude = torch.mean(torch.norm(paradox, dim=-1)).item()
        
        # Use paradox as nonlinearity
        hidden = hidden_enhanced * torch.sigmoid(paradox)
        
        # Use paradox magnitude to modulate temperature
        self_pred_accuracy = torch.mean(paradox**2).item()
        adaptive_temp = self.current_temp * (1 + self.temp_lr * self_pred_accuracy)
        adaptive_temp = max(self.min_temp, adaptive_temp)  # Don't go below minimum
        
        return hidden, adaptive_temp
    
    def forward(self, x: torch.Tensor, next_layer: Optional['DiscretePatternLayer'], 
                layer_idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        # Ensure x is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Apply unified self-processing (coordination + nonlinearity + temperature)
        hidden, adaptive_temp = self.apply_unified_self_processing(x)
        
        if next_layer is not None:
            # Temporarily update temperature for this forward pass
            original_temp = self.current_temp
            self.current_temp = adaptive_temp
            
            # Compress and predict next layer
            my_compressed, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            predicted_next = my_compressed
            
            # Get actual next layer transformation
            with torch.no_grad():
                actual_next, _ = next_layer.apply_unified_self_processing(hidden)
                compressed_next, _ = next_layer.compress_activity(actual_next, is_next_layer=True)
            
            # Restore original temperature
            self.current_temp = original_temp
            
            # Match dimensions
            min_dim = min(predicted_next.size(1), compressed_next.size(1))
            predicted_next = predicted_next[:, :min_dim]
            compressed_next = compressed_next[:, :min_dim]
            
            # Prediction error
            pred_error = torch.mean((compressed_next - predicted_next)**2, dim=1, keepdim=True)
            
            # Route based on prediction accuracy
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Add routing cost
            routing_balance = confidence * (1 - confidence)
            routing_cost = 0.1 * torch.mean(routing_balance)
            pred_error = pred_error + routing_cost
            
            # Route information
            penultimate_features = self.to_penultimate(hidden)
            penultimate_contribution = penultimate_features * confidence
            continue_up = hidden * (1 - confidence)
            
            # Track comprehensive statistics
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.mean(torch.norm(continue_up.detach(), dim=1)),
                layer_idx=layer_idx,
                pattern_usage=my_patterns.detach().mean(0),
                pattern_entropy=self.last_entropy,
                self_paradox_magnitude=self.last_paradox_magnitude,
                pattern_attention_entropy=self.last_pattern_attention_entropy,
                adaptive_temperature=adaptive_temp
            )
            
            return continue_up, penultimate_contribution, pred_error
            
        else:
            # Last layer processing
            penultimate_contribution = self.to_penultimate(hidden)
            _, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            
            self.last_stats = LayerStats(
                prediction_errors=torch.zeros(1, 1, device=x.device),
                confidence_values=torch.ones(1, 1, device=x.device),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.tensor(0.0, device=x.device),
                layer_idx=layer_idx,
                pattern_usage=my_patterns.detach().mean(0),
                pattern_entropy=self.last_entropy,
                self_paradox_magnitude=self.last_paradox_magnitude,
                pattern_attention_entropy=self.last_pattern_attention_entropy,
                adaptive_temperature=adaptive_temp
            )
            
            return None, penultimate_contribution, None

class DiscretePatternPredictiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, penultimate_dim, output_dim, n_patterns=8,
                 initial_temp=1.0, min_temp=0.1, temp_decay=0.99, temp_lr=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        # Create layers
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else penultimate_dim
            
            layer = DiscretePatternLayer(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                next_dim=next_dim,
                penultimate_dim=penultimate_dim,
                n_patterns=n_patterns,
                initial_temp=initial_temp,
                min_temp=min_temp,
                temp_decay=temp_decay,
                temp_lr=temp_lr
            )
            self.layers.append(layer)
            current_dim = hidden_dim
        
        self.final = nn.Linear(penultimate_dim, output_dim)
    
    def update_temperatures(self):
        """Update base temperature for all layers"""
        for layer in self.layers:
            layer.update_temperature()
    
    def get_layer_stats(self) -> List[LayerStats]:
        """Get comprehensive statistics including all new metrics"""
        return [layer.last_stats for layer in self.layers if layer.last_stats is not None]
    
    def get_pattern_correlations(self) -> List[torch.Tensor]:
        """Get correlation matrices between patterns"""
        pattern_correlations = []
        for layer in self.layers:
            patterns = layer.pattern_dict
            normalized = F.normalize(patterns, dim=1)
            corr = normalized @ normalized.T
            pattern_correlations.append(corr)
        return pattern_correlations
    
    def get_self_paradox_stats(self) -> List[float]:
        """Get self-paradox magnitudes for all layers"""
        return [layer.last_paradox_magnitude for layer in self.layers]
    
    def get_adaptive_temperatures(self) -> List[float]:
        """Get adaptive temperatures for all layers"""
        stats = self.get_layer_stats()
        return [stat.adaptive_temperature for stat in stats if stat is not None]
    
    def get_pattern_attention_stats(self) -> List[float]:
        """Get pattern attention entropies (coordination measure)"""
        stats = self.get_layer_stats()
        return [stat.pattern_attention_entropy for stat in stats if stat is not None]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        penultimate_contributions = []
        current = x
        all_errors = []
        
        # Process through layers
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