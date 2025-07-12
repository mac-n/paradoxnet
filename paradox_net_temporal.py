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
    pattern_attention_entropy: float = 0.0
    adaptive_temperature: float = 0.0
    # NEW: Add temporal temperature to stats for tracking
    temporal_temperature: float = 1.0

class DiscretePatternLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8,
                 initial_temp=1.0, min_temp=0.1, temp_decay=0.99, temp_lr=0.1,
                 # NEW: Temporal loop parameters
                 temporal_lr=0.1, temporal_base_temp=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        self.last_entropy = 0.0
        self.temp_lr = temp_lr

        # Temperature parameters
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay

        # --- NEW: Temporal Regularization Parameters ---
        self.temporal_lr = temporal_lr
        # Register buffers to make them part of the model's state_dict, but not trainable parameters
        self.register_buffer('temporal_temperature', torch.tensor(temporal_base_temp))
        self.register_buffer('is_first_temporal_epoch', torch.tensor(True, dtype=torch.bool))
        # Initialize previous_pattern_dict buffer, it will be populated on the first update
        self.register_buffer('previous_pattern_dict', None, persistent=True)
        # --- End of New Temporal Parameters ---

        # Main processing pathway
        self.process = nn.Linear(input_dim, hidden_dim)
        self.self_predictor = nn.Linear(hidden_dim, hidden_dim)

        # Pattern dictionaries
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim) / hidden_dim**0.5)
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        self.next_pattern_dict = nn.Parameter(torch.randn(n_patterns, next_dim) / next_dim**0.5)
        self.next_pattern_attention = nn.Linear(hidden_dim, n_patterns)

        # Self-attention between patterns for coordination
        self.pattern_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=1, batch_first=True, dropout=0.0
        )

        # Output pathway
        self.to_penultimate = nn.Linear(hidden_dim, penultimate_dim)

        # Stats tracking
        self.last_stats: Optional[LayerStats] = None
        self.last_paradox_magnitude: float = 0.0
        self.last_pattern_attention_entropy: float = 0.0

    def update_base_temperature(self):
        """Anneal base temperature (the original temperature update)."""
        self.current_temp = max(self.min_temp, self.current_temp * self.temp_decay)

    def _get_effective_temperature(self):
        """Calculate the final temperature by combining base, adaptive, and temporal factors."""
        # The temporal temperature acts as a multiplier on the base annealing temperature.
        return self.current_temp * self.temporal_temperature

    def apply_unified_self_processing(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Apply pattern coordination + self-paradox nonlinearity + temperature modulation"""
        hidden_linear = self.process(x)
        
        # Get the combined effective temperature for this forward pass
        effective_temp = self._get_effective_temperature()

        attn = self.pattern_attention(hidden_linear)
        pattern_weights = F.softmax(attn / effective_temp, dim=-1)

        with torch.no_grad():
            entropy = -torch.sum(pattern_weights * torch.log(pattern_weights + 1e-10), dim=-1)
            self.last_entropy = entropy.mean().item()

        patterns = self.pattern_dict.unsqueeze(0).expand(hidden_linear.size(0), -1, -1)
        if patterns.size(0) > 0:
            attended_patterns, attention_weights = self.pattern_self_attention(patterns, patterns, patterns)
            with torch.no_grad():
                attn_flat = attention_weights.view(-1, attention_weights.size(-1))
                attn_entropy = -torch.sum(attn_flat * torch.log(attn_flat + 1e-10), dim=-1)
                self.last_pattern_attention_entropy = attn_entropy.mean().item()
        else:
            attended_patterns = patterns
            self.last_pattern_attention_entropy = 0.0

        coordinated_contribution = (pattern_weights.unsqueeze(-1) * attended_patterns).sum(dim=1)
        hidden_enhanced = hidden_linear + coordinated_contribution

        self_prediction = self.self_predictor(hidden_enhanced)
        paradox = self_prediction - hidden_enhanced

        with torch.no_grad():
            self.last_paradox_magnitude = torch.mean(torch.norm(paradox, dim=-1)).item()

        hidden = hidden_enhanced * torch.sigmoid(paradox)

        self_pred_accuracy = torch.mean(paradox**2).item()
        # The adaptive temp now modulates the *effective* temperature
        adaptive_temp = effective_temp * (1 + self.temp_lr * self_pred_accuracy)
        adaptive_temp = max(self.min_temp, adaptive_temp)

        return hidden, adaptive_temp

    def forward(self, x: torch.Tensor, next_layer: Optional['DiscretePatternLayer'],
                layer_idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        hidden, adaptive_temp = self.apply_unified_self_processing(x)

        if next_layer is not None:
            original_temp = self.current_temp
            self.current_temp = adaptive_temp
            my_compressed, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            predicted_next = my_compressed
            with torch.no_grad():
                actual_next, _ = next_layer.apply_unified_self_processing(hidden)
                compressed_next, _ = next_layer.compress_activity(actual_next, is_next_layer=True)
            self.current_temp = original_temp
            min_dim = min(predicted_next.size(1), compressed_next.size(1))
            predicted_next = predicted_next[:, :min_dim]
            compressed_next = compressed_next[:, :min_dim]
            pred_error = torch.mean((compressed_next - predicted_next)**2, dim=1, keepdim=True)
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            routing_balance = confidence * (1 - confidence)
            routing_cost = 0.1 * torch.mean(routing_balance)
            pred_error = pred_error + routing_cost
            penultimate_features = self.to_penultimate(hidden)
            penultimate_contribution = penultimate_features * confidence
            continue_up = hidden * (1 - confidence)
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
                adaptive_temperature=adaptive_temp,
                temporal_temperature=self.temporal_temperature.item()
            )
            return continue_up, penultimate_contribution, pred_error
        else:
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
                adaptive_temperature=adaptive_temp,
                temporal_temperature=self.temporal_temperature.item()
            )
            return None, penultimate_contribution, None

    def compress_activity(self, x: torch.Tensor, is_next_layer: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1: x = x.unsqueeze(0)
        attention_layer = self.next_pattern_attention if is_next_layer else self.pattern_attention
        patterns = self.next_pattern_dict if is_next_layer else self.pattern_dict
        effective_temp = self._get_effective_temperature()
        attn = attention_layer(x)
        pattern_weights = F.softmax(attn / effective_temp, dim=-1)
        with torch.no_grad():
            entropy = -torch.sum(pattern_weights * torch.log(pattern_weights + 1e-10), dim=-1)
            self.last_entropy = entropy.mean().item()
        compressed = pattern_weights @ patterns
        return compressed, pattern_weights

class DiscretePatternPredictiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, penultimate_dim, output_dim, n_patterns=8,
                 initial_temp=1.0, min_temp=0.1, temp_decay=0.99, temp_lr=0.1,
                 temporal_lr=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else penultimate_dim
            layer = DiscretePatternLayer(
                input_dim=current_dim, hidden_dim=hidden_dim, next_dim=next_dim,
                penultimate_dim=penultimate_dim, n_patterns=n_patterns,
                initial_temp=initial_temp, min_temp=min_temp, temp_decay=temp_decay,
                temp_lr=temp_lr, temporal_lr=temporal_lr
            )
            self.layers.append(layer)
            current_dim = hidden_dim
        self.final = nn.Linear(penultimate_dim, output_dim)

    def update_temporal_temperatures(self):
        with torch.no_grad():
            for layer in self.layers:
                if layer.is_first_temporal_epoch:
                    if layer.previous_pattern_dict is None:
                         layer.register_buffer('previous_pattern_dict', layer.pattern_dict.clone(), persistent=True)
                    else:
                         layer.previous_pattern_dict.copy_(layer.pattern_dict)
                    layer.is_first_temporal_epoch.fill_(False)
                else:
                    temporal_error = F.mse_loss(layer.pattern_dict, layer.previous_pattern_dict)
                    new_temp = 1.0 + layer.temporal_lr * temporal_error
                    layer.temporal_temperature.fill_(new_temp)
                    layer.previous_pattern_dict.copy_(layer.pattern_dict)

    def update_base_temperatures(self):
        for layer in self.layers:
            layer.update_base_temperature()

    def get_layer_stats(self) -> List[LayerStats]:
        return [layer.last_stats for layer in self.layers if layer.last_stats is not None]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        penultimate_contributions = []
        current = x
        all_errors = []
        for i, layer in enumerate(self.layers):
            next_layer = self.layers[i+1] if i < len(self.layers)-1 else None
            current, penultimate, error = layer(current, next_layer, i)
            if error is not None: all_errors.append(error)
            penultimate_contributions.append(penultimate)
        penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
        output = self.final(penultimate)
        return output, torch.cat(all_errors, dim=1) if all_errors else None
