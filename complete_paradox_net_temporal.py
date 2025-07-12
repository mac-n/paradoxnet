import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Note: This version incorporates a fully predictive penultimate layer.

@dataclass
class LayerStats:
    """Track statistics for a single layer during forward pass"""
    layer_idx: int
    prediction_errors: Optional[torch.Tensor] = None
    confidence_values: Optional[torch.Tensor] = None
    penultimate_magnitude: Optional[torch.Tensor] = None
    continue_magnitude: Optional[torch.Tensor] = None
    pattern_usage: Optional[torch.Tensor] = None
    pattern_entropy: float = 0.0
    self_paradox_magnitude: float = 0.0
    temporal_temperature: float = 1.0

class DiscretePatternLayer(nn.Module):
    """The standard hidden layer from our previous version."""
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8, temporal_lr=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        self.temporal_lr = temporal_lr

        self.register_buffer('temporal_temperature', torch.tensor(1.0))
        self.register_buffer('is_first_temporal_epoch', torch.tensor(True, dtype=torch.bool))
        self.register_buffer('previous_pattern_dict', None, persistent=True)

        self.process = nn.Linear(input_dim, hidden_dim)
        self.self_predictor = nn.Linear(hidden_dim, hidden_dim)
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim) / hidden_dim**0.5)
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        self.next_pattern_dict = nn.Parameter(torch.randn(n_patterns, next_dim) / next_dim**0.5)
        self.next_pattern_attention = nn.Linear(hidden_dim, n_patterns)
        self.to_penultimate = nn.Linear(hidden_dim, penultimate_dim)
        self.last_stats: Optional[LayerStats] = None

    def _get_effective_temperature(self):
        return self.temporal_temperature

    def apply_self_processing(self, x: torch.Tensor) -> torch.Tensor:
        hidden_linear = self.process(x)
        effective_temp = self._get_effective_temperature()
        attn = self.pattern_attention(hidden_linear)
        pattern_weights = F.softmax(attn / effective_temp, dim=-1)
        
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox)
        
        with torch.no_grad():
            if self.last_stats:
                self.last_stats.pattern_usage = pattern_weights.mean(0)
                self.last_stats.self_paradox_magnitude = torch.mean(torch.norm(paradox, dim=-1)).item()
        return hidden

    def forward(self, x: torch.Tensor, next_layer: nn.Module, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.last_stats = LayerStats(layer_idx=layer_idx, temporal_temperature=self.temporal_temperature.item())
        
        hidden = self.apply_self_processing(x)

        with torch.no_grad():
            # This is the key part that requires next_layer to have apply_self_processing
            next_hidden = next_layer.apply_self_processing(hidden)
            actual_next_state = next_layer.pattern_attention(next_hidden) @ next_layer.pattern_dict
        
        predicted_next_state = self.next_pattern_attention(hidden) @ self.next_pattern_dict
        
        pred_error = F.mse_loss(predicted_next_state, actual_next_state, reduction='none').mean(dim=1, keepdim=True)
        confidence = torch.exp(-pred_error)

        penultimate_features = self.to_penultimate(hidden)
        penultimate_contribution = penultimate_features * confidence
        continue_up = hidden * (1 - confidence)

        with torch.no_grad():
            self.last_stats.prediction_errors = pred_error
            self.last_stats.confidence_values = confidence
            self.last_stats.penultimate_magnitude = torch.mean(torch.norm(penultimate_contribution, dim=1))
            self.last_stats.continue_magnitude = torch.mean(torch.norm(continue_up, dim=1))

        return continue_up, penultimate_contribution, pred_error

class PenultimatePatternLayer(nn.Module):
    """A new predictive layer for the end of the network."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_patterns=8, temporal_lr=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_patterns = n_patterns
        self.temporal_lr = temporal_lr

        self.register_buffer('temporal_temperature', torch.tensor(1.0))
        self.register_buffer('is_first_temporal_epoch', torch.tensor(True, dtype=torch.bool))
        self.register_buffer('previous_pattern_dict', None, persistent=True)

        self.process = nn.Linear(input_dim, hidden_dim)
        self.self_predictor = nn.Linear(hidden_dim, hidden_dim)
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim) / hidden_dim**0.5)
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        self.output_predictor = nn.Linear(hidden_dim, output_dim)
        self.last_stats: Optional[LayerStats] = None

    def _get_effective_temperature(self):
        return self.temporal_temperature

    # CORRECTED: Added the missing apply_self_processing method for consistency
    def apply_self_processing(self, x: torch.Tensor) -> torch.Tensor:
        """This method is now consistent with DiscretePatternLayer."""
        hidden_linear = self.process(x)
        effective_temp = self._get_effective_temperature()
        attn = self.pattern_attention(hidden_linear)
        pattern_weights = F.softmax(attn / effective_temp, dim=-1)
        
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox)

        with torch.no_grad():
            if self.last_stats:
                self.last_stats.pattern_usage = pattern_weights.mean(0)
                self.last_stats.self_paradox_magnitude = torch.mean(torch.norm(paradox, dim=-1)).item()
        return hidden

    def forward(self, x: torch.Tensor, y: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.last_stats = LayerStats(layer_idx=layer_idx, temporal_temperature=self.temporal_temperature.item())
        
        hidden = self.apply_self_processing(x)

        predicted_output = self.output_predictor(hidden)
        
        y_one_hot = F.one_hot(y.long(), num_classes=self.output_dim).float()
        pred_error = F.mse_loss(predicted_output, y_one_hot, reduction='none').mean(dim=1, keepdim=True)
        
        self.last_stats.prediction_errors = pred_error.detach()
        
        return predicted_output, pred_error

class CompleteParadoxNetTemporal(nn.Module):
    def __init__(self, input_dim, hidden_dims, penultimate_dim, output_dim, n_patterns=8, temporal_lr=0.1):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList()
        current_dim = input_dim
        
        for i in range(len(hidden_dims)):
            # The next_dim for the last hidden layer is the input_dim of the penultimate layer
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else penultimate_dim
            layer = DiscretePatternLayer(
                input_dim=current_dim, hidden_dim=hidden_dims[i], next_dim=next_dim,
                penultimate_dim=penultimate_dim, n_patterns=n_patterns, temporal_lr=temporal_lr
            )
            self.hidden_layers.append(layer)
            current_dim = hidden_dims[i]

        self.penultimate_layer = PenultimatePatternLayer(
            input_dim=penultimate_dim, hidden_dim=penultimate_dim, output_dim=output_dim,
            n_patterns=n_patterns, temporal_lr=temporal_lr
        )

    def update_temporal_temperatures(self):
        with torch.no_grad():
            all_layers = list(self.hidden_layers) + [self.penultimate_layer]
            for layer in all_layers:
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

    def get_layer_stats(self) -> List[LayerStats]:
        return [layer.last_stats for layer in self.hidden_layers if layer.last_stats] + \
               ([self.penultimate_layer.last_stats] if self.penultimate_layer.last_stats else [])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        penultimate_contributions = []
        current = x
        all_errors = []

        for i, layer in enumerate(self.hidden_layers):
            # Determine the next layer in the sequence for prediction
            next_layer = self.hidden_layers[i+1] if i < len(self.hidden_layers)-1 else self.penultimate_layer
            current, penultimate, error = layer(current, next_layer, i)
            all_errors.append(error)
            penultimate_contributions.append(penultimate)
        
        penultimate_input = torch.sum(torch.stack(penultimate_contributions), dim=0)
        final_output, penultimate_error = self.penultimate_layer(penultimate_input, y, layer_idx=len(self.hidden_layers))
        all_errors.append(penultimate_error)
        
        total_prediction_error = torch.cat(all_errors, dim=1)
        
        return final_output, total_prediction_error

