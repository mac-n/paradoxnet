import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Note: This version implements "Explicit Surprise Tagging" and fixes the runtime error.

@dataclass
class LayerStats:
    layer_idx: int
    prediction_errors: Optional[torch.Tensor] = None
    confidence_values: Optional[torch.Tensor] = None
    penultimate_magnitude: Optional[torch.Tensor] = None
    continue_magnitude: Optional[torch.Tensor] = None
    pattern_usage: Optional[torch.Tensor] = None
    self_paradox_magnitude: float = 0.0
    temporal_temperatures: Optional[torch.Tensor] = None
    adaptive_temperature_factor: float = 1.0

class DiscretePatternLayer(nn.Module):
    """Hidden layer with combined temperature mechanisms."""
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8, temporal_lr=0.1, temp_lr=0.1):
        super().__init__()
        self.input_dim, self.hidden_dim, self.next_dim = input_dim, hidden_dim, next_dim
        self.penultimate_dim = penultimate_dim
        self.n_patterns, self.temporal_lr, self.temp_lr = n_patterns, temporal_lr, temp_lr

        self.base_temp = 1.0
        self.register_buffer('temporal_temperatures', torch.ones(n_patterns))
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

    def _get_base_effective_temperatures(self):
        return self.base_temp * self.temporal_temperatures

    def apply_self_processing(self, x: torch.Tensor) -> torch.Tensor:
        hidden_linear = self.process(x)
        base_effective_temps = self._get_base_effective_temperatures()
        
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        
        with torch.no_grad():
            self_pred_accuracy = torch.mean(paradox**2).item()
            adaptive_temp_factor = 1.0 + self.temp_lr * self_pred_accuracy
        
        final_effective_temps = base_effective_temps * adaptive_temp_factor
        attn = self.pattern_attention(hidden_linear)
        pattern_weights = F.softmax(attn / final_effective_temps, dim=-1)
        hidden = hidden_linear * torch.sigmoid(paradox)
        
        with torch.no_grad():
            if self.last_stats:
                self.last_stats.pattern_usage = pattern_weights.mean(0)
                self.last_stats.self_paradox_magnitude = torch.mean(torch.norm(paradox, dim=-1)).item()
                self.last_stats.adaptive_temperature_factor = adaptive_temp_factor
        return hidden

    def forward(self, x: torch.Tensor, next_layer: nn.Module, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.last_stats = LayerStats(layer_idx=layer_idx, temporal_temperatures=self.temporal_temperatures.detach())
        hidden = self.apply_self_processing(x)

        # CORRECTED LOGIC: Check if the next layer is the special penultimate layer
        if isinstance(next_layer, PenultimatePatternLayer):
            # This is the last hidden layer. It cannot predict the penultimate layer.
            # Assume full confidence and pass information up.
            pred_error = torch.zeros(x.size(0), 1, device=x.device)
            confidence = torch.ones(x.size(0), 1, device=x.device)
        else:
            # This is a regular hidden layer, do the prediction as before
            with torch.no_grad():
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
    """Penultimate layer, now accepts a larger concatenated input."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_patterns=8, temporal_lr=0.1, temp_lr=0.1):
        super().__init__()
        self.input_dim, self.hidden_dim, self.output_dim = input_dim, hidden_dim, output_dim
        self.n_patterns, self.temporal_lr, self.temp_lr = n_patterns, temporal_lr, temp_lr

        self.base_temp = 1.0
        self.register_buffer('temporal_temperatures', torch.ones(n_patterns))
        self.register_buffer('is_first_temporal_epoch', torch.tensor(True, dtype=torch.bool))
        self.register_buffer('previous_pattern_dict', None, persistent=True)

        self.process = nn.Linear(input_dim, hidden_dim)
        self.self_predictor = nn.Linear(hidden_dim, hidden_dim)
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim) / hidden_dim**0.5)
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        self.output_predictor = nn.Linear(hidden_dim, output_dim)
        self.last_stats: Optional[LayerStats] = None

    def _get_base_effective_temperatures(self):
        return self.base_temp * self.temporal_temperatures

    def apply_self_processing(self, x: torch.Tensor) -> torch.Tensor:
        hidden_linear = self.process(x)
        base_effective_temps = self._get_base_effective_temperatures()
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        with torch.no_grad():
            self_pred_accuracy = torch.mean(paradox**2).item()
            adaptive_temp_factor = 1.0 + self.temp_lr * self_pred_accuracy
        final_effective_temps = base_effective_temps * adaptive_temp_factor
        attn = self.pattern_attention(hidden_linear)
        pattern_weights = F.softmax(attn / final_effective_temps, dim=-1)
        hidden = hidden_linear * torch.sigmoid(paradox)
        with torch.no_grad():
            if self.last_stats:
                self.last_stats.pattern_usage = pattern_weights.mean(0)
                self.last_stats.self_paradox_magnitude = torch.mean(torch.norm(paradox, dim=-1)).item()
                self.last_stats.adaptive_temperature_factor = adaptive_temp_factor
        return hidden

    def forward(self, x: torch.Tensor, y: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.last_stats = LayerStats(layer_idx=layer_idx, temporal_temperatures=self.temporal_temperatures.detach())
        hidden = self.apply_self_processing(x)
        predicted_output = self.output_predictor(hidden)
        y_one_hot = F.one_hot(y.long(), num_classes=self.output_dim).float()
        pred_error = F.mse_loss(predicted_output, y_one_hot, reduction='none').mean(dim=1, keepdim=True)
        self.last_stats.prediction_errors = pred_error.detach()
        return predicted_output, pred_error

class CompleteParadoxNetSurpriseTag(nn.Module):
    def __init__(self, input_dim, hidden_dims, penultimate_dim, output_dim, n_patterns=8, temporal_lr=0.1, temp_lr=0.1):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        current_dim = input_dim
        
        for i in range(len(hidden_dims)):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else penultimate_dim
            self.hidden_layers.append(DiscretePatternLayer(
                current_dim, hidden_dims[i], next_dim, penultimate_dim, n_patterns, temporal_lr, temp_lr
            ))
            current_dim = hidden_dims[i]

        penultimate_input_dim = (len(hidden_dims) * penultimate_dim) + hidden_dims[-1]

        self.penultimate_layer = PenultimatePatternLayer(
            penultimate_input_dim, penultimate_dim, output_dim, n_patterns, temporal_lr, temp_lr
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
                    temporal_error_per_pattern = F.mse_loss(layer.pattern_dict, layer.previous_pattern_dict, reduction='none').mean(dim=1)
                    new_temps = 1.0 + layer.temporal_lr * temporal_error_per_pattern
                    layer.temporal_temperatures.copy_(new_temps)
                    layer.previous_pattern_dict.copy_(layer.pattern_dict)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        penultimate_contributions = []
        current = x
        all_errors = []

        for i, layer in enumerate(self.hidden_layers):
            next_layer = self.hidden_layers[i+1] if i < len(self.hidden_layers)-1 else self.penultimate_layer
            current, penultimate, error = layer(current, next_layer, i)
            all_errors.append(error)
            penultimate_contributions.append(penultimate)
        
        final_surprise_signal = current
        tagged_signals = penultimate_contributions + [final_surprise_signal]
        penultimate_input = torch.cat(tagged_signals, dim=1)
        
        final_output, penultimate_error = self.penultimate_layer(penultimate_input, y, layer_idx=len(self.hidden_layers))
        all_errors.append(penultimate_error)
        
        total_prediction_error = torch.cat(all_errors, dim=1)
        return final_output, total_prediction_error
