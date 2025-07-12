import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

# THE ULTIMATE PARADOX NET: Complex numbers + ALL the architectural innovations!

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
    temporal_temperatures: Optional[torch.Tensor] = None
    adaptive_temperature_factor: float = 1.0

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
    """A linear layer that operates on complex-valued tensors, now handling sequences."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Note: We divide in_features by 2 because the complex dimension is half the real dimension
        self.weight_re = nn.Parameter(torch.randn(in_features // 2, out_features // 2) * 0.02)
        self.weight_im = nn.Parameter(torch.randn(in_features // 2, out_features // 2) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a complex tensor. It can be [batch, out_features/2] or [batch, seq_len, out_features/2]
        x_re, x_im = x.real, x.imag

        if x.dim() == 3: # Has a sequence dimension
            # Using einsum for clarity: 'bsi,io->bso'
            # b: batch, s: sequence, i: in_features, o: out_features
            out_re = torch.einsum('bsi,io->bso', x_re, self.weight_re) - torch.einsum('bsi,io->bso', x_im, self.weight_im)
            out_im = torch.einsum('bsi,io->bso', x_re, self.weight_im) + torch.einsum('bsi,io->bso', x_im, self.weight_re)
        else: # No sequence dimension
            out_re = x_re @ self.weight_re - x_im @ self.weight_im
            out_im = x_re @ self.weight_im + x_im @ self.weight_re
            
        return torch.complex(out_re, out_im)

class DiscretePatternLayer(nn.Module):
    """Ultimate hidden layer: Complex numbers + confidence routing + temporal temp + next-layer prediction!"""
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8, temporal_lr=0.1, temp_lr=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  
        self.next_dim = next_dim
        self.penultimate_dim = penultimate_dim
        self.n_patterns = n_patterns
        self.temporal_lr = temporal_lr
        self.temp_lr = temp_lr
        
        # Temperature management
        self.base_temp = 1.0
        self.register_buffer('temporal_temperatures', torch.ones(n_patterns))
        self.register_buffer('is_first_temporal_epoch', torch.tensor(True, dtype=torch.bool))
        self.register_buffer('previous_pattern_dict', None, persistent=True)
        
        # Core processing (complex-valued!)
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # Pattern dictionaries (complex-valued!)
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02)
        self.pattern_attention = ComplexLinear(hidden_dim, n_patterns * 2)  # Real output for softmax
        
        # NEXT-LAYER PREDICTION (the secret sauce!)
        self.next_pattern_dict = nn.Parameter(torch.randn(n_patterns, next_dim // 2, dtype=torch.cfloat) * 0.02)
        self.next_pattern_attention = ComplexLinear(hidden_dim, n_patterns * 2)  # Real output for softmax
        
        # Output pathways
        self.to_penultimate = ComplexLinear(hidden_dim, penultimate_dim)
        
        # Stats tracking
        self.last_stats: Optional[LayerStats] = None

    def _get_effective_temperatures(self):
        """Get per-pattern temporal temperatures"""
        return self.base_temp * self.temporal_temperatures

    def apply_self_processing(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Enhanced self-processing with complex numbers and adaptive temperature"""
        hidden_linear = self.process(x)
        
        # Get temporal temperatures
        effective_temps = self._get_effective_temperatures()
        
        # Self-paradox mechanism (complex version!)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        
        # Adaptive temperature based on self-prediction accuracy
        with torch.no_grad():
            self_pred_accuracy = torch.mean(paradox.abs()**2).item()  # Complex magnitude squared
            adaptive_temp_factor = 1.0 + self.temp_lr * self_pred_accuracy
        
        # Apply gating with complex-aware sigmoid
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        
        # Pattern attention with effective temperature
        # Convert complex attention to real for softmax
        attn_complex = self.pattern_attention(hidden)
        attn_real = attn_complex.real  # Take real part for softmax
        
        # Apply temperature (broadcast across patterns)
        if attn_real.dim() == 3:  # [batch, seq, n_patterns*2] -> [batch, seq, n_patterns]
            attn_real = attn_real[:, :, :self.n_patterns]
        else:  # [batch, n_patterns*2] -> [batch, n_patterns]
            attn_real = attn_real[:, :self.n_patterns]
            
        pattern_weights = F.softmax(attn_real / effective_temps, dim=-1)
        
        # Track stats
        with torch.no_grad():
            if self.last_stats:
                self.last_stats.pattern_usage = pattern_weights.mean(0) if pattern_weights.dim() > 1 else pattern_weights
                self.last_stats.self_paradox_magnitude = torch.mean(paradox.abs()).item()
                self.last_stats.adaptive_temperature_factor = adaptive_temp_factor
                
        return hidden, adaptive_temp_factor

    def predict_next_layer(self, hidden: torch.Tensor, next_layer) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict what the next layer will compute - THE CONFIDENCE ROUTING CORE!"""
        
        # Our prediction of next layer's state
        # Convert to real for attention, then apply to complex patterns
        next_attn_complex = self.next_pattern_attention(hidden)
        next_attn_real = next_attn_complex.real
        
        # Handle different tensor dimensions
        if next_attn_real.dim() == 3:  # [batch, seq, n_patterns*2]
            next_attn_real = next_attn_real[:, :, :self.n_patterns]
        else:  # [batch, n_patterns*2]
            next_attn_real = next_attn_real[:, :self.n_patterns]
            
        next_pattern_weights = F.softmax(next_attn_real, dim=-1)
        
        # Predict next state - fix einsum dimensions and complex types
        # Convert real weights to complex for computation
        next_pattern_weights_complex = next_pattern_weights.to(dtype=self.next_pattern_dict.dtype)
        
        if next_pattern_weights.dim() == 3 and self.next_pattern_dict.dim() == 2:
            # [batch, seq, patterns] @ [patterns, hidden] -> [batch, seq, hidden]
            predicted_next_state = torch.einsum('bsp,ph->bsh', next_pattern_weights_complex, self.next_pattern_dict)
            predicted_next_state = predicted_next_state.mean(dim=1)  # Pool sequence
        elif next_pattern_weights.dim() == 2 and self.next_pattern_dict.dim() == 2:
            # [batch, patterns] @ [patterns, hidden] -> [batch, hidden]
            predicted_next_state = torch.einsum('bp,ph->bh', next_pattern_weights_complex, self.next_pattern_dict)
        else:
            # Fallback: use matrix multiplication with proper reshaping
            if next_pattern_weights.dim() == 3:
                next_pattern_weights_complex = next_pattern_weights_complex.mean(dim=1)  # Pool sequence
            predicted_next_state = next_pattern_weights_complex @ self.next_pattern_dict
        
        # Get actual next layer computation
        with torch.no_grad():
            if hasattr(next_layer, 'apply_self_processing'):
                # If it's another DiscretePatternLayer
                next_hidden, _ = next_layer.apply_self_processing(hidden)
                
                # Get next layer's actual pattern activation
                next_attn_actual_complex = next_layer.pattern_attention(next_hidden)
                next_attn_actual_real = next_attn_actual_complex.real
                
                if next_attn_actual_real.dim() == 3:
                    next_attn_actual_real = next_attn_actual_real[:, :, :next_layer.n_patterns]
                else:
                    next_attn_actual_real = next_attn_actual_real[:, :next_layer.n_patterns]
                    
                next_actual_weights = F.softmax(next_attn_actual_real, dim=-1)
                
                # Same dimension fix for actual computation - handle complex types
                next_actual_weights_complex = next_actual_weights.to(dtype=next_layer.pattern_dict.dtype)
                
                if next_actual_weights.dim() == 3 and next_layer.pattern_dict.dim() == 2:
                    actual_next_state = torch.einsum('bsp,ph->bsh', next_actual_weights_complex, next_layer.pattern_dict)
                    actual_next_state = actual_next_state.mean(dim=1)  # Pool sequence
                elif next_actual_weights.dim() == 2 and next_layer.pattern_dict.dim() == 2:
                    actual_next_state = torch.einsum('bp,ph->bh', next_actual_weights_complex, next_layer.pattern_dict)
                else:
                    if next_actual_weights.dim() == 3:
                        next_actual_weights_complex = next_actual_weights_complex.mean(dim=1)
                    actual_next_state = next_actual_weights_complex @ next_layer.pattern_dict
            else:
                # If it's the penultimate layer, predict its input processing
                hidden_for_penultimate = hidden.mean(dim=1) if hidden.dim() == 3 else hidden
                next_processed = next_layer.process(hidden_for_penultimate)
                actual_next_state = next_processed
                
                # Ensure predicted state matches actual state dimensions
                if predicted_next_state.dim() > actual_next_state.dim():
                    predicted_next_state = predicted_next_state.mean(dim=1)
        
        # Compute prediction error (complex MSE) - ensure matching dimensions
        if predicted_next_state.shape != actual_next_state.shape:
            # Reshape to match
            min_dim = min(predicted_next_state.size(-1), actual_next_state.size(-1))
            predicted_next_state = predicted_next_state[..., :min_dim]
            actual_next_state = actual_next_state[..., :min_dim]
        
        pred_error = F.mse_loss(predicted_next_state.real, actual_next_state.real, reduction='none').mean(dim=-1, keepdim=True)
        pred_error += F.mse_loss(predicted_next_state.imag, actual_next_state.imag, reduction='none').mean(dim=-1, keepdim=True)
        
        # Convert to confidence
        confidence = torch.exp(-pred_error)
        
        return confidence, pred_error

    def forward(self, x: torch.Tensor, next_layer, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """THE ULTIMATE FORWARD PASS: Complex + Confidence + Prediction + Everything!"""
        self.last_stats = LayerStats(layer_idx=layer_idx, temporal_temperatures=self.temporal_temperatures.detach())
        
        # Enhanced self-processing
        hidden, adaptive_temp = self.apply_self_processing(x)
        
        # CONFIDENCE-BASED ROUTING (the heart of the architecture!)
        if isinstance(next_layer, PenultimatePatternLayer):
            # CORRECTED LOGIC: Last hidden layer passes EVERYTHING as residual
            # Its job is to produce the final "unexplained" representation
            continue_up = hidden
            # It contributes NOTHING to consensus - but must match other layers' penultimate size
            penultimate_features = self.to_penultimate(hidden)
            if penultimate_features.dim() == 3:  # Has sequence dimension
                penultimate_features = penultimate_features.mean(dim=1)  # Pool over sequence
            penultimate_contribution = torch.zeros_like(penultimate_features)
            # No prediction error since it doesn't predict
            pred_error = torch.zeros(hidden.size(0), 1, device=hidden.device)
        else:
            # Predict next layer and route based on confidence
            confidence, pred_error = self.predict_next_layer(hidden, next_layer)
            
            # Route information based on confidence!
            penultimate_features = self.to_penultimate(hidden)
            if penultimate_features.dim() == 3:  # Has sequence dimension
                penultimate_features = penultimate_features.mean(dim=1)  # Pool over sequence
                
            penultimate_contribution = penultimate_features * confidence
            
            # Continue up signal (what we're uncertain about)
            continue_up = hidden * (1 - confidence.unsqueeze(-1))
        
        # Track comprehensive stats
        with torch.no_grad():
            self.last_stats.prediction_errors = pred_error
            # Handle confidence values for stats (use dummy for last layer)
            if isinstance(next_layer, PenultimatePatternLayer):
                dummy_confidence = torch.zeros(hidden.size(0), 1, device=hidden.device)  # Show 0 confidence for last layer
                self.last_stats.confidence_values = dummy_confidence
            else:
                self.last_stats.confidence_values = confidence
            self.last_stats.penultimate_magnitude = torch.mean(torch.norm(penultimate_contribution.abs(), dim=1))
            self.last_stats.continue_magnitude = torch.mean(torch.norm(continue_up.abs(), dim=1))
        
        return continue_up, penultimate_contribution, pred_error

class PenultimatePatternLayer(nn.Module):
    """Ultimate penultimate layer with complex numbers and temporal temperature"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_patterns=8, temporal_lr=0.1, temp_lr=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_patterns = n_patterns
        self.temporal_lr = temporal_lr
        self.temp_lr = temp_lr
        
        # Temperature management
        self.base_temp = 1.0
        self.register_buffer('temporal_temperatures', torch.ones(n_patterns))
        self.register_buffer('is_first_temporal_epoch', torch.tensor(True, dtype=torch.bool))
        self.register_buffer('previous_pattern_dict', None, persistent=True)
        
        # Complex processing
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02)
        self.pattern_attention = ComplexLinear(hidden_dim, n_patterns * 2)
        
        # Real-valued output
        self.output_predictor = nn.Linear(hidden_dim // 2, output_dim)
        self.last_stats: Optional[LayerStats] = None

    def apply_self_processing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex self-processing with temporal temperature"""
        hidden_linear = self.process(x)
        effective_temps = self.base_temp * self.temporal_temperatures
        
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        
        # Adaptive temperature
        with torch.no_grad():
            self_pred_accuracy = torch.mean(paradox.abs()**2).item()
            adaptive_temp_factor = 1.0 + self.temp_lr * self_pred_accuracy
        
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        
        # Pattern attention
        attn_complex = self.pattern_attention(hidden)
        attn_real = attn_complex.real[:, :self.n_patterns]
        pattern_weights = F.softmax(attn_real / (effective_temps * adaptive_temp_factor), dim=-1)
        
        return hidden

    def forward(self, x: torch.Tensor, y: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with output prediction"""
        self.last_stats = LayerStats(layer_idx=layer_idx, temporal_temperatures=self.temporal_temperatures.detach())
        
        hidden = self.apply_self_processing(x)
        predicted_output = self.output_predictor(hidden.real)
        
        # Compute prediction error
        y_one_hot = F.one_hot(y.long(), num_classes=self.output_dim).float()
        pred_error = F.mse_loss(predicted_output, y_one_hot, reduction='none').mean(dim=1, keepdim=True)
        
        self.last_stats.prediction_errors = pred_error.detach()
        return predicted_output, pred_error

class ParadoxNetComplexUltimate(nn.Module):
    """THE ULTIMATE PARADOX NET: All innovations combined!"""
    def __init__(self, vocab_size, embedding_dim, hidden_dims, penultimate_dim, n_patterns=8, temporal_lr=0.1, temp_lr=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Build the hidden layers with full connectivity info
        self.hidden_layers = nn.ModuleList()
        current_dim = embedding_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else penultimate_dim
            layer = DiscretePatternLayer(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                next_dim=next_dim,
                penultimate_dim=penultimate_dim,
                n_patterns=n_patterns,
                temporal_lr=temporal_lr,
                temp_lr=temp_lr
            )
            self.hidden_layers.append(layer)
            current_dim = hidden_dim
        
        # Consensus + residual input dimension
        penultimate_input_dim = penultimate_dim + hidden_dims[-1]
        
        self.penultimate_layer = PenultimatePatternLayer(
            input_dim=penultimate_input_dim,
            hidden_dim=penultimate_dim,
            output_dim=vocab_size,
            n_patterns=n_patterns,
            temporal_lr=temporal_lr,
            temp_lr=temp_lr
        )

    def update_temporal_temperatures(self):
        """Update temporal temperatures based on pattern dictionary changes"""
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
                    # Complex MSE for temporal error
                    temporal_error_per_pattern = (
                        F.mse_loss(layer.pattern_dict.real, layer.previous_pattern_dict.real, reduction='none').mean(dim=1) +
                        F.mse_loss(layer.pattern_dict.imag, layer.previous_pattern_dict.imag, reduction='none').mean(dim=1)
                    )
                    new_temps = 1.0 + layer.temporal_lr * temporal_error_per_pattern
                    layer.temporal_temperatures.copy_(new_temps)
                    layer.previous_pattern_dict.copy_(layer.pattern_dict)

    def get_layer_stats(self) -> List[LayerStats]:
        """Get all layer statistics for analysis"""
        stats = [layer.last_stats for layer in self.hidden_layers if layer.last_stats]
        if self.penultimate_layer.last_stats:
            stats.append(self.penultimate_layer.last_stats)
        return stats

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """THE ULTIMATE FORWARD PASS: Everything working together!"""
        batch_size, seq_len = x.shape
        
        # Embedding + RoPE
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

        # Process through hidden layers with confidence routing!
        penultimate_contributions = []
        all_errors = []
        
        for i, layer in enumerate(self.hidden_layers):
            next_layer = self.hidden_layers[i+1] if i < len(self.hidden_layers)-1 else self.penultimate_layer
            current_seq, penultimate, error = layer(current_seq, next_layer, i)
            all_errors.append(error)
            penultimate_contributions.append(penultimate)
        
        # CONSENSUS VS RESIDUAL ARCHITECTURE
        consensus_view = torch.sum(torch.stack(penultimate_contributions), dim=0)
        recursive_residual = current_seq.mean(dim=1)  # Pool sequence for residual
        
        # Concatenate for final processing
        penultimate_input = torch.cat([consensus_view, recursive_residual], dim=1)
        
        # Final prediction
        final_output, penultimate_error = self.penultimate_layer(penultimate_input, y, layer_idx=len(self.hidden_layers))
        all_errors.append(penultimate_error)
        
        # RECURSIVE RESIDUAL LOSS (interpretability pressure!)
        recursive_residual_loss = torch.mean(torch.norm(recursive_residual.abs(), p=2, dim=1))
        
        total_prediction_error = torch.cat(all_errors, dim=1)
        
        return final_output, total_prediction_error, recursive_residual_loss