import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Complex ParadoxNet with working complex attention over patterns - 2025_07_15

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

class ComplexPatternAttention(nn.Module):
    """Complex attention over pattern dictionary"""
    def __init__(self, d_model, n_patterns):
        super().__init__()
        self.d_model = d_model
        self.n_patterns = n_patterns
        self.scale = 1 / (d_model ** 0.5)
        
        # Complex pattern dictionary
        self.patterns = nn.Parameter(torch.randn(n_patterns, d_model, dtype=torch.cfloat) * 0.02)
    
    def forward(self, hidden):
        """Apply complex attention to compress hidden state"""
        if hidden.dim() == 3:  # sequence case
            B, L, d = hidden.shape
            patterns_expanded = self.patterns.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
            
            # Hermitian inner product
            scores = torch.einsum('bld,blpd->blp', hidden, patterns_expanded.conj()) * self.scale
            attn_weights = F.softmax(scores.real, dim=-1)
            
            # Convert to complex for multiplication
            attn_complex = attn_weights.to(dtype=torch.cfloat)
            compressed = torch.einsum('blp,blpd->bld', attn_complex, patterns_expanded)
            
        else:  # no sequence case
            B, d = hidden.shape
            patterns_expanded = self.patterns.unsqueeze(0).expand(B, -1, -1)
            
            scores = torch.einsum('bd,bpd->bp', hidden, patterns_expanded.conj()) * self.scale
            attn_weights = F.softmax(scores.real, dim=-1)
            
            attn_complex = attn_weights.to(dtype=torch.cfloat)
            compressed = torch.einsum('bp,bpd->bd', attn_complex, patterns_expanded)
        
        return compressed, attn_weights

class DiscretePatternLayer(nn.Module):
    """Hidden layer with working complex attention over patterns."""
    def __init__(self, input_dim, hidden_dim, n_patterns=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_patterns = n_patterns
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # Replace vestigial patterns with working complex attention
        self.pattern_attention = ComplexPatternAttention(hidden_dim // 2, n_patterns)
        
        self.to_penultimate = ComplexLinear(hidden_dim, hidden_dim)

    def apply_self_processing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply paradox transformation - same as original"""
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        return hidden

    def forward(self, x: torch.Tensor, next_layer: Optional['DiscretePatternLayer'] = None, 
                layer_idx: int = 0) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Apply paradox transformation, pattern attention, and routing"""
        hidden = self.apply_self_processing(x)
        
        if next_layer is not None:
            # Compress own activity with complex attention
            my_compressed, my_patterns = self.pattern_attention(hidden)
            predicted_next = my_compressed
            
            # Get actual next layer transformation
            with torch.no_grad():
                actual_next = next_layer.apply_self_processing(hidden)
                compressed_next, _ = next_layer.pattern_attention(actual_next)
            
            # Match dimensions for prediction error
            if predicted_next.shape != compressed_next.shape:
                min_dim = min(predicted_next.shape[-1], compressed_next.shape[-1])
                if predicted_next.dim() == 3:
                    predicted_next = predicted_next[..., :min_dim]
                    compressed_next = compressed_next[..., :min_dim]
                else:
                    predicted_next = predicted_next[:, :min_dim]
                    compressed_next = compressed_next[:, :min_dim]
            
            # Complex prediction error (use magnitude for routing)
            complex_error = compressed_next - predicted_next
            pred_error = torch.mean(complex_error.abs()**2, dim=-1, keepdim=True)
            
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
            
            return continue_up, penultimate_contribution, pred_error
        
        else:
            # No next layer - just return penultimate contribution
            compressed, _ = self.pattern_attention(hidden)
            penultimate_contribution = self.to_penultimate(compressed)
            return None, penultimate_contribution, None

class PenultimatePatternLayer(nn.Module):
    """Penultimate layer with working complex attention."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_patterns=8):
        super().__init__()
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # Replace vestigial patterns with working complex attention
        self.pattern_attention = ComplexPatternAttention(hidden_dim // 2, n_patterns)
        
        # Final output predictor is real-valued
        self.output_predictor = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply paradox transformation then pattern attention"""
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        
        # Apply complex attention to compress
        compressed, attention_weights = self.pattern_attention(hidden)
        
        # Final output (real-valued)
        predicted_output = self.output_predictor(compressed.real)
        return predicted_output

class ParadoxNetComplex(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        self.hidden_layers = nn.ModuleList()
        current_dim = embedding_dim
        for h_dim in hidden_dims:
            self.hidden_layers.append(DiscretePatternLayer(current_dim, h_dim, n_patterns))
            current_dim = h_dim
            
        self.penultimate_layer = PenultimatePatternLayer(hidden_dims[-1], hidden_dims[-1], vocab_size, n_patterns)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with routing and prediction error collection"""
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

        penultimate_contributions = []
        prediction_errors = []
        
        # Process through hidden layers with routing
        for i, layer in enumerate(self.hidden_layers):
            next_layer = self.hidden_layers[i+1] if i+1 < len(self.hidden_layers) else None
            
            continue_up, penultimate_contrib, pred_error = layer(
                current_seq, next_layer, layer_idx=i
            )
            
            if penultimate_contrib is not None:
                penultimate_contributions.append(penultimate_contrib.mean(dim=1))
            
            if pred_error is not None:
                prediction_errors.append(pred_error.mean())
            
            # Update current sequence for next layer
            if continue_up is not None:
                current_seq = continue_up
            else:
                current_seq = layer.apply_self_processing(current_seq)
        
        # Consensus view + recursive residual
        consensus_view = torch.sum(torch.stack(penultimate_contributions), dim=0)
        recursive_residual = current_seq.mean(dim=1)
        penultimate_input = consensus_view + recursive_residual
        
        final_output = self.penultimate_layer(penultimate_input)
        
        # Combine prediction errors for loss
        combined_pred_errors = None
        if prediction_errors:
            combined_pred_errors = torch.stack(prediction_errors)
        
        return final_output, combined_pred_errors

# Test the updated version
if __name__ == "__main__":
    print("Testing complex paradox net with working complex attention...")
    
    model = ParadoxNetComplex(vocab_size=50, embedding_dim=32, hidden_dims=[32, 32], n_patterns=8)
    x = torch.randint(0, 50, (4, 20))
    
    try:
        output, pred_errors = model(x)
        print(f"✅ Works! Output shape: {output.shape}")
        print(f"✅ Prediction errors shape: {pred_errors.shape if pred_errors is not None else None}")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test loss computation with prediction errors
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        targets = torch.randint(0, 50, (4,))
        
        task_loss = criterion(output, targets)
        total_loss = task_loss
        
        if pred_errors is not None:
            pred_loss = 0.1 * torch.mean(pred_errors)
            total_loss = task_loss + pred_loss
            print(f"✅ Task loss: {task_loss.item():.4f}, Pred loss: {pred_loss.item():.6f}")
            print(f"✅ Prediction errors: {pred_errors.detach().numpy()}")
        
        # Test gradient flow
        total_loss.backward()
        print("✅ Gradients flow correctly!")
        
        print("\n🎉 Complex attention with routing successfully integrated!")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


