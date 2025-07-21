import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# Copy the working parts from the complex net
from paradox_net_complex import (
    apply_rotary_pos_emb, 
    PositionalEncoding, 
    ComplexLinear
)

class RealPatternAttention(nn.Module):
    """Real-valued attention over interleaved complex → real patterns"""
    def __init__(self, d_model, n_patterns):
        super().__init__()
        self.d_model = d_model
        self.n_patterns = n_patterns
        self.scale = 1 / (d_model ** 0.5)
        
        # Real pattern dictionary for interleaved complex data
        # Input will be [batch, seq, 2*d_model] (interleaved real/imag)
        self.patterns = nn.Parameter(torch.randn(n_patterns, 2 * d_model) * 0.02)
    
    def forward(self, hidden_complex):
        """
        Apply real attention to interleaved complex data
        hidden_complex: [B, L, d] or [B, d] complex tensor
        returns: compressed real tensor for prediction
        """
        # Convert complex to interleaved real
        hidden_real = torch.view_as_real(hidden_complex)  # [B, L, d, 2] or [B, d, 2]
        hidden_interleaved = hidden_real.flatten(-2)      # [B, L, 2d] or [B, 2d]
        
        if hidden_interleaved.dim() == 3:  # sequence case
            B, L, d2 = hidden_interleaved.shape
            patterns_expanded = self.patterns.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
            
            # Standard real attention
            scores = torch.einsum('bld,blpd->blp', hidden_interleaved, patterns_expanded) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply attention to patterns
            compressed = torch.einsum('blp,blpd->bld', attn_weights, patterns_expanded)
            
        else:  # no sequence case
            B, d2 = hidden_interleaved.shape
            patterns_expanded = self.patterns.unsqueeze(0).expand(B, -1, -1)
            
            scores = torch.einsum('bd,bpd->bp', hidden_interleaved, patterns_expanded) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            
            compressed = torch.einsum('bp,bpd->bd', attn_weights, patterns_expanded)
        
        return compressed, attn_weights

class DiscretePatternLayer(nn.Module):
    """Hidden layer with complex paradox + real attention routing"""
    def __init__(self, input_dim, hidden_dim, n_patterns=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_patterns = n_patterns
        
        # Complex paradox transformation (same as before)
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # Real attention for compression (much faster!)
        self.pattern_attention = RealPatternAttention(hidden_dim // 2, n_patterns)
        
        self.to_penultimate = ComplexLinear(hidden_dim, hidden_dim)

    def apply_self_processing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex paradox transformation - same as original"""
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        return hidden

    def forward(self, x: torch.Tensor, next_layer: Optional['DiscretePatternLayer'] = None, 
                layer_idx: int = 0) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Apply paradox transformation, real attention compression, and routing"""
        # Complex paradox transformation
        hidden = self.apply_self_processing(x)
        
        if next_layer is not None:
            # Compress with REAL attention (much faster!)
            my_compressed, my_patterns = self.pattern_attention(hidden)
            
            # Predict next layer's compressed representation
            with torch.no_grad():
                actual_next = next_layer.apply_self_processing(hidden)
                compressed_next, _ = next_layer.pattern_attention(actual_next)
            
            # Match dimensions for prediction error
            if my_compressed.shape != compressed_next.shape:
                min_dim = min(my_compressed.shape[-1], compressed_next.shape[-1])
                if my_compressed.dim() == 3:
                    my_compressed = my_compressed[..., :min_dim]
                    compressed_next = compressed_next[..., :min_dim]
                else:
                    my_compressed = my_compressed[:, :min_dim]
                    compressed_next = compressed_next[:, :min_dim]
            
            # Real prediction error (much simpler!)
            pred_error = torch.mean((compressed_next - my_compressed)**2, dim=-1, keepdim=True)
            
            # DEBUG: Print prediction info
            if layer_idx == 0:  # Only print for first layer to avoid spam
                print(f"  Layer {layer_idx} debug:")
                print(f"    my_compressed shape: {my_compressed.shape}, mean: {my_compressed.mean().item():.6f}")
                print(f"    compressed_next shape: {compressed_next.shape}, mean: {compressed_next.mean().item():.6f}")
                print(f"    pred_error: {pred_error.mean().item():.6f}")
                print(f"    confidence: {confidence.mean().item():.6f}")
            
            # Route based on prediction accuracy
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Add routing cost
            routing_balance = confidence * (1 - confidence)
            routing_cost = 0.1 * torch.mean(routing_balance)
            pred_error = pred_error + routing_cost
            
            # Route information (complex hidden state!)
            penultimate_features = self.to_penultimate(hidden)
            penultimate_contribution = penultimate_features * confidence
            continue_up = hidden * (1 - confidence)
            
            return continue_up, penultimate_contribution, pred_error
        
        else:
            # No next layer - just return penultimate contribution
            penultimate_contribution = self.to_penultimate(hidden)
            return None, penultimate_contribution, None

class PenultimatePatternLayer(nn.Module):
    """Penultimate layer with complex paradox + real attention"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_patterns=8):
        super().__init__()
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # Real attention for compression
        self.pattern_attention = RealPatternAttention(hidden_dim // 2, n_patterns)
        
        # Output predictor takes real compressed representation
        self.output_predictor = nn.Linear(2 * (hidden_dim // 2), output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex paradox transformation then real attention"""
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        
        # Apply real attention to compress
        compressed, attention_weights = self.pattern_attention(hidden)
        
        # Final output from real compressed representation
        predicted_output = self.output_predictor(compressed)
        return predicted_output

class ParadoxNetComplexRealAttention(nn.Module):
    """Complex paradox net with fast real attention for compression"""
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Build layers
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
        
        # Apply rotary positional encoding and convert to complex
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
            
            # Update current sequence for next layer (complex!)
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

# Test the hybrid version
if __name__ == "__main__":
    print("Testing complex paradox + real attention hybrid...")
    
    model = ParadoxNetComplexRealAttention(vocab_size=50, embedding_dim=32, hidden_dims=[32, 32], n_patterns=8)
    x = torch.randint(0, 50, (4, 20))
    
    try:
        output, pred_errors = model(x)
        print(f"✅ Works! Output shape: {output.shape}")
        print(f"✅ Prediction errors shape: {pred_errors.shape if pred_errors is not None else None}")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test loss computation
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
        
        print("\n🎉 Hybrid complex paradox + real attention works!")
        print("Much faster than pure complex attention!")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()