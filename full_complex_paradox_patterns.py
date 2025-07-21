import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

# Copy the exact working parts from the original
from paradox_net_complex import (
    apply_rotary_pos_emb, 
    PositionalEncoding, 
    ComplexLinear
)

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

class FullComplexPatternLayer(nn.Module):
    """Full pipeline: paradox + own patterns + next patterns + routing"""
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        
        # 1. Paradox transformation components
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # 2. Own pattern dictionary (compress current activity)
        self.own_patterns = ComplexPatternAttention(hidden_dim // 2, n_patterns)
        
        # 3. Next pattern dictionary (predict next layer's compressed activity)
        self.next_patterns = ComplexPatternAttention(next_dim // 2, n_patterns)
        
        # 4. Output pathway
        self.to_penultimate = ComplexLinear(hidden_dim, penultimate_dim)
        
        # Stats tracking
        self.last_stats: Optional[LayerStats] = None
        self.last_paradox_magnitude: float = 0.0
        self.last_entropy: float = 0.0

    def apply_self_paradox_nonlinearity(self, x):
        """Step 1: Apply self-prediction paradox transformation"""
        # Pure linear transformation first
        hidden_linear = self.process(x)
        
        # Predict what I should become
        self_prediction = self.self_predictor(hidden_linear)
        
        # The paradox: difference between prediction and current state
        paradox = self_prediction - hidden_linear
        
        # Track paradox magnitude for statistics
        with torch.no_grad():
            self.last_paradox_magnitude = torch.mean(paradox.abs()).item()
        
        # Use paradox magnitude as nonlinearity
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        
        return hidden

    def compress_activity(self, hidden, is_next_layer=False):
        """Step 2: Compress activity using appropriate pattern dictionary"""
        if is_next_layer:
            compressed, pattern_weights = self.next_patterns(hidden)
        else:
            compressed, pattern_weights = self.own_patterns(hidden)
        
        # Track entropy
        with torch.no_grad():
            entropy = -torch.sum(pattern_weights * torch.log(pattern_weights + 1e-8), dim=-1)
            self.last_entropy = entropy.mean().item()
        
        return compressed, pattern_weights

    def forward(self, x, next_layer: Optional['FullComplexPatternLayer'] = None, 
                layer_idx: int = 0) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Full pipeline with routing"""
        
        # Step 1: Apply paradox transformation
        hidden = self.apply_self_paradox_nonlinearity(x)
        
        if next_layer is not None:
            # Step 2: Compress own activity and predict next layer
            my_compressed, my_patterns = self.compress_activity(hidden, is_next_layer=False)
            predicted_next = my_compressed
            
            # Step 3: Get actual next layer result
            with torch.no_grad():
                actual_next = next_layer.apply_self_paradox_nonlinearity(hidden)
                compressed_next, _ = next_layer.compress_activity(actual_next, is_next_layer=True)
            
            # Step 4: Match dimensions and calculate prediction error
            if predicted_next.shape != compressed_next.shape:
                min_dim = min(predicted_next.shape[-1], compressed_next.shape[-1])
                if predicted_next.dim() == 3:
                    predicted_next = predicted_next[..., :min_dim]
                    compressed_next = compressed_next[..., :min_dim]
                else:
                    predicted_next = predicted_next[:, :min_dim]
                    compressed_next = compressed_next[:, :min_dim]
            
            # Complex prediction error (use magnitude for confidence)
            complex_error = compressed_next - predicted_next
            pred_error = torch.mean(complex_error.abs()**2, dim=-1, keepdim=True)
            
            # Step 5: Calculate routing confidence
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Step 6: Route based on confidence
            penultimate_features = self.to_penultimate(hidden)
            penultimate_contribution = penultimate_features * confidence
            continue_up = hidden * (1 - confidence)
            
            # Track statistics
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach().abs(), dim=-1)),
                continue_magnitude=torch.mean(torch.norm(continue_up.detach().abs(), dim=-1)),
                layer_idx=layer_idx,
                pattern_usage=my_patterns.detach().mean(0),
                pattern_entropy=self.last_entropy,
                self_paradox_magnitude=self.last_paradox_magnitude
            )
            
            return continue_up, penultimate_contribution, pred_error
        
        else:
            # No next layer - just return penultimate contribution
            penultimate_contribution = self.to_penultimate(hidden)
            return None, penultimate_contribution, None

class ComplexPenultimateLayer(nn.Module):
    """Penultimate layer for final output"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_patterns=8):
        super().__init__()
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        self.patterns = ComplexPatternAttention(hidden_dim // 2, n_patterns)
        
        # Final output predictor is real-valued
        self.output_predictor = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        # Paradox transformation
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        
        # Pattern compression
        compressed, _ = self.patterns(hidden)
        
        # Final output (real-valued)
        predicted_output = self.output_predictor(compressed.real)
        return predicted_output

class ComplexParadoxPatternsNet(nn.Module):
    """Full complex paradox net with pattern routing"""
    def __init__(self, vocab_size, embedding_dim, hidden_dims, penultimate_dim, n_patterns=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Build layers with proper dimensions for routing
        self.hidden_layers = nn.ModuleList()
        current_dim = embedding_dim
        
        for i, h_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i+1] if i+1 < len(hidden_dims) else h_dim
            layer = FullComplexPatternLayer(
                input_dim=current_dim,
                hidden_dim=h_dim,
                next_dim=next_dim,
                penultimate_dim=penultimate_dim,
                n_patterns=n_patterns
            )
            self.hidden_layers.append(layer)
            current_dim = h_dim
        
        self.penultimate_layer = ComplexPenultimateLayer(
            input_dim=penultimate_dim,
            hidden_dim=hidden_dims[-1],
            output_dim=vocab_size,
            n_patterns=n_patterns
        )

    def forward(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with prediction error collection"""
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        # Apply rotary positional encoding and convert to complex
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

        prediction_errors = []
        
        # Process through hidden layers with routing - NO CONSENSUS VIEW
        for i, layer in enumerate(self.hidden_layers):
            next_layer = self.hidden_layers[i+1] if i+1 < len(self.hidden_layers) else None
            
            continue_up, penultimate_contrib, pred_error = layer(
                current_seq, next_layer, layer_idx=i
            )
            
            # Just collect prediction errors, ignore penultimate contributions
            if pred_error is not None:
                error_val = pred_error.mean()
                prediction_errors.append(error_val)
                print(f"  Layer {i} prediction error: {error_val.item():.6f}")
            
            # Update current sequence for next layer
            if continue_up is not None:
                current_seq = continue_up
            else:
                current_seq = layer.apply_self_paradox_nonlinearity(current_seq)
        
        # Final layer processes whatever made it through the routing chain
        final_hidden = current_seq.mean(dim=1)  # Just the final routed information
        final_output = self.penultimate_layer(final_hidden)
        
        # Combine prediction errors for loss
        combined_pred_errors = None
        if prediction_errors:
            combined_pred_errors = torch.stack(prediction_errors)
        
        return final_output, combined_pred_errors

def create_pattern_data(vocab_size=10, seq_length=5, n_samples=100):
    """Create simple repeating pattern: [1,2,3,4,5,1,2,3,4,5,...]"""
    pattern = list(range(1, vocab_size))
    sequences = []
    targets = []
    
    for i in range(n_samples):
        start_idx = i % len(pattern)
        seq = []
        for j in range(seq_length):
            seq.append(pattern[(start_idx + j) % len(pattern)])
        target = pattern[(start_idx + seq_length) % len(pattern)]
        sequences.append(seq)
        targets.append(target)
    
    return torch.tensor(sequences), torch.tensor(targets)

# Test the full pipeline!
if __name__ == "__main__":
    print("Testing full complex paradox + patterns pipeline...")
    
    print("\n=== Test 1: Random Data (current) ===")
    model1 = ComplexParadoxPatternsNet(
        vocab_size=50, 
        embedding_dim=32, 
        hidden_dims=[32, 32], 
        penultimate_dim=32, 
        n_patterns=8
    )
    
    x_random = torch.randint(0, 50, (4, 20))
    targets_random = torch.randint(0, 50, (4,))
    
    try:
        output1, pred_errors1 = model1(x_random)
        print(f"✅ Random data works! Output shape: {output1.shape}")
        if pred_errors1 is not None:
            print(f"✅ Random prediction errors: {pred_errors1.detach().numpy()}")
        
    except Exception as e:
        print(f"❌ Random data failed: {e}")
    
    print("\n=== Test 2: Structured Pattern Data ===")
    model2 = ComplexParadoxPatternsNet(
        vocab_size=10, 
        embedding_dim=32, 
        hidden_dims=[32, 32], 
        penultimate_dim=32, 
        n_patterns=8
    )
    
    # Create structured pattern data
    x_pattern, targets_pattern = create_pattern_data(vocab_size=10, seq_length=5, n_samples=8)
    print(f"Pattern data examples:")
    print(f"  Sequences: {x_pattern[:3]}")
    print(f"  Targets: {targets_pattern[:3]}")
    
    try:
        output2, pred_errors2 = model2(x_pattern)
        print(f"✅ Pattern data works! Output shape: {output2.shape}")
        
        # Test loss computation
        criterion = nn.CrossEntropyLoss()
        task_loss = criterion(output2, targets_pattern)
        
        if pred_errors2 is not None:
            pred_loss = 0.1 * torch.mean(pred_errors2)
            total_loss = task_loss + pred_loss
            print(f"✅ Task loss: {task_loss.item():.4f}, Pred loss: {pred_loss.item():.8f}")
            print(f"✅ Pattern prediction errors: {pred_errors2.detach().numpy()}")
            print(f"✅ Mean prediction error: {torch.mean(pred_errors2).item():.8f}")
        else:
            print("⚠️  No prediction errors generated")
        
        # Test backprop
        total_loss.backward()
        print("✅ Gradients flow correctly!")
        
    except Exception as e:
        print(f"❌ Pattern data failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Testing complete!")