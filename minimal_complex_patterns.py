import torch
import torch.nn as nn
import torch.nn.functional as F

# Copy the exact working parts from the original
from paradox_net_complex import (
    apply_rotary_pos_emb, 
    PositionalEncoding, 
    ComplexLinear,
    ParadoxNetComplex
)

class SimplePatternLayer(nn.Module):
    """Just add basic pattern compression to the working complex layer"""
    def __init__(self, input_dim, hidden_dim, n_patterns=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_patterns = n_patterns
        
        # Same as original
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        self.to_penultimate = ComplexLinear(hidden_dim, hidden_dim)
        
        # Simple pattern addition
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02)
        self.pattern_weights = nn.Linear(hidden_dim // 2, n_patterns)  # Real-valued weights

    def apply_self_processing(self, x):
        """Same as original"""
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        return hidden

    def forward(self, x):
        """Same structure as original, just add pattern compression"""
        hidden = self.apply_self_processing(x)
        
        # Simple pattern compression - use real part for attention
        attention_input = hidden.real if hidden.dim() == 3 else hidden.real
        pattern_scores = self.pattern_weights(attention_input)
        pattern_probs = F.softmax(pattern_scores, dim=-1)
        
        # Simple weighted sum - just use @ for matrix multiply
        if hidden.dim() == 3:  # sequence
            batch_size, seq_len, _ = hidden.shape
            # Reshape and multiply
            pattern_probs_flat = pattern_probs.view(-1, self.n_patterns)
            patterns_flat = self.pattern_dict.view(self.n_patterns, -1)
            compressed_flat = pattern_probs_flat @ patterns_flat
            compressed = compressed_flat.view(batch_size, seq_len, -1)
        else:  # no sequence
            compressed = pattern_probs @ self.pattern_dict.view(self.n_patterns, -1)
        
        # Mix original and compressed
        mixed = 0.5 * hidden + 0.5 * compressed
        penultimate = self.to_penultimate(mixed)
        
        return mixed, penultimate

class MinimalComplexPatterns(nn.Module):
    """Minimal version - just add patterns to working complex net"""
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Build with simple pattern layers
        self.hidden_layers = nn.ModuleList()
        current_dim = embedding_dim
        for h_dim in hidden_dims:
            self.hidden_layers.append(SimplePatternLayer(current_dim, h_dim, n_patterns))
            current_dim = h_dim
            
        # Same penultimate as original
        from paradox_net_complex import PenultimatePatternLayer
        self.penultimate_layer = PenultimatePatternLayer(hidden_dims[-1], hidden_dims[-1], vocab_size, n_patterns)

    def forward(self, x):
        """Same as original structure"""
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

        penultimate_contributions = []
        
        # Process layers
        for layer in self.hidden_layers:
            current_seq, penultimate = layer(current_seq)
            penultimate_contributions.append(penultimate.mean(dim=1))
            
        # Same as original
        consensus_view = torch.sum(torch.stack(penultimate_contributions), dim=0)
        recursive_residual = current_seq.mean(dim=1)
        
        penultimate_input = consensus_view + recursive_residual
        final_output = self.penultimate_layer(penultimate_input)
        
        return final_output

# Simple test
if __name__ == "__main__":
    model = MinimalComplexPatterns(vocab_size=50, embedding_dim=32, hidden_dims=[32, 32])
    x = torch.randint(0, 50, (4, 20))
    
    print("Testing minimal version...")
    try:
        output = model(x)
        print(f"✅ Works! Output shape: {output.shape}")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()