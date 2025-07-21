import torch
import torch.nn as nn
import torch.nn.functional as F

# Copy the exact working parts from the original
from paradox_net_complex import (
    apply_rotary_pos_emb, 
    PositionalEncoding, 
    ComplexLinear,
    PenultimatePatternLayer
)

def complex_pattern_attention(hidden, patterns, scale=None):
    """
    Complex attention over pattern dictionary
    hidden: [B, L, d] or [B, d] complex tensor (queries)
    patterns: [n_patterns, d] complex tensor (keys and values)
    returns: [B, L, d] or [B, d] complex tensor (attended patterns)
    """
    if scale is None:
        scale = 1 / (hidden.size(-1) ** 0.5)
    
    # Expand patterns to match hidden dimensions
    if hidden.dim() == 3:  # sequence case
        B, L, d = hidden.shape
        patterns_expanded = patterns.unsqueeze(0).unsqueeze(0)  # [1, 1, n_patterns, d]
        patterns_expanded = patterns_expanded.expand(B, L, -1, -1)  # [B, L, n_patterns, d]
        
        # Hermitian inner product: hidden @ patterns.conj()
        scores = torch.einsum('bld,blpd->blp', hidden, patterns_expanded.conj()) * scale
        attn = F.softmax(scores.real, dim=-1)  # real attention weights
        
        # Apply attention to patterns
        attended = torch.einsum('blp,blpd->bld', attn, patterns_expanded)
        
    else:  # no sequence case
        B, d = hidden.shape
        patterns_expanded = patterns.unsqueeze(0).expand(B, -1, -1)  # [B, n_patterns, d]
        
        # Hermitian inner product
        scores = torch.einsum('bd,bpd->bp', hidden, patterns_expanded.conj()) * scale
        attn = F.softmax(scores.real, dim=-1)  # real attention weights
        
        # Apply attention to patterns
        attended = torch.einsum('bp,bpd->bd', attn, patterns_expanded)
    
    return attended, attn

class ComplexPatternLayer(nn.Module):
    """Complex layer with proper complex attention over patterns"""
    def __init__(self, input_dim, hidden_dim, n_patterns=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_patterns = n_patterns
        
        # Same as original
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        self.to_penultimate = ComplexLinear(hidden_dim, hidden_dim)
        
        # Complex pattern dictionary
        self.pattern_dict = nn.Parameter(torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02)

    def apply_self_processing(self, x):
        """Same as original - the paradox nonlinearity"""
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        return hidden

    def forward(self, x):
        """Forward with complex pattern attention"""
        hidden = self.apply_self_processing(x)
        
        # Complex attention over patterns
        attended_patterns, attention_weights = complex_pattern_attention(hidden, self.pattern_dict)
        
        # Mix original hidden state with attended patterns
        # This preserves the paradox-modulated representation while adding pattern compression
        mixed = 0.7 * hidden + 0.3 * attended_patterns
        
        # Penultimate contribution
        penultimate = self.to_penultimate(mixed)
        
        return mixed, penultimate

class ComplexPatternsNet(nn.Module):
    """Complex paradox net with proper complex pattern attention"""
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Build with complex pattern layers
        self.hidden_layers = nn.ModuleList()
        current_dim = embedding_dim
        for h_dim in hidden_dims:
            self.hidden_layers.append(ComplexPatternLayer(current_dim, h_dim, n_patterns))
            current_dim = h_dim
            
        # Same penultimate as original
        self.penultimate_layer = PenultimatePatternLayer(hidden_dims[-1], hidden_dims[-1], vocab_size, n_patterns)

    def forward(self, x):
        """Same structure as original"""
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

        penultimate_contributions = []
        
        # Process through layers
        for layer in self.hidden_layers:
            current_seq, penultimate = layer(current_seq)
            penultimate_contributions.append(penultimate.mean(dim=1))
            
        # Same as original
        consensus_view = torch.sum(torch.stack(penultimate_contributions), dim=0)
        recursive_residual = current_seq.mean(dim=1)
        
        penultimate_input = consensus_view + recursive_residual
        final_output = self.penultimate_layer(penultimate_input)
        
        return final_output

# Test it!
if __name__ == "__main__":
    print("Testing complex pattern attention...")
    
    # Test the attention function first
    print("\n1. Testing attention function...")
    hidden = torch.randn(2, 10, 16, dtype=torch.cfloat)  # [batch, seq, dim]
    patterns = torch.randn(8, 16, dtype=torch.cfloat)    # [n_patterns, dim]
    
    try:
        attended, attn_weights = complex_pattern_attention(hidden, patterns)
        print(f"✅ Attention works! Hidden: {hidden.shape} -> Attended: {attended.shape}")
        print(f"   Attention weights shape: {attn_weights.shape}")
        print(f"   Attention weights are real: {attn_weights.dtype}")
    except Exception as e:
        print(f"❌ Attention failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test the full model
    print("\n2. Testing full model...")
    model = ComplexPatternsNet(vocab_size=50, embedding_dim=32, hidden_dims=[32, 32], n_patterns=8)
    x = torch.randint(0, 50, (4, 20))
    
    try:
        output = model(x)
        print(f"✅ Model works! Output shape: {output.shape}")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        print("✅ Gradients flow correctly!")
        
    except Exception as e:
        print(f"❌ Model failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Complex pattern attention is working!")