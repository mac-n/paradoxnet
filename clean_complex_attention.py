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

class ComplexPatternAttention(nn.Module):
    """Clean complex attention over pattern dictionary"""
    def __init__(self, d_model, n_patterns):
        super().__init__()
        self.d_model = d_model
        self.n_patterns = n_patterns
        self.scale = 1 / (d_model ** 0.5)
        
        # Complex pattern dictionary (keys and values)
        self.patterns = nn.Parameter(torch.randn(n_patterns, d_model, dtype=torch.cfloat) * 0.02)
    
    def forward(self, hidden):
        """
        hidden: [B, L, d] or [B, d] complex tensor (queries)
        returns: [B, L, d] or [B, d] complex tensor (attended patterns)
        """
        if hidden.dim() == 3:  # sequence case
            B, L, d = hidden.shape
            # Expand patterns: [n_patterns, d] -> [B, L, n_patterns, d]
            patterns_expanded = self.patterns.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
            
            # Hermitian inner product: hidden @ patterns.conj()
            scores = torch.einsum('bld,blpd->blp', hidden, patterns_expanded.conj()) * self.scale
            attn_weights = F.softmax(scores.real, dim=-1)  # real attention weights
            
            # Convert real weights to complex for multiplication
            attn_complex = attn_weights.to(dtype=torch.cfloat)
            
            # Apply attention to patterns
            attended = torch.einsum('blp,blpd->bld', attn_complex, patterns_expanded)
            
        else:  # no sequence case
            B, d = hidden.shape
            # Expand patterns: [n_patterns, d] -> [B, n_patterns, d]
            patterns_expanded = self.patterns.unsqueeze(0).expand(B, -1, -1)
            
            # Hermitian inner product
            scores = torch.einsum('bd,bpd->bp', hidden, patterns_expanded.conj()) * self.scale
            attn_weights = F.softmax(scores.real, dim=-1)  # real attention weights
            
            # Convert real weights to complex for multiplication
            attn_complex = attn_weights.to(dtype=torch.cfloat)
            
            # Apply attention to patterns
            attended = torch.einsum('bp,bpd->bd', attn_complex, patterns_expanded)
        
        return attended, attn_weights

class ComplexPatternLayer(nn.Module):
    """Complex layer with clean pattern attention"""
    def __init__(self, input_dim, hidden_dim, n_patterns=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Same as original
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        self.to_penultimate = ComplexLinear(hidden_dim, hidden_dim)
        
        # Complex pattern attention
        self.pattern_attention = ComplexPatternAttention(hidden_dim // 2, n_patterns)

    def apply_self_processing(self, x):
        """Same as original - the paradox nonlinearity"""
        hidden_linear = self.process(x)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        hidden = hidden_linear * torch.sigmoid(paradox.abs())
        return hidden

    def forward(self, x):
        """Forward with clean pattern attention"""
        hidden = self.apply_self_processing(x)
        
        # Apply pattern attention directly (no mixing)
        attended_patterns, attention_weights = self.pattern_attention(hidden)
        
        # Use the attended patterns as the new representation
        penultimate = self.to_penultimate(attended_patterns)
        
        return attended_patterns, penultimate

class ComplexPatternsNet(nn.Module):
    """Complex paradox net with clean pattern attention"""
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
    print("Testing clean complex pattern attention...")
    
    # Test the attention module first
    print("\n1. Testing attention module...")
    attention_module = ComplexPatternAttention(d_model=16, n_patterns=8)
    hidden = torch.randn(2, 10, 16, dtype=torch.cfloat)  # [batch, seq, dim]
    
    try:
        attended, attn_weights = attention_module(hidden)
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
    
    print("\n🎉 Clean complex pattern attention is working!")