import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Copy the working temporal entropy recycling and just change the final output
from entropy_recycling_temporal import TemporalEntropyRecyclingNet, TemporalEntropyRecyclingLayer, LayerStats, PositionalEncoding

class SequenceOutputTemporalNet(TemporalEntropyRecyclingNet):
    """Simple modification: process sequences but predict all positions"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8, max_seq_len=20):
        super().__init__(vocab_size, embedding_dim, hidden_dims, n_patterns)
        self.max_seq_len = max_seq_len
        
        # Replace final layer to handle sequences
        self.final = nn.Linear(hidden_dims[-1], vocab_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass with sequence output - predict next token for each position"""
        # Embedding + positional encoding
        embedded = self.embedding(x)  # [batch, seq, embed_dim]
        embedded = self.pos_encoder(embedded)  # Add positional encoding
        batch_size, seq_len, embed_dim = embedded.shape
        
        # Process each position through the temporal entropy recycling layers
        outputs = []
        
        for pos in range(seq_len):
            # Get embedding for this position
            pos_embedded = embedded[:, pos, :]  # [batch, embed_dim]
            
            # Process through temporal entropy recycling layers (unchanged)
            penultimate_contributions = []
            current = pos_embedded
            all_errors = []
            all_entropy = []
            
            # Track temporal prediction error
            temporal_prediction_error = None
            
            # Single pass: process all layers with Layer 0 using previous epoch's entropy
            for i, layer in enumerate(self.layers):
                next_layer = self.layers[i+1] if i < len(self.layers)-1 else None
                
                if i == 0:
                    # Layer 0: Use entropy from previous epoch (temporal recycling)
                    current, penultimate, error, entropy = layer(current, next_layer, i, temporal_entropy=self.previous_entropy)
                else:
                    current, penultimate, error, entropy = layer(current, next_layer, i)
                
                if error is not None:
                    all_errors.append(error)
                penultimate_contributions.append(penultimate)
                
                # Collect entropy for NEXT epoch (except from layer 0)
                if i > 0:
                    all_entropy.append(entropy)
            
            # Prepare entropy for next epoch AND compute temporal prediction error (only once)
            if pos == 0:  # Only do this once per batch, not per position
                if all_entropy:
                    # Sum all entropy for next epoch (real space - simpler)
                    total_entropy = torch.stack(all_entropy).sum(dim=0)
                    
                    # TEMPORAL INPUT PREDICTION: Compare Layer 0's prediction to actual entropy
                    if self.previous_entropy is not None:
                        layer_0 = self.layers[0]
                        if hasattr(layer_0, 'temporal_entropy_predictor'):
                            # Layer 0 predicts what optimal entropy processing should look like
                            temporal_prediction = layer_0.temporal_entropy_predictor(pos_embedded)
                            # Prediction error: predicted optimal processing vs actual accumulated entropy
                            temporal_prediction_error = F.mse_loss(temporal_prediction, total_entropy)
                    
                    # Store for next epoch (detach to avoid gradient accumulation)
                    self.previous_entropy = total_entropy.detach()
                else:
                    self.previous_entropy = None
                    temporal_prediction_error = None
            
            # Combine penultimate contributions for this position
            penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
            
            # Final output for this position
            pos_output = self.final(penultimate)
            outputs.append(pos_output)
        
        # Stack outputs: [seq_len, batch, vocab_size] -> [batch, seq_len, vocab_size]
        sequence_output = torch.stack(outputs, dim=1)
        
        return sequence_output, torch.cat(all_errors, dim=1) if all_errors else None, temporal_prediction_error

# Factory function
def create_sequence_simple_temporal_net(sequence_length=20, hidden_dims=[64, 64, 64], n_patterns=8):
    """Create simple sequence temporal entropy recycling version"""
    return SequenceOutputTemporalNet(
        vocab_size=128,  # Will be set by experiment
        embedding_dim=64,
        hidden_dims=hidden_dims,
        n_patterns=n_patterns,
        max_seq_len=sequence_length
    )

if __name__ == "__main__":
    print("📚 TESTING SIMPLE SEQUENCE TEMPORAL ENTROPY RECYCLING 📚")
    
    # Create network
    net = create_sequence_simple_temporal_net()
    
    # Test data
    x = torch.randint(0, 57, (5, 10))  # Smaller test: 5 batches, 10 sequence length
    
    # Forward pass
    output, errors, temporal_error = net(x)
    
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Errors: {errors.shape if errors is not None else None}")
    print(f"Temporal error: {temporal_error.item() if temporal_error is not None else None}")
    
    # Check entropy statistics
    print(f"\n=== SIMPLE SEQUENCE TEMPORAL ENTROPY STATISTICS ===")
    for i, layer in enumerate(net.layers):
        if layer.last_stats:
            stats = layer.last_stats
            print(f"Layer {i}:")
            print(f"  Entropy magnitude: {stats.entropy_magnitude:.3f}")
            print(f"  Temporal entropy magnitude: {stats.temporal_entropy_magnitude:.3f}")
            print(f"  Composition alpha: {stats.composition_alpha:.3f}")
    
    print(f"\n✅ Simple sequence temporal entropy recycling working!")