import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import json
from dataclasses import dataclass
from typing import List, Optional
from torch.utils.data import DataLoader, TensorDataset

from data_generators import get_tiny_shakespeare_data
from temporal_symmetry_layer import TemporalSymmetryLayer

# --- RoPE Functions (same as your other experiments) ---
def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embedding to the input tensor."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0)
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

@dataclass
class TemporalSymmetryConfig:
    """Configuration for temporal symmetry experiment"""
    embedding_dim: int = 64
    hidden_dims: List[int] = None
    n_patterns: int = 8
    sequence_length: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64, 64]

class TemporalSymmetryNet(nn.Module):
    """Full network with temporal symmetry: past entropy + future pattern prediction"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Store entropy from previous epoch (for bottom layer)
        self.previous_entropy = None
        
        # Create layers with temporal symmetry
        self.layers = nn.ModuleList()
        current_dim = embedding_dim * 2  # Account for complex interleaving (real + imaginary)
        
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else hidden_dim
            
            layer = TemporalSymmetryLayer(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                next_dim=next_dim,
                penultimate_dim=hidden_dim,
                n_patterns=n_patterns,
                is_bottom=(i == 0),  # First layer handles past entropy
                is_top=(i == len(hidden_dims) - 1)  # Last layer predicts future
            )
            self.layers.append(layer)
            current_dim = hidden_dim
        
        # Final output
        self.final = nn.Linear(hidden_dims[-1], vocab_size)
    
    def forward(self, x: torch.Tensor):
        """Forward pass with temporal symmetry"""
        batch_size, seq_len = x.shape
        
        # Embedding + RoPE
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))
        
        # Mean pool and convert back to real space with interleaving (like your baseline)
        current_complex = current_seq.mean(dim=1)
        current = torch.view_as_real(current_complex).flatten(start_dim=-1)
        
        # Process through temporal symmetry layers
        penultimate_contributions = []
        all_temporal_consistency_losses = []
        all_entropy = []
        
        # Track temporal prediction error
        temporal_prediction_error = None
        
        for i, layer in enumerate(self.layers):
            if i == 0:
                # Bottom layer: Use entropy from previous epoch
                current, penultimate, entropy, consistency_loss = layer(
                    current, temporal_entropy=self.previous_entropy, layer_idx=i
                )
            else:
                current, penultimate, entropy, consistency_loss = layer(
                    current, layer_idx=i
                )
            
            penultimate_contributions.append(penultimate)
            
            # Collect entropy for next epoch (except from bottom layer)
            if i > 0:
                all_entropy.append(entropy)
                
            # Collect temporal consistency losses from top layer
            if consistency_loss is not None:
                all_temporal_consistency_losses.append(consistency_loss)
        
        # Prepare entropy for next epoch
        if all_entropy:
            total_entropy = torch.stack(all_entropy).sum(dim=0)
            
            # TEMPORAL INPUT PREDICTION (like your other experiments)
            if self.previous_entropy is not None:
                layer_0 = self.layers[0]
                if hasattr(layer_0, 'temporal_entropy_predictor'):
                    temporal_prediction = layer_0.temporal_entropy_predictor(current)
                    temporal_prediction_error = F.mse_loss(temporal_prediction, total_entropy)
            
            self.previous_entropy = total_entropy.detach()
        else:
            self.previous_entropy = None
            temporal_prediction_error = None
        
        # Combine penultimate contributions
        penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
        output = self.final(penultimate)
        
        # Combine temporal consistency losses
        temporal_consistency_loss = torch.stack(all_temporal_consistency_losses).mean() if all_temporal_consistency_losses else None
        
        return output, temporal_prediction_error, temporal_consistency_loss

def run_temporal_symmetry_experiment(config: TemporalSymmetryConfig):
    """Run the temporal symmetry experiment"""
    
    print(f"🌪️ TEMPORAL SYMMETRY EXPERIMENT: PAST ↔ FUTURE ⚡")
    print(f"Device: {config.device}")
    print(f"Hidden dims: {config.hidden_dims}")
    print(f"Patterns: {config.n_patterns}")
    print(f"Innovation: Bottom layer (past entropy) + Top layer (future patterns)")
    print(f"Mathematics: Temporal derivatives hierarchy!")
    
    # Load data - back to 10K baseline
    print("Loading 10KB Tiny Shakespeare (baseline)...")
    X, y, metadata = get_tiny_shakespeare_data(sequence_length=config.sequence_length)
    vocab_size = metadata["vocab_size"]
    print(f"Vocab size: {vocab_size}")
    print(f"Total sequences: {len(X)}")
    
    # Train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Data loaders
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
    
    train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model = TemporalSymmetryNet(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        n_patterns=config.n_patterns
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    results = {
        "train_losses": [],
        "test_losses": [],
        "config": config.__dict__
    }
    
    best_test_loss = float('inf')
    
    print(f"Starting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, (sequences, targets) in enumerate(train_data):
            sequences = sequences.to(config.device)
            targets = targets.to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass with temporal symmetry
            outputs, temporal_error, consistency_loss = model(sequences)
            task_loss = criterion(outputs, targets)
            
            # Multi-component loss
            total_loss = task_loss
            
            # Add temporal entropy prediction loss
            if temporal_error is not None:
                total_loss = total_loss + 0.1 * temporal_error
            
            # Add temporal consistency loss (NEW!)
            if consistency_loss is not None:
                total_loss = total_loss + 0.1 * consistency_loss
            
            loss = total_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Debug loss components
            if batch_idx % 50 == 0 and epoch % 20 == 0 and batch_idx == 0:
                temporal_val = temporal_error.item() if temporal_error is not None else 0.0
                consistency_val = consistency_loss.item() if consistency_loss is not None else 0.0
                print(f"    Loss - Task: {task_loss.item():.4f}, "
                      f"Temporal: {temporal_val:.4f}, Consistency: {consistency_val:.4f}")
        
        # Testing
        model.eval()
        test_losses = []
        with torch.no_grad():
            for sequences, targets in test_data:
                sequences = sequences.to(config.device)
                targets = targets.to(config.device)
                outputs, _, _ = model(sequences)
                loss = criterion(outputs, targets)
                test_losses.append(loss.item())
        
        # Record results
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        results["train_losses"].append(train_loss)
        results["test_losses"].append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        # Print progress
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}: Train={train_loss:.4f}, Test={test_loss:.4f}")
    
    # Final results
    print(f"\n🎯 TEMPORAL SYMMETRY RESULTS:")
    print(f"Best Test Loss: {best_test_loss:.4f}")
    print(f"Total Parameters: {total_params:,}")
    
    print(f"\n📊 COMPARISON TO YOUR BASELINES:")
    print(f"Previous Complex Temporal (10K): ~2.64")
    print(f"Temporal Symmetry (10K): {best_test_loss:.4f}")
    
    if best_test_loss < 2.64:
        improvement = ((2.64 - best_test_loss) / 2.64) * 100
        print(f"🎉 TEMPORAL SYMMETRY WINS by {improvement:.1f}%!")
        print(f"🌪️ The derivative hierarchy works! ⚡")
    else:
        gap = ((best_test_loss - 2.64) / best_test_loss) * 100
        print(f"Close but not quite: {gap:.1f}% behind")
    
    # Save results
    timestamp = int(time.time())
    results_file = f"temporal_symmetry_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configure the temporal symmetry experiment
    config = TemporalSymmetryConfig(
        embedding_dim=64,
        hidden_dims=[64, 64, 64],
        n_patterns=8,
        epochs=100,
        learning_rate=3e-4
    )
    
    print("🔥 TESTING TEMPORAL DERIVATIVES HIERARCHY 🔥")
    print("Past entropy ↔ Future patterns ↔ Learning acceleration")
    print("The beautiful symmetry of temporal prediction! ⚡🔮")
    
    results = run_temporal_symmetry_experiment(config)