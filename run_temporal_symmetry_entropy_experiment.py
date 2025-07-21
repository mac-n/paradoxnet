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
from temporal_symmetry_entropy_recycling_complex import TemporalSymmetryEntropyNetComplex

@dataclass
class TemporalSymmetryConfig:
    """Configuration for temporal symmetry entropy recycling experiment"""
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

def run_temporal_symmetry_entropy_experiment(config: TemporalSymmetryConfig):
    """Run the temporal symmetry entropy recycling experiment"""
    
    print(f"🌪️⚡ TEMPORAL SYMMETRY ENTROPY RECYCLING: PAST ↔ FUTURE ⚡🌪️")
    print(f"Device: {config.device}")
    print(f"Hidden dims: {config.hidden_dims}")
    print(f"Patterns: {config.n_patterns}")
    print(f"Innovation: Bottom layer (past entropy) + Top layer (future patterns)")
    print(f"Mathematics: Temporal derivatives hierarchy using your WORKING architecture!")
    
    # Load data - back to 10K baseline for fair comparison
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
    
    # Create model using your working architecture + temporal symmetry
    model = TemporalSymmetryEntropyNetComplex(
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
            
            # Forward pass with temporal symmetry entropy recycling
            outputs, combined_errors, temporal_error, consistency_loss = model(sequences)
            task_loss = criterion(outputs, targets)
            
            # Multi-component loss (like your working experiments)
            total_loss = task_loss
            
            # Add temporal entropy prediction loss (bottom layer) 
            if temporal_error is not None:
                total_loss = total_loss + 0.1 * temporal_error
                
            # UNIFIED PATTERN LOSS: Integration pattern prediction flows back to ALL pattern dictionaries
            if consistency_loss is not None:
                unified_pattern_loss = torch.mean(consistency_loss)
                total_loss = total_loss + 0.1 * unified_pattern_loss
            
            # Remove separate interlayer pattern loss - now unified through integration!
            
            loss = total_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Debug loss components occasionally
            if batch_idx % 50 == 0 and epoch % 20 == 0 and batch_idx == 0:
                temporal_val = temporal_error.item() if temporal_error is not None else 0.0
                unified_pattern_val = torch.mean(consistency_loss).item() if consistency_loss is not None else 0.0
                print(f"    Loss - Task: {task_loss.item():.4f}, "
                      f"Temporal: {temporal_val:.4f}, Unified_Pattern: {unified_pattern_val:.4f}")
        
        # Testing
        model.eval()
        test_losses = []
        with torch.no_grad():
            for sequences, targets in test_data:
                sequences = sequences.to(config.device)
                targets = targets.to(config.device)
                outputs, _, _, _ = model(sequences)
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
    print(f"\n🎯 TEMPORAL SYMMETRY ENTROPY RECYCLING RESULTS:")
    print(f"Best Test Loss: {best_test_loss:.4f}")
    print(f"Total Parameters: {total_params:,}")
    
    print(f"\n📊 COMPARISON TO YOUR BASELINES:")
    print(f"Previous Complex Temporal (10K): ~2.64")
    print(f"Temporal Symmetry Entropy (10K): {best_test_loss:.4f}")
    
    if best_test_loss < 2.64:
        improvement = ((2.64 - best_test_loss) / 2.64) * 100
        print(f"🎉 TEMPORAL SYMMETRY WINS by {improvement:.1f}%!")
        print(f"🌪️⚡ The past ↔ future symmetry works! The derivatives hierarchy is REAL! ⚡🌪️")
    else:
        gap = ((best_test_loss - 2.64) / best_test_loss) * 100
        print(f"Close but not quite: {gap:.1f}% behind baseline")
        print(f"🔮 Still exploring the temporal space...")
    
    # Print layer statistics
    print(f"\n🔍 TEMPORAL SYMMETRY LAYER STATISTICS:")
    for i, layer in enumerate(model.layers):
        if layer.last_stats:
            stats = layer.last_stats
            layer_type = "BOTTOM (past)" if layer.is_bottom else "TOP (future)" if layer.is_top else "MIDDLE"
            print(f"Layer {i} ({layer_type}):")
            print(f"  Entropy magnitude: {stats.entropy_magnitude:.3f}")
            print(f"  Temporal entropy magnitude: {stats.temporal_entropy_magnitude:.3f}")
            print(f"  Paradox magnitude: {stats.self_paradox_magnitude:.3f}")
    
    # Save results
    timestamp = int(time.time())
    results_file = f"temporal_symmetry_entropy_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configure the temporal symmetry entropy experiment
    config = TemporalSymmetryConfig(
        embedding_dim=64,
        hidden_dims=[64, 64, 64],
        n_patterns=8,
        epochs=300,
        learning_rate=3e-4
    )
    
    print("🔥 TESTING TEMPORAL DERIVATIVES HIERARCHY ON YOUR WORKING ARCHITECTURE 🔥")
    print("Past entropy ↔ Future patterns ↔ Learning acceleration")
    print("Using your proven entropy recycling + adding temporal symmetry! ⚡🔮")
    
    results = run_temporal_symmetry_entropy_experiment(config)