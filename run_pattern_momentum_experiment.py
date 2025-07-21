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
from pattern_momentum_acceleration import TemporalSymmetryPatternMomentumNetComplex

@dataclass
class PatternMomentumConfig:
    """Configuration for temporal symmetry + pattern momentum experiment"""
    embedding_dim: int = 64
    hidden_dims: List[int] = None
    n_patterns: int = 8
    sequence_length: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 300
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64, 64]

def run_pattern_momentum_experiment(config: PatternMomentumConfig):
    """Run the temporal symmetry + pattern momentum experiment"""
    
    print(f"🚀⚡ TEMPORAL SYMMETRY + PATTERN MOMENTUM: PAST ↔ FUTURE + ACCELERATION ⚡🚀")
    print(f"Device: {config.device}")
    print(f"Hidden dims: {config.hidden_dims}")
    print(f"Patterns: {config.n_patterns}")
    print(f"Innovation: Bottom (past entropy) + Top (future patterns) + Momentum acceleration!")
    print(f"Mathematics: Temporal derivatives hierarchy with self-accelerating learning!")
    
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
    
    # Create model with temporal symmetry + pattern momentum
    model = TemporalSymmetryPatternMomentumNetComplex(
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
    
    # Training loop with detailed loss tracking
    results = {
        "train_losses": [],
        "test_losses": [],
        "task_losses": [],
        "temporal_losses": [],
        "pattern_evolution_losses": [],
        "config": config.__dict__
    }
    
    best_test_loss = float('inf')
    
    print(f"Starting training for {config.epochs} epochs...")
    print(f"🎯 THREE-LOSS STRUCTURE: task + 0.1*temporal + 0.1*pattern_evolution")
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []
        epoch_task_losses = []
        epoch_temporal_losses = []
        epoch_pattern_evolution_losses = []
        
        for batch_idx, (sequences, targets) in enumerate(train_data):
            sequences = sequences.to(config.device)
            targets = targets.to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass with temporal symmetry + pattern momentum
            outputs, temporal_error, pattern_evolution_error = model(sequences)
            task_loss = criterion(outputs, targets)
            
            # THREE-LOSS STRUCTURE (perfect symmetry!)
            total_loss = task_loss
            temporal_val = 0.0
            pattern_evolution_val = 0.0
            
            # Add temporal entropy prediction loss (bottom layer: PAST → PRESENT) 
            if temporal_error is not None:
                temporal_val = temporal_error.item()
                total_loss = total_loss + 0.5 * temporal_error
                
            # Add pattern evolution prediction loss (top layer: PRESENT → FUTURE)
            if pattern_evolution_error is not None:
                pattern_evolution_loss = torch.mean(pattern_evolution_error)
                pattern_evolution_val = pattern_evolution_loss.item()
                total_loss = total_loss + 0.5 * pattern_evolution_loss
            
            loss = total_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Collect detailed loss components
            train_losses.append(loss.item())
            epoch_task_losses.append(task_loss.item())
            epoch_temporal_losses.append(temporal_val)
            epoch_pattern_evolution_losses.append(pattern_evolution_val)
        
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
        task_loss_avg = np.mean(epoch_task_losses)
        temporal_loss_avg = np.mean(epoch_temporal_losses)
        pattern_evolution_loss_avg = np.mean(epoch_pattern_evolution_losses)
        
        results["train_losses"].append(train_loss)
        results["test_losses"].append(test_loss)
        results["task_losses"].append(task_loss_avg)
        results["temporal_losses"].append(temporal_loss_avg)
        results["pattern_evolution_losses"].append(pattern_evolution_loss_avg)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        # Print progress EVERY EPOCH for graphing
        print(f"Epoch {epoch:3d}: Train={train_loss:.4f}, Test={test_loss:.4f}, "
              f"Task={task_loss_avg:.4f}, Temp={temporal_loss_avg:.4f}, PatEvol={pattern_evolution_loss_avg:.4f}")
    
    # Final results
    print(f"\n🎯 TEMPORAL SYMMETRY + PATTERN MOMENTUM RESULTS:")
    print(f"Best Test Loss: {best_test_loss:.4f}")
    print(f"Total Parameters: {total_params:,}")
    
    print(f"\n📊 COMPARISON TO BASELINES:")
    print(f"Standard Complex Temporal (10K): ~2.64")
    print(f"Unified Pattern Loss (10K): ~2.58 (your current run)")
    print(f"Pattern Momentum (10K): {best_test_loss:.4f}")
    
    if best_test_loss < 2.58:
        improvement = ((2.58 - best_test_loss) / 2.58) * 100
        print(f"🎉 PATTERN MOMENTUM WINS by {improvement:.1f}%!")
        print(f"🚀⚡ The temporal derivatives hierarchy + momentum acceleration works! ⚡🚀")
    elif best_test_loss < 2.64:
        improvement = ((2.64 - best_test_loss) / 2.64) * 100
        print(f"🎊 Still beats the complex temporal baseline by {improvement:.1f}%!")
        print(f"🌪️ Pattern momentum making progress in the temporal space!")
    else:
        gap = ((best_test_loss - 2.58) / best_test_loss) * 100
        print(f"Learning still: {gap:.1f}% behind best baseline")
        print(f"🔮 The temporal symmetry is complex but promising...")
    
    # Print temporal symmetry statistics
    print(f"\n🔍 TEMPORAL SYMMETRY + PATTERN MOMENTUM STATISTICS:")
    for i, layer in enumerate(model.layers):
        if layer.last_stats:
            stats = layer.last_stats
            layer_type = "BOTTOM (past entropy)" if layer.is_bottom else "MIDDLE"
            print(f"Layer {i} ({layer_type}):")
            print(f"  Entropy magnitude: {stats.entropy_magnitude:.3f}")
            print(f"  Temporal entropy magnitude: {stats.temporal_entropy_magnitude:.3f}")
            print(f"  Paradox magnitude: {stats.self_paradox_magnitude:.3f}")
    
    # Integration layer statistics
    if hasattr(model.integration_layer, 'last_stats') and model.integration_layer.last_stats:
        stats = model.integration_layer.last_stats
        print(f"Integration Layer (TOP - future patterns + momentum):")
        print(f"  Pattern evolution prediction working: {stats.prediction_errors.mean().item():.3f}")
        print(f"  Momentum acceleration magnitude: {stats.entropy_magnitude:.3f}")
    
    # Save results
    timestamp = int(time.time())
    results_file = f"pattern_momentum_results_{timestamp}.json"
    csv_file = f"pattern_momentum_losses_{timestamp}.csv"
    
    # Save JSON results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV for easy graphing
    import csv
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'test_loss', 'task_loss', 'temporal_loss', 'pattern_evolution_loss'])
        for i in range(len(results['train_losses'])):
            writer.writerow([
                i,
                results['train_losses'][i],
                results['test_losses'][i], 
                results['task_losses'][i],
                results['temporal_losses'][i],
                results['pattern_evolution_losses'][i]
            ])
    
    print(f"Results saved to: {results_file}")
    print(f"CSV for graphing saved to: {csv_file}")
    
    return results

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configure the temporal symmetry + pattern momentum experiment
    config = PatternMomentumConfig(
        embedding_dim=64,
        hidden_dims=[64, 64, 64],
        n_patterns=8,
        epochs=300,
        learning_rate=3e-4
    )
    
    print("🔥 TESTING TEMPORAL DERIVATIVES HIERARCHY + PATTERN MOMENTUM 🔥")
    print("Past entropy ↔ Future pattern evolution ↔ Self-accelerating learning!")
    print("Perfect three-loss symmetry: task + temporal + pattern_evolution ⚡🌪️")
    
    results = run_pattern_momentum_experiment(config)