import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from dataclasses import dataclass
from typing import List
from torch.utils.data import DataLoader, TensorDataset

from entropy_recycling_temporal import TemporalEntropyRecyclingNet
from data_generators import get_tiny_shakespeare_data

@dataclass
class TemporalConfig:
    """Configuration for temporal entropy recycling experiment"""
    embedding_dim: int = 64
    hidden_dims: List[int] = None
    n_patterns: int = 8
    sequence_length: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64, 64]

def run_temporal_entropy_experiment(config: TemporalConfig):
    """Run temporal entropy recycling experiment"""
    
    print(f"⏰ TEMPORAL ENTROPY RECYCLING + PATTERN PREDICTION EXPERIMENT ⏰")
    print(f"Device: {config.device}")
    print(f"Hidden dims: {config.hidden_dims}")
    print(f"Patterns: {config.n_patterns}")
    print(f"Loss components: Task + 0.1*Temporal + 0.1*Pattern")
    
    # Load data
    print("Loading Tiny Shakespeare...")
    X, y, metadata = get_tiny_shakespeare_data(sequence_length=config.sequence_length)
    vocab_size = metadata["vocab_size"]
    print(f"Vocab size: {vocab_size}")
    
    # Create train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
    
    train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model = TemporalEntropyRecyclingNet(
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
        "entropy_stats": [],
        "temporal_entropy_evolution": [],  # Track how temporal entropy evolves
        "config": config.__dict__
    }
    
    best_test_loss = float('inf')
    
    print(f"Starting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []
        epoch_entropy_stats = []
        
        for batch_idx, (sequences, targets) in enumerate(train_data):
            sequences = sequences.to(config.device)
            targets = targets.to(config.device)
            
            optimizer.zero_grad()
            outputs, errors, temporal_error = model(sequences)
            task_loss = criterion(outputs, targets)
            
            # Add temporal prediction error and pattern prediction errors to loss
            total_loss = task_loss
            temporal_loss_val = 0.0
            pattern_loss_val = 0.0
            
            if temporal_error is not None:
                total_loss = total_loss + 0.1 * temporal_error  # Weight temporal prediction
                temporal_loss_val = temporal_error.item()
            
            if errors is not None:
                pattern_prediction_loss = torch.mean(errors)  # Average all pattern prediction errors
                total_loss = total_loss + 0.1 * pattern_prediction_loss  # Weight pattern prediction
                pattern_loss_val = pattern_prediction_loss.item()
            
            loss = total_loss
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Track loss components for debugging
            if batch_idx % 50 == 0 and epoch % 10 == 0 and batch_idx == 0:  # Print occasionally
                print(f"    Loss components - Task: {task_loss.item():.4f}, "
                      f"Temporal: {temporal_loss_val:.4f}, Pattern: {pattern_loss_val:.4f}")
            
            # Collect entropy statistics every 20 batches
            if batch_idx % 20 == 0:
                entropy_stats = []
                temporal_entropy_magnitude = 0.0
                
                for i, layer in enumerate(model.layers):
                    if hasattr(layer, 'last_stats') and layer.last_stats:
                        stats = layer.last_stats
                        entropy_stats.append({
                            'layer': i,
                            'entropy_magnitude': stats.entropy_magnitude,
                            'temporal_entropy_magnitude': stats.temporal_entropy_magnitude,
                            'composition_alpha': stats.composition_alpha,
                            'paradox_magnitude': stats.self_paradox_magnitude
                        })
                        
                        # Track temporal entropy from Layer 0
                        if i == 0:
                            temporal_entropy_magnitude = stats.temporal_entropy_magnitude
                
                epoch_entropy_stats.append(entropy_stats)
                
                # Track temporal entropy evolution across training
                results["temporal_entropy_evolution"].append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'temporal_entropy_magnitude': temporal_entropy_magnitude
                })
        
        # Testing
        model.eval()
        test_losses = []
        with torch.no_grad():
            for sequences, targets in test_data:
                sequences = sequences.to(config.device)
                targets = targets.to(config.device)
                outputs, errors, temporal_error = model(sequences)
                loss = criterion(outputs, targets)
                test_losses.append(loss.item())
        
        # Record results
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        results["train_losses"].append(train_loss)
        results["test_losses"].append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        results["entropy_stats"].append(epoch_entropy_stats[-1] if epoch_entropy_stats else [])
        
        # Print progress
        print(f"Epoch {epoch:3d}: Train={train_loss:.4f}, Test={test_loss:.4f}")
        
        # Print entropy and temporal statistics every 5 epochs
        if epoch % 5 == 0 and epoch_entropy_stats:
            print("  Temporal entropy stats:")
            for stat in epoch_entropy_stats[-1]:
                print(f"    Layer {stat['layer']}: entropy={stat['entropy_magnitude']:.3f}, "
                      f"temporal={stat['temporal_entropy_magnitude']:.3f}, "
                      f"composition={stat['composition_alpha']:.3f}")
        
        # Track temporal entropy magnitude across epochs
        if model.previous_entropy is not None:
            temporal_mag = torch.mean(torch.norm(model.previous_entropy, dim=-1)).item()
            print(f"  Stored temporal entropy magnitude: {temporal_mag:.3f}")
    
    # Final results
    final_train = results["train_losses"][-1]
    final_test = results["test_losses"][-1]
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Final Train Loss: {final_train:.4f}")
    print(f"Final Test Loss: {final_test:.4f}")
    print(f"Best Test Loss: {best_test_loss:.4f}")
    
    # Compare to baselines
    print(f"\n📊 COMPARISON TO BASELINES:")
    print(f"Standard Transformer: ~2.68")
    print(f"Transformer + RoPE: ~2.61") 
    print(f"Entropy Recycling (real): ~3.16")
    print(f"Temporal + Pattern Prediction: {best_test_loss:.4f}")
    
    gap_from_standard = ((best_test_loss - 2.68) / 2.68) * 100
    gap_from_rope = ((best_test_loss - 2.61) / 2.61) * 100
    gap_from_real = ((best_test_loss - 3.16) / 3.16) * 100
    
    print(f"\nPerformance gaps:")
    print(f"  vs Standard Transformer: {gap_from_standard:+.1f}%")
    print(f"  vs Transformer + RoPE: {gap_from_rope:+.1f}%")
    print(f"  vs Real Entropy Recycling: {gap_from_real:+.1f}%")
    
    if best_test_loss < 3.16:
        print("🎉 Temporal entropy recycling beats standard entropy recycling!")
    if best_test_loss < 3.0:
        print("🚀 Breaking the 3.0 barrier!")
    if best_test_loss < 2.8:
        print("🔥 Getting competitive with transformers!")
    
    # Analyze temporal entropy evolution
    if results["temporal_entropy_evolution"]:
        temporal_mags = [x['temporal_entropy_magnitude'] for x in results["temporal_entropy_evolution"]]
        non_zero_temporal = [x for x in temporal_mags if x > 0]
        
        if non_zero_temporal:
            print(f"\n⏰ TEMPORAL ENTROPY ANALYSIS:")
            print(f"  Temporal entropy became active after {len(temporal_mags) - len(non_zero_temporal)} steps")
            print(f"  Average temporal entropy magnitude: {np.mean(non_zero_temporal):.3f}")
            print(f"  Temporal entropy range: {min(non_zero_temporal):.3f} - {max(non_zero_temporal):.3f}")
        else:
            print(f"\n⚠️ Temporal entropy never became active - no previous epochs processed")
    
    # Save results
    timestamp = int(time.time())
    results_file = f"temporal_entropy_recycling_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, list):
            results_serializable[key] = value
        else:
            results_serializable[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    # Run experiment with temporal configuration
    config = TemporalConfig(
        embedding_dim=64,
        hidden_dims=[64, 64, 64],
        n_patterns=8,
        epochs=50,
        learning_rate=3e-4
    )
    
    results = run_temporal_entropy_experiment(config)