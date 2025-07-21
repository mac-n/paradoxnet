import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from dataclasses import dataclass
from typing import List
from torch.utils.data import DataLoader, TensorDataset

from entropy_recycling_complex import EntropyRecyclingComplexNet
from data_generators import get_tiny_shakespeare_data

@dataclass
class ComplexConfig:
    """Configuration for complex entropy recycling experiment"""
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
            self.hidden_dims = [32, 32, 32]  # Complex dimensions

def run_complex_entropy_experiment(config: ComplexConfig):
    """Run complex entropy recycling with RoPE experiment"""
    
    print(f"🌀 COMPLEX ENTROPY RECYCLING + ROPE EXPERIMENT 🌀")
    print(f"Device: {config.device}")
    print(f"Hidden dims (complex): {config.hidden_dims}")
    print(f"Patterns: {config.n_patterns}")
    
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
    model = EntropyRecyclingComplexNet(
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
        "config": config.__dict__
    }
    
    best_test_loss = float('inf')
    
    print(f"Starting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []
        epoch_entropy_stats = []
        
        if epoch == 0:
            print(f"Processing first epoch...")
        
        for batch_idx, (sequences, targets) in enumerate(train_data):
            sequences = sequences.to(config.device)
            targets = targets.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping for stability (important for complex numbers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Collect entropy statistics every 10 batches
            if batch_idx % 10 == 0:
                entropy_stats = []
                for i, layer in enumerate(model.layers):
                    if hasattr(layer, 'last_stats') and layer.last_stats:
                        stats = layer.last_stats
                        entropy_stats.append({
                            'layer': i,
                            'entropy_magnitude': stats.entropy_magnitude,
                            'phase_diversity': stats.phase_diversity,
                            'composition_alpha': stats.composition_alpha,
                            'paradox_magnitude': stats.self_paradox_magnitude
                        })
                epoch_entropy_stats.append(entropy_stats)
        
        # Testing
        model.eval()
        test_losses = []
        with torch.no_grad():
            for sequences, targets in test_data:
                sequences = sequences.to(config.device)
                targets = targets.to(config.device)
                outputs = model(sequences)
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
        
        # Print progress (more frequent for debugging)
        if epoch % 1 == 0:  # Print every epoch
            print(f"Epoch {epoch:3d}: Train={train_loss:.4f}, Test={test_loss:.4f}")
            
            # Print entropy and phase statistics
            if epoch_entropy_stats:
                print("  Complex entropy stats:")
                for stat in epoch_entropy_stats[-1]:
                    print(f"    Layer {stat['layer']}: entropy={stat['entropy_magnitude']:.3f}, "
                          f"phase_div={stat['phase_diversity']:.3f}, "
                          f"composition={stat['composition_alpha']:.3f}")
    
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
    print(f"Complex Entropy + RoPE: {best_test_loss:.4f}")
    
    if best_test_loss < 3.0:
        print("🎉 Complex entropy recycling beats real version!")
    if best_test_loss < 2.8:
        print("🚀 Getting competitive with transformers!")
    if best_test_loss < 2.65:
        print("🔥 BEATING THE BASELINES!")
    
    # Save results
    timestamp = int(time.time())
    results_file = f"complex_entropy_recycling_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    # Run experiment with complex configuration
    config = ComplexConfig(
        embedding_dim=64,          # Real embedding dimension
        hidden_dims=[32, 32, 32],  # Complex hidden dimensions
        n_patterns=8,
        epochs=50,
        learning_rate=3e-4
    )
    
    results = run_complex_entropy_experiment(config)