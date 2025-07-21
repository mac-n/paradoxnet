import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from dataclasses import dataclass
from typing import List
from torch.utils.data import DataLoader, TensorDataset

from transformer_with_rope import TransformerWithRoPE, StandardTransformer
from data_generators import get_tiny_shakespeare_data

@dataclass
class TransformerConfig:
    """Configuration for transformer comparison"""
    d_model: int = 48
    n_heads: int = 3
    d_ff: int = 96
    n_layers: int = 3
    sequence_length: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train_transformer(model, train_data, test_data, config, model_name):
    """Train a transformer model and return results"""
    
    print(f"\n🤖 TRAINING {model_name.upper()} 🤖")
    
    model = model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    results = {
        "model_name": model_name,
        "train_losses": [],
        "test_losses": [],
        "total_params": total_params,
        "config": config.__dict__
    }
    
    best_test_loss = float('inf')
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []
        
        for sequences, targets in train_data:
            sequences = sequences.to(config.device)
            targets = targets.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
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
        
        # Print progress
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}: Train={train_loss:.4f}, Test={test_loss:.4f}")
    
    results["final_train_loss"] = results["train_losses"][-1]
    results["final_test_loss"] = results["test_losses"][-1]
    results["best_test_loss"] = best_test_loss
    
    print(f"Final: Train={results['final_train_loss']:.4f}, Test={results['final_test_loss']:.4f}")
    print(f"Best Test: {results['best_test_loss']:.4f}")
    
    return results

def run_rope_comparison():
    """Compare standard transformer vs transformer with RoPE"""
    
    print("🔬 TRANSFORMER ROPE SANITY CHECK 🔬")
    print("Comparing: Standard Positional Encoding vs RoPE")
    
    config = TransformerConfig()
    print(f"Config: {config}")
    
    # Load data
    print("\nLoading Tiny Shakespeare...")
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
    
    # Train both models
    all_results = {}
    
    # 1. Standard Transformer (baseline)
    standard_model = StandardTransformer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers
    )
    
    results_standard = train_transformer(
        standard_model, train_data, test_data, config, "Standard Transformer"
    )
    all_results["standard"] = results_standard
    
    # 2. Transformer with RoPE  
    rope_model = TransformerWithRoPE(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers
    )
    
    results_rope = train_transformer(
        rope_model, train_data, test_data, config, "Transformer with RoPE"
    )
    all_results["rope"] = results_rope
    
    # Final comparison
    print(f"\n📊 FINAL COMPARISON 📊")
    print(f"Standard Transformer:")
    print(f"  Best Test Loss: {results_standard['best_test_loss']:.4f}")
    print(f"  Final Test Loss: {results_standard['final_test_loss']:.4f}")
    print(f"  Parameters: {results_standard['total_params']:,}")
    
    print(f"\nTransformer with RoPE:")
    print(f"  Best Test Loss: {results_rope['best_test_loss']:.4f}")
    print(f"  Final Test Loss: {results_rope['final_test_loss']:.4f}")
    print(f"  Parameters: {results_rope['total_params']:,}")
    
    improvement = results_standard['best_test_loss'] - results_rope['best_test_loss']
    percent_improvement = (improvement / results_standard['best_test_loss']) * 100
    
    print(f"\nRoPE Improvement: {improvement:.4f} ({percent_improvement:+.1f}%)")
    
    if improvement > 0.05:
        print("🚨 RoPE provides significant improvement! Your architecture might be benefiting from better positional encoding.")
    elif improvement > 0.01:
        print("⚠️  RoPE provides modest improvement. Some benefit from better positional encoding.")
    else:
        print("✅ RoPE provides minimal improvement. Your architecture's benefits are likely architectural, not just positional encoding!")
    
    # Save results
    timestamp = int(time.time())
    results_file = f"transformer_rope_comparison_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return all_results

if __name__ == "__main__":
    results = run_rope_comparison()