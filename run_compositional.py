import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional
from torch.utils.data import DataLoader, TensorDataset

# Import the compositional architecture
from compositional_paradox_net import CompositionalDiscretePatternPredictiveNet
from data_generators import get_tiny_shakespeare_data

# Model wrapper for text processing
class CompositionalParadoxNetText(nn.Module):
    """Wrapper to handle text embedding for compositional ParadoxNet"""
    
    def __init__(self, sequence_length, vocab_size, embedding_dim=16, 
                 hidden_dims=[64, 32], penultimate_dim=32, n_patterns=8):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dim = sequence_length * embedding_dim
        
        self.compositional_net = CompositionalDiscretePatternPredictiveNet(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            penultimate_dim=penultimate_dim,
            output_dim=vocab_size,
            n_patterns=n_patterns
        )

    def forward(self, x):
        # Embed and flatten (like stable ParadoxNet)
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        
        # Forward through compositional network
        output, pred_errors = self.compositional_net(flattened)
        
        return output, pred_errors
    
    def get_layer_stats(self):
        """Get compositional statistics"""
        return self.compositional_net.get_layer_stats()
    
    def get_composition_hierarchy(self):
        """Get hierarchy description"""
        return self.compositional_net.get_composition_hierarchy()

# Factory function
def create_compositional_text_net(sequence_length, vocab_size, **kwargs):
    """Factory for compositional text model"""
    return CompositionalParadoxNetText(
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        embedding_dim=16,
        hidden_dims=[64, 32],
        penultimate_dim=32,
        n_patterns=16  # More patterns for richer composition
    )

# Experiment harness
class CompositionalTextHarness:
    def __init__(self, data_generator: Callable, epochs: int = 50, batch_size: int = 32,
                 pattern_loss_weight: float = 0.1):
        self.data_generator = data_generator
        self.epochs = epochs
        self.batch_size = batch_size
        self.pattern_loss_weight = pattern_loss_weight
        self.criterion = nn.CrossEntropyLoss()

    def run_trial(self, seed: int):
        start_time = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X, y, metadata = self.data_generator()
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        
        # Create compositional model
        model = create_compositional_text_net(
            sequence_length=X.shape[1],
            vocab_size=metadata['vocab_size']
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print("🔗 Training Compositional ParadoxNet on Text 🔗")
        print(f"Pattern loss weight: {self.pattern_loss_weight}")
        print(f"Hierarchy: {model.get_composition_hierarchy()}")
        
        best_test_loss = float('inf')
        
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0
            epoch_task_loss = 0
            epoch_pattern_loss = 0
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                final_output, pred_errors = model(X_batch)
                
                # Main task loss
                task_loss = self.criterion(final_output, y_batch.long())
                
                # Pattern prediction loss (same as stable ParadoxNet)
                if pred_errors is not None:
                    pattern_loss = torch.mean(pred_errors)
                else:
                    pattern_loss = torch.tensor(0.0, device=final_output.device)
                
                # Combined loss
                total_loss = task_loss + self.pattern_loss_weight * pattern_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_train_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_pattern_loss += pattern_loss.item()
                num_batches += 1

            # Evaluation
            model.eval()
            with torch.no_grad():
                test_output, _ = model(X_test)
                test_loss = self.criterion(test_output, y_test.long()).item()
                best_test_loss = min(best_test_loss, test_loss)
                
                avg_train_loss = epoch_train_loss / num_batches
                avg_task_loss = epoch_task_loss / num_batches
                avg_pattern_loss = epoch_pattern_loss / num_batches
            
            # Print progress and compositional stats
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch}: Total = {avg_train_loss:.4f} "
                      f"(Task: {avg_task_loss:.4f}, Pattern: {avg_pattern_loss:.4f}), "
                      f"Test = {test_loss:.4f}")
                
                # Show compositional learning progress
                if epoch % 20 == 0:
                    stats = model.get_layer_stats()
                    print("  Compositional Stats:")
                    for i, stat in enumerate(stats):
                        print(f"    Layer {i}: Entropy={stat.pattern_entropy:.2f}, "
                              f"Composition={stat.composition_alpha:.2f}, "
                              f"Paradox={stat.self_paradox_magnitude:.2f}")

        print(f"\n🔗 Trial finished in {time.time() - start_time:.2f} seconds.")
        print(f"Best test loss: {best_test_loss:.4f}")
        
        # Final compositional analysis
        print(f"\n=== FINAL COMPOSITIONAL ANALYSIS ===")
        stats = model.get_layer_stats()
        for i, stat in enumerate(stats):
            print(f"Layer {i}:")
            print(f"  Pattern entropy: {stat.pattern_entropy:.3f}")
            print(f"  Composition alpha: {stat.composition_alpha:.3f} "
                  f"({'diverse' if stat.composition_alpha > 0.5 else 'focused'} composition)")
            print(f"  Self-paradox: {stat.self_paradox_magnitude:.3f}")
        
        return best_test_loss

# Comparison function
def compare_compositional_vs_baseline():
    """Compare compositional vs non-compositional ParadoxNet"""
    
    print("=" * 60)
    print("🔗 COMPOSITIONAL vs BASELINE PARADOXNET COMPARISON 🔗")
    print("=" * 60)
    
    data_generator = get_tiny_shakespeare_data
    
    # Test compositional version
    print("\n1. COMPOSITIONAL ParadoxNet:")
    harness = CompositionalTextHarness(data_generator, epochs=50, pattern_loss_weight=0.1)
    compositional_loss = harness.run_trial(seed=42)
    
    print(f"\n" + "=" * 60)
    print("COMPARISON TARGETS:")
    print("• Transformer: 2.29 (champion)")
    print("• Stable ParadoxNet: 2.58 (stable baseline)")  
    print(f"• Compositional ParadoxNet: {compositional_loss:.4f}")
    
    if compositional_loss < 2.58:
        improvement = (2.58 - compositional_loss) / 2.58 * 100
        print(f"✅ Compositional IMPROVED by {improvement:.1f}% over stable ParadoxNet!")
        
        if compositional_loss < 2.35:
            print("🎉 Getting close to transformer performance!")
    else:
        degradation = (compositional_loss - 2.58) / 2.58 * 100
        print(f"❌ Compositional was {degradation:.1f}% worse than stable ParadoxNet")
    
    print("=" * 60)

# Quick test function
def quick_test():
    """Quick 20-epoch test to see if it's working"""
    print("🚀 QUICK COMPOSITIONAL TEST (20 epochs)")
    
    data_generator = get_tiny_shakespeare_data
    harness = CompositionalTextHarness(data_generator, epochs=20, pattern_loss_weight=0.1)
    test_loss = harness.run_trial(seed=123)
    
    print(f"\n⚡ Quick test result: {test_loss:.4f}")
    print("🎯 If this looks promising, run full comparison!")

if __name__ == "__main__":
    # Choose what to run
    print("Choose test:")
    print("1. Quick test (20 epochs)")
    print("2. Full comparison (50 epochs)")
    
    # For now, run quick test
    quick_test()
    
    print("\n💡 To run full comparison, uncomment the line below:")
    print("# compare_compositional_vs_baseline()")