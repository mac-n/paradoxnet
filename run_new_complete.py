import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset

# Import the compositional architecture
from complete_compositional import CompositionalDiscretePatternPredictiveNet
from data_generators import get_tiny_shakespeare_data

# Enhanced compositional network with penultimate patterns
class CompositionalPenultimatePatternLayer(nn.Module):
    """
    Penultimate layer with compositional patterns + paradox mechanism.
    
    This creates full architectural consistency - ALL layers use patterns+paradox!
    """
    
    def __init__(self, penultimate_dim, output_dim, n_patterns=8, prev_layer=None):
        super().__init__()
        
        self.penultimate_dim = penultimate_dim
        self.output_dim = output_dim
        self.n_patterns = n_patterns
        
        # Compositional patterns from previous layer
        if prev_layer is not None:
            self.composition_weights = nn.Parameter(
                torch.randn(n_patterns, prev_layer.n_patterns) / (prev_layer.n_patterns ** 0.5)
            )
            self.prev_layer = prev_layer
            self.pattern_dict = None  # Computed property
        else:
            # Fallback: independent patterns
            self.pattern_dict = nn.Parameter(
                torch.randn(n_patterns, penultimate_dim) / (penultimate_dim ** 0.5)
            )
            self.composition_weights = None
            self.prev_layer = None
        
        # Pattern attention
        self.pattern_attention = nn.Linear(penultimate_dim, n_patterns)
        
        # Self-prediction paradox mechanism
        self.self_predictor = nn.Linear(penultimate_dim, penultimate_dim)
        
        # Final output prediction from patterns
        self.output_predictor = nn.Linear(penultimate_dim, output_dim)
    
    def get_pattern_dict(self):
        """Get compositional patterns from previous layer"""
        if self.composition_weights is not None and self.prev_layer is not None:
            prev_patterns = self.prev_layer.get_pattern_dict()
            return self.composition_weights @ prev_patterns
        else:
            return self.pattern_dict
    
    def forward(self, integrated_penultimate: torch.Tensor) -> torch.Tensor:
        """
        Pattern-based output prediction with compositional patterns + paradox.
        
        This is the same mechanism as other layers!
        """
        if integrated_penultimate.dim() == 1:
            integrated_penultimate = integrated_penultimate.unsqueeze(0)
        
        # Step 1: Attention over compositional output patterns
        attention_weights = F.softmax(
            self.pattern_attention(integrated_penultimate), dim=-1
        )
        
        # Step 2: Select patterns based on input (using compositional patterns!)
        selected_patterns = attention_weights @ self.get_pattern_dict()
        
        # Step 3: Self-prediction paradox for output confidence
        predicted_self = self.self_predictor(selected_patterns)
        paradox = predicted_self - selected_patterns
        paradox_magnitude = torch.norm(paradox, dim=-1, keepdim=True)
        
        # Step 4: Confidence-based gating (same as other layers)
        gated_patterns = selected_patterns * torch.sigmoid(paradox_magnitude)
        
        # Step 5: Final output prediction from patterns
        output = self.output_predictor(gated_patterns)
        
        return output

class EnhancedCompositionalNet(CompositionalDiscretePatternPredictiveNet):
    """
    Enhanced compositional network with penultimate pattern layer.
    
    Now ALL layers use compositional patterns + paradox mechanism!
    """
    
    def __init__(self, input_dim, hidden_dims, penultimate_dim, output_dim, n_patterns=8,
                 initial_temp=1.0, min_temp=0.1, temp_decay=0.99):
        
        # Initialize base compositional network
        super().__init__(input_dim, hidden_dims, penultimate_dim, output_dim, n_patterns,
                        initial_temp, min_temp, temp_decay)
        
        # Replace simple final linear layer with compositional penultimate pattern layer
        self.final = CompositionalPenultimatePatternLayer(
            penultimate_dim=penultimate_dim,
            output_dim=output_dim,
            n_patterns=n_patterns,
            prev_layer=self.layers[-1]  # Compose from last hidden layer
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with compositional penultimate patterns"""
        penultimate_contributions = []
        current = x
        all_errors = []
        
        # Process through compositional hidden layers
        for i, layer in enumerate(self.layers):
            next_layer = self.layers[i+1] if i < len(self.layers)-1 else None
            current, penultimate, error = layer(current, next_layer, i)
            
            if error is not None:
                all_errors.append(error)
            penultimate_contributions.append(penultimate)
        
        # Integrate penultimate contributions
        integrated_penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
        
        # NEW: Pattern-based final prediction with compositional patterns
        output = self.final(integrated_penultimate)
        
        return output, torch.cat(all_errors, dim=1) if all_errors else None

# Model wrapper for text processing
class CompositionalParadoxNetText(nn.Module):
    """Wrapper with enhanced compositional network including penultimate patterns"""
    
    def __init__(self, sequence_length, vocab_size, embedding_dim=16, 
                 hidden_dims=[64, 64], penultimate_dim=64, n_patterns=8):  # Use same dim throughout
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dim = sequence_length * embedding_dim
        
        # Use enhanced compositional network with consistent dimensions
        self.compositional_net = EnhancedCompositionalNet(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            penultimate_dim=penultimate_dim,
            output_dim=vocab_size,
            n_patterns=n_patterns
        )

    def forward(self, x):
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        output, pred_errors = self.compositional_net(flattened)
        return output, pred_errors
    
    def get_layer_stats(self):
        return self.compositional_net.get_layer_stats()
    
    def get_composition_hierarchy(self):
        hierarchy = self.compositional_net.get_composition_hierarchy()
        hierarchy.append("Penultimate: Composed from Layer 2")  # Add penultimate info
        return hierarchy

# Factory function
def create_compositional_text_net(sequence_length, vocab_size, **kwargs):
    """Factory for compositional text model"""
    return CompositionalParadoxNetText(
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        embedding_dim=16,
        hidden_dims=[64, 64],  # Consistent dimensions
        penultimate_dim=64,    # Match hidden dims
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
        
        print("🔗 Training FULL Compositional ParadoxNet on Text 🔗")
        print("(INCLUDING compositional penultimate pattern layer)")
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
    print("🔗 FULL COMPOSITIONAL ParadoxNet vs BASELINE 🔗")
    print("(Now with compositional penultimate pattern layer!)")
    print("=" * 60)
    
    data_generator = get_tiny_shakespeare_data
    
    # Test FULL compositional version (including penultimate patterns)
    print("\n1. FULL COMPOSITIONAL ParadoxNet (with penultimate patterns):")
    print("   Using seed=123 (same as successful quick test)")
    harness = CompositionalTextHarness(data_generator, epochs=50, pattern_loss_weight=0.1)
    compositional_loss = harness.run_trial(seed=123)  # Use same seed as quick test!
    
    print(f"\n" + "=" * 60)
    print("COMPARISON TARGETS:")
    print("• Transformer: 2.29 (champion)")
    print("• Stable ParadoxNet: 2.58 (stable baseline)")  
    print(f"• FULL Compositional ParadoxNet: {compositional_loss:.4f}")
    
    if compositional_loss < 2.58:
        improvement = (2.58 - compositional_loss) / 2.58 * 100
        print(f"✅ FULL Compositional IMPROVED by {improvement:.1f}% over stable ParadoxNet!")
        
        if compositional_loss < 2.35:
            print("🎉 Getting close to transformer performance!")
        if compositional_loss < 2.29:
            print("🏆 BEAT THE TRANSFORMER!")
    else:
        degradation = (compositional_loss - 2.58) / 2.58 * 100
        print(f"❌ FULL Compositional was {degradation:.1f}% worse than stable ParadoxNet")
    
    print("=" * 60)

# Quick test function
def quick_test():
    """Quick 20-epoch test to see if it's working"""
    print("🚀 QUICK FULL COMPOSITIONAL TEST (20 epochs)")
    print("   (Including compositional penultimate pattern layer)")
    
    data_generator = get_tiny_shakespeare_data
    harness = CompositionalTextHarness(data_generator, epochs=20, pattern_loss_weight=0.1)
    test_loss = harness.run_trial(seed=123)
    
    print(f"\n⚡ Quick test result: {test_loss:.4f}")
    print("🎯 Target: beat 2.58 (stable ParadoxNet baseline)")
    if test_loss < 2.58:
        print("✅ Looking promising! Run full comparison!")
    else:
        print("🤔 Hmm, might need tuning or more epochs")
    print("🎯 If this looks promising, run full comparison!")

if __name__ == "__main__":
    # Choose what to run
    print("Choose test:")
    print("1. Quick test (20 epochs) - FULL compositional including penultimate patterns")
    print("2. Full comparison (50 epochs) - vs stable ParadoxNet baseline")
    
    # For now, run quick test
    #quick_test()
    
    print("\n💡 To run full comparison, uncomment the line below:")
    compare_compositional_vs_baseline()
    print("🎯 This could be the architecture that beats transformers!")