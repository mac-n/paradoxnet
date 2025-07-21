import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional
from torch.utils.data import DataLoader, TensorDataset

# Import the complex-valued model AS IS
from paradox_net_complex import ParadoxNetComplex
from data_generators import get_tiny_shakespeare_data

# Simple Enhanced Model - just add pattern loss to existing complex model
class SimpleEnhancedComplexParadoxNet(ParadoxNetComplex):
    """
    Add simple pattern prediction loss to complex ParadoxNet.
    Don't change the architecture, just add auxiliary loss.
    """
    
    def get_pattern_prediction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pattern prediction loss similar to base_predictive_net.py
        but simplified for the complex version.
        """
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        current_seq_real = self.apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

        pattern_losses = []
        
        # Simple pattern prediction between adjacent layers
        for i in range(len(self.hidden_layers) - 1):
            current_layer = self.hidden_layers[i]
            next_layer = self.hidden_layers[i + 1]
            
            # Process through current layer
            current_hidden = current_layer.apply_self_processing(current_seq)
            
            # Simple pattern attention on mean-pooled hidden state
            pooled_hidden = current_hidden.mean(dim=1)  # (batch, hidden_dim)
            
            # Get pattern attention weights (real-valued)
            attn_logits = current_layer.pattern_attention(pooled_hidden)
            pattern_weights = torch.softmax(attn_logits.real[:, :current_layer.n_patterns], dim=-1)
            
            # Predict next layer's processing (simplified)
            with torch.no_grad():
                next_hidden = next_layer.apply_self_processing(current_hidden)
                next_pooled = next_hidden.mean(dim=1)
            
            # Simple prediction loss - predict the magnitude of next layer's output
            predicted_next_mag = pattern_weights @ torch.abs(current_layer.pattern_dict).mean(dim=-1)
            actual_next_mag = torch.abs(next_pooled).mean(dim=-1)
            
            # L2 loss between predicted and actual magnitude
            pattern_loss = torch.mean((predicted_next_mag - actual_next_mag) ** 2)
            pattern_losses.append(pattern_loss)
            
            # Update current_seq for next iteration
            current_seq = current_hidden
        
        # Return average pattern prediction loss
        if pattern_losses:
            return sum(pattern_losses) / len(pattern_losses)
        else:
            return torch.tensor(0.0, device=x.device)
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Helper method to make RoPE accessible"""
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * freqs_cis
        x_out = torch.view_as_real(x_rotated).flatten(2)
        return x_out.type_as(x)


# Simple Experiment Harness with Pattern Loss
class SimplePatternHarness:
    def __init__(self, data_generator: Callable, epochs: int = 50, batch_size: int = 32, 
                 pattern_loss_weight: float = 0.1):
        self.data_generator = data_generator
        self.epochs = epochs
        self.batch_size = batch_size
        self.pattern_loss_weight = pattern_loss_weight
        self.criterion = nn.CrossEntropyLoss()

    def run_trial(self, seed: int, use_patterns: bool = True):
        start_time = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X, y, metadata = self.data_generator()
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        
        # Create model
        if use_patterns:
            model = SimpleEnhancedComplexParadoxNet(
                vocab_size=metadata['vocab_size'],
                embedding_dim=64,
                hidden_dims=[64, 64],
                n_patterns=16
            )
            print(f"Training Complex ParadoxNet WITH pattern loss (weight: {self.pattern_loss_weight})")
        else:
            model = ParadoxNetComplex(
                vocab_size=metadata['vocab_size'],
                embedding_dim=64,
                hidden_dims=[64, 64],
                n_patterns=16
            )
            print("Training Complex ParadoxNet WITHOUT pattern loss")
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        best_test_loss = float('inf')
        
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0
            epoch_task_loss = 0
            epoch_pattern_loss = 0
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Main forward pass
                final_output = model(X_batch)
                task_loss = self.criterion(final_output, y_batch.long())
                
                # Add pattern loss if using patterns
                if use_patterns:
                    pattern_loss = model.get_pattern_prediction_loss(X_batch)
                    total_loss = task_loss + self.pattern_loss_weight * pattern_loss
                    epoch_pattern_loss += pattern_loss.item()
                else:
                    total_loss = task_loss
                    epoch_pattern_loss = 0
                
                total_loss.backward()
                optimizer.step()
                
                epoch_train_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                num_batches += 1

            # Evaluation
            model.eval()
            with torch.no_grad():
                test_output = model(X_test)
                test_loss = self.criterion(test_output, y_test.long()).item()
                best_test_loss = min(best_test_loss, test_loss)
                
                avg_train_loss = epoch_train_loss / num_batches
                avg_task_loss = epoch_task_loss / num_batches
                avg_pattern_loss = epoch_pattern_loss / num_batches if use_patterns else 0
            
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                if use_patterns:
                    print(f"Epoch {epoch}: Total = {avg_train_loss:.4f} "
                          f"(Task: {avg_task_loss:.4f}, Pattern: {avg_pattern_loss:.4f}), "
                          f"Test = {test_loss:.4f}")
                else:
                    print(f"Epoch {epoch}: Train = {avg_train_loss:.4f}, Test = {test_loss:.4f}")

        print(f"Trial finished in {time.time() - start_time:.2f} seconds.")
        print(f"Best test loss: {best_test_loss:.4f}")
        return best_test_loss


def compare_pattern_effect():
    """Simple comparison: with vs without pattern loss"""
    
    print("=" * 60)
    print("COMPARING: Complex ParadoxNet WITH vs WITHOUT Pattern Loss")
    print("=" * 60)
    
    data_generator = get_tiny_shakespeare_data
    harness = SimplePatternHarness(data_generator, epochs=40, pattern_loss_weight=0.1)
    
    # Run without patterns
    # print("\n1. WITHOUT Pattern Loss:")
    # test_loss_without = harness.run_trial(seed=42, use_patterns=False)
    
    # print("\n" + "=" * 60)
    
    # Run with patterns
    print("\n2. WITH Pattern Loss:")
    test_loss_with = harness.run_trial(seed=42, use_patterns=True)
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS:")
    print(f"Without patterns: {test_loss_without:.4f}")
    print(f"With patterns:    {test_loss_with:.4f}")
    
    if test_loss_with < test_loss_without:
        improvement = (test_loss_without - test_loss_with) / test_loss_without * 100
        print(f"✅ Pattern loss IMPROVED performance by {improvement:.1f}%!")
    else:
        degradation = (test_loss_with - test_loss_without) / test_loss_without * 100  
        print(f"❌ Pattern loss HURT performance by {degradation:.1f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    compare_pattern_effect()
    
    print("\n🎯 Quick test completed!")
    print("💡 Try different pattern_loss_weight values (0.05, 0.2, 0.5) if needed")