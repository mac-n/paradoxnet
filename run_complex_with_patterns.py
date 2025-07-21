import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional
from torch.utils.data import DataLoader, TensorDataset

# Import the complex-valued model (we'll need to modify it slightly)
from paradox_net_complex import ParadoxNetComplex
from data_generators import get_tiny_shakespeare_data

# Enhanced Complex ParadoxNet with Pattern Training
class EnhancedComplexParadoxNet(ParadoxNetComplex):
    """
    Add pattern prediction training to the complex ParadoxNet.
    
    This combines:
    - Complex-valued processing
    - Self-prediction paradox nonlinearity  
    - Pattern prediction training signal
    """
    
    def compress_activity_complex(self, x: torch.Tensor, layer_idx: int) -> tuple:
        """
        Compress complex-valued activity using pattern dictionary.
        Similar to the base version but adapted for complex numbers.
        """
        if layer_idx >= len(self.hidden_layers):
            return x, torch.ones(x.shape[0], 8, device=x.device) / 8  # Uniform weights for final layer
            
        layer = self.hidden_layers[layer_idx]
        
        # Get pattern attention (this outputs real values for softmax)
        attention_logits = layer.pattern_attention(x)
        # Reshape to get proper attention weights
        pattern_weights = torch.softmax(attention_logits.real.view(x.shape[0], -1)[:, :layer.n_patterns], dim=-1)
        
        # Compress using weighted combination of complex patterns
        compressed = pattern_weights @ layer.pattern_dict
        
        return compressed, pattern_weights
    
    def forward_with_pattern_prediction(self, x: torch.Tensor) -> tuple:
        """
        Forward pass that also computes pattern prediction errors for training.
        
        Returns:
            final_output: The main task prediction
            pattern_errors: List of pattern prediction errors for auxiliary loss
        """
        batch_size, seq_len = x.shape
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        current_seq_real = self.apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

        penultimate_contributions = []
        pattern_errors = []
        
        # Process through layers with pattern prediction
        for i, layer in enumerate(self.hidden_layers):
            # Standard processing
            current_seq, penultimate = layer(current_seq)
            penultimate_contributions.append(penultimate.mean(dim=1))
            
            # Pattern prediction training
            if i < len(self.hidden_layers) - 1:  # Not the last layer
                # Compress current layer's activity
                my_compressed, my_patterns = self.compress_activity_complex(current_seq.mean(dim=1), i)
                
                # Predict what next layer will do
                predicted_next = my_compressed
                
                # Get actual next layer processing (detached for stability)
                with torch.no_grad():
                    next_layer = self.hidden_layers[i + 1]
                    actual_next = next_layer.apply_self_processing(current_seq)
                    actual_compressed, _ = self.compress_activity_complex(actual_next.mean(dim=1), i + 1)
                
                # Match dimensions for comparison
                min_dim = min(predicted_next.shape[-1], actual_compressed.shape[-1])
                if predicted_next.shape[-1] != actual_compressed.shape[-1]:
                    predicted_next = predicted_next[..., :min_dim]
                    actual_compressed = actual_compressed[..., :min_dim]
                
                # Pattern prediction error (real-valued loss from complex tensors)
                pred_error = torch.mean((actual_compressed - predicted_next).abs() ** 2)
                pattern_errors.append(pred_error)
        
        # Standard final processing
        consensus_view = torch.sum(torch.stack(penultimate_contributions), dim=0)
        recursive_residual = current_seq.mean(dim=1)
        penultimate_input = consensus_view + recursive_residual
        final_output = self.penultimate_layer(penultimate_input)
        
        return final_output, pattern_errors
    
    # Helper method to apply rotary pos emb (make it accessible)
    def apply_rotary_pos_emb(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Applies rotary positional embedding to the input tensor."""
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * freqs_cis
        x_out = torch.view_as_real(x_rotated).flatten(2)
        return x_out.type_as(x)


# Enhanced Experiment Harness with Pattern Training
class EnhancedExperimentHarness:
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
        
        # Create the enhanced model
        model = EnhancedComplexParadoxNet(
            vocab_size=metadata['vocab_size'],
            embedding_dim=64,
            hidden_dims=[64, 64],
            n_patterns=16
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print("Starting training for Enhanced Complex Paradox Net (with pattern training)...")
        print(f"Pattern loss weight: {self.pattern_loss_weight}")
        
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0
            epoch_task_loss = 0
            epoch_pattern_loss = 0
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass with pattern prediction
                final_output, pattern_errors = model.forward_with_pattern_prediction(X_batch)
                
                # Main task loss
                task_loss = self.criterion(final_output, y_batch.long())
                
                # Pattern prediction loss
                if pattern_errors:
                    pattern_loss = sum(pattern_errors) / len(pattern_errors)
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
                test_output, _ = model.forward_with_pattern_prediction(X_test)
                test_loss = self.criterion(test_output, y_test.long()).item()
                
                avg_train_loss = epoch_train_loss / num_batches
                avg_task_loss = epoch_task_loss / num_batches
                avg_pattern_loss = epoch_pattern_loss / num_batches
            
            print(f"Epoch {epoch}: Total Loss = {avg_train_loss:.4f} "
                  f"(Task: {avg_task_loss:.4f}, Pattern: {avg_pattern_loss:.4f}), "
                  f"Test Loss = {test_loss:.4f}")

        print(f"\nTrial finished in {time.time() - start_time:.2f} seconds.")
        return test_loss

# Comparison function
def compare_with_and_without_patterns():
    """Compare the enhanced version with pattern training vs without."""
    
    print("=" * 60)
    print("COMPARISON: Complex ParadoxNet WITH vs WITHOUT Pattern Training")
    print("=" * 60)
    
    data_generator = get_tiny_shakespeare_data
    
    # Test without pattern training (original)
    print("\n1. WITHOUT Pattern Training:")
    harness_without = ExperimentHarness(data_generator, epochs=30)
    harness_without.run_trial(seed=42)
    
    print("\n" + "=" * 60)
    
    # Test with pattern training (enhanced)
    print("\n2. WITH Pattern Training:")
    harness_with = EnhancedExperimentHarness(data_generator, epochs=30, pattern_loss_weight=0.1)
    test_loss_with_patterns = harness_with.run_trial(seed=42)
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("Check if pattern training improves the complex ParadoxNet!")
    

# Original harness for comparison
class ExperimentHarness:
    def __init__(self, data_generator: Callable, epochs: int = 50, batch_size: int = 32):
        self.data_generator = data_generator
        self.epochs = epochs
        self.batch_size = batch_size
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
        
        # Create the original model
        model = ParadoxNetComplex(
            vocab_size=metadata['vocab_size'],
            embedding_dim=64,
            hidden_dims=[64, 64],
            n_patterns=16
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print("Training original Complex Paradox Net (NO pattern training)...")
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0
            num_batches = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                final_output = model(X_batch)
                task_loss = self.criterion(final_output, y_batch.long())
                
                task_loss.backward()
                optimizer.step()
                epoch_train_loss += task_loss.item()
                num_batches += 1

            model.eval()
            with torch.no_grad():
                test_output = model(X_test)
                test_loss = self.criterion(test_output, y_test.long()).item()
                avg_train_loss = epoch_train_loss / num_batches
            
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}")

        print(f"\nTrial finished in {time.time() - start_time:.2f} seconds.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Run the comparison
    compare_with_and_without_patterns()
    
    print("\n" + "=" * 60)
    print("Want to run more trials? Modify the script to test different:")
    print("- Pattern loss weights (0.05, 0.1, 0.2)")  
    print("- Number of patterns (8, 16, 32)")
    print("- Hidden dimensions ([32,32], [64,64], [128,64])")
    print("=" * 60)