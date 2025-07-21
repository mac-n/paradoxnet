import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional
from torch.utils.data import DataLoader, TensorDataset

# Import our new routing version
from paradox_net_complex_with_attention_2025_07_15 import ParadoxNetComplex
from data_generators import get_tiny_shakespeare_data

# --- Minimal Experiment Harness ---
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
        
        # Create the model with routing
        model = ParadoxNetComplex(
            vocab_size=metadata['vocab_size'],
            embedding_dim=64,
            hidden_dims=[64, 64],
            n_patterns=16
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print("Starting training for the Complex-Valued Paradox Net with Routing...")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0
            epoch_pred_loss = 0
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass returns output and prediction errors
                final_output, pred_errors = model(X_batch)
                
                # Main task loss
                task_loss = self.criterion(final_output, y_batch.long())
                total_loss = task_loss
                
                # Add prediction error loss
                if pred_errors is not None:
                    pred_loss = 0.1 * torch.mean(pred_errors)
                    total_loss = task_loss + pred_loss
                    epoch_pred_loss += pred_loss.item()
                
                total_loss.backward()
                optimizer.step()
                epoch_train_loss += task_loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_train_loss / num_batches
            avg_pred_loss = epoch_pred_loss / num_batches
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Task Loss = {avg_train_loss:.4f}, Pred Loss = {avg_pred_loss:.6f}")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            X_test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)
            test_loss = 0
            num_test_batches = 0
            
            for X_batch, y_batch in X_test_loader:
                final_output, _ = model(X_batch)
                batch_loss = self.criterion(final_output, y_batch.long())
                test_loss += batch_loss.item()
                num_test_batches += 1
            
            avg_test_loss = test_loss / num_test_batches
            print(f"Final test loss: {avg_test_loss:.4f}")
            
            return avg_test_loss

# --- Run the experiment ---
if __name__ == "__main__":
    print("Complex Paradox Net with Routing on Tiny Shakespeare")
    print("=" * 50)
    
    # Check if data exists
    try:
        harness = ExperimentHarness(get_tiny_shakespeare_data, epochs=50, batch_size=32)
        
        # Run a few trials
        results = []
        for seed in [42, 123, 456]:
            print(f"\nRunning trial with seed {seed}...")
            result = harness.run_trial(seed)
            results.append(result)
            print(f"Trial result: {result:.4f}")
        
        # Summary
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print(f"Mean test loss: {np.mean(results):.4f}")
        print(f"Std test loss: {np.std(results):.4f}")
        print(f"All results: {results}")
        
        # Compare to original complex net result
        print("\nCompare to original complex net result (from your logs):")
        print("Original complex net: ~2.5753 (stable learning)")
        print("Our routing version: {:.4f}".format(np.mean(results)))
        
        if np.mean(results) < 2.5753:
            print("🎉 IMPROVEMENT! Routing + patterns helped!")
        else:
            print("🤔 Similar performance - but now with interpretable routing!")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()