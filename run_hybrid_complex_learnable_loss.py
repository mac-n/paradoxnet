


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional
from torch.utils.data import DataLoader, TensorDataset

# Import the new hybrid model
from paradox_net_hybrid_complex import ParadoxNetHybridComplex
from data_generators import get_tiny_shakespeare_data

# --- Minimal Experiment Harness ---
class ExperimentHarness:
    def __init__(self, data_generator: Callable, epochs: int = 50, batch_size: int = 32):
        self.data_generator = data_generator
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.log_pred_loss_weight = nn.Parameter(torch.tensor(0.0))

    def run_trial(self, seed: int):
        start_time = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X, y, metadata = self.data_generator()
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        
        # Create the hybrid model
        model = ParadoxNetHybridComplex(
            vocab_size=metadata['vocab_size'],
            embedding_dim=64,
            hidden_dims=[64, 64],
            penultimate_dim=64, # Added penultimate_dim
            n_patterns=16
        )
        optimizer = torch.optim.Adam(list(model.parameters()) + [self.log_pred_loss_weight], lr=1e-3)
        
        print("Starting training for the Hybrid Complex Paradox Net...")
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0
            num_batches = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                final_output, pred_errors = model(X_batch)
                
                task_loss = self.criterion(final_output, y_batch.long())
                
                # Add prediction error to the loss
                if pred_errors is not None:
                    pred_loss_weight = torch.exp(self.log_pred_loss_weight)
                    pred_loss = pred_loss_weight * torch.mean(pred_errors)
                    total_loss = task_loss + pred_loss
                else:
                    total_loss = task_loss
                
                total_loss.backward()
                optimizer.step()
                epoch_train_loss += total_loss.item()
                num_batches += 1

            model.eval()
            with torch.no_grad():
                test_output, _ = model(X_test)
                test_loss = self.criterion(test_output, y_test.long()).item()
                avg_train_loss = epoch_train_loss / num_batches
            
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}, Pred Loss Weight: {torch.exp(self.log_pred_loss_weight).item():.4f}")

        print(f"\nTrial finished in {time.time() - start_time:.2f} seconds.")

# --- Main Execution Block ---
if __name__ == "__main__":
    data_generator = get_tiny_shakespeare_data
    harness = ExperimentHarness(data_generator, epochs=50)
    harness.run_trial(seed=0)


