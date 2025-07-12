import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional
from torch.utils.data import DataLoader, TensorDataset

# Import the new, complex-valued model
from paradox_net_complex import ParadoxNetComplex
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
        
        # Create the model
        model = ParadoxNetComplex(
            vocab_size=metadata['vocab_size'],
            embedding_dim=64,
            hidden_dims=[64, 64],
            n_patterns=16
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print("Starting training for the Complex-Valued Paradox Net...")
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0
            num_batches = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                final_output = model(X_batch)
                
                # The final output is now the magnitude of the complex logits
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
    data_generator = get_tiny_shakespeare_data
    harness = ExperimentHarness(data_generator, epochs=50)
    harness.run_trial(seed=0)