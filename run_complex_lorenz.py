import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional
from torch.utils.data import DataLoader, TensorDataset

# Import the new, regression-adapted complex model
from paradox_net_complex_regression import ParadoxNetComplexRegression
from data_generators import generate_lorenz_data

# --- Minimal Experiment Harness for Regression ---
class ExperimentHarness:
    def __init__(self, data_generator: Callable, epochs: int = 100, batch_size: int = 32):
        self.data_generator = data_generator
        self.epochs = epochs
        self.batch_size = batch_size
        # REGRESSION CHANGE: Use Mean Squared Error Loss
        self.criterion = nn.MSELoss()

    def run_trial(self, seed: int):
        start_time = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X, y = self.data_generator()
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        
        # Create the model
        model = ParadoxNetComplexRegression(
            input_dim=1, # Lorenz data has 1 feature per time step
            embedding_dim=64,
            hidden_dims=[64, 64],
            output_dim=1, # Predict a single value
            n_patterns=16
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print("Starting Lorenz sanity check for the Complex-Valued Paradox Net...")
        for epoch in range(self.epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                final_output = model(X_batch)
                loss = self.criterion(final_output, y_batch)
                loss.backward()
                optimizer.step()

            # Simple evaluation at the end of each epoch
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_output = model(X_test)
                    test_loss = self.criterion(test_output, y_test).item()
                print(f"Epoch {epoch}: Test Loss = {test_loss:.6f}")

        model.eval()
        with torch.no_grad():
            final_test_output = model(X_test)
            final_test_loss = self.criterion(final_test_output, y_test).item()
        print(f"\nSanity Check Finished. Final Test Loss: {final_test_loss:.6f}")
        print(f"Trial finished in {time.time() - start_time:.2f} seconds.")

# --- Main Execution Block ---
if __name__ == "__main__":
    data_generator = generate_lorenz_data
    harness = ExperimentHarness(data_generator, epochs=100)
    harness.run_trial(seed=0)
