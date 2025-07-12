import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional
from torch.utils.data import DataLoader, TensorDataset

# Import the new, complete architecture
from complete_paradox_net_temporal import CompleteParadoxNetTemporal
from data_generators import get_tiny_shakespeare_data

# --- DataClass for Results ---
@dataclass
class ExperimentResult:
    train_losses: List[float]
    test_losses: List[float]
    final_test_loss: float
    trial_duration: Optional[float] = None

# --- Model Wrapper ---
class ParadoxNetText(nn.Module):
    """A wrapper to handle the text embedding for the new CompleteParadoxNetTemporal."""
    def __init__(self, sequence_length, vocab_size, embedding_dim=16, hidden_dims=[64, 32], penultimate_dim=32, n_patterns=8, temporal_lr=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dim = sequence_length * embedding_dim
        
        self.ppn = CompleteParadoxNetTemporal(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            penultimate_dim=penultimate_dim,
            output_dim=vocab_size,
            n_patterns=n_patterns,
            temporal_lr=temporal_lr
        )

    def forward(self, x, y):
        """Note: The forward pass now requires the target 'y'."""
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        # Pass both x (flattened) and y to the core model
        return self.ppn(flattened, y)

    def update_temporal_temperatures(self):
        """Expose the temporal update method."""
        self.ppn.update_temporal_temperatures()

# --- Model Factory ---
def create_complete_paradox_net(sequence_length, vocab_size, **kwargs):
    """Factory function for the new complete architecture."""
    return ParadoxNetText(
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        embedding_dim=16,
        hidden_dims=[64, 32],
        penultimate_dim=32,
        n_patterns=16,
        temporal_lr=0.5
    )

# --- Minimal Experiment Harness ---
class ExperimentHarness:
    def __init__(self, data_generator: Callable, model_factory: Callable, epochs: int = 50, batch_size: int = 32):
        self.data_generator = data_generator
        self.model_factory = model_factory
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
        
        model_params = metadata or {}
        model_params['sequence_length'] = X.shape[1]
        model = self.model_factory(**model_params)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print("Starting training for the Complete Paradox Net...")
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0
            num_batches = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # KEY CHANGE: Pass y_batch to the model's forward pass
                final_output, pred_errors = model(X_batch, y_batch)
                
                # The main task loss is still CrossEntropy on the final output
                task_loss = self.criterion(final_output, y_batch.long())
                
                # The total loss now also includes the prediction errors from all layers
                total_loss = task_loss + 0.1 * torch.mean(pred_errors)
                
                total_loss.backward()
                optimizer.step()
                epoch_train_loss += total_loss.item()
                num_batches += 1

            # Update temporal temperatures at the end of the epoch
            if hasattr(model, 'update_temporal_temperatures'):
                model.update_temporal_temperatures()

            # Evaluation
            model.eval()
            with torch.no_grad():
                # Pass y_test for the evaluation forward pass
                test_output, _ = model(X_test, y_test)
                test_loss = self.criterion(test_output, y_test.long()).item()
                avg_train_loss = epoch_train_loss / num_batches
            
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}")

        print(f"\nTrial finished in {time.time() - start_time:.2f} seconds.")

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Define the data generator
    data_generator = get_tiny_shakespeare_data
    
    # 2. Define the model factory for our new model
    model_factory = create_complete_paradox_net
    
    # 3. Create and run the harness
    harness = ExperimentHarness(data_generator, model_factory, epochs=50)
    harness.run_trial(seed=0)
