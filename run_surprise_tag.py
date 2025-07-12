import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional
from torch.utils.data import DataLoader, TensorDataset

# Import the new, surprise-tagged model
from complete_paradox_net_surprise_tag import CompleteParadoxNetSurpriseTag
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
    """Wrapper for the new surprise-tagged model."""
    def __init__(self, sequence_length, vocab_size, embedding_dim=16, hidden_dims=[64, 32], penultimate_dim=32, n_patterns=8, temporal_lr=0.1, temp_lr=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dim = sequence_length * embedding_dim
        
        self.ppn = CompleteParadoxNetSurpriseTag(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            penultimate_dim=penultimate_dim,
            output_dim=vocab_size,
            n_patterns=n_patterns,
            temporal_lr=temporal_lr,
            temp_lr=temp_lr
        )

    def forward(self, x, y):
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        return self.ppn(flattened, y)

    def update_temporal_temperatures(self):
        self.ppn.update_temporal_temperatures()

# --- Model Factory ---
def create_surprise_tag_net(sequence_length, vocab_size, **kwargs):
    return ParadoxNetText(
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        embedding_dim=16,
        hidden_dims=[64, 32],
        penultimate_dim=32,
        n_patterns=16,
        temporal_lr=0.5,
        temp_lr=0.1
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
        
        print("Starting training for the Surprise-Tagged Paradox Net...")
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0
            num_batches = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                final_output, pred_errors = model(X_batch, y_batch)
                task_loss = self.criterion(final_output, y_batch.long())
                total_loss = task_loss + 0.1 * torch.mean(pred_errors)
                total_loss.backward()
                optimizer.step()
                epoch_train_loss += total_loss.item()
                num_batches += 1

            if hasattr(model, 'update_temporal_temperatures'):
                model.update_temporal_temperatures()

            model.eval()
            with torch.no_grad():
                test_output, _ = model(X_test, y_test)
                test_loss = self.criterion(test_output, y_test.long()).item()
                avg_train_loss = epoch_train_loss / num_batches
            
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}")

        print(f"\nTrial finished in {time.time() - start_time:.2f} seconds.")

# --- Main Execution Block ---
if __name__ == "__main__":
    data_generator = get_tiny_shakespeare_data
    model_factory = create_surprise_tag_net
    harness = ExperimentHarness(data_generator, model_factory, epochs=50)
    harness.run_trial(seed=0)



