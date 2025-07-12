import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
# CORRECTED: Added missing imports from the typing module
from typing import Dict, List, Callable, Optional
import json
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

# Import the new temporal model
from paradox_net_temporal import DiscretePatternPredictiveNet
# Keep original transformer for comparison
from transformer_net import TransformerModel
from data_generators import get_tiny_shakespeare_data

# --- DataClass Definitions (same as before) ---
@dataclass
class ExperimentResult:
    train_losses: List[float]
    test_losses: List[float]
    final_test_loss: float
    trial_duration: Optional[float] = None

# --- Model Definitions for Language Tasks ---

class StandardNetText(nn.Module):
    def __init__(self, sequence_length, vocab_size, embedding_dim=32, hidden_dims=[48, 96, 48]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dim = sequence_length * embedding_dim
        layers = []
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.extend([nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, vocab_size)])
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        return self.network(flattened)

class ParadoxNetText(nn.Module):
    """Wrapper for the NEW Temporal DiscretePatternPredictiveNet."""
    def __init__(self, sequence_length, vocab_size, embedding_dim=16, hidden_dims=[64, 32, 16], n_patterns=8, temporal_lr=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dim = sequence_length * embedding_dim
        self.ppn = DiscretePatternPredictiveNet(
            input_dim=self.input_dim, hidden_dims=hidden_dims,
            penultimate_dim=32, output_dim=vocab_size, n_patterns=n_patterns,
            # Pass the temporal learning rate to the core model
            temporal_lr=temporal_lr
        )
    def forward(self, x):
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        return self.ppn(flattened)
    # Expose the new method from the core model
    def update_temporal_temperatures(self):
        self.ppn.update_temporal_temperatures()
    def update_base_temperatures(self):
        self.ppn.update_base_temperatures()

# --- Model Factory Functions ---

def create_standard_net_text(sequence_length, vocab_size, **kwargs):
    return StandardNetText(sequence_length=sequence_length, vocab_size=vocab_size, embedding_dim=48, hidden_dims=[96, 48])

def create_paradox_net_temporal(sequence_length, vocab_size, **kwargs):
    """Factory for the NEW ParadoxNet with the temporal loop."""
    return ParadoxNetText(
        sequence_length=sequence_length, vocab_size=vocab_size,
        embedding_dim=16, hidden_dims=[64, 32, 16], n_patterns=16,
        # Define the temporal learning rate here. This is a key hyperparameter to tune.
        temporal_lr=0.5
    )

def create_transformer_net_text(sequence_length, vocab_size, **kwargs):
    return TransformerModel(
        input_dim=sequence_length, d_model=48, n_heads=3, d_ff=96, n_layers=3,
        output_dim=vocab_size, is_text=True, vocab_size=vocab_size
    )

# --- Experiment Harness ---

class ExperimentHarness:
    def __init__(self, data_generator: Callable, model_factory: Callable, n_seeds: int = 5, epochs: int = 100, batch_size: int = 32, is_classification: bool = False):
        self.data_generator = data_generator
        self.model_factory = model_factory
        self.n_seeds = n_seeds
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_classification = is_classification
        self.criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    def run_trial(self, seed: int) -> ExperimentResult:
        start_time = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X, y, metadata = self.data_generator()
        # Simple train/test split for this example
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        
        model_params = metadata or {}
        model_params['sequence_length'] = X.shape[1]
        model = self.model_factory(**model_params)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        train_losses, test_losses = [], []
        
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0
            num_batches = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                outputs = output[0] if isinstance(output, tuple) else output
                task_loss = self.criterion(outputs, y_batch.long())
                total_loss = task_loss
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    total_loss += 0.1 * torch.mean(output[1])
                total_loss.backward()
                optimizer.step()
                epoch_train_loss += total_loss.item()
                num_batches += 1

            # --- KEY CHANGE: Update temperatures at the end of the epoch ---
            if hasattr(model, 'update_base_temperatures'):
                model.update_base_temperatures()
            if hasattr(model, 'update_temporal_temperatures'):
                model.update_temporal_temperatures()
            # --- End of Key Change ---

            # Evaluation
            model.eval()
            with torch.no_grad():
                test_output = model(X_test)
                test_outputs = test_output[0] if isinstance(test_output, tuple) else test_output
                test_loss = self.criterion(test_outputs, y_test.long()).item()
                
                avg_train_loss = epoch_train_loss / num_batches
                train_losses.append(avg_train_loss)
                test_losses.append(test_loss)
            print(f"Seed {seed}, Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}")

        final_test_loss = test_losses[-1]
        return ExperimentResult(train_losses, test_losses, final_test_loss, time.time() - start_time)

# --- Comparison Runner (simplified for this example) ---
def run_comparison(data_generators, model_factories, n_seeds, epochs, save_path):
    # This function would be similar to your original, running the harness
    # for each model and saving the results.
    print("Running comparison...")
    all_results = {}
    for data_name, data_gen in data_generators.items():
        all_results[data_name] = {}
        for model_name, model_fac in model_factories.items():
            print(f"\n--- Running {model_name} on {data_name} ---")
            harness = ExperimentHarness(data_gen, model_fac, n_seeds, epochs, is_classification=True)
            # A full implementation would collect and save results here.
            # For now, we just run one trial to demonstrate.
            results = harness.run_trial(seed=0)
            all_results[data_name][model_name] = results.final_test_loss
    print("\n--- Final Test Losses ---")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    data_generators = {"tiny_shakespeare": get_tiny_shakespeare_data}
    model_factories = {
        "paradox_temporal": create_paradox_net_temporal,
        "transformer": create_transformer_net_text
    }
    run_comparison(data_generators, model_factories, n_seeds=1, epochs=50, save_path="temporal_results")
