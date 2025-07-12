import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
import json
from scipy import stats

# Assuming these files are in the same directory or accessible in the python path
from paradox_net import DiscretePatternPredictiveNet
from transformer_net import TransformerModel

# --- DataClass Definitions ---

@dataclass
class EpochStats:
    """Statistics for a single epoch"""
    layer_confidences: Dict[int, float]
    layer_pred_errors: Dict[int, float]
    penultimate_flows: Dict[int, float]
    continue_flows: Dict[int, float]
    train_loss: float
    prediction_loss: Optional[float] = None
    pattern_entropy: Optional[Dict[int, float]] = None

@dataclass
class ExperimentResult:
    """Store results from a single experimental run"""
    train_losses: List[float]
    test_losses: List[float]
    final_test_loss: float
    prediction_errors: Optional[List[float]] = None
    epoch_stats: Optional[List[EpochStats]] = None
    model_state_dict: Optional[Dict] = None
    trial_duration: Optional[float] = None

# --- Model Definitions for Language Tasks ---

class StandardNetText(nn.Module):
    """A standard feed-forward network adapted for text classification."""
    def __init__(self, sequence_length, vocab_size, embedding_dim=32, hidden_dims=[48, 96, 48]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dim = sequence_length * embedding_dim
        
        layers = []
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        # Add a penultimate layer and final output layer
        layers.append(nn.Linear(in_dim, 32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(32, vocab_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x is expected to be of shape (batch_size, sequence_length) and type Long
        embedded = self.embedding(x)
        # Flatten the embeddings
        flattened = embedded.view(embedded.size(0), -1)
        return self.network(flattened)

class ParadoxNetText(nn.Module):
    """Wrapper for DiscretePatternPredictiveNet to handle text data."""
    def __init__(self, sequence_length, vocab_size, embedding_dim=16, hidden_dims=[64, 32, 16], n_patterns=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dim = sequence_length * embedding_dim
        
        self.ppn = DiscretePatternPredictiveNet(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            penultimate_dim=32,
            output_dim=vocab_size,
            n_patterns=n_patterns
        )

    def forward(self, x):
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        return self.ppn(flattened)
        
    def get_layer_stats(self):
        return self.ppn.get_layer_stats()

    def update_temperatures(self):
        self.ppn.update_temperatures()


# --- Model Factory Functions ---

def create_standard_net_text(sequence_length, vocab_size, **kwargs):
    """Factory for the standard text model."""
    return StandardNetText(
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        embedding_dim=48, # Matched to transformer d_model for fairness
        hidden_dims=[96, 48]
    )

def create_paradox_net_text(sequence_length, vocab_size, **kwargs):
    """Factory for the ParadoxNet text model."""
    return ParadoxNetText(
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        embedding_dim=16, # Smaller embedding to keep params reasonable
        hidden_dims=[64, 32, 16],
        n_patterns=16
    )

def create_transformer_net_text(sequence_length, vocab_size, **kwargs):
    """Factory for the Transformer text model."""
    return TransformerModel(
        input_dim=sequence_length,
        d_model=48,
        n_heads=3,
        d_ff=96,
        n_layers=3,
        output_dim=vocab_size,
        is_text=True,
        vocab_size=vocab_size
    )


# --- Experiment Harness ---

class ExperimentHarness:
    def __init__(
        self,
        data_generator: Callable,
        model_factory: Callable,
        n_seeds: int = 5,
        epochs: int = 100,
        batch_size: int = 32,
        test_size: float = 0.2,
        eval_frequency: int = 5,
        is_classification: bool = False
    ):
        self.data_generator = data_generator
        self.model_factory = model_factory
        self.n_seeds = n_seeds
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.eval_frequency = eval_frequency
        self.is_classification = is_classification
        self.criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    def run_trial(self, seed: int) -> ExperimentResult:
        """Run a single trial with a given seed."""
        start_time = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        train_loader, test_loader, metadata = self.prepare_data()
        
        # Pass metadata to model factory (e.g., vocab_size)
        model_params = metadata or {}
        model_params['sequence_length'] = train_loader.dataset.tensors[0].shape[1]
        model = self.model_factory(**model_params)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        train_losses, test_losses, prediction_errors, epoch_stats_list = [], [], [], []
        
        for epoch in range(self.epochs):
            model.train()
            epoch_loss, epoch_pred_errors, n_batches = 0, [], 0
            
            for X, y in train_loader:
                optimizer.zero_grad()
                output = model(X)
                
                outputs = output[0] if isinstance(output, tuple) else output
                
                # For classification, target y should be Long and not one-hot
                if self.is_classification:
                    y = y.long()

                task_loss = self.criterion(outputs, y)
                total_loss = task_loss
                
                # Handle PPN-specific prediction error loss
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    pred_loss = 0.1 * torch.mean(output[1])
                    total_loss += pred_loss
                    epoch_pred_errors.append(pred_loss.item())
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += task_loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)
            
            if hasattr(model, 'update_temperatures'):
                model.update_temperatures()
            
            if epoch % self.eval_frequency == 0:
                test_loss = self.evaluate_model(model, test_loader)
                test_losses.append(test_loss)
                print(f"Seed {seed}, Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}")

        final_test_loss = self.evaluate_model(model, test_loader)
        
        return ExperimentResult(
            train_losses=train_losses,
            test_losses=test_losses,
            final_test_loss=final_test_loss,
            trial_duration=time.time() - start_time
        )
        
    def prepare_data(self):
        """Prepare train and test dataloaders."""
        generator_output = self.data_generator()
        metadata = None
        if len(generator_output) == 3:
            X, y, metadata = generator_output
        else:
            X, y = generator_output
        
        n_samples = len(X)
        n_test = int(n_samples * self.test_size)
        indices = torch.randperm(n_samples)
        
        train_indices, test_indices = indices[:-n_test], indices[-n_test:]
        
        train_dataset = TensorDataset(X[train_indices], y[train_indices])
        test_dataset = TensorDataset(X[test_indices], y[test_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_loader, test_loader, metadata

    def evaluate_model(self, model, dataloader) -> float:
        """Evaluate model on a given dataloader."""
        model.eval()
        total_loss, n_batches = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                output = model(X)
                outputs = output[0] if isinstance(output, tuple) else output
                
                if self.is_classification:
                    y = y.long()

                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / n_batches

    def run_experiment(self) -> Dict[int, ExperimentResult]:
        """Run the full experiment over multiple seeds."""
        results = {}
        for seed in range(self.n_seeds):
            print(f"\nRunning trial with seed {seed}")
            results[seed] = self.run_trial(seed)
        return results

# --- Comparison Runner ---

def run_comparison(
        data_generators: Dict[str, Callable],
        model_factories: Dict[str, Callable],
        n_seeds: int = 5,
        epochs: int = 100,
        save_path: str = "experiment_results",
        is_classification: bool = False
    ):
    """
    Run a comparison between different models across various datasets.
    """
    all_results = {
        "experiment_results": {},
        "summary": {}
    }

    for data_name, data_generator in data_generators.items():
        print(f"\n----- Running experiments for {data_name} data -----")
        all_results["experiment_results"][data_name] = {}
        all_results["summary"][data_name] = {}
        
        losses = {}
        
        for model_name, model_factory in model_factories.items():
            print(f"\n  --- Running {model_name} model ---")
            harness = ExperimentHarness(
                data_generator=data_generator,
                model_factory=model_factory,
                n_seeds=n_seeds,
                epochs=epochs,
                is_classification=is_classification
            )
            results = harness.run_experiment()

            # Process and store results
            trial_losses = [r.final_test_loss for r in results.values()]
            trial_durations = [r.trial_duration for r in results.values()]
            
            losses[model_name] = trial_losses
            
            all_results["experiment_results"][data_name][model_name] = {
               "losses": trial_losses,
               "mean_loss": float(np.mean(trial_losses)),
               "std_loss": float(np.std(trial_losses)),
                "trial_durations": trial_durations,
                "mean_duration": float(np.mean(trial_durations)),
                "std_duration": float(np.std(trial_durations)),
             }

        # Perform t-tests between models
        model_names = list(model_factories.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                t_stat, p_value = stats.ttest_ind(losses[model1], losses[model2])
                all_results["summary"][data_name][f"{model1}_vs_{model2}"] = {
                    "t_stat": float(t_stat),
                    "p_value": float(p_value)
                }

    # Save results to file
    try:
        with open(f"{save_path}/detailed_experiment_results.json", 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nDetailed results saved to {save_path}/detailed_experiment_results.json")
    except Exception as e:
        print(f"Error saving detailed results to file: {e}")
