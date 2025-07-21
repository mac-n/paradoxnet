import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
import json
from scipy import stats

# Import our new complex pattern networks
from paradox_net_complex_patterns_2025_07_15 import (
    create_complex_patterns_softmax,
    create_complex_patterns_gumbel, 
    create_complex_patterns_no_consensus
)

# Import comparison models
from transformer_net import TransformerModel
from paradox_net_complex import ParadoxNetComplex

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
    paradox_magnitudes: Optional[Dict[int, float]] = None

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

class StandardNetText(nn.Module):
    """A standard feed-forward network adapted for text classification."""
    def __init__(self, sequence_length, vocab_size, embedding_dim=32, hidden_dims=[48, 96, 48]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dim = sequence_length * embedding_dim
        
        layers = []
        current_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, vocab_size))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        return self.layers(flattened)

def create_datasets(data_array, sequence_length, device):
    """Create datasets for character-level language modeling."""
    sequences = []
    targets = []
    
    for i in range(len(data_array) - sequence_length):
        seq = data_array[i:i+sequence_length]
        target = data_array[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    sequences = torch.tensor(sequences, dtype=torch.long).to(device)
    targets = torch.tensor(targets, dtype=torch.long).to(device)
    
    # Split into train/test
    split_idx = int(0.8 * len(sequences))
    train_dataset = TensorDataset(sequences[:split_idx], targets[:split_idx])
    test_dataset = TensorDataset(sequences[split_idx:], targets[split_idx:])
    
    return train_dataset, test_dataset

def train_epoch(model, train_loader, optimizer, criterion, device, collect_stats=False):
    """Train for one epoch with optional statistics collection."""
    model.train()
    total_loss = 0
    total_pred_loss = 0
    epoch_stats = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Forward pass - handle models that return prediction errors
        output = model(data)
        pred_errors = None
        
        if isinstance(output, tuple):
            output, pred_errors = output
        
        # Main task loss
        task_loss = criterion(output, target)
        
        # Add prediction error loss if available
        total_loss_batch = task_loss
        if pred_errors is not None:
            pred_loss = 0.1 * torch.mean(pred_errors)
            total_loss_batch = task_loss + pred_loss
            total_pred_loss += pred_loss.item()
        
        total_loss_batch.backward()
        optimizer.step()
        total_loss += task_loss.item()
        
        # Collect detailed statistics if requested
        if collect_stats and hasattr(model, 'hidden_layers'):
            layer_stats = {}
            for i, layer in enumerate(model.hidden_layers):
                if hasattr(layer, 'last_stats') and layer.last_stats:
                    stats = layer.last_stats
                    layer_stats[i] = {
                        'confidence': float(torch.mean(stats.confidence_values)),
                        'pred_error': float(torch.mean(stats.prediction_errors)),
                        'penultimate_flow': float(stats.penultimate_magnitude),
                        'continue_flow': float(stats.continue_magnitude),
                        'pattern_entropy': float(stats.pattern_entropy),
                        'paradox_magnitude': float(stats.self_paradox_magnitude)
                    }
            
            if layer_stats:
                epoch_stats.append(layer_stats)
    
    avg_loss = total_loss / len(train_loader)
    avg_pred_loss = total_pred_loss / len(train_loader) if total_pred_loss > 0 else None
    
    return avg_loss, avg_pred_loss, epoch_stats

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Handle models that return prediction errors
            if isinstance(output, tuple):
                output, _ = output
            
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def run_single_experiment(model_fn, data_array, sequence_length, device, 
                         epochs=100, lr=0.001, batch_size=32, collect_stats=False, 
                         patience=20, min_delta=0.001):
    """Run a single experiment with early stopping."""
    start_time = time.time()
    
    # Create datasets
    train_dataset, test_dataset = create_datasets(data_array, sequence_length, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    vocab_size = len(set(data_array))
    model = model_fn(vocab_size, sequence_length).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    epoch_stats_list = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss, pred_loss, epoch_stats = train_epoch(
            model, train_loader, optimizer, criterion, device, collect_stats
        )
        
        # Evaluate
        test_loss = evaluate_model(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if collect_stats and epoch_stats:
            # Aggregate statistics across batches
            aggregated_stats = {}
            for layer_idx in epoch_stats[0].keys():
                aggregated_stats[layer_idx] = {
                    'confidence': np.mean([batch[layer_idx]['confidence'] for batch in epoch_stats]),
                    'pred_error': np.mean([batch[layer_idx]['pred_error'] for batch in epoch_stats]),
                    'penultimate_flow': np.mean([batch[layer_idx]['penultimate_flow'] for batch in epoch_stats]),
                    'continue_flow': np.mean([batch[layer_idx]['continue_flow'] for batch in epoch_stats]),
                    'pattern_entropy': np.mean([batch[layer_idx]['pattern_entropy'] for batch in epoch_stats]),
                    'paradox_magnitude': np.mean([batch[layer_idx]['paradox_magnitude'] for batch in epoch_stats])
                }
            
            epoch_stats_obj = EpochStats(
                layer_confidences={k: v['confidence'] for k, v in aggregated_stats.items()},
                layer_pred_errors={k: v['pred_error'] for k, v in aggregated_stats.items()},
                penultimate_flows={k: v['penultimate_flow'] for k, v in aggregated_stats.items()},
                continue_flows={k: v['continue_flow'] for k, v in aggregated_stats.items()},
                train_loss=train_loss,
                prediction_loss=pred_loss,
                pattern_entropy={k: v['pattern_entropy'] for k, v in aggregated_stats.items()},
                paradox_magnitudes={k: v['paradox_magnitude'] for k, v in aggregated_stats.items()}
            )
            epoch_stats_list.append(epoch_stats_obj)
        
        # Early stopping
        if test_loss < best_loss - min_delta:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")
    
    duration = time.time() - start_time
    
    return ExperimentResult(
        train_losses=train_losses,
        test_losses=test_losses,
        final_test_loss=best_loss,
        epoch_stats=epoch_stats_list if collect_stats else None,
        trial_duration=duration
    )

def compare_models(data_array, sequence_length, n_trials=5, epochs=100, device='cuda'):
    """Compare different model architectures."""
    
    def create_standard_net(vocab_size, seq_length):
        return StandardNetText(seq_length, vocab_size, embedding_dim=32, hidden_dims=[48, 96, 48])
    
    def create_transformer(vocab_size, seq_length):
        return TransformerModel(vocab_size, embedding_dim=32, num_heads=4, 
                              num_layers=3, dim_feedforward=64, max_seq_length=seq_length)
    
    def create_original_complex(vocab_size, seq_length):
        return ParadoxNetComplex(vocab_size, embedding_dim=32, hidden_dims=[48, 96])
    
    def create_complex_softmax(vocab_size, seq_length):
        return create_complex_patterns_softmax(vocab_size, embedding_dim=32, 
                                             hidden_dims=[48, 96], penultimate_dim=48, n_patterns=8)
    
    def create_complex_gumbel(vocab_size, seq_length):
        return create_complex_patterns_gumbel(vocab_size, embedding_dim=32,
                                            hidden_dims=[48, 96], penultimate_dim=48, 
                                            n_patterns=8, gumbel_temp=1.0)
    
    def create_complex_no_consensus(vocab_size, seq_length):
        return create_complex_patterns_no_consensus(vocab_size, embedding_dim=32,
                                                  hidden_dims=[48, 96], penultimate_dim=48, n_patterns=8)
    
    models = {
        'Standard Feed-Forward': create_standard_net,
        'Transformer': create_transformer,
        'Original Complex ParadoxNet': create_original_complex,
        'Complex + Patterns (Softmax)': create_complex_softmax,
        'Complex + Patterns (Gumbel)': create_complex_gumbel,
        'Complex + Patterns (No Consensus)': create_complex_no_consensus
    }
    
    results = {}
    
    for model_name, model_fn in models.items():
        print(f"\nTesting {model_name}...")
        model_results = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}")
            try:
                result = run_single_experiment(
                    model_fn, data_array, sequence_length, device, 
                    epochs=epochs, collect_stats=True
                )
                model_results.append(result.final_test_loss)
                print(f"    Final test loss: {result.final_test_loss:.4f}")
            except Exception as e:
                print(f"    Trial failed: {e}")
                continue
        
        if model_results:
            results[model_name] = {
                'mean_loss': np.mean(model_results),
                'std_loss': np.std(model_results),
                'all_results': model_results
            }
            print(f"  Average: {results[model_name]['mean_loss']:.4f} ± {results[model_name]['std_loss']:.4f}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Test with a small Shakespeare sample
    import requests
    
    # Download Shakespeare if not available
    try:
        with open('shakespeare.txt', 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print("Downloading Shakespeare...")
        url = "https://www.gutenberg.org/files/100/100-0.txt"
        text = requests.get(url).text
        with open('shakespeare.txt', 'w') as f:
            f.write(text)
    
    # Prepare data
    chars = sorted(list(set(text)))
    char_to_int = {char: i for i, char in enumerate(chars)}
    data_array = [char_to_int[char] for char in text[:10000]]  # Use first 10k chars for testing
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run comparison
    results = compare_models(data_array, sequence_length=20, n_trials=3, epochs=50, device=device)
    
    # Save results
    with open('complex_patterns_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nFinal Results:")
    for model_name, result in results.items():
        print(f"{model_name}: {result['mean_loss']:.4f} ± {result['std_loss']:.4f}")