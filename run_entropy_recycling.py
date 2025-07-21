import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import json
import math
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset

# Import the entropy recycling architecture
from entropy_recycling_compositional import EntropyRecyclingNet, EntropyRecyclingLayer
from data_generators import get_tiny_shakespeare_data

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].squeeze(1).unsqueeze(0)
        return x

# Enhanced penultimate layer with entropy recycling compatibility
class EntropyRecyclingPenultimateLayer(nn.Module):
    """Penultimate layer compatible with entropy recycling system"""
    
    def __init__(self, penultimate_dim, output_dim, n_patterns=8, prev_layer=None):
        super().__init__()
        
        self.penultimate_dim = penultimate_dim
        self.output_dim = output_dim
        self.n_patterns = n_patterns
        
        # Compositional patterns from previous layer
        if prev_layer is not None:
            self.composition_weights = nn.Parameter(
                torch.randn(n_patterns, prev_layer.n_patterns) / (prev_layer.n_patterns ** 0.5)
            )
            self.prev_layer = prev_layer
            self.pattern_dict = None
        else:
            self.pattern_dict = nn.Parameter(
                torch.randn(n_patterns, penultimate_dim) / (penultimate_dim ** 0.5)
            )
            self.composition_weights = None
            self.prev_layer = None
        
        # Pattern attention
        self.pattern_attention = nn.Linear(penultimate_dim, n_patterns)
        
        # Self-prediction paradox mechanism
        self.self_predictor = nn.Linear(penultimate_dim, penultimate_dim)
        
        # Final output
        self.output_predictor = nn.Linear(penultimate_dim, output_dim)
    
    def get_pattern_dict(self):
        """Get compositional pattern dictionary"""
        if self.composition_weights is not None and self.prev_layer is not None:
            prev_patterns = self.prev_layer.get_pattern_dict()
            composed_patterns = self.composition_weights @ prev_patterns
            return composed_patterns
        else:
            return self.pattern_dict
    
    def forward(self, penultimate_input: torch.Tensor) -> torch.Tensor:
        """Forward pass with patterns + paradox on penultimate"""
        # Extract patterns
        patterns = self.get_pattern_dict()
        attn = self.pattern_attention(penultimate_input)
        pattern_weights = F.softmax(attn, dim=-1)
        pattern_reconstruction = pattern_weights @ patterns
        
        # Apply paradox mechanism
        self_prediction = self.self_predictor(pattern_reconstruction)
        paradox = self_prediction - pattern_reconstruction
        paradox_magnitude = torch.norm(paradox, dim=-1, keepdim=True)
        
        # "I'm confused about myself → let more through"
        gated_patterns = pattern_reconstruction * torch.sigmoid(paradox_magnitude)
        
        # Final output
        output = self.output_predictor(gated_patterns)
        return output

class EntropyRecyclingLanguageNet(nn.Module):
    """Complete entropy recycling network for language modeling"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Create entropy recycling layers
        self.layers = nn.ModuleList()
        current_dim = embedding_dim
        
        prev_layer = None
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else hidden_dim
            
            layer = EntropyRecyclingLayer(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                next_dim=next_dim,
                penultimate_dim=hidden_dim,  # Match for consistency
                n_patterns=n_patterns,
                composition_from_prev=(i > 0),
                prev_layer=prev_layer,
                is_bottom=(i == 0)
            )
            self.layers.append(layer)
            prev_layer = layer
            current_dim = hidden_dim
        
        # Enhanced penultimate layer with patterns + paradox
        self.penultimate_layer = EntropyRecyclingPenultimateLayer(
            penultimate_dim=hidden_dims[-1],
            output_dim=vocab_size,
            n_patterns=n_patterns,
            prev_layer=self.layers[-1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for language modeling"""
        # Embedding + positional encoding
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        embedded = self.pos_encoder(embedded)  # Add positional encoding
        batch_size, seq_len, embed_dim = embedded.shape
        
        # Process sequence
        penultimate_contributions = []
        current = embedded.mean(dim=1)  # Mean pooling after positional encoding
        all_errors = []
        all_entropy = []
        
        # First pass: collect entropy from all layers
        for i, layer in enumerate(self.layers):
            next_layer = self.layers[i+1] if i < len(self.layers)-1 else None
            
            if i == 0:
                current, penultimate, error, entropy = layer(current, next_layer, i, accumulated_entropy=None)
            else:
                current, penultimate, error, entropy = layer(current, next_layer, i)
            
            if error is not None:
                all_errors.append(error)
            penultimate_contributions.append(penultimate)
            
            # Collect entropy for recycling (except from layer 0)
            if i > 0:
                all_entropy.append(entropy)
        
        # Second pass: Layer 0 processes accumulated entropy
        if all_entropy:
            # Project all entropy to Layer 0's dimension and sum
            layer_0_dim = self.layers[0].hidden_dim
            projected_entropy = []
            
            for i, entropy in enumerate(all_entropy):
                if entropy.size(-1) != layer_0_dim:
                    layer_idx = i + 1
                    projector_name = f'entropy_projector_to_0_from_{layer_idx}'
                    if not hasattr(self, projector_name):
                        projector = nn.Linear(entropy.size(-1), layer_0_dim).to(entropy.device)
                        setattr(self, projector_name, projector)
                    else:
                        projector = getattr(self, projector_name)
                    entropy = projector(entropy)
                projected_entropy.append(entropy)
            
            total_entropy = torch.sum(torch.stack(projected_entropy), dim=0)
            
            # Re-process Layer 0 with accumulated entropy
            layer_0 = self.layers[0]
            next_layer_0 = self.layers[1] if len(self.layers) > 1 else None
            _, penultimate_0_enhanced, error_0_enhanced, _ = layer_0(
                embedded.mean(dim=1), next_layer_0, 0, accumulated_entropy=total_entropy
            )
            
            # Replace Layer 0's contribution with enhanced version
            penultimate_contributions[0] = penultimate_0_enhanced
        
        # Combine penultimate contributions
        penultimate = torch.sum(torch.stack(penultimate_contributions), dim=0)
        
        # Final prediction through enhanced penultimate layer
        output = self.penultimate_layer(penultimate)
        
        return output

@dataclass
class ExperimentConfig:
    """Configuration for entropy recycling experiments"""
    vocab_size: int = 128
    embedding_dim: int = 64
    hidden_dims: List[int] = None
    n_patterns: int = 8
    sequence_length: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64, 64]

def run_entropy_recycling_experiment(config: ExperimentConfig):
    """Run entropy recycling language modeling experiment"""
    
    print(f"🔄 ENTROPY RECYCLING LANGUAGE EXPERIMENT 🔄")
    print(f"Device: {config.device}")
    print(f"Hidden dims: {config.hidden_dims}")
    print(f"Patterns: {config.n_patterns}")
    
    # Load data
    print("Loading Tiny Shakespeare...")
    X, y, metadata = get_tiny_shakespeare_data(sequence_length=config.sequence_length)
    vocab_size = metadata["vocab_size"]
    config.vocab_size = vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # Create train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
    
    train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model = EntropyRecyclingLanguageNet(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        n_patterns=config.n_patterns
    ).to(config.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    results = {
        "train_losses": [],
        "test_losses": [],
        "entropy_stats": [],
        "config": config.__dict__
    }
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []
        epoch_entropy_stats = []
        
        for batch_idx, (sequences, targets) in enumerate(train_data):
            sequences = sequences.to(config.device)
            targets = targets.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Collect entropy statistics every 10 batches
            if batch_idx % 10 == 0:
                entropy_stats = []
                for i, layer in enumerate(model.layers):
                    if hasattr(layer, 'last_stats') and layer.last_stats:
                        stats = layer.last_stats
                        entropy_stats.append({
                            'layer': i,
                            'entropy_magnitude': stats.entropy_magnitude,
                            'pattern_entropy': stats.pattern_entropy,
                            'composition_alpha': stats.composition_alpha,
                            'paradox_magnitude': stats.self_paradox_magnitude
                        })
                epoch_entropy_stats.append(entropy_stats)
        
        # Testing
        model.eval()
        test_losses = []
        with torch.no_grad():
            for sequences, targets in test_data:
                sequences = sequences.to(config.device)
                targets = targets.to(config.device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                test_losses.append(loss.item())
        
        # Record results
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        results["train_losses"].append(train_loss)
        results["test_losses"].append(test_loss)
        results["entropy_stats"].append(epoch_entropy_stats[-1] if epoch_entropy_stats else [])
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Train={train_loss:.4f}, Test={test_loss:.4f}")
            
            # Print entropy statistics
            if epoch_entropy_stats:
                print("  Entropy stats:")
                for stat in epoch_entropy_stats[-1]:
                    print(f"    Layer {stat['layer']}: entropy={stat['entropy_magnitude']:.3f}, "
                          f"composition={stat['composition_alpha']:.3f}")
    
    # Final results
    final_train = results["train_losses"][-1]
    final_test = results["test_losses"][-1]
    best_test = min(results["test_losses"])
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Final Train Loss: {final_train:.4f}")
    print(f"Final Test Loss: {final_test:.4f}")
    print(f"Best Test Loss: {best_test:.4f}")
    
    # Save results
    timestamp = int(time.time())
    results_file = f"entropy_recycling_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    # Run experiment with default config
    config = ExperimentConfig(
        hidden_dims=[64, 64, 64],
        n_patterns=8,
        epochs=50,
        learning_rate=3e-4
    )
    
    results = run_entropy_recycling_experiment(config)