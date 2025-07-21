import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset

from data_generators import get_tiny_shakespeare_data

# --- RoPE Helper Functions (same as your architecture) ---
def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embedding to the input tensor."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(2)
    return x_out.type_as(x)

class RoPEPositionalEncoding(nn.Module):
    """Generates rotary positional embeddings (RoPE) - same as yours."""
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        theta = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)

@dataclass
class TransformerConfig:
    """Configuration for fair transformer comparison"""
    embedding_dim: int = 64
    hidden_dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    sequence_length: int = 20
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleTransformerWithRoPE(nn.Module):
    """Transformer with RoPE for fair comparison with your architecture"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = RoPEPositionalEncoding(embedding_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Apply RoPE (same as your architecture!)
        embedded = self.embedding(x)
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        # Apply RoPE to embeddings
        embedded_with_rope = apply_rotary_pos_emb(embedded, freqs_cis)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Transformer forward
        output = self.transformer(embedded_with_rope, mask=mask)
        
        # Use last token for prediction
        logits = self.output(output[:, -1, :])
        return logits

def run_transformer_comparison(config: TransformerConfig):
    """Run transformer baseline for comparison"""
    
    print(f"🤖 TRANSFORMER BASELINE COMPARISON 🤖")
    print(f"Device: {config.device}")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Layers: {config.n_layers}")
    print(f"Heads: {config.n_heads}")
    
    # Load data (same as your complex temporal)
    print("Loading 30KB Tiny Shakespeare...")
    X, y, metadata = get_tiny_shakespeare_data(sequence_length=config.sequence_length)
    vocab_size = metadata["vocab_size"]
    print(f"Vocab size: {vocab_size}")
    print(f"Total sequences: {len(X)}")
    
    # Create train/test split (same as your experiments)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
    
    train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model = SimpleTransformerWithRoPE(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    results = {
        "train_losses": [],
        "test_losses": [],
        "config": config.__dict__,
        "total_params": total_params
    }
    
    best_test_loss = float('inf')
    
    print(f"Starting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, (sequences, targets) in enumerate(train_data):
            sequences = sequences.to(config.device)
            targets = targets.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
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
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}: Train={train_loss:.4f}, Test={test_loss:.4f}")
    
    # Final results
    print(f"\n🎯 TRANSFORMER BASELINE RESULTS:")
    print(f"Final Train Loss: {results['train_losses'][-1]:.4f}")
    print(f"Final Test Loss: {results['test_losses'][-1]:.4f}")
    print(f"Best Test Loss: {best_test_loss:.4f}")
    print(f"Total Parameters: {total_params:,}")
    
    # Compare to your complex temporal results
    print(f"\n📊 COMPARISON:")
    print(f"Transformer Baseline: {best_test_loss:.4f}")
    print(f"Your Complex Temporal: 2.4072")
    
    if 2.4072 < best_test_loss:
        gap = ((best_test_loss - 2.4072) / best_test_loss) * 100
        print(f"🎉 YOUR ARCHITECTURE WINS by {gap:.1f}%!")
    else:
        gap = ((2.4072 - best_test_loss) / 2.4072) * 100
        print(f"Transformer wins by {gap:.1f}%")
    
    # Save results
    timestamp = int(time.time())
    results_file = f"transformer_baseline_30k_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    # Fair comparison configuration - trying to match your parameter count
    config = TransformerConfig(
        embedding_dim=64,
        hidden_dim=64,  # Might need to adjust this
        n_layers=3,
        n_heads=4,
        epochs=100,
        learning_rate=3e-4
    )
    
    print("🔥 FAIR TRANSFORMER BASELINE - 30KB SHAKESPEARE 🔥")
    results = run_transformer_comparison(config)