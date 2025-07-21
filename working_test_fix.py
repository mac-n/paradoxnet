import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Let's create a minimal working test first
def create_simple_dataset(vocab_size=50, seq_length=20, n_samples=1000):
    """Create a simple synthetic dataset for testing."""
    sequences = torch.randint(0, vocab_size, (n_samples, seq_length))
    targets = torch.randint(0, vocab_size, (n_samples,))
    
    split_idx = int(0.8 * n_samples)
    train_dataset = TensorDataset(sequences[:split_idx], targets[:split_idx])
    test_dataset = TensorDataset(sequences[split_idx:], targets[split_idx:])
    
    return train_dataset, test_dataset

def test_original_complex():
    """Test that the original complex net works."""
    from paradox_net_complex import ParadoxNetComplex
    
    device = torch.device('cpu')  # Use CPU to avoid memory issues
    vocab_size = 50
    seq_length = 20
    
    train_dataset, test_dataset = create_simple_dataset(vocab_size, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Smaller batch
    
    print("=== Testing Original Complex ParadoxNet ===")
    try:
        model = ParadoxNetComplex(vocab_size=vocab_size, embedding_dim=32, hidden_dims=[32, 32])
        model = model.to(device)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        test_batch = next(iter(train_loader))
        data, target = test_batch[0].to(device), test_batch[1].to(device)
        
        print(f"Input shape: {data.shape}")
        output = model(data)
        print(f"Output shape: {output.shape}")
        print(f"✅ Original Complex works!")
        
        # Test one training step
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"✅ Training step works! Loss: {loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Original Complex failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_original_complex()
    if success:
        print("\n🎉 Original complex net is working! Now I can debug the pattern version.")
    else:
        print("\n💥 Even the original is broken. Need to check basic setup.")