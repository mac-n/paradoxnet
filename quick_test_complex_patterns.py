import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Import our models
from paradox_net_complex_patterns_2025_07_15 import create_complex_patterns_softmax
from paradox_net_complex import ParadoxNetComplex

def create_simple_dataset(vocab_size=50, seq_length=20, n_samples=1000):
    """Create a simple synthetic dataset for testing."""
    # Generate random sequences
    sequences = torch.randint(0, vocab_size, (n_samples, seq_length))
    targets = torch.randint(0, vocab_size, (n_samples,))
    
    # Split train/test
    split_idx = int(0.8 * n_samples)
    train_dataset = TensorDataset(sequences[:split_idx], targets[:split_idx])
    test_dataset = TensorDataset(sequences[split_idx:], targets[split_idx:])
    
    return train_dataset, test_dataset

def test_model(model, train_loader, test_loader, device, epochs=10):
    """Test a model with basic training loop."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_pred_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            pred_errors = None
            
            if isinstance(output, tuple):
                output, pred_errors = output
            
            # Calculate loss
            task_loss = criterion(output, target)
            total_loss = task_loss
            
            if pred_errors is not None:
                pred_loss = 0.1 * torch.mean(pred_errors)
                total_loss = task_loss + pred_loss
                total_pred_loss += pred_loss.item()
            
            total_loss.backward()
            optimizer.step()
            train_loss += task_loss.item()
            
            if batch_idx == 0:  # Just test first batch works
                break
        
        # Test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if isinstance(output, tuple):
                    output, _ = output
                
                loss = criterion(output, target)
                test_loss += loss.item()
                break  # Just test first batch
        
        avg_train_loss = train_loss
        avg_test_loss = test_loss
        avg_pred_loss = total_pred_loss if total_pred_loss > 0 else 0
        
        print(f"Epoch {epoch}: Train={avg_train_loss:.4f}, Test={avg_test_loss:.4f}, Pred={avg_pred_loss:.4f}")
    
    return avg_test_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create simple dataset
    vocab_size = 50
    seq_length = 20
    train_dataset, test_dataset = create_simple_dataset(vocab_size, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print("Testing models on synthetic data...")
    
    # Test 1: Original Complex ParadoxNet
    print("\n=== Testing Original Complex ParadoxNet ===")
    try:
        model1 = ParadoxNetComplex(vocab_size=vocab_size, embedding_dim=32, hidden_dims=[48, 96])
        result1 = test_model(model1, train_loader, test_loader, device, epochs=5)
        print(f"✅ Original Complex works! Final loss: {result1:.4f}")
    except Exception as e:
        print(f"❌ Original Complex failed: {e}")
    
    # Test 2: Complex + Patterns (Softmax)
    print("\n=== Testing Complex + Patterns (Softmax) ===")
    try:
        model2 = create_complex_patterns_softmax(
            vocab_size=vocab_size, 
            embedding_dim=32, 
            hidden_dims=[48, 96], 
            penultimate_dim=48
        )
        result2 = test_model(model2, train_loader, test_loader, device, epochs=5)
        print(f"✅ Complex + Patterns works! Final loss: {result2:.4f}")
        
        # Check if prediction errors are being generated
        model2.eval()
        with torch.no_grad():
            test_batch = next(iter(test_loader))
            output = model2(test_batch[0].to(device))
            if isinstance(output, tuple) and output[1] is not None:
                print(f"✅ Prediction errors working! Shape: {output[1].shape}")
            else:
                print("⚠️ No prediction errors generated")
                
    except Exception as e:
        print(f"❌ Complex + Patterns failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Quick Test Complete ===")

if __name__ == "__main__":
    main()