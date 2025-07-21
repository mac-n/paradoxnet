import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset

# Import our hybrid version
from paradox_net_complex_real_attention_2025_07_15 import ParadoxNetComplexRealAttention
from data_generators import get_tiny_shakespeare_data

def quick_shakespeare_test():
    """Quick test on Shakespeare with the hybrid approach"""
    print("Hybrid Complex Paradox + Real Attention on Shakespeare")
    print("=" * 55)
    
    # Get data
    X, y, metadata = get_tiny_shakespeare_data()
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    # Create hybrid model
    model = ParadoxNetComplexRealAttention(
        vocab_size=metadata['vocab_size'],
        embedding_dim=64,
        hidden_dims=[64, 64],
        n_patterns=16
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Training...")
    start_time = time.time()
    
    for epoch in range(30):  # Reduced epochs for quick test
        model.train()
        epoch_train_loss = 0
        epoch_pred_loss = 0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            output, pred_errors = model(X_batch)
            
            # Loss computation
            task_loss = criterion(output, y_batch.long())
            total_loss = task_loss
            
            if pred_errors is not None:
                pred_loss = 0.1 * torch.mean(pred_errors)
                total_loss = task_loss + pred_loss
                epoch_pred_loss += pred_loss.item()
            
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += task_loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        avg_pred_loss = epoch_pred_loss / num_batches
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Task Loss = {avg_train_loss:.4f}, Pred Loss = {avg_pred_loss:.6f}")
    
    # Final evaluation
    model.eval()
    test_loss = 0
    num_test_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output, _ = model(X_batch)
            batch_loss = criterion(output, y_batch.long())
            test_loss += batch_loss.item()
            num_test_batches += 1
    
    avg_test_loss = test_loss / num_test_batches
    training_time = time.time() - start_time
    
    print(f"\nFinal test loss: {avg_test_loss:.4f}")
    print(f"Training time: {training_time:.1f} seconds")
    
    # Compare to previous results
    print("\nComparison to previous results:")
    print("Original complex net: ~2.5753 (stable learning)")
    print("Pure complex + routing: ~2.67 (slow)")
    print(f"Hybrid approach: {avg_test_loss:.4f} (fast)")
    
    if avg_test_loss < 2.5753:
        print("🎉 IMPROVEMENT! Hybrid approach beats original!")
    elif avg_test_loss < 2.67:
        print("✅ GOOD! Faster than pure complex with similar performance!")
    else:
        print("🤔 Similar performance - but much faster!")
    
    return avg_test_loss

if __name__ == "__main__":
    try:
        result = quick_shakespeare_test()
        print(f"\n🎯 Final result: {result:.4f}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()