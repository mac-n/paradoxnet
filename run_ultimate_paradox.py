import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional
from torch.utils.data import DataLoader, TensorDataset

# Import the ULTIMATE beast
from paradox_net_complex_ultimate import ParadoxNetComplexUltimate
from data_generators import get_tiny_shakespeare_data

class UltimateExperimentHarness:
    def __init__(self, data_generator: Callable, epochs: int = 50, batch_size: int = 32, 
                 lambda_residual: float = 0.1, save_path: str = 'ultimate_model.pt'):
        self.data_generator = data_generator
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_residual = lambda_residual  # Weight for recursive residual loss
        self.save_path = save_path

    def run_trial(self, seed: int):
        start_time = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X, y, metadata = self.data_generator()
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        
        # Create the ULTIMATE model
        model = ParadoxNetComplexUltimate(
            vocab_size=metadata['vocab_size'],
            embedding_dim=64,
            hidden_dims=[64, 64],
            penultimate_dim=48,  # Slightly smaller penultimate for cleaner routing
            n_patterns=16,
            temporal_lr=0.1,
            temp_lr=0.1
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print("🚀 Starting training for the ULTIMATE ParadoxNet!")
        print(f"📊 Using two-way loss: CrossEntropy + {self.lambda_residual} * RecursiveResidualLoss")
        print(f"🧠 Architecture: Complex numbers + Confidence routing + Temporal temp + Everything!")
        print("=" * 80)
        
        best_test_loss = float('inf')
        
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0
            epoch_task_loss = 0
            epoch_residual_loss = 0
            epoch_avg_confidence = 0
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                
                # THE ULTIMATE FORWARD PASS
                final_output, total_prediction_error, recursive_residual_loss = model(X_batch, y_batch)
                
                # Two-way loss: Task performance + Interpretability pressure
                task_loss = F.cross_entropy(final_output, y_batch.long())
                total_loss = task_loss + self.lambda_residual * recursive_residual_loss
                
                total_loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_train_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_residual_loss += recursive_residual_loss.item()
                
                # Track confidence (from diagnostic prediction errors)
                with torch.no_grad():
                    if total_prediction_error.numel() > 0:
                        avg_pred_error = torch.mean(total_prediction_error)
                        avg_confidence = torch.exp(-avg_pred_error)
                        epoch_avg_confidence += avg_confidence.item()
                
                num_batches += 1

            # Update temporal temperatures after each epoch
            model.update_temporal_temperatures()

            # Evaluation
            model.eval()
            with torch.no_grad():
                test_output, test_prediction_error, test_residual_loss = model(X_test, y_test)
                test_task_loss = F.cross_entropy(test_output, y_test.long()).item()
                test_total_loss = test_task_loss + self.lambda_residual * test_residual_loss.item()
                
                # Calculate test accuracy
                test_predictions = torch.argmax(test_output, dim=1)
                test_accuracy = (test_predictions == y_test).float().mean().item()
                
                # Average training metrics
                avg_train_loss = epoch_train_loss / num_batches
                avg_task_loss = epoch_task_loss / num_batches
                avg_residual_loss = epoch_residual_loss / num_batches
                avg_confidence = epoch_avg_confidence / num_batches if num_batches > 0 else 0
            
            # Print comprehensive stats
            print(f"Epoch {epoch:2d}: "
                  f"Train={avg_train_loss:.4f} (Task={avg_task_loss:.4f}, Res={avg_residual_loss:.4f}) | "
                  f"Test={test_total_loss:.4f} (Task={test_task_loss:.4f}, Acc={test_accuracy:.3f}) | "
                  f"Conf={avg_confidence:.3f}")
            
            # Save best model
            if test_total_loss < best_test_loss:
                best_test_loss = test_total_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'test_loss': test_total_loss,
                    'test_accuracy': test_accuracy,
                    'hyperparameters': {
                        'vocab_size': metadata['vocab_size'],
                        'embedding_dim': 64,
                        'hidden_dims': [64, 64],
                        'penultimate_dim': 48,
                        'n_patterns': 16,
                        'lambda_residual': self.lambda_residual
                    }
                }, self.save_path)

        print("=" * 80)
        print(f"🎯 Training complete! Best test loss: {best_test_loss:.4f}")
        print(f"💾 Model saved to {self.save_path}")
        print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")
        
        # Print final layer statistics
        print("\n📈 Final Layer Statistics:")
        stats = model.get_layer_stats()
        for i, stat in enumerate(stats):
            if stat.confidence_values is not None:
                avg_conf = torch.mean(stat.confidence_values).item()
                avg_pred_err = torch.mean(stat.prediction_errors).item() if stat.prediction_errors is not None else 0
                print(f"   Layer {i}: Confidence={avg_conf:.3f}, PredError={avg_pred_err:.4f}, "
                      f"SelfParadox={stat.self_paradox_magnitude:.4f}")
        
        return model, best_test_loss

# --- Main Execution Block ---
if __name__ == "__main__":
    print("🔥 ULTIMATE PARADOX NET EXPERIMENT 🔥")
    print("Complex numbers + Confidence routing + Temporal temp + Recursive residual loss")
    print("Let's see if all the architectural brilliance works together!\n")
    
    data_generator = get_tiny_shakespeare_data
    harness = UltimateExperimentHarness(
        data_generator, 
        epochs=50, 
        lambda_residual=0.1,  # Adjust this to control interpretability pressure
        save_path='ultimate_paradox_model.pt'
    )
    
    model, final_loss = harness.run_trial(seed=42)
    
    print(f"\n🎉 EXPERIMENT COMPLETE!")
    print(f"Final loss: {final_loss:.4f}")
    print("Your ultimate interpretable transformer is ready! 🚀")