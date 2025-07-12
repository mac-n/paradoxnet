import torch
import torch.nn as nn
import torch.nn.functional as F # <-- The missing import is added here
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List

# Import the architecture and data generator
from paradox_net_complex import ParadoxNetComplex, apply_rotary_pos_emb
from data_generators import get_tiny_shakespeare_data

class ComplexPatternAnalyzer:
    """A class to analyze the trained complex-valued Paradox Net."""

    def __init__(self, model_path: str, model_config: Dict, idx_to_char: Dict[int, str]):
        """
        Initializes the analyzer by loading the trained model.
        """
        print("Initializing analyzer and loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate the model architecture from the config
        self.model = ParadoxNetComplex(**model_config).to(self.device)
        
        # Load the saved weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.idx_to_char = idx_to_char
        print("Model loaded successfully.")
        
        # Create a directory for the analysis results
        self.output_dir = "analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_learned_patterns(self):
        """
        Visualizes the magnitude and phase of the learned patterns in each layer.
        """
        print("\nVisualizing learned patterns (magnitude and phase)...")
        all_layers = list(self.model.hidden_layers) + [self.model.penultimate_layer]
        
        for i, layer in enumerate(all_layers):
            patterns = layer.pattern_dict.detach().cpu()
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            sns.heatmap(patterns.abs(), ax=axes[0], cmap='viridis')
            axes[0].set_title(f"Layer {i}: Pattern Magnitudes")
            axes[0].set_xlabel("Pattern Feature Dimension")
            axes[0].set_ylabel("Pattern Index")
            
            sns.heatmap(patterns.angle(), ax=axes[1], cmap='hsv')
            axes[1].set_title(f"Layer {i}: Pattern Phases (Angles)")
            axes[1].set_xlabel("Pattern Feature Dimension")
            axes[1].set_ylabel("Pattern Index")
            
            fig.tight_layout()
            save_path = os.path.join(self.output_dir, f"layer_{i}_patterns.pdf")
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"  - Saved pattern visualization for layer {i} to {save_path}")

    def analyze_character_associations(self, X_data: torch.Tensor):
        """
        Analyzes and visualizes which patterns are activated by which characters.
        """
        print("\nAnalyzing pattern-character associations...")
        char_activations = {i: [] for i in self.idx_to_char.keys()}
        
        with torch.no_grad():
            # Process data in batches
            for i in range(0, len(X_data), 32):
                X_batch = X_data[i:i+32].to(self.device)
                
                # --- Manually perform the forward pass to intercept attention weights ---
                batch_size, seq_len = X_batch.shape
                embedded = self.model.embedding(X_batch)
                freqs_cis = self.model.pos_encoder.freqs_cis[:seq_len]
                current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
                current_seq_complex = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))

                # We'll analyze the first hidden layer's attention
                layer = self.model.hidden_layers[0]
                hidden = layer.apply_self_processing(current_seq_complex)
                
                attn_output_complex = layer.pattern_attention(hidden)
                attn_logits = attn_output_complex.real
                
                attn_weights = F.softmax(attn_logits, dim=-1)

                # Store activations
                for batch_item_idx in range(X_batch.shape[0]):
                    for seq_pos_idx in range(X_batch.shape[1]):
                        char_idx = X_batch[batch_item_idx, seq_pos_idx].item()
                        if char_idx in char_activations:
                            activations = attn_weights[batch_item_idx, seq_pos_idx, :].cpu().numpy()
                            char_activations[char_idx].append(activations)
        
        # --- Process and visualize the results ---
        n_patterns = self.model.hidden_layers[0].n_patterns
        sorted_chars = sorted(self.idx_to_char.keys())
        avg_activations = np.zeros((len(sorted_chars), n_patterns))
        
        for i, char_idx in enumerate(sorted_chars):
            activations_list = char_activations[char_idx]
            if activations_list:
                avg_activations[i] = np.mean(activations_list, axis=0)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            avg_activations,
            yticklabels=[self.idx_to_char.get(i, '?') for i in sorted_chars],
            xticklabels=[f"P{i}" for i in range(n_patterns)],
            cmap='viridis'
        )
        plt.title("Average Pattern Activation per Character (Layer 0)")
        plt.xlabel("Pattern Index")
        plt.ylabel("Character")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "character_associations.pdf")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  - Saved character association heatmap to {save_path}")


def main():
    # --- Configuration ---
    MODEL_PATH = 'complex_model.pt'
    MODEL_CONFIG = {
        'embedding_dim': 64,
        'hidden_dims': [64, 64],
        'n_patterns': 16
    }

    # --- Load Data and Metadata ---
    print("Loading data and metadata...")
    X, y, metadata = get_tiny_shakespeare_data()
    
    MODEL_CONFIG['vocab_size'] = metadata['vocab_size']
    
    idx_to_char = {v: k for k, v in metadata['char_to_idx'].items()}

    # --- Run Analysis ---
    analyzer = ComplexPatternAnalyzer(
        model_path=MODEL_PATH,
        model_config=MODEL_CONFIG,
        idx_to_char=idx_to_char
    )
    
    analyzer.visualize_learned_patterns()
    analyzer.analyze_character_associations(X)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

