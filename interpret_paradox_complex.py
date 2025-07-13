#!/usr/bin/env python3
"""
Advanced interpretability analysis for ParadoxNetComplex.
Helps understand what the model is learning despite distributed heatmaps.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import json

class ParadoxComplexInterpreter:
    """Analyzes the internal representations of ParadoxNetComplex."""
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.activations = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook
        
        # Hook hidden layers
        for i, layer in enumerate(self.model.hidden_layers):
            self.hooks.append(layer.register_forward_hook(make_hook(f'hidden_{i}')))
        
        # Hook penultimate layer
        self.hooks.append(
            self.model.penultimate_layer.register_forward_hook(make_hook('penultimate'))
        )
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_text(self, text: str, vocab=None) -> Dict[str, Any]:
        """Analyze a text sample and return comprehensive interpretability data."""
        if vocab is None:
            # Simple character tokenization
            chars = sorted(list(set(text)))
            char_to_idx = {ch: i for i, ch in enumerate(chars)}
            tokens = torch.tensor([char_to_idx.get(ch, 0) for ch in text[:50]], dtype=torch.long)
        else:
            tokens = torch.tensor([vocab.get(ch, 0) for ch in text[:50]], dtype=torch.long)
        
        tokens = tokens.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            output = self.model(tokens)
        
        analysis = {
            'input_text': text[:50],
            'tokens': tokens.squeeze().tolist(),
            'layer_analysis': self._analyze_layers(),
            'pattern_analysis': self._analyze_patterns(),
            'paradox_analysis': self._analyze_paradox_activations(tokens),
            'attention_analysis': self._analyze_attention_patterns(),
            'consensus_analysis': self._analyze_consensus_mechanisms()
        }
        
        return analysis
    
    def _analyze_layers(self) -> Dict[str, Any]:
        """Analyze what each layer is computing."""
        layer_stats = {}
        
        for layer_name, activation in self.activations.items():
            if 'hidden' in layer_name:
                # Complex tensor analysis
                if activation.dtype == torch.complex64 or activation.dtype == torch.complex128:
                    magnitude = activation.abs()
                    phase = activation.angle()
                    
                    layer_stats[layer_name] = {
                        'magnitude_mean': magnitude.mean().item(),
                        'magnitude_std': magnitude.std().item(),
                        'phase_mean': phase.mean().item(),
                        'phase_std': phase.std().item(),
                        'sparsity': (magnitude < 0.1).float().mean().item(),
                        'max_magnitude': magnitude.max().item(),
                        'magnitude_distribution': magnitude.flatten().numpy()[:100].tolist()
                    }
                else:
                    # Real tensor analysis
                    layer_stats[layer_name] = {
                        'mean': activation.mean().item(),
                        'std': activation.std().item(),
                        'sparsity': (activation.abs() < 0.1).float().mean().item(),
                        'max_value': activation.max().item(),
                        'min_value': activation.min().item()
                    }
        
        return layer_stats
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze pattern dictionary usage and specialization."""
        pattern_analysis = {}
        
        for i, layer in enumerate(self.model.hidden_layers):
            patterns = layer.pattern_dict.detach().cpu()
            
            # Pattern similarity analysis
            pattern_similarities = []
            for p1_idx in range(patterns.shape[0]):
                for p2_idx in range(p1_idx + 1, patterns.shape[0]):
                    p1, p2 = patterns[p1_idx], patterns[p2_idx]
                    # Use magnitude for complex cosine similarity
                    p1_flat = p1.flatten()
                    p2_flat = p2.flatten()
                    similarity = F.cosine_similarity(p1_flat.abs().unsqueeze(0), 
                                                   p2_flat.abs().unsqueeze(0)).item()
                    pattern_similarities.append(similarity)
            
            # Pattern magnitude analysis
            pattern_magnitudes = patterns.abs().mean(dim=1).numpy().tolist()
            
            pattern_analysis[f'layer_{i}'] = {
                'pattern_similarities': pattern_similarities,
                'pattern_magnitudes': pattern_magnitudes,
                'similarity_mean': np.mean(pattern_similarities),
                'similarity_std': np.std(pattern_similarities),
                'specialization_score': 1.0 - np.mean(pattern_similarities)  # Higher = more specialized
            }
        
        return pattern_analysis
    
    def _analyze_paradox_activations(self, tokens: torch.Tensor) -> Dict[str, Any]:
        """Analyze the paradox mechanism (self_prediction - hidden) activations."""
        paradox_stats = {}
        
        # Need to manually compute paradox activations
        with torch.no_grad():
            current = self.model.embedding(tokens)
            freqs_cis = self.model.pos_encoder.freqs_cis[:tokens.shape[1]]
            
            from paradox_net_complex import apply_rotary_pos_emb
            current_real = apply_rotary_pos_emb(current, freqs_cis)
            current = torch.view_as_complex(current_real.float().reshape(*current_real.shape[:-1], -1, 2))
            
            for i, layer in enumerate(self.model.hidden_layers):
                hidden_linear = layer.process(current)
                self_prediction = layer.self_predictor(hidden_linear)
                paradox = self_prediction - hidden_linear
                paradox_magnitude = paradox.abs()
                gating = torch.sigmoid(paradox_magnitude)
                
                paradox_stats[f'layer_{i}'] = {
                    'paradox_magnitude_mean': paradox_magnitude.mean().item(),
                    'paradox_magnitude_std': paradox_magnitude.std().item(),
                    'gating_mean': gating.mean().item(),
                    'gating_std': gating.std().item(),
                    'high_paradox_ratio': (paradox_magnitude > paradox_magnitude.mean()).float().mean().item(),
                    'paradox_distribution': paradox_magnitude.flatten().numpy()[:50].tolist()
                }
                
                current, _ = layer(current)
        
        return paradox_stats
    
    def _analyze_attention_patterns(self) -> Dict[str, Any]:
        """Analyze pattern attention weights to see what patterns are being selected."""
        # This would require modifying the model to return attention weights
        # For now, return placeholder
        return {"note": "Attention analysis requires model modification to expose attention weights"}
    
    def _analyze_consensus_mechanisms(self) -> Dict[str, Any]:
        """Analyze how consensus vs residual mechanisms work."""
        return {"note": "Consensus analysis requires access to penultimate contributions"}
    
    def generate_report(self, analysis: Dict[str, Any], save_path: str = None) -> str:
        """Generate a human-readable interpretability report."""
        report = []
        report.append("=== ParadoxNetComplex Interpretability Report ===\n")
        
        report.append(f"Input: '{analysis['input_text']}'\n")
        
        # Layer analysis
        report.append("LAYER ANALYSIS:")
        for layer_name, stats in analysis['layer_analysis'].items():
            report.append(f"  {layer_name}:")
            if 'magnitude_mean' in stats:
                report.append(f"    Complex magnitude: {stats['magnitude_mean']:.4f} ± {stats['magnitude_std']:.4f}")
                report.append(f"    Phase variation: {stats['phase_std']:.4f}")
                report.append(f"    Sparsity: {stats['sparsity']:.2%}")
            else:
                report.append(f"    Mean activation: {stats['mean']:.4f}")
                report.append(f"    Sparsity: {stats['sparsity']:.2%}")
        
        # Pattern analysis
        report.append("\nPATTERN SPECIALIZATION:")
        for layer_name, stats in analysis['pattern_analysis'].items():
            report.append(f"  {layer_name}:")
            report.append(f"    Specialization score: {stats['specialization_score']:.4f}")
            report.append(f"    Pattern similarity: {stats['similarity_mean']:.4f} ± {stats['similarity_std']:.4f}")
        
        # Paradox analysis
        report.append("\nPARADOX MECHANISM:")
        for layer_name, stats in analysis['paradox_analysis'].items():
            report.append(f"  {layer_name}:")
            report.append(f"    Paradox magnitude: {stats['paradox_magnitude_mean']:.4f}")
            report.append(f"    Gating strength: {stats['gating_mean']:.4f}")
            report.append(f"    High paradox ratio: {stats['high_paradox_ratio']:.2%}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def create_visualizations(self, analysis: Dict[str, Any], save_dir: str = None):
        """Create visualization plots for the analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('ParadoxNetComplex Interpretability Analysis')
        
        # Plot 1: Layer magnitude distributions
        ax = axes[0, 0]
        for layer_name, stats in analysis['layer_analysis'].items():
            if 'magnitude_distribution' in stats:
                ax.hist(stats['magnitude_distribution'], alpha=0.6, label=layer_name, bins=20)
        ax.set_title('Magnitude Distributions')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Plot 2: Pattern specialization scores
        ax = axes[0, 1]
        layers = list(analysis['pattern_analysis'].keys())
        scores = [analysis['pattern_analysis'][layer]['specialization_score'] for layer in layers]
        ax.bar(layers, scores)
        ax.set_title('Pattern Specialization by Layer')
        ax.set_ylabel('Specialization Score')
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Plot 3: Paradox magnitude distributions
        ax = axes[0, 2]
        for layer_name, stats in analysis['paradox_analysis'].items():
            ax.hist(stats['paradox_distribution'], alpha=0.6, label=layer_name, bins=20)
        ax.set_title('Paradox Magnitude Distributions')
        ax.set_xlabel('Paradox Magnitude')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Plot 4: Gating strength by layer
        ax = axes[1, 0]
        layers = list(analysis['paradox_analysis'].keys())
        gating = [analysis['paradox_analysis'][layer]['gating_mean'] for layer in layers]
        ax.plot(layers, gating, 'o-')
        ax.set_title('Gating Strength by Layer')
        ax.set_ylabel('Mean Gating')
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Plot 5: Sparsity by layer
        ax = axes[1, 1]
        layers = list(analysis['layer_analysis'].keys())
        sparsity = [analysis['layer_analysis'][layer]['sparsity'] for layer in layers]
        ax.bar(layers, sparsity)
        ax.set_title('Activation Sparsity by Layer')
        ax.set_ylabel('Sparsity')
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Plot 6: Pattern similarity heatmap (for first layer)
        ax = axes[1, 2]
        if analysis['pattern_analysis']:
            first_layer = list(analysis['pattern_analysis'].keys())[0]
            similarities = analysis['pattern_analysis'][first_layer]['pattern_similarities']
            # Convert to matrix form
            n_patterns = int(np.sqrt(2 * len(similarities)) + 1)
            sim_matrix = np.eye(n_patterns)
            idx = 0
            for i in range(n_patterns):
                for j in range(i+1, n_patterns):
                    sim_matrix[i, j] = sim_matrix[j, i] = similarities[idx]
                    idx += 1
            
            sns.heatmap(sim_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title(f'Pattern Similarity Matrix ({first_layer})')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/paradox_analysis.png", dpi=150, bbox_inches='tight')
        
        return fig

def main():
    """Example usage of the interpreter."""
    from paradox_net_complex import ParadoxNetComplex
    
    # Create a dummy model for demonstration
    model = ParadoxNetComplex(
        vocab_size=100,
        embedding_dim=64,
        hidden_dims=[48, 48],
        n_patterns=8
    )
    
    interpreter = ParadoxComplexInterpreter(model)
    
    # Analyze sample text
    sample_text = "The quick brown fox jumps over the lazy dog"
    analysis = interpreter.analyze_text(sample_text)
    
    # Generate report
    report = interpreter.generate_report(analysis)
    print(report)
    
    # Create visualizations
    interpreter.create_visualizations(analysis)
    plt.show()
    
    # Cleanup
    interpreter.remove_hooks()

if __name__ == "__main__":
    main()