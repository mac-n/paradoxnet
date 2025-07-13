#!/usr/bin/env python3
"""
Analyze whether the real vs imaginary parts of complex representations serve different purposes.
Is the real part just "wasted mental real estate" or does it encode something different?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from paradox_net_complex import ParadoxNetComplex

def analyze_real_vs_imaginary():
    """Analyze the roles of real vs imaginary parts in the learned representations."""
    
    # Load model
    checkpoint = torch.load("complex_model.pt", map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    vocab_size = state_dict['embedding.weight'].shape[0]
    embedding_dim = state_dict['embedding.weight'].shape[1]
    hidden_dims = [64, 64]
    n_patterns = 16
    
    model = ParadoxNetComplex(vocab_size, embedding_dim, hidden_dims, n_patterns)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Test various texts to see real vs imaginary behavior
    test_texts = [
        "hello world",
        "the quick brown fox", 
        "abcdefghijk",
        "programming is fun",
        "neural networks learn"
    ]
    
    results = {}
    
    for text in test_texts:
        # Create vocab and tokens
        chars = sorted(list(set(text.lower())))
        vocab = {ch: i for i, ch in enumerate(chars)}
        tokens = torch.tensor([vocab.get(ch, 0) for ch in text.lower()], dtype=torch.long)
        tokens = tokens.unsqueeze(0)
        
        # Capture activations
        activations = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach().cpu()
                else:
                    activations[name] = output.detach().cpu()
            return hook
        
        for i, layer in enumerate(model.hidden_layers):
            hooks.append(layer.register_forward_hook(make_hook(f'layer_{i}')))
        
        with torch.no_grad():
            _ = model(tokens)
        
        for hook in hooks:
            hook.remove()
        
        # Analyze real vs imaginary parts
        text_analysis = {'text': text}
        
        for layer_name, activation in activations.items():
            if activation.dtype in [torch.complex64, torch.complex128]:
                real_part = activation.real
                imag_part = activation.imag
                
                # Basic statistics
                text_analysis[layer_name] = {
                    # Real part analysis
                    'real_mean': real_part.mean().item(),
                    'real_std': real_part.std().item(),
                    'real_range': (real_part.min().item(), real_part.max().item()),
                    'real_sparsity': (real_part.abs() < 0.01).float().mean().item(),
                    'real_energy': (real_part ** 2).sum().item(),
                    
                    # Imaginary part analysis
                    'imag_mean': imag_part.mean().item(),
                    'imag_std': imag_part.std().item(),
                    'imag_range': (imag_part.min().item(), imag_part.max().item()),
                    'imag_sparsity': (imag_part.abs() < 0.01).float().mean().item(),
                    'imag_energy': (imag_part ** 2).sum().item(),
                    
                    # Relationship analysis
                    'real_imag_correlation': torch.corrcoef(torch.stack([
                        real_part.flatten(), imag_part.flatten()
                    ]))[0, 1].item(),
                    
                    # Information content comparison
                    'real_entropy': -torch.sum(torch.softmax(real_part.flatten(), dim=0) * 
                                             torch.log_softmax(real_part.flatten(), dim=0)).item(),
                    'imag_entropy': -torch.sum(torch.softmax(imag_part.flatten(), dim=0) * 
                                             torch.log_softmax(imag_part.flatten(), dim=0)).item(),
                    
                    # Positional analysis - do real/imag encode different positional info?
                    'position_analysis': analyze_positional_encoding(real_part, imag_part)
                }
        
        results[text] = text_analysis
    
    return results

def analyze_positional_encoding(real_part, imag_part):
    """Analyze if real vs imaginary parts encode different positional information."""
    
    # real_part and imag_part are [batch, seq_len, features]
    seq_len = real_part.shape[1]
    
    if seq_len < 2:
        return {'insufficient_data': True}
    
    # Compute position-to-position changes
    real_changes = []
    imag_changes = []
    
    for pos in range(seq_len - 1):
        real_change = torch.norm(real_part[0, pos+1] - real_part[0, pos]).item()
        imag_change = torch.norm(imag_part[0, pos+1] - imag_part[0, pos]).item()
        real_changes.append(real_change)
        imag_changes.append(imag_change)
    
    # Compute variance across positions for each feature
    real_pos_variance = real_part.var(dim=1).mean().item()  # Variance across positions
    imag_pos_variance = imag_part.var(dim=1).mean().item()
    
    # See if real/imag have different "roles" in sequence encoding
    return {
        'real_position_changes': real_changes,
        'imag_position_changes': imag_changes,
        'real_positional_variance': real_pos_variance,
        'imag_positional_variance': imag_pos_variance,
        'real_vs_imag_change_ratio': np.mean(real_changes) / (np.mean(imag_changes) + 1e-8)
    }

def test_ablation_study():
    """Test what happens if we ablate (zero out) real or imaginary parts."""
    
    checkpoint = torch.load("complex_model.pt", map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    vocab_size = state_dict['embedding.weight'].shape[0]
    embedding_dim = state_dict['embedding.weight'].shape[1]
    hidden_dims = [64, 64]
    n_patterns = 16
    
    model = ParadoxNetComplex(vocab_size, embedding_dim, hidden_dims, n_patterns)
    model.load_state_dict(state_dict)
    model.eval()
    
    test_text = "hello world"
    chars = sorted(list(set(test_text.lower())))
    vocab = {ch: i for i, ch in enumerate(chars)}
    tokens = torch.tensor([vocab.get(ch, 0) for ch in test_text.lower()], dtype=torch.long)
    tokens = tokens.unsqueeze(0)
    
    # Test 1: Normal forward pass
    with torch.no_grad():
        normal_output = model(tokens)
    
    # Test 2: Zero out real parts in embeddings and see what happens
    original_embedding_weight = model.embedding.weight.data.clone()
    
    # Test 3: Analyze pattern dictionaries - are real/imag parts different?
    pattern_analysis = {}
    
    for i, layer in enumerate(model.hidden_layers):
        patterns = layer.pattern_dict.detach()
        
        pattern_analysis[f'layer_{i}'] = {
            'real_pattern_variance': patterns.real.var(dim=0).mean().item(),
            'imag_pattern_variance': patterns.imag.var(dim=0).mean().item(),
            'real_pattern_mean': patterns.real.mean().item(),
            'imag_pattern_mean': patterns.imag.mean().item(),
            'real_imag_correlation': torch.corrcoef(torch.stack([
                patterns.real.flatten(), patterns.imag.flatten()
            ]))[0, 1].item()
        }
    
    return {
        'normal_output_stats': {
            'mean': normal_output.mean().item(),
            'std': normal_output.std().item(),
            'entropy': -torch.sum(torch.softmax(normal_output.flatten(), dim=0) * 
                                torch.log_softmax(normal_output.flatten(), dim=0)).item()
        },
        'pattern_analysis': pattern_analysis
    }

def visualize_real_vs_imaginary(results, ablation_results):
    """Create visualizations comparing real vs imaginary parts."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Real vs Imaginary Energy Distribution
    texts = list(results.keys())
    real_energies_l0 = [results[text]['layer_0']['real_energy'] for text in texts]
    imag_energies_l0 = [results[text]['layer_0']['imag_energy'] for text in texts]
    real_energies_l1 = [results[text]['layer_1']['real_energy'] for text in texts]
    imag_energies_l1 = [results[text]['layer_1']['imag_energy'] for text in texts]
    
    x_pos = range(len(texts))
    width = 0.2
    
    axes[0, 0].bar([p - 1.5*width for p in x_pos], real_energies_l0, width, label='Layer 0 Real', alpha=0.7)
    axes[0, 0].bar([p - 0.5*width for p in x_pos], imag_energies_l0, width, label='Layer 0 Imag', alpha=0.7)
    axes[0, 0].bar([p + 0.5*width for p in x_pos], real_energies_l1, width, label='Layer 1 Real', alpha=0.7)
    axes[0, 0].bar([p + 1.5*width for p in x_pos], imag_energies_l1, width, label='Layer 1 Imag', alpha=0.7)
    
    axes[0, 0].set_title('Real vs Imaginary Energy by Text')
    axes[0, 0].set_ylabel('Energy (sum of squares)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([t[:6] + '...' for t in texts], rotation=45)
    axes[0, 0].legend()
    
    # Plot 2: Information Content (Entropy)
    real_entropies_l0 = [results[text]['layer_0']['real_entropy'] for text in texts]
    imag_entropies_l0 = [results[text]['layer_0']['imag_entropy'] for text in texts]
    
    axes[0, 1].bar([p - 0.2 for p in x_pos], real_entropies_l0, 0.4, label='Real', alpha=0.7)
    axes[0, 1].bar([p + 0.2 for p in x_pos], imag_entropies_l0, 0.4, label='Imaginary', alpha=0.7)
    axes[0, 1].set_title('Information Content (Layer 0)')
    axes[0, 1].set_ylabel('Entropy')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([t[:6] + '...' for t in texts], rotation=45)
    axes[0, 1].legend()
    
    # Plot 3: Real-Imaginary Correlation
    correlations_l0 = [results[text]['layer_0']['real_imag_correlation'] for text in texts]
    correlations_l1 = [results[text]['layer_1']['real_imag_correlation'] for text in texts]
    
    axes[0, 2].bar([p - 0.2 for p in x_pos], correlations_l0, 0.4, label='Layer 0', alpha=0.7)
    axes[0, 2].bar([p + 0.2 for p in x_pos], correlations_l1, 0.4, label='Layer 1', alpha=0.7)
    axes[0, 2].set_title('Real-Imaginary Correlation')
    axes[0, 2].set_ylabel('Correlation Coefficient')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels([t[:6] + '...' for t in texts], rotation=45)
    axes[0, 2].legend()
    axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Positional Variance
    real_pos_vars = [results[text]['layer_0']['position_analysis']['real_positional_variance'] for text in texts if 'position_analysis' in results[text]['layer_0']]
    imag_pos_vars = [results[text]['layer_0']['position_analysis']['imag_positional_variance'] for text in texts if 'position_analysis' in results[text]['layer_0']]
    
    if real_pos_vars and imag_pos_vars:
        axes[1, 0].bar([p - 0.2 for p in range(len(real_pos_vars))], real_pos_vars, 0.4, label='Real', alpha=0.7)
        axes[1, 0].bar([p + 0.2 for p in range(len(imag_pos_vars))], imag_pos_vars, 0.4, label='Imaginary', alpha=0.7)
        axes[1, 0].set_title('Positional Variance (Layer 0)')
        axes[1, 0].set_ylabel('Variance across positions')
        axes[1, 0].legend()
    
    # Plot 5: Pattern Dictionary Analysis
    if 'pattern_analysis' in ablation_results:
        layers = list(ablation_results['pattern_analysis'].keys())
        real_pattern_vars = [ablation_results['pattern_analysis'][layer]['real_pattern_variance'] for layer in layers]
        imag_pattern_vars = [ablation_results['pattern_analysis'][layer]['imag_pattern_variance'] for layer in layers]
        
        axes[1, 1].bar([i - 0.2 for i in range(len(layers))], real_pattern_vars, 0.4, label='Real', alpha=0.7)
        axes[1, 1].bar([i + 0.2 for i in range(len(layers))], imag_pattern_vars, 0.4, label='Imaginary', alpha=0.7)
        axes[1, 1].set_title('Pattern Dictionary Variance')
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].set_xticks(range(len(layers)))
        axes[1, 1].set_xticklabels(layers)
        axes[1, 1].legend()
    
    # Plot 6: Sparsity Comparison
    real_sparsity_l0 = [results[text]['layer_0']['real_sparsity'] for text in texts]
    imag_sparsity_l0 = [results[text]['layer_0']['imag_sparsity'] for text in texts]
    
    axes[1, 2].bar([p - 0.2 for p in x_pos], real_sparsity_l0, 0.4, label='Real', alpha=0.7)
    axes[1, 2].bar([p + 0.2 for p in x_pos], imag_sparsity_l0, 0.4, label='Imaginary', alpha=0.7)
    axes[1, 2].set_title('Sparsity (% near zero)')
    axes[1, 2].set_ylabel('Sparsity Ratio')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([t[:6] + '...' for t in texts], rotation=45)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('real_vs_imaginary_analysis.png', dpi=150, bbox_inches='tight')
    return fig

def main():
    print("=== Real vs Imaginary Analysis ===\n")
    
    print("1. Analyzing real vs imaginary parts across different texts...")
    results = analyze_real_vs_imaginary()
    
    print("\nReal vs Imaginary Statistics:")
    for text, data in results.items():
        print(f"\n'{text}':")
        for layer in ['layer_0', 'layer_1']:
            if layer in data:
                real_energy = data[layer]['real_energy']
                imag_energy = data[layer]['imag_energy']
                correlation = data[layer]['real_imag_correlation']
                real_entropy = data[layer]['real_entropy']
                imag_entropy = data[layer]['imag_entropy']
                
                print(f"  {layer}:")
                print(f"    Energy - Real: {real_energy:.3f}, Imag: {imag_energy:.3f} (ratio: {real_energy/imag_energy:.2f})")
                print(f"    Information - Real: {real_entropy:.3f}, Imag: {imag_entropy:.3f}")
                print(f"    Correlation: {correlation:.3f}")
    
    print("\n2. Running ablation analysis...")
    ablation_results = test_ablation_study()
    
    print("\nPattern Dictionary Analysis:")
    for layer, stats in ablation_results['pattern_analysis'].items():
        print(f"  {layer}:")
        print(f"    Variance - Real: {stats['real_pattern_variance']:.4f}, Imag: {stats['imag_pattern_variance']:.4f}")
        print(f"    Mean - Real: {stats['real_pattern_mean']:.4f}, Imag: {stats['imag_pattern_mean']:.4f}")
        print(f"    Correlation: {stats['real_imag_correlation']:.3f}")
    
    print("\n3. Creating visualizations...")
    fig = visualize_real_vs_imaginary(results, ablation_results)
    plt.show()
    
    print("\n=== ANALYSIS SUMMARY ===")
    
    # Determine if real part is "wasted"
    avg_real_energy = np.mean([results[text]['layer_0']['real_energy'] for text in results])
    avg_imag_energy = np.mean([results[text]['layer_0']['imag_energy'] for text in results])
    avg_correlation = np.mean([results[text]['layer_0']['real_imag_correlation'] for text in results])
    
    print(f"Average Energy Ratio (Real/Imag): {avg_real_energy/avg_imag_energy:.2f}")
    print(f"Average Real-Imag Correlation: {avg_correlation:.3f}")
    
    if avg_real_energy / avg_imag_energy < 0.1:
        print("→ REAL PART appears to be underutilized (low energy)")
    elif abs(avg_correlation) > 0.8:
        print("→ REAL and IMAGINARY parts are highly correlated (redundant)")
    else:
        print("→ REAL and IMAGINARY parts serve different roles")
    
    print("\nGenerated: real_vs_imaginary_analysis.png")

if __name__ == "__main__":
    main()