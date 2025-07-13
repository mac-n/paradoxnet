#!/usr/bin/env python3
"""
Analyze what Layer 1 and the penultimate layer actually do in terms of output prediction.
Map their representations to the final vocabulary predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from paradox_net_complex import ParadoxNetComplex

def analyze_layer_to_output_mapping():
    """Analyze how each layer contributes to final output predictions."""
    
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
    
    # Test with different input types
    test_cases = [
        ("simple_repeat", "aaaa"),
        ("common_word", "the"),
        ("vowel_sequence", "aei"),
        ("consonant_cluster", "str"),
        ("punctuation", "..."),
        ("mixed_case", "Hello"),
    ]
    
    results = {}
    
    for case_name, text in test_cases:
        # Create vocab and tokens
        chars = sorted(list(set(text.lower())))
        # Add some common characters to ensure we have a decent vocab
        all_chars = sorted(list(set(text.lower() + "abcdefghijklmnopqrstuvwxyz .,!?")))
        vocab = {ch: i for i, ch in enumerate(all_chars)}
        reverse_vocab = {i: ch for ch, i in vocab.items()}
        
        tokens = torch.tensor([vocab.get(ch, 0) for ch in text.lower()], dtype=torch.long)
        tokens = tokens.unsqueeze(0)
        
        # Capture detailed activations including intermediate steps
        detailed_activations = {}
        hooks = []
        
        def make_detailed_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    detailed_activations[name] = {
                        'hidden': output[0].detach().cpu(),
                        'penultimate_contrib': output[1].detach().cpu() if len(output) > 1 else None
                    }
                else:
                    detailed_activations[name] = {'output': output.detach().cpu()}
            return hook
        
        # Hook all layers
        for i, layer in enumerate(model.hidden_layers):
            hooks.append(layer.register_forward_hook(make_detailed_hook(f'hidden_{i}')))
        
        hooks.append(model.penultimate_layer.register_forward_hook(make_detailed_hook('penultimate')))
        
        # Forward pass
        with torch.no_grad():
            final_output = model(tokens)
            final_probs = F.softmax(final_output, dim=-1)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze what each layer contributes
        case_analysis = {
            'text': text,
            'vocab_size': len(vocab),
            'final_output': final_output.squeeze().numpy(),
            'final_probs': final_probs.squeeze().numpy(),
            'predicted_tokens': torch.topk(final_probs.squeeze(), k=5)[1].numpy(),
            'predicted_chars': [reverse_vocab.get(idx.item(), '?') for idx in torch.topk(final_probs.squeeze(), k=5)[1]],
            'predicted_probs': torch.topk(final_probs.squeeze(), k=5)[0].numpy(),
        }
        
        # Analyze layer contributions
        for layer_name, activation_data in detailed_activations.items():
            if 'hidden' in activation_data:
                hidden = activation_data['hidden']
                if hidden.dtype in [torch.complex64, torch.complex128]:
                    # Analyze complex representations
                    seq_mean = hidden.mean(dim=1).squeeze()  # Average over sequence
                    
                    case_analysis[layer_name] = {
                        'magnitude_profile': torch.abs(seq_mean).numpy(),
                        'phase_profile': torch.angle(seq_mean).numpy(),
                        'representation_norm': torch.norm(seq_mean).item(),
                        'dominant_features': torch.topk(torch.abs(seq_mean), k=5)[0].numpy(),
                        'dominant_indices': torch.topk(torch.abs(seq_mean), k=5)[1].numpy(),
                    }
                    
                    # If this is a hidden layer with penultimate contribution
                    if activation_data['penultimate_contrib'] is not None:
                        penult_contrib = activation_data['penultimate_contrib'].mean(dim=1).squeeze()
                        case_analysis[layer_name]['penultimate_contribution'] = {
                            'magnitude': torch.abs(penult_contrib).mean().item(),
                            'phase_coherence': torch.abs(torch.mean(torch.exp(1j * torch.angle(penult_contrib)))).item(),
                            'contribution_norm': torch.norm(penult_contrib).item()
                        }
        
        # Try to understand the consensus mechanism
        # We need to manually compute what the consensus view would be
        if 'hidden_0' in detailed_activations and 'hidden_1' in detailed_activations:
            h0_contrib = detailed_activations['hidden_0']['penultimate_contrib']
            h1_contrib = detailed_activations['hidden_1']['penultimate_contrib']
            
            if h0_contrib is not None and h1_contrib is not None:
                h0_mean = h0_contrib.mean(dim=1)
                h1_mean = h1_contrib.mean(dim=1)
                consensus = h0_mean + h1_mean  # Sum as in the model
                
                case_analysis['consensus_analysis'] = {
                    'consensus_magnitude': torch.abs(consensus).mean().item(),
                    'consensus_phase_std': torch.angle(consensus).std().item(),
                    'layer_balance': torch.norm(h0_mean).item() / (torch.norm(h1_mean).item() + 1e-8)
                }
        
        results[case_name] = case_analysis
    
    return results

def analyze_output_sensitivity():
    """Test how sensitive the output is to changes in each layer."""
    
    checkpoint = torch.load("complex_model.pt", map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    vocab_size = state_dict['embedding.weight'].shape[0]
    embedding_dim = state_dict['embedding.weight'].shape[1]
    hidden_dims = [64, 64]
    n_patterns = 16
    
    model = ParadoxNetComplex(vocab_size, embedding_dim, hidden_dims, n_patterns)
    model.load_state_dict(state_dict)
    model.eval()
    
    test_text = "hello"
    chars = sorted(list(set(test_text + "abcdefghijklmnopqrstuvwxyz ")))
    vocab = {ch: i for i, ch in enumerate(chars)}
    tokens = torch.tensor([vocab.get(ch, 0) for ch in test_text], dtype=torch.long)
    tokens = tokens.unsqueeze(0)
    
    # Get baseline output
    with torch.no_grad():
        baseline_output = model(tokens)
    
    # Test ablations - what happens if we zero out each layer's contribution?
    ablation_results = {}
    
    # This is tricky because we need to intervene in the forward pass
    # For now, let's analyze the pattern dictionaries to see what each layer "knows"
    
    layer_knowledge = {}
    for i, layer in enumerate(model.hidden_layers):
        patterns = layer.pattern_dict.detach()
        
        # Analyze pattern "vocabulary" - which patterns are most distinctive?
        pattern_distances = torch.zeros(n_patterns, n_patterns)
        for p1 in range(n_patterns):
            for p2 in range(p1+1, n_patterns):
                dist = torch.norm(patterns[p1] - patterns[p2]).item()
                pattern_distances[p1, p2] = dist
                pattern_distances[p2, p1] = dist
        
        layer_knowledge[f'layer_{i}'] = {
            'pattern_diversity': pattern_distances.mean().item(),
            'most_unique_pattern': pattern_distances.sum(dim=1).argmax().item(),
            'pattern_magnitudes': torch.abs(patterns).mean(dim=1).numpy(),
            'pattern_phase_spread': torch.angle(patterns).std(dim=1).numpy()
        }
    
    return layer_knowledge

def create_layer_output_visualization(results, layer_knowledge):
    """Visualize how layers contribute to outputs."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Output prediction confidence by input type
    case_names = list(results.keys())
    max_probs = [results[case]['predicted_probs'][0] for case in case_names]  # Top prediction confidence
    
    axes[0, 0].bar(case_names, max_probs)
    axes[0, 0].set_title('Prediction Confidence by Input Type')
    axes[0, 0].set_ylabel('Max Probability')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Layer representation norms
    layer_norms_by_case = {}
    for layer in ['hidden_0', 'hidden_1']:
        layer_norms_by_case[layer] = [results[case].get(layer, {}).get('representation_norm', 0) for case in case_names]
    
    x_pos = range(len(case_names))
    width = 0.35
    axes[0, 1].bar([p - width/2 for p in x_pos], layer_norms_by_case['hidden_0'], width, label='Layer 0', alpha=0.7)
    axes[0, 1].bar([p + width/2 for p in x_pos], layer_norms_by_case['hidden_1'], width, label='Layer 1', alpha=0.7)
    axes[0, 1].set_title('Layer Representation Strength')
    axes[0, 1].set_ylabel('Representation Norm')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(case_names, rotation=45)
    axes[0, 1].legend()
    
    # Plot 3: Consensus vs individual contributions
    consensus_mags = [results[case].get('consensus_analysis', {}).get('consensus_magnitude', 0) for case in case_names]
    
    axes[0, 2].bar(case_names, consensus_mags)
    axes[0, 2].set_title('Consensus Mechanism Strength')
    axes[0, 2].set_ylabel('Consensus Magnitude')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Pattern diversity by layer
    layers = list(layer_knowledge.keys())
    diversities = [layer_knowledge[layer]['pattern_diversity'] for layer in layers]
    
    axes[1, 0].bar(layers, diversities)
    axes[1, 0].set_title('Pattern Diversity by Layer')
    axes[1, 0].set_ylabel('Average Pattern Distance')
    
    # Plot 5: Top predicted characters by case
    for i, case in enumerate(case_names[:6]):  # First 6 cases
        predicted_chars = results[case]['predicted_chars'][:3]  # Top 3
        predicted_probs = results[case]['predicted_probs'][:3]
        
        if i < 6:
            row, col = 1, (i % 3) + 1
            if row < 2 and col < 3:
                axes[row, col].bar(predicted_chars, predicted_probs)
                axes[row, col].set_title(f'{case}: {results[case]["text"]}')
                axes[row, col].set_ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig('layer_output_analysis.png', dpi=150, bbox_inches='tight')
    return fig

def main():
    print("=== Layer-to-Output Analysis ===\n")
    
    print("1. Analyzing how layers contribute to final predictions...")
    results = analyze_layer_to_output_mapping()
    
    print("\nOutput Analysis by Input Type:")
    for case_name, data in results.items():
        print(f"\n{case_name}: '{data['text']}'")
        print(f"  Top predictions: {list(zip(data['predicted_chars'][:3], data['predicted_probs'][:3]))}")
        
        if 'hidden_0' in data:
            print(f"  Layer 0 strength: {data['hidden_0']['representation_norm']:.3f}")
        if 'hidden_1' in data:
            print(f"  Layer 1 strength: {data['hidden_1']['representation_norm']:.3f}")
        
        if 'consensus_analysis' in data:
            consensus = data['consensus_analysis']
            print(f"  Consensus strength: {consensus['consensus_magnitude']:.3f}")
            print(f"  Layer balance (L0/L1): {consensus['layer_balance']:.2f}")
    
    print("\n2. Analyzing layer-specific knowledge...")
    layer_knowledge = analyze_output_sensitivity()
    
    print("\nLayer Knowledge Analysis:")
    for layer_name, knowledge in layer_knowledge.items():
        print(f"\n{layer_name}:")
        print(f"  Pattern diversity: {knowledge['pattern_diversity']:.3f}")
        print(f"  Most unique pattern: {knowledge['most_unique_pattern']}")
        print(f"  Pattern magnitude range: {knowledge['pattern_magnitudes'].min():.4f} - {knowledge['pattern_magnitudes'].max():.4f}")
    
    print("\n3. Creating visualizations...")
    fig = create_layer_output_visualization(results, layer_knowledge)
    plt.show()
    
    print("\n=== LAYER FUNCTION SUMMARY ===")
    
    # Determine what each layer does based on analysis
    avg_layer0_strength = np.mean([results[case].get('hidden_0', {}).get('representation_norm', 0) for case in results])
    avg_layer1_strength = np.mean([results[case].get('hidden_1', {}).get('representation_norm', 0) for case in results])
    
    print(f"Layer 0 average strength: {avg_layer0_strength:.3f}")
    print(f"Layer 1 average strength: {avg_layer1_strength:.3f}")
    
    if avg_layer1_strength > avg_layer0_strength * 1.2:
        print("→ Layer 1 appears to be the PRIMARY processor")
    elif avg_layer0_strength > avg_layer1_strength * 1.2:
        print("→ Layer 0 appears to be the PRIMARY processor")
    else:
        print("→ Layers 0 and 1 work in BALANCED cooperation")
    
    print("\nGenerated: layer_output_analysis.png")

if __name__ == "__main__":
    main()