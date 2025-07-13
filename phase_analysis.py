#!/usr/bin/env python3
"""
Analyze the phase-based encoding discovered in ParadoxNetComplex.
Your model seems to use rotational relationships instead of magnitude differences!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from paradox_net_complex import ParadoxNetComplex
from typing import Dict, List, Tuple

class PhaseAnalyzer:
    """Analyze phase-based information encoding in complex representations."""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def analyze_character_phase_encoding(self, test_chars: List[str]) -> Dict:
        """Test how different characters map to phase patterns."""
        
        results = {}
        
        for char in test_chars:
            # Simple vocab for single character
            vocab = {char: 0}
            tokens = torch.tensor([0], dtype=torch.long).unsqueeze(0)
            
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
            
            for i, layer in enumerate(self.model.hidden_layers):
                hooks.append(layer.register_forward_hook(make_hook(f'layer_{i}')))
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(tokens)
            
            # Clean up hooks
            for hook in hooks:
                hook.remove()
            
            # Extract phase information
            layer_phases = {}
            for layer_name, activation in activations.items():
                if activation.dtype in [torch.complex64, torch.complex128]:
                    phases = torch.angle(activation).squeeze()
                    layer_phases[layer_name] = {
                        'mean_phase': phases.mean().item(),
                        'phase_std': phases.std().item(),
                        'phase_range': (phases.min().item(), phases.max().item()),
                        'phase_vector': phases.flatten()[:32].numpy()  # First 32 for comparison
                    }
            
            results[char] = layer_phases
            
        return results
    
    def find_phase_relationships(self, phase_data: Dict) -> Dict:
        """Find systematic phase relationships between characters."""
        
        chars = list(phase_data.keys())
        relationships = {}
        
        for layer_name in ['layer_0', 'layer_1']:
            if layer_name not in phase_data[chars[0]]:
                continue
                
            layer_relationships = {}
            
            # Compute phase differences between character pairs
            for i, char1 in enumerate(chars):
                for j, char2 in enumerate(chars[i+1:], i+1):
                    if layer_name in phase_data[char1] and layer_name in phase_data[char2]:
                        phase1 = phase_data[char1][layer_name]['phase_vector']
                        phase2 = phase_data[char2][layer_name]['phase_vector']
                        
                        # Compute phase differences (handling wraparound)
                        phase_diff = np.angle(np.exp(1j * (phase1 - phase2)))
                        
                        layer_relationships[f'{char1}-{char2}'] = {
                            'mean_phase_diff': np.mean(phase_diff),
                            'phase_diff_std': np.std(phase_diff),
                            'phase_coherence': np.abs(np.mean(np.exp(1j * phase_diff))),  # How consistent is the phase relationship
                            'raw_diff': phase_diff[:10].tolist()  # Sample for inspection
                        }
            
            relationships[layer_name] = layer_relationships
            
        return relationships
    
    def test_sequence_phase_evolution(self, text: str) -> Dict:
        """Analyze how phases evolve through a sequence."""
        
        # Create character vocab
        chars = sorted(list(set(text.lower())))
        vocab = {ch: i for i, ch in enumerate(chars)}
        tokens = torch.tensor([vocab.get(ch, 0) for ch in text[:20]], dtype=torch.long)
        tokens = tokens.unsqueeze(0)
        
        # Capture position-specific activations
        activations = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach().cpu()
                else:
                    activations[name] = output.detach().cpu()
            return hook
        
        for i, layer in enumerate(self.model.hidden_layers):
                hooks.append(layer.register_forward_hook(make_hook(f'layer_{i}')))
        
        with torch.no_grad():
            _ = self.model(tokens)
        
        for hook in hooks:
            hook.remove()
        
        # Analyze phase evolution across sequence positions
        evolution = {}
        for layer_name, activation in activations.items():
            if activation.dtype in [torch.complex64, torch.complex128]:
                # activation shape: [batch, seq_len, features]
                seq_phases = torch.angle(activation).squeeze(0)  # [seq_len, features]
                
                position_analysis = []
                for pos in range(min(seq_phases.shape[0], len(text))):
                    pos_phases = seq_phases[pos]
                    position_analysis.append({
                        'position': pos,
                        'character': text[pos] if pos < len(text) else '',
                        'mean_phase': pos_phases.mean().item(),
                        'phase_std': pos_phases.std().item(),
                        'dominant_phase': pos_phases[pos_phases.abs().argmax()].item()
                    })
                
                evolution[layer_name] = position_analysis
        
        return evolution

    def analyze_contextual_phase(self, sequences: List[str], target_char_index: int = 1) -> Dict:
        """
        Analyzes the phase of a target character within different sequence contexts.
        For example, to see how 'e' is encoded, you could pass sequences like ['he', 'we', 'le'].
        This version re-initializes the model for each sequence to ensure no state is carried over.
        """
        results = {}
        all_chars = sorted(list(set("".join(sequences))))
        vocab = {ch: i for i, ch in enumerate(all_chars)}

        # Load the base state dict once
        try:
            base_checkpoint = torch.load("complex_model.pt", map_location='cpu')
            base_state_dict = base_checkpoint['model_state_dict'] if 'model_state_dict' in base_checkpoint else base_checkpoint
            vocab_size = base_state_dict['embedding.weight'].shape[0]
            embedding_dim = base_state_dict['embedding.weight'].shape[1]
            hidden_dims = []
            i = 0
            while f'hidden_layers.{i}.process.weight_re' in base_state_dict:
                hidden_dims.append(base_state_dict[f'hidden_layers.{i}.process.weight_re'].shape[1] * 2)
                i += 1
            n_patterns = 16 # Assuming this is fixed
        except Exception as e:
            print(f"Error loading base model state: {e}")
            return {}

        for seq in sequences:
            if len(seq) <= target_char_index:
                continue

            # Re-initialize the model for each sequence
            model = ParadoxNetComplex(vocab_size, embedding_dim, hidden_dims, n_patterns)
            model.load_state_dict(base_state_dict)
            model.eval()

            tokens = torch.tensor([vocab.get(ch, 0) for ch in seq], dtype=torch.long).unsqueeze(0)

            activations = {}
            hooks = []
            def make_hook(name):
                def hook(module, input, output):
                    activations[name] = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
                return hook

            for i, layer in enumerate(model.hidden_layers):
                hooks.append(layer.register_forward_hook(make_hook(f'layer_{i}')))

            with torch.no_grad():
                _ = model(tokens)

            for hook in hooks:
                hook.remove()

            seq_results = {}
            for layer_name, activation in activations.items():
                if activation.dtype in [torch.complex64, torch.complex128]:
                    seq_phases = torch.angle(activation).squeeze(0)
                    target_phases = seq_phases[target_char_index]
                    seq_results[layer_name] = {
                        'mean_phase': target_phases.mean().item(),
                        'phase_std': target_phases.std().item(),
                        'phase_vector': target_phases.numpy()
                    }
            results[seq] = seq_results
        return results

    def visualize_contextual_phase_rotation(self, contextual_data: Dict, layer_name: str, save_path: str = None):
        """
        Creates a polar plot to visualize contextual phase rotation.
        `contextual_data` should be the output from `analyze_contextual_phase`.
        """
        if not contextual_data:
            print("No data to visualize.")
            return

        labels = list(contextual_data.keys())
        mean_phases = [data[layer_name]['mean_phase'] for data in contextual_data.values() if layer_name in data]
        
        if not mean_phases:
            print(f"No data found for layer '{layer_name}'.")
            return

        # Use standard deviation for the radius, normalized for better visualization
        stds = [data[layer_name]['phase_std'] for data in contextual_data.values() if layer_name in data]
        radii = 1.0 - (np.array(stds) / np.pi) # Lower std = higher confidence = longer radius

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)

        # Plot each point
        for i, (label, phase, radius) in enumerate(zip(labels, mean_phases, radii)):
            ax.plot([phase, phase], [0, radius], marker='o', markersize=8, linestyle='-', label=label)
            # Add labels slightly offset from the point
            ax.text(phase, radius + 0.1, label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=12)

        ax.set_title(f'Contextual Phase Encoding in {layer_name.replace("_", " ").title()}', fontsize=16)
        ax.set_yticklabels([]) # Hide radial ticks
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'])
        ax.grid(True)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_phase_encoding(self, phase_data: Dict, save_path: str = None):
        """Create visualizations of phase-based encoding."""
        
        chars = list(phase_data.keys())
        n_chars = len(chars)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Mean phase per character per layer
        layer_names = ['layer_0', 'layer_1']
        for i, layer_name in enumerate(layer_names):
            if layer_name in phase_data[chars[0]]:
                mean_phases = [phase_data[char][layer_name]['mean_phase'] for char in chars]
                axes[0, i].bar(chars, mean_phases)
                axes[0, i].set_title(f'{layer_name.replace("_", " ").title()} - Mean Phase per Character')
                axes[0, i].set_ylabel('Mean Phase (radians)')
                axes[0, i].tick_params(axis='x', rotation=45)
        
        # Plot 2: Phase standard deviation (how spread out phases are)
        for i, layer_name in enumerate(layer_names):
            if layer_name in phase_data[chars[0]]:
                phase_stds = [phase_data[char][layer_name]['phase_std'] for char in chars]
                axes[1, i].bar(chars, phase_stds)
                axes[1, i].set_title(f'{layer_name.replace("_", " ").title()} - Phase Spread per Character')
                axes[1, i].set_ylabel('Phase Std Dev')
                axes[1, i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_phase_relationships(self, relationships: Dict, save_path: str = None):
        """Visualize phase relationships between character pairs."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, layer_name in enumerate(['layer_0', 'layer_1']):
            if layer_name in relationships:
                pairs = list(relationships[layer_name].keys())
                coherences = [relationships[layer_name][pair]['phase_coherence'] for pair in pairs]
                
                axes[i].bar(range(len(pairs)), coherences)
                axes[i].set_title(f'{layer_name.replace("_", " ").title()} - Phase Coherence Between Characters')
                axes[i].set_ylabel('Phase Coherence (0-1)')
                axes[i].set_xlabel('Character Pairs')
                axes[i].set_xticks(range(len(pairs)))
                axes[i].set_xticklabels(pairs, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig

def main():
    """Run phase analysis on the trained model."""
    
    print("=== Phase-Based Encoding Analysis ===\n")
    
    # Load model
    try:
        checkpoint = torch.load("complex_model.pt", map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        vocab_size = state_dict['embedding.weight'].shape[0]
        embedding_dim = state_dict['embedding.weight'].shape[1]
        
        hidden_dims = []
        n_patterns = 16  # We know this from previous analysis
        i = 0
        while f'hidden_layers.{i}.process.weight_re' in state_dict:
            hidden_dims.append(state_dict[f'hidden_layers.{i}.process.weight_re'].shape[1] * 2)
            i += 1
        
        model = ParadoxNetComplex(vocab_size, embedding_dim, hidden_dims, n_patterns)
        model.load_state_dict(state_dict)
        print(f"Loaded model: vocab_size={vocab_size}, embedding_dim={embedding_dim}, hidden_dims={hidden_dims}, n_patterns={n_patterns}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    analyzer = PhaseAnalyzer(model)
    
    # Test common characters
    test_chars = ['a', 'e', 'i', 'o', 'u', 't', 'h', 'n', 's', 'r']  # Vowels + common consonants
    print(f"Testing characters: {test_chars}")
    
    print("\n1. Analyzing character-specific phase encoding...")
    phase_data = analyzer.analyze_character_phase_encoding(test_chars)
    
    # Print some results
    for char in test_chars[:3]:  # First 3 chars
        print(f"\nCharacter '{char}':")
        for layer_name, data in phase_data[char].items():
            print(f"  {layer_name}: mean_phase={data['mean_phase']:.3f}, std={data['phase_std']:.3f}")
    
    print("\n2. Finding phase relationships between characters...")
    relationships = analyzer.find_phase_relationships(phase_data)
    
    # Print interesting relationships
    for layer_name, layer_rels in relationships.items():
        print(f"\n{layer_name} relationships:")
        # Sort by coherence to find strongest relationships
        sorted_pairs = sorted(layer_rels.items(), key=lambda x: x[1]['phase_coherence'], reverse=True)
        for pair, data in sorted_pairs[:5]:  # Top 5 most coherent relationships
            print(f"  {pair}: coherence={data['phase_coherence']:.3f}, mean_diff={data['mean_phase_diff']:.3f}")
    
    print("\n3. Testing sequence evolution...")
    evolution = analyzer.test_sequence_phase_evolution("hello world")
    
    for layer_name, pos_data in evolution.items():
        print(f"\n{layer_name} sequence evolution:")
        for item in pos_data[:5]:  # First 5 positions
            print(f"  pos {item['position']} ('{item['character']}'): mean_phase={item['mean_phase']:.3f}")
    
    print("\n4. Creating visualizations...")
    fig1 = analyzer.visualize_phase_encoding(phase_data, "phase_encoding_analysis.png")
    plt.show()
    
    fig2 = analyzer.visualize_phase_relationships(relationships, "phase_relationships.png")
    plt.show()

    print("\n5. Analyzing contextual phase rotation...")
    # We will analyze the phase of the character 'e' when it is preceded by different characters.
    # This will show how the network's representation of 'e' changes based on context.
    context_sequences = ['he', 've', 're', 'le', 'de', 'ne', 'se', 'ce']
    contextual_data = analyzer.analyze_contextual_phase(context_sequences, target_char_index=1)

    print("Contextual analysis for character 'e':")
    for seq, data in contextual_data.items():
        if 'layer_1' in data:
            print(f"  Sequence '{seq}': Layer 1 Mean Phase = {data['layer_1']['mean_phase']:.3f}")

    print("\n6. Creating new contextual visualization...")
    fig3 = analyzer.visualize_contextual_phase_rotation(contextual_data, 'layer_1', 'contextual_phase_rotation.png')
    plt.show()
    
    print("\nPhase analysis complete!")
    print("Generated files:")
    print("  - phase_encoding_analysis.png: How different characters map to phases")
    print("  - phase_relationships.png: Phase coherence between character pairs")
    print("  - contextual_phase_rotation.png: How a character's phase changes with context")


if __name__ == "__main__":
    main()
