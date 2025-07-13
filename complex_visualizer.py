#!/usr/bin/env python3
"""
Advanced visualization tools for complex number representations in ParadoxNetComplex.
Helps understand the geometry and dynamics of complex-valued activations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.colors import hsv_to_rgb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional

class ComplexVisualizer:
    """Visualizes complex number representations and their geometric properties."""
    
    def __init__(self):
        self.color_schemes = {
            'phase': 'hsv',
            'magnitude': 'viridis',
            'default': 'coolwarm'
        }
    
    def plot_complex_plane(self, complex_tensor: torch.Tensor, 
                          title: str = "Complex Plane Visualization",
                          max_points: int = 1000,
                          save_path: str = None) -> plt.Figure:
        """Plot complex numbers on the complex plane with color-coded phase/magnitude."""
        
        # Flatten and sample if too many points
        flat_complex = complex_tensor.flatten()
        if len(flat_complex) > max_points:
            indices = torch.randperm(len(flat_complex))[:max_points]
            flat_complex = flat_complex[indices]
        
        # Extract real and imaginary parts
        real_parts = flat_complex.real.numpy()
        imag_parts = flat_complex.imag.numpy()
        magnitudes = torch.abs(flat_complex).numpy()
        phases = torch.angle(flat_complex).numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Magnitude-colored scatter
        scatter1 = axes[0].scatter(real_parts, imag_parts, c=magnitudes, 
                                  cmap='viridis', alpha=0.6, s=20)
        axes[0].set_xlabel('Real')
        axes[0].set_ylabel('Imaginary') 
        axes[0].set_title('Magnitude-Colored')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linewidth=0.5)
        axes[0].axvline(x=0, color='k', linewidth=0.5)
        plt.colorbar(scatter1, ax=axes[0], label='Magnitude')
        
        # Plot 2: Phase-colored scatter
        scatter2 = axes[1].scatter(real_parts, imag_parts, c=phases, 
                                  cmap='hsv', alpha=0.6, s=20)
        axes[1].set_xlabel('Real')
        axes[1].set_ylabel('Imaginary')
        axes[1].set_title('Phase-Colored')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linewidth=0.5)
        axes[1].axvline(x=0, color='k', linewidth=0.5)
        plt.colorbar(scatter2, ax=axes[1], label='Phase (radians)')
        
        # Plot 3: Magnitude distribution
        axes[2].hist(magnitudes, bins=50, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Magnitude')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Magnitude Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_phase_magnitude_evolution(self, complex_history: List[torch.Tensor],
                                     titles: Optional[List[str]] = None,
                                     save_path: str = None) -> plt.Figure:
        """Plot how phase and magnitude distributions evolve over time/layers."""
        
        n_timesteps = len(complex_history)
        if titles is None:
            titles = [f'Step {i}' for i in range(n_timesteps)]
        
        fig, axes = plt.subplots(2, n_timesteps, figsize=(4*n_timesteps, 8))
        if n_timesteps == 1:
            axes = axes.reshape(2, 1)
        
        for i, (complex_tensor, title) in enumerate(zip(complex_history, titles)):
            flat_complex = complex_tensor.flatten()
            magnitudes = torch.abs(flat_complex).numpy()
            phases = torch.angle(flat_complex).numpy()
            
            # Magnitude histogram
            axes[0, i].hist(magnitudes, bins=30, alpha=0.7, edgecolor='black')
            axes[0, i].set_title(f'{title} - Magnitude')
            axes[0, i].set_xlabel('Magnitude')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(True, alpha=0.3)
            
            # Phase histogram
            axes[1, i].hist(phases, bins=30, alpha=0.7, edgecolor='black', 
                           range=(-np.pi, np.pi))
            axes[1, i].set_title(f'{title} - Phase')
            axes[1, i].set_xlabel('Phase (radians)')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].set_xlim(-np.pi, np.pi)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_complex_heatmap(self, complex_tensor: torch.Tensor,
                            mode: str = 'magnitude',
                            title: str = None,
                            save_path: str = None) -> plt.Figure:
        """Plot complex tensor as a heatmap (magnitude, phase, or real/imag)."""
        
        if complex_tensor.dim() > 2:
            # Take the first 2D slice if higher dimensional
            complex_2d = complex_tensor.view(-1, complex_tensor.shape[-1])
        else:
            complex_2d = complex_tensor
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if mode == 'magnitude':
            magnitude = torch.abs(complex_2d).numpy()
            im1 = axes[0].imshow(magnitude, cmap='viridis', aspect='auto')
            axes[0].set_title('Magnitude')
            plt.colorbar(im1, ax=axes[0])
            
            phase = torch.angle(complex_2d).numpy()
            im2 = axes[1].imshow(phase, cmap='hsv', aspect='auto', vmin=-np.pi, vmax=np.pi)
            axes[1].set_title('Phase')
            plt.colorbar(im2, ax=axes[1])
            
        elif mode == 'real_imag':
            real_part = complex_2d.real.numpy()
            imag_part = complex_2d.imag.numpy()
            
            im1 = axes[0].imshow(real_part, cmap='RdBu_r', aspect='auto')
            axes[0].set_title('Real Part')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(imag_part, cmap='RdBu_r', aspect='auto')
            axes[1].set_title('Imaginary Part')
            plt.colorbar(im2, ax=axes[1])
        
        if title:
            plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_interactive_3d_plot(self, complex_tensor: torch.Tensor,
                                  max_points: int = 2000,
                                  title: str = "3D Complex Visualization") -> go.Figure:
        """Create interactive 3D plot with real, imaginary, and magnitude as dimensions."""
        
        flat_complex = complex_tensor.flatten()
        if len(flat_complex) > max_points:
            indices = torch.randperm(len(flat_complex))[:max_points]
            flat_complex = flat_complex[indices]
        
        real_parts = flat_complex.real.numpy()
        imag_parts = flat_complex.imag.numpy()
        magnitudes = torch.abs(flat_complex).numpy()
        phases = torch.angle(flat_complex).numpy()
        
        # Normalize phases to [0, 1] for color mapping
        normalized_phases = (phases + np.pi) / (2 * np.pi)
        
        fig = go.Figure(data=go.Scatter3d(
            x=real_parts,
            y=imag_parts,
            z=magnitudes,
            mode='markers',
            marker=dict(
                size=3,
                color=normalized_phases,
                colorscale='HSV',
                showscale=True,
                colorbar=dict(title="Phase"),
                opacity=0.7
            ),
            text=[f'Real: {r:.3f}<br>Imag: {i:.3f}<br>Mag: {m:.3f}<br>Phase: {p:.3f}' 
                  for r, i, m, p in zip(real_parts, imag_parts, magnitudes, phases)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Real',
                yaxis_title='Imaginary',
                zaxis_title='Magnitude'
            )
        )
        
        return fig
    
    def plot_pattern_geometry(self, pattern_dict: torch.Tensor,
                             title: str = "Pattern Dictionary Geometry",
                             save_path: str = None) -> plt.Figure:
        """Visualize the geometric relationships between patterns."""
        
        n_patterns = pattern_dict.shape[0]
        
        # Compute pairwise distances and similarities
        distances = torch.zeros(n_patterns, n_patterns)
        similarities = torch.zeros(n_patterns, n_patterns)
        
        for i in range(n_patterns):
            for j in range(n_patterns):
                p1 = pattern_dict[i].flatten()
                p2 = pattern_dict[j].flatten()
                
                # Complex distance
                distances[i, j] = torch.abs(p1 - p2).mean()
                
                # Complex similarity (using conjugate)
                similarity = torch.real(torch.sum(p1 * torch.conj(p2))) / (
                    torch.abs(p1).sum() * torch.abs(p2).sum() + 1e-8
                )
                similarities[i, j] = similarity
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Distance matrix
        im1 = axes[0, 0].imshow(distances.numpy(), cmap='viridis')
        axes[0, 0].set_title('Pattern Distance Matrix')
        axes[0, 0].set_xlabel('Pattern Index')
        axes[0, 0].set_ylabel('Pattern Index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Similarity matrix  
        im2 = axes[0, 1].imshow(similarities.numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 1].set_title('Pattern Similarity Matrix')
        axes[0, 1].set_xlabel('Pattern Index')
        axes[0, 1].set_ylabel('Pattern Index')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Pattern magnitudes
        pattern_mags = torch.abs(pattern_dict).mean(dim=1).numpy()
        axes[1, 0].bar(range(n_patterns), pattern_mags)
        axes[1, 0].set_title('Pattern Magnitudes')
        axes[1, 0].set_xlabel('Pattern Index')
        axes[1, 0].set_ylabel('Mean Magnitude')
        
        # Pattern phases
        pattern_phases = torch.angle(pattern_dict).mean(dim=1).numpy()
        axes[1, 1].bar(range(n_patterns), pattern_phases)
        axes[1, 1].set_title('Pattern Phases')
        axes[1, 1].set_xlabel('Pattern Index')
        axes[1, 1].set_ylabel('Mean Phase (radians)')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_rotation_dynamics(self, complex_sequence: List[torch.Tensor],
                              pattern_idx: int = 0,
                              save_path: str = None) -> plt.Figure:
        """Visualize how a specific pattern rotates in complex space over time."""
        
        if len(complex_sequence) < 2:
            raise ValueError("Need at least 2 time points to show dynamics")
        
        # Extract the specific pattern over time
        pattern_evolution = [seq[pattern_idx] if seq.dim() > 1 else seq for seq in complex_sequence]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot trajectory in complex plane
        for i in range(len(pattern_evolution)):
            pattern = pattern_evolution[i].flatten()
            real_parts = pattern.real.numpy()
            imag_parts = pattern.imag.numpy()
            
            # Color by time step
            colors = plt.cm.viridis(i / len(pattern_evolution))
            axes[0].scatter(real_parts, imag_parts, c=[colors], alpha=0.7, 
                           s=20, label=f'Step {i}')
            
            # Draw arrows showing movement
            if i > 0:
                prev_pattern = pattern_evolution[i-1].flatten()
                prev_real = prev_pattern.real.numpy()
                prev_imag = prev_pattern.imag.numpy()
                
                for j in range(min(len(real_parts), 10)):  # Show only first 10 for clarity
                    axes[0].arrow(prev_real[j], prev_imag[j], 
                                 real_parts[j] - prev_real[j], 
                                 imag_parts[j] - prev_imag[j],
                                 head_width=0.01, head_length=0.01, 
                                 fc=colors, ec=colors, alpha=0.5)
        
        axes[0].set_xlabel('Real')
        axes[0].set_ylabel('Imaginary')
        axes[0].set_title(f'Pattern {pattern_idx} Trajectory')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot magnitude and phase evolution
        magnitudes = [torch.abs(p).mean().item() for p in pattern_evolution]
        phases = [torch.angle(p).mean().item() for p in pattern_evolution]
        
        ax2_twin = axes[1].twinx()
        
        line1 = axes[1].plot(magnitudes, 'b-o', label='Magnitude')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Magnitude', color='b')
        axes[1].tick_params(axis='y', labelcolor='b')
        
        line2 = ax2_twin.plot(phases, 'r-s', label='Phase')
        ax2_twin.set_ylabel('Phase (radians)', color='r')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        
        axes[1].set_title(f'Pattern {pattern_idx} Magnitude/Phase Evolution')
        
        # Combine legends
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_analysis(self, model, input_text: str, vocab: Dict = None):
        """Create a comprehensive visual analysis of complex representations."""
        
        # Simple tokenization if no vocab provided
        if vocab is None:
            chars = sorted(list(set(input_text)))
            vocab = {ch: i for i, ch in enumerate(chars)}
        
        tokens = torch.tensor([vocab.get(ch, 0) for ch in input_text[:50]], dtype=torch.long)
        tokens = tokens.unsqueeze(0)
        
        # Capture activations
        activations = {}
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach().cpu()
                else:
                    activations[name] = output.detach().cpu()
            return hook
        
        hooks = []
        for i, layer in enumerate(model.hidden_layers):
            hooks.append(layer.register_forward_hook(make_hook(f'layer_{i}')))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(tokens)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Create visualizations
        print("Creating complex plane visualizations...")
        for layer_name, activation in activations.items():
            if activation.dtype in [torch.complex64, torch.complex128]:
                fig = self.plot_complex_plane(activation, 
                                            title=f"{layer_name.replace('_', ' ').title()} - Complex Plane")
                plt.show()
        
        print("Creating pattern geometry analysis...")
        for i, layer in enumerate(model.hidden_layers):
            if hasattr(layer, 'pattern_dict'):
                fig = self.plot_pattern_geometry(layer.pattern_dict.detach().cpu(),
                                               title=f"Layer {i} Pattern Geometry")
                plt.show()

def main():
    """Example usage of the complex visualizer."""
    from paradox_net_complex import ParadoxNetComplex
    
    # Create a dummy model
    model = ParadoxNetComplex(
        vocab_size=100,
        embedding_dim=64,
        hidden_dims=[48, 48],
        n_patterns=8
    )
    
    visualizer = ComplexVisualizer()
    
    # Generate some complex data for demonstration
    complex_data = torch.randn(100, 24, dtype=torch.cfloat)
    
    # Create various visualizations
    print("Creating complex plane plot...")
    fig1 = visualizer.plot_complex_plane(complex_data)
    plt.show()
    
    print("Creating complex heatmap...")
    fig2 = visualizer.plot_complex_heatmap(complex_data[:10, :10])
    plt.show()
    
    print("Creating pattern geometry plot...")
    pattern_data = torch.randn(8, 24, dtype=torch.cfloat)
    fig3 = visualizer.plot_pattern_geometry(pattern_data)
    plt.show()
    
    # Interactive 3D plot (uncomment if plotly is available)
    # print("Creating interactive 3D plot...")
    # fig4 = visualizer.create_interactive_3d_plot(complex_data)
    # fig4.show()

if __name__ == "__main__":
    main()