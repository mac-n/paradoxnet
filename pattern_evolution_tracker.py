#!/usr/bin/env python3
"""
Track pattern dictionary evolution during ParadoxNetComplex training.
Helps understand how patterns specialize over time.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
from collections import defaultdict

class PatternEvolutionTracker:
    """Tracks how pattern dictionaries evolve during training."""
    
    def __init__(self, model):
        self.model = model
        self.pattern_history = defaultdict(list)
        self.similarity_history = defaultdict(list)
        self.specialization_history = defaultdict(list)
        self.epoch_count = 0
        
    def capture_patterns(self, epoch: int = None):
        """Capture current pattern state for all layers."""
        if epoch is None:
            epoch = self.epoch_count
            self.epoch_count += 1
        
        for i, layer in enumerate(self.model.hidden_layers):
            patterns = layer.pattern_dict.detach().cpu().clone()
            
            # Store raw patterns
            self.pattern_history[f'layer_{i}'].append({
                'epoch': epoch,
                'patterns': patterns.numpy(),
                'magnitudes': patterns.abs().mean(dim=1).numpy(),
                'phases': patterns.angle().mean(dim=1).numpy()
            })
            
            # Compute pattern similarities
            similarities = self._compute_pattern_similarities(patterns)
            self.similarity_history[f'layer_{i}'].append({
                'epoch': epoch,
                'similarities': similarities,
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities)
            })
            
            # Compute specialization metrics
            specialization = self._compute_specialization_metrics(patterns)
            self.specialization_history[f'layer_{i}'].append({
                'epoch': epoch,
                **specialization
            })
    
    def _compute_pattern_similarities(self, patterns: torch.Tensor) -> List[float]:
        """Compute pairwise cosine similarities between patterns."""
        similarities = []
        for i in range(patterns.shape[0]):
            for j in range(i + 1, patterns.shape[0]):
                p1 = patterns[i].flatten()
                p2 = patterns[j].flatten()
                sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item()
                similarities.append(sim)
        return similarities
    
    def _compute_specialization_metrics(self, patterns: torch.Tensor) -> Dict[str, float]:
        """Compute various specialization metrics for patterns."""
        # Magnitude diversity
        magnitudes = patterns.abs().mean(dim=1)
        magnitude_diversity = magnitudes.std() / (magnitudes.mean() + 1e-8)
        
        # Phase diversity
        phases = patterns.angle().mean(dim=1)
        phase_diversity = torch.std(phases) / (torch.pi + 1e-8)
        
        # Pattern orthogonality (lower similarity = higher orthogonality)
        similarities = self._compute_pattern_similarities(patterns)
        orthogonality = 1.0 - np.mean(similarities)
        
        # Pattern norm variance
        norms = torch.norm(patterns, dim=1)
        norm_variance = norms.std() / (norms.mean() + 1e-8)
        
        return {
            'magnitude_diversity': magnitude_diversity.item(),
            'phase_diversity': phase_diversity.item(),
            'orthogonality': orthogonality,
            'norm_variance': norm_variance.item()
        }
    
    def analyze_pattern_evolution(self, layer_idx: int = 0) -> Dict[str, Any]:
        """Analyze how patterns evolved for a specific layer."""
        layer_key = f'layer_{layer_idx}'
        
        if layer_key not in self.pattern_history:
            return {'error': f'No data for {layer_key}'}
        
        history = self.pattern_history[layer_key]
        sim_history = self.similarity_history[layer_key]
        spec_history = self.specialization_history[layer_key]
        
        n_epochs = len(history)
        n_patterns = history[0]['patterns'].shape[0]
        
        analysis = {
            'n_epochs': n_epochs,
            'n_patterns': n_patterns,
            'pattern_trajectories': self._analyze_pattern_trajectories(history),
            'similarity_evolution': self._analyze_similarity_evolution(sim_history),
            'specialization_evolution': self._analyze_specialization_evolution(spec_history),
            'pattern_stability': self._analyze_pattern_stability(history),
            'convergence_analysis': self._analyze_convergence(history)
        }
        
        return analysis
    
    def _analyze_pattern_trajectories(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze how individual patterns moved through parameter space."""
        n_patterns = history[0]['patterns'].shape[0]
        
        trajectories = {
            'magnitude_trajectories': [],
            'phase_trajectories': [],
            'displacement_per_epoch': []
        }
        
        for pattern_idx in range(n_patterns):
            mag_traj = [h['magnitudes'][pattern_idx] for h in history]
            phase_traj = [h['phases'][pattern_idx] for h in history]
            
            trajectories['magnitude_trajectories'].append(mag_traj)
            trajectories['phase_trajectories'].append(phase_traj)
            
            # Compute displacement between consecutive epochs
            displacements = []
            for i in range(1, len(history)):
                prev_pattern = history[i-1]['patterns'][pattern_idx]
                curr_pattern = history[i]['patterns'][pattern_idx]
                displacement = np.linalg.norm(curr_pattern - prev_pattern)
                displacements.append(displacement)
            trajectories['displacement_per_epoch'].append(displacements)
        
        # Summary statistics
        trajectories['mean_final_magnitude'] = np.mean([traj[-1] for traj in trajectories['magnitude_trajectories']])
        trajectories['magnitude_convergence'] = np.mean([
            abs(traj[-1] - traj[-5]) if len(traj) >= 5 else abs(traj[-1] - traj[0])
            for traj in trajectories['magnitude_trajectories']
        ])
        
        return trajectories
    
    def _analyze_similarity_evolution(self, sim_history: List[Dict]) -> Dict[str, Any]:
        """Analyze how pattern similarities changed over time."""
        epochs = [h['epoch'] for h in sim_history]
        mean_sims = [h['mean_similarity'] for h in sim_history]
        std_sims = [h['std_similarity'] for h in sim_history]
        
        return {
            'epochs': epochs,
            'mean_similarities': mean_sims,
            'std_similarities': std_sims,
            'initial_similarity': mean_sims[0] if mean_sims else 0,
            'final_similarity': mean_sims[-1] if mean_sims else 0,
            'similarity_trend': np.polyfit(epochs, mean_sims, 1)[0] if len(epochs) > 1 else 0
        }
    
    def _analyze_specialization_evolution(self, spec_history: List[Dict]) -> Dict[str, Any]:
        """Analyze how specialization metrics evolved."""
        metrics = ['magnitude_diversity', 'phase_diversity', 'orthogonality', 'norm_variance']
        
        evolution = {}
        for metric in metrics:
            values = [h[metric] for h in spec_history]
            evolution[metric] = {
                'values': values,
                'initial': values[0] if values else 0,
                'final': values[-1] if values else 0,
                'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
            }
        
        return evolution
    
    def _analyze_pattern_stability(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze which patterns became stable vs. which kept changing."""
        if len(history) < 2:
            return {'error': 'Need at least 2 epochs for stability analysis'}
        
        n_patterns = history[0]['patterns'].shape[0]
        pattern_stability = []
        
        for pattern_idx in range(n_patterns):
            # Compute variance in pattern parameters over last few epochs
            recent_patterns = [h['patterns'][pattern_idx] for h in history[-5:]]
            variance = np.var(recent_patterns, axis=0).mean()
            pattern_stability.append(variance)
        
        # Find most and least stable patterns
        most_stable_idx = np.argmin(pattern_stability)
        least_stable_idx = np.argmax(pattern_stability)
        
        return {
            'pattern_stabilities': pattern_stability,
            'most_stable_pattern': most_stable_idx,
            'least_stable_pattern': least_stable_idx,
            'stability_range': max(pattern_stability) - min(pattern_stability),
            'mean_stability': np.mean(pattern_stability)
        }
    
    def _analyze_convergence(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze convergence properties of pattern learning."""
        if len(history) < 5:
            return {'converged': False, 'reason': 'Insufficient epochs'}
        
        # Check if patterns have converged by looking at recent changes
        recent_changes = []
        for i in range(-5, -1):
            curr_patterns = history[i]['patterns']
            next_patterns = history[i+1]['patterns']
            change = np.linalg.norm(curr_patterns - next_patterns)
            recent_changes.append(change)
        
        convergence_threshold = 0.01
        converged = all(change < convergence_threshold for change in recent_changes)
        
        return {
            'converged': converged,
            'recent_changes': recent_changes,
            'convergence_threshold': convergence_threshold,
            'average_recent_change': np.mean(recent_changes)
        }
    
    def create_evolution_plots(self, layer_idx: int = 0, save_path: str = None):
        """Create comprehensive plots showing pattern evolution."""
        analysis = self.analyze_pattern_evolution(layer_idx)
        
        if 'error' in analysis:
            print(f"Cannot create plots: {analysis['error']}")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Pattern Evolution Analysis - Layer {layer_idx}')
        
        # Plot 1: Magnitude trajectories
        ax = axes[0, 0]
        trajectories = analysis['pattern_trajectories']['magnitude_trajectories']
        for i, traj in enumerate(trajectories):
            ax.plot(traj, label=f'Pattern {i}', alpha=0.7)
        ax.set_title('Pattern Magnitude Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Magnitude')
        ax.legend()
        
        # Plot 2: Similarity evolution
        ax = axes[0, 1]
        sim_data = analysis['similarity_evolution']
        ax.plot(sim_data['epochs'], sim_data['mean_similarities'], 'b-', label='Mean')
        ax.fill_between(sim_data['epochs'], 
                       np.array(sim_data['mean_similarities']) - np.array(sim_data['std_similarities']),
                       np.array(sim_data['mean_similarities']) + np.array(sim_data['std_similarities']),
                       alpha=0.3, color='blue')
        ax.set_title('Pattern Similarity Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cosine Similarity')
        ax.legend()
        
        # Plot 3: Specialization metrics
        ax = axes[0, 2]
        spec_data = analysis['specialization_evolution']
        for metric, data in spec_data.items():
            ax.plot(data['values'], label=metric.replace('_', ' ').title())
        ax.set_title('Specialization Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.legend()
        
        # Plot 4: Pattern stability
        ax = axes[1, 0]
        stability_data = analysis['pattern_stability']
        ax.bar(range(len(stability_data['pattern_stabilities'])), 
               stability_data['pattern_stabilities'])
        ax.set_title('Pattern Stability (Lower = More Stable)')
        ax.set_xlabel('Pattern Index')
        ax.set_ylabel('Variance')
        
        # Plot 5: Displacement per epoch
        ax = axes[1, 1]
        displacements = analysis['pattern_trajectories']['displacement_per_epoch']
        for i, disp in enumerate(displacements):
            ax.plot(disp, alpha=0.7, label=f'Pattern {i}')
        ax.set_title('Pattern Displacement per Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Parameter Space Distance')
        
        # Plot 6: Convergence analysis
        ax = axes[1, 2]
        conv_data = analysis['convergence_analysis']
        if 'recent_changes' in conv_data:
            ax.plot(conv_data['recent_changes'], 'ro-')
            ax.axhline(y=conv_data['convergence_threshold'], color='g', linestyle='--', 
                      label=f"Threshold ({conv_data['convergence_threshold']})")
            ax.set_title(f"Convergence (Converged: {conv_data['converged']})")
            ax.set_xlabel('Recent Epochs')
            ax.set_ylabel('Pattern Change')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def save_tracking_data(self, filename: str):
        """Save all tracking data to a file."""
        data = {
            'pattern_history': {k: [
                {**item, 'patterns': item['patterns'].tolist(), 
                 'magnitudes': item['magnitudes'].tolist(),
                 'phases': item['phases'].tolist()}
                for item in v
            ] for k, v in self.pattern_history.items()},
            'similarity_history': dict(self.similarity_history),
            'specialization_history': dict(self.specialization_history)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_tracking_data(self, filename: str):
        """Load tracking data from a file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert back to numpy arrays
        for layer_key, history in data['pattern_history'].items():
            for item in history:
                item['patterns'] = np.array(item['patterns'])
                item['magnitudes'] = np.array(item['magnitudes'])
                item['phases'] = np.array(item['phases'])
        
        self.pattern_history = defaultdict(list, data['pattern_history'])
        self.similarity_history = defaultdict(list, data['similarity_history'])
        self.specialization_history = defaultdict(list, data['specialization_history'])

def create_training_callback(tracker: PatternEvolutionTracker, capture_every: int = 1):
    """Create a training callback that captures patterns at specified intervals."""
    
    def callback(epoch: int, model, loss: float = None):
        if epoch % capture_every == 0:
            print(f"Capturing patterns at epoch {epoch}")
            tracker.capture_patterns(epoch)
    
    return callback

def main():
    """Example usage of the pattern evolution tracker."""
    from paradox_net_complex import ParadoxNetComplex
    
    # Create a dummy model
    model = ParadoxNetComplex(
        vocab_size=100,
        embedding_dim=64,
        hidden_dims=[48, 48],
        n_patterns=8
    )
    
    # Create tracker
    tracker = PatternEvolutionTracker(model)
    
    # Simulate training evolution by capturing patterns at different points
    for epoch in range(0, 50, 5):
        # Simulate some parameter changes
        for layer in model.hidden_layers:
            layer.pattern_dict.data += torch.randn_like(layer.pattern_dict.data) * 0.01
        
        tracker.capture_patterns(epoch)
    
    # Analyze evolution
    analysis = tracker.analyze_pattern_evolution(layer_idx=0)
    print("Evolution Analysis:")
    print(f"Convergence: {analysis['convergence_analysis']['converged']}")
    print(f"Most stable pattern: {analysis['pattern_stability']['most_stable_pattern']}")
    
    # Create plots
    tracker.create_evolution_plots(layer_idx=0)
    plt.show()
    
    # Save data
    tracker.save_tracking_data('pattern_evolution.json')

if __name__ == "__main__":
    main()