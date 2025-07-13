#!/usr/bin/env python3
"""
Simple script to run interpretability analysis on a trained ParadoxNetComplex model.
"""

import torch
import matplotlib.pyplot as plt
from interpret_paradox_complex import ParadoxComplexInterpreter
from complex_visualizer import ComplexVisualizer
from paradox_net_complex import ParadoxNetComplex

def load_model_and_analyze():
    """Load trained model and run analysis."""
    
    model_path = "complex_model.pt"
    print(f"Loading model from {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Try to infer model parameters from state dict
        vocab_size = state_dict['embedding.weight'].shape[0]
        embedding_dim = state_dict['embedding.weight'].shape[1]
        
        # Infer hidden dims and n_patterns from layer weights
        hidden_dims = []
        n_patterns = 8  # default
        i = 0
        while f'hidden_layers.{i}.process.weight_re' in state_dict:
            hidden_dims.append(state_dict[f'hidden_layers.{i}.process.weight_re'].shape[1] * 2)
            # Infer n_patterns from pattern_dict shape
            if f'hidden_layers.{i}.pattern_dict' in state_dict:
                n_patterns = state_dict[f'hidden_layers.{i}.pattern_dict'].shape[0]
            i += 1
        
        print(f"Detected n_patterns: {n_patterns}")
        model = ParadoxNetComplex(vocab_size, embedding_dim, hidden_dims, n_patterns)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model: vocab_size={vocab_size}, embedding_dim={embedding_dim}, hidden_dims={hidden_dims}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating demo model instead...")
        model = ParadoxNetComplex(vocab_size=65, embedding_dim=64, hidden_dims=[48, 48])
    
    return model

def main():
    """Run the complete analysis pipeline."""
    
    print("=== ParadoxNetComplex Interpretability Analysis ===\n")
    
    # Load model
    model = load_model_and_analyze()
    model.eval()
    
    # Create interpreters
    interpreter = ParadoxComplexInterpreter(model)
    visualizer = ComplexVisualizer()
    
    # Sample text to analyze
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is the question",
        "Hello world! This is a test of the neural network."
    ]
    
    print("1. Running text analysis...")
    for i, text in enumerate(sample_texts):
        print(f"\nAnalyzing: '{text[:30]}...'")
        
        # Create simple character vocab
        chars = sorted(list(set(text.lower())))
        vocab = {ch: i for i, ch in enumerate(chars)}
        
        try:
            analysis = interpreter.analyze_text(text, vocab)
            
            # Generate report
            report = interpreter.generate_report(analysis, f"analysis_report_{i}.txt")
            print("Generated report:")
            print(report[:500] + "..." if len(report) > 500 else report)
            
            # Create visualizations
            interpreter.create_visualizations(analysis, save_dir=".")
            plt.savefig(f"analysis_{i}.png")
            plt.close()
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            continue
    
    print("\n2. Analyzing pattern dictionaries...")
    try:
        for i, layer in enumerate(model.hidden_layers):
            if hasattr(layer, 'pattern_dict'):
                print(f"Layer {i} patterns:")
                patterns = layer.pattern_dict.detach()
                
                # Basic pattern analysis
                magnitudes = torch.abs(patterns).mean(dim=1)
                phases = torch.angle(patterns).mean(dim=1)
                
                print(f"  Pattern magnitudes: {magnitudes.numpy()}")
                print(f"  Pattern phases: {phases.numpy()}")
                
                # Create pattern visualization
                fig = visualizer.plot_pattern_geometry(patterns, 
                                                     title=f"Layer {i} Pattern Geometry")
                plt.savefig(f"patterns_layer_{i}.png")
                plt.close()
                
                # Create complex plane visualization
                fig = visualizer.plot_complex_plane(patterns, 
                                                   title=f"Layer {i} Pattern Complex Plane")
                plt.savefig(f"complex_plane_layer_{i}.png")
                plt.close()
                
    except Exception as e:
        print(f"Error in pattern analysis: {e}")
    
    print("\n3. Creating comprehensive visualization...")
    try:
        # Test with simple input
        test_text = "hello world"
        chars = sorted(list(set(test_text)))
        vocab = {ch: i for i, ch in enumerate(chars)}
        
        visualizer.create_comprehensive_analysis(model, test_text, vocab)
        
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
    
    # Cleanup
    interpreter.remove_hooks()
    
    print("\nAnalysis complete! Check the generated files:")
    print("  - analysis_report_*.txt: Detailed text analysis reports")  
    print("  - analysis_*.png: Analysis visualizations")
    print("  - patterns_layer_*.png: Pattern geometry plots")
    print("  - complex_plane_layer_*.png: Complex plane visualizations")

if __name__ == "__main__":
    main()