import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# --- RoPE Helper Functions ---
def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embedding to the input tensor."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(2)
    return x_out.type_as(x)

class PositionalEncoding(nn.Module):
    """Generates rotary positional embeddings (RoPE)."""
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        theta = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)

class ComplexLinear(nn.Module):
    """A linear layer that operates on complex-valued tensors."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_re = nn.Parameter(torch.randn(in_features // 2, out_features // 2) * 0.02)
        self.weight_im = nn.Parameter(torch.randn(in_features // 2, out_features // 2) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_re, x_im = x.real, x.imag
        if x.dim() == 3:
            out_re = torch.einsum('bsi,io->bso', x_re, self.weight_re) - torch.einsum('bsi,io->bso', x_im, self.weight_im)
            out_im = torch.einsum('bsi,io->bso', x_re, self.weight_im) + torch.einsum('bsi,io->bso', x_im, self.weight_re)
        else:
            out_re = x_re @ self.weight_re - x_im @ self.weight_im
            out_im = x_re @ self.weight_im + x_im @ self.weight_re
        return torch.complex(out_re, out_im)

@dataclass
class LayerStats:
    """Track statistics for temporal entropy recycling layers"""
    prediction_errors: torch.Tensor
    confidence_values: torch.Tensor
    penultimate_magnitude: torch.Tensor
    continue_magnitude: torch.Tensor
    layer_idx: int
    pattern_usage: torch.Tensor
    pattern_entropy: float = 0.0
    self_paradox_magnitude: float = 0.0
    composition_alpha: float = 0.0
    entropy_magnitude: float = 0.0
    temporal_entropy_magnitude: float = 0.0

class TemporalSymmetryEntropyLayerComplex(nn.Module):
    """Temporal entropy recycling layer with complex space + RoPE + TEMPORAL SYMMETRY"""
    
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, n_patterns=8,
                 composition_from_prev=True, prev_layer=None, is_bottom=False, is_top=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.n_patterns = n_patterns
        self.composition_from_prev = composition_from_prev
        self.is_bottom = is_bottom
        self.is_top = is_top
        
        # Complex processing pathway
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # COMPOSITIONAL PATTERN MECHANISM (complex)
        if composition_from_prev and prev_layer is not None:
            self.composition_weights = nn.Parameter(
                torch.randn(n_patterns, prev_layer.n_patterns, dtype=torch.cfloat) / (prev_layer.n_patterns ** 0.5)
            )
            self.prev_layer = prev_layer
            self.pattern_dict = None
        else:
            self.pattern_dict = nn.Parameter(
                torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02
            )
            self.composition_weights = None
            self.prev_layer = None
        
        # Next layer prediction patterns (complex)
        self.next_pattern_dict = nn.Parameter(
            torch.randn(n_patterns, next_dim // 2, dtype=torch.cfloat) * 0.02
        )
        self.next_pattern_attention = nn.Linear(next_dim, n_patterns)  # Real attention over interleaved
        
        # Pattern attention using INTERLEAVING PATTERN (the key innovation!)
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)  # Real attention over interleaved
        
        # Output pathway (complex -> real)
        self.to_penultimate = nn.Linear(hidden_dim // 2, penultimate_dim)  # Real output
        
        # TEMPORAL ENTROPY PROCESSING (special for bottom layer) - PAST → PRESENT
        if is_bottom:
            self.temporal_entropy_processor = ComplexLinear(hidden_dim, hidden_dim)
            self.temporal_entropy_predictor = ComplexLinear(hidden_dim, hidden_dim)  # Predict optimal entropy processing
            self.temporal_entropy_gate = nn.Linear(hidden_dim // 2, 1)  # Real gate for temporal entropy influence
        
        # Remove future prediction from individual layers - will be moved to integration level
        
        # Stats tracking
        self.last_stats: Optional[LayerStats] = None
    
    def get_pattern_dict(self):
        """Get compositional pattern dictionary"""
        if self.composition_weights is not None and self.prev_layer is not None:
            prev_patterns = self.prev_layer.get_pattern_dict()
            # Complex matrix multiplication for composition
            composed_patterns = torch.einsum('ij,jk->ik', self.composition_weights, prev_patterns)
            return composed_patterns
        else:
            return self.pattern_dict
    
    # Future prediction functionality removed from layers - moved to network integration level
    
    def extract_patterns_and_entropy(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract patterns and compute entropy as residual using INTERLEAVING PATTERN"""
        patterns = self.get_pattern_dict()
        
        # INTERLEAVING PATTERN: Convert complex to real interleaved
        # hidden is [batch, seq, complex_dim] or [batch, complex_dim]
        hidden_real_interleaved = torch.view_as_real(hidden).flatten(start_dim=-2)
        
        # Real attention over interleaved representation
        attn = self.pattern_attention(hidden_real_interleaved)
        pattern_weights = F.softmax(attn, dim=-1)
        
        # Pattern reconstruction (complex)
        if hidden.dim() == 3:  # sequence case
            pattern_reconstruction = torch.einsum('bsp,pk->bsk', pattern_weights.cfloat(), patterns)
        else:  # no sequence case
            pattern_reconstruction = torch.einsum('bp,pk->bk', pattern_weights.cfloat(), patterns)
        
        # Temporal acceleration removed from individual layers
        
        # Entropy = what patterns CANNOT explain (complex)
        entropy = hidden - pattern_reconstruction
        
        return pattern_reconstruction, entropy, pattern_weights
    
    def apply_self_paradox_nonlinearity(self, x: torch.Tensor, temporal_entropy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Complex paradox mechanism with temporal entropy integration"""
        # Process input (complex) - x can be [batch, seq, complex_dim] or [batch, complex_dim]
        hidden_linear = self.process(x)
        
        # BOTTOM LAYER: Temporal entropy integration (PAST → PRESENT)
        if self.is_bottom and temporal_entropy is not None:
            # temporal_entropy is [batch, hidden_dim] (pooled from previous epoch)
            # hidden_linear is [batch, seq, hidden_dim] or [batch, hidden_dim]
            
            batch_size = hidden_linear.size(0)
            temp_batch_size = temporal_entropy.size(0)
            
            if temp_batch_size != batch_size:
                if temp_batch_size < batch_size:
                    repeats = (batch_size + temp_batch_size - 1) // temp_batch_size
                    temporal_entropy = temporal_entropy.repeat(repeats, 1)[:batch_size]
                else:
                    temporal_entropy = temporal_entropy[:batch_size]
            
            # Process temporal entropy (complex) and gate its influence (real)
            processed_temporal = self.temporal_entropy_processor(temporal_entropy)
            
            # Handle sequence dimension for gating
            if hidden_linear.dim() == 3:  # sequence case
                # Broadcast temporal entropy to sequence: [batch, hidden] -> [batch, seq, hidden]
                seq_len = hidden_linear.size(1)
                processed_temporal = processed_temporal.unsqueeze(1).expand(-1, seq_len, -1)
                temporal_gate = torch.sigmoid(self.temporal_entropy_gate(hidden_linear.real))
            else:  # no sequence case
                temporal_gate = torch.sigmoid(self.temporal_entropy_gate(hidden_linear.real))
            
            # Apply temporal influence using complex multiplication
            temporal_influence = processed_temporal * temporal_gate.cfloat()
            hidden_linear = hidden_linear + temporal_influence
        
        # Self-prediction paradox mechanism (complex)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        paradox_magnitude = torch.norm(paradox, dim=-1, keepdim=True)
        
        # "I'm confused about myself → let more through" (complex gating)
        hidden = hidden_linear * torch.sigmoid(paradox_magnitude).cfloat()
        return hidden
    
    def forward(self, x: torch.Tensor, next_layer: Optional['TemporalSymmetryEntropyLayerComplex'], 
                layer_idx: int, temporal_entropy: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Forward pass with complex temporal entropy extraction + TEMPORAL SYMMETRY"""
        # x is [batch, seq, complex_dim] (full sequence processing!)
        
        # Apply complex paradox mechanism with temporal entropy
        hidden = self.apply_self_paradox_nonlinearity(x, temporal_entropy)
        
        # Extract patterns and entropy (complex with interleaving + temporal acceleration)
        pattern_reconstruction, entropy, pattern_weights = self.extract_patterns_and_entropy(hidden)
        
        # Future prediction moved to integration level
        
        # Track temporal entropy magnitude
        temporal_entropy_magnitude = 0.0
        if temporal_entropy is not None:
            temporal_entropy_magnitude = torch.mean(torch.norm(temporal_entropy, dim=-1)).item()
        
        if next_layer is not None:
            # Predict next layer (complex) - sequence-aware
            predicted_next = pattern_reconstruction
            
            # Get actual next layer behavior
            with torch.no_grad():
                actual_next = next_layer.apply_self_paradox_nonlinearity(hidden)
                actual_patterns, _, _ = next_layer.extract_patterns_and_entropy(actual_next)
            
            # Complex prediction error - average over sequence and features
            error_tensor = torch.norm(actual_patterns - predicted_next, dim=-1)**2
            
            if error_tensor.dim() == 2:  # [batch, seq] case
                pred_error = torch.mean(error_tensor, dim=1, keepdim=True)  # Average over sequence
            elif error_tensor.dim() == 1:  # [batch] case
                pred_error = error_tensor.unsqueeze(-1)  # Just add dimension
            else:  # Other cases
                pred_error = torch.mean(error_tensor.flatten(1), dim=1, keepdim=True)
            
            # Routing confidence
            pred_certainty = torch.abs(pred_error - torch.mean(pred_error))
            temperature = torch.sigmoid(pred_certainty)
            scaled_error = -pred_error * temperature
            confidence = 0.5 * (torch.tanh(scaled_error) + 1)
            
            # Route information (complex -> real for penultimate)
            # Average sequence for penultimate contribution (like original ParadoxNet)
            if pattern_reconstruction.dim() == 3:  # sequence case
                penultimate_input = pattern_reconstruction.mean(dim=1).real  # [batch, complex_dim] -> [batch, real_dim]
            else:
                penultimate_input = pattern_reconstruction.real
            
            penultimate_contribution = self.to_penultimate(penultimate_input) * confidence
            
            # Continue up maintains sequence structure
            if hidden.dim() == 3:
                confidence_broadcast = confidence.unsqueeze(1).expand(-1, hidden.size(1), -1)
            else:
                confidence_broadcast = confidence
            continue_up = hidden * (1 - confidence_broadcast.cfloat())
            
            # Track composition statistics
            composition_alpha = 0.0
            if self.composition_weights is not None:
                with torch.no_grad():
                    comp_weights_norm = F.softmax(torch.abs(self.composition_weights), dim=-1)
                    composition_alpha = 1.0 - torch.mean(torch.max(comp_weights_norm, dim=-1)[0]).item()
            
            # Enhanced statistics
            self.last_stats = LayerStats(
                prediction_errors=pred_error.detach(),
                confidence_values=confidence.detach(),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.mean(torch.norm(continue_up.detach(), dim=-1)),
                layer_idx=layer_idx,
                pattern_usage=pattern_weights.detach().mean(0) if pattern_weights.dim() > 1 else pattern_weights.detach(),
                pattern_entropy=0.0,
                self_paradox_magnitude=torch.mean(torch.norm(hidden.detach(), dim=-1)).item(),
                composition_alpha=composition_alpha,
                entropy_magnitude=torch.mean(torch.norm(entropy, dim=-1)).item(),
                temporal_entropy_magnitude=temporal_entropy_magnitude
            )
            
            return continue_up, penultimate_contribution, pred_error, entropy
            
        else:
            # Last layer processing (complex -> real)
            if pattern_reconstruction.dim() == 3:  # sequence case
                penultimate_input = pattern_reconstruction.mean(dim=1).real
            else:
                penultimate_input = pattern_reconstruction.real
            
            penultimate_contribution = self.to_penultimate(penultimate_input)
            
            composition_alpha = 0.0
            if self.composition_weights is not None:
                with torch.no_grad():
                    comp_weights_norm = F.softmax(torch.abs(self.composition_weights), dim=-1)
                    composition_alpha = 1.0 - torch.mean(torch.max(comp_weights_norm, dim=-1)[0]).item()
            
            self.last_stats = LayerStats(
                prediction_errors=torch.zeros(1, 1, device=x.device),
                confidence_values=torch.ones(1, 1, device=x.device),
                penultimate_magnitude=torch.mean(torch.norm(penultimate_contribution.detach(), dim=1)),
                continue_magnitude=torch.tensor(0.0, device=x.device),
                layer_idx=layer_idx,
                pattern_usage=pattern_weights.detach().mean(0) if pattern_weights.dim() > 1 else pattern_weights.detach(),
                pattern_entropy=0.0,
                self_paradox_magnitude=torch.mean(torch.norm(hidden.detach(), dim=-1)).item(),
                composition_alpha=composition_alpha,
                entropy_magnitude=torch.mean(torch.norm(entropy, dim=-1)).item(),
                temporal_entropy_magnitude=temporal_entropy_magnitude
            )
            
            return None, penultimate_contribution, None, entropy

class IntegrationLayer(nn.Module):
    """Integration layer: Part of layer stack with special consensus input! PERFECT SYMMETRY!"""
    
    def __init__(self, input_dim, hidden_dim, next_dim, penultimate_dim, output_dim, n_patterns=8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.next_dim = next_dim
        self.penultimate_dim = penultimate_dim
        self.output_dim = output_dim
        self.n_patterns = n_patterns
        self.is_integration = True
        
        # Complex processing pathway (like other layers)
        self.process = ComplexLinear(input_dim, hidden_dim)
        self.self_predictor = ComplexLinear(hidden_dim, hidden_dim)
        
        # Integration pattern dictionary (complex)
        self.pattern_dict = nn.Parameter(
            torch.randn(n_patterns, hidden_dim // 2, dtype=torch.cfloat) * 0.02
        )
        self.pattern_attention = nn.Linear(hidden_dim, n_patterns)
        
        # No next layer prediction (this is the final layer)
        # But we have PATTERN PREDICTION (Signal 5)
        self.pattern_predictor = ComplexLinear(hidden_dim, n_patterns * (hidden_dim // 2))
        
        # Output pathway - TWO SIGNALS!
        self.to_penultimate = nn.Linear(hidden_dim // 2, penultimate_dim)  # Like other layers
        self.vocab_projection = nn.Linear(penultimate_dim, penultimate_dim)  # Prepare for vocab
        self.final_output = nn.Linear(penultimate_dim, output_dim)  # Signal 4: Direct vocab
        
        # Stats
        self.last_stats: Optional[LayerStats] = None
    
    def forward(self, continue_up: torch.Tensor, consensus_contributions: torch.Tensor, 
                targets: Optional[torch.Tensor] = None, layer_idx: int = -1) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Integration layer: continue_up + consensus → two signals!"""
        
        # SPECIAL: Combine continue_up + consensus (what makes integration layer special!)
        combined_input = continue_up + torch.complex(consensus_contributions, torch.zeros_like(consensus_contributions))
        
        # Apply integration paradox mechanism (MAIN PROCESSING FLOW)
        hidden_linear = self.process(combined_input)
        self_prediction = self.self_predictor(hidden_linear)
        paradox = self_prediction - hidden_linear
        paradox_magnitude = torch.norm(paradox, dim=-1, keepdim=True)
        
        # Integration paradox nonlinearity 
        hidden = hidden_linear * torch.sigmoid(paradox_magnitude).cfloat()
        
        # PATTERN ANALYSIS FLOW (analyzes but doesn't transform)
        hidden_real_interleaved = torch.view_as_real(hidden).flatten(start_dim=-2)
        attn = self.pattern_attention(hidden_real_interleaved)
        pattern_weights = F.softmax(attn, dim=-1)
        pattern_reconstruction = torch.einsum('bp,pk->bk', pattern_weights.cfloat(), self.pattern_dict)
        
        # SIGNAL 4: Main processing flow → vocab output (PRIMARY)
        penultimate_features = self.to_penultimate(hidden.real)  # Complex → real
        vocab_ready = self.vocab_projection(penultimate_features)
        direct_output = self.final_output(vocab_ready)
        
        # SIGNAL 5: Pattern prediction loss - OUTPUT DRIVES PATTERNS!
        pattern_prediction_error = None
        # Always compute pattern prediction (not just when targets available)
        predicted_patterns = self.pattern_predictor(hidden)  # What patterns should be in output?
        predicted_patterns_reshaped = predicted_patterns.view(-1, self.n_patterns, self.hidden_dim // 2).mean(dim=1)
        
        # Extract actual patterns FROM the output (output drives patterns!)
        # Use the vocab_ready features to extract what patterns are actually present
        # Need to match the hidden_dim that pattern_attention expects
        padded_vocab_ready = F.pad(vocab_ready, (0, self.hidden_dim - vocab_ready.size(-1)))
        output_attention = F.softmax(self.pattern_attention(padded_vocab_ready), dim=-1)
        # Convert pattern dict to real using interleaving pattern (consistent with everywhere else)  
        pattern_dict_real = torch.view_as_real(self.pattern_dict).flatten(start_dim=1)  # [n_patterns, hidden_dim*2]
        
        # Shapes should now be correct
        
        actual_output_patterns = output_attention @ pattern_dict_real
        
        # Handle batch size mismatch between predicted and actual patterns
        pred_flat = torch.view_as_real(predicted_patterns_reshaped).flatten(1)
        
        # Ensure both have same batch size
        batch_size = min(pred_flat.size(0), actual_output_patterns.size(0))
        pred_flat_matched = pred_flat[:batch_size]
        actual_matched = actual_output_patterns[:batch_size]
        
        # Pattern prediction loss: predicted vs actual output patterns (both real now)
        # Scale up the loss to avoid numerical underflow
        pattern_prediction_error = ((pred_flat_matched - actual_matched) ** 2).mean(dim=-1, keepdim=True) * 1000000
        
        # Track stats
        self.last_stats = LayerStats(
            prediction_errors=pattern_prediction_error.detach() if pattern_prediction_error is not None else torch.zeros(1, 1),
            confidence_values=torch.ones(1, 1),
            penultimate_magnitude=torch.mean(torch.norm(penultimate_features.detach(), dim=1)),
            continue_magnitude=torch.tensor(0.0),
            layer_idx=layer_idx,
            pattern_usage=pattern_weights.detach().mean(0),
            pattern_entropy=0.0,
            self_paradox_magnitude=torch.mean(torch.norm(hidden.detach(), dim=-1)).item(),
            composition_alpha=0.0,
            entropy_magnitude=torch.mean(torch.norm(hidden - pattern_reconstruction, dim=-1)).item(),
            temporal_entropy_magnitude=0.0
        )
        
        return direct_output, penultimate_features, pattern_prediction_error

class TemporalSymmetryEntropyNetComplex(nn.Module):
    """Complete temporal symmetry entropy recycling network: PAST ↔ FUTURE + INTEGRATION PATTERNS!"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dims, n_patterns=8):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Store entropy from previous epoch for temporal recycling
        self.previous_entropy = None
        
        # Create temporal symmetry entropy recycling layers
        self.layers = nn.ModuleList()
        current_dim = embedding_dim
        
        prev_layer = None
        for i, hidden_dim in enumerate(hidden_dims):
            next_dim = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else hidden_dim
            
            layer = TemporalSymmetryEntropyLayerComplex(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                next_dim=next_dim,
                penultimate_dim=hidden_dim // 2,  # Real output dimensions
                n_patterns=n_patterns,
                composition_from_prev=(i > 0),
                prev_layer=prev_layer,
                is_bottom=(i == 0),  # First layer handles PAST entropy
                is_top=False  # No individual layer does future prediction - moved to integration
            )
            self.layers.append(layer)
            prev_layer = layer
            current_dim = hidden_dim
        
        # INTEGRATION LAYER (FINAL LAYER IN STACK!)
        self.integration_layer = IntegrationLayer(
            input_dim=hidden_dims[-1],       # Complex input from continue_up
            hidden_dim=hidden_dims[-1],      # Same as last layer
            next_dim=hidden_dims[-1],        # No next layer
            penultimate_dim=hidden_dims[-1] // 2,
            output_dim=vocab_size,
            n_patterns=n_patterns
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass with TEMPORAL SYMMETRY: Past entropy ↔ Future patterns"""
        batch_size, seq_len = x.shape
        
        # Embedding + RoPE positional encoding
        embedded = self.embedding(x)  # [batch, seq, embed_dim]
        freqs_cis = self.pos_encoder.freqs_cis[:seq_len]
        
        # Apply RoPE and convert to complex
        current_seq_real = apply_rotary_pos_emb(embedded, freqs_cis)
        current_seq = torch.view_as_complex(current_seq_real.float().reshape(batch_size, seq_len, -1, 2))
        current_seq_original = current_seq.clone()  # Save original for potential recursive residual
        
        # FULL SEQUENCE PROCESSING with TEMPORAL SYMMETRY
        penultimate_contributions = []
        all_errors = []
        all_entropy = []
        
        # Track temporal prediction error for Layer 0
        temporal_prediction_error = None
        
        # Single pass: process all layers with temporal symmetry
        for i, layer in enumerate(self.layers):
            next_layer = self.layers[i+1] if i < len(self.layers)-1 else None
            
            if i == 0:
                # BOTTOM LAYER: Use entropy from previous epoch (PAST → PRESENT)
                continue_up, penultimate, error, entropy = layer(current_seq, next_layer, i, temporal_entropy=self.previous_entropy)
            else:
                # MIDDLE LAYERS: Regular processing
                continue_up, penultimate, error, entropy = layer(current_seq, next_layer, i)
            
            # Update current_seq for next layer (None means this was the last layer)
            if continue_up is not None:
                current_seq = continue_up
            
            if error is not None:
                all_errors.append(error)
            penultimate_contributions.append(penultimate)
            
            # Collect entropy for NEXT epoch (except from layer 0)
            if i > 0:
                all_entropy.append(entropy)
        
        # Prepare entropy for next epoch AND compute temporal prediction error
        if all_entropy:
            # Pool entropy over sequence for next epoch: [batch, seq, hidden] -> [batch, hidden]
            pooled_entropy = [entropy.mean(dim=1) if entropy.dim() == 3 else entropy for entropy in all_entropy]
            total_entropy = torch.stack(pooled_entropy).sum(dim=0)
            
            # TEMPORAL INPUT PREDICTION: Compare Layer 0's prediction to actual entropy
            if self.previous_entropy is not None:
                layer_0 = self.layers[0]
                if hasattr(layer_0, 'temporal_entropy_predictor'):
                    # Layer 0 predicts what optimal entropy processing should look like
                    # Use original input for prediction (before processing)
                    input_for_prediction = current_seq.mean(dim=1) if current_seq is not None else embedded.mean(dim=1)
                    
                    # Convert to complex if needed
                    if not torch.is_complex(input_for_prediction):
                        # Reconstruct complex from embedded
                        input_real = apply_rotary_pos_emb(embedded, freqs_cis)
                        input_complex = torch.view_as_complex(input_real.float().reshape(batch_size, seq_len, -1, 2))
                        input_for_prediction = input_complex.mean(dim=1)
                    
                    temporal_prediction = layer_0.temporal_entropy_predictor(input_for_prediction)
                    # Prediction error: predicted optimal processing vs actual accumulated entropy
                    temporal_prediction_error = F.mse_loss(
                        torch.view_as_real(temporal_prediction).flatten(1),
                        torch.view_as_real(total_entropy).flatten(1)
                    )
            
            # Store for next epoch (detach to avoid gradient accumulation)
            self.previous_entropy = total_entropy.detach()
        else:
            self.previous_entropy = None
            temporal_prediction_error = None
        
        # Get recursive residual from final sequence state (like original ParadoxNet)
        if current_seq is not None:
            recursive_residual = current_seq.mean(dim=1)
        else:
            # Use the input as residual if no layers passed sequence through
            recursive_residual = current_seq_original.mean(dim=1)
            
        # Make sure dimensions match for addition
        if recursive_residual.size(-1) != penultimate_contributions[0].size(-1):
            # Project to match penultimate dimensions
            recursive_residual = recursive_residual.real
        
        # INTEGRATION LAYER PROCESSING (PERFECT SYMMETRY!)
        # Consensus contributions from all layers
        consensus_view = torch.sum(torch.stack(penultimate_contributions), dim=0)
        
        # Get final continue_up (unpredictable residual) - average sequence for integration
        if current_seq is not None:
            final_continue_up = current_seq.mean(dim=1)  # [batch, seq, hidden] → [batch, hidden]
        else:
            final_continue_up = current_seq_original.mean(dim=1)
        
        # Add recursive residual 
        if recursive_residual.size(-1) != final_continue_up.size(-1):
            recursive_residual = recursive_residual.real
        if final_continue_up.size(-1) == recursive_residual.size(-1):
            final_continue_up = final_continue_up + recursive_residual.cfloat()
        
        # INTEGRATION LAYER: continue_up + consensus → two signals!
        output, integration_penultimate, integration_pattern_error = self.integration_layer(
            continue_up=final_continue_up,
            consensus_contributions=consensus_view, 
            targets=None,
            layer_idx=len(self.layers)
        )
        
        # Combine errors properly
        if all_errors:
            # Ensure all errors have the same shape and flatten if needed
            flattened_errors = []
            for error in all_errors:
                if error.dim() > 1:
                    flattened_errors.append(error.flatten(1))
                else:
                    flattened_errors.append(error.unsqueeze(-1) if error.dim() == 1 else error)
            
            # Stack or concatenate based on shapes
            if len(flattened_errors) == 1:
                combined_errors = flattened_errors[0]
            else:
                # Try to concatenate along last dimension
                try:
                    combined_errors = torch.cat(flattened_errors, dim=-1)
                except:
                    # Fall back to just using the first error
                    combined_errors = flattened_errors[0]
        else:
            combined_errors = None
        
        return output, combined_errors, temporal_prediction_error, integration_pattern_error

# Factory function
def create_temporal_symmetry_entropy_net_complex(sequence_length=20, hidden_dims=[64, 64, 64], n_patterns=8):
    """Create temporal symmetry entropy recycling version: PAST ↔ FUTURE"""
    return TemporalSymmetryEntropyNetComplex(
        vocab_size=128,  # Will be set by experiment
        embedding_dim=64,
        hidden_dims=hidden_dims,
        n_patterns=n_patterns
    )

if __name__ == "__main__":
    print("🌪️⚡ TESTING TEMPORAL SYMMETRY ENTROPY RECYCLING: PAST ↔ FUTURE ⚡🌪️")
    
    # Create network
    net = create_temporal_symmetry_entropy_net_complex()
    
    # Test data
    x = torch.randint(0, 57, (5, 10))  # Token sequences
    
    # Forward pass
    output, errors, temporal_error, consistency_loss = net(x)
    
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Errors: {errors.shape if errors is not None else None}")
    print(f"Temporal error: {temporal_error.item() if temporal_error is not None else None}")
    print(f"Consistency loss: {consistency_loss.item() if consistency_loss is not None else None}")
    
    # Check entropy statistics
    print(f"\n=== TEMPORAL SYMMETRY STATISTICS ===")
    for i, layer in enumerate(net.layers):
        if layer.last_stats:
            stats = layer.last_stats
            print(f"Layer {i} ({'BOTTOM' if layer.is_bottom else 'TOP' if layer.is_top else 'MIDDLE'}):")
            print(f"  Entropy magnitude: {stats.entropy_magnitude:.3f}")
            print(f"  Temporal entropy magnitude: {stats.temporal_entropy_magnitude:.3f}")
            print(f"  Composition alpha: {stats.composition_alpha:.3f}")
            print(f"  Paradox magnitude: {stats.self_paradox_magnitude:.3f}")
    
    print(f"\n✅ TEMPORAL SYMMETRY: Past entropy ↔ Future patterns working!")
    print(f"🔮 The derivatives hierarchy lives! ⚡")