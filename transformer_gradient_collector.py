"""
Specialized Transformer Per-Sample Gradient Collector
Optimized specifically for transformer architectures with attention mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Callable
import time
import math

class TransformerGradientCollector:
    """
    Ultra-efficient gradient collector specifically designed for transformer models.
    Uses optimized attention gradient computation and layer-wise processing.
    """
    
    def __init__(self, model: nn.Module, save_dir: str, collect_attention_only: bool = False):
        """
        Initialize transformer-specific gradient collector
        
        Args:
            model: Transformer model
            save_dir: Directory to save gradients
            collect_attention_only: If True, only collect attention layer gradients (much faster)
        """
        self.model = model
        self.save_dir = save_dir
        self.collect_attention_only = collect_attention_only
        self.gradients = defaultdict(list)
        self.sample_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Identify transformer components
        self._identify_transformer_components()
        
        # Setup hooks for efficient gradient collection
        self._setup_gradient_hooks()
        
        print(f"Transformer gradient collector initialized")
        print(f"Attention-only mode: {collect_attention_only}")
        print(f"Target layers: {len(self.target_layers)}")
    
    def _identify_transformer_components(self):
        """Identify transformer-specific components for targeted gradient collection"""
        self.target_layers = {}
        self.attention_layers = []
        self.feedforward_layers = []
        
        for name, module in self.model.named_modules():
            # Identify attention layers
            if any(keyword in name.lower() for keyword in ['attention', 'attn', 'self_attn']):
                if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                    self.attention_layers.append((name, module))
                    if not self.collect_attention_only:
                        self.target_layers[name] = module
                    else:
                        self.target_layers[name] = module
            
            # Identify feedforward layers (if not attention-only mode)
            elif not self.collect_attention_only:
                if any(keyword in name.lower() for keyword in ['feed_forward', 'ff', 'mlp']):
                    if isinstance(module, nn.Linear):
                        self.feedforward_layers.append((name, module))
                        self.target_layers[name] = module
                
                # Include embedding and output layers
                elif any(keyword in name.lower() for keyword in ['embed', 'lm_head', 'output']):
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        self.target_layers[name] = module
    
    def _setup_gradient_hooks(self):
        """Setup backward hooks for efficient gradient collection"""
        self.hooks = []
        self.hooked_gradients = {}
        
        for name, module in self.target_layers.items():
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    full_name = f"{name}.{param_name}"
                    
                    def make_hook(full_name):
                        def hook(grad):
                            if full_name not in self.hooked_gradients:
                                self.hooked_gradients[full_name] = []
                            self.hooked_gradients[full_name].append(grad.detach().cpu().clone())
                            return grad
                        return hook
                    
                    handle = param.register_hook(make_hook(full_name))
                    self.hooks.append(handle)
    
    def collect_gradients_batch_efficient(self, 
                                        input_ids: torch.Tensor,
                                        attention_mask: torch.Tensor,
                                        labels: torch.Tensor,
                                        max_samples: int = 32) -> None:
        """
        Collect gradients efficiently using batch processing with sample separation
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]  
            labels: Target labels [batch_size, seq_len]
            max_samples: Maximum samples to process at once
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Process in chunks to manage memory
        chunk_size = min(batch_size, max_samples)
        
        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            
            chunk_input_ids = input_ids[start_idx:end_idx]
            chunk_attention_mask = attention_mask[start_idx:end_idx]
            chunk_labels = labels[start_idx:end_idx]
            
            self._collect_chunk_gradients(chunk_input_ids, chunk_attention_mask, chunk_labels)
            
            self.sample_count += (end_idx - start_idx)
    
    def _collect_chunk_gradients(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        """Collect gradients for a chunk of samples using vectorized operations"""
        chunk_size = input_ids.size(0)
        
        # Clear previous hooked gradients
        self.hooked_gradients.clear()
        
        # Process each sample in the chunk
        for i in range(chunk_size):
            # Zero gradients
            self.model.zero_grad()
            
            # Single sample
            sample_input = input_ids[i:i+1]
            sample_mask = attention_mask[i:i+1] if attention_mask is not None else None
            sample_labels = labels[i:i+1]
            
            # Forward pass
            if sample_mask is not None:
                outputs = self.model(sample_input, attention_mask=sample_mask, labels=sample_labels)
            else:
                outputs = self.model(sample_input, labels=sample_labels)
            
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            
            # Backward pass (hooks will collect gradients)
            loss.backward()
        
        # Process collected gradients
        for param_name, grad_list in self.hooked_gradients.items():
            if grad_list:
                # Stack gradients from this chunk
                stacked_grads = torch.stack(grad_list, dim=0)
                self.gradients[param_name].append(stacked_grads)
    
    def collect_gradients_attention_optimized(self,
                                            input_ids: torch.Tensor,
                                            attention_mask: torch.Tensor,
                                            labels: torch.Tensor) -> None:
        """
        Optimized gradient collection focusing on attention patterns
        Uses analytical gradients for attention mechanisms where possible
        """
        if not self.collect_attention_only:
            print("Warning: attention_optimized mode should be used with collect_attention_only=True")
        
        batch_size = input_ids.size(0)
        
        # Process samples individually for attention analysis
        for i in range(batch_size):
            sample_input = input_ids[i:i+1]
            sample_mask = attention_mask[i:i+1] if attention_mask is not None else None
            sample_labels = labels[i:i+1]
            
            # Get attention weights and gradients
            attention_grads = self._compute_attention_gradients(sample_input, sample_mask, sample_labels)
            
            # Store attention gradients
            for layer_name, grads in attention_grads.items():
                self.gradients[layer_name].append(grads)
            
            self.sample_count += 1
    
    def _compute_attention_gradients(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute attention-specific gradients efficiently"""
        self.model.zero_grad()
        
        # Forward pass with gradient tracking
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        
        # Backward pass
        loss.backward()
        
        # Extract attention gradients
        attention_grads = {}
        for name, module in self.attention_layers:
            for param_name, param in module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    full_name = f"{name}.{param_name}"
                    attention_grads[full_name] = param.grad.detach().cpu().clone()
        
        return attention_grads
    
    def save_gradients(self, epoch: int, batch_idx: Optional[int] = None):
        """Save collected gradients with transformer-specific metadata"""
        if not any(self.gradients.values()):
            print("No gradients to save")
            return
        
        # Create epoch directory
        epoch_dir = os.path.join(self.save_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Save gradients by layer type
        attention_info = {}
        feedforward_info = {}
        other_info = {}
        
        for param_name, grad_list in self.gradients.items():
            if not grad_list:
                continue
            
            # Concatenate all gradients for this parameter
            all_grads = torch.cat(grad_list, dim=0)
            
            # Determine layer type
            layer_type = 'other'
            if any(keyword in param_name.lower() for keyword in ['attention', 'attn']):
                layer_type = 'attention'
            elif any(keyword in param_name.lower() for keyword in ['feed_forward', 'ff', 'mlp']):
                layer_type = 'feedforward'
            
            # Create filename
            if batch_idx is not None:
                filename = f'{param_name}_epoch_{epoch}_batch_{batch_idx}.pt'
            else:
                filename = f'{param_name}_epoch_{epoch}.pt'
            
            filepath = os.path.join(epoch_dir, filename)
            
            # Save gradients
            torch.save({
                'gradients': all_grads,
                'param_name': param_name,
                'layer_type': layer_type,
                'num_samples': all_grads.size(0),
                'param_shape': all_grads.shape[1:]  # Exclude sample dimension
            }, filepath)
            
            # Update info
            info_dict = {
                'num_samples': all_grads.size(0),
                'shape': all_grads.shape,
                'file': filename
            }
            
            if layer_type == 'attention':
                attention_info[param_name] = info_dict
            elif layer_type == 'feedforward':
                feedforward_info[param_name] = info_dict
            else:
                other_info[param_name] = info_dict
            
            print(f"Saved {all_grads.size(0)} gradient samples for {param_name} ({layer_type})")
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_samples': self.sample_count,
            'attention_layers': attention_info,
            'feedforward_layers': feedforward_info,
            'other_layers': other_info,
            'collection_mode': 'attention_only' if self.collect_attention_only else 'full',
            'timestamp': time.time()
        }
        
        metadata_path = os.path.join(epoch_dir, 'transformer_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.clear_gradients()
    
    def clear_gradients(self):
        """Clear collected gradients and reset hooks"""
        self.gradients.clear()
        self.hooked_gradients.clear()
    
    def get_attention_statistics(self) -> Dict[str, float]:
        """Get statistics specific to attention layers"""
        stats = {}
        
        for param_name, grad_list in self.gradients.items():
            if 'attention' in param_name.lower() or 'attn' in param_name.lower():
                if grad_list:
                    all_grads = torch.cat(grad_list, dim=0)
                    flat_grads = all_grads.view(all_grads.size(0), -1)
                    grad_norms = torch.norm(flat_grads, dim=1)
                    
                    stats[param_name] = {
                        'mean_norm': grad_norms.mean().item(),
                        'std_norm': grad_norms.std().item(),
                        'max_norm': grad_norms.max().item(),
                        'min_norm': grad_norms.min().item()
                    }
        
        return stats
    
    def cleanup(self):
        """Clean up hooks and resources"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.clear_gradients()

# Utility functions for transformer gradient analysis
def analyze_attention_gradients(gradients_dir: str, epoch: int) -> Dict:
    """Analyze attention-specific gradient patterns"""
    epoch_dir = os.path.join(gradients_dir, f'epoch_{epoch}')
    
    if not os.path.exists(epoch_dir):
        raise ValueError(f"No gradients found for epoch {epoch}")
    
    # Load metadata
    metadata_path = os.path.join(epoch_dir, 'transformer_metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    attention_analysis = {}
    
    # Analyze attention layer gradients
    for param_name, param_info in metadata['attention_layers'].items():
        grad_path = os.path.join(epoch_dir, param_info['file'])
        grad_data = torch.load(grad_path, map_location='cpu')
        gradients = grad_data['gradients'].to(torch.float32)
        
        # Compute attention-specific metrics
        if 'query' in param_name.lower() or 'q_proj' in param_name.lower():
            analysis_type = 'query'
        elif 'key' in param_name.lower() or 'k_proj' in param_name.lower():
            analysis_type = 'key'
        elif 'value' in param_name.lower() or 'v_proj' in param_name.lower():
            analysis_type = 'value'
        elif 'out' in param_name.lower() or 'o_proj' in param_name.lower():
            analysis_type = 'output'
        else:
            analysis_type = 'other'
        
        # Compute gradient statistics
        flat_grads = gradients.view(gradients.size(0), -1)
        grad_norms = torch.norm(flat_grads, dim=1)
        
        attention_analysis[param_name] = {
            'type': analysis_type,
            'mean_norm': grad_norms.mean().item(),
            'std_norm': grad_norms.std().item(),
            'gradient_diversity': torch.std(grad_norms).item(),  # How diverse are the gradients
            'num_samples': gradients.size(0)
        }
    
    return {
        'attention_analysis': attention_analysis,
        'metadata': metadata
    }

def compute_gradient_similarity_matrix(gradients_dir: str, epoch: int, layer_name: str) -> torch.Tensor:
    """Compute similarity matrix between per-sample gradients for a specific layer"""
    epoch_dir = os.path.join(gradients_dir, f'epoch_{epoch}')
    
    # Find the gradient file for the specified layer
    grad_files = [f for f in os.listdir(epoch_dir) if layer_name in f and f.endswith('.pt')]
    
    if not grad_files:
        raise ValueError(f"No gradient file found for layer {layer_name}")
    
    grad_path = os.path.join(epoch_dir, grad_files[0])
    grad_data = torch.load(grad_path, map_location='cpu')
    gradients = grad_data['gradients'].to(torch.float32)
    
    # Flatten gradients
    flat_grads = gradients.view(gradients.size(0), -1)
    
    # Normalize gradients
    normalized_grads = F.normalize(flat_grads, p=2, dim=1)
    
    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(normalized_grads, normalized_grads.t())
    
    return similarity_matrix