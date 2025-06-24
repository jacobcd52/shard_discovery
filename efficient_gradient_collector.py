"""
Highly Efficient Per-Sample Gradient Collector
Uses vectorized operations and torch.func for massive speedup over sample-by-sample processing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from collections import OrderedDict, defaultdict
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import time

try:
    from torch.func import functional_call, vmap, grad
    FUNCTORCH_AVAILABLE = True
except ImportError:
    FUNCTORCH_AVAILABLE = False
    print("Warning: torch.func not available. Falling back to manual vectorization.")

class EfficientPerSampleGradientCollector:
    """
    Highly efficient per-sample gradient collector using vectorized operations.
    Can be 10-100x faster than the naive approach.
    """
    
    def __init__(self, model: nn.Module, save_dir: str, max_samples_per_collection: int = 64):
        """
        Initialize efficient gradient collector
        
        Args:
            model: PyTorch model
            save_dir: Directory to save gradients
            max_samples_per_collection: Maximum samples to process at once (memory vs speed tradeoff)
        """
        self.model = model
        self.save_dir = save_dir
        self.max_samples_per_collection = max_samples_per_collection
        self.gradients = defaultdict(list)
        self.sample_count = 0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Store parameter info for efficient access
        self.param_names = []
        self.param_shapes = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                self.param_shapes[name] = param.shape
        
        # Choose implementation based on availability
        self.use_functorch = FUNCTORCH_AVAILABLE and self._is_compatible_model()
        
        if self.use_functorch:
            print("Using torch.func for maximum efficiency")
            self._setup_functorch()
        else:
            print("Using manual vectorization")
            self._setup_manual_vectorization()
    
    def _is_compatible_model(self) -> bool:
        """Check if model is compatible with functorch approach"""
        # Most standard layers are compatible, but some custom layers might not be
        incompatible_types = (nn.DataParallel, nn.parallel.DistributedDataParallel)
        return not isinstance(self.model, incompatible_types)
    
    def _setup_functorch(self):
        """Setup functorch-based gradient computation"""
        # Create functional version of model
        self.func_model = lambda params, x, *args, **kwargs: functional_call(self.model, params, (x,) + args, kwargs)
        
        # Get parameters as a dict
        self.params = dict(self.model.named_parameters())
    
    def _setup_manual_vectorization(self):
        """Setup manual vectorization for gradient computation"""
        # We'll use gradient accumulation with per-sample processing
        # but optimized for batch operations where possible
        pass
    
    def collect_gradients_for_batch(self, 
                                  data: torch.Tensor, 
                                  target: torch.Tensor, 
                                  criterion: Callable,
                                  sample_indices: Optional[List[int]] = None) -> None:
        """
        Collect per-sample gradients for a batch efficiently
        
        Args:
            data: Input data [batch_size, ...]
            target: Target labels [batch_size, ...]
            criterion: Loss function that takes (model_output, target) and returns scalar loss
            sample_indices: Optional indices for tracking sample order
        """
        batch_size = data.size(0)
        
        # Store original training mode
        original_training_mode = self.model.training
        
        # For gradient collection, we want to be in eval mode to avoid dropout randomness
        # This makes the gradients more deterministic and compatible with vmap
        self.model.eval()
        
        try:
            # Process in chunks if batch is too large
            chunk_size = min(batch_size, self.max_samples_per_collection)
            
            for start_idx in range(0, batch_size, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size)
                chunk_data = data[start_idx:end_idx]
                chunk_target = target[start_idx:end_idx]
                
                if self.use_functorch:
                    self._collect_gradients_functorch(chunk_data, chunk_target, criterion)
                else:
                    self._collect_gradients_manual(chunk_data, chunk_target, criterion)
                
                self.sample_count += (end_idx - start_idx)
        finally:
            # Restore original training mode
            self.model.train(original_training_mode)
    
    def _collect_gradients_functorch(self, data: torch.Tensor, target: torch.Tensor, criterion: Callable):
        """Collect gradients using functorch for maximum efficiency"""
        try:
            def compute_loss_for_sample(params, sample_data, sample_target):
                # Forward pass for single sample
                if sample_data.dim() == data.dim() - 1:
                    sample_data = sample_data.unsqueeze(0)
                if sample_target.dim() == target.dim() - 1:
                    sample_target = sample_target.unsqueeze(0)
                
                output = functional_call(self.model, params, (sample_data,))
                return criterion(output, sample_target)
            
            # Compute per-sample gradients using vmap
            # Use 'different' randomness to ensure dropout works correctly for each sample
            try:
                per_sample_grads = vmap(
                    grad(compute_loss_for_sample), 
                    in_dims=(None, 0, 0),
                    randomness='different'  # Each sample gets different dropout patterns
                )(self.params, data, target)
            except TypeError:
                # Fallback for older PyTorch versions without randomness parameter
                print("PyTorch version doesn't support randomness parameter, trying without...")
                per_sample_grads = vmap(
                    grad(compute_loss_for_sample), 
                    in_dims=(None, 0, 0)
                )(self.params, data, target)
            
            # Store gradients
            for name in self.param_names:
                if name in per_sample_grads:
                    grad_tensor = per_sample_grads[name].detach().cpu().to(torch.float32)
                    self.gradients[name].append(grad_tensor)
                    
        except Exception as e:
            print(f"FuncTorch method failed: {e}")
            print("Falling back to manual vectorization...")
            # Fall back to manual method if functorch fails
            self._collect_gradients_manual(data, target, criterion)
    
    def _collect_gradients_manual(self, data: torch.Tensor, target: torch.Tensor, criterion: Callable):
        """Collect gradients using manual vectorization (fallback method)"""
        batch_size = data.size(0)
        device = data.device
        
        # Store original requires_grad state
        original_requires_grad = {}
        for name, param in self.model.named_parameters():
            original_requires_grad[name] = param.requires_grad
        
        # Initialize storage for this batch
        batch_gradients = {name: [] for name in self.param_names}
        
        # Process each sample (optimized loop)
        for i in range(batch_size):
            # Zero gradients
            self.model.zero_grad()
            
            # Single sample forward pass
            sample_data = data[i:i+1]
            sample_target = target[i:i+1]
            
            # Forward pass
            output = self.model(sample_data)
            loss = criterion(output, sample_target)
            
            # Backward pass
            loss.backward()
            
            # Collect gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad.detach().cpu().to(torch.float32).clone()
                    batch_gradients[name].append(grad)
        
        # Stack gradients for this batch
        for name in self.param_names:
            if batch_gradients[name]:
                stacked_grads = torch.stack(batch_gradients[name], dim=0)
                self.gradients[name].append(stacked_grads)
    
    def collect_gradients_memory_efficient(self, 
                                         data: torch.Tensor, 
                                         target: torch.Tensor, 
                                         criterion: Callable,
                                         max_samples: int = 16) -> None:
        """
        Memory-efficient gradient collection for large batches
        """
        batch_size = data.size(0)
        
        # Process in small chunks to avoid OOM
        for start_idx in range(0, min(batch_size, max_samples)):
            sample_data = data[start_idx:start_idx+1]
            sample_target = target[start_idx:start_idx+1]
            
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(sample_data)
            loss = criterion(output, sample_target)
            
            # Backward pass
            loss.backward()
            
            # Collect and immediately save gradients to avoid memory buildup
            sample_gradients = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    sample_gradients[name] = param.grad.detach().cpu().to(torch.float32).clone()
            
            # Save immediately (streaming approach)
            self._save_single_sample_gradients(sample_gradients, self.sample_count + start_idx)
    
    def _save_single_sample_gradients(self, gradients: Dict[str, torch.Tensor], sample_idx: int):
        """Save gradients for a single sample immediately"""
        # Create subdirectory for streaming saves
        stream_dir = os.path.join(self.save_dir, 'streaming')
        os.makedirs(stream_dir, exist_ok=True)
        
        # Save each parameter's gradient
        for name, grad in gradients.items():
            filename = f'{name}_sample_{sample_idx}.pt'
            filepath = os.path.join(stream_dir, filename)
            torch.save(grad, filepath)
    
    def save_gradients(self, epoch: int, batch_idx: Optional[int] = None, clear_after_save: bool = True):
        """
        Save collected gradients to disk
        
        Args:
            epoch: Current epoch number
            batch_idx: Optional batch index
            clear_after_save: Whether to clear gradients after saving (recommended)
        """
        if not any(self.gradients.values()):
            print("No gradients to save")
            return
        
        # Create epoch directory
        epoch_dir = os.path.join(self.save_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Save gradients for each parameter
        saved_info = {}
        for name in self.param_names:
            if name in self.gradients and self.gradients[name]:
                # Concatenate all collected gradients for this parameter
                all_grads = torch.cat(self.gradients[name], dim=0)
                
                # Create filename
                if batch_idx is not None:
                    filename = f'{name}_epoch_{epoch}_batch_{batch_idx}.pt'
                else:
                    filename = f'{name}_epoch_{epoch}.pt'
                
                filepath = os.path.join(epoch_dir, filename)
                
                # Save gradients with compression
                torch.save({
                    'gradients': all_grads,
                    'param_shape': self.param_shapes[name],
                    'num_samples': all_grads.size(0)
                }, filepath)
                
                saved_info[name] = {
                    'num_samples': all_grads.size(0),
                    'shape': all_grads.shape,
                    'file': filename
                }
                
                print(f"Saved {all_grads.size(0)} gradient samples for {name}")
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_samples': self.sample_count,
            'parameter_info': saved_info,
            'collection_method': 'functorch' if self.use_functorch else 'manual',
            'timestamp': time.time()
        }
        
        metadata_path = os.path.join(epoch_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        if clear_after_save:
            self.clear_gradients()
    
    def clear_gradients(self):
        """Clear collected gradients to free memory"""
        self.gradients.clear()
        # Don't reset sample_count as it tracks total across collections
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage of stored gradients in MB"""
        usage = {}
        total_mb = 0
        
        for name, grad_list in self.gradients.items():
            if grad_list:
                # Calculate memory usage
                total_elements = sum(g.numel() for g in grad_list)
                mb = total_elements * 4 / (1024 * 1024)  # Assuming float32
                usage[name] = mb
                total_mb += mb
        
        usage['total'] = total_mb
        return usage

# Utility functions for gradient analysis
def load_and_analyze_gradients(gradients_dir: str, epoch: int, analysis_type: str = 'basic') -> Dict:
    """
    Load and analyze collected gradients
    
    Args:
        gradients_dir: Directory containing saved gradients
        epoch: Epoch number to analyze
        analysis_type: Type of analysis ('basic', 'clustering', 'similarity')
    """
    epoch_dir = os.path.join(gradients_dir, f'epoch_{epoch}')
    
    if not os.path.exists(epoch_dir):
        raise ValueError(f"No gradients found for epoch {epoch}")
    
    # Load metadata
    metadata_path = os.path.join(epoch_dir, 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    results = {'metadata': metadata}
    
    if analysis_type == 'basic':
        results.update(_analyze_gradient_statistics(epoch_dir, metadata))
    elif analysis_type == 'clustering':
        results.update(_analyze_gradient_clustering(epoch_dir, metadata))
    elif analysis_type == 'similarity':
        results.update(_analyze_gradient_similarity(epoch_dir, metadata))
    
    return results

def _analyze_gradient_statistics(epoch_dir: str, metadata: Dict) -> Dict:
    """Compute basic gradient statistics"""
    stats = {}
    
    for param_name, param_info in metadata['parameter_info'].items():
        # Load gradients
        grad_path = os.path.join(epoch_dir, param_info['file'])
        grad_data = torch.load(grad_path, map_location='cpu')
        gradients = grad_data['gradients'].to(torch.float32)
        
        # Flatten gradients for analysis
        flat_grads = gradients.view(gradients.size(0), -1)
        
        # Compute statistics
        grad_norms = torch.norm(flat_grads, dim=1)
        
        stats[param_name] = {
            'mean_norm': grad_norms.mean().item(),
            'std_norm': grad_norms.std().item(),
            'min_norm': grad_norms.min().item(),
            'max_norm': grad_norms.max().item(),
            'median_norm': grad_norms.median().item(),
            'num_samples': gradients.size(0),
            'param_size': gradients.numel() // gradients.size(0)
        }
    
    return {'statistics': stats}

def _analyze_gradient_clustering(epoch_dir: str, metadata: Dict) -> Dict:
    """Analyze gradient clustering patterns"""
    # This would implement more advanced clustering analysis
    # For now, return placeholder
    return {'clustering': 'Not implemented'}

def _analyze_gradient_similarity(epoch_dir: str, metadata: Dict) -> Dict:
    """Analyze gradient similarity patterns"""
    # This would implement similarity analysis
    # For now, return placeholder
    return {'similarity': 'Not implemented'}

# Benchmark utility
def benchmark_gradient_collectors(model, data_batch, target_batch, criterion, num_runs=5):
    """
    Benchmark different gradient collection methods
    """
    print("Benchmarking gradient collectors...")
    
    # Original collector
    from gradient_collector import PerSampleGradientCollector
    original_collector = PerSampleGradientCollector(model, 'tmp_benchmark_original')
    
    # New efficient collector
    efficient_collector = EfficientPerSampleGradientCollector(model, 'tmp_benchmark_efficient')
    
    # Benchmark original
    start_time = time.time()
    for _ in range(num_runs):
        original_collector.collect_gradients_for_batch(data_batch, target_batch, criterion)
        original_collector.clear_gradients()
    original_time = (time.time() - start_time) / num_runs
    
    # Benchmark efficient
    start_time = time.time()
    for _ in range(num_runs):
        efficient_collector.collect_gradients_for_batch(data_batch, target_batch, criterion)
        efficient_collector.clear_gradients()
    efficient_time = (time.time() - start_time) / num_runs
    
    # Clean up
    import shutil
    shutil.rmtree('tmp_benchmark_original', ignore_errors=True)
    shutil.rmtree('tmp_benchmark_efficient', ignore_errors=True)
    
    speedup = original_time / efficient_time if efficient_time > 0 else float('inf')
    
    print(f"Original collector: {original_time:.3f}s per batch")
    print(f"Efficient collector: {efficient_time:.3f}s per batch")
    print(f"Speedup: {speedup:.1f}x")
    
    return {
        'original_time': original_time,
        'efficient_time': efficient_time,
        'speedup': speedup
    }