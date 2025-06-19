"""
Gradient collection utilities for per-sample gradient tracking
"""
import torch
import torch.nn as nn
import os
import pickle
from collections import OrderedDict
import numpy as np

class PerSampleGradientCollector:
    def __init__(self, model, save_dir):
        """
        Initialize gradient collector
        
        Args:
            model: PyTorch model
            save_dir: Directory to save gradients
        """
        self.model = model
        self.save_dir = save_dir
        self.gradients = OrderedDict()
        self.sample_count = 0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize gradient storage for each parameter
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Get the shape of the parameter
                param_shape = param.shape
                # We'll store gradients as [num_samples, *param_shape]
                # Initialize with None, will be filled as we collect
                self.gradients[name] = {
                    'shapes': [],
                    'gradients': [],
                    'param_shape': param_shape
                }
    
    def collect_gradients_for_batch(self, data, target, criterion, sample_indices=None):
        """
        Collect per-sample gradients for a batch
        
        Args:
            data: Input data [batch_size, ...]
            target: Target labels [batch_size]
            criterion: Loss function
            sample_indices: Optional indices for tracking sample order
        """
        batch_size = data.size(0)
        
        # Process each sample individually
        for i in range(batch_size):
            # Extract single sample
            single_data = data[i:i+1]  # Keep batch dimension
            single_target = target[i:i+1]
            
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(single_data)
            loss = criterion(output, single_target)
            
            # Backward pass
            loss.backward()
            
            # Collect gradients for this sample
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Detach and move to CPU to save memory
                    grad = param.grad.detach().cpu().clone()
                    
                    # Store gradient
                    self.gradients[name]['gradients'].append(grad)
                    self.gradients[name]['shapes'].append(grad.shape)
            
            self.sample_count += 1
            
            if self.sample_count % 1000 == 0:
                print(f"Collected gradients for {self.sample_count} samples")
    
    def save_gradients(self, epoch, batch_idx=None):
        """
        Save collected gradients to disk
        
        Args:
            epoch: Current epoch number
            batch_idx: Optional batch index for more granular saving
        """
        # Create epoch directory
        epoch_dir = os.path.join(self.save_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Save gradients for each parameter
        for name, grad_info in self.gradients.items():
            if grad_info['gradients']:
                # Stack all gradients for this parameter
                stacked_grads = torch.stack(grad_info['gradients'], dim=0)
                
                # Create filename
                if batch_idx is not None:
                    filename = f'{name}_epoch_{epoch}_batch_{batch_idx}.pt'
                else:
                    filename = f'{name}_epoch_{epoch}.pt'
                
                filepath = os.path.join(epoch_dir, filename)
                
                # Save gradients
                torch.save({
                    'gradients': stacked_grads,
                    'param_shape': grad_info['param_shape'],
                    'num_samples': len(grad_info['gradients'])
                }, filepath)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_samples': self.sample_count,
            'parameter_names': list(self.gradients.keys())
        }
        
        metadata_path = os.path.join(epoch_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def clear_gradients(self):
        """Clear collected gradients to free memory"""
        for name in self.gradients:
            self.gradients[name]['gradients'] = []
            self.gradients[name]['shapes'] = []
        self.sample_count = 0

def compute_gradient_statistics(gradients_dir, epoch):
    """
    Compute statistics over collected gradients
    
    Args:
        gradients_dir: Directory containing saved gradients
        epoch: Epoch number to analyze
    """
    epoch_dir = os.path.join(gradients_dir, f'epoch_{epoch}')
    
    if not os.path.exists(epoch_dir):
        print(f"No gradients found for epoch {epoch}")
        return
    
    stats = {}
    
    # Load metadata
    metadata_path = os.path.join(epoch_dir, 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Analyzing gradients for epoch {epoch}")
    print(f"Total samples: {metadata['total_samples']}")
    
    # Analyze each parameter
    for param_name in metadata['parameter_names']:
        # Find gradient file
        grad_files = [f for f in os.listdir(epoch_dir) if f.startswith(param_name) and f.endswith('.pt')]
        
        if not grad_files:
            continue
        
        # Load gradients (assuming single file per parameter for now)
        grad_file = grad_files[0]
        grad_path = os.path.join(epoch_dir, grad_file)
        
        grad_data = torch.load(grad_path)
        gradients = grad_data['gradients']  # Shape: [num_samples, *param_shape]
        
        # Compute statistics
        grad_norms = torch.norm(gradients.view(gradients.size(0), -1), dim=1)
        
        stats[param_name] = {
            'mean_norm': grad_norms.mean().item(),
            'std_norm': grad_norms.std().item(),
            'min_norm': grad_norms.min().item(),
            'max_norm': grad_norms.max().item(),
            'shape': gradients.shape
        }
        
        print(f"{param_name}: mean_norm={stats[param_name]['mean_norm']:.6f}, "
              f"std_norm={stats[param_name]['std_norm']:.6f}")
    
    # Save statistics
    stats_path = os.path.join(epoch_dir, 'gradient_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    return stats 