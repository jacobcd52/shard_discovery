"""
Utility to load all gradients from disk and return as stacked tensors
"""
import torch
import os
import pickle
import glob
import re
from collections import OrderedDict

def load_stacked_gradients(gradients_dir, epoch, batch_idx=None):
    """
    Load all gradients for all state dict entries and return as stacked tensors.
    
    Args:
        gradients_dir: Directory containing saved gradients
        epoch: Epoch number to load
        batch_idx: Optional batch index (if saving per batch)
    
    Returns:
        Dictionary with keys from state_dict and values as tensors of shape
        [num_samples, weight_shape_0, weight_shape_1, ...]
    """
    epoch_dir = os.path.join(gradients_dir, f'epoch_{epoch}')
    
    if not os.path.exists(epoch_dir):
        raise FileNotFoundError(f"Gradients directory not found: {epoch_dir}")
    
    # Load metadata
    metadata_path = os.path.join(epoch_dir, 'metadata.pkl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"Loaded metadata: {metadata}")
    else:
        print("Warning: No metadata file found")
        metadata = None
    
    # Find all gradient files
    if batch_idx is not None:
        # Load specific batch
        pattern = f"*_epoch_{epoch}_batch_{batch_idx}.pt"
    else:
        # Load all batches for the epoch (since files are saved per-batch)
        pattern = f"*_epoch_{epoch}_batch_*.pt"
    
    gradient_files = glob.glob(os.path.join(epoch_dir, pattern))
    gradient_files = [f for f in gradient_files if not f.endswith("_stats.pt")]
    
    if not gradient_files:
        raise FileNotFoundError(f"No gradient files found matching pattern: {pattern}")
    
    print(f"Found {len(gradient_files)} gradient files")
    
    # Load and stack gradients
    stacked_gradients = OrderedDict()
    
    for grad_file in gradient_files:
        # Extract parameter name from filename
        filename = os.path.basename(grad_file)
        
        # Handle the actual filename format: param_name_epoch_X_batch_Y.pt
        if batch_idx is not None:
            # Remove the batch suffix
            suffix = f"_epoch_{epoch}_batch_{batch_idx}.pt"
        else:
            # Remove the batch suffix using regex
            param_name = re.sub(r'_epoch_\d+_batch_\d+\.pt$', '', filename)
        
        if batch_idx is not None:
            if filename.endswith(suffix):
                param_name = filename[:-len(suffix)]
            else:
                print(f"Warning: Could not parse filename {filename}")
                continue
        
        try:
            grad_data = torch.load(grad_file, map_location='cpu')
            
            gradients = grad_data['gradients']  # Shape: [num_samples, *param_shape]
            param_shape = grad_data['param_shape']
            num_samples = grad_data['num_samples']
            
            # Verify shape consistency
            expected_shape = (num_samples,) + param_shape
            if gradients.shape != expected_shape:
                print(f"  Warning: Expected shape {expected_shape}, got {gradients.shape}")
            
            # If loading all batches, we need to accumulate gradients
            if batch_idx is None:
                if param_name not in stacked_gradients:
                    stacked_gradients[param_name] = []
                stacked_gradients[param_name].append(gradients)
            else:
                stacked_gradients[param_name] = gradients
            
        except Exception as e:
            print(f"  Error loading {param_name}: {e}")
            continue
    
    # If loading all batches, concatenate the gradients
    if batch_idx is None:
        print("Concatenating gradients from all batches...")
        final_gradients = OrderedDict()
        for param_name, grad_list in stacked_gradients.items():
            if grad_list:
                # Concatenate along the sample dimension
                concatenated = torch.cat(grad_list, dim=0)
                final_gradients[param_name] = concatenated
        stacked_gradients = final_gradients
    
    print(f"Successfully loaded {len(stacked_gradients)} parameters")
    
    # Print final shapes
    total_memory = 0
    for param_name, gradients in stacked_gradients.items():
        memory_mb = gradients.numel() * gradients.element_size() / (1024 * 1024)
        total_memory += memory_mb
        print(f"  {param_name}: {gradients.shape} ({memory_mb:.2f} MB)")
    
    print(f"Total memory: {total_memory:.2f} MB")
    
    return stacked_gradients

def load_gradients_by_batch(gradients_dir, epoch):
    """
    Load all gradients organized by batch for a given epoch.
    
    Args:
        gradients_dir: Directory containing saved gradients
        epoch: Epoch number to load
    
    Returns:
        Dictionary with batch indices as keys and stacked gradients as values
    """
    epoch_dir = os.path.join(gradients_dir, f'epoch_{epoch}')
    
    if not os.path.exists(epoch_dir):
        raise FileNotFoundError(f"Gradients directory not found: {epoch_dir}")
    
    # Find all batch files
    batch_files = glob.glob(os.path.join(epoch_dir, "*_batch_*.pt"))
    
    if not batch_files:
        raise FileNotFoundError(f"No batch gradient files found in {epoch_dir}")
    
    # Group files by batch index
    batch_gradients = {}
    
    for grad_file in batch_files:
        filename = os.path.basename(grad_file)
        
        # Extract batch index from filename using regex
        # Expected format: param_name_epoch_X_batch_Y.pt
        batch_match = re.search(r'_batch_(\d+)\.pt$', filename)
        if batch_match:
            batch_idx = int(batch_match.group(1))
        else:
            print(f"Warning: Could not extract batch index from {filename}")
            continue
        
        if batch_idx not in batch_gradients:
            batch_gradients[batch_idx] = OrderedDict()
        
        # Extract parameter name by removing the batch suffix
        param_name = re.sub(r'_epoch_\d+_batch_\d+\.pt$', '', filename)
        
        try:
            grad_data = torch.load(grad_file, map_location='cpu')
            batch_gradients[batch_idx][param_name] = grad_data['gradients']
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"Loaded gradients for {len(batch_gradients)} batches")
    for batch_idx, params in batch_gradients.items():
        print(f"  Batch {batch_idx}: {len(params)} parameters")
    
    return batch_gradients 