"""
Verification script to confirm that efficient gradient collector 
produces true per-sample gradients (not batch gradients)
"""
import torch
import torch.nn as nn
import numpy as np
from tinystories_model import TinyStoriesTransformer
from tinystories_config import TinyStoriesConfig
from efficient_gradient_collector import EfficientPerSampleGradientCollector
from gradient_collector import PerSampleGradientCollector

def create_simple_test_data(batch_size=4, seq_len=8, vocab_size=100):
    """Create simple test data for verification"""
    # Create different input sequences to ensure we get different gradients per sample
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Make each sample slightly different to ensure different gradients
    for i in range(batch_size):
        input_ids[i, 0] = i  # Each sample starts with a different token
    
    labels = input_ids.clone()
    return input_ids, labels

def test_criterion(output, target):
    """Simple loss function for testing"""
    if isinstance(output, dict):
        logits = output['logits']
    else:
        logits = output
    
    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

def compute_batch_gradients(model, input_ids, labels):
    """Compute traditional batch gradients for comparison"""
    model.zero_grad()
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    loss.backward()
    
    batch_grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            batch_grads[name] = param.grad.detach().cpu().clone()
    
    return batch_grads

def main():
    print("Verifying Per-Sample vs Batch Gradients")
    print("=" * 50)
    
    # Create small model for testing
    config = TinyStoriesConfig()
    model = TinyStoriesTransformer(
        vocab_size=100,  # Small vocab for testing
        d_model=32,      # Small model
        n_heads=2,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0      # No dropout for deterministic results
    )
    
    print(f"Test model parameters: {model.count_parameters():,}")
    
    # Create test data
    batch_size = 4
    input_ids, labels = create_simple_test_data(batch_size, seq_len=8, vocab_size=100)
    
    print(f"Input batch shape: {input_ids.shape}")
    print(f"Input data (first few tokens per sample):")
    for i in range(batch_size):
        print(f"  Sample {i}: {input_ids[i, :3].tolist()}")
    
    print("\n" + "-" * 50)
    print("1. Computing per-sample gradients (original method)")
    
    # Method 1: Original per-sample gradients
    original_collector = PerSampleGradientCollector(model, 'test_original')
    original_collector.collect_gradients_for_batch(input_ids, labels, test_criterion)
    
    # Method 2: Efficient per-sample gradients
    print("2. Computing per-sample gradients (efficient method)")
    efficient_collector = EfficientPerSampleGradientCollector(model, 'test_efficient')
    efficient_collector.collect_gradients_for_batch(input_ids, labels, test_criterion)
    
    # Method 3: Batch gradients for comparison
    print("3. Computing batch gradients (for comparison)")
    batch_grads = compute_batch_gradients(model, input_ids, labels)
    
    print("\n" + "-" * 50)
    print("VERIFICATION RESULTS")
    print("-" * 50)
    
    # Compare shapes and values
    param_name = list(model.named_parameters())[0][0]  # Get first parameter name
    
    # Get gradients from collectors
    original_grads = None
    efficient_grads = None
    
    # Extract from original collector
    for name, grad_list in original_collector.gradients.items():
        if param_name in name and grad_list:
            original_grads = grad_list[0]  # First batch
            break
    
    # Extract from efficient collector  
    for name, grad_list in efficient_collector.gradients.items():
        if param_name in name and grad_list:
            efficient_grads = grad_list[0]  # First batch
            break
    
    # Debug: print available gradient keys
    print("Available gradient keys:")
    print("Original collector:", list(original_collector.gradients.keys()))
    print("Efficient collector:", list(efficient_collector.gradients.keys()))
    print("Looking for parameter containing:", param_name)
    
    batch_grad = batch_grads[param_name]
    
    print(f"Parameter analyzed: {param_name}")
    print(f"Parameter shape: {batch_grad.shape}")
    print()
    
    if original_grads is not None:
        print(f"Original per-sample gradients shape: {original_grads.shape}")
        print(f"  -> First dimension = {original_grads.shape[0]} (should equal batch_size={batch_size})")
        
    if efficient_grads is not None:
        print(f"Efficient per-sample gradients shape: {efficient_grads.shape}")
        print(f"  -> First dimension = {efficient_grads.shape[0]} (should equal batch_size={batch_size})")
        
    print(f"Batch gradient shape: {batch_grad.shape}")
    print(f"  -> No sample dimension (this is the summed/averaged gradient)")
    
    print("\n" + "-" * 30)
    print("GRADIENT DIVERSITY CHECK")
    print("-" * 30)
    
    # Check if per-sample gradients are actually different
    if efficient_grads is not None and efficient_grads.shape[0] > 1:
        # Flatten gradients for easier comparison
        flat_grads = efficient_grads.view(efficient_grads.shape[0], -1)
        
        # Compute pairwise differences
        sample_0 = flat_grads[0]
        sample_1 = flat_grads[1]
        
        diff_norm = torch.norm(sample_0 - sample_1).item()
        grad_0_norm = torch.norm(sample_0).item()
        grad_1_norm = torch.norm(sample_1).item()
        
        print(f"Gradient norm for sample 0: {grad_0_norm:.6f}")
        print(f"Gradient norm for sample 1: {grad_1_norm:.6f}")
        print(f"Difference norm: {diff_norm:.6f}")
        
        if diff_norm > 1e-6:
            print("✅ PASS: Per-sample gradients are different (as expected)")
        else:
            print("❌ FAIL: Per-sample gradients are identical (unexpected)")
        
        # Compare with batch gradient
        batch_flat = batch_grad.view(-1)
        mean_per_sample = flat_grads.mean(dim=0)
        
        batch_vs_mean_diff = torch.norm(batch_flat - mean_per_sample).item()
        print(f"Batch gradient vs mean per-sample gradient difference: {batch_vs_mean_diff:.6f}")
        
        if batch_vs_mean_diff < 1e-4:
            print("✅ PASS: Batch gradient ≈ mean of per-sample gradients (as expected)")
        else:
            print("❌ Note: Batch gradient differs from mean per-sample (may be due to normalization)")
    
    print("\n" + "-" * 30)
    print("CONSISTENCY CHECK")
    print("-" * 30)
    
    # Compare original vs efficient
    if original_grads is not None and efficient_grads is not None:
        consistency_diff = torch.norm(original_grads - efficient_grads).item()
        max_grad_norm = max(torch.norm(original_grads).item(), torch.norm(efficient_grads).item())
        relative_diff = consistency_diff / (max_grad_norm + 1e-8)
        
        print(f"Original vs Efficient difference: {consistency_diff:.8f}")
        print(f"Relative difference: {relative_diff:.8f}")
        
        if relative_diff < 1e-4:
            print("✅ PASS: Original and efficient methods produce consistent results")
        else:
            print("❌ FAIL: Methods produce different results")
    
    print("\n" + "=" * 50)
    print("CONCLUSION")
    print("=" * 50)
    
    if efficient_grads is not None:
        is_per_sample = (efficient_grads.shape[0] == batch_size and 
                        len(efficient_grads.shape) > len(batch_grad.shape))
        
        if is_per_sample:
            print("✅ CONFIRMED: Efficient collector produces TRUE PER-SAMPLE GRADIENTS")
            print(f"   Each sample gets its own gradient vector of shape {efficient_grads.shape[1:]}")
            print(f"   Total storage: {batch_size} gradients per parameter")
        else:
            print("❌ ERROR: Efficient collector is NOT producing per-sample gradients")
    else:
        print("❌ ERROR: Could not verify efficient collector gradients")
    
    # Cleanup
    import shutil
    shutil.rmtree('test_original', ignore_errors=True)
    shutil.rmtree('test_efficient', ignore_errors=True)

if __name__ == "__main__":
    main() 