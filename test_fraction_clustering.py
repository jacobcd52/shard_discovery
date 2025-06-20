#!/usr/bin/env python3
"""
Test script to verify start_fraction, end_fraction, and min_grad_percentile functionality in kmeans_gradients.
"""

import torch
import numpy as np
from k_means import kmeans_gradients

def test_fraction_clustering():
    """Test that fraction-based clustering works correctly."""
    
    # Create a simple test tensor with 100 batches
    print("Creating test tensor with 100 batches...")
    num_batches = 100
    batch_size = 10
    feature_dim = 5
    
    # Create synthetic gradients with some structure
    torch.manual_seed(42)
    gradients = torch.randn(num_batches, batch_size, feature_dim)
    
    # Add some structure to make clustering meaningful
    # First 30 batches: cluster 0
    gradients[:30] += 2.0
    # Next 30 batches: cluster 1  
    gradients[30:60] -= 1.0
    # Last 40 batches: cluster 2
    gradients[60:] += 0.5
    
    print(f"Test tensor shape: {gradients.shape}")
    
    # Test 1: Full clustering (should use all batches)
    print("\n" + "="*50)
    print("TEST 1: Full clustering (start_fraction=0.0, end_fraction=1.0)")
    print("="*50)
    
    results_full = kmeans_gradients(
        gradients, 
        n_clusters=3, 
        random_state=42, 
        max_iter=50, 
        n_init=3,
        use_gpu=False,  # Use CPU for testing
        start_fraction=0.0, 
        end_fraction=1.0
    )
    
    print(f"Full clustering cluster sizes: {results_full['cluster_sizes']}")
    print(f"Full clustering cluster labels shape: {results_full['cluster_labels'].shape}")
    print(f"Full clustering subset info: {results_full['subset_size']}/{results_full['num_samples']}")
    
    # Test 2: Partial clustering (should use only middle 60% of batches)
    print("\n" + "="*50)
    print("TEST 2: Partial clustering (start_fraction=0.2, end_fraction=0.8)")
    print("="*50)
    
    results_partial = kmeans_gradients(
        gradients, 
        n_clusters=3, 
        random_state=42, 
        max_iter=50, 
        n_init=3,
        use_gpu=False,  # Use CPU for testing
        start_fraction=0.2, 
        end_fraction=0.8
    )
    
    print(f"Partial clustering cluster sizes: {results_partial['cluster_sizes']}")
    print(f"Partial clustering cluster labels shape: {results_partial['cluster_labels'].shape}")
    print(f"Partial clustering subset info: {results_partial['subset_size']}/{results_partial['num_samples']}")
    print(f"Partial clustering subset indices: {results_partial['subset_indices']}")
    
    # Test 3: Gradient norm filtering (should use only gradients with high norms)
    print("\n" + "="*50)
    print("TEST 3: Gradient norm filtering (min_grad_percentile=80.0)")
    print("="*50)
    
    results_norm_filtered = kmeans_gradients(
        gradients, 
        n_clusters=3, 
        random_state=42, 
        max_iter=50, 
        n_init=3,
        use_gpu=False,  # Use CPU for testing
        start_fraction=0.2, 
        end_fraction=0.8,
        min_grad_percentile=80.0
    )
    
    print(f"Norm filtered clustering cluster sizes: {results_norm_filtered['cluster_sizes']}")
    print(f"Norm filtered clustering cluster labels shape: {results_norm_filtered['cluster_labels'].shape}")
    print(f"Norm filtered clustering subset info: {results_norm_filtered['subset_size']}/{results_norm_filtered['num_samples']}")
    print(f"Norm filtered clustering filtered info: {results_norm_filtered['filtered_size']}/{results_norm_filtered['subset_size']}")
    print(f"Norm filtered clustering original batch indices: {results_norm_filtered['original_batch_indices']}")
    
    # Test 4: Verify that cluster labels correspond to original batch indices
    print("\n" + "="*50)
    print("TEST 4: Verifying cluster label correspondence")
    print("="*50)
    
    # Check that we get cluster assignments for all original batches
    assert len(results_partial['cluster_labels']) == num_batches, f"Expected {num_batches} cluster labels, got {len(results_partial['cluster_labels'])}"
    assert len(results_norm_filtered['cluster_labels']) == num_batches, f"Expected {num_batches} cluster labels, got {len(results_norm_filtered['cluster_labels'])}"
    
    # Check that the subset was actually used for clustering
    start_idx, end_idx = results_partial['subset_indices']
    expected_subset_size = end_idx - start_idx
    assert results_partial['subset_size'] == expected_subset_size, f"Expected subset size {expected_subset_size}, got {results_partial['subset_size']}"
    
    # Check that norm filtering was applied
    assert results_norm_filtered['filtered_size'] <= results_norm_filtered['subset_size'], f"Filtered size should be <= subset size"
    assert results_norm_filtered['original_batch_indices'] is not None, "Original batch indices should be available when norm filtering is applied"
    
    print("✓ All tests passed!")
    print(f"✓ Cluster labels shape matches original tensor: {results_partial['cluster_labels'].shape[0]} == {num_batches}")
    print(f"✓ Subset size is correct: {results_partial['subset_size']} == {expected_subset_size}")
    print(f"✓ Norm filtering was applied: {results_norm_filtered['filtered_size']} <= {results_norm_filtered['subset_size']}")
    
    # Show some example cluster assignments
    print(f"\nExample cluster assignments (first 10 batches):")
    for i in range(10):
        print(f"  Batch {i}: Cluster {results_norm_filtered['cluster_labels'][i]}")
    
    print(f"\nExample cluster assignments (last 10 batches):")
    for i in range(num_batches-10, num_batches):
        print(f"  Batch {i}: Cluster {results_norm_filtered['cluster_labels'][i]}")
    
    # Show which batches were actually used for clustering
    if results_norm_filtered['original_batch_indices'] is not None:
        print(f"\nBatches used for clustering (out of {results_norm_filtered['subset_size']} in subset):")
        print(f"  Original batch indices: {results_norm_filtered['original_batch_indices']}")
        print(f"  Number of batches used: {len(results_norm_filtered['original_batch_indices'])}")

if __name__ == "__main__":
    test_fraction_clustering() 