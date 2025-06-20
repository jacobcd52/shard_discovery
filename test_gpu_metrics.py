#!/usr/bin/env python3
"""
Test script to verify GPU-accelerated k-means clustering.
"""

import torch
import numpy as np
import time
from k_means import kmeans_gradients_gpu

def test_gpu_clustering():
    """Test GPU-accelerated k-means clustering."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic gradient data
    print("Creating synthetic gradient data...")
    num_samples = 1000
    feature_dim = 100
    
    # Create clusters with some structure
    torch.manual_seed(42)
    gradients = torch.randn(num_samples, feature_dim, device=device)
    
    # Add some cluster structure
    cluster_centers = torch.randn(5, feature_dim, device=device) * 2.0
    cluster_labels = torch.randint(0, 5, (num_samples,), device=device)
    
    # Add cluster structure to data
    for i in range(num_samples):
        gradients[i] += cluster_centers[cluster_labels[i]] * 0.5
    
    print(f"Data shape: {gradients.shape}")
    
    # Test full k-means with GPU
    print("\nTesting GPU-accelerated k-means clustering...")
    start_time = time.time()
    
    # Reshape gradients to simulate parameter tensor
    gradients_tensor = gradients.view(num_samples, 10, 10)  # Reshape to 2D parameter
    
    results = kmeans_gradients_gpu(
        gradients_tensor, 
        n_clusters=5, 
        random_state=42, 
        max_iter=100, 
        n_init=3,
        device=device
    )
    
    full_time = time.time() - start_time
    print(f"Full k-means clustering time: {full_time:.4f} seconds")
    print(f"Clustering time: {results['clustering_time']:.4f} seconds")
    print(f"Cluster sizes: {results['cluster_sizes']}")
    print(f"Inertia: {results['inertia']:.2f}")
    
    # Test with different number of clusters
    print("\nTesting with different numbers of clusters...")
    for n_clusters in [3, 7, 10]:
        print(f"\nTesting with {n_clusters} clusters...")
        start_time = time.time()
        
        results_n = kmeans_gradients_gpu(
            gradients_tensor, 
            n_clusters=n_clusters, 
            random_state=42, 
            max_iter=50, 
            n_init=2,
            device=device
        )
        
        time_taken = time.time() - start_time
        print(f"Time: {time_taken:.4f}s, Inertia: {results_n['inertia']:.2f}, Sizes: {results_n['cluster_sizes']}")

if __name__ == "__main__":
    test_gpu_clustering() 