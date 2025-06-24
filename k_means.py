"""
K-means clustering for gradient tensors (GPU accelerated)
"""
import torch
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from tqdm import tqdm

def kmeans_plusplus_gpu(X, n_clusters, random_state=None, n_local_trials=None, device=None):
    """
    GPU-accelerated k-means++ initialization.
    
    Args:
        X: Tensor of shape [n_samples, n_features]
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        n_local_trials: Number of seeding trials for each center (except the first)
        device: Device to use
    
    Returns:
        centers: Tensor of shape [n_clusters, n_features]
        indices: Tensor of shape [n_clusters] with indices of chosen centers
    """
    if device is None:
        device = X.device
    
    n_samples, n_features = X.shape
    
    if n_samples < n_clusters:
        raise ValueError(f"n_samples={n_samples} should be >= n_clusters={n_clusters}.")
    
    # Set random seed
    if random_state is not None:
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))
    
    centers = torch.empty((n_clusters, n_features), dtype=X.dtype, device=device)
    indices = torch.full((n_clusters,), -1, dtype=torch.long, device=device)
    
    # Pick first center randomly
    center_id = torch.randint(0, n_samples, (1,), device=device).item()
    centers[0] = X[center_id]
    indices[0] = center_id
    
    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = torch.cdist(centers[0:1].to(device), X, p=2).squeeze(0) ** 2
    current_pot = torch.sum(closest_dist_sq)
    
    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = torch.rand(n_local_trials, device=device) * current_pot
        
        # Compute cumulative sum for sampling
        cumsum = torch.cumsum(closest_dist_sq, dim=0)
        cumsum = cumsum / cumsum[-1]  # Normalize to [0, 1]
        
        # Find candidate indices using binary search
        candidate_ids = torch.searchsorted(cumsum, rand_vals, right=True)
        candidate_ids = torch.clamp(candidate_ids, 0, n_samples - 1)
        
        # Compute distances to center candidates
        distance_to_candidates = torch.cdist(X[candidate_ids], X, p=2) ** 2
        
        # Update closest distances squared and potential for each candidate
        min_distances = torch.minimum(closest_dist_sq.unsqueeze(0), distance_to_candidates)
        candidates_pot = torch.sum(min_distances, dim=1)
        
        # Decide which candidate is the best
        best_candidate_idx = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate_idx]
        closest_dist_sq = min_distances[best_candidate_idx]
        best_candidate = candidate_ids[best_candidate_idx]
        
        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    
    return centers, indices

def kmeans_gradients_gpu(gradients_tensor, n_clusters=5, random_state=42, max_iter=300, n_init=10, device=None, init_method='kmeans++', start_fraction=0.0, end_fraction=1.0, min_grad_percentile=0.0):
    """
    Perform k-means clustering on a gradient tensor using GPU acceleration.
    
    Args:
        gradients_tensor: Tensor of shape [num_samples, weight_shape_0, weight_shape_1, ...]
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations for k-means
        n_init: Number of times to run k-means with different centroid seeds
        device: Device to use (defaults to gradients_tensor.device)
        init_method: Initialization method ('kmeans++' or 'random')
        start_fraction: Fraction of batches to start from (0.0 to 1.0)
        end_fraction: Fraction of batches to end at (0.0 to 1.0)
        min_grad_percentile: Minimum gradient norm percentile to include (0.0 to 100.0)
    
    Returns:
        Dictionary containing all clustering results with cluster_labels corresponding to original batch indices
    """
    if device is None:
        device = gradients_tensor.device
    
    # Ensure the main tensor is on the correct device from the start
    gradients_tensor = gradients_tensor.to(device)
    
    print(f"Performing GPU-accelerated k-means clustering on gradients tensor of shape {gradients_tensor.shape}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Initialization method: {init_method}")
    print(f"Device: {device}")
    print(f"Using batches from {start_fraction:.1%} to {end_fraction:.1%} for clustering")
    print(f"Minimum gradient norm percentile: {min_grad_percentile:.1f}%")
    
    # Calculate batch indices for the subset
    num_samples = gradients_tensor.shape[0]
    start_idx = int(start_fraction * num_samples)
    end_idx = int(end_fraction * num_samples)
    subset_size = end_idx - start_idx
    
    print(f"Clustering on {subset_size} batches (indices {start_idx} to {end_idx-1}) out of {num_samples} total batches")
    
    # Extract subset for clustering
    subset_gradients = gradients_tensor[start_idx:end_idx]
    
    # Calculate gradient norms for the subset
    subset_shape = subset_gradients.shape
    subset_flat = subset_gradients.view(subset_size, -1)
    gradient_norms = torch.norm(subset_flat, dim=1)
    
    # Calculate percentile threshold
    if min_grad_percentile > 0.0:
        percentile_threshold = torch.quantile(gradient_norms, min_grad_percentile / 100.0)
        print(f"Gradient norm percentile threshold: {percentile_threshold:.4f}")
        
        # Filter gradients by norm
        norm_mask = gradient_norms >= percentile_threshold
        filtered_indices = torch.nonzero(norm_mask).squeeze(-1)
        filtered_size = len(filtered_indices)
        
        print(f"Filtered to {filtered_size} gradients (out of {subset_size}) with norm >= {percentile_threshold:.4f}")
        
        if filtered_size < n_clusters:
            print(f"Warning: Only {filtered_size} gradients remain after filtering, but {n_clusters} clusters requested")
            print("Using all remaining gradients for clustering")
            filtered_indices = torch.arange(subset_size, device=device)
            filtered_size = subset_size
        
        # Extract filtered gradients
        filtered_gradients = subset_flat[filtered_indices]
        
        # Create mapping from filtered indices back to original batch indices
        # filtered_indices contains indices within the subset (0 to subset_size-1)
        # We need to map these to original batch indices (start_idx to end_idx-1)
        original_batch_indices = start_idx + filtered_indices
        
    else:
        # No filtering, use all gradients in subset
        filtered_gradients = subset_flat
        filtered_size = subset_size
        original_batch_indices = torch.arange(start_idx, end_idx, device=device)
        print(f"No gradient norm filtering applied, using all {filtered_size} gradients")
    
    print(f"Final clustering shape: {filtered_gradients.shape}")
    print(f"Memory usage: {filtered_gradients.numel() * filtered_gradients.element_size() / (1024**2):.2f} MB")
    
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
    
    # Perform k-means clustering with multiple initializations on filtered subset
    start_time = time.time()
    
    best_inertia = float('inf')
    best_cluster_labels = None
    best_cluster_centers = None
    
    print(f"Running k-means with {n_init} different initializations...")
    
    for init_idx in tqdm(range(n_init), desc="K-means initializations"):
        # Initialize cluster centers
        if init_method == 'kmeans++':
            cluster_centers, _ = kmeans_plusplus_gpu(
                filtered_gradients, n_clusters, random_state=random_state + init_idx, device=device
            )
        else:  # random initialization
            indices = torch.randperm(filtered_size, device=device)[:n_clusters]
            cluster_centers = filtered_gradients[indices].clone()
        
        # K-means iterations with progress bar
        for iteration in tqdm(range(max_iter), desc=f"Init {init_idx+1}/{n_init} iterations", leave=False):
            # Compute distances to all cluster centers
            # Shape: [filtered_size, n_clusters]
            distances = torch.cdist(filtered_gradients, cluster_centers, p=2)
            
            # Assign samples to nearest cluster
            cluster_labels = torch.argmin(distances, dim=1)
            
            # Update cluster centers
            new_centers = torch.zeros_like(cluster_centers)
            for k in range(n_clusters):
                cluster_mask = cluster_labels == k
                if cluster_mask.sum() > 0:
                    new_centers[k] = filtered_gradients[cluster_mask].mean(dim=0)
                else:
                    # If cluster is empty, keep the old center
                    new_centers[k] = cluster_centers[k]
            
            # Check for convergence
            if torch.allclose(cluster_centers, new_centers, atol=1e-6):
                break
                
            cluster_centers = new_centers
        
        # Compute inertia for this initialization
        distances = torch.cdist(filtered_gradients, cluster_centers, p=2)
        min_distances = torch.min(distances, dim=1)[0]
        inertia = torch.sum(min_distances ** 2).item()
        
        # Keep the best result
        if inertia < best_inertia:
            best_inertia = inertia
            best_cluster_labels = cluster_labels.cpu().numpy()
            best_cluster_centers = cluster_centers.cpu().numpy()
    
    end_time = time.time()
    clustering_time = end_time - start_time
    
    print(f"Clustering completed in {clustering_time:.2f} seconds")
    
    # Now assign all original batches to clusters using the learned centers
    print("Assigning all original batches to clusters...")
    flattened_all = gradients_tensor.view(num_samples, -1)
    distances_to_centers = torch.cdist(flattened_all, torch.tensor(best_cluster_centers, device=device), p=2).cpu().numpy()
    all_cluster_labels = np.argmin(distances_to_centers, axis=1)
    min_distances = np.min(distances_to_centers, axis=1)
    
    # Cluster statistics for all samples
    unique_labels, counts = np.unique(all_cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))
    
    # Compute cluster statistics
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        cluster_mask = all_cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Calculate inertia for this cluster
        cluster_inertia = np.sum(min_distances[cluster_mask] ** 2)
        
        # Calculate silhouette score for a sample of the cluster
        if cluster_sizes.get(cluster_id, 0) > 1:
            try:
                sample_size = min(1000, cluster_sizes[cluster_id])
                sample_indices = np.random.choice(cluster_indices, sample_size, replace=False)
                dist_sample = distances_to_centers[sample_indices, :]
                silhouette_val = silhouette_score(dist_sample, all_cluster_labels[sample_indices])
            except ValueError:
                silhouette_val = -1 # Score is undefined for single-cluster samples
        else:
            silhouette_val = -1  # Not defined for single-point clusters
            
        cluster_stats[cluster_id] = {
            'size': cluster_sizes.get(cluster_id, 0),
            'inertia': cluster_inertia,
            'avg_silhouette': silhouette_val
        }
    
    # Prepare results
    results = {
        'cluster_labels': best_cluster_labels,
        'all_cluster_labels': all_cluster_labels,
        'cluster_centers': best_cluster_centers,
        'inertia': best_inertia,
        'clustering_time': clustering_time,
        'cluster_sizes': cluster_sizes,
        'cluster_stats': cluster_stats,
        'distances_to_centers': distances_to_centers,
        'min_distances': min_distances,
        'flattened_gradients': flattened_all.cpu().numpy(),
        'original_shape': gradients_tensor.shape,
        'n_clusters': n_clusters,
        'num_samples': num_samples,
        'device_used': str(device),
        'init_method': init_method,
        'start_fraction': start_fraction,
        'end_fraction': end_fraction,
        'min_grad_percentile': min_grad_percentile,
        'subset_size': subset_size,
        'subset_indices': (start_idx, end_idx),
        'filtered_size': filtered_size,
        'original_batch_indices': original_batch_indices.cpu().numpy() if min_grad_percentile > 0.0 else None
    }
    
    # Print summary
    print(f"\nClustering Results Summary:")
    print(f"  Inertia: {best_inertia:.2f}")
    print(f"  Cluster sizes: {cluster_sizes}")
    print(f"  Clustering performed on {filtered_size}/{subset_size} batches in subset ({filtered_size/subset_size:.1%})")
    print(f"  Subset: {subset_size}/{num_samples} total batches ({subset_size/num_samples:.1%})")
    
    return results

def kmeans_gradients(gradients_tensor, n_clusters=5, random_state=42, max_iter=300, n_init=10, use_gpu=True, init_method='kmeans++', start_fraction=0.0, end_fraction=1.0, min_grad_percentile=0.0): 
    """
    Perform k-means clustering on a gradient tensor (GPU or CPU).
    
    Args:
        gradients_tensor: Tensor of shape [num_samples, weight_shape_0, weight_shape_1, ...]
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations for k-means
        n_init: Number of times to run k-means with different centroid seeds
        use_gpu: Whether to use GPU acceleration
        init_method: Initialization method ('kmeans++' or 'random')
        start_fraction: Fraction of batches to start from (0.0 to 1.0)
        end_fraction: Fraction of batches to end at (0.0 to 1.0)
        min_grad_percentile: Minimum gradient norm percentile to include (0.0 to 100.0)
    
    Returns:
        Dictionary containing all clustering results with cluster_labels corresponding to original batch indices
    """
    if use_gpu and torch.cuda.is_available():
        return kmeans_gradients_gpu(gradients_tensor, n_clusters, random_state, max_iter, n_init, init_method=init_method, start_fraction=start_fraction, end_fraction=end_fraction, min_grad_percentile=min_grad_percentile)
    else:
        # Fall back to CPU implementation
        print("Using CPU implementation (GPU not available or disabled)")
        return kmeans_gradients_cpu(gradients_tensor, n_clusters, random_state, max_iter, n_init, init_method=init_method, start_fraction=start_fraction, end_fraction=end_fraction, min_grad_percentile=min_grad_percentile)

def kmeans_gradients_cpu(gradients_tensor, n_clusters=5, random_state=42, max_iter=300, n_init=10, init_method='kmeans++', start_fraction=0.0, end_fraction=1.0, min_grad_percentile=0.0):
    """
    CPU implementation of k-means clustering (fallback).
    """
    from sklearn.cluster import KMeans
    
    print(f"Performing CPU k-means clustering on gradients tensor of shape {gradients_tensor.shape}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Initialization method: {init_method}")
    print(f"Using batches from {start_fraction:.1%} to {end_fraction:.1%} for clustering")
    print(f"Minimum gradient norm percentile: {min_grad_percentile:.1f}%")
    
    # Calculate batch indices for the subset
    num_samples = gradients_tensor.shape[0]
    start_idx = int(start_fraction * num_samples)
    end_idx = int(end_fraction * num_samples)
    subset_size = end_idx - start_idx
    
    print(f"Clustering on {subset_size} batches (indices {start_idx} to {end_idx-1}) out of {num_samples} total batches")
    
    # Extract subset for clustering
    subset_gradients = gradients_tensor[start_idx:end_idx]
    
    # Calculate gradient norms for the subset
    subset_shape = subset_gradients.shape
    subset_flat = subset_gradients.view(subset_size, -1).numpy()
    gradient_norms = np.linalg.norm(subset_flat, axis=1)
    
    # Calculate percentile threshold
    if min_grad_percentile > 0.0:
        percentile_threshold = np.percentile(gradient_norms, min_grad_percentile)
        print(f"Gradient norm percentile threshold: {percentile_threshold:.4f}")
        
        # Filter gradients by norm
        norm_mask = gradient_norms >= percentile_threshold
        filtered_indices = np.nonzero(norm_mask)[0]
        filtered_size = len(filtered_indices)
        
        print(f"Filtered to {filtered_size} gradients (out of {subset_size}) with norm >= {percentile_threshold:.4f}")
        
        if filtered_size < n_clusters:
            print(f"Warning: Only {filtered_size} gradients remain after filtering, but {n_clusters} clusters requested")
            print("Using all remaining gradients for clustering")
            filtered_indices = np.arange(subset_size)
            filtered_size = subset_size
        
        # Extract filtered gradients
        filtered_gradients = subset_flat[filtered_indices]
        
        # Create mapping from filtered indices back to original batch indices
        # filtered_indices contains indices within the subset (0 to subset_size-1)
        # We need to map these to original batch indices (start_idx to end_idx-1)
        original_batch_indices = start_idx + filtered_indices
        
    else:
        # No filtering, use all gradients in subset
        filtered_gradients = subset_flat
        filtered_size = subset_size
        original_batch_indices = np.arange(start_idx, end_idx)
        print(f"No gradient norm filtering applied, using all {filtered_size} gradients")
    
    print(f"Final clustering shape: {filtered_gradients.shape}")
    print(f"Memory usage: {filtered_gradients.nbytes / (1024**2):.2f} MB")
    
    # Perform k-means clustering on filtered subset
    start_time = time.time()
    
    # Map our init_method to sklearn's init parameter
    sklearn_init = 'k-means++' if init_method == 'kmeans++' else 'random'
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
        init=sklearn_init,
        verbose=0
    )
    
    subset_cluster_labels = kmeans.fit_predict(filtered_gradients)
    cluster_centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    end_time = time.time()
    clustering_time = end_time - start_time
    
    print(f"Clustering completed in {clustering_time:.2f} seconds")
    
    # Now assign all original batches to clusters using the learned centers
    print("Assigning all original batches to clusters...")
    flattened_all = gradients_tensor.view(num_samples, -1).numpy()
    distances_to_centers = kmeans.transform(flattened_all)
    all_cluster_labels = np.argmin(distances_to_centers, axis=1)
    min_distances = np.min(distances_to_centers, axis=1)
    
    # Cluster statistics for all samples
    unique_labels, counts = np.unique(all_cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))
    
    # Compute cluster statistics
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        cluster_mask = all_cluster_labels == cluster_id
        cluster_distances = min_distances[cluster_mask]
        
        cluster_stats[cluster_id] = {
            'size': int(cluster_sizes[cluster_id]),
            'percentage': float(cluster_sizes[cluster_id] / num_samples * 100),
            'mean_distance': float(np.mean(cluster_distances)),
            'std_distance': float(np.std(cluster_distances)),
            'min_distance': float(np.min(cluster_distances)),
            'max_distance': float(np.max(cluster_distances))
        }
    
    # Prepare results
    results = {
        'cluster_labels': subset_cluster_labels,
        'all_cluster_labels': all_cluster_labels,
        'cluster_centers': cluster_centers,
        'inertia': inertia,
        'clustering_time': clustering_time,
        'cluster_sizes': cluster_sizes,
        'cluster_stats': cluster_stats,
        'distances_to_centers': distances_to_centers,
        'min_distances': min_distances,
        'flattened_gradients': flattened_all,
        'original_shape': gradients_tensor.shape,
        'n_clusters': n_clusters,
        'num_samples': num_samples,
        'device_used': 'cpu',
        'init_method': init_method,
        'start_fraction': start_fraction,
        'end_fraction': end_fraction,
        'min_grad_percentile': min_grad_percentile,
        'subset_size': subset_size,
        'subset_indices': (start_idx, end_idx),
        'filtered_size': filtered_size,
        'original_batch_indices': original_batch_indices if min_grad_percentile > 0.0 else None
    }
    
    # Print summary
    print(f"\nClustering Results Summary:")
    print(f"  Inertia: {inertia:.2f}")
    print(f"  Cluster sizes: {cluster_sizes}")
    print(f"  Clustering performed on {filtered_size}/{subset_size} batches in subset ({filtered_size/subset_size:.1%})")
    print(f"  Subset: {subset_size}/{num_samples} total batches ({subset_size/num_samples:.1%})")
    
    return results

def plot_clustering_results(results, param_name="gradients"):
    """
    Plot comprehensive clustering results.
    
    Args:
        results: Results dictionary from kmeans_gradients
        param_name: Name of the parameter for plot titles
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cluster size distribution
    ax1 = axes[0, 0]
    cluster_ids = list(results['cluster_sizes'].keys())
    cluster_sizes = list(results['cluster_sizes'].values())
    ax1.bar(cluster_ids, cluster_sizes, alpha=0.7, edgecolor='black')
    ax1.set_title(f'Cluster Size Distribution - {param_name}')
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of Samples')
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for i, (cluster_id, size) in enumerate(zip(cluster_ids, cluster_sizes)):
        percentage = size / results['num_samples'] * 100
        ax1.text(cluster_id, size + max(cluster_sizes) * 0.01, f'{percentage:.1f}%', 
                ha='center', va='bottom')
    
    # 2. Distance distribution
    ax2 = axes[0, 1]
    ax2.hist(results['min_distances'], bins=50, alpha=0.7, edgecolor='black')
    ax2.set_title(f'Distance to Nearest Cluster Center - {param_name}')
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cluster center distances heatmap
    ax3 = axes[1, 0]
    distances = results['distances_to_centers']
    im = ax3.imshow(distances[:1000], cmap='viridis', aspect='auto')  # Show first 1000 samples
    ax3.set_title(f'Distance to Cluster Centers (First 1000 samples) - {param_name}')
    ax3.set_xlabel('Cluster ID')
    ax3.set_ylabel('Sample Index')
    plt.colorbar(im, ax=ax3)
    
    # 4. Cluster statistics
    ax4 = axes[1, 1]
    cluster_ids = list(results['cluster_stats'].keys())
    mean_distances = [results['cluster_stats'][cid]['mean_distance'] for cid in cluster_ids]
    std_distances = [results['cluster_stats'][cid]['std_distance'] for cid in cluster_ids]
    
    ax4.bar(cluster_ids, mean_distances, yerr=std_distances, alpha=0.7, edgecolor='black', capsize=5)
    ax4.set_title(f'Mean Distance to Cluster Center - {param_name}')
    ax4.set_xlabel('Cluster ID')
    ax4.set_ylabel('Mean Distance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_cluster_characteristics(results):
    """
    Analyze and print detailed cluster characteristics.
    
    Args:
        results: Results dictionary from kmeans_gradients
    """
    print(f"\nDetailed Cluster Analysis:")
    print("=" * 60)
    
    for cluster_id in range(results['n_clusters']):
        stats = results['cluster_stats'][cluster_id]
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {stats['size']} samples ({stats['percentage']:.1f}%)")
        print(f"  Mean distance to center: {stats['mean_distance']:.4f}")
        print(f"  Std distance to center: {stats['std_distance']:.4f}")
        print(f"  Min distance to center: {stats['min_distance']:.4f}")
        print(f"  Max distance to center: {stats['max_distance']:.4f}")
    
    print(f"\nOverall Metrics:")
    print(f"  Inertia: {results['inertia']:.2f}")
    print(f"  Clustering time: {results['clustering_time']:.2f} seconds")
    print(f"  Device used: {results['device_used']}")

def find_optimal_clusters(gradients_tensor, max_clusters=10, random_state=42, use_gpu=True):
    """
    Find optimal number of clusters using elbow method and silhouette analysis.
    
    Args:
        gradients_tensor: Tensor of shape [num_samples, weight_shape_0, weight_shape_1, ...]
        max_clusters: Maximum number of clusters to try
        random_state: Random seed for reproducibility
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Dictionary with elbow analysis results
    """
    print(f"Finding optimal number of clusters (max: {max_clusters})")
    if use_gpu and torch.cuda.is_available():
        print("Using GPU acceleration")
    else:
        print("Using CPU")
    
    # Flatten the gradients
    num_samples = gradients_tensor.shape[0]
    if use_gpu and torch.cuda.is_available():
        flattened_gradients = gradients_tensor.view(num_samples, -1)
        device = flattened_gradients.device
    else:
        flattened_gradients = gradients_tensor.view(num_samples, -1).numpy()
        device = 'cpu'
    
    # Try different numbers of clusters
    n_clusters_range = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    
    for n_clusters in tqdm(n_clusters_range, desc="Testing cluster numbers"):
        if use_gpu and torch.cuda.is_available():
            # GPU implementation
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
            
            # Initialize cluster centers randomly
            indices = torch.randperm(num_samples, device=device)[:n_clusters]
            cluster_centers = flattened_gradients[indices].clone()
            
            # K-means iterations
            for iteration in tqdm(range(100), desc=f"Testing {n_clusters} clusters", leave=False):  # Reduced max_iter for speed
                distances = torch.cdist(flattened_gradients, cluster_centers, p=2)
                cluster_labels = torch.argmin(distances, dim=1)
                
                new_centers = torch.zeros_like(cluster_centers)
                for k in range(n_clusters):
                    cluster_mask = cluster_labels == k
                    if cluster_mask.sum() > 0:
                        new_centers[k] = flattened_gradients[cluster_mask].mean(dim=0)
                    else:
                        new_centers[k] = cluster_centers[k]
                
                if torch.allclose(cluster_centers, new_centers, atol=1e-6):
                    break
                    
                cluster_centers = new_centers
            
            # Compute metrics
            distances = torch.cdist(flattened_gradients, cluster_centers, p=2)
            min_distances = torch.min(distances, dim=1)[0]
            inertia = torch.sum(min_distances ** 2).item()
            
            # Move to CPU for sklearn metrics
            cluster_labels_cpu = cluster_labels.cpu().numpy()
            flattened_gradients_cpu = flattened_gradients.cpu().numpy()
            
        else:
            # CPU implementation
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5, max_iter=100)
            cluster_labels_cpu = kmeans.fit_predict(flattened_gradients)
            inertia = kmeans.inertia_
            flattened_gradients_cpu = flattened_gradients
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette_score(flattened_gradients_cpu, cluster_labels_cpu))
        calinski_scores.append(calinski_harabasz_score(flattened_gradients_cpu, cluster_labels_cpu))
        davies_scores.append(davies_bouldin_score(flattened_gradients_cpu, cluster_labels_cpu))
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Elbow plot
    axes[0, 0].plot(n_clusters_range, inertias, 'bo-')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].grid(True)
    
    # Silhouette score
    axes[0, 1].plot(n_clusters_range, silhouette_scores, 'ro-')
    axes[0, 1].set_title('Silhouette Score (Higher is Better)')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].grid(True)
    
    # Calinski-Harabasz score
    axes[1, 0].plot(n_clusters_range, calinski_scores, 'go-')
    axes[1, 0].set_title('Calinski-Harabasz Score (Higher is Better)')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].grid(True)
    
    # Davies-Bouldin score
    axes[1, 1].plot(n_clusters_range, davies_scores, 'mo-')
    axes[1, 1].set_title('Davies-Bouldin Score (Lower is Better)')
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('Davies-Bouldin Score')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal clusters
    optimal_silhouette = n_clusters_range[np.argmax(silhouette_scores)]
    optimal_calinski = n_clusters_range[np.argmax(calinski_scores)]
    optimal_davies = n_clusters_range[np.argmin(davies_scores)]
    
    print(f"\nOptimal number of clusters:")
    print(f"  Silhouette method: {optimal_silhouette}")
    print(f"  Calinski-Harabasz method: {optimal_calinski}")
    print(f"  Davies-Bouldin method: {optimal_davies}")
    
    return {
        'n_clusters_range': list(n_clusters_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_scores': davies_scores,
        'optimal_silhouette': optimal_silhouette,
        'optimal_calinski': optimal_calinski,
        'optimal_davies': optimal_davies
    } 