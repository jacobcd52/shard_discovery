#!/usr/bin/env python3
"""
Utility to visualize MNIST images from clustering results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from gradient_loader import load_stacked_gradients
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

def load_gradients_for_parameter(param_name, epoch, data_dir="gradients"):
    """
    Load gradients for a specific parameter from a specific epoch.
    
    Args:
        param_name: Name of the parameter (e.g., 'fc1.weight')
        epoch: Epoch number to load
        data_dir: Directory containing gradient data
    
    Returns:
        Tensor of gradients for the parameter or None if not found
    """
    try:
        stacked_gradients = load_stacked_gradients(data_dir, epoch)
        if param_name in stacked_gradients:
            return stacked_gradients[param_name]
        else:
            print(f"Parameter {param_name} not found in epoch {epoch}")
            print(f"Available parameters: {list(stacked_gradients.keys())}")
            return None
    except Exception as e:
        print(f"Error loading gradients for {param_name}: {e}")
        return None

def visualize_cluster_samples(clustering_results, param_name, epoch, data_dir="results/gradients", num_samples_per_cluster=10):
    """
    Visualize MNIST samples from each cluster.
    
    Args:
        clustering_results: Results from k-means clustering
        param_name: Name of the parameter
        epoch: Epoch number
        data_dir: Directory containing gradient data
        num_samples_per_cluster: Number of samples to show per cluster
    """
    
    cluster_labels = clustering_results['cluster_labels']
    n_clusters = clustering_results['n_clusters']
    
    # Create subplot grid
    fig, axes = plt.subplots(n_clusters, num_samples_per_cluster, 
                            figsize=(num_samples_per_cluster * 1.5, n_clusters * 1.5))
    
    # If only one cluster, make axes 2D
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    # For each cluster
    for cluster_id in range(n_clusters):
        # Get indices of samples in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            print(f"Warning: Cluster {cluster_id} has no samples")
            continue
        
        # Randomly sample from this cluster
        if len(cluster_indices) > num_samples_per_cluster:
            selected_indices = np.random.choice(cluster_indices, num_samples_per_cluster, replace=False)
        else:
            selected_indices = cluster_indices
            # If we don't have enough samples, just use what we have
            if len(selected_indices) < num_samples_per_cluster:
                print(f"Warning: Cluster {cluster_id} only has {len(selected_indices)} samples, showing all of them")
        
        # Load and display MNIST images for these samples
        for i, sample_idx in enumerate(selected_indices):
            try:
                # Load the MNIST image for this sample
                image = load_mnist_image_for_sample(sample_idx, epoch, data_dir)
                
                ax = axes[cluster_id, i]
                ax.imshow(image, cmap='gray')
                ax.set_title(f'Sample {sample_idx}', fontsize=8)
                ax.axis('off')
                
            except Exception as e:
                ax = axes[cluster_id, i]
                ax.text(0.5, 0.5, f'Error\n{sample_idx}\n{str(e)[:20]}...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=6, color='red')
                ax.axis('off')
        
        # Add cluster label
        cluster_size = len(cluster_indices)
        cluster_percentage = cluster_size / len(cluster_labels) * 100
        axes[cluster_id, 0].set_ylabel(f'Cluster {cluster_id}\n({cluster_size} samples, {cluster_percentage:.1f}%)', 
                                     fontsize=10, rotation=0, ha='right', va='center')
    
    plt.suptitle(f'MNIST Samples by Cluster - {param_name} (Epoch {epoch})', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Print cluster statistics
    print(f"\nCluster Statistics for {param_name} (Epoch {epoch}):")
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_size = len(cluster_indices)
        cluster_percentage = cluster_size / len(cluster_labels) * 100
        print(f"  Cluster {cluster_id}: {cluster_size} samples ({cluster_percentage:.1f}%)")

def load_mnist_image_for_sample(sample_idx, epoch, data_dir="results/gradients"):
    """
    Load the MNIST image for a specific sample index.
    
    Args:
        sample_idx: Index of the sample
        epoch: Epoch number
        data_dir: Directory containing data
    
    Returns:
        MNIST image as numpy array (28x28) or raises error if not found
    """
    try:
        # Try to load from saved MNIST images
        image_path = os.path.join(data_dir, f"mnist_images_epoch_{epoch}.pt")
        if os.path.exists(image_path):
            images = torch.load(image_path)
            if sample_idx < len(images):
                image = images[sample_idx].numpy()
                
                # Reshape and denormalize the image
                if image.ndim == 3 and image.shape[0] == 1:
                    # Remove channel dimension: (1, 28, 28) -> (28, 28)
                    image = image.squeeze(0)
                elif image.ndim == 1 and image.shape[0] == 784:
                    # Reshape flattened: (784,) -> (28, 28)
                    image = image.reshape(28, 28)
                
                # Denormalize: reverse the normalization applied during training
                # Original normalization: (x - 0.1307) / 0.3081
                # So denormalization: x * 0.3081 + 0.1307
                image = image * 0.3081 + 0.1307
                
                # Clip to valid range [0, 1]
                image = np.clip(image, 0, 1)
                
                return image
            else:
                raise ValueError(f"Sample index {sample_idx} out of range (max: {len(images)-1})")
        else:
            raise FileNotFoundError(f"MNIST images file not found: {image_path}")
        
    except Exception as e:
        raise RuntimeError(f"Error loading image for sample {sample_idx}: {e}")

def visualize_cluster_centers(clustering_results, param_name, epoch, data_dir="results/gradients"):
    """
    Visualize the cluster centers as gradient patterns.
    
    Args:
        clustering_results: Results from k-means clustering
        param_name: Name of the parameter
        epoch: Epoch number
        data_dir: Directory containing gradient data
    """
    
    cluster_centers = clustering_results['cluster_centers']
    original_shape = clustering_results['original_shape']
    n_clusters = clustering_results['n_clusters']
    
    # Reshape centers back to original parameter shape
    centers_reshaped = cluster_centers.reshape(n_clusters, *original_shape[1:])
    
    # Create visualization
    if len(original_shape) == 3:  # 2D parameter (e.g., conv weight)
        fig, axes = plt.subplots(1, n_clusters, figsize=(n_clusters * 4, 4))
        if n_clusters == 1:
            axes = [axes]
        
        for i in range(n_clusters):
            center = centers_reshaped[i]
            # Take mean across first dimension if it's a conv layer
            if center.shape[0] > 1:
                center_viz = center.mean(axis=0)
            else:
                center_viz = center.squeeze(0)
            
            # Check if the result is 1D with 784 dimensions (first layer)
            if center_viz.shape == (784,):
                center_viz = center_viz.reshape(28, 28)
            
            im = axes[i].imshow(center_viz, cmap='RdBu_r', aspect='auto')
            axes[i].set_title(f'Cluster {i} Center')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
    
    elif len(original_shape) == 2:  # 1D parameter (e.g., fc weight)
        # Check if this is the first layer (input dimension 784 = 28*28)
        if original_shape[1] == 784:
            # Reshape to 2D image for visualization
            fig, axes = plt.subplots(1, n_clusters, figsize=(n_clusters * 4, 4))
            if n_clusters == 1:
                axes = [axes]
            
            for i in range(n_clusters):
                center = centers_reshaped[i]
                # Reshape from (784,) to (28, 28)
                center_viz = center.reshape(28, 28)
                
                im = axes[i].imshow(center_viz, cmap='RdBu_r', aspect='auto')
                axes[i].set_title(f'Cluster {i} Center')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i])
        else:
            # Regular 1D parameter visualization
            fig, axes = plt.subplots(n_clusters, 1, figsize=(10, n_clusters * 3))
            if n_clusters == 1:
                axes = [axes]
            
            for i in range(n_clusters):
                center = centers_reshaped[i]
                axes[i].plot(center.flatten())
                axes[i].set_title(f'Cluster {i} Center')
                axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'Cluster Centers - {param_name} (Epoch {epoch})', fontsize=16)
    plt.tight_layout()
    plt.show()

def analyze_cluster_diversity(clustering_results, param_name, epoch, data_dir="results/gradients"):
    """
    Analyze the diversity of samples within each cluster.
    
    Args:
        clustering_results: Results from k-means clustering
        param_name: Name of the parameter
        epoch: Epoch number
        data_dir: Directory containing gradient data
    """
    
    cluster_labels = clustering_results['cluster_labels']
    n_clusters = clustering_results['n_clusters']
    
    print(f"\nCluster Diversity Analysis for {param_name} (Epoch {epoch}):")
    print("=" * 60)
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_size = len(cluster_indices)
        
        if cluster_size == 0:
            print(f"Cluster {cluster_id}: No samples")
            continue
        
        # Calculate diversity metrics
        cluster_gradients = clustering_results['flattened_gradients'][cluster_indices]
        
        # Variance within cluster
        variance = np.var(cluster_gradients, axis=0).mean()
        
        # Average distance to cluster center
        cluster_center = clustering_results['cluster_centers'][cluster_id]
        distances = np.linalg.norm(cluster_gradients - cluster_center, axis=1)
        avg_distance = distances.mean()
        std_distance = distances.std()
        
        print(f"Cluster {cluster_id} ({cluster_size} samples):")
        print(f"  Average variance: {variance:.6f}")
        print(f"  Average distance to center: {avg_distance:.6f} ± {std_distance:.6f}")
        print(f"  Distance range: {distances.min():.6f} - {distances.max():.6f}")
        print()

def visualize_specific_cluster(clustering_results, param_name, epoch, cluster_id, data_dir="results/gradients", num_samples=10):
    """
    Visualize samples from a specific cluster.
    
    Args:
        clustering_results: Results from k-means clustering
        param_name: Name of the parameter
        epoch: Epoch number
        cluster_id: ID of the cluster to visualize
        data_dir: Directory containing gradient data
        num_samples: Number of samples to display (default: 10)
    """
    
    cluster_labels = clustering_results['cluster_labels']
    n_clusters = clustering_results['n_clusters']
    
    if cluster_id >= n_clusters:
        print(f"Error: Cluster ID {cluster_id} is out of range. Valid range: 0-{n_clusters-1}")
        return
    
    # Get indices of samples in this cluster
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    
    if len(cluster_indices) == 0:
        print(f"Warning: Cluster {cluster_id} has no samples")
        return
    
    # Randomly sample from this cluster
    if len(cluster_indices) > num_samples:
        selected_indices = np.random.choice(cluster_indices, num_samples, replace=False)
    else:
        selected_indices = cluster_indices
        if len(selected_indices) < num_samples:
            print(f"Warning: Cluster {cluster_id} only has {len(selected_indices)} samples, showing all of them")
    
    # Calculate grid dimensions for 10 columns
    num_cols = 10
    num_rows = (len(selected_indices) + num_cols - 1) // num_cols  # Ceiling division
    
    # Create visualization with grid layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))
    
    # Handle case where there's only one row
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Display MNIST images in grid
    for i, sample_idx in enumerate(selected_indices):
        try:
            # Load the MNIST image for this sample
            image = load_mnist_image_for_sample(sample_idx, epoch, data_dir)
            
            ax = axes_flat[i]
            ax.imshow(image, cmap='gray')
            ax.set_title(f'{sample_idx}', fontsize=8)
            ax.axis('off')
            
        except Exception as e:
            ax = axes_flat[i]
            ax.text(0.5, 0.5, f'Error\n{sample_idx}\n{str(e)[:15]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=6, color='red')
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(selected_indices), len(axes_flat)):
        axes_flat[i].axis('off')
    
    cluster_size = len(cluster_indices)
    cluster_percentage = cluster_size / len(cluster_labels) * 100
    plt.suptitle(f'Cluster {cluster_id} Samples - {param_name} (Epoch {epoch})\n'
                f'{cluster_size} samples ({cluster_percentage:.1f}%)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Print cluster statistics
    print(f"\nCluster {cluster_id} Statistics for {param_name} (Epoch {epoch}):")
    print(f"  Size: {cluster_size} samples ({cluster_percentage:.1f}%)")
    print(f"  Sample indices: {selected_indices.tolist()}")

def print_cluster_info(cluster_labels, param_name, n_clusters=10, samples_per_cluster=5):
    """Print information about each cluster and show sample indices."""
    print(f"\nCluster Information for {param_name}:")
    print(f"Total samples: {len(cluster_labels)}")
    print(f"Number of clusters: {n_clusters}")
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = len(cluster_indices)
        cluster_percentage = 100.0 * cluster_size / len(cluster_labels)
        
        # Select a few sample indices to show
        selected_indices = cluster_indices[:samples_per_cluster] if len(cluster_indices) >= samples_per_cluster else cluster_indices
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {cluster_size} samples ({cluster_percentage:.1f}%)")
        print(f"  Sample indices: {selected_indices.tolist()}")

def visualize_tsne_gradients(clustering_results, param_name, epoch, data_dir="results/gradients", 
                           n_samples=5000, perplexity=30, random_state=42, figsize=(20, 6)):
    """
    Create t-SNE visualization of gradients with three different colorings:
    1. Points colored by true labels
    2. Points colored by cluster assignments
    3. Points colored by model predictions
    
    Args:
        clustering_results: Results from k-means clustering
        param_name: Name of the parameter
        epoch: Epoch number
        data_dir: Directory containing gradient data
        n_samples: Number of samples to use for t-SNE (for performance)
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
        figsize: Figure size for the plot
    """
    
    print(f"Creating t-SNE visualization for {param_name} (epoch {epoch})")
    
    # Get cluster assignments
    cluster_labels = clustering_results['cluster_labels']
    n_clusters = clustering_results['n_clusters']
    
    # Load gradients for the parameter
    stacked_gradients = load_stacked_gradients(data_dir, epoch)
    if param_name not in stacked_gradients:
        print(f"Parameter {param_name} not found in epoch {epoch}")
        return
    
    gradients_tensor = stacked_gradients[param_name]
    print(f"Loaded gradients tensor: {gradients_tensor.shape}")
    
    # Flatten gradients for t-SNE
    # gradients_tensor shape: [num_samples, *param_shape]
    original_shape = gradients_tensor.shape
    flattened_gradients = gradients_tensor.view(original_shape[0], -1)
    
    # Sample a subset for t-SNE (for performance)
    if n_samples < len(flattened_gradients):
        # Random sampling
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(flattened_gradients), n_samples, replace=False)
        sample_gradients = flattened_gradients[sample_indices].cpu().numpy()
        sample_cluster_labels = cluster_labels[sample_indices]
    else:
        sample_gradients = flattened_gradients.cpu().numpy()
        sample_cluster_labels = cluster_labels
        sample_indices = np.arange(len(flattened_gradients))
    
    print(f"Using {len(sample_gradients)} samples for t-SNE")
    
    # Load true labels for the sampled indices
    print("Loading true labels...")
    true_labels = load_true_labels_for_samples(sample_indices, epoch, data_dir)
    
    if true_labels is None:
        print("Warning: Could not load true labels, creating visualization with cluster labels only")
        true_labels = sample_cluster_labels  # Use cluster labels as fallback
    
    # Load model predictions for the sampled indices
    print("Loading model predictions...")
    model_predictions = load_model_predictions_for_samples(sample_indices, epoch, data_dir)
    
    if model_predictions is None:
        print("Warning: Could not load model predictions, creating visualization with cluster labels only")
        model_predictions = sample_cluster_labels  # Use cluster labels as fallback
    
    # Apply PCA for dimensionality reduction before t-SNE (for better performance)
    print("Applying PCA for dimensionality reduction...")
    if sample_gradients.shape[1] > 50:
        pca = PCA(n_components=50, random_state=random_state)
        reduced_gradients = pca.fit_transform(sample_gradients)
        print(f"Reduced from {sample_gradients.shape[1]} to {reduced_gradients.shape[1]} dimensions")
    else:
        reduced_gradients = sample_gradients
    
    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    tsne_result = tsne.fit_transform(reduced_gradients)
    
    # Create visualization with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Colored by true labels
    scatter1 = ax1.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                          c=true_labels, cmap='tab10', alpha=0.7, s=20)
    ax1.set_title(f't-SNE: True Labels\n{param_name} (Epoch {epoch})', fontsize=11)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    
    # Add colorbar for true labels
    cbar1 = plt.colorbar(scatter1, ax=ax1, ticks=range(10))
    cbar1.set_label('True Label')
    
    # Plot 2: Colored by cluster assignments
    scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                          c=sample_cluster_labels, cmap='tab10', alpha=0.7, s=20)
    ax2.set_title(f't-SNE: Cluster Assignment\n{param_name} (Epoch {epoch})', fontsize=11)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    
    # Add colorbar for cluster labels
    cbar2 = plt.colorbar(scatter2, ax=ax2, ticks=range(n_clusters))
    cbar2.set_label('Cluster ID')
    
    # Plot 3: Colored by model predictions
    scatter3 = ax3.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                          c=model_predictions, cmap='tab10', alpha=0.7, s=20)
    ax3.set_title(f't-SNE: Model Predictions\n{param_name} (Epoch {epoch})', fontsize=11)
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    
    # Add colorbar for model predictions
    cbar3 = plt.colorbar(scatter3, ax=ax3, ticks=range(10))
    cbar3.set_label('Predicted Label')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join("results", f"tsne_visualization_{param_name.replace('.', '_')}_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE visualization to: {save_path}")
    
    plt.show()
    
    # Print some statistics
    print(f"\nt-SNE Visualization Statistics:")
    print(f"  Parameter: {param_name}")
    print(f"  Epoch: {epoch}")
    print(f"  Samples used: {len(sample_gradients)}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  t-SNE perplexity: {perplexity}")
    
    # Calculate label-cluster correspondence
    if len(np.unique(true_labels)) > 1:  # Only if we have multiple true labels
        print(f"\nLabel-Cluster Correspondence:")
        for cluster_id in range(n_clusters):
            cluster_mask = sample_cluster_labels == cluster_id
            if np.any(cluster_mask):
                cluster_true_labels = true_labels[cluster_mask]
                unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
                dominant_label = unique_labels[np.argmax(counts)]
                dominant_percentage = 100 * np.max(counts) / len(cluster_true_labels)
                print(f"  Cluster {cluster_id}: Dominant true label {dominant_label} ({dominant_percentage:.1f}%)")
    
    # Calculate prediction-cluster correspondence
    if len(np.unique(model_predictions)) > 1:  # Only if we have multiple predictions
        print(f"\nPrediction-Cluster Correspondence:")
        for cluster_id in range(n_clusters):
            cluster_mask = sample_cluster_labels == cluster_id
            if np.any(cluster_mask):
                cluster_predictions = model_predictions[cluster_mask]
                unique_predictions, counts = np.unique(cluster_predictions, return_counts=True)
                dominant_prediction = unique_predictions[np.argmax(counts)]
                dominant_percentage = 100 * np.max(counts) / len(cluster_predictions)
                print(f"  Cluster {cluster_id}: Dominant prediction {dominant_prediction} ({dominant_percentage:.1f}%)")
    
    # Calculate overall prediction accuracy
    if len(np.unique(true_labels)) > 1 and len(np.unique(model_predictions)) > 1:
        accuracy = 100 * np.mean(true_labels == model_predictions)
        print(f"\nOverall Model Accuracy on Sample: {accuracy:.1f}%")

def load_true_labels_for_samples(sample_indices, epoch, data_dir="results/gradients"):
    """
    Load true labels for specific sample indices.
    
    Args:
        sample_indices: Array of sample indices
        epoch: Epoch number
        data_dir: Directory containing data
    
    Returns:
        Array of true labels or None if not found
    """
    try:
        # Try to load from saved labels file
        labels_path = os.path.join(data_dir, f"true_labels_epoch_{epoch}.pt")
        if os.path.exists(labels_path):
            all_labels = torch.load(labels_path)
            if len(all_labels) > max(sample_indices):
                return all_labels[sample_indices].numpy()
            else:
                print(f"Warning: Labels file has {len(all_labels)} labels but max index is {max(sample_indices)}")
                return None
        else:
            print(f"Labels file not found: {labels_path}")
            return None
            
    except Exception as e:
        print(f"Error loading true labels: {e}")
        return None

def load_model_predictions_for_samples(sample_indices, epoch, data_dir="results/gradients"):
    """
    Load model predictions for specific sample indices.
    
    Args:
        sample_indices: Array of sample indices
        epoch: Epoch number
        data_dir: Directory containing data
    
    Returns:
        Array of model predictions or None if not found
    """
    try:
        # Try to load from saved predictions file
        predictions_path = os.path.join(data_dir, f"model_predictions_epoch_{epoch}.pt")
        if os.path.exists(predictions_path):
            all_predictions = torch.load(predictions_path)
            max_index = max(sample_indices)
            
            if len(all_predictions) > max_index:
                return all_predictions[sample_indices].numpy()
            else:
                print(f"Warning: Predictions file has {len(all_predictions)} predictions but max index needed is {max_index}")
                print(f"Missing {max_index + 1 - len(all_predictions)} predictions")
                return None
        else:
            print(f"Predictions file not found: {predictions_path}")
            return None
            
    except Exception as e:
        print(f"Error loading model predictions: {e}")
        return None 