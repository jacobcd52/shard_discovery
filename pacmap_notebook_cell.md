# PaCMAP Visualization Cell (add this to your notebook)
# ======================================================

```python
# Import PaCMAP visualization function
from cluster_visualization import visualize_pacmap_gradients
from config import Config

# Create PaCMAP visualization using the clustering results
print("🎯 Creating PaCMAP Visualization")
print("=" * 40)

# Load config to check for digit filtering
config = Config()
if hasattr(config, 'filter_digits') and config.filter_digits is not None:
    print(f"Digit filtering detected: {config.filter_digits}")
    print("PaCMAP will show original digits for better visualization")
else:
    print("No digit filtering - showing all digits")

# Use the same parameter and clustering results from previous cells
print(f"Creating PaCMAP visualization for: {param_name}")
print(f"Using clustering results with {clustering_results['n_clusters']} clusters")

# PaCMAP visualization with filtering options
# Filtering parameters (adjust these as needed)
start_fraction = 0.8    # Start from 80% through training (skip early samples)
end_fraction = 1.0      # Use samples up to 100% (end of training)
min_grad_percentile = 0.5  # Only use samples with gradient norm >= 0.5th percentile

print(f"PaCMAP filtering settings:")
print(f"  Training position: {start_fraction:.1%} to {end_fraction:.1%}")
print(f"  Gradient percentile: >= {min_grad_percentile:.1f}%")

visualize_pacmap_gradients(
    clustering_results=clustering_results,
    param_name=param_name,
    epoch=EPOCH_TO_LOAD,
    data_dir=GRADIENTS_DIR,
    n_samples=5000,          # Use 5000 samples for good performance
    n_neighbors=10,          # PaCMAP parameter: number of neighbors (default: 10)
    MN_ratio=0.5,            # PaCMAP parameter: mid-near ratio (default: 0.5)
    FP_ratio=2.0,            # PaCMAP parameter: further pair ratio (default: 2.0)
    random_state=42,         # For reproducibility
    figsize=(20, 6),         # Wide figure to show all three plots
    config=config,           # Pass config for filtering info
    start_fraction=start_fraction,        # Filter by training position
    end_fraction=end_fraction,            # Filter by training position
    min_grad_percentile=min_grad_percentile,  # Filter by gradient norm
    normalize_gradients=True  # Set to False to use original gradient magnitudes
)

print("\n✅ PaCMAP visualization completed!")
print("The visualization shows:")
print("  - Left: Points colored by true digit labels (original digits if filtered)")
print("  - Middle: Points colored by k-means cluster assignments") 
print("  - Right: Points colored by model predictions")
print("\nNote: PaCMAP parameters can be adjusted:")
print("  - n_neighbors: Number of neighbors to consider (try 5-50)")
print("  - MN_ratio: Mid-near pair ratio - higher preserves local structure (try 0.1-1.0)")
print("  - FP_ratio: Further pair ratio - higher preserves global structure (try 1.0-5.0)")
print("  - normalize_gradients: Set to False to preserve gradient magnitudes")

# Optional: Parameter comparison
print("\n" + "="*60)
print("Optional: Compare different PaCMAP parameter settings")
print("="*60)

# You can uncomment and run this section to compare different parameters:
"""
# Compare local vs global structure preservation
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# More local structure (higher MN_ratio, lower FP_ratio)
visualize_pacmap_gradients(
    clustering_results=clustering_results,
    param_name=param_name,
    epoch=EPOCH_TO_LOAD,
    data_dir=GRADIENTS_DIR,
    n_samples=3000,
    MN_ratio=0.8,    # Higher - more local
    FP_ratio=1.0,    # Lower - less global
    figsize=(8, 6)
)

# Balanced (default parameters)
visualize_pacmap_gradients(
    clustering_results=clustering_results,
    param_name=param_name,
    epoch=EPOCH_TO_LOAD,
    data_dir=GRADIENTS_DIR,
    n_samples=3000,
    MN_ratio=0.5,    # Default
    FP_ratio=2.0,    # Default
    figsize=(8, 6)
)

# More global structure (lower MN_ratio, higher FP_ratio)
visualize_pacmap_gradients(
    clustering_results=clustering_results,
    param_name=param_name,
    epoch=EPOCH_TO_LOAD,
    data_dir=GRADIENTS_DIR,
    n_samples=3000,
    MN_ratio=0.2,    # Lower - less local
    FP_ratio=3.0,    # Higher - more global
    figsize=(8, 6)
)
"""
```

## Installation
To use PaCMAP, install it with:
```bash
pip install pacmap
```

## PaCMAP vs t-SNE vs UMAP

| Method | Strengths | Best For |
|--------|-----------|----------|
| **t-SNE** | Good local structure | Dense, well-separated clusters |
| **UMAP** | Balance local/global, faster | General-purpose, large datasets |
| **PaCMAP** | Best local+global preservation | When you need both fine details and overall structure |

## Key PaCMAP Parameters

- **`n_neighbors`** (default: 10): Number of neighbors to consider
  - Lower (5-8): Focus on very local structure
  - Higher (15-30): More global awareness

- **`MN_ratio`** (default: 0.5): Mid-near pair ratio
  - Higher (0.7-1.0): Preserves more local structure
  - Lower (0.2-0.4): Allows more global optimization

- **`FP_ratio`** (default: 2.0): Further pair ratio  
  - Higher (3.0-5.0): Better global structure preservation
  - Lower (1.0-1.5): Focus more on local neighborhoods 