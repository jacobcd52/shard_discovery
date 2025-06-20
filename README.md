# MNIST MLP Training Project

This project implements a Multi-Layer Perceptron (MLP) for MNIST digit classification with comprehensive configuration management, training visualization, data exploration capabilities, **per-sample gradient collection**, and **digit filtering** for studying clustering patterns on subsets of MNIST digits.

## Project Structure

```
.
├── config.py              # Configuration file with all parameters
├── model.py               # MLP model definition
├── utils.py               # Utility functions for visualization
├── gradient_collector.py  # Per-sample gradient collection utilities
├── train.py               # Main training script
├── test_gradients.py      # Test script for gradient collection
├── requirements.txt       # Python dependencies
└── results/               # Output directory (created automatically)
    ├── mnist_samples.png      # Sample MNIST images
    ├── training_curves.png    # Training/validation curves
    ├── predictions.png        # Model predictions visualization
    ├── mnist_mlp.pth         # Trained model weights
    └── gradients/             # Per-sample gradients (if enabled)
        └── epoch_1/
            ├── network.0.weight_epoch_1.pt
            ├── network.1.weight_epoch_1.pt
            ├── network.2.weight_epoch_1.pt
            ├── metadata.pkl
            └── gradient_stats.pkl
```

## Features

- **Configurable Architecture**: All hyperparameters and model settings in `config.py`
- **Automatic Visualization**: Training curves, sample images, and predictions
- **Reproducible Training**: Fixed random seeds for consistent results
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Comprehensive Logging**: Detailed training progress and metrics
- **Per-Sample Gradient Collection**: Save gradients for each individual training sample
- **🆕 Digit Filtering**: Train on subsets of MNIST digits to study clustering patterns
- **🆕 t-SNE Visualization**: Advanced gradient clustering visualization with t-SNE
- **🆕 GPU-Accelerated K-means**: Fast clustering analysis of gradient data

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the training script:
```bash
python train.py
```

This will:
1. Download MNIST dataset (if not already present)
2. Create the MLP model with specified architecture
3. Train for the configured number of epochs
4. Generate visualizations in the `results/` folder
5. Save the best model weights
6. **Collect and save per-sample gradients** (if enabled)

### Gradient Collection

The project includes a sophisticated per-sample gradient collection system:

#### Configuration

Enable gradient collection in `config.py`:
```python
# Gradient collection parameters
collect_gradients = True  # Whether to collect per-sample gradients
gradients_dir = 'results/gradients'  # Directory to save gradients
save_gradients_every_epoch = True  # Save gradients after each epoch
save_gradients_every_batch = False  # Save gradients after each batch (memory intensive)
```

#### How It Works

1. **Per-Sample Processing**: Each sample in a batch is processed individually to compute separate gradients
2. **Gradient Storage**: Gradients are stored as tensors with shape `[num_samples, *parameter_shape]`
3. **Memory Management**: Gradients are moved to CPU and detached to save memory
4. **Statistics Computation**: Automatic computation of gradient statistics (norms, means, etc.)

#### Output Format

For each model parameter, gradients are saved as:
- **File**: `{parameter_name}_epoch_{epoch}.pt`
- **Content**: Dictionary containing:
  - `gradients`: Tensor of shape `[num_samples, *parameter_shape]`
  - `param_shape`: Original parameter shape
  - `num_samples`: Number of samples processed

#### Testing Gradient Collection

Test the gradient collection functionality:
```bash
python test_gradients.py
```

### Configuration

All parameters can be modified in `config.py`:

```python
class Config:
    # Data parameters
    batch_size = 64
    num_workers = 4
    
    # Model parameters
    input_size = 784  # 28x28 flattened
    hidden_sizes = [512]  # Hidden layer sizes
    output_size = 10  # 10 classes (0-9)
    dropout_rate = 0.
    
    # Training parameters
    learning_rate = 0.001
    num_epochs = 1
    weight_decay = 1e-4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Gradient collection parameters
    collect_gradients = True
    gradients_dir = 'results/gradients'
    save_gradients_every_epoch = True
    save_gradients_every_batch = False
```

### Output Files

After training, the following files will be created in the `results/` directory:

1. **`mnist_samples.png`**: Grid of sample MNIST images with labels
2. **`training_curves.png`**: Training and validation loss/accuracy curves
3. **`predictions.png`**: Model predictions on test samples (correct/incorrect highlighted)
4. **`mnist_mlp.pth`**: Best model weights (saved based on validation accuracy)
5. **`gradients/`**: Per-sample gradients (if enabled)
   - `epoch_1/`: Gradients for each epoch
   - `metadata.pkl`: Information about collected gradients
   - `gradient_stats.pkl`: Statistical analysis of gradients

## Model Architecture

The MLP consists of:
- Input layer: 784 neurons (28×28 flattened images)
- Hidden layers: Configurable sizes with ReLU activation and dropout
- Output layer: 10 neurons (one per digit class)
- Loss function: Cross-entropy loss
- Optimizer: Adam with weight decay

## Expected Performance

With the default configuration:
- Training accuracy: ~99%+
- Test accuracy: ~97-98%
- Training time: ~5-10 minutes on CPU, ~1-2 minutes on GPU
- **Gradient storage**: ~100-500MB per epoch (depending on model size and dataset)

## Customization

### Changing Model Architecture

Modify the `hidden_sizes` parameter in `config.py`:
```python
hidden_sizes = [1024, 512, 256, 128]  # Deeper network
hidden_sizes = [256, 128]             # Smaller network
```

### Adjusting Training Parameters

Modify training parameters in `config.py`:
```python
learning_rate = 0.0001  # Lower learning rate
num_epochs = 50         # More training epochs
batch_size = 128        # Larger batch size
```

### Gradient Collection Settings

Adjust gradient collection behavior:
```python
collect_gradients = False              # Disable gradient collection
save_gradients_every_batch = True      # Save after each batch (memory intensive)
gradients_dir = 'custom_gradients/'    # Custom save directory
```

### Visualization Settings

Adjust visualization parameters:
```python
num_samples_to_visualize = 25  # More samples in visualizations
plot_dpi = 600                 # Higher resolution plots
```

## 🆕 Digit Filtering

The project now supports training on subsets of MNIST digits to study how gradient clustering patterns change when certain digits are removed.

### Quick Start with Digit Filtering

1. **Run the demo script** to see available filtering options:
```bash
python demo_digit_filtering.py
```

2. **Choose a filtering configuration** (e.g., binary classification with digits 0 and 1):
```python
# In config.py
filter_digits = [0, 1]  # Only use digits 0 and 1
```

3. **Generate filtered labels**:
```bash
python generate_true_labels.py
```

4. **Train the model**:
```bash
python train.py
```

5. **Analyze results** in `inspect_gradients.ipynb` - the t-SNE visualization will automatically handle filtered datasets!

### Configuration Examples

#### Binary Classification (Digits 0 and 1)
```python
class Config:
    filter_digits = [0, 1]  # Binary classification
    # Model output will be automatically adjusted to 2 classes
```

#### Even Digits Only
```python
class Config:
    filter_digits = [0, 2, 4, 6, 8]  # Even digits
    # Creates 5-class problem: 0→0, 2→1, 4→2, 6→3, 8→4
```

#### Prime Digits
```python
class Config:
    filter_digits = [2, 3, 5, 7]  # Prime digits
    # Creates 4-class problem: 2→0, 3→1, 5→2, 7→3
```

#### Similar Digits (Challenging)
```python
class Config:
    filter_digits = [6, 8, 9]  # Visually similar digits
    # Tests if gradients can distinguish similar shapes
```

### How It Works

1. **Automatic Dataset Filtering**: The `FilteredMNIST` class automatically filters the training and test datasets
2. **Label Remapping**: Original digit labels are remapped to a continuous range (0, 1, 2, ...)
3. **Model Adaptation**: The model's output layer is automatically resized based on the number of filtered digits
4. **Visualization Support**: t-SNE and clustering visualizations show both filtered labels and original digits

### Benefits for Research

- **Study Digit Relationships**: See how removing certain digits affects gradient clustering
- **Simplify Analysis**: Focus on specific digit pairs or groups
- **Test Hypotheses**: E.g., "Do visually similar digits cluster together in gradient space?"
- **Reduce Complexity**: Start with binary classification, then gradually add more digits

## 🆕 t-SNE Visualization

The project includes advanced t-SNE visualization of gradient clustering patterns.

### Features

- **Three-Panel Visualization**: 
  - Original digit labels (left)
  - K-means cluster assignments (middle) 
  - Model predictions (right)
- **Filtered Dataset Support**: Shows original digits even when training on filtered subsets
- **PCA Preprocessing**: Dimensionality reduction before t-SNE for better performance
- **Configurable Parameters**: Adjust sample size, perplexity, and random seed

### Usage

In `inspect_gradients.ipynb`, run Cell 4 after performing k-means clustering:

```python
# The t-SNE visualization will automatically:
# 1. Load gradient data
# 2. Apply PCA preprocessing
# 3. Run t-SNE dimensionality reduction
# 4. Create three-panel visualization
# 5. Save high-resolution plots
visualize_tsne_gradients(
    clustering_results=clustering_results,
    param_name="network.3.weight",
    epoch=0,
    n_samples=5000,
    perplexity=30
)
```

### Expected Results

- **Clear Cluster Separation**: t-SNE typically reveals 10 distinct clusters (or fewer with digit filtering)
- **Label-Cluster Correspondence**: Clusters often correspond to digit classes
- **Emergent Structure**: The clustering structure emerges naturally from gradient patterns

## Memory Considerations

- **Gradient Collection**: Can be memory-intensive, especially with large models
- **Digit Filtering**: Reduces memory usage by training on smaller datasets
- **Batch Processing**: Consider reducing batch size if running out of memory
- **Gradient Storage**: Gradients are automatically moved to CPU to save GPU memory
- **Cleanup**: Gradients are cleared after each epoch to free memory

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- numpy
- matplotlib
- Pillow

## License

This project is open source and available under the MIT License.