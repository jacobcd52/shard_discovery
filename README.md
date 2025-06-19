# MNIST MLP Training Project

This project implements a Multi-Layer Perceptron (MLP) for MNIST digit classification with comprehensive configuration management, training visualization, data exploration capabilities, and **per-sample gradient collection**.

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

## Memory Considerations

- **Gradient Collection**: Can be memory-intensive, especially with large models
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