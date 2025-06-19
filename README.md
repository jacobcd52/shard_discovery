# MNIST MLP Training Project

This project implements a Multi-Layer Perceptron (MLP) for MNIST digit classification with comprehensive configuration management, training visualization, and data exploration capabilities.

## Project Structure

```
.
├── config.py          # Configuration file with all parameters
├── model.py           # MLP model definition
├── utils.py           # Utility functions for visualization
├── train.py           # Main training script
├── requirements.txt   # Python dependencies
└── results/           # Output directory (created automatically)
    ├── mnist_samples.png      # Sample MNIST images
    ├── training_curves.png    # Training/validation curves
    ├── predictions.png        # Model predictions visualization
    └── mnist_mlp.pth         # Trained model weights
```

## Features

- **Configurable Architecture**: All hyperparameters and model settings in `config.py`
- **Automatic Visualization**: Training curves, sample images, and predictions
- **Reproducible Training**: Fixed random seeds for consistent results
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Comprehensive Logging**: Detailed training progress and metrics

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

### Configuration

All parameters can be modified in `config.py`:

```python
class Config:
    # Data parameters
    batch_size = 64
    num_workers = 4
    
    # Model parameters
    input_size = 784  # 28x28 flattened
    hidden_sizes = [512, 256, 128]  # Hidden layer sizes
    output_size = 10  # 10 classes (0-9)
    dropout_rate = 0.2
    
    # Training parameters
    learning_rate = 0.001
    num_epochs = 20
    weight_decay = 1e-4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Output Files

After training, the following files will be created in the `results/` directory:

1. **`mnist_samples.png`**: Grid of sample MNIST images with labels
2. **`training_curves.png`**: Training and validation loss/accuracy curves
3. **`predictions.png`**: Model predictions on test samples (correct/incorrect highlighted)
4. **`mnist_mlp.pth`**: Best model weights (saved based on validation accuracy)

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

### Visualization Settings

Adjust visualization parameters:
```python
num_samples_to_visualize = 25  # More samples in visualizations
plot_dpi = 600                 # Higher resolution plots
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- numpy
- matplotlib
- Pillow

## License

This project is open source and available under the MIT License.