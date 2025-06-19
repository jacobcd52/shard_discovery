"""
Configuration file for MNIST MLP training
"""
import torch

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
    
    # Paths
    results_dir = 'results'
    model_save_path = 'results/mnist_mlp.pth'
    
    # Visualization parameters
    num_samples_to_visualize = 16
    plot_dpi = 300
    
    # Random seed
    random_seed = 42 