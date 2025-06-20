"""
Configuration file for MNIST MLP training
"""
import torch

class Config:
    # Data parameters
    batch_size = 64
    num_workers = 4
    
    # Digit filtering parameters
    # filter_digits = None  # List of digits to include (None = all digits)
    # Examples:
    filter_digits = [0, 1, 2, 3, 4, 5, 6, 7]  # Only digits 0-4
    # filter_digits = [0, 1]           # Only digits 0 and 1 (binary classification)
    # filter_digits = [2, 3, 5, 7]     # Only prime digits
    
    # Model parameters
    input_size = 784  # 28x28 flattened
    hidden_sizes = [32]  # small for testing
    output_size = 10  # 10 classes (0-9) - will be adjusted based on filter_digits
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
    
    # Gradient collection parameters
    collect_gradients = True  # Whether to collect per-sample gradients
    gradients_dir = 'results/gradients'  # Directory to save gradients
    save_gradients_every_epoch = False
    save_gradients_every_batch = True
    
    def get_output_size(self):
        """Get the actual output size based on digit filtering"""
        if self.filter_digits is None:
            return 10  # All digits
        else:
            return len(self.filter_digits)
    
    def get_label_mapping(self):
        """Get mapping from original labels to filtered labels"""
        if self.filter_digits is None:
            return {i: i for i in range(10)}  # Identity mapping
        else:
            return {digit: idx for idx, digit in enumerate(self.filter_digits)}
    
    def get_reverse_label_mapping(self):
        """Get mapping from filtered labels back to original labels"""
        if self.filter_digits is None:
            return {i: i for i in range(10)}  # Identity mapping
        else:
            return {idx: digit for idx, digit in enumerate(self.filter_digits)}