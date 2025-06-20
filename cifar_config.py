"""
Configuration file for CIFAR-10 CNN training
"""
import torch

class CIFARConfig:
    # Jacob: added this, idk what it does
    filter_digits = True
    
    # Data parameters
    batch_size = 128
    num_workers = 4
    
    # Model architecture parameters
    # Convolutional layers
    conv_channels = [3, 16, 32, 32]  # Input channels for each conv layer (3->16)
    conv_kernel_sizes = [3, 3, 3, 3]   # Kernel sizes for conv layers
    conv_padding = [1, 1, 1, 1]        # Padding for conv layers
    
    # Pooling
    pool_kernel_size = 2
    pool_stride = 2
    
    # Input image size
    input_size = 32  # CIFAR-10 images are 32x32
    
    # Fully connected layers (input_size will be calculated automatically)
    fc_hidden_sizes = [64]  # Hidden sizes for FC layers (excluding input size)
    
    # Model parameters
    num_classes = 10  # CIFAR-10 has 10 classes
    dropout_rate = 0.
    
    # Training parameters
    learning_rate = 0.001
    num_epochs = 1
    weight_decay = 1e-4
    
    # Device and precision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16  # Use bfloat16 by default
    autocast_enabled = False  # Disable autocast for bfloat16 (GradScaler doesn't support bfloat16)
    
    # Paths
    results_dir = 'cifar_results'
    model_save_path = 'cifar_results/cifar_cnn.pth'
    
    # Visualization parameters
    num_samples_to_visualize = 16
    plot_dpi = 300
    
    # Random seed
    random_seed = 42
    
    # Gradient collection parameters
    collect_gradients = True
    gradients_dir = 'cifar_results/gradients'  # Directory to save gradients
    save_gradients_every_epoch = False
    save_gradients_every_batch = True
    
    def calculate_feature_size(self):
        """Calculate the size of features after conv+pool layers"""
        # Start with input size
        current_size = self.input_size
        
        # Apply each conv+pool operation
        for i in range(len(self.conv_channels) - 1):
            # Conv layer doesn't change spatial size (with padding=1)
            # Pool layer reduces size by pool_stride
            current_size = current_size // self.pool_stride
        
        # Final feature map size
        final_channels = self.conv_channels[-1]
        feature_size = current_size * current_size * final_channels
        
        return feature_size
    
    @property
    def fc_sizes(self):
        """Get FC layer sizes with automatically calculated input size"""
        feature_size = self.calculate_feature_size()
        return [feature_size] + self.fc_hidden_sizes 