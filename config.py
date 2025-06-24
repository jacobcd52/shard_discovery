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
    hidden_sizes = [32]  # small for testing
    output_size = 10  # 10 classes (0-9)
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
    results_dir = 'results'
    model_save_path = 'results/mnist_mlp.pth'
    
    # Visualization parameters
    num_samples_to_visualize = 16
    plot_dpi = 300
    
    # Random seed
    random_seed = 42
    
    # Gradient collection parameters
    collect_gradients = True  # Re-enabled
    gradients_dir = 'results/gradients'  # Directory to save gradients
    save_gradients_every_epoch = False
    save_gradients_every_batch = True
    gradient_collector_type = 'efficient'  # 'efficient' or 'original'
    max_gradient_samples = 64  # For efficient collector
    
    # Fine-tuning / Steering Vector parameters
    steering_vector_dir = "results/steering_vectors"
    finetuning_steps = 10  # Number of backprop steps for fine-tuning
    finetuning_learning_rate = 1e-4 # Use a smaller learning rate for fine-tuning