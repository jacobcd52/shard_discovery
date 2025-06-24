"""
Configuration file for TinyStories transformer training
"""
import torch
import math

class TinyStoriesConfig:
    # Model parameters (targeting ~1M parameters)
    vocab_size = 2048  # Will be set based on tokenizer
    d_model = 128  # Hidden dimension
    n_heads = 4  # Number of attention heads
    n_layers = 3  # Number of transformer layers
    d_ff = 512  # Feed-forward dimension (4 * d_model)
    max_seq_len = 512  # Maximum sequence length
    dropout = 0.1
    
    # Data parameters
    batch_size = 1024
    max_length = 256
    num_workers = 16
    
    # Training parameters
    learning_rate = 1e-4  # Much smaller learning rate to prevent NaN
    num_epochs = 2
    warmup_steps = 250
    weight_decay = 0.01
    grad_clip_norm = 0.5  # More aggressive gradient clipping
    
    # Device and precision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16  # Use float32 for stability
    use_amp = True  # Disable mixed precision for stability
    
    # Paths
    results_dir = 'tinystories_results'
    model_save_path = 'tinystories_results/tinystories_transformer.pth'
    tokenizer_save_path = 'tinystories_results/tokenizer'
    
    # Gradient collection parameters
    collect_gradients = True
    gradients_dir = 'tinystories_results/gradients'
    gradient_collection_frequency = 10  # Collect gradients every N steps (much more frequent now!)
    max_gradient_samples = 64  # Maximum samples per gradient collection (can handle larger batches now)
    gradient_collector_type = 'efficient'  # 'efficient' or 'transformer' or 'original'
    collect_attention_only = False  # Only collect attention layer gradients (much faster)
    
    # Logging and saving
    log_every = 100
    save_every = 2000
    eval_every = 1000
    
    # Random seed
    random_seed = 42
    
    @classmethod
    def calculate_parameters(cls):
        """Calculate approximate number of parameters"""
        # Embedding layer: vocab_size * d_model
        embedding_params = cls.vocab_size * cls.d_model
        
        # Transformer layers
        # Each layer has:
        # - Self-attention: 4 * d_model^2 (Q, K, V, O projections)
        # - Feed-forward: 2 * d_model * d_ff
        # - Layer norms: 2 * d_model (small, can ignore)
        attention_params_per_layer = 4 * cls.d_model * cls.d_model
        ff_params_per_layer = 2 * cls.d_model * cls.d_ff
        params_per_layer = attention_params_per_layer + ff_params_per_layer
        transformer_params = cls.n_layers * params_per_layer
        
        # Output projection (if not tied with input embedding)
        output_params = cls.vocab_size * cls.d_model
        
        total_params = embedding_params + transformer_params + output_params
        
        print(f"Estimated parameters:")
        print(f"  Embedding: {embedding_params:,}")
        print(f"  Transformer ({cls.n_layers} layers): {transformer_params:,}")
        print(f"  Output: {output_params:,}")
        print(f"  Total: {total_params:,}")
        print(f"  Target: ~1M parameters")
        
        return total_params 