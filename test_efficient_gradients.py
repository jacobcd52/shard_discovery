"""
Test script to benchmark the efficient gradient collector
"""
import torch
import torch.nn as nn
import time
import sys
from tinystories_model import TinyStoriesTransformer
from tinystories_config import TinyStoriesConfig
from gradient_collector import PerSampleGradientCollector  
from efficient_gradient_collector import EfficientPerSampleGradientCollector, benchmark_gradient_collectors

def create_test_data(batch_size=16, seq_len=128, vocab_size=2048):
    """Create test data for benchmarking"""
    # Create random input tokens
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    return input_ids, attention_mask, labels

def test_criterion(output, target):
    """Test loss function for language modeling"""
    if isinstance(output, dict):
        logits = output['logits']
    else:
        logits = output
    
    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

def main():
    print("Testing Efficient Gradient Collector")
    print("=" * 50)
    
    # Create config and model
    config = TinyStoriesConfig()
    
    # Create smaller model for faster testing
    model = TinyStoriesTransformer(
        vocab_size=config.vocab_size,
        d_model=64,  # Smaller for faster testing
        n_heads=2,
        n_layers=2,
        d_ff=256,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create test data
    batch_sizes = [8, 16, 32]
    seq_len = 128
    
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size: {batch_size}")
        print("-" * 30)
        
        input_ids, attention_mask, labels = create_test_data(batch_size, seq_len, config.vocab_size)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
            print("Using GPU")
        
        # Test original collector
        print("Testing original gradient collector...")
        start_time = time.time()
        
        try:
            original_collector = PerSampleGradientCollector(model, 'test_gradients_original')
            original_collector.collect_gradients_for_batch(input_ids, labels, test_criterion)
            original_time = time.time() - start_time
            original_collector.clear_gradients()
            print(f"Original collector: {original_time:.3f}s")
        except Exception as e:
            print(f"Original collector failed: {e}")
            original_time = float('inf')
        
        # Test efficient collector
        print("Testing efficient gradient collector...")
        start_time = time.time()
        
        try:
            efficient_collector = EfficientPerSampleGradientCollector(
                model, 
                'test_gradients_efficient',
                max_samples_per_collection=batch_size
            )
            efficient_collector.collect_gradients_for_batch(input_ids, labels, test_criterion)
            efficient_time = time.time() - start_time
            efficient_collector.clear_gradients()
            print(f"Efficient collector: {efficient_time:.3f}s")
            
            # Show memory usage
            memory_usage = efficient_collector.get_memory_usage()
            print(f"Memory usage: {memory_usage.get('total', 0):.2f} MB")
            
        except Exception as e:
            print(f"Efficient collector failed: {e}")
            efficient_time = float('inf')
        
        # Calculate speedup
        if efficient_time > 0 and original_time < float('inf'):
            speedup = original_time / efficient_time
            print(f"Speedup: {speedup:.1f}x")
        elif efficient_time > 0:
            print("Speedup: Original failed, efficient succeeded!")
        else:
            print("Both methods failed")
    
    print("\n" + "=" * 50)
    print("Testing with different torch.func availability...")
    
    # Test memory-efficient mode
    print("\nTesting memory-efficient gradient collection...")
    input_ids, attention_mask, labels = create_test_data(16, seq_len, config.vocab_size)
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        labels = labels.cuda()
    
    efficient_collector = EfficientPerSampleGradientCollector(model, 'test_gradients_memory_efficient')
    
    start_time = time.time()
    efficient_collector.collect_gradients_memory_efficient(input_ids, labels, test_criterion, max_samples=8)
    memory_efficient_time = time.time() - start_time
    
    print(f"Memory-efficient mode: {memory_efficient_time:.3f}s for 8 samples")
    
    # Clean up
    import shutil
    shutil.rmtree('test_gradients_original', ignore_errors=True)
    shutil.rmtree('test_gradients_efficient', ignore_errors=True)
    shutil.rmtree('test_gradients_memory_efficient', ignore_errors=True)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 