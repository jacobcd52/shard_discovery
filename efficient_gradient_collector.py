"""
Enhanced gradient collection for transformer models with token awareness and progressive batch saving.
This module provides memory-efficient gradient collection with detailed token tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import pickle
import glob
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, Callable
from tqdm import tqdm
import gc


class TokenAwareGradientCollector:
    """Enhanced gradient collector that saves data progressively in batches with token tracking"""
    
    def __init__(self, model, tokenizer, save_dir, token_context_window=5, save_batch_size=50, use_gpu=True):
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.token_context_window = token_context_window
        self.save_batch_size = save_batch_size
        self.device = model.device if use_gpu else torch.device('cpu')
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Storage for current batch
        self.gradients = defaultdict(list)
        self.token_data = []
        self.sample_count = 0
        self.batch_count = 0
        
        # Create batch directory
        self.batch_dir = os.path.join(save_dir, 'batches')
        os.makedirs(self.batch_dir, exist_ok=True)
        
        # Get parameter names
        self.param_names = [name for name, param in model.named_parameters() if param.requires_grad]
        print(f"Tracking {len(self.param_names)} parameters")
        print(f"Will save batches of {save_batch_size} samples to {self.batch_dir}")
        print(f"Using device: {self.device} (GPU: {self.use_gpu})")
    
    def collect_gradients(self, texts, max_length=128):
        """Collect gradients for a batch of texts"""
        # Tokenize texts
        encoding = self.tokenizer(
            texts, 
            max_length=max_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].to(self.model.device)
        attention_mask = encoding['attention_mask'].to(self.model.device)
        
        batch_size = input_ids.size(0)
        
        # Collect gradients for each sample
        for i in range(batch_size):
            sample_input = input_ids[i:i+1]
            sample_mask = attention_mask[i:i+1]
            
            # Create labels (next token prediction)
            sample_labels = sample_input.clone()
            sample_labels[:, :-1] = sample_input[:, 1:]
            sample_labels[:, -1] = self.tokenizer.eos_token_id
            
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(sample_input, attention_mask=sample_mask)
            logits = outputs.logits
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                sample_labels.view(-1), 
                ignore_index=-100
            )
            
            # Backward pass
            loss.backward()
            
            # Collect gradients
            sample_gradients = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if self.use_gpu:
                        sample_gradients[name] = param.grad.clone().to(self.device)
                    else:
                        sample_gradients[name] = param.grad.clone().cpu().float()
            
            # Store gradients
            for name in self.param_names:
                if name in sample_gradients:
                    self.gradients[name].append(sample_gradients[name])
            
            # Store token information
            self._store_token_info(texts[i], sample_input[0].cpu(), self.sample_count)
            
            self.sample_count += 1
            
            # Save batch if we've reached the batch size
            if len(self.token_data) >= self.save_batch_size:
                self._save_current_batch()
            
            if self.sample_count % 25 == 0:
                print(f"Processed {self.sample_count} samples (saved {self.batch_count} batches)")
    
    def _store_token_info(self, original_text, token_ids, sample_idx):
        """Store token information with context"""
        tokens = token_ids.numpy()
        seq_len = len(tokens)
        
        token_contexts = []
        for pos in range(seq_len):
            start_pos = max(0, pos - self.token_context_window)
            end_pos = min(seq_len, pos + self.token_context_window + 1)
            
            context_tokens = tokens[start_pos:end_pos]
            context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=False)
            
            token_info = {
                'position': pos,
                'token_id': int(tokens[pos]),
                'token_text': self.tokenizer.decode([tokens[pos]], skip_special_tokens=False),
                'context_text': context_text,
                'target_position': pos - start_pos
            }
            token_contexts.append(token_info)
        
        sample_data = {
            'sample_idx': sample_idx,
            'original_text': original_text,
            'token_contexts': token_contexts,
            'sequence_length': seq_len
        }
        
        self.token_data.append(sample_data)
    
    def _save_current_batch(self):
        """Save current batch to disk and clear memory"""
        if not self.gradients or not self.token_data:
            return
        
        batch_file = os.path.join(self.batch_dir, f'batch_{self.batch_count:04d}')
        
        # Save gradients
        gradient_tensors = {}
        for name, grad_list in self.gradients.items():
            if grad_list:
                gradient_tensors[name] = torch.stack(grad_list, dim=0)
        
        gradients_file = f'{batch_file}_gradients.pkl'
        with open(gradients_file, 'wb') as f:
            pickle.dump(gradient_tensors, f)
        
        # Save token data
        tokens_file = f'{batch_file}_tokens.json'
        with open(tokens_file, 'w') as f:
            json.dump(self.token_data, f)
        
        print(f"💾 Saved batch {self.batch_count} with {len(self.token_data)} samples")
        
        # Clear memory
        self.gradients.clear()
        self.token_data.clear()
        self.batch_count += 1
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def finalize_collection(self):
        """Save any remaining data and finalize collection"""
        if self.gradients or self.token_data:
            self._save_current_batch()
        
        print(f"✅ Collection complete! Saved {self.batch_count} batches with {self.sample_count} total samples")
        return self.batch_count
    
    def load_all_batches(self):
        """Load and concatenate all saved batches"""
        print("📂 Loading all saved batches...")
        
        # Find all batch files
        gradient_files = sorted(glob.glob(os.path.join(self.batch_dir, '*_gradients.pkl')))
        token_files = sorted(glob.glob(os.path.join(self.batch_dir, '*_tokens.json')))
        
        if not gradient_files:
            print("❌ No batch files found!")
            return None, None
        
        print(f"Found {len(gradient_files)} gradient batches and {len(token_files)} token batches")
        
        # Load and concatenate gradients
        all_gradients = {}
        for grad_file in gradient_files:
            with open(grad_file, 'rb') as f:
                batch_grads = pickle.load(f)
                for param_name, grads in batch_grads.items():
                    if param_name not in all_gradients:
                        all_gradients[param_name] = []
                    all_gradients[param_name].append(grads)
        
        # Concatenate gradient tensors
        final_gradients = {}
        for param_name, grad_list in all_gradients.items():
            final_gradients[param_name] = torch.cat(grad_list, dim=0)
            print(f"  {param_name}: {final_gradients[param_name].shape}")
        
        # Load and concatenate token data
        all_token_data = []
        for token_file in token_files:
            with open(token_file, 'r') as f:
                batch_tokens = json.load(f)
                all_token_data.extend(batch_tokens)
        
        print(f"✅ Loaded {len(final_gradients)} parameters and {len(all_token_data)} token samples")
        
        return final_gradients, all_token_data
    
    def analyze_diversity_from_batches(self):
        """Analyze gradient diversity from saved batches"""
        print("📊 Analyzing gradient diversity from saved batches...")
        
        gradient_files = sorted(glob.glob(os.path.join(self.batch_dir, '*_gradients.pkl')))
        if not gradient_files:
            return {'error': 'No batch files found'}
        
        # Sample gradients from multiple batches for diversity analysis
        sample_gradients = {}
        samples_per_batch = 10  # Sample fewer from each batch for memory efficiency
        
        for grad_file in gradient_files[:5]:  # Check first 5 batches
            with open(grad_file, 'rb') as f:
                batch_grads = pickle.load(f)
                for param_name, grads in batch_grads.items():
                    if param_name not in sample_gradients:
                        sample_gradients[param_name] = []
                    
                    # Sample a few gradients from this batch
                    n_samples = min(samples_per_batch, grads.size(0))
                    indices = torch.randperm(grads.size(0))[:n_samples]
                    sample_gradients[param_name].append(grads[indices])
        
        # Analyze diversity
        diversity_stats = {}
        for param_name, grad_list in sample_gradients.items():
            if not grad_list:
                continue
                
            grads = torch.cat(grad_list, dim=0)
            flat_grads = grads.view(grads.size(0), -1)
            
            # Compute similarities
            similarities = torch.cosine_similarity(
                flat_grads.unsqueeze(1), flat_grads.unsqueeze(0), dim=2
            )
            
            # Remove diagonal
            mask = ~torch.eye(similarities.size(0), dtype=torch.bool)
            off_diag = similarities[mask]
            
            diversity_stats[param_name] = {
                'num_samples': grads.size(0),
                'mean_similarity': float(off_diag.mean()),
                'std_similarity': float(off_diag.std()),
                'high_similarity_count': int((off_diag > 0.95).sum()),
                'gradient_norm_mean': float(torch.norm(flat_grads, dim=1).mean()),
                'gradient_norm_std': float(torch.norm(flat_grads, dim=1).std())
            }
        
        return diversity_stats
    
    def get_memory_usage(self):
        """Get memory usage statistics"""
        if self.use_gpu and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            gpu_reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            print(f"GPU Memory - Allocated: {gpu_memory:.2f} GB, Reserved: {gpu_reserved:.2f} GB")
            
            # Count gradient memory usage
            total_elements = 0
            for name, grad_list in self.gradients.items():
                for grad in grad_list:
                    total_elements += grad.numel()
            
            # Estimate gradient memory (assuming float32)
            gradient_memory_gb = total_elements * 4 / 1024**3
            print(f"Estimated gradient storage: {gradient_memory_gb:.2f} GB ({total_elements:,} elements)")
            
            return {
                'gpu_allocated_gb': gpu_memory,
                'gpu_reserved_gb': gpu_reserved,
                'gradient_storage_gb': gradient_memory_gb,
                'total_elements': total_elements
            }
        else:
            return {'gpu_memory': 'Not using GPU'}
    
    def clear_gradients(self):
        """Clear stored gradients to free memory"""
        self.gradients.clear()
        self.sample_count = 0
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 Cleared gradients and GPU cache")


def load_single_layer_gradients(collector, target_layer=None):
    """
    Load gradients for only one layer with progress bar
    
    Args:
        collector: The gradient collector instance
        target_layer: Name of the layer to load (e.g., "transformer.wte.weight")
                     If None, will show available layers
    
    Returns:
        gradient_tensors: Dict with layer name and gradients
        token_data: List of token data if available
    """
    save_dir = collector.save_dir
    
    # Find all gradient files
    gradient_files = []
    token_files = []
    
    print("🔍 Scanning for gradient files...")
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if file.endswith('_gradients.pt'):
                gradient_files.append(os.path.join(root, file))
            elif file.endswith('_tokens.json'):
                token_files.append(os.path.join(root, file))
    
    if not gradient_files:
        print("❌ No gradient files found!")
        return None, None
    
    print(f"Found {len(gradient_files)} gradient files total")
    
    # Show available layers if no target specified
    if target_layer is None:
        print("\n📋 Available layers:")
        layers = set()
        for f in gradient_files:
            basename = os.path.basename(f)
            layer_name = basename.replace('_gradients.pt', '').replace('_', '.')
            layers.add(layer_name)
        
        for i, layer in enumerate(sorted(layers)):
            print(f"  {i+1}. {layer}")
        
        print("\n💡 Specify target_layer parameter to load a specific layer")
        print("   Example: load_single_layer_gradients(collector, 'transformer.wte.weight')")
        return None, None
    
    # Filter for target layer
    target_files = []
    target_key = target_layer.replace('.', '_')
    
    for f in gradient_files:
        basename = os.path.basename(f)
        if target_key in basename:
            target_files.append(f)
    
    if not target_files:
        print(f"❌ No files found for layer: {target_layer}")
        print("Available layers:")
        layers = set()
        for f in gradient_files:
            basename = os.path.basename(f)
            layer_name = basename.replace('_gradients.pt', '').replace('_', '.')
            layers.add(layer_name)
        for layer in sorted(layers):
            print(f"  - {layer}")
        return None, None
    
    print(f"📁 Found {len(target_files)} files for layer: {target_layer}")
    
    # Load gradients with progress bar
    all_gradients = []
    all_token_data = []
    
    print(f"💾 Loading gradients for {target_layer}...")
    for grad_file in tqdm(target_files, desc="Loading batches", unit="file"):
        try:
            gradients = torch.load(grad_file, map_location='cpu')
            all_gradients.append(gradients)
            
            # Try to load corresponding token data
            token_file = grad_file.replace('_gradients.pt', '_tokens.json')
            # Check if it's in a tokens subdirectory
            tokens_dir = os.path.join(os.path.dirname(grad_file), 'tokens')
            token_file_alt = os.path.join(tokens_dir, os.path.basename(token_file))
            
            token_file_to_use = None
            if os.path.exists(token_file):
                token_file_to_use = token_file
            elif os.path.exists(token_file_alt):
                token_file_to_use = token_file_alt
            
            if token_file_to_use:
                with open(token_file_to_use, 'r') as f:
                    token_data = json.load(f)
                    all_token_data.extend(token_data)
            
        except Exception as e:
            print(f"❌ Error loading {grad_file}: {e}")
            continue
    
    if not all_gradients:
        print("❌ No gradients loaded successfully!")
        return None, None
    
    # Concatenate all gradients
    print("🔗 Concatenating gradients...")
    concatenated_gradients = torch.cat(all_gradients, dim=0)
    
    # Create result dictionary
    gradient_tensors = {target_layer: concatenated_gradients}
    
    return gradient_tensors, all_token_data


def analyze_gradient_diversity(gradients: Dict[str, torch.Tensor], 
                             device: Optional[torch.device] = None) -> Dict[str, Dict]:
    """Analyze gradient diversity across samples"""
    diversity_stats = {}
    
    for name, grad_tensor in gradients.items():
        if grad_tensor.numel() == 0:
            continue
            
        grads = grad_tensor
        flat_grads = grads.view(grads.size(0), -1)
        
        # Move to specified device if needed
        if device is not None and flat_grads.device != device:
            flat_grads = flat_grads.to(device)
        
        # Compute similarities
        similarities = torch.cosine_similarity(
            flat_grads.unsqueeze(1), flat_grads.unsqueeze(0), dim=2
        )
        
        # Remove diagonal
        mask = ~torch.eye(similarities.size(0), dtype=torch.bool, device=similarities.device)
        off_diag = similarities[mask]
        
        diversity_stats[name] = {
            'num_samples': grads.size(0),
            'mean_similarity': float(off_diag.mean()),
            'std_similarity': float(off_diag.std()),
            'high_similarity_count': int((off_diag > 0.95).sum()),
            'gradient_norm_mean': float(torch.norm(flat_grads, dim=1).mean()),
            'gradient_norm_std': float(torch.norm(flat_grads, dim=1).std())
        }
    
    return diversity_stats


def collect_gradients_from_stories(model, tokenizer, stories, save_dir, 
                                 max_samples=2500, max_length=128, batch_size=8, 
                                 save_batch_size=50, token_context_window=5):
    """
    Convenient function to collect gradients from a list of stories
    
    Args:
        model: The model to collect gradients from
        tokenizer: Tokenizer for the model
        stories: List of text stories
        save_dir: Directory to save gradients
        max_samples: Maximum number of samples to process
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        save_batch_size: Number of samples per saved batch
        token_context_window: Context window for token tracking
    
    Returns:
        collector: The gradient collector instance
    """
    # Initialize collector
    collector = TokenAwareGradientCollector(
        model, tokenizer, save_dir, token_context_window, save_batch_size
    )
    
    print("🔄 Starting gradient collection with progressive batch saving...")
    
    # Process stories in batches
    for i in range(0, min(len(stories), max_samples), batch_size):
        batch_stories = stories[i:i+batch_size]
        collector.collect_gradients(batch_stories, max_length)
        
        if i % (batch_size * 10) == 0:
            print(f"Processed {i + len(batch_stories)} / {min(len(stories), max_samples)} stories")
    
    # Finalize collection
    num_batches = collector.finalize_collection()
    
    print(f"✅ Gradient collection complete!")
    print(f"Total samples processed: {collector.sample_count}")
    print(f"Total batches saved: {num_batches}")
    print(f"Data saved to: {collector.batch_dir}")
    
    return collector