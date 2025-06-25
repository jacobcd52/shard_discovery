"""
Token-specific gradient collection for transformer models with efficient storage.
This module provides memory-efficient gradient collection with precise token-level attribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import gc


class TokenSpecificGradientCollector:
    """
    Token-specific gradient collector that computes gradients for individual token positions
    and saves them directly in an optimized per-layer format.
    """
    
    def __init__(self, model, tokenizer, save_dir, token_context_window=5, save_batch_size=100, use_gpu=True):
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.token_context_window = token_context_window
        self.save_batch_size = save_batch_size
        self.device = model.device if use_gpu else torch.device('cpu')
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Storage for current batch (per layer)
        self.gradients = defaultdict(list)
        self.token_data = []
        self.sample_count = 0  # Counts individual tokens
        self.batch_count = 0
        
        # Create optimized directory structure directly
        self.layers_dir = os.path.join(save_dir, 'layers')
        self.tokens_dir = os.path.join(save_dir, 'tokens')
        os.makedirs(self.layers_dir, exist_ok=True)
        os.makedirs(self.tokens_dir, exist_ok=True)
        
        # Get parameter names and create layer directories
        self.param_names = [name for name, param in model.named_parameters() if param.requires_grad]
        for param_name in self.param_names:
            layer_dir = os.path.join(self.layers_dir, param_name.replace('.', '_'))
            os.makedirs(layer_dir, exist_ok=True)
        
        print(f"🎯 Token-specific gradient collection initialized")
        print(f"Tracking {len(self.param_names)} parameters")
        print(f"Will save batches of {save_batch_size} token gradients")
        print(f"Using optimized per-layer storage format")
        print(f"Using device: {self.device} (GPU: {self.use_gpu})")
    
    def collect_gradients(self, texts, max_length=128):
        """Collect gradients for individual token positions in each text"""
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
        
        # Process each sequence
        for seq_idx in range(batch_size):
            sample_input = input_ids[seq_idx:seq_idx+1]
            sample_mask = attention_mask[seq_idx:seq_idx+1]
            
            # Get actual sequence length (excluding padding)
            seq_len = sample_mask.sum().item()
            
            # Collect gradients for each token position
            for pos in range(seq_len - 1):  # Exclude last position (no next token to predict)
                # Create target for this specific position
                target_token = sample_input[0, pos + 1].item()
                
                # Zero gradients
                self.model.zero_grad()
                
                # Forward pass
                outputs = self.model(sample_input, attention_mask=sample_mask)
                logits = outputs.logits
                
                # Compute loss for ONLY this token position
                position_logits = logits[0, pos, :]  # [vocab_size]
                loss = F.cross_entropy(position_logits.unsqueeze(0), torch.tensor([target_token], device=self.device))
                
                # Backward pass
                loss.backward()
                
                # Collect gradients
                token_gradients = {}
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if self.use_gpu:
                            token_gradients[name] = param.grad.clone().to(self.device)
                        else:
                            token_gradients[name] = param.grad.clone().cpu().float()
                
                # Store gradients for this token
                for name in self.param_names:
                    if name in token_gradients:
                        self.gradients[name].append(token_gradients[name])
                
                # Store token information for this specific position
                self._store_token_info(
                    original_text=texts[seq_idx],
                    token_ids=sample_input[0].cpu(),
                    token_position=pos,
                    target_token_id=target_token,
                    sample_idx=self.sample_count
                )
                
                self.sample_count += 1
                
                # Save batch if we've reached the batch size
                if len(self.token_data) >= self.save_batch_size:
                    self._save_current_batch()
                
                if self.sample_count % 50 == 0:
                    print(f"Processed {self.sample_count} token positions (saved {self.batch_count} batches)")
    
    def _store_token_info(self, original_text, token_ids, token_position, target_token_id, sample_idx):
        """Store information for a specific token position"""
        tokens = token_ids.numpy()
        seq_len = len(tokens)
        
        # Get the current token and target token
        current_token_id = tokens[token_position]
        current_token_text = self.tokenizer.decode([current_token_id], skip_special_tokens=False)
        target_token_text = self.tokenizer.decode([target_token_id], skip_special_tokens=False)
        
        # Create context window around this position
        start_pos = max(0, token_position - self.token_context_window)
        end_pos = min(seq_len, token_position + self.token_context_window + 1)
        
        context_tokens = tokens[start_pos:end_pos]
        context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=False)
        
        # Position of the target token within the context
        context_target_pos = token_position - start_pos
        
        token_data = {
            'sample_idx': sample_idx,
            'original_text': original_text,
            'sequence_length': seq_len,
            'token_position': token_position,
            'current_token_id': int(current_token_id),
            'current_token_text': current_token_text,
            'target_token_id': int(target_token_id),
            'target_token_text': target_token_text,
            'context_text': context_text,
            'context_start_pos': start_pos,
            'context_target_pos': context_target_pos,
            'prediction_task': f"'{current_token_text}' → '{target_token_text}'"
        }
        
        self.token_data.append(token_data)
    
    def _save_current_batch(self):
        """Save current batch directly in optimized per-layer format"""
        if not self.gradients or not self.token_data:
            return
        
        # Save each layer separately in optimized format
        for param_name, grad_list in self.gradients.items():
            if grad_list:
                # Stack gradients for this layer
                layer_gradients = torch.stack(grad_list, dim=0)
                
                # Save in layer-specific directory
                layer_dir = os.path.join(self.layers_dir, param_name.replace('.', '_'))
                layer_file = os.path.join(layer_dir, f'batch_{self.batch_count:04d}_gradients.pt')
                
                # Save as PyTorch tensor (faster than pickle)
                torch.save(layer_gradients, layer_file)
        
        # Save token data
        tokens_file = os.path.join(self.tokens_dir, f'batch_{self.batch_count:04d}_tokens.json')
        with open(tokens_file, 'w') as f:
            json.dump(self.token_data, f)
        
        print(f"💾 Saved optimized batch {self.batch_count} with {len(self.token_data)} token gradients")
        
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
        
        print(f"✅ Token-specific collection complete!")
        print(f"   Total token gradients: {self.sample_count}")
        print(f"   Saved in {self.batch_count} batches")
        print(f"   Optimized storage: {self.layers_dir}")
        return self.batch_count


def load_single_layer_gradients(collector, target_layer):
    """
    Load gradients for a single layer from optimized storage
    
    Args:
        collector: The gradient collector instance
        target_layer: Name of the layer to load
    
    Returns:
        gradient_tensors: Dict with layer name and gradients
        token_data: List of token data
    """
    save_dir = collector.save_dir
    layers_dir = os.path.join(save_dir, 'layers')
    tokens_dir = os.path.join(save_dir, 'tokens')
    
    if not os.path.exists(layers_dir):
        print("❌ No layers directory found!")
        return None, None
    
    # Show available layers if no target specified
    if target_layer is None:
        layer_dirs = [d for d in os.listdir(layers_dir) if os.path.isdir(os.path.join(layers_dir, d))]
        if layer_dirs:
            print("\n📋 Available layers:")
            for i, layer_dir in enumerate(sorted(layer_dirs)):
                layer_name = layer_dir.replace('_', '.')
                print(f"  {i+1}. {layer_name}")
        else:
            print("❌ No layer directories found!")
        
        print("\n💡 Specify target_layer parameter to load a specific layer")
        print("   Example: load_single_layer_gradients(collector, 'transformer.wte.weight')")
        return None, None
    
    # Convert layer name to directory name
    layer_dir_name = target_layer.replace('.', '_')
    layer_path = os.path.join(layers_dir, layer_dir_name)
    
    if not os.path.exists(layer_path):
        print(f"❌ Layer '{target_layer}' not found!")
        print("Available layers:")
        layer_dirs = [d for d in os.listdir(layers_dir) if os.path.isdir(os.path.join(layers_dir, d))]
        for layer_dir in sorted(layer_dirs):
            layer_name = layer_dir.replace('_', '.')
            print(f"  - {layer_name}")
        return None, None
    
    # Find gradient files for this layer
    grad_files = sorted(glob.glob(os.path.join(layer_path, '*_gradients.pt')))
    if not grad_files:
        print(f"❌ No gradient files found for layer {target_layer}")
        return None, None
    
    print(f"🎯 Loading gradients for layer: {target_layer}")
    print(f"Found {len(grad_files)} batch files")
    
    # Load gradients
    all_gradients = []
    for grad_file in tqdm(grad_files, desc="Loading batches", unit="file"):
        try:
            layer_grads = torch.load(grad_file, map_location='cpu')
            all_gradients.append(layer_grads)
        except Exception as e:
            print(f"❌ Error loading {grad_file}: {e}")
            continue
    
    if not all_gradients:
        print("❌ No gradients loaded successfully!")
        return None, None
    
    # Load token data
    token_files = sorted(glob.glob(os.path.join(tokens_dir, '*_tokens.json')))
    all_token_data = []
    
    for token_file in token_files:
        try:
            with open(token_file, 'r') as f:
                token_data = json.load(f)
                all_token_data.extend(token_data)
        except Exception as e:
            print(f"⚠️  Error loading token file {token_file}: {e}")
            continue
    
    # Concatenate gradients
    concatenated_gradients = torch.cat(all_gradients, dim=0)
    gradient_tensors = {target_layer: concatenated_gradients}
    
    print(f"✅ Loaded {concatenated_gradients.shape[0]} token gradients for layer {target_layer}")
    print(f"   Shape: {concatenated_gradients.shape}")
    memory_mb = concatenated_gradients.nelement() * concatenated_gradients.element_size() / (1024**2)
    print(f"   Memory: {memory_mb:.1f} MB")
    
    return gradient_tensors, all_token_data


def check_existing_gradients(save_dir):
    """
    Check if gradients already exist in the save directory
    
    Args:
        save_dir: Directory where gradients would be saved
    
    Returns:
        bool: True if existing gradients found, False otherwise
        int: Number of batch files found
        list: List of available layers
    """
    layers_dir = os.path.join(save_dir, 'layers')
    tokens_dir = os.path.join(save_dir, 'tokens')
    
    if not os.path.exists(layers_dir) or not os.path.exists(tokens_dir):
        return False, 0, []
    
    # Count batch files
    token_files = glob.glob(os.path.join(tokens_dir, '*_tokens.json'))
    
    # Get available layers
    available_layers = []
    if os.path.exists(layers_dir):
        layer_dirs = [d for d in os.listdir(layers_dir) if os.path.isdir(os.path.join(layers_dir, d))]
        available_layers = [layer_dir.replace('_', '.') for layer_dir in layer_dirs]
    
    return len(token_files) > 0, len(token_files), available_layers


def list_existing_gradients(save_dir):
    """
    List existing gradients in a user-friendly format
    
    Args:
        save_dir: Directory where gradients are saved
    """
    has_gradients, num_batches, available_layers = check_existing_gradients(save_dir)
    
    if not has_gradients:
        print(f"❌ No existing gradients found in {save_dir}")
        print("💡 Run gradient collection first!")
        return
    
    print(f"✅ Found existing gradients in {save_dir}")
    print(f"📊 Batch files: {num_batches}")
    print(f"🔗 Available layers: {len(available_layers)}")
    
    if available_layers:
        print("\n📋 Available layers:")
        
        # Group by layer type for better organization
        embedding_layers = [l for l in available_layers if 'wte' in l or 'wpe' in l]
        attention_layers = [l for l in available_layers if 'attn' in l]
        mlp_layers = [l for l in available_layers if 'mlp' in l]
        norm_layers = [l for l in available_layers if 'ln' in l]
        
        if embedding_layers:
            print("  🔤 Embedding layers:")
            for layer in sorted(embedding_layers):
                print(f"    - {layer}")
        
        if attention_layers:
            print("  🎯 Attention layers:")
            for layer in sorted(attention_layers):
                print(f"    - {layer}")
        
        if mlp_layers:
            print("  🧠 MLP layers:")
            for layer in sorted(mlp_layers):
                print(f"    - {layer}")
        
        if norm_layers:
            print("  📏 Normalization layers:")
            for layer in sorted(norm_layers):
                print(f"    - {layer}")
    
    print(f"\n💡 Use load_single_layer_gradients(collector, '<layer_name>') to load any layer")


def collect_token_gradients(model, tokenizer, stories, save_dir, 
                           max_samples=500, max_length=128, batch_size=4, 
                           save_batch_size=100, token_context_window=5,
                           force_recollect=False):
    """
    Collect token-specific gradients from a list of stories with optimized storage
    
    Args:
        model: The model to collect gradients from
        tokenizer: Tokenizer for the model
        stories: List of text stories
        save_dir: Directory to save gradients
        max_samples: Maximum number of stories to process
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        save_batch_size: Number of token gradients per saved batch
        token_context_window: Context window for token tracking
        force_recollect: If True, ignore existing gradients and recollect
    
    Returns:
        collector: The gradient collector instance
    """
    # Check for existing gradients first
    if not force_recollect:
        has_gradients, num_batches, available_layers = check_existing_gradients(save_dir)
        
        if has_gradients:
            print("🎯 Found existing gradients!")
            print(f"   📁 Directory: {save_dir}")
            print(f"   📊 Batch files: {num_batches}")
            print(f"   🔗 Available layers: {len(available_layers)}")
            print("   ✅ Using existing gradients (no collection needed)")
            
            if len(available_layers) > 0:
                print(f"\n📋 First few layers available:")
                for i, layer in enumerate(available_layers[:5]):
                    print(f"     {i+1}. {layer}")
                if len(available_layers) > 5:
                    print(f"     ... and {len(available_layers) - 5} more layers")
            
            print(f"\n💡 To force recollection, use: force_recollect=True")
            
            # Create a mock collector for compatibility
            collector = TokenSpecificGradientCollector(
                model, tokenizer, save_dir, token_context_window, save_batch_size
            )
            collector.batch_count = num_batches
            
            return collector
    
    # Initialize collector for new collection
    collector = TokenSpecificGradientCollector(
        model, tokenizer, save_dir, token_context_window, save_batch_size
    )
    
    print("🎯 Starting token-specific gradient collection...")
    print(f"   Processing {min(len(stories), max_samples)} stories")
    print(f"   Each token gets its own gradient (precise attribution!)")
    print(f"   Saving directly in optimized format (no reorganization needed)")
    
    # Process stories in batches
    for i in range(0, min(len(stories), max_samples), batch_size):
        batch_stories = stories[i:i+batch_size]
        collector.collect_gradients(batch_stories, max_length)
        
        if i % (batch_size * 5) == 0:
            print(f"Processed {i + len(batch_stories)} / {min(len(stories), max_samples)} stories")
    
    # Finalize collection
    num_batches = collector.finalize_collection()
    
    print(f"✅ Token-specific gradient collection complete!")
    print(f"Total token gradients: {collector.sample_count}")
    print(f"Total batches saved: {num_batches}")
    print(f"Optimized storage: {collector.layers_dir}")
    
    return collector


def analyze_gradient_diversity(gradients: Dict[str, torch.Tensor], 
                             device: Optional[torch.device] = None) -> Dict[str, Dict]:
    """Analyze gradient diversity across token samples"""
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
