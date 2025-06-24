"""
TinyStories Transformer Training Script with Gradient Collection
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import json
import pickle
from collections import defaultdict

from tinystories_config import TinyStoriesConfig
from tinystories_model import TinyStoriesTransformer
from efficient_gradient_collector import EfficientPerSampleGradientCollector
from transformer_gradient_collector import TransformerGradientCollector

class TinyStoriesDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For language modeling
        }

def prepare_tokenizer(dataset_sample, config):
    """Prepare tokenizer for TinyStories dataset"""
    print("Preparing tokenizer...")
    
    # Try to load existing tokenizer
    if os.path.exists(config.tokenizer_save_path):
        print(f"Loading existing tokenizer from {config.tokenizer_save_path}")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_save_path)
        return tokenizer
    
    # Create new tokenizer from scratch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from tokenizers.normalizers import NFD, Lowercase, StripAccents
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = NFD()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Train tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    )
    
    # Collect texts for training
    texts = []
    for item in tqdm(dataset_sample, desc="Collecting texts for tokenizer training"):
        if 'text' in item:
            texts.append(item['text'])
        elif 'story' in item:
            texts.append(item['story'])
    
    # Train tokenizer
    tokenizer.train_from_iterator(texts, trainer)
    
    # Convert to HuggingFace tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    hf_tokenizer.pad_token = "<pad>"
    hf_tokenizer.bos_token = "<s>"
    hf_tokenizer.eos_token = "</s>"
    hf_tokenizer.unk_token = "<unk>"
    hf_tokenizer.mask_token = "<mask>"
    
    # Save tokenizer
    os.makedirs(config.tokenizer_save_path, exist_ok=True)
    hf_tokenizer.save_pretrained(config.tokenizer_save_path)
    
    print(f"Tokenizer saved to {config.tokenizer_save_path}")
    print(f"Vocab size: {len(hf_tokenizer)}")
    
    return hf_tokenizer

def load_tinystories_data(config):
    """Load and prepare TinyStories dataset"""
    print("Loading TinyStories dataset...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Get train and validation splits
    train_dataset = dataset['train']
    val_dataset = dataset['validation'] if 'validation' in dataset else None
    
    # Sample for tokenizer training (first 10k samples)
    sample_dataset = train_dataset.select(range(min(10000, len(train_dataset))))
    
    # Prepare tokenizer
    tokenizer = prepare_tokenizer(sample_dataset, config)
    
    # Update config with actual vocab size
    config.vocab_size = len(tokenizer)
    
    # Extract texts
    train_texts = [item['text'] if 'text' in item else item['story'] 
                   for item in tqdm(train_dataset, desc="Extracting train texts")]
    
    val_texts = []
    if val_dataset:
        val_texts = [item['text'] if 'text' in item else item['story'] 
                     for item in tqdm(val_dataset, desc="Extracting val texts")]
    else:
        # Use last 1000 samples for validation
        val_texts = train_texts[-1000:]
        train_texts = train_texts[:-1000]
    
    # Limit dataset size for training (optional, remove for full dataset)
    # train_texts = train_texts[:50000]  # Use first 50k samples
    # val_texts = val_texts[:1000]  # Use first 1k samples for validation
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    
    return train_texts, val_texts, tokenizer

def create_dataloaders(train_texts, val_texts, tokenizer, config):
    """Create train and validation dataloaders"""
    
    # Create datasets
    train_dataset = TinyStoriesDataset(train_texts, tokenizer, config.max_length)
    val_dataset = TinyStoriesDataset(val_texts, tokenizer, config.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create a linear learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(config):
    """Main training function"""
    
    # Set random seeds
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Load data
    train_texts, val_texts, tokenizer = load_tinystories_data(config)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_texts, val_texts, tokenizer, config)
    
    # Initialize model
    model = TinyStoriesTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"Model parameters: {total_params:,}")
    
    # Move to device
    model = model.to(config.device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Initialize scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if config.use_amp else None
    
    # Initialize gradient collector
    gradient_collector = None
    if config.collect_gradients:
        if config.gradient_collector_type == 'transformer':
            gradient_collector = TransformerGradientCollector(
                model, 
                config.gradients_dir,
                collect_attention_only=config.collect_attention_only
            )
        elif config.gradient_collector_type == 'efficient':
            gradient_collector = EfficientPerSampleGradientCollector(
                model, 
                config.gradients_dir,
                max_samples_per_collection=config.max_gradient_samples
            )
        else:  # original
            from gradient_collector import PerSampleGradientCollector
            gradient_collector = PerSampleGradientCollector(model, config.gradients_dir)
    
    # Training metrics
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        epoch_train_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if config.use_amp:
                with autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss']
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at step {global_step}! Skipping batch.")
                    continue
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at step {global_step}! Skipping batch.")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Check for NaN gradients
                nan_grads = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"NaN gradient detected in {name} at step {global_step}!")
                        nan_grads = True
                        break
                
                if nan_grads:
                    optimizer.zero_grad()
                    continue
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
            
            scheduler.step()
            
            # Collect gradients
            if (gradient_collector is not None and 
                global_step % config.gradient_collection_frequency == 0 and 
                global_step > 0):
                
                print(f"\nCollecting gradients at step {global_step}...")
                
                # Use a subset of the current batch for gradient collection
                batch_size = min(config.max_gradient_samples, input_ids.size(0))
                sample_input_ids = input_ids[:batch_size]
                sample_attention_mask = attention_mask[:batch_size] if attention_mask is not None else None
                sample_labels = labels[:batch_size]
                
                # Collect gradients based on collector type
                if config.gradient_collector_type == 'transformer':
                    # Use transformer-specific collection method
                    if config.collect_attention_only:
                        gradient_collector.collect_gradients_attention_optimized(
                            sample_input_ids, sample_attention_mask, sample_labels
                        )
                    else:
                        gradient_collector.collect_gradients_batch_efficient(
                            sample_input_ids, sample_attention_mask, sample_labels,
                            max_samples=config.max_gradient_samples
                        )
                else:
                    # Use standard collection method for efficient and original collectors
                    def custom_criterion(output, target):
                        if isinstance(output, dict):
                            logits = output['logits']
                        else:
                            logits = output
                        
                        # Shift for causal LM
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = target[..., 1:].contiguous()
                        
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                        return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    gradient_collector.collect_gradients_for_batch(
                        sample_input_ids, 
                        sample_labels,
                        custom_criterion,
                        sample_indices=None
                    )
                
                # Save gradients
                gradient_collector.save_gradients(epoch, batch_idx)
                gradient_collector.clear_gradients()
            
            # Update metrics
            epoch_train_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'step': global_step
            })
            
            # Log metrics
            if global_step % config.log_every == 0:
                train_losses.append(loss.item())
                learning_rates.append(scheduler.get_last_lr()[0])
            
            # Save checkpoint
            if global_step % config.save_every == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'config': config.__dict__,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
                
                checkpoint_path = os.path.join(config.results_dir, f'checkpoint_step_{global_step}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at step {global_step}")
            
            # Validation
            if global_step % config.eval_every == 0:
                val_loss = evaluate_model(model, val_loader, config)
                val_losses.append(val_loss)
                print(f"Validation loss at step {global_step}: {val_loss:.4f}")
                model.train()  # Switch back to training mode
        
        # End of epoch
        avg_train_loss = epoch_train_loss / num_batches
        print(f"Average training loss for epoch {epoch + 1}: {avg_train_loss:.4f}")
        
        # Validation at end of epoch
        val_loss = evaluate_model(model, val_loader, config)
        print(f"Validation loss at end of epoch {epoch + 1}: {val_loss:.4f}")
        model.train()
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': config.num_epochs,
        'global_step': global_step,
        'config': config.__dict__,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_parameters': total_params
    }
    
    torch.save(final_checkpoint, config.model_save_path)
    print(f"Final model saved to {config.model_save_path}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, learning_rates, config)
    
    # Generate sample text
    generate_sample_text(model, tokenizer, config)
    
    return model, train_losses, val_losses

def evaluate_model(model, val_loader, config):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            if config.use_amp:
                with autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss']
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def plot_training_curves(train_losses, val_losses, learning_rates, config):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', alpha=0.7)
    if val_losses:
        # Interpolate validation losses to match training loss points
        val_steps = np.linspace(0, len(train_losses)-1, len(val_losses))
        ax1.plot(val_steps, val_losses, label='Validation Loss', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate
    ax2.plot(learning_rates, alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_sample_text(model, tokenizer, config):
    """Generate sample text using the trained model"""
    model.eval()
    
    # Sample prompts
    prompts = [
        "Once upon a time,",
        "The little girl",
        "In a magical forest,",
        "The brave knight"
    ]
    
    generated_texts = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(config.device)
            
            # Generate
            output_ids = model.generate(
                input_ids,
                max_length=100,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            # Decode
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 50)
    
    # Save generated texts
    with open(os.path.join(config.results_dir, 'generated_samples.txt'), 'w') as f:
        for i, text in enumerate(generated_texts):
            f.write(f"Sample {i+1}:\n{text}\n\n")
    
    model.train()

def main():
    """Main function"""
    config = TinyStoriesConfig()
    
    # Calculate and print expected parameters
    config.calculate_parameters()
    
    print(f"Training TinyStories Transformer")
    print(f"Device: {config.device}")
    print(f"Precision: {config.dtype}")
    print(f"Gradient collection: {config.collect_gradients}")
    
    # Train model
    model, train_losses, val_losses = train_model(config)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 