import torch
from datasets import load_dataset
import wandb
import os
from tqdm import tqdm
import random

import config as config
from model import AutoEncoderTopK
from trainer import SAETrainer
from utils import (
    get_model_and_tokenizer,
    generate_gradient_batch,
    upload_to_hf,
)

def main():
    """Main training loop."""
    wandb_config = {k: v for k, v in config.__dict__.items() if k.isupper()}
    run = wandb.init(project=config.WANDB_PROJECT, config=wandb_config)

    # Load model and tokenizer for gradient generation
    model, tokenizer = get_model_and_tokenizer(config.MODEL_NAME)
    model.eval()
    
    # Create a streaming dataset
    dataset = load_dataset(config.DATASET_NAME, split="train", streaming=True)
    dataset_iter = iter(dataset)

    # Initialize SAE
    # We need to get one batch of gradients to know the activation_dim
    print("Generating one batch to determine activation dimension...")
    text_batch = [next(dataset_iter)[config.TEXT_COLUMN] for _ in range(config.EFFECTIVE_BATCH_SIZE)]
    initial_grads = generate_gradient_batch(model, tokenizer, text_batch)
    activation_dim = initial_grads.shape[-1]
    print(f"Activation dimension: {activation_dim}")

    sae = AutoEncoderTopK(
        activation_dim=activation_dim,
        dict_size=config.DICT_SIZE,
        k=config.SAE_K,
        device=config.DEVICE
    ).to(config.DEVICE, dtype=config.DTYPE)
    
    # The trainer needs to know the total steps for the LR scheduler
    # We will estimate it.
    total_steps = config.TOTAL_TRAINING_TOKENS // config.BATCH_SIZE
    trainer = SAETrainer(
        model=sae,
        lr=config.LR,
        warmup_steps=int(total_steps * 0.0),
        total_steps=total_steps,
        device=config.DEVICE
    )

    # Training loop
    gradient_buffer = [initial_grads]
    tokens_in_buffer = initial_grads.shape[0]
    total_tokens_trained = 0
    
    pbar = tqdm(total=config.TOTAL_TRAINING_TOKENS, desc="Training SAE")
    while total_tokens_trained < config.TOTAL_TRAINING_TOKENS:
        # Fill buffer
        while tokens_in_buffer < config.GRADIENT_BUFFER_SIZE:
            text_batch = [next(dataset_iter)[config.TEXT_COLUMN] for _ in range(config.EFFECTIVE_BATCH_SIZE)]
            new_grads = generate_gradient_batch(model, tokenizer, text_batch)
            if new_grads is not None:
                gradient_buffer.append(new_grads)
                tokens_in_buffer += new_grads.shape[0]
        
        # Train on buffer
        buffer_tensor = torch.cat(gradient_buffer, dim=0)
        gradient_buffer = [buffer_tensor] # Keep remaining for next round
        
        # Create batches from buffer
        for i in range(0, len(buffer_tensor), config.BATCH_SIZE):
            batch = buffer_tensor[i:i+config.BATCH_SIZE].to(config.DEVICE).to(config.DTYPE)
            if len(batch) == 0: continue

            metrics = trainer.train_step(batch)
            metrics['tokens_trained'] = total_tokens_trained + len(batch)
            
            if (trainer.step_count % 100) == 0:
                print(f"Step {trainer.step_count}, Loss: {metrics['loss']:.4f}")
                wandb.log(metrics)

            total_tokens_trained += len(batch)
            pbar.update(len(batch))

            if total_tokens_trained >= config.TOTAL_TRAINING_TOKENS:
                break
        
        # Clear part of buffer used for training
        if len(buffer_tensor) > 0:
            gradient_buffer = [buffer_tensor[i+config.BATCH_SIZE:]]
            tokens_in_buffer = gradient_buffer[0].shape[0]
        else:
            gradient_buffer = []
            tokens_in_buffer = 0


    # Save and upload model
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "sae_final.pt")
    torch.save(sae.state_dict(), checkpoint_path)
    print(f"Final model saved to {checkpoint_path}")

    upload_to_hf(sae, config.HF_REPO_ID)
    wandb.finish()

if __name__ == "__main__":
    main() 