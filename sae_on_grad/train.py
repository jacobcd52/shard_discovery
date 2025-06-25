import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb
import os

import config as config
from sae_on_grad.model import AutoEncoderTopK
from sae_on_grad.trainer import SAETrainer
from sae_on_grad.utils import (
    get_model_and_tokenizer,
    generate_and_save_gradients,
    GradientDataset,
    upload_to_hf,
)

def main():
    """Main training loop."""
    # wandb.login()
    run = wandb.init(project=config.WANDB_PROJECT, config=config.__dict__)

    # Ensure gradient directory exists
    if not os.path.exists(config.GRADIENT_SAVE_DIR) or not os.listdir(config.GRADIENT_SAVE_DIR):
        print("Gradient directory not found or empty, generating gradients...")
        model, tokenizer = get_model_and_tokenizer(config.MODEL_NAME)
        dataset = load_dataset(config.DATASET_NAME, split="train")
        generate_and_save_gradients(model, tokenizer, dataset, config.WEIGHT_NAME, config.GRADIENT_SAVE_DIR)
        del model, tokenizer, dataset # Free up memory
    else:
        print("Gradients found, skipping generation.")

    # Load dataset
    grad_dataset = GradientDataset(config.GRADIENT_SAVE_DIR)
    activation_dim = grad_dataset.activation_dim
    
    # Initialize SAE
    sae = AutoEncoderTopK(
        activation_dim=activation_dim,
        dict_size=config.DICT_SIZE,
        k=config.SAE_K,
        device=config.DEVICE
    ).to(config.DEVICE, dtype=config.DTYPE)
    
    dataloader = DataLoader(grad_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    total_steps = len(dataloader) * config.N_EPOCHS
    trainer = SAETrainer(
        model=sae,
        lr=config.LR,
        warmup_steps=int(total_steps * 0.1), # 10% warmup
        total_steps=total_steps,
        device=config.DEVICE
    )

    # Training loop
    for epoch in range(config.N_EPOCHS):
        print(f"--- Epoch {epoch+1}/{config.N_EPOCHS} ---")
        for i, batch in enumerate(dataloader):
            batch = batch.to(config.DEVICE).to(config.DTYPE)
            metrics = trainer.train_step(batch)
            
            if (i % 100) == 0:
                print(f"Step {i}/{len(dataloader)}, Loss: {metrics['loss']:.4f}")
                wandb.log(metrics)

    # Save and upload model
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "sae_final.pt")
    torch.save(sae.state_dict(), checkpoint_path)
    print(f"Final model saved to {checkpoint_path}")

    # Upload to Hugging Face
    upload_to_hf(sae, config.HF_REPO_ID)

    wandb.finish()

if __name__ == "__main__":
    main() 