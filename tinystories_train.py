import os
import json
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import wandb

# ───────────────────── Local project imports ────────────────────────────────
from tinystories_config import TinyStoriesConfig  # ← new config module
from tinystories_model import TinyStoriesTransformer  # ← new model
from efficient_gradient_collector import EfficientPerSampleGradientCollector
from transformer_gradient_collector import TransformerGradientCollector


# ───────────────────────────── Dataset class ────────────────────────────────
class TinyStoriesDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # causal LM target
        }


# ─────────────────────── Tokenizer utilities ────────────────────────────────

def prepare_tokenizer(dataset_sample, config):
    """Either load an existing tokenizer or train a new Byte‑Level BPE one."""
    if os.path.exists(config.tokenizer_save_path):
        print(f"Loading tokenizer from {config.tokenizer_save_path}")
        return AutoTokenizer.from_pretrained(config.tokenizer_save_path)

    print("Training new tokenizer …")
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from tokenizers.normalizers import NFD, Lowercase, StripAccents

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = NFD()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    )
    texts = [item.get("text", item.get("story")) for item in dataset_sample]
    tokenizer.train_from_iterator(texts, trainer)

    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    hf_tok.pad_token = "<pad>"
    hf_tok.bos_token = "<s>"
    hf_tok.eos_token = "</s>"
    hf_tok.unk_token = "<unk>"
    hf_tok.mask_token = "<mask>"
    hf_tok.save_pretrained(config.tokenizer_save_path)
    return hf_tok


# ───────────────────────────── Data loading ─────────────────────────────────

def load_tinystories_data(config):
    dataset = load_dataset("roneneldan/TinyStories")
    train_ds = dataset["train"]
    val_ds = dataset.get("validation")

    # tokenizer training sample (10k)
    sample_ds = train_ds.select(range(min(10_000, len(train_ds))))
    tokenizer = prepare_tokenizer(sample_ds, config)
    config.vocab_size = len(tokenizer)

    train_texts = [item.get("text", item.get("story")) for item in train_ds]
    if val_ds:
        val_texts = [item.get("text", item.get("story")) for item in val_ds]
    else:
        val_texts = train_texts[-1000:]
        train_texts = train_texts[:-1000]

    return train_texts, val_texts, tokenizer


def create_dataloaders(train_texts, val_texts, tokenizer, config):
    train_set = TinyStoriesDataset(train_texts, tokenizer, config.max_length)
    val_set = TinyStoriesDataset(val_texts, tokenizer, config.max_length)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# ───────────────────────── LR schedule (linear warm‑up) ──────────────────────

def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ───────────────────────────── Training loop ────────────────────────────────

def train_model(config):
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    os.makedirs(config.results_dir, exist_ok=True)

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity=config.wandb_entity,
            config={
                "vocab_size": config.vocab_size,
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "n_layers": config.n_layers,
                "d_ff": config.d_ff,
                "max_seq_len": config.max_seq_len,
                "dropout": config.dropout,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "warmup_steps": config.warmup_steps,
                "weight_decay": config.weight_decay,
                "grad_clip_norm": config.grad_clip_norm,
                "use_amp": config.use_amp,
            }
        )

    train_texts, val_texts, tokenizer = load_tinystories_data(config)
    train_loader, val_loader = create_dataloaders(train_texts, val_texts, tokenizer, config)

    model = TinyStoriesTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )
    total_params = model.count_parameters()
    print(f"Parameter count: {total_params:,}")
    model = model.to(config.device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)

    scaler = GradScaler() if config.use_amp else None

    # -------- Gradient collector (optional) ----------------------------------
    gradient_collector = None
    if config.collect_gradients:
        if config.gradient_collector_type == "transformer":
            gradient_collector = TransformerGradientCollector(
                model, config.gradients_dir, collect_attention_only=config.collect_attention_only
            )
        elif config.gradient_collector_type == "efficient":
            gradient_collector = EfficientPerSampleGradientCollector(
                model, config.gradients_dir, max_samples_per_collection=config.max_gradient_samples
            )

    # ---------------------- Training iterations ------------------------------
    train_losses, val_losses, lrs = [], [], []
    global_step = 0
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        model.train()
        pbar = tqdm(train_loader, desc="Training")
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)

            if config.use_amp:
                with autocast():
                    loss = model(input_ids, attention_mask=attention_mask, labels=labels)["loss"]
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss = model(input_ids, attention_mask=attention_mask, labels=labels)["loss"]
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Gradient collection (optional) -- still per mini-batch
            if (
                gradient_collector is not None and
                global_step % config.gradient_collection_frequency == 0 and
                global_step > 0
            ):
                batch_size = min(config.max_gradient_samples, input_ids.size(0))
                sample_input_ids = input_ids[:batch_size]
                sample_labels = labels[:batch_size]
                gradient_collector.collect_gradients_for_batch(
                    sample_input_ids,
                    sample_labels,
                    lambda out, tgt: nn.CrossEntropyLoss(ignore_index=-100)(
                        out["logits"][..., :-1, :].contiguous().view(-1, config.vocab_size),
                        tgt[..., 1:].contiguous().view(-1),
                    ),
                )
                gradient_collector.save_gradients(epoch, batch_idx)
                gradient_collector.clear_gradients()

            # Only step optimizer every N steps
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging and global step update
                train_losses.append(loss.item() * config.gradient_accumulation_steps)  # scale back
                lrs.append(scheduler.get_last_lr()[0])
                global_step += 1
                pbar.set_postfix({"loss": f"{train_losses[-1]:.4f}", "lr": f"{lrs[-1]:.1e}"})

                # Log to wandb
                if config.use_wandb:
                    wandb.log({
                        "train_loss": train_losses[-1],
                        "learning_rate": lrs[-1],
                        "global_step": global_step,
                        "epoch": epoch + 1,
                    })

                # Checkpointing & validation
                if global_step % config.eval_every == 0:
                    val_loss = evaluate_model(model, val_loader, config)
                    val_losses.append(val_loss)
                    print(f"Validation loss: {val_loss:.4f}")
                    if config.use_wandb:
                        wandb.log({
                            "val_loss": val_loss,
                            "global_step": global_step,
                            "epoch": epoch + 1,
                        })
                    model.train()

        print(f"Epoch {epoch + 1} finished. Avg train loss: {np.mean(train_losses[-len(train_loader):]):.4f}")

    # ─────────────── Save final checkpoint & plots / samples ────────────────
    torch.save(model.state_dict(), config.model_save_path)
    print(f"Model saved to {config.model_save_path}")
    plot_training_curves(train_losses, val_losses, lrs, config)
    generate_sample_text(model, tokenizer, config)
    
    # Finish wandb run
    if config.use_wandb:
        wandb.finish()
    
    return model, train_losses, val_losses


# ───────────────────────────── Evaluation loop ───────────────────────────────

def evaluate_model(model, val_loader, config):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            with autocast(enabled=config.use_amp):
                loss = model(input_ids, attention_mask=attention_mask, labels=labels)["loss"]
            losses.append(loss.item())
    return float(np.mean(losses))


# ───────────────────────── Plotting & text generation ───────────────────────

def plot_training_curves(train_losses, val_losses, lrs, config):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label="train", alpha=0.7)
    if val_losses:
        val_x = np.linspace(0, len(train_losses) - 1, len(val_losses))
        ax1.plot(val_x, val_losses, label="val", alpha=0.7)
    ax1.set_title("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(lrs, alpha=0.7);
    ax2.set_title("Learning rate")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "training_curves.png"), dpi=300)
    plt.close()


def generate_sample_text(model, tokenizer, config):
    model.eval()
    prompts = [
        "Once upon a time,",
        "The little girl",
        "In a magical forest,",
        "The brave knight",
    ]
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
            output_ids = model.generate(input_ids, max_length=100)
            print("Prompt:", prompt)
            print("→", tokenizer.decode(output_ids[0], skip_special_tokens=True))
            print("-" * 60)


# ─────────────────────────────────── main ────────────────────────────────────

def main():
    config = TinyStoriesConfig()
    config.calculate_parameters()
    print(f"Device: {config.device} | AMP: {config.use_amp}")
    train_model(config)


if __name__ == "__main__":
    main()