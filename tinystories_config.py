import torch
import math


class TinyStoriesConfig:
    # ───────────────────────── Model hyper‑parameters ──────────────────────────
    vocab_size: int = 10_000  # set by tokenizer; 10k ≈ paper default
    d_model: int = 64         # smaller hidden size → allows 8 layers at ~1 M params
    n_heads: int = 16         # head dim = 4 (64 // 16)
    n_layers: int = 8         # depth helps more than width at this scale
    d_ff: int = 256           # 4 × d_model (standard multiplier)
    max_seq_len: int = 2048   # modelʼs context window; train on ≤512‑token chunks
    dropout: float = 0.1

    # If you implement alternating global/local attention, encode the pattern here
    # e.g. ["g", "l", ...] length = n_layers. Leave None to use full global.
    attention_pattern = ["g", "l"] * 4  # g/l alternation as in TinyStories models

    # ──────────────────────── Data loader parameters ───────────────────────────
    batch_size: int = 128
    max_length: int = 512         # truncate/pack training sequences to this length
    num_workers: int = 16

    # ───────────────────────── Training hyper‑parameters ───────────────────────
    learning_rate: float = 3e-4    # peak LR with cosine decay (AdamW)
    num_epochs: int = 2
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip_norm: float = 0.5
    gradient_accumulation_steps: int = 8  # Number of steps to accumulate gradients

    # ─────────────────────────── Device / precision ────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16           # AMP recommended; switch to float32 if unstable
    use_amp: bool = True

    # ─────────────────────── Check‑point & output paths ────────────────────────
    results_dir: str = "tinystories_results"
    model_save_path: str = f"{results_dir}/tinystories_transformer.pth"
    tokenizer_save_path: str = f"{results_dir}/tokenizer"

    # ─────────────────── Gradient collection (optional) ────────────────────────
    collect_gradients: bool = True
    gradients_dir: str = f"{results_dir}/gradients"
    gradient_collection_frequency: int = 10
    max_gradient_samples: int = 64
    gradient_collector_type: str = "efficient"  # "efficient" | "transformer" | "original"
    collect_attention_only: bool = False

    # ──────────────────────── Logging / evaluation ─────────────────────────────
    log_every: int = 100
    save_every: int = 2_000
    eval_every: int = 1_000
    
    # ──────────────────────────── Wandb logging ────────────────────────────────
    use_wandb: bool = True
    wandb_project: str = "tinystories-transformer"
    wandb_run_name: str = None  # will be auto-generated if None
    wandb_entity: str = None    # your wandb username/team

    # ─────────────────────────── Miscellaneous ────────────────────────────────
    random_seed: int = 42

    # ───────────────────── Parameter‑count utilities ───────────────────────────
    tie_embeddings: bool = True  # share input & output embeddings → ~1 M total params

    @classmethod
    def calculate_parameters(cls):
        """Rough estimate of the number of trainable parameters."""
        # Embedding layer
        embedding_params = cls.vocab_size * cls.d_model

        # Per‑layer Transformer params
        attention_params_per_layer = 4 * cls.d_model * cls.d_model  # Q,K,V,O
        ff_params_per_layer = 2 * cls.d_model * cls.d_ff            # W1 & W2
        params_per_layer = attention_params_per_layer + ff_params_per_layer
        transformer_params = cls.n_layers * params_per_layer

        # Output projection (unless tied with input embedding)
        output_params = 0 if cls.tie_embeddings else cls.vocab_size * cls.d_model

        total_params = embedding_params + transformer_params + output_params

        print("Estimated parameters:")
        print(f"  Embedding: {embedding_params:,}")
        print(f"  Transformer ({cls.n_layers} layers): {transformer_params:,}")
        if not cls.tie_embeddings:
            print(f"  Output: {output_params:,}")
        print(f"  Total: {total_params:,}")
        print("  Target: ~1M parameters")

        return total_params
