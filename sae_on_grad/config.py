import torch

# DTYPE = torch.float32
DTYPE = torch.bfloat16

# SAE Parameters
SAE_K = 16 
DICT_SIZE = 8192 


# Dataset Parameters
DATASET_NAME = "roneneldan/TinyStories"
TEXT_COLUMN = "text"
TOTAL_TRAINING_TOKENS = 20_000_000
CONTEXT_LENGTH = 128

# Model Parameters
MODEL_NAME = "roneneldan/TinyStories-1M"
WEIGHT_NAME = "transformer.h.0.attn.attention.v_proj.weight"


# Training Parameters
BATCH_SIZE = 2048 # Batch size for training the SAE
EFFECTIVE_BATCH_SIZE = 32 # Batch size for generating gradients
GRADIENT_BUFFER_SIZE = 50_000 # Number of gradients to buffer in memory
LR = 1e-4
N_EPOCHS = 1

# Other Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
WANDB_PROJECT = "sae_on_grad"
GRADIENT_SAVE_DIR = f"sae_on_grad/gradients_{MODEL_NAME.replace('/', '_')}_{WEIGHT_NAME}"
CHECKPOINT_DIR = f"sae_on_grad/checkpoints_{MODEL_NAME.replace('/', '_')}_{WEIGHT_NAME}"
HF_REPO_ID = f"jacobcd52/{MODEL_NAME.replace('/', '_')}_{WEIGHT_NAME}_sae_k{SAE_K}" 