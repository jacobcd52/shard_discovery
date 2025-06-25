import torch

# DTYPE = torch.float32
DTYPE = torch.bfloat16

# SAE Parameters
# SAE_K = 256 # Tinystories
SAE_K = 64 # Openwebtext
# DICT_SIZE = 131072 # Tinystories
DICT_SIZE = 65536 # Openwebtext


# Dataset Parameters
DATASET_NAME = "roneneldan/TinyStories"
TEXT_COLUMN = "text"
N_BATCHES_TO_SAVE = 100 # TODO: increase
CONTEXT_LENGTH = 256

# Model Parameters
MODEL_NAME = "roneneldan/TinyStories-1M"
WEIGHT_NAME = "transformer.h.0.attn.attention.v_proj.weight"


# Training Parameters
BATCH_SIZE = 4096 # Batch size for training the SAE
EFFECTIVE_BATCH_SIZE = 4 # Batch size for generating gradients
LR = 4e-4
N_EPOCHS = 1

# Other Parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
WANDB_PROJECT = "sae_on_grad"
GRADIENT_SAVE_DIR = f"sae_on_grad/gradients_{MODEL_NAME.replace('/', '_')}_{WEIGHT_NAME}"
CHECKPOINT_DIR = f"sae_on_grad/checkpoints_{MODEL_NAME.replace('/', '_')}_{WEIGHT_NAME}"
HF_REPO_ID = f"TahaDouaji/{MODEL_NAME.replace('/', '_')}_{WEIGHT_NAME}_sae_k{SAE_K}" 