import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
from tqdm import tqdm
import sae_on_grad.config as config
from huggingface_hub import HfApi, create_repo

def get_model_and_tokenizer(model_name: str):
    """Load model and tokenizer from Hugging Face."""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=config.DTYPE).to(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_module_by_name(model, module_name):
    """Get a module from a model by its name."""
    for name, module in model.named_modules():
        if name == module_name:
            return module
    raise ValueError(f"Module '{module_name}' not found in model.")

def generate_and_save_gradients(model, tokenizer, dataset, weight_name, save_dir):
    """Generate and save per-token gradients to disk."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    
    module_name = ".".join(weight_name.split('.')[:-1])
    target_module = get_module_by_name(model, module_name)

    activations = None
    def forward_hook(module, input, output):
        nonlocal activations
        # input is a tuple
        activations = input[0].detach()

    grad_outputs = None
    def backward_hook(module, grad_input, grad_output):
        nonlocal grad_outputs
        # grad_output is a tuple
        grad_outputs = grad_output[0].detach()

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)

    for i in tqdm(range(config.N_BATCHES_TO_SAVE), desc="Generating Gradients"):
        batch = dataset[i * config.EFFECTIVE_BATCH_SIZE: (i + 1) * config.EFFECTIVE_BATCH_SIZE][config.TEXT_COLUMN]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=config.CONTEXT_LENGTH).to(config.DEVICE)
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        model.zero_grad()
        loss.backward()

        # activations: [batch, seq_len, d_in]
        # grad_outputs: [batch, seq_len, d_out]
        
        if activations is None or grad_outputs is None:
            print("Warning: Hooks did not run. Skipping batch.")
            continue
            
        per_token_grads = torch.einsum("bsd,bsi->bsdi", grad_outputs, activations)
        
        # flatten to [batch*seq, d_out*d_in]
        d_out, d_in = per_token_grads.shape[-2:]
        per_token_grads_flat = per_token_grads.view(-1, d_out * d_in)

        torch.save(per_token_grads_flat.cpu(), os.path.join(save_dir, f"batch_{i}.pt"))
    
    forward_handle.remove()
    backward_handle.remove()


class GradientDataset(Dataset):
    """PyTorch dataset for loading saved gradients."""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.files = sorted([os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.pt')])
        # This can be memory intensive, for larger datasets, consider memory-mapping
        self.data_tensors = [torch.load(f) for f in tqdm(self.files, desc="Loading gradients")]
        self.data = torch.cat(self.data_tensors, dim=0)
        self.activation_dim = self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def upload_to_hf(model, repo_id: str):
    """Upload model to Hugging Face Hub."""
    api = HfApi()
    create_repo(repo_id, exist_ok=True, repo_type="model")
    
    checkpoint_path = "sae.pt"
    torch.save(model.state_dict(), checkpoint_path)

    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="sae.pt",
        repo_id=repo_id,
    )
    print(f"Model uploaded to https://huggingface.co/{repo_id}")
    os.remove(checkpoint_path) 