import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
from tqdm import tqdm
import config as config
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

def generate_gradient_batch(model, tokenizer, text_batch):
    """Generate a single batch of per-token gradients."""
    module_name = ".".join(config.WEIGHT_NAME.split('.')[:-1])
    target_module = get_module_by_name(model, module_name)

    activations = None
    def forward_hook(module, input, output):
        nonlocal activations
        activations = input[0].detach()

    grad_outputs = None
    def backward_hook(module, grad_input, grad_output):
        nonlocal grad_outputs
        grad_outputs = grad_output[0].detach()

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    
    inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=config.CONTEXT_LENGTH).to(config.DEVICE)
    
    inputs["labels"] = inputs["input_ids"].clone()
    outputs = model(**inputs)
    loss = outputs.loss
    
    model.zero_grad()
    loss.backward()

    forward_handle.remove()
    backward_handle.remove()
    
    if activations is None or grad_outputs is None:
        return None
        
    per_token_grads = torch.einsum("bsd,bsi->bsdi", grad_outputs, activations)
    
    d_out, d_in = per_token_grads.shape[-2:]
    per_token_grads_flat = per_token_grads.view(-1, d_out * d_in)

    return per_token_grads_flat.cpu()

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