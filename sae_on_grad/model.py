import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import os


class AutoEncoderTopK(nn.Module):
    """
    Top-K Sparse Autoencoder.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int, device: str = "cpu"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k
        self.device = device

        self.encoder = nn.Linear(activation_dim, dict_size, device=device)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        self.b_dec = nn.Parameter(torch.zeros(activation_dim, device=device))

        # Initialization
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        self.decoder.weight.data = self.encoder.weight.data.T.clone()


    def encode(self, x: torch.Tensor):
        x_cent = x - self.b_dec
        f_pre = self.encoder(x_cent)
        f_relu = torch.relu(f_pre)

        top_k_values, top_k_indices = torch.topk(f_relu, self.k, dim=-1)

        f = torch.zeros_like(f_relu)
        f.scatter_(-1, top_k_indices, top_k_values)
        return f

    def decode(self, f: torch.Tensor):
        return self.decoder(f) + self.b_dec

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    @classmethod
    def from_pretrained(cls, path_or_repo_id: str, k: int, device: str="cpu", **kwargs):
        """
        Load a pretrained autoencoder from a file or Hugging Face Hub.
        """
        if os.path.exists(path_or_repo_id):
            state_dict = torch.load(path_or_repo_id, map_location=device)
        else:
            # Assume it's a Hub repo_id
            try:
                model_file = hf_hub_download(repo_id=path_or_repo_id, filename="sae.pt", **kwargs)
                state_dict = torch.load(model_file, map_location=device)
            except Exception as e:
                raise IOError(f"Could not load model from {path_or_repo_id}. Ensure it's a valid file path or HF Hub repo_id.") from e
        
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        model = cls(activation_dim=activation_dim, dict_size=dict_size, k=k, device=device)
        model.load_state_dict(state_dict)
        model.to(device)
        return model 