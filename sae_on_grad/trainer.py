import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import config as config

@torch.no_grad()
def geometric_median(points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median of a set of points."""
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess.clone()
        weights = 1 / torch.norm(points - guess, dim=1)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess

def get_lr_schedule(steps: int, warmup_steps: int, decay_start: int = 0):
    """Learning rate schedule with linear warmup and cosine decay."""
    if decay_start == 0:
        decay_start = warmup_steps
    
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < decay_start:
            return 1.0
        else:
            progress = (current_step - decay_start) / (steps - decay_start)
            return float(0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))
    return lr_lambda


def set_decoder_norm_to_unit_norm(decoder_weight: nn.Parameter):
    """Normalize decoder weights to have unit norm."""
    norms = torch.norm(decoder_weight, dim=1, keepdim=True)
    decoder_weight.data /= norms
    decoder_weight.data = torch.nan_to_num(decoder_weight.data)


def remove_gradient_parallel_to_decoder_directions(decoder_weight: nn.Parameter, grad: torch.Tensor):
    """Remove gradient components parallel to decoder directions."""
    if grad is None:
        return None
    
    parallel_component = torch.einsum("di,dj->ij", grad, decoder_weight)
    return grad - torch.einsum("ij,dj->di", parallel_component, decoder_weight)


class SAETrainer:
    def __init__(self, model, lr, warmup_steps, total_steps, device):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr)
        lr_schedule = get_lr_schedule(total_steps, warmup_steps)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_schedule)
        self.device = device
        self.step_count = 0

    def train_step(self, batch):
        """Perform a single training step."""
        if self.step_count == 0:
            # Initialize decoder bias with geometric median
            median = geometric_median(batch)
            self.model.b_dec.data = median.to(self.model.b_dec.dtype)

        self.optimizer.zero_grad()
        
        x_hat, _ = self.model(batch)
        mse = (x_hat - batch).pow(2).sum()
        var = (batch - batch.mean()).pow(2).sum()
        loss = mse / var
        
        loss.backward()

        # Gradient clipping and normalization
        self.model.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.model.decoder.weight, self.model.decoder.weight.grad
        )
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()

        # Re-normalize decoder weights
        set_decoder_norm_to_unit_norm(self.model.decoder.weight)
        
        self.step_count += 1
        
        return {"loss": loss.item()} 