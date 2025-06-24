import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import copy
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools

from model import MLP
from config import Config

def get_test_loader(config):
    """Creates a DataLoader for the test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    return loader

def test_epoch(model, test_loader, criterion, device, config):
    """Test for one epoch and return accuracy and predictions."""
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
    return np.array(all_predictions)

def plot_distribution(predictions, title, save_path):
    """Plots and saves the prediction distribution."""
    plt.figure(figsize=(10, 5))
    sns.countplot(x=predictions, palette="viridis")
    plt.title(title)
    plt.xlabel("Predicted Digit")
    plt.ylabel("Count")
    plt.savefig(save_path, dpi=300)
    print(f"Saved fine-tuned model distribution plot to {save_path}")
    plt.close()

def get_mnist_dataset():
    """Returns the full MNIST training dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    return train_dataset

def create_subset_loader(dataset, digits, config):
    """Creates a DataLoader for a subset of digits."""
    indices = [i for i, (_, label) in enumerate(dataset) if label in digits]
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    return loader

def finetune_model(model, loader, config, device):
    """Fine-tunes a model on the provided data loader for a specific number of steps."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.finetuning_learning_rate)
    criterion = nn.CrossEntropyLoss()

    data_iterator = itertools.cycle(loader)

    print(f"Starting fine-tuning for {config.finetuning_steps} steps...")
    for step in range(config.finetuning_steps):
        try:
            data, target = next(data_iterator)
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0 or (step + 1) == config.finetuning_steps:
                 print(f"  Fine-tuning step {step + 1}/{config.finetuning_steps} complete.")
        except StopIteration:
            print("Warning: DataLoader exhausted before reaching target number of steps. Fine-tuning stopped early.")
            break
    
    print("Fine-tuning complete.")
    return model

def calculate_steering_weight(steer_towards, steer_away, config=None, force_recalculate=False, output_dir=None):
    """
    Calculates a steering weight vector by fine-tuning two models and taking their difference.
    Caches the result to avoid re-calculation.

    Args:
        steer_towards (list): List of integer digits to steer towards.
        steer_away (list): List of integer digits to steer away from.
        config (Config, optional): Configuration object. Defaults to None.
        force_recalculate (bool, optional): Whether to force recalculation of steering weights. Defaults to False.
        output_dir (str, optional): Directory to save prediction distribution plots. Defaults to None.

    Returns:
        dict: A dictionary mapping parameter names to steering weight tensors.
    """
    if config is None:
        config = Config()

    # Define cache path
    os.makedirs(config.steering_vector_dir, exist_ok=True)
    towards_str = "".join(map(str, sorted(steer_towards)))
    away_str = "".join(map(str, sorted(steer_away)))
    cache_filename = f"steering_weights_towards_{towards_str}_away_{away_str}.pt"
    cache_path = os.path.join(config.steering_vector_dir, cache_filename)

    # Check if cached version exists
    if not force_recalculate and os.path.exists(cache_path):
        print(f"Loading cached steering weights from {cache_path}")
        return torch.load(cache_path)

    print("No cached steering weights found. Calculating from scratch...")

    # --- Data Preparation ---
    full_train_dataset = get_mnist_dataset()
    loader_towards = create_subset_loader(full_train_dataset, steer_towards, config)
    loader_away = create_subset_loader(full_train_dataset, steer_away, config)

    # --- Model Loading and Copying ---
    print("Loading base model...")
    base_model = MLP(
        input_size=config.input_size,
        hidden_sizes=config.hidden_sizes,
        output_size=config.output_size
    ).to(config.device)
    model_path = config.model_save_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Base model not found at {model_path}. Please train a model first by running train.py.")
    base_model.load_state_dict(torch.load(model_path))

    model_towards = copy.deepcopy(base_model)
    model_away = copy.deepcopy(base_model)

    # --- Fine-tuning ---
    print(f"Fine-tuning 'towards' model on digits: {steer_towards}...")
    finetune_model(model_towards, loader_towards, config, config.device)

    print(f"Fine-tuning 'away' model on digits: {steer_away}...")
    finetune_model(model_away, loader_away, config, config.device)

    # --- Evaluation and Plotting of fine-tuned models ---
    if output_dir:
        print("\n--- Evaluating fine-tuned models on test set ---")
        test_loader = get_test_loader(config)
        criterion = nn.CrossEntropyLoss()

        # Evaluate 'towards' model
        preds_towards = test_epoch(model_towards, test_loader, criterion, config.device, config)
        plot_distribution(
            preds_towards,
            f"Prediction Distribution of Model Fine-tuned on Digits: {steer_towards}",
            os.path.join(output_dir, "finetuned_model_dist_towards.png")
        )

        # Evaluate 'away' model
        preds_away = test_epoch(model_away, test_loader, criterion, config.device, config)
        plot_distribution(
            preds_away,
            f"Prediction Distribution of Model Fine-tuned on Digits: {steer_away}",
            os.path.join(output_dir, "finetuned_model_dist_away.png")
        )
        print("--- Finished evaluating fine-tuned models ---\n")

    # --- Calculate Steering Weights ---
    print("Calculating steering weights (difference of model weights)...")
    steering_weights = OrderedDict()
    params_towards = model_towards.state_dict()
    params_away = model_away.state_dict()

    for name, param_towards in params_towards.items():
        if name in params_away:
            steering_weights[name] = param_towards - params_away[name]

    # --- Caching and Returning ---
    print(f"Saving steering weights to {cache_path}")
    torch.save(steering_weights, cache_path)

    return steering_weights

if __name__ == '__main__':
    # Example usage:
    print("--- Example: Calculating steering vector for 9 vs 0 ---")
    example_config = Config()
    # Add finetuning-specific parameters to config if they don't exist
    if not hasattr(example_config, 'steering_vector_dir'):
        example_config.steering_vector_dir = "results/steering_vectors"

    # Create a dummy output dir for the example
    os.makedirs("steering_results/example_output", exist_ok=True)

    steering_vector = calculate_steering_weight(steer_towards=[9], steer_away=[0], config=example_config, force_recalculate=True, output_dir="steering_results/example_output")
    print("\nCalculation complete. Steering vector dictionary contains keys:")
    for name, tensor in steering_vector.items():
        print(f"- {name}: shape={tensor.shape}, norm={torch.norm(tensor).item():.4f}")

    print("\n--- Example: Using cached vector ---")
    # This call should be much faster
    steering_vector_cached = calculate_steering_weight(steer_towards=[9], steer_away=[0], config=example_config)
    print("\nCached calculation complete.") 