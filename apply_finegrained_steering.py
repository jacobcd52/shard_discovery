import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import copy
import logging

from config import Config
from model import MLP
from calculate_finegrained_steering_weight import calculate_steering_weight

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data_loaders(config, batch_size=1000):
    """Create test data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    return test_loader

def test_epoch(model, test_loader, criterion, device, config):
    """Test for one epoch and return accuracy and predictions."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=config.dtype if config else torch.float32), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = 100. * correct / total
    
    return epoch_accuracy, np.array(all_predictions)

def plot_distribution_difference(original_preds, steered_preds, output_dir, desc, param_name, alpha):
    """Plots the difference in prediction counts before and after steering."""
    plt.figure(figsize=(12, 6))
    
    original_counts = Counter(original_preds)
    steered_counts = Counter(steered_preds)
    
    digits = sorted(list(set(original_preds) | set(steered_preds)))
    diff = [steered_counts.get(d, 0) - original_counts.get(d, 0) for d in digits]
    
    sns.barplot(x=digits, y=diff, palette="vlag")
    title = (
        f"Change in Prediction Counts after Steering\n{desc}\n"
        f"Steered Params: {param_name} | Equation: $W_{{steered}} = W_{{orig}} + ({alpha}) \\cdot (W_{{towards}} - W_{{away}})$"
    )
    plt.title(title, fontsize=10)
    plt.xlabel("Predicted Digit")
    plt.ylabel("Change in Count (Steered - Original)")
    plt.axhline(0, color='grey', linewidth=0.8)
    save_path = os.path.join(output_dir, "prediction_distribution_difference.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved prediction difference plot to {save_path}")
    plt.close()

def steer_model_with_weights(model, steering_weights, alpha, layers_to_steer, device):
    """
    Alters the model's parameters by adding scaled steering weights.
    """
    steered_model = copy.deepcopy(model)
    
    print("\n--- Applying Steering Weights ---")
    with torch.no_grad():
        model_state_dict = steered_model.state_dict()
        for param_name in layers_to_steer:
            if param_name in steering_weights and param_name in model_state_dict:
                param = model_state_dict[param_name]
                steering_vector = steering_weights[param_name].to(param.device, dtype=param.dtype)
                
                if param.shape != steering_vector.shape:
                    logging.warning(f"Shape mismatch for {param_name}. Param: {param.shape}, Vector: {steering_vector.shape}. Skipping.")
                    continue
                
                param.add_(steering_vector, alpha=alpha)
                print(f"  - Steered parameter '{param_name}'")
            else:
                logging.warning(f"Parameter '{param_name}' not found in steering weights or model state. Skipping.")
            
    return steered_model

def main(args):
    # Setup
    config = Config()
    device = config.device
    
    # Override config with command-line arguments if provided
    if args.finetuning_steps is not None:
        config.finetuning_steps = args.finetuning_steps
    
    # Create a unique output directory for this experiment
    towards_str = "".join(map(str, sorted(args.steer_towards)))
    away_str = "".join(map(str, sorted(args.steer_away)))
    layers_str = "all" if "all" in args.layers_to_steer else "_".join(args.layers_to_steer).replace('.', '_')
    output_dir_name = (
        f"finegrained_steer_t{towards_str}_a{away_str}_alpha{args.alpha}_"
        f"layers_{layers_str}_finetuning_steps_{config.finetuning_steps}"
    )
    output_dir = os.path.abspath(os.path.join("steering_results", output_dir_name))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Model
    logging.info("Loading base model...")
    model = MLP(
        input_size=config.input_size,
        hidden_sizes=config.hidden_sizes,
        output_size=config.output_size,
        dtype=config.dtype
    ).to(device)
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    
    # If 'all' is specified for layers, get all parameter names
    if 'all' in args.layers_to_steer:
        layers_to_steer = list(model.state_dict().keys())
    else:
        layers_to_steer = args.layers_to_steer

    # Load Test Data & Evaluate Original Model
    logging.info("Loading MNIST test data and evaluating original model...")
    test_loader = get_data_loaders(config)
    criterion = nn.CrossEntropyLoss()
    original_accuracy, original_predictions = test_epoch(model, test_loader, criterion, device, config)
    logging.info(f"Accuracy of original model: {original_accuracy:.2f}%")

    # --- Calculate/Load Steering Weights ---
    logging.info(f"Getting steering weights for towards={args.steer_towards} and away={args.steer_away}")
    steering_weights = calculate_steering_weight(
        args.steer_towards, 
        args.steer_away, 
        config,
        force_recalculate=args.rerun_sw_calc,
        output_dir=output_dir
    )

    if not steering_weights:
        logging.error("Failed to prepare steering weights. Aborting.")
        return

    # --- Steer The Model ---
    steered_model = steer_model_with_weights(model, steering_weights, args.alpha, layers_to_steer, device)
    
    # --- Evaluate Steered Model ---
    logging.info("Evaluating steered model...")
    steered_accuracy, steered_predictions = test_epoch(steered_model, test_loader, criterion, device, config)
    
    # --- Visualize Results ---
    desc = f"Towards {args.steer_towards}, Away From {args.steer_away}"
    
    # Original Distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(x=original_predictions, palette="viridis")
    plt.title("Original Model Prediction Distribution")
    plt.xlabel("Predicted Digit")
    plt.ylabel("Count")
    save_path = os.path.join(output_dir, "original_prediction_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Steered Distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(x=steered_predictions, palette="magma")
    title = (f"Post-Steering Distribution\n{desc} | $\\alpha = {args.alpha}$")
    plt.title(title, fontsize=12)
    plt.xlabel("Predicted Digit")
    plt.ylabel("Count")
    save_path = os.path.join(output_dir, "post_steering_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Difference Plot
    plot_distribution_difference(original_predictions, steered_predictions, output_dir, desc, ", ".join(layers_to_steer), args.alpha)

    # --- Final Report ---
    print("\n--- Steering Experiment Summary ---")
    print(f"Steering Towards: {args.steer_towards}, Away From: {args.steer_away}")
    print(f"Steered {len(layers_to_steer)} parameter(s): {', '.join(layers_to_steer)}")
    print(f"Steering Strength (alpha): {args.alpha}")
    print("-" * 30)
    print(f"Original Model Accuracy: {original_accuracy:.2f}%")
    print(f"Steered Model Accuracy:  {steered_accuracy:.2f}%")
    print(f"Accuracy Change: {steered_accuracy - original_accuracy:+.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply fine-grained steering vectors to a trained model.")
    parser.add_argument('--steer_towards', type=int, nargs='+', required=True, help='List of digits to steer towards.')
    parser.add_argument('--steer_away', type=int, nargs='+', required=True, help='List of digits to steer away from.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Steering strength.')
    parser.add_argument('--layers_to_steer', nargs='+', default=['all'], help="List of parameter names to steer (e.g., 'network.0.weight'). Default is 'all'.")
    parser.add_argument('--rerun_sw_calc', action='store_true', help='Force recalculation of steering weights even if cached.')
    parser.add_argument('--finetuning_steps', type=int, default=None, help='Number of fine-tuning steps. Overrides config.')
    
    args = parser.parse_args()
    main(args) 