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
from gradient_loader import load_stacked_gradients
from k_means import kmeans_gradients_gpu

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data_loaders(config, batch_size=1000):
    """Create train and test data loaders."""
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
        f"Change in Prediction Counts after Steering on {desc}\n"
        f"Steered Params: {param_name} | Equation: $W_{{steered}} = W_{{orig}} + ({alpha}) \\cdot G_{{avg}}$"
    )
    plt.title(title, fontsize=10)
    plt.xlabel("Predicted Digit")
    plt.ylabel("Change in Count (Steered - Original)")
    plt.axhline(0, color='grey', linewidth=0.8)
    save_path = os.path.join(output_dir, "prediction_distribution_difference.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved prediction difference plot to {save_path}")
    plt.close()

def steer_model_with_vectors(model, steering_vectors, alpha, device):
    """
    Alter the model's parameters by adding multiple scaled steering vectors.
    """
    steered_model = copy.deepcopy(model) # Don't modify the original model
    
    print("\n--- Applying Steering Vectors ---")
    with torch.no_grad():
        for param_name, steering_vector in steering_vectors.items():
            if param_name in steered_model.state_dict():
                param = steered_model.state_dict()[param_name]
                # Ensure the vector is a tensor and on the correct device
                steer_tensor = torch.from_numpy(steering_vector).to(param.device, dtype=param.dtype)
                
                # START Jacob change
                steer_tensor_unit = steer_tensor / steer_tensor.norm()
                overlap = torch.dot(param.view(-1), steer_tensor_unit.view(-1))
                param.add_(steer_tensor * overlap, alpha=alpha)
                # END Jacob change
                
                print(f"  - Steered parameter '{param_name}'")
            
    return steered_model

def prepare_steering_vectors(config, target_digit, layers_to_steer, n_clusters, device):
    """
    Loads gradients, performs clustering, and extracts steering vectors for specified layers.
    """
    logging.info(f"Preparing steering vectors for target digit: {target_digit}")
    steering_vectors = {}
    
    try:
        stacked_gradients = load_stacked_gradients(config.gradients_dir, epoch=0)
        labels_path = os.path.join(config.gradients_dir, "true_labels_epoch_0.pt")
        true_labels = torch.load(labels_path).numpy()
    except FileNotFoundError as e:
        logging.error(f"Could not load required gradient/label files: {e}. Please run `train.py` first.")
        return None

    for param_name in layers_to_steer:
        logging.info(f"--- Processing parameter: {param_name} ---")
        if param_name not in stacked_gradients:
            logging.warning(f"Gradients for {param_name} not found, skipping.")
            continue

        param_gradients = stacked_gradients[param_name]
        
        logging.info(f"Running K-means for {param_name}...")
        kmeans_results = kmeans_gradients_gpu(
            param_gradients, n_clusters=n_clusters, random_state=config.random_seed, device=device
        )
        
        all_cluster_labels = kmeans_results['all_cluster_labels']
        best_cluster_id, max_digit_proportion = -1, -1

        for i in range(n_clusters):
            cluster_indices = np.where(all_cluster_labels == i)[0]
            if len(cluster_indices) == 0: continue
            
            digit_counts = Counter(true_labels[cluster_indices])
            if digit_counts[target_digit] > 0:
                proportion = digit_counts[target_digit] / len(cluster_indices)
                if proportion > max_digit_proportion:
                    max_digit_proportion, best_cluster_id = proportion, i
        
        if best_cluster_id != -1:
            logging.info(f"Found best cluster for digit {target_digit}: Cluster {best_cluster_id} "
                         f"with {max_digit_proportion:.2%} target digits.")
            
            cluster_center_grad = kmeans_results['cluster_centers'][best_cluster_id]
            param_shape = param_gradients.shape[1:]
            steering_vectors[param_name] = cluster_center_grad.reshape(param_shape)
            logging.info(f"Saved steering vector for {param_name}")
        else:
            logging.warning(f"Could not find a cluster for digit {target_digit} in {param_name}.")

    return steering_vectors

def main(args):
    # Setup
    config = Config()
    device = config.device
    
    # Use an absolute path for the output directory for robustness
    output_dir = os.path.abspath("steering_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Model
    logging.info("Loading trained model...")
    model = MLP(
        input_size=config.input_size,
        hidden_sizes=config.hidden_sizes,
        output_size=config.output_size,
        dtype=config.dtype
    ).to(device)
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    
    # Load Test Data & Evaluate Original Model
    logging.info("Loading MNIST test data and evaluating original model...")
    test_loader = get_data_loaders(config)
    criterion = nn.CrossEntropyLoss()
    original_accuracy, original_predictions = test_epoch(model, test_loader, criterion, device, config)
    logging.info(f"Accuracy of original model: {original_accuracy:.2f}%")

    # Visualize Original Model Predictions
    plt.figure(figsize=(10, 5))
    sns.countplot(x=original_predictions, palette="viridis")
    plt.title("Original Model Prediction Distribution")
    plt.xlabel("Predicted Digit")
    plt.ylabel("Count")
    save_path = os.path.join(output_dir, "original_prediction_distribution.png")
    plt.savefig(save_path, dpi=300)
    logging.info(f"Saved original prediction distribution to {save_path}")
    plt.close()
    
    # --- Prepare Steering Vectors ---
    steering_vectors = prepare_steering_vectors(config, args.target_digit, args.layers_to_steer, args.n_clusters, device)

    if not steering_vectors:
        logging.error("Failed to prepare steering vectors. Aborting.")
        return

    # --- Steer The Model ---
    steered_model = steer_model_with_vectors(model, steering_vectors, args.alpha, device)
    
    # --- Evaluate Steered Model ---
    logging.info("Evaluating steered model...")
    steered_accuracy, steered_predictions = test_epoch(steered_model, test_loader, criterion, device, config)
    
    # --- Visualize Post-Steering Results ---
    desc = f"Digit '{args.target_digit}'"
    plt.figure(figsize=(10, 5))
    sns.countplot(x=steered_predictions, palette="magma")
    title = (
        f"Post-Steering Prediction Distribution (Steered on {desc})\n"
        f"Steered Params: {', '.join(args.layers_to_steer)} | $\\alpha = {args.alpha}$"
    )
    plt.title(title, fontsize=10)
    plt.xlabel("Predicted Digit")
    plt.ylabel("Count")
    save_path = os.path.join(output_dir, "post_steering_distribution.png")
    plt.savefig(save_path, dpi=300)
    logging.info(f"Saved post-steering distribution to {save_path}")
    plt.close()

    plot_distribution_difference(original_predictions, steered_predictions, output_dir, desc, ", ".join(args.layers_to_steer), args.alpha)

    # --- Final Report ---
    print("\n--- Steering Experiment Summary ---")
    print(f"Steered {len(steering_vectors)} parameter(s) towards digit '{args.target_digit}': {', '.join(steering_vectors.keys())}")
    print(f"Steering Strength (alpha): {args.alpha}")
    print("-" * 30)
    print(f"Original Model Accuracy: {original_accuracy:.2f}%")
    print(f"Steered Model Accuracy:  {steered_accuracy:.2f}%")
    print(f"Accuracy Change: {steered_accuracy - original_accuracy:+.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform post-training gradient clustering and model steering.")
    parser.add_argument('--target_digit', type=int, default=9, help='The digit to find representative clusters for and steer towards.')
    parser.add_argument('--layers_to_steer', nargs='+', default=['network.3.bias'], help='List of parameter names to steer.')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for K-means.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Steering strength.')
    
    args = parser.parse_args()
    main(args) 