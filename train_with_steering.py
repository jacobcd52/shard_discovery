import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import logging
from collections import Counter

from config import Config
from model import MLP
from gradient_loader import load_stacked_gradients
from k_means import kmeans_gradients_gpu
from gradient_steering_optimizer import SteeringVectorAdamW
from train import get_data_loaders, train_epoch, test_epoch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_steering_vectors(config, target_digit, layers_to_steer, n_clusters, device):
    """
    Loads gradients, performs clustering, and extracts steering vectors for specified layers.
    """
    logging.info(f"Preparing steering vectors for target digit: {target_digit}")
    steering_vectors = {}
    
    # Load all available gradients first
    try:
        stacked_gradients = load_stacked_gradients(config.gradients_dir, epoch=0)
    except FileNotFoundError as e:
        logging.error(f"Could not load gradients: {e}. Please run `train.py` first.")
        return None

    # Load true labels to identify clusters
    labels_path = os.path.join(config.gradients_dir, "true_labels_epoch_0.pt")
    if not os.path.exists(labels_path):
        logging.error(f"True labels file not found at {labels_path}")
        return None
    true_labels = torch.load(labels_path).numpy()

    for param_name in layers_to_steer:
        logging.info(f"--- Processing parameter: {param_name} ---")
        if param_name not in stacked_gradients:
            logging.warning(f"Gradients for {param_name} not found, skipping.")
            continue

        param_gradients = stacked_gradients[param_name]
        
        # Cluster the gradients for this parameter
        logging.info(f"Running K-means for {param_name}...")
        kmeans_results = kmeans_gradients_gpu(
            param_gradients,
            n_clusters=n_clusters,
            random_state=config.random_seed,
            device=device
        )
        
        # Find the cluster most associated with the target digit
        all_cluster_labels = kmeans_results['all_cluster_labels']
        best_cluster_id = -1
        max_digit_proportion = -1

        for i in range(n_clusters):
            cluster_indices = np.where(all_cluster_labels == i)[0]
            if len(cluster_indices) == 0:
                continue
            
            labels_in_cluster = true_labels[cluster_indices]
            digit_counts = Counter(labels_in_cluster)
            
            if digit_counts[target_digit] > 0:
                proportion = digit_counts[target_digit] / len(cluster_indices)
                if proportion > max_digit_proportion:
                    max_digit_proportion = proportion
                    best_cluster_id = i
        
        if best_cluster_id != -1:
            logging.info(f"Found best cluster for digit {target_digit}: Cluster {best_cluster_id} "
                         f"with {max_digit_proportion:.2%} target digits.")
            
            # Get the average gradient (cluster center) for the best cluster
            cluster_center_grad = kmeans_results['cluster_centers'][best_cluster_id]
            param_shape = param_gradients.shape[1:] # Get the original parameter shape
            
            # Reshape and store the steering vector
            steering_vector = torch.from_numpy(cluster_center_grad.reshape(param_shape)).to(device, dtype=config.dtype)
            steering_vectors[param_name] = steering_vector
            logging.info(f"Added steering vector for {param_name} with shape {steering_vector.shape}")
        else:
            logging.warning(f"Could not find any cluster associated with digit {target_digit} "
                            f"for parameter {param_name}. Skipping.")

    return steering_vectors

def main(args):
    config = Config()
    device = config.device
    
    # --- 1. Prepare Steering Vectors ---
    layers_to_steer = ['network.0.weight', 'network.0.bias', 'network.3.weight', 'network.3.bias']
    steering_vectors = prepare_steering_vectors(config, args.target_digit, layers_to_steer, args.n_clusters, device)

    if not steering_vectors:
        logging.error("Failed to prepare steering vectors. Aborting training.")
        return

    # --- 2. Setup Model and Data ---
    logging.info("Setting up model and data loaders...")
    train_loader, test_loader, _, _ = get_data_loaders(config)
    
    # Re-initialize model to train from scratch
    model = MLP(
        input_size=config.input_size,
        hidden_sizes=config.hidden_sizes,
        output_size=config.output_size,
        dtype=config.dtype
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # --- 3. Initialize Steering Optimizer ---
    logging.info("Initializing SteeringVectorAdamW optimizer...")
    optimizer = SteeringVectorAdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        named_parameters=list(model.named_parameters()),
        steering_vectors=steering_vectors,
        alpha=args.alpha
    )

    # --- 4. Training Loop ---
    logging.info(f"Starting steered training for {config.num_epochs} epochs...")
    best_accuracy = 0.0

    for epoch in range(config.num_epochs):
        logging.info(f"--- Epoch {epoch+1}/{config.num_epochs} ---")
        
        # We need a slightly modified train_epoch that doesn't save gradients
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device, dtype=config.dtype), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step() # The optimizer handles the steering
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        
        # Evaluate on test set
        test_loss, test_accuracy = test_epoch(model, test_loader, criterion, device, config)
        
        logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                     f"Test Accuracy: {test_accuracy:.2f}%")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            logging.info(f"New best accuracy: {best_accuracy:.2f}%")

    logging.info("--- Steered Training Complete ---")
    logging.info(f"Final model accuracy with steering: {best_accuracy:.2f}%")
    logging.info(f"Steered towards digit '{args.target_digit}' with alpha={args.alpha} "
                 f"on {len(steering_vectors)} parameter(s).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with online gradient steering.")
    parser.add_argument('--target_digit', type=int, default=9, help='The digit to steer the model towards.')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters to use for finding steering vectors.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Steering strength.')
    
    args = parser.parse_args()
    main(args) 