#!/usr/bin/env python3
"""
Script to generate true labels for MNIST dataset.
This creates the true_labels_epoch_0.pt file needed for t-SNE visualization.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import os
from config import Config

def generate_true_labels():
    """Generate and save true labels from MNIST training dataset."""
    
    print("Loading MNIST training dataset...")
    
    # Use the same transform as in training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST training dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Extract all true labels
    true_labels = []
    for _, label in train_dataset:
        true_labels.append(label)
    
    true_labels_tensor = torch.tensor(true_labels)
    print(f"Extracted {len(true_labels_tensor)} true labels")
    print(f"Label distribution: {torch.bincount(true_labels_tensor)}")
    
    # Save to the gradients directory
    config = Config()
    os.makedirs(config.gradients_dir, exist_ok=True)
    
    save_path = os.path.join(config.gradients_dir, "true_labels_epoch_0.pt")
    torch.save(true_labels_tensor, save_path)
    print(f"Saved true labels to: {save_path}")
    
    return true_labels_tensor

if __name__ == "__main__":
    labels = generate_true_labels()
    print("True labels generation completed!") 