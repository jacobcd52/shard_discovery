#!/usr/bin/env python3
"""
Script to generate true labels for CIFAR-10 dataset.
This creates the true_labels_epoch_0.pt file needed for t-SNE visualization.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import os
from cifar_config import CIFARConfig

def generate_cifar_true_labels():
    """Generate and save true labels from CIFAR-10 training dataset."""
    
    config = CIFARConfig()
    print("Loading CIFAR-10 training dataset...")
    
    # Use the same transform as in training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean and std
    ])
    
    # Load original CIFAR-10 training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    print("Extracting true labels from CIFAR-10 training dataset...")
    # Extract all true labels
    true_labels = []
    for _, label in train_dataset:
        true_labels.append(label)
    
    true_labels_tensor = torch.tensor(true_labels)
    print(f"Extracted {len(true_labels_tensor)} true labels")
    print(f"Label distribution: {torch.bincount(true_labels_tensor)}")
    
    # Save to the gradients directory
    os.makedirs(config.gradients_dir, exist_ok=True)
    
    save_path = os.path.join(config.gradients_dir, "true_labels_epoch_0.pt")
    torch.save(true_labels_tensor, save_path)
    print(f"Saved true labels to: {save_path}")
    
    return true_labels_tensor

if __name__ == "__main__":
    labels = generate_cifar_true_labels()
    print("CIFAR-10 true labels generation completed!") 