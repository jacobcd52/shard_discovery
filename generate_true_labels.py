#!/usr/bin/env python3
"""
Script to generate true labels for MNIST dataset.
This creates the true_labels_epoch_0.pt file needed for t-SNE visualization.
Supports filtered datasets - will generate labels matching the filtered training set.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import os
from config import Config
from utils import FilteredMNIST

def generate_true_labels():
    """Generate and save true labels from MNIST training dataset (with filtering support)."""
    
    config = Config()
    print("Loading MNIST training dataset...")
    
    # Use the same transform as in training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load original MNIST training dataset
    original_train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Apply same filtering as in training
    if config.filter_digits is not None:
        print(f"Applying digit filtering: {config.filter_digits}")
        label_mapping = config.get_label_mapping()
        train_dataset = FilteredMNIST(original_train_dataset, config.filter_digits, label_mapping)
        
        # Extract filtered and remapped labels
        true_labels = []
        original_labels = []  # Keep track of original digits too
        
        for i in range(len(train_dataset)):
            _, filtered_label = train_dataset[i]
            original_label = train_dataset.get_original_label(filtered_label)
            true_labels.append(filtered_label)
            original_labels.append(original_label)
        
        true_labels_tensor = torch.tensor(true_labels)
        original_labels_tensor = torch.tensor(original_labels)
        
        print(f"Extracted {len(true_labels_tensor)} filtered labels")
        print(f"Filtered label distribution: {torch.bincount(true_labels_tensor)}")
        print(f"Original digits: {sorted(set(original_labels))}")
        
        # Save both filtered and original labels
        os.makedirs(config.gradients_dir, exist_ok=True)
        
        # Filtered labels (for clustering analysis)
        filtered_save_path = os.path.join(config.gradients_dir, "true_labels_epoch_0.pt")
        torch.save(true_labels_tensor, filtered_save_path)
        print(f"Saved filtered labels to: {filtered_save_path}")
        
        # Original labels (for visualization)
        original_save_path = os.path.join(config.gradients_dir, "original_labels_epoch_0.pt")
        torch.save(original_labels_tensor, original_save_path)
        print(f"Saved original labels to: {original_save_path}")
        
        return true_labels_tensor, original_labels_tensor
        
    else:
        print("No digit filtering applied - using all digits")
        # Extract all true labels (no filtering)
        true_labels = []
        for _, label in original_train_dataset:
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
    labels = generate_true_labels()
    print("True labels generation completed!") 