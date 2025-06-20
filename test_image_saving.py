#!/usr/bin/env python3
"""
Test script to verify MNIST image saving functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from config import Config

def test_image_saving():
    """Test that MNIST images are saved correctly during training."""
    
    config = Config()
    
    # Create data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of batches: {len(train_loader)}")
    
    # Initialize image saving
    saved_images = []
    saved_image_indices = []
    
    print("Testing image saving...")
    
    # Simulate the training loop for one epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Processing batch {batch_idx + 1}/{len(train_loader)}")
        
        # Save MNIST images for visualization (only for first epoch to save space)
        if config is not None:
            # Get the current batch images and their global indices
            batch_images = data.cpu()  # Move to CPU for saving
            current_count = len(saved_images)
            batch_indices = list(range(current_count, current_count + len(batch_images)))
            
            print(f"  Batch {batch_idx}: {len(batch_images)} images, indices {batch_indices[0]}-{batch_indices[-1]}")
            
            # Extend the saved images lists
            saved_images.extend(batch_images)
            saved_image_indices.extend(batch_indices)
            
            # Only process first few batches for testing
            if batch_idx >= 2:  # Test with just 3 batches
                # Force save for testing
                print(f"Saving {len(saved_images)} MNIST images for visualization...")
                os.makedirs(config.gradients_dir, exist_ok=True)
                
                image_path = os.path.join(config.gradients_dir, "mnist_images_epoch_0.pt")
                indices_path = os.path.join(config.gradients_dir, "mnist_image_indices_epoch_0.pt")
                
                torch.save(torch.stack(saved_images), image_path)
                torch.save(torch.tensor(saved_image_indices), indices_path)
                
                print("MNIST images saved successfully!")
                print(f"Images saved to: {image_path}")
                print(f"Indices saved to: {indices_path}")
                
                # Verify the files were created
                if os.path.exists(image_path):
                    print(f"✓ Image file created: {os.path.getsize(image_path)} bytes")
                else:
                    print("✗ Image file not created!")
                    
                if os.path.exists(indices_path):
                    print(f"✓ Indices file created: {os.path.getsize(indices_path)} bytes")
                else:
                    print("✗ Indices file not created!")
                break
        
        # Only process first few batches for testing
        if batch_idx >= 2:  # Test with just 3 batches
            break
    
    print(f"Total images collected: {len(saved_images)}")
    print(f"Total indices collected: {len(saved_image_indices)}")

if __name__ == "__main__":
    test_image_saving() 