"""
Utility functions for visualization and plotting
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import Dataset

def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies, save_path):
    """
    Plot training and test loss/accuracy curves
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Plot loss difference
    loss_diff = [abs(t - v) for t, v in zip(train_losses, test_losses)]
    ax3.plot(epochs, loss_diff, 'g-', label='|Train Loss - Test Loss|')
    ax3.set_title('Loss Difference (Overfitting Indicator)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.legend()
    ax3.grid(True)
    
    # Plot accuracy difference
    acc_diff = [abs(t - v) for t, v in zip(train_accuracies, test_accuracies)]
    ax4.plot(epochs, acc_diff, 'g-', label='|Train Acc - Test Acc|')
    ax4.set_title('Accuracy Difference (Overfitting Indicator)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Difference')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")

def visualize_mnist_samples(dataloader, num_samples=16, save_path=None):
    """
    Visualize MNIST samples as a grid
    """
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Take only the first num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Ensure tensors are on CPU and convert to numpy
    images = images.cpu().detach()
    labels = labels.cpu().detach()
    
    # Create grid
    grid = make_grid(images, nrow=4, normalize=True, padding=2)
    
    # Convert to numpy for plotting
    grid_np = grid.numpy().transpose((1, 2, 0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_np, cmap='gray')
    ax.set_title(f'MNIST Samples (Labels: {labels.tolist()})')
    ax.axis('off')
    
    # Add labels on each image
    for i, label in enumerate(labels):
        row = i // 4
        col = i % 4
        ax.text(col * 7.5 + 3.5, row * 7.5 + 3.5, str(label.item()), 
                ha='center', va='center', color='red', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MNIST samples visualization saved to {save_path}")
    
    plt.show()

def visualize_predictions(model, dataloader, device, num_samples=16, save_path=None):
    """
    Visualize model predictions on test samples
    """
    model.eval()
    data_iter = iter(dataloader)
    images, true_labels = next(data_iter)
    
    # Take only the first num_samples
    images = images[:num_samples]
    true_labels = true_labels[:num_samples]
    
    with torch.no_grad():
        images_device = images.to(device)
        outputs = model(images_device)
        _, predicted_labels = torch.max(outputs, 1)
    
    # Ensure tensors are on CPU and convert to numpy
    images = images.cpu().detach()
    true_labels = true_labels.cpu().detach()
    predicted_labels = predicted_labels.cpu().detach()
    
    # Create grid
    grid = make_grid(images, nrow=4, normalize=True, padding=2)
    grid_np = grid.numpy().transpose((1, 2, 0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid_np, cmap='gray')
    ax.set_title('Model Predictions on MNIST Test Samples')
    ax.axis('off')
    
    # Add labels on each image
    for i, (true_label, pred_label) in enumerate(zip(true_labels, predicted_labels)):
        row = i // 4
        col = i % 4
        color = 'green' if true_label == pred_label else 'red'
        ax.text(col * 7.5 + 3.5, row * 7.5 + 1.5, f'True: {true_label.item()}', 
                ha='center', va='center', color='blue', fontsize=10)
        ax.text(col * 7.5 + 3.5, row * 7.5 + 5.5, f'Pred: {pred_label.item()}', 
                ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    
    plt.show()

def create_results_dir(results_dir):
    """Create results directory if it doesn't exist"""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")

class FilteredMNIST(Dataset):
    """
    Wrapper for MNIST dataset that filters specific digits and remaps labels.
    
    Args:
        dataset: Original MNIST dataset
        filter_digits: List of digits to include (e.g., [0, 1, 2, 3, 4])
        label_mapping: Dictionary mapping original labels to new labels
    """
    
    def __init__(self, dataset, filter_digits=None, label_mapping=None):
        self.dataset = dataset
        self.filter_digits = filter_digits
        self.label_mapping = label_mapping or {}
        
        if filter_digits is not None:
            # Find indices of samples with desired digits
            self.filtered_indices = []
            self.filtered_labels = []
            
            for idx, (_, label) in enumerate(dataset):
                if label in filter_digits:
                    self.filtered_indices.append(idx)
                    # Map to new label space (0, 1, 2, ...)
                    new_label = self.label_mapping.get(label, label)
                    self.filtered_labels.append(new_label)
            
            print(f"Filtered dataset: {len(self.filtered_indices)} samples from {len(dataset)} total")
            print(f"Original digits: {sorted(set(filter_digits))}")
            print(f"New label mapping: {self.label_mapping}")
            
            # Print distribution
            unique_labels, counts = np.unique(self.filtered_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                original_digit = [k for k, v in self.label_mapping.items() if v == label][0]
                print(f"  Digit {original_digit} -> Label {label}: {count} samples")
        else:
            # No filtering - use all samples
            self.filtered_indices = list(range(len(dataset)))
            self.filtered_labels = [dataset[i][1] for i in self.filtered_indices]
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        data, _ = self.dataset[original_idx]
        label = self.filtered_labels[idx]
        return data, label
    
    def get_original_label(self, filtered_label):
        """Convert filtered label back to original digit"""
        if self.filter_digits is None:
            return filtered_label
        
        # Find original digit for this filtered label
        for original_digit, mapped_label in self.label_mapping.items():
            if mapped_label == filtered_label:
                return original_digit
        return filtered_label 