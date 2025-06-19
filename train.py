"""
Main training script for MNIST MLP
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import os

from config import Config
from model import MLP
from utils import plot_training_curves, visualize_mnist_samples, visualize_predictions, create_results_dir

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(config):
    """Create train and test data loaders"""
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100. * correct / total
    
    return epoch_loss, epoch_accuracy

def test_epoch(model, test_loader, criterion, device):
    """Test for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = 100. * correct / total
    
    return epoch_loss, epoch_accuracy

def main():
    # Load configuration
    config = Config()
    
    # Set random seed
    set_random_seed(config.random_seed)
    
    # Create results directory
    create_results_dir(config.results_dir)
    
    # Get data loaders
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Visualize some training samples
    print("Visualizing training samples...")
    visualize_mnist_samples(
        train_loader, 
        num_samples=config.num_samples_to_visualize,
        save_path=os.path.join(config.results_dir, 'mnist_samples.png')
    )
    
    # Create model
    print("Creating MLP model...")
    model = MLP(
        input_size=config.input_size,
        hidden_sizes=config.hidden_sizes,
        output_size=config.output_size,
        dropout_rate=config.dropout_rate
    ).to(config.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    print(f"Device: {config.device}")
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_test_accuracy = 0.0
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print("-" * 20)
        
        # Train
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, config.device)
        
        # Test
        test_loss, test_accuracy = test_epoch(model, test_loader, criterion, config.device)
        
        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Save best model
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), config.model_save_path)
            print(f"New best model saved! Test accuracy: {test_accuracy:.2f}%")
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_test_accuracy:.2f}%")
    
    # Plot training curves
    print("Plotting training curves...")
    plot_training_curves(
        train_losses, test_losses, train_accuracies, test_accuracies,
        save_path=os.path.join(config.results_dir, 'training_curves.png')
    )
    
    # Load best model and visualize predictions
    print("Loading best model for prediction visualization...")
    model.load_state_dict(torch.load(config.model_save_path))
    visualize_predictions(
        model, test_loader, config.device,
        num_samples=config.num_samples_to_visualize,
        save_path=os.path.join(config.results_dir, 'predictions.png')
    )
    
    print(f"\nAll results saved to {config.results_dir}/")
    print("Files created:")
    print(f"  - {config.results_dir}/mnist_samples.png")
    print(f"  - {config.results_dir}/training_curves.png")
    print(f"  - {config.results_dir}/predictions.png")
    print(f"  - {config.model_save_path}")

if __name__ == "__main__":
    main() 