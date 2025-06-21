"""
Main training script for CIFAR-10 CNN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import os
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from cifar_config import CIFARConfig
from cifar_model import CIFARCNN
from cifar_utils import plot_training_curves, visualize_cifar_samples, visualize_predictions, create_results_dir
from gradient_collector import PerSampleGradientCollector, compute_gradient_statistics

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
    """Create train and test data loaders for CIFAR-10"""
    # Define transforms for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean and std
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
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
    
    return train_loader, test_loader, train_dataset, test_dataset

def train_epoch(model, train_loader, criterion, optimizer, device, gradient_collector=None, config=None, saved_images=None, saved_image_indices=None, epoch=0, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize image saving for first epoch
    if epoch == 0:
        saved_images = saved_images or []
        saved_image_indices = saved_image_indices or []
        saved_true_labels = []
        saved_model_predictions = []
        print("Initializing CIFAR-10 image collection for visualization...")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=config.dtype if config else torch.float32), target.to(device)
        
        # Save CIFAR-10 images for visualization (only for first epoch to save space)
        if epoch == 0 and config is not None and saved_images is not None and saved_image_indices is not None:
            # Get the current batch images and their global indices
            batch_images = data.cpu().float()  # Convert to float for saving
            current_count = len(saved_images)
            batch_indices = list(range(current_count, current_count + len(batch_images)))
            
            # Extend the saved images lists
            saved_images.extend(batch_images)
            saved_image_indices.extend(batch_indices)
            
            # Save at the end of epoch
            if batch_idx == len(train_loader) - 1:
                print(f"Saving {len(saved_images)} CIFAR-10 images for visualization...")
                os.makedirs(config.gradients_dir, exist_ok=True)
                torch.save(torch.stack(saved_images), os.path.join(config.gradients_dir, f"cifar_images_epoch_{epoch}.pt"))
                torch.save(torch.tensor(saved_image_indices), os.path.join(config.gradients_dir, f"cifar_image_indices_epoch_{epoch}.pt"))
                print("CIFAR-10 images saved successfully!")
                print(f"Images saved to: {config.gradients_dir}/cifar_images_epoch_{epoch}.pt")
        
        # Collect per-sample gradients if enabled
        if gradient_collector is not None and config is not None and config.collect_gradients:
            gradient_collector.collect_gradients_for_batch(data, target, criterion)
            
            # Save gradients every batch if configured (memory intensive)
            if config.save_gradients_every_batch:
                gradient_collector.save_gradients(epoch=epoch, batch_idx=batch_idx)
                gradient_collector.clear_gradients()
        
        # Normal training step with autocast for mixed precision
        optimizer.zero_grad()
        
        if config and config.autocast_enabled and scaler is not None:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Direct bfloat16 training without autocast
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Collect true labels and model predictions for first epoch
        if epoch == 0 and config is not None:
            _, predicted = torch.max(output.data, 1)
            saved_true_labels.extend(target.cpu().tolist())
            saved_model_predictions.extend(predicted.cpu().tolist())
            
            # Save at the end of epoch
            if batch_idx == len(train_loader) - 1:
                print(f"Saving {len(saved_true_labels)} true labels and model predictions...")
                os.makedirs(config.gradients_dir, exist_ok=True)
                torch.save(torch.tensor(saved_true_labels), os.path.join(config.gradients_dir, f"true_labels_epoch_{epoch}.pt"))
                torch.save(torch.tensor(saved_model_predictions), os.path.join(config.gradients_dir, f"model_predictions_epoch_{epoch}.pt"))
                print("True labels and model predictions saved successfully!")
                print(f"Labels saved to: {config.gradients_dir}/true_labels_epoch_{epoch}.pt")
                print(f"Predictions saved to: {config.gradients_dir}/model_predictions_epoch_{epoch}.pt")
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100. * correct / total
    
    return epoch_loss, epoch_accuracy, saved_images, saved_image_indices

def test_epoch(model, test_loader, criterion, device, config):
    """Test for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=config.dtype if config else torch.float32), target.to(device)
            
            if config and config.autocast_enabled:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
            else:
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
    config = CIFARConfig()
    
    # Set random seed
    set_random_seed(config.random_seed)
    
    # Create results directory
    create_results_dir(config.results_dir)
    
    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(config)
    print(f"Training samples: {len(train_dataset) if hasattr(train_dataset, '__len__') else 'Unknown'}")
    print(f"Test samples: {len(test_dataset) if hasattr(test_dataset, '__len__') else 'Unknown'}")
    
    # Visualize some training samples
    print("Visualizing training samples...")
    visualize_cifar_samples(
        train_loader, 
        num_samples=config.num_samples_to_visualize,
        save_path=os.path.join(config.results_dir, 'cifar_samples.png')
    )
    
    # Create model
    print("Creating CIFAR-10 CNN model...")
    model = CIFARCNN(config).to(config.device, dtype=config.dtype)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Using dtype: {config.dtype}")
    print(f"Autocast enabled: {config.autocast_enabled}")
    
    # Initialize gradient collector if enabled
    gradient_collector = None
    if config.collect_gradients:
        print("Initializing gradient collector...")
        gradient_collector = PerSampleGradientCollector(model, config.gradients_dir)
        print(f"Gradient collection enabled. Gradients will be saved to {config.gradients_dir}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = None
    if config.autocast_enabled:
        scaler = GradScaler()
        print("Gradient scaler initialized for mixed precision training")
    
    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    print(f"Device: {config.device}")
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_test_accuracy = 0.0
    saved_images = None
    saved_image_indices = None
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print("-" * 20)
        
        # Train
        train_loss, train_accuracy, saved_images, saved_image_indices = train_epoch(
            model, train_loader, criterion, optimizer, config.device, 
            gradient_collector, config, saved_images, saved_image_indices, epoch, scaler
        )
        
        # Save gradients after epoch if enabled
        if gradient_collector is not None and config.save_gradients_every_epoch:
            print("Saving gradients for this epoch...")
            gradient_collector.save_gradients(epoch=epoch)
            
            # Compute and display gradient statistics
            print("Computing gradient statistics...")
            compute_gradient_statistics(config.gradients_dir, epoch)
            
            # Clear gradients to free memory
            gradient_collector.clear_gradients()
        
        # Test
        test_loss, test_accuracy = test_epoch(model, test_loader, criterion, config.device, config)
        
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
    print(f"  - {config.results_dir}/cifar_samples.png")
    print(f"  - {config.results_dir}/training_curves.png")
    print(f"  - {config.results_dir}/predictions.png")
    print(f"  - {config.model_save_path}")
    
    if config.collect_gradients:
        print(f"  - {config.gradients_dir}/ (per-sample gradients)")

if __name__ == "__main__":
    main() 