#!/usr/bin/env python3
"""
Script to clean up gradient files when switching between different digit filtering configurations.
This is necessary because gradient tensor shapes change when the model output size changes.
"""

import os
import shutil
from config import Config

def clean_gradients():
    """Remove all gradient files to start fresh with new filtering configuration."""
    
    config = Config()
    gradients_dir = config.gradients_dir
    
    print("🧹 Cleaning Gradient Files")
    print("=" * 40)
    
    if config.filter_digits is not None:
        print(f"Current filtering: {config.filter_digits} ({len(config.filter_digits)} digits)")
        print(f"Model output size: {config.get_output_size()}")
    else:
        print("Current filtering: None (all 10 digits)")
        print("Model output size: 10")
    
    if os.path.exists(gradients_dir):
        print(f"\nFound gradient directory: {gradients_dir}")
        
        # List what will be deleted
        contents = []
        for root, dirs, files in os.walk(gradients_dir):
            for file in files:
                contents.append(os.path.join(root, file))
        
        if contents:
            print(f"Found {len(contents)} gradient files to delete:")
            for item in contents[:10]:  # Show first 10
                print(f"  {item}")
            if len(contents) > 10:
                print(f"  ... and {len(contents) - 10} more files")
            
            # Ask for confirmation
            response = input(f"\nDelete all gradient files? (y/n): ").strip().lower()
            
            if response == 'y':
                shutil.rmtree(gradients_dir)
                print(f"✅ Deleted gradient directory: {gradients_dir}")
                print("\nNext steps:")
                print("1. Run: python generate_true_labels.py")
                print("2. Run: python train.py")
                print("3. Open inspect_gradients.ipynb and run all cells")
            else:
                print("❌ Cleanup cancelled")
        else:
            print("No gradient files found - directory is already clean")
    else:
        print(f"No gradient directory found: {gradients_dir}")
        print("Nothing to clean!")

def clean_specific_files():
    """More targeted cleanup - remove only files with dimension mismatches."""
    config = Config()
    gradients_dir = config.gradients_dir
    
    print("🎯 Targeted Gradient Cleanup")
    print("=" * 40)
    
    expected_output_size = config.get_output_size()
    print(f"Expected output size: {expected_output_size}")
    
    if not os.path.exists(gradients_dir):
        print("No gradient directory found")
        return
    
    # This would require loading each file and checking dimensions
    # For now, recommend full cleanup for simplicity
    print("For dimension mismatch issues, recommend full cleanup:")
    print("Run: python clean_gradients.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--targeted":
        clean_specific_files()
    else:
        clean_gradients() 