"""
Small CNN model for CIFAR-10 classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFARCNN(nn.Module):
    def __init__(self, config):
        super(CIFARCNN, self).__init__()
        
        # Extract hyperparameters from config
        conv_channels = config.conv_channels
        conv_kernel_sizes = config.conv_kernel_sizes
        conv_padding = config.conv_padding
        pool_kernel_size = config.pool_kernel_size
        pool_stride = config.pool_stride
        fc_sizes = config.fc_sizes  # This is now a property that auto-calculates
        dropout_rate = config.dropout_rate
        dtype = getattr(config, 'dtype', torch.float32)  # Default to float32 if not specified
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        
        for i in range(len(conv_channels) - 1):
            # Conv layer
            conv = nn.Conv2d(
                conv_channels[i], 
                conv_channels[i + 1], 
                kernel_size=conv_kernel_sizes[i], 
                padding=conv_padding[i],
                dtype=dtype
            )
            self.conv_layers.append(conv)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size after convolutions and pooling
        # Input: 32x32x3
        # After each conv+pool: size gets halved
        # After 3 conv+pool operations: 32 -> 16 -> 8 -> 4
        # Final feature map: 4x4x128 = 2048 features
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_sizes) - 1):
            fc = nn.Linear(fc_sizes[i], fc_sizes[i + 1], dtype=dtype)
            self.fc_layers.append(fc)
    
    def forward(self, x):
        # Convolutional layers with ReLU
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for i, fc in enumerate(self.fc_layers[:-1]):
            x = F.relu(fc(x))
            x = self.dropout(x)
        
        # Final layer (no activation, no dropout)
        x = self.fc_layers[-1](x)
        
        return x
    
    def get_features(self, x):
        """Get features from the last hidden layer before classification"""
        # Convolutional layers with ReLU
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Get features from the last hidden FC layer
        for i, fc in enumerate(self.fc_layers[:-1]):
            x = F.relu(fc(x))
            if i < len(self.fc_layers) - 2:  # Apply dropout to all but last hidden layer
                x = self.dropout(x)
        
        return x 