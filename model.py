"""
MLP model for MNIST classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2, dtype=torch.float32):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size, dtype=dtype),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, output_size, dtype=dtype))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)
        return self.network(x)
    
    def get_features(self, x):
        """Get features from the last hidden layer before classification"""
        x = x.view(x.size(0), -1)
        
        # Pass through all layers except the last one
        for i, layer in enumerate(self.network[:-1]):
            x = layer(x)
        
        return x 