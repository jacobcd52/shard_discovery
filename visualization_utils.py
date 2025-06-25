"""
Streamlined visualization utilities for token-specific gradient analysis.
Only supports token-specific data (precise attribution) with modern hover functionality.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
import torch
from typing import Dict, List, Optional, Any, Tuple
from umap import UMAP


def create_tsne_with_token_data(gradients: Dict[str, torch.Tensor], 
                               token_data: List[Dict],
                               layer_name: str,
                               max_samples: int = 2000,
                               perplexity: int = 30,
                               random_state: int = 42,
                               show_prediction_task: bool = True,
                               use_pca: bool = True,
                               pca_components: int = 50) -> go.Figure:
    """
    Create t-SNE visualization with token-specific hover information.
    
    Args:
        gradients: Dictionary mapping layer names to gradient tensors
        token_data: List of token data dictionaries
        layer_name: Name of the layer to visualize
        max_samples: Maximum number of samples to visualize
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
        show_prediction_task: Whether to show prediction task in hover
        use_pca: Whether to apply PCA preprocessing (recommended for high-dim data)
        pca_components: Number of PCA components to keep (ignored if use_pca=False)
    
    Returns:
        Plotly figure object
    """
    if layer_name not in gradients:
        raise ValueError(f"Layer '{layer_name}' not found in gradients")
    
    gradient_tensor = gradients[layer_name]
    
    # Handle tensor reshaping
    if gradient_tensor.dim() == 4:  # Conv layer: [samples, out_ch, in_ch, kernel]
        flat_grads = gradient_tensor.view(gradient_tensor.size(0), -1)
    elif gradient_tensor.dim() == 3:  # Some 3D tensor
        flat_grads = gradient_tensor.view(gradient_tensor.size(0), -1)
    elif gradient_tensor.dim() == 2:  # Linear layer: [samples, features]
        flat_grads = gradient_tensor
    else:
        flat_grads = gradient_tensor.view(gradient_tensor.size(0), -1)
    
    # Subsample if necessary
    n_samples = min(flat_grads.size(0), max_samples)
    if n_samples < flat_grads.size(0):
        indices = torch.randperm(flat_grads.size(0))[:n_samples]
        flat_grads = flat_grads[indices]
        sampled_token_data = [token_data[i] for i in indices.tolist()]
    else:
        sampled_token_data = token_data[:n_samples]
    
    # Ensure token data matches gradient data
    if len(sampled_token_data) != n_samples:
        print(f"⚠️  Token data length ({len(sampled_token_data)}) doesn't match gradient samples ({n_samples})")
        sampled_token_data = sampled_token_data[:n_samples]
    
    print(f"🔍 Computing t-SNE for {n_samples} token gradients...")
    print(f"Original gradient shape: {flat_grads.shape}")
    
    # Convert to numpy for sklearn
    grad_data = flat_grads.cpu().numpy()
    
    # Optional PCA preprocessing (recommended for high-dimensional data)
    if use_pca and grad_data.shape[1] > pca_components:
        print(f"🔧 Applying PCA preprocessing: {grad_data.shape[1]} → {pca_components} dimensions")
        pca = PCA(n_components=pca_components, random_state=random_state)
        grad_data = pca.fit_transform(grad_data)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"📊 PCA explained variance: {explained_var:.1%}")
    elif use_pca:
        print(f"ℹ️  Skipping PCA: data already has {grad_data.shape[1]} ≤ {pca_components} dimensions")
    else:
        print(f"⚠️  No PCA preprocessing (use_pca=False) - using {grad_data.shape[1]} dimensions")
    
    print(f"Final data shape for t-SNE: {grad_data.shape}")
    
    # Compute t-SNE
    perplexity = min(perplexity, (n_samples - 1) // 3)  # Ensure valid perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, verbose=1)
    embedding = tsne.fit_transform(grad_data)
    
    # Create hover text for token-specific data
    hover_texts = []
    current_tokens = []
    target_tokens = []
    contexts = []
    
    for token_info in sampled_token_data:
        if show_prediction_task:
            prediction = token_info.get('prediction_task', 'Unknown')
            hover_text = f"<b>Prediction:</b> {prediction}<br>"
        else:
            current = token_info.get('current_token_text', 'Unknown')
            target = token_info.get('target_token_text', 'Unknown')
            hover_text = f"<b>Current:</b> {current}<br><b>Target:</b> {target}<br>"
        
        context = token_info.get('context_text', 'No context')
        position = token_info.get('token_position', 'Unknown')
        sample_idx = token_info.get('sample_idx', 'Unknown')
        
        hover_text += f"<b>Context:</b> {context}<br>"
        hover_text += f"<b>Position:</b> {position}<br>"
        hover_text += f"<b>Sample:</b> {sample_idx}<br>"
        
        # Add original text if available and not too long
        original = token_info.get('original_text', '')
        if original and len(original) < 200:
            hover_text += f"<b>Story:</b> {original[:100]}..."
        
        hover_texts.append(hover_text)
        
        # For coloring
        current_tokens.append(token_info.get('current_token_text', 'Unknown'))
        target_tokens.append(token_info.get('target_token_text', 'Unknown'))
        contexts.append(token_info.get('context_text', 'Unknown'))
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'hover_text': hover_texts,
        'current_token': current_tokens,
        'target_token': target_tokens,
        'context': contexts
    })
    
    # Create figure with token-specific coloring
    fig = px.scatter(
        df, 
        x='x', 
        y='y',
        color='current_token',
        hover_name='current_token',
        title=f't-SNE Visualization: {layer_name} (Token-Specific Gradients)',
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
        width=900,
        height=700
    )
    
    # Update hover template to show custom text
    fig.update_traces(
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts
    )
    
    # Update layout with PCA info
    pca_info = f" (PCA: {grad_data.shape[1]}D)" if use_pca and grad_data.shape[1] < flat_grads.shape[1] else ""
    fig.update_layout(
        title={
            'text': f't-SNE Visualization: {layer_name}<br><sub>Token-Specific Gradients ({n_samples} samples{pca_info})</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=True,
        legend=dict(
            title="Current Token",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    return fig


def create_umap_with_token_data(gradients: Dict[str, torch.Tensor], 
                               token_data: List[Dict],
                               layer_name: str,
                               max_samples: int = 2000,
                               n_neighbors: int = 15,
                               min_dist: float = 0.1,
                               random_state: int = 42,
                               show_prediction_task: bool = True,
                               use_pca: bool = True,
                               pca_components: int = 50) -> go.Figure:
    """
    Create UMAP visualization with token-specific hover information.
    
    Args:
        gradients: Dictionary mapping layer names to gradient tensors
        token_data: List of token data dictionaries
        layer_name: Name of the layer to visualize
        max_samples: Maximum number of samples to visualize
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed for reproducibility
        show_prediction_task: Whether to show prediction task in hover
        use_pca: Whether to apply PCA preprocessing (recommended for high-dim data)
        pca_components: Number of PCA components to keep (ignored if use_pca=False)
    
    Returns:
        Plotly figure object
    """
    if layer_name not in gradients:
        raise ValueError(f"Layer '{layer_name}' not found in gradients")
    
    gradient_tensor = gradients[layer_name]
    
    # Handle tensor reshaping
    if gradient_tensor.dim() == 4:  # Conv layer: [samples, out_ch, in_ch, kernel]
        flat_grads = gradient_tensor.view(gradient_tensor.size(0), -1)
    elif gradient_tensor.dim() == 3:  # Some 3D tensor
        flat_grads = gradient_tensor.view(gradient_tensor.size(0), -1)
    elif gradient_tensor.dim() == 2:  # Linear layer: [samples, features]
        flat_grads = gradient_tensor
    else:
        flat_grads = gradient_tensor.view(gradient_tensor.size(0), -1)
    
    # Subsample if necessary
    n_samples = min(flat_grads.size(0), max_samples)
    if n_samples < flat_grads.size(0):
        indices = torch.randperm(flat_grads.size(0))[:n_samples]
        flat_grads = flat_grads[indices]
        sampled_token_data = [token_data[i] for i in indices.tolist()]
    else:
        sampled_token_data = token_data[:n_samples]
    
    # Ensure token data matches gradient data
    if len(sampled_token_data) != n_samples:
        print(f"⚠️  Token data length ({len(sampled_token_data)}) doesn't match gradient samples ({n_samples})")
        sampled_token_data = sampled_token_data[:n_samples]
    
    print(f"🔍 Computing UMAP for {n_samples} token gradients...")
    print(f"Original gradient shape: {flat_grads.shape}")
    
    # Convert to numpy for sklearn
    grad_data = flat_grads.cpu().numpy()
    
    # Optional PCA preprocessing (recommended for high-dimensional data)
    if use_pca and grad_data.shape[1] > pca_components:
        print(f"🔧 Applying PCA preprocessing: {grad_data.shape[1]} → {pca_components} dimensions")
        pca = PCA(n_components=pca_components, random_state=random_state)
        grad_data = pca.fit_transform(grad_data)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"📊 PCA explained variance: {explained_var:.1%}")
    elif use_pca:
        print(f"ℹ️  Skipping PCA: data already has {grad_data.shape[1]} ≤ {pca_components} dimensions")
    else:
        print(f"⚠️  No PCA preprocessing (use_pca=False) - using {grad_data.shape[1]} dimensions")
    
    print(f"Final data shape for UMAP: {grad_data.shape}")
    
    # Compute UMAP
    n_neighbors = min(n_neighbors, n_samples - 1)  # Ensure valid n_neighbors
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, verbose=True)
    embedding = reducer.fit_transform(grad_data)
    
    # Create hover text for token-specific data
    hover_texts = []
    current_tokens = []
    target_tokens = []
    
    for token_info in sampled_token_data:
        if show_prediction_task:
            prediction = token_info.get('prediction_task', 'Unknown')
            hover_text = f"<b>Prediction:</b> {prediction}<br>"
        else:
            current = token_info.get('current_token_text', 'Unknown')
            target = token_info.get('target_token_text', 'Unknown')
            hover_text = f"<b>Current:</b> {current}<br><b>Target:</b> {target}<br>"
        
        context = token_info.get('context_text', 'No context')
        position = token_info.get('token_position', 'Unknown')
        sample_idx = token_info.get('sample_idx', 'Unknown')
        
        hover_text += f"<b>Context:</b> {context}<br>"
        hover_text += f"<b>Position:</b> {position}<br>"
        hover_text += f"<b>Sample:</b> {sample_idx}<br>"
        
        # Add original text if available and not too long
        original = token_info.get('original_text', '')
        if original and len(original) < 200:
            hover_text += f"<b>Story:</b> {original[:100]}..."
        
        hover_texts.append(hover_text)
        
        # For coloring
        current_tokens.append(token_info.get('current_token_text', 'Unknown'))
        target_tokens.append(token_info.get('target_token_text', 'Unknown'))
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'hover_text': hover_texts,
        'current_token': current_tokens,
        'target_token': target_tokens
    })
    
    # Create figure with token-specific coloring
    fig = px.scatter(
        df, 
        x='x', 
        y='y',
        color='current_token',
        hover_name='current_token',
        title=f'UMAP Visualization: {layer_name} (Token-Specific Gradients)',
        labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
        width=900,
        height=700
    )
    
    # Update hover template to show custom text
    fig.update_traces(
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts
    )
    
    # Update layout with PCA info
    pca_info = f" (PCA: {grad_data.shape[1]}D)" if use_pca and grad_data.shape[1] < flat_grads.shape[1] else ""
    fig.update_layout(
        title={
            'text': f'UMAP Visualization: {layer_name}<br><sub>Token-Specific Gradients ({n_samples} samples{pca_info})</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=True,
        legend=dict(
            title="Current Token",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    return fig


def print_token_gradient_summary(gradients: Dict[str, torch.Tensor], 
                                token_data: List[Dict], 
                                layer_name: str = None):
    """Print a summary of token-specific gradient data"""
    print("🎯 Token-Specific Gradient Summary")
    print("=" * 50)
    
    total_tokens = len(token_data)
    print(f"Total token gradients: {total_tokens}")
    
    if token_data:
        # Analyze prediction tasks
        prediction_tasks = [item.get('prediction_task', 'Unknown') for item in token_data]
        unique_tasks = len(set(prediction_tasks))
        print(f"Unique prediction tasks: {unique_tasks}")
        
        # Analyze tokens
        current_tokens = [item.get('current_token_text', 'Unknown') for item in token_data]
        target_tokens = [item.get('target_token_text', 'Unknown') for item in token_data]
        
        unique_current = len(set(current_tokens))
        unique_targets = len(set(target_tokens))
        
        print(f"Unique current tokens: {unique_current}")
        print(f"Unique target tokens: {unique_targets}")
        
        # Show most common prediction tasks
        from collections import Counter
        task_counts = Counter(prediction_tasks)
        print(f"\nMost common prediction tasks:")
        for task, count in task_counts.most_common(5):
            print(f"  {task}: {count} times")
    
    print("\nGradient tensors:")
    for name, tensor in gradients.items():
        memory_mb = tensor.nelement() * tensor.element_size() / (1024**2)
        dimensions = tensor.view(tensor.size(0), -1).shape[1]
        print(f"  {name}: {tensor.shape} ({memory_mb:.1f} MB, {dimensions} dims)")
    
    if layer_name and layer_name in gradients:
        layer_tensor = gradients[layer_name]
        flat_dims = layer_tensor.view(layer_tensor.size(0), -1).shape[1]
        print(f"\nFocused layer: {layer_name}")
        print(f"  Shape: {layer_tensor.shape}")
        print(f"  Flattened dimensions: {flat_dims}")
        print(f"  Memory: {layer_tensor.nelement() * layer_tensor.element_size() / (1024**2):.1f} MB")
        print(f"  Data type: {layer_tensor.dtype}")
        print(f"  Device: {layer_tensor.device}")
        
        # PCA recommendation
        if flat_dims > 100:
            print(f"  💡 Recommendation: Use PCA preprocessing (current: {flat_dims}D → suggested: 50D)")
        else:
            print(f"  ✅ Dimensionality looks good for direct embedding")
    
    print(f"\n💡 Each gradient represents a specific token prediction task!")
    print(f"   Example: 'little' → 'girl' (predicting 'girl' after seeing 'little')")
    print(f"   This gives precise attribution compared to sequence-level gradients")

