"""
Visualization utilities for gradient analysis with token information.
Contains functions to create interactive t-SNE plots and other visualizations.
"""

import torch
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Any, Tuple


def create_tsne_with_tokens(gradients, token_data, param_name, n_samples=None, perplexity=30):
    """
    Create t-SNE visualization with token hover information
    
    Args:
        gradients: Dictionary of gradient tensors
        token_data: List of token data with context information
        param_name: Name of the parameter to visualize
        n_samples: Number of samples to use (None for all)
        perplexity: t-SNE perplexity parameter
    
    Returns:
        fig: Plotly figure object
        tsne_result: t-SNE coordinates array
    """
    
    if param_name not in gradients:
        print(f"Parameter {param_name} not found in gradients")
        return None, None
    
    # Get gradients for this parameter
    param_grads = gradients[param_name]
    print(f"Original gradient shape: {param_grads.shape}")
    
    # Flatten gradients for t-SNE - move to CPU if needed
    if param_grads.device.type == 'cuda':
        flat_grads = param_grads.view(param_grads.size(0), -1).cpu().numpy()
        print("Moved gradients from GPU to CPU for t-SNE computation")
    else:
        flat_grads = param_grads.view(param_grads.size(0), -1).numpy()
    
    # Subsample if requested
    if n_samples and n_samples < flat_grads.shape[0]:
        indices = np.random.choice(flat_grads.shape[0], n_samples, replace=False)
        flat_grads = flat_grads[indices]
        sample_token_data = [token_data[i] for i in indices]
    else:
        sample_token_data = token_data
        indices = np.arange(len(token_data))
    
    print(f"Using {flat_grads.shape[0]} samples for t-SNE")
    
    # Standardize gradients
    scaler = StandardScaler()
    flat_grads_scaled = scaler.fit_transform(flat_grads)
    
    # Apply t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, verbose=1)
    tsne_result = tsne.fit_transform(flat_grads_scaled)
    
    # Prepare hover information with bold tokens
    hover_texts = []
    colors = []
    
    for i, sample_data in enumerate(sample_token_data):
        # Create hover text with token context
        hover_parts = []
        hover_parts.append(f"Sample {sample_data['sample_idx']}")
        hover_parts.append(f"Sequence length: {sample_data['sequence_length']}")
        hover_parts.append("")
        hover_parts.append("Text preview:")
        hover_parts.append(sample_data['original_text'][:150] + "...")
        hover_parts.append("")
        
        # Add token examples with context
        hover_parts.append("Token examples (context with <b>target</b> token):")
        for j, token_ctx in enumerate(sample_data['token_contexts'][:8]):  # Show first 8 tokens
            # Split context text and bold the target token
            context_tokens = token_ctx['context_text'].split()
            target_pos = token_ctx['target_position']
            
            if target_pos < len(context_tokens):
                context_tokens[target_pos] = f"<b>{context_tokens[target_pos]}</b>"
            
            formatted_context = " ".join(context_tokens)
            hover_parts.append(f"  {j}: {formatted_context}")
        
        hover_text = "<br>".join(hover_parts)
        hover_texts.append(hover_text)
        
        # Color by sequence length
        colors.append(sample_data['sequence_length'])
    
    # Create interactive plot
    fig = go.Figure(data=go.Scatter(
        x=tsne_result[:, 0],
        y=tsne_result[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            colorscale='Viridis',
            colorbar=dict(title="Sequence Length"),
            opacity=0.7
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Gradient Points'
    ))
    
    fig.update_layout(
        title=f't-SNE Visualization of Gradients: {param_name}',
        xaxis_title='t-SNE 1',
        yaxis_title='t-SNE 2',
        width=900,
        height=700,
        hovermode='closest'
    )
    
    return fig, tsne_result


def create_multiple_tsne_visualizations(gradient_tensors, token_data, save_dir, 
                                      keywords=['embed', 'attention', 'attn', 'lm_head'],
                                      n_samples=300, perplexity=30):
    """
    Create t-SNE visualizations for multiple parameters
    
    Args:
        gradient_tensors: Dictionary of gradient tensors
        token_data: List of token data
        save_dir: Directory to save HTML files
        keywords: Keywords to filter interesting parameters
        n_samples: Number of samples for visualization
        perplexity: t-SNE perplexity parameter
    
    Returns:
        figures: Dictionary of parameter names to figure objects
    """
    print("🎨 Creating t-SNE visualizations...")

    # Select interesting parameters to visualize
    params_to_visualize = []
    for name in gradient_tensors.keys():
        if any(keyword in name.lower() for keyword in keywords):
            params_to_visualize.append(name)

    # If no specific parameters found, use first few
    if not params_to_visualize:
        params_to_visualize = list(gradient_tensors.keys())[:3]

    print(f"Visualizing parameters: {params_to_visualize}")

    # Create visualizations
    figures = {}
    for param_name in params_to_visualize:
        print(f"\nCreating t-SNE for {param_name}...")
        fig, tsne_coords = create_tsne_with_tokens(
            gradient_tensors, 
            token_data, 
            param_name, 
            n_samples=min(n_samples, len(token_data)),
            perplexity=perplexity
        )
        
        if fig is not None:
            figures[param_name] = fig
            fig.show()
            
            # Save figure
            fig_path = os.path.join(save_dir, f'tsne_{param_name.replace(".", "_")}.html')
            fig.write_html(fig_path)
            print(f"Saved interactive plot to {fig_path}")

    print("\n✅ t-SNE visualizations complete!")
    
    return figures


def print_gradient_summary(collector, gradient_tensors=None, token_data=None, 
                         diversity_stats=None, figures=None, model_name=None):
    """
    Print a comprehensive summary of gradient collection and analysis
    
    Args:
        collector: The gradient collector instance
        gradient_tensors: Dictionary of gradient tensors (optional)
        token_data: List of token data (optional)
        diversity_stats: Gradient diversity statistics (optional)
        figures: Dictionary of visualization figures (optional)
        model_name: Name of the model (optional)
    """
    print("📋 GRADIENT COLLECTION SUMMARY")
    print("=" * 50)
    if model_name:
        print(f"Model: {model_name}")
    print(f"Samples processed: {collector.sample_count}")
    print(f"Batch size for saving: {collector.save_batch_size}")
    print(f"Token context window: ±{collector.token_context_window} tokens")
    print(f"Save directory: {collector.save_dir}")
    print(f"Batch directory: {collector.batch_dir}")

    print("\n📊 PROGRESSIVE SAVING BENEFITS:")
    print("=" * 35)
    print("✅ Memory efficient: Data saved in small batches")
    print("✅ Progress preserved: Intermediate results saved")
    print("✅ Large scale ready: Can handle thousands of samples")
    print("✅ Resume capability: Can continue from where you left off")

    if diversity_stats:
        print("\n📊 DIVERSITY ANALYSIS SUMMARY")
        print("=" * 30)

        high_similarity_params = []
        low_variance_params = []

        for param_name, stats in diversity_stats.items():
            if 'mean_similarity' in stats and stats['mean_similarity'] > 0.8:
                high_similarity_params.append(param_name)
            if ('gradient_norm_std' in stats and 'gradient_norm_mean' in stats and 
                stats['gradient_norm_std'] < stats['gradient_norm_mean'] * 0.1):
                low_variance_params.append(param_name)

        if high_similarity_params:
            print(f"⚠️  Parameters with high similarity (potential duplication):")
            for param in high_similarity_params:
                print(f"   - {param}")
        else:
            print("✅ Good gradient diversity across analyzed parameters")

        if low_variance_params:
            print(f"\n⚠️  Parameters with low variance:")
            for param in low_variance_params:
                print(f"   - {param}")
        else:
            print("\n✅ Good gradient variance across analyzed parameters")

    if figures:
        print("\n🎨 VISUALIZATION FILES CREATED:")
        print("=" * 35)
        for param_name in figures.keys():
            filename = f'tsne_{param_name.replace(".", "_")}.html'
            print(f"  📊 {filename}")

    print("\n💾 BATCH FILES LOCATION:")
    print("=" * 25)
    print(f"📁 {collector.batch_dir}")
    import glob
    batch_files = glob.glob(os.path.join(collector.batch_dir, '*'))
    print(f"📄 Total batch files: {len(batch_files)}")

    print("\n💡 RECOMMENDATIONS:")
    print("=" * 18)
    print("1. Open the HTML files in a web browser for interactive exploration")
    print("2. Hover over points in t-SNE plots to see token context with bolded target tokens")
    print("3. Look for clusters in the t-SNE plot - they may represent similar linguistic patterns")
    print("4. Batch files are preserved - you can reload data anytime with collector.load_all_batches()")
    print("5. To process more samples, just increase MAX_SAMPLES and run again")
    print("6. Memory usage is controlled by SAVE_BATCH_SIZE (smaller = less memory)")

    print("\n✅ Analysis complete! Progressive batch saving enabled efficient large-scale processing.") 