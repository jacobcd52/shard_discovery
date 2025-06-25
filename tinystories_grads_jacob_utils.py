import torch
import psutil
from typing import Tuple, List, Optional
from datasets import IterableDataset
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm 
import plotly.graph_objects as go
from umap import UMAP

def get_memory_usage_gb():
    """Returns memory usage in GB."""
    return psutil.virtual_memory().used / (1024 ** 3)

def collect_gradients(
    model: HookedTransformer,
    dataset: IterableDataset,
    layer: int,
    batch_size: int,
    n_tokens_to_collect: int,
    print_freq: int,
    max_seq_len: int,
    hover_context_chars: int = 20,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    Runs the model on the dataset and collects gradients of the loss with respect
    to the pre-residual stream activations at a specified layer.
    
    Returns a tuple of (gradients, hover_texts, tokens).
    """
    collected_gradients = []
    hover_texts = []
    collected_tokens = []
    tokens_processed = 0
    
    dataset_iter = iter(dataset)
    
    pbar = tqdm(total=n_tokens_to_collect, desc="Collecting Gradients")
    
    batch_idx = 0
    while tokens_processed < n_tokens_to_collect:
        batch_texts = []
        for _ in range(batch_size):
            try:
                # Assuming the dataset yields dictionaries with a 'text' key
                batch_texts.append(next(dataset_iter)['text'])
            except StopIteration:
                break
        
        if not batch_texts:
            print("Dataset exhausted.")
            break

        tokens = model.to_tokens(batch_texts)
        
        if tokens.shape[1] > max_seq_len:
            tokens = tokens[:, :max_seq_len]

        collected_tokens.append(tokens.cpu())

        # Generate hover texts for this batch
        for i in range(tokens.shape[0]):
            # Process one sequence at a time as per user feedback
            str_tokens = model.to_str_tokens(tokens[i], prepend_bos=False)
            safe_str_tokens = [
                tok.replace('<', '&lt;').replace('>', '&gt;') for tok in str_tokens
            ]
            for j in range(len(str_tokens)):
                before_str = ' '.join(safe_str_tokens[:j])
                bold_token_str = f"&lt;<b>{safe_str_tokens[j]}</b>&gt;"
                after_str = ' '.join(safe_str_tokens[j+1:])
                
                if before_str:
                    before_str += " "
                if after_str:
                    after_str = " " + after_str

                context_before = before_str[-hover_context_chars:]
                context_after = after_str[:hover_context_chars]

                ellipsis_before = "..." if len(before_str) > hover_context_chars else ""
                ellipsis_after = "..." if len(after_str) > hover_context_chars else ""
                
                hover_str = f"{ellipsis_before}{context_before}{bold_token_str}{context_after}{ellipsis_after}"
                hover_texts.append(hover_str)

        if hasattr(model.cfg, 'device') and model.cfg.device is not None:
            tokens = tokens.to(model.cfg.device)

        activations_storage = []
        def hook_fn(activation, hook):
            activation.retain_grad()
            activations_storage.append(activation)
        
        hook_name = f"blocks.{layer}.hook_resid_pre"
        
        model.zero_grad()
        loss = model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[(hook_name, hook_fn)]
        )
        loss.backward()

        for act in activations_storage:
            collected_gradients.append(act.grad.detach().cpu())
        
        tokens_in_batch = tokens.numel()
        tokens_processed += tokens_in_batch
        pbar.update(tokens_in_batch)

        batch_idx += 1
        if batch_idx % print_freq == 0:
            mem_usage = get_memory_usage_gb()
            pbar.set_postfix_str(f"CPU Mem: {mem_usage:.2f}GB")

    pbar.close()

    if not collected_gradients:
        return torch.empty(0, model.cfg.d_model), [], torch.empty(0)

    all_gradients = torch.cat(collected_gradients, dim=0)
    all_gradients = all_gradients.reshape(-1, model.cfg.d_model)
    
    all_tokens = torch.cat(collected_tokens, dim=0)
    all_tokens = all_tokens.reshape(-1)

    # Truncate if we've collected more than requested
    if all_gradients.shape[0] > n_tokens_to_collect:
        all_gradients = all_gradients[:n_tokens_to_collect]
        hover_texts = hover_texts[:n_tokens_to_collect]
        all_tokens = all_tokens[:n_tokens_to_collect]

    final_mem_usage = get_memory_usage_gb()
    print(f"\nFinished. Collected gradients for {all_gradients.shape[0]} tokens.")
    print(f"Final CPU Memory Usage: {final_mem_usage:.2f} GB")
    
    return all_gradients, hover_texts, all_tokens






def plot_umap_with_hover(
    data: torch.Tensor,
    hover_texts: List[str],
    title: str,
    tokens: Optional[torch.Tensor] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    opacity: float = 0.8,
):
    """
    Applies UMAP to the data and creates an interactive Plotly scatter plot.

    Args:
        data (torch.Tensor): The high-dimensional data (e.g., gradients).
        hover_texts (List[str]): A list of strings for hover labels.
        title (str): The title for the plot.
        tokens (Optional[torch.Tensor]): Optional tensor of token IDs to color the points.
        n_neighbors (int): UMAP n_neighbors parameter.
        min_dist (float): UMAP min_dist parameter.
        random_state (int): Random state for UMAP for reproducibility.
        opacity (float): Opacity of the scatter plot points (0.0 to 1.0).
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().to(torch.float32).numpy()

    print("Running UMAP...")
    umap_reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        metric='cosine'
    )
    embedding = umap_reducer.fit_transform(data)

    print("Creating plot...")
    fig = go.Figure()

    marker_properties = dict(
        size=5,
        opacity=opacity,
    )
    if tokens is not None:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        marker_properties['color'] = tokens
        marker_properties['colorscale'] = 'viridis'
        marker_properties['showscale'] = True
        marker_properties['colorbar'] = dict(title='Token ID')

    fig.add_trace(
        go.Scattergl(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode='markers',
            marker=marker_properties,
            text=hover_texts,
            hoverinfo='text',
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Rockwell"
        )
    )

    fig.show()
