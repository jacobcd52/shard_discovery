from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ─────────────────────────────── Attention ────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """Self‑attention that can be *global* (full causal) or *local* (windowed)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        attention_type: str = "g",  # "g" = global, "l" = local
        local_window: int = 128,      # tokens of left context for local attention
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert attention_type in {"g", "l"}, "attention_type must be 'g' or 'l'"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.attention_type = attention_type
        self.local_window = local_window

        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    # ---------------------------------------------------------------------
    def _local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a mask that restricts attention to `local_window` past tokens."""
        # Mask positions *further* than local_window to the left (and all future)
        # 1 = keep, 0 = block
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device))
        if self.local_window is None:
            return causal
        window_mask = torch.triu(causal, diagonal=-self.local_window)
        return window_mask

    # ---------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                 # [B, T, D]
        external_mask: Optional[torch.Tensor] = None,  # [B, 1, T, T] or None
    ) -> torch.Tensor:                  # returns [B, T, D]
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, T, T]

        # Build (or combine) masks ------------------------------------------------
        if self.attention_type == "l":
            local_mask = self._local_mask(seq_len, x.device)
            # broadcast to [1, 1, T, T] then to batch/head dims
            local_mask = local_mask.unsqueeze(0).unsqueeze(0)
            mask = local_mask if external_mask is None else external_mask * local_mask
        else:
            mask = external_mask

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights + 1e-9  # avoid NaN in backward
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(context)


# ───────────────────────────── Feed‑forward ───────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ─────────────────────────── Transformer block ───────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_type: str = "g",
        local_window: int = 128,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout, attention_type, local_window
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


# ─────────────────────────────── Transformer ─────────────────────────────────
class TinyStoriesTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 16,
        n_layers: int = 8,
        d_ff: int = 256,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        attention_pattern: Optional[List[str]] = None,  # e.g. ["g", "l", ...]
        local_window: int = 128,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        # Embeddings --------------------------------------------------------
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Default pattern = alternate g ↔ l
        if attention_pattern is None:
            attention_pattern = ["g" if i % 2 == 0 else "l" for i in range(n_layers)]
        assert len(attention_pattern) == n_layers, "attention_pattern length must equal n_layers"

        # Transformer blocks -----------------------------------------------
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    attention_type=attention_pattern[i],
                    local_window=local_window,
                )
                for i in range(n_layers)
            ]
        )

        # Output projection -------------------------------------------------
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Initialise weights -----------------------------------------------
        self.apply(self._init_weights)

    # ---------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(self.d_model)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ---------------------------------------------------------------------
    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

    # ---------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,          # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T]
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.size()
        device = input_ids.device

        # Embedding lookup --------------------------------------------------
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(position_ids))

        # Base causal mask --------------------------------------------------
        causal_mask = self._build_causal_mask(seq_len, device)
        if attention_mask is not None:
            # [B, T] -> [B, 1, 1, T]
            attn = attention_mask.unsqueeze(1).unsqueeze(1)
            mask = attn * causal_mask  # broadcasting
        else:
            mask = causal_mask

        # Transformer -------------------------------------------------------
        for block in self.blocks:
            x = block(x, mask)

        # Output ------------------------------------------------------------
        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
            )
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    # ---------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------------------------------------------------------------
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Greedy sampling + top‑k/p filtering (identical to the previous impl)."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids)
                logits = outputs["logits"][:, -1, :] / temperature

                # top‑k --------------------------------------------------
                if top_k > 0:
                    topk_val, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < topk_val[:, [-1]]] = -float("inf")

                # top‑p --------------------------------------------------
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cdf = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_idx_to_remove = cdf > top_p
                    sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
                    sorted_idx_to_remove[..., 0] = 0
                    idx_to_remove = sorted_idx_to_remove.scatter(1, sorted_idx, sorted_idx_to_remove)
                    logits[idx_to_remove] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if input_ids.size(-1) >= self.max_seq_len:
                    break
        self.train()
        return input_ids