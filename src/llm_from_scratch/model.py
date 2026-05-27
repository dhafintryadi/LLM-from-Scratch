from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 64
    emb_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    drop_rate: float = 0.1
    qkv_bias: bool = False


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        if cfg.emb_dim % cfg.n_heads != 0:
            raise ValueError("emb_dim must be divisible by n_heads.")

        self.num_heads = cfg.n_heads
        self.head_dim = cfg.emb_dim // cfg.n_heads
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.W_query = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_key = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_value = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.dropout = nn.Dropout(cfg.drop_rate)
        mask = torch.triu(torch.ones(cfg.context_length, cfg.context_length), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, emb_dim = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, emb_dim)
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        if seq_len > self.cfg.context_length:
            raise ValueError("Input sequence exceeds configured context length.")

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
