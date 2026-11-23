"""Transformer encoder with multi-head self-attention for 2-D feature maps."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_2d(h, w, dim, device):
    """Fixed 2-D sine/cosine positional encoding -> (h*w, dim). Works for any h, w."""
    assert dim % 4 == 0, "embed_dim must be divisible by 4"
    q = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(q, device=device, dtype=torch.float32) / q))
    ys = torch.arange(h, device=device, dtype=torch.float32)[:, None] * omega[None]
    xs = torch.arange(w, device=device, dtype=torch.float32)[:, None] * omega[None]
    pe_y = torch.cat([ys.sin(), ys.cos()], 1)                       # (h, 2q)
    pe_x = torch.cat([xs.sin(), xs.cos()], 1)                       # (w, 2q)
    pe = torch.cat([pe_y[:, None, :].expand(h, w, 2 * q), pe_x[None, :, :].expand(h, w, 2 * q)], -1)
    return pe.reshape(h * w, dim)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "embed_dim must be divisible by heads"
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                                           # x: (B, N, C)
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        # softmax(QK^T / sqrt(d)) V  -- scaled dot-product self-attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class TransformerEncoder(nn.Module):
    """Feature map (B,C,H,W) -> tokens -> `depth` blocks -> feature map (B,C,H,W)."""
    def __init__(self, dim=256, depth=2, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)                            # (B, HW, C) tokens
        t = t + sincos_2d(H, W, C, x.device).to(t.dtype)[None]
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t).transpose(1, 2).reshape(B, C, H, W)
