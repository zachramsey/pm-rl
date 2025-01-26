'''
Derived from: github.com/jiahaoli57/LSRE-CAAN/blob/main/LSRE_CAAN.py
'''

import torch
from torch import nn
from einops import rearrange
import numpy as np


class QuickGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, q_dim, kv_dim):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads

        self.q_linear = nn.Linear(q_dim, inner_dim, bias=False)
        self.k_linear = nn.Linear(kv_dim, inner_dim, bias=False)
        self.v_linear = nn.Linear(kv_dim, inner_dim, bias=False)
        self.out_linear = nn.Linear(inner_dim, q_dim)

    def forward(self, q, kv, is_causal = False):
        ''' ### Forward pass of Attention
        Args:
            q (torch.Tensor): Query tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
            kv (torch.Tensor): Key-Value tensor of shape (batch_dim*asset_dim, window_size, feat_dim)
            is_causal (bool): Whether to apply causal masking
        Returns:
            out (torch.Tensor): Output tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
        '''
        query = self.q_linear(q)    # (b*a, n, l) -> (b*a, n, h*d)
        key = self.k_linear(kv)     # (b*a, w, f) -> (b*a, w, h*d)
        value = self.v_linear(kv)   # (b*a, w, f) -> (b*a, w, h*d)
        
        query = rearrange(query, 'b w (h d) -> (b h) w d', h=self.num_heads)    # (b*a, n, h*d) -> (b*a*h, n, d)
        key = rearrange(key, 'b w (h d) -> (b h) w d', h=self.num_heads)        # (b*a, w, h*d) -> (b*a*h, w, d)
        value = rearrange(value, 'b w (h d) -> (b h) w d', h=self.num_heads)    # (b*a, w, h*d) -> (b*a*h, w, d)

        # (b*a*h, n, d) x (b*a*h, w, d) -> (b*a*h, n, w)
        scores = torch.einsum('b i d, b j d -> b i j', query, key) * self.scale

        # Derived from: pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        if is_causal:
            mask = torch.ones(q.shape[-2], kv.shape[-2], dtype=torch.bool).tril(diagonal=0)
            scores.masked_fill_(mask.logical_not(), float('-inf'))
            scores.to(query.dtype)

        # (b*a*h, n, w) -> (b*a*h, n, w)
        attn = torch.softmax(scores, dim=-1)

        # (b*a*h, n, w) x (b*a*h, w, d) -> (a*h, n, d)
        out = torch.einsum('b i j, b j d -> b i d', attn, value)
        
        # (b*a*h, n, d) -> (b*a, n, h*d)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)

        # (b*a, n, h*d) -> (b*a, n, l)
        out = self.out_linear(out)
        
        return out
    

class AttentionBlock(nn.Module):
    def __init__(self, num_heads, head_dim, q_dim, kv_dim=None):
        super().__init__()
        inner_dim = np.power(2, np.ceil(np.log2(q_dim))).astype(int)    # Next power of 2

        self.norm_q = nn.LayerNorm(q_dim)
        if kv_dim is None: kv_dim = q_dim
        else: self.norm_kv = nn.LayerNorm(kv_dim)

        self.attn = Attention(num_heads, head_dim, q_dim, kv_dim)
        self.norm_z = nn.LayerNorm(q_dim)
        self.ff = nn.Sequential(
            nn.Linear(q_dim, inner_dim),
            QuickGELU(),
            nn.Linear(inner_dim, q_dim)
        )

    def forward(self, q, kv = None):
        ''' ### Forward pass of AttentionBlock
        Args:
            q (torch.Tensor): Query tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
            kv (torch.Tensor): Key-Value tensor of shape (batch_dim*asset_dim, window_size, feat_dim)
        Returns:
            z (torch.Tensor): Output tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
        '''
        q_norm = self.norm_q(q)
        kv_norm = q_norm if kv is None else self.norm_kv(kv)
        z = self.attn(q_norm, kv_norm)
        z = z + q
        z = z + self.ff(self.norm_z(z))
        return z
