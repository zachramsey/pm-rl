'''
Derived from: github.com/jiahaoli57/LSRE-CAAN/blob/main/LSRE_CAAN.py
'''

import torch
from torch import nn
from einops import rearrange
import numpy as np
from config.base import DROPOUT

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

        self.q_proj = nn.Linear(q_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, inner_dim, bias=False)

        # self.dropout = nn.Dropout(DROPOUT)

        self.out_proj = nn.Linear(inner_dim, q_dim)

    def forward(self, q, kv):#, is_causal = False):
        ''' ### Forward pass of Attention
        Args:
            q (torch.Tensor): Query tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
            kv (torch.Tensor): Key-Value tensor of shape (batch_dim*asset_dim, window_size, feat_dim)
            is_causal (bool): Whether to apply causal masking
        Returns:
            out (torch.Tensor): Output tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
        '''
        q = self.q_proj(q)      # (b*a, n, l) -> (b*a, n, h*d)
        k = self.k_proj(kv)     # (b*a, w, f) -> (b*a, w, h*d)
        v = self.v_proj(kv)     # (b*a, w, f) -> (b*a, w, h*d)
        
        # (b*a, n, h*d) -> (b*a, h, n, d) | (b*a, w, h*d) -> (b*a, h, w, d)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        # (b*a, h, n, d) x (b*a, h, d, w) -> (b*a, h, n, w)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        # NOTE: Causal masking?
        attn = torch.softmax(scores, dim=-1)
        # attn = self.dropout(attn)

        # (b*a, h, n, w) x (b*a, h, w, d) -> (b*a, h, n, d)
        out = (attn @ v)
        
        # (b*a, h, n, d) -> (b*a, n, h*d)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # (b*a, n, h*d) -> (b*a, n, l)
        out = self.out_proj(out)
        return out
    

class AttentionBlock(nn.Module):
    def __init__(self, num_heads, head_dim, q_dim, kv_dim=None):
        super().__init__()
        self.q_norm = nn.LayerNorm(q_dim)
        if kv_dim is not None:
            self.kv_norm = nn.LayerNorm(kv_dim)
        else:
            kv_dim = q_dim
        self.attn = Attention(num_heads, head_dim, q_dim, kv_dim)
        # self.dropout1 = nn.Dropout(DROPOUT)

        self.norm = nn.LayerNorm(q_dim)
        ff_dim = 4 * np.power(2, np.ceil(np.log2(q_dim))).astype(int)   # 4 times the input dim
        self.ff = nn.Sequential(
            nn.Linear(q_dim, ff_dim),
            QuickGELU(),
            nn.Linear(ff_dim, q_dim)
        )
        # self.dropout2 = nn.Dropout(DROPOUT)

    def forward(self, q, kv = None):
        ''' ### Forward pass of AttentionBlock
        Args:
            q (torch.Tensor): Query tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
            kv (torch.Tensor): Key-Value tensor of shape (batch_dim*asset_dim, window_size, feat_dim)
        Returns:
            z (torch.Tensor): Output tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
        '''
        q_norm = self.q_norm(q)
        kv_norm = q_norm if kv is None else self.kv_norm(kv)
        # q = q + self.dropout1(self.attn(q_norm, kv_norm))
        # q = q + self.dropout2(self.ff(self.norm(q)))
        q = q + self.attn(q_norm, kv_norm)
        q = q + self.ff(self.norm(q))
        return q
