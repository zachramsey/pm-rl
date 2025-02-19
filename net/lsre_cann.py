'''
Derived from: github.com/jiahaoli57/LSRE-CAAN/blob/main/LSRE_CAAN.py
'''

import torch
import torch.nn as nn
from einops import rearrange, repeat

from config.base import NUM_ASSETS, WINDOW_SIZE
from config.base import DEPTH, NUM_LATENTS, LATENT_DIM, NUM_CROSS_HEADS, CROSS_HEAD_DIM, NUM_SELF_HEADS, SELF_HEAD_DIM, DROPOUT
from net.attention import AttentionBlock

class LSRE(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()

        self.z = nn.Buffer(torch.randn(NUM_ASSETS, NUM_LATENTS, LATENT_DIM))
        self.cross_attn = AttentionBlock(NUM_CROSS_HEADS, CROSS_HEAD_DIM, LATENT_DIM, feat_dim)
        self.self_attns = nn.ModuleList([
            AttentionBlock(NUM_SELF_HEADS, SELF_HEAD_DIM, LATENT_DIM) 
            for _ in range(DEPTH)
        ])
        self.out = nn.Linear(NUM_LATENTS, 1)

    def forward(self, x):
        ''' ### Forward pass of LSRE
        Args:
            x (torch.Tensor): Input tensor of shape (batch_dim, asset_dim, window_size, feat_dim)
        Returns:
            z (torch.Tensor): Latent tensor of shape (batch_dim, asset_dim, latent_dim)
        '''
        batch_dim = x.shape[0]
        z = repeat(self.z, 'a n d -> b a n d', b=batch_dim)
        z = rearrange(z, 'b a n d -> (b a) n d')
        x = rearrange(x, 'b a w f -> (b a) w f')

        # (b*a, n, d), (b*a, w, f) -> (b*a, n, d)
        z = self.cross_attn(z, x)

        # (b*a, n, d) -> (b*a, n, d)
        for self_attn in self.self_attns:
            z = self_attn(z)

        # self.z = z

        # (b*a, n, d) -> (b*a, d, n)
        z = z.transpose(1, 2)
        
        # (b*a, d, n) -> (b*a, d, 1)
        z = self.out(z)

        # (b*a, d, 1) -> (b, a, d)
        z = rearrange(z, '(b a) d 1 -> b a d', b=batch_dim)

        return z

class CANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = LATENT_DIM ** -0.5
        
        self.q_linear = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.k_linear = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.v_linear = nn.Linear(LATENT_DIM, LATENT_DIM)

        self.out = nn.Linear(LATENT_DIM, 1)

    def forward(self, z):
        ''' ### Forward pass of CANN
        Args:
            z (torch.Tensor): Latent tensor of shape (batch_dim, asset_dim, latent_dim)
        Returns:
            h (torch.Tensor): Hidden state of shape (batch_dim, asset_dim, 1)
        '''
        q = self.q_linear(z)        # (B, A, D)
        k = self.k_linear(z)        # (B, A, D)
        v = self.v_linear(z)        # (B, A, D)

        # (B, A, D) x (B, D, A) -> (B, A, A)
        beta = torch.matmul(q, k.transpose(1, 2)) * self.scale
        beta = torch.softmax(beta, dim=-1).unsqueeze(-1)

        # (B, 1, A, A) x (B, A, D, 1) -> (B, A, A, D)
        h = v.unsqueeze(1) * beta

        # (B, A, A, D) -> (B, A, D)
        h = torch.sum(h, dim=2)
        
        # (B, A, D) -> (B, A, 1)
        h = self.out(h)
        
        return h

class LSRE_CANN(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.pos_emb = nn.Embedding(WINDOW_SIZE, feat_dim)
        self.lsre = LSRE(feat_dim)
        self.dropout = nn.Dropout(DROPOUT)
        self.cann = CANN()

    def forward(self, x):
        ''' ### Forward pass of LSRE_CANN
        Args:
            x (torch.Tensor): Input tensor of shape (batch_dim, asset_dim, window_size, feat_dim)
            reset (bool): Whether to reset the latent tensor
        Returns:
            h (torch.Tensor): Hidden state of shape (batch_dim, asset_dim, 1)
        '''
        # (asset_dim, window_size, feat_dim) -> (1, asset_dim, window_size, feat_dim)
        if x.ndim == 3: x = x.unsqueeze(0)

        pos_emb = self.pos_emb(torch.arange(WINDOW_SIZE, device=x.device))
        x = x + repeat(pos_emb, "n d -> b a n d", b=x.shape[0], a=x.shape[1])

        # (batch_dim, asset_dim, window_size, feat_dim) -> (batch_dim, asset_dim, latent_dim)
        z = self.lsre(x)
        z = self.dropout(z)

        # (batch_dim, asset_dim, latent_dim) -> (batch_dim, asset_dim, 1)
        h = self.cann(z)

        return h
        