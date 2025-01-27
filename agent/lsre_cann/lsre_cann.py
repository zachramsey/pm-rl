'''
Derived from: github.com/jiahaoli57/LSRE-CAAN/blob/main/LSRE_CAAN.py
'''

import torch
import torch.nn as nn
from einops import rearrange, repeat

from config.base import NUM_ASSETS
from config.lsre_cann import NUM_LATENTS, LATENT_DIM, NUM_CROSS_HEADS, CROSS_HEAD_DIM, NUM_SELF_HEADS, SELF_HEAD_DIM, DEPTH, DROPOUT
from agent.lsre_cann.attention import AttentionBlock

# class PositionalEncoding(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         position = torch.arange(cfg["window_size"]).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, cfg["feat_dim"], 2) * -(np.log(10000.0) / cfg["feat_dim"]))
#         pe = torch.zeros(cfg["window_size"], cfg["feat_dim"])
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.pe = nn.Buffer(pe)

#     def forward(self, x):
#         ''' ### Forward pass of PositionalEncoding
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_dim*asset_dim, window_size, feat_dim)
#         Returns:
#             x (torch.Tensor): Output tensor of shape (batch_dim*asset_dim, window_size, feat_dim)
#         '''
#         x = rearrange(x, 'b w f -> w b f')
#         x = x + self.pe[:x.size(0)]
#         x = rearrange(x, 'w b f -> b w f')
#         return x


class LSRE_Encode(nn.Module):
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


class CANN_Encode(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = LATENT_DIM ** -0.5
        
        self.q_linear = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.k_linear = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.v_linear = nn.Linear(LATENT_DIM, LATENT_DIM)

    def forward(self, z):
        ''' ### Forward pass of CANN
        Args:
            z (torch.Tensor): Latent tensor of shape (batch_dim, asset_dim, latent_dim)
        Returns:
            h (torch.Tensor): Hidden state of shape (batch_dim, asset_dim, latent_dim)
        '''
        q = self.q_linear(z)    # (batch_dim, asset_dim, latent_dim)
        k = self.k_linear(z)    # (batch_dim, asset_dim, latent_dim)
        v = self.v_linear(z)    # (batch_dim, asset_dim, latent_dim)

        # (batch_dim, asset_dim, latent_dim) x (batch_dim, latent_dim, asset_dim) 
        beta = torch.matmul(q, k.transpose(1, 2)) * self.scale  # (batch_dim, asset_dim, asset_dim)
        beta = torch.softmax(beta, dim=-1)

        # (batch_dim, asset_dim, asset_dim) x (batch_dim, asset_dim, latent_dim)
        h = torch.matmul(beta, v)   # (batch_dim, asset_dim, latent_dim)
        
        return h


class LSRE_CANN(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        # self.pos_enc = PositionalEncoding(cfg)
        # self.pos_emb = nn.Embedding(cfg["window_size"], cfg["feat_dim"])  #NOTE Original implementation
        self.lsre = LSRE_Encode(feat_dim)
        self.dropout = nn.Dropout(DROPOUT)
        self.cann = CANN_Encode()

    def forward(self, x):
        ''' ### Forward pass of LSRE_CANN
        Args:
            x (torch.Tensor): Input tensor of shape (batch_dim, asset_dim, window_size, feat_dim)
            reset (bool): Whether to reset the latent tensor
        Returns:
            h (torch.Tensor): Hidden state of shape (batch_dim, asset_dim, latent_dim)
        '''
        # (asset_dim, window_size, feat_dim) -> (1, asset_dim, window_size, feat_dim)
        if x.ndim == 3: x = x.unsqueeze(0)

        # x = self.pos_enc(x)
        # pos_emb = self.pos_emb(torch.arange(self.window_size))    #NOTE Original implementation
        # x = x + rearrange(pos_emb, "n d -> () n d")

        # (batch_dim, asset_dim, window_size, feat_dim) -> (batch_dim, asset_dim, latent_dim)
        z = self.lsre(x)
        z = self.dropout(z)

        # (batch_dim, asset_dim, latent_dim) -> (batch_dim, asset_dim, latent_dim)
        h = self.cann(z)

        return h
        