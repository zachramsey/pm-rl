'''
Derived from: github.com/jiahaoli57/LSRE-CAAN/blob/main/LSRE_CAAN.py
'''

import torch
import torch.nn as nn
from einops import rearrange, repeat

from srl.attention import AttentionBlock

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
    def __init__(self, cfg):
        super().__init__()
        asset_dim = cfg['asset_dim']                # Number of assets
        feat_dim = cfg['feat_dim']                         # Number of features

        num_latents = cfg['num_latents']            # Number of latents
        latent_dim = cfg['latent_dim']              # Dimension of latents

        num_cross_heads = cfg['num_cross_heads']    # Number of cross attention heads
        cross_head_dim = cfg['cross_head_dim']      # Dimension of cross attention heads

        num_latent_heads = cfg['num_latent_heads']  # Number of self attention heads
        latent_head_dim = cfg['latent_head_dim']    # Dimension of self attention heads

        self.z = nn.Buffer(torch.randn(asset_dim, num_latents, latent_dim))
        self.cross_attn = AttentionBlock(num_cross_heads, cross_head_dim, latent_dim, feat_dim)
        self.self_attns = nn.ModuleList([
            AttentionBlock(num_latent_heads, latent_head_dim, latent_dim) 
            for _ in range(cfg['depth'])
        ])
        self.out = nn.Linear(num_latents, 1)

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
    def __init__(self, cfg):
        super().__init__()
        latent_dim = cfg['latent_dim']
        self.scale = latent_dim ** -0.5
        
        self.q_linear = nn.Linear(latent_dim, latent_dim)
        self.k_linear = nn.Linear(latent_dim, latent_dim)
        self.v_linear = nn.Linear(latent_dim, latent_dim)

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
    def __init__(self, cfg):
        super().__init__()
        # self.pos_enc = PositionalEncoding(cfg)
        # self.pos_emb = nn.Embedding(cfg["window_size"], cfg["feat_dim"])  #NOTE Original implementation
        self.lsre = LSRE_Encode(cfg)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.cann = CANN_Encode(cfg)

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
        