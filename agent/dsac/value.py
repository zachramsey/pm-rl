import torch
import torch.nn as nn

# class Critic(nn.Module):
#     def __init__(self, input_dim):
#         super(Critic, self).__init__()
#         self.self_attn = AttentionBlock(4, 64, input_dim)
#         self.cross_attn = AttentionBlock(4, 64, input_dim, 1)
#         self.input = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU())
#         self.hidden = nn.Sequential(nn.Linear(256, 256), nn.GELU())
#         self.output = nn.Sequential(nn.Linear(256, 2), nn.GELU())

#     def forward(self, s, a):
#         ''' ### Forward pass of Critic
#         Args:
#             s (torch.Tensor): State tensor of shape (batch_dim, asset_dim, latent_dim)
#             a (torch.Tensor): Action tensor of shape (batch_dim, asset_dim, 1)
#         Returns:
#             mu (torch.Tensor): Q-value tensor of shape (batch_dim, asset_dim, 1)
#             log_std (torch.Tensor): Standard deviation tensor of shape (batch_dim, asset_dim)
#         '''
#         s = self.self_attn(s)
#         x = self.cross_attn(s, a)
#         x = self.input(x)
#         x = self.hidden(x)
#         x = self.output(x)
#         mu, std = torch.chunk(x, chunks=2, dim=-1)
#         log_std = nn.functional.softplus(std)
#         return mu, log_std

#--------------------------------------------------------------------------------------------------------------

from config.lsre_cann import NUM_CROSS_HEADS, CROSS_HEAD_DIM, LATENT_DIM
from agent.lsre_cann.lsre_cann import LSRE_CANN
from agent.lsre_cann.attention import AttentionBlock

class LSRE_CANN_Critic(nn.Module):
    def __init__(self, feat_dim):
        super(LSRE_CANN_Critic, self).__init__()
        self.repr = LSRE_CANN(feat_dim)
        self.attn = AttentionBlock(NUM_CROSS_HEADS, CROSS_HEAD_DIM, LATENT_DIM, 1)
        self.out = nn.Linear(LATENT_DIM, 2)

    def forward(self, s, a):
        ''' ### Forward pass of Critic
        Args:
            s (torch.Tensor): State tensor of shape (batch_dim, asset_dim, window_dim, feature_dim)
            a (torch.Tensor): Action tensor of shape (batch_dim, asset_dim, 1)
        Returns:
            mu (torch.Tensor): Q-value tensor of shape (batch_dim, asset_dim, 1)
            log_std (torch.Tensor): Standard deviation tensor of shape (batch_dim, asset_dim)
        '''
        # (b, a, w, f) -> (b, a, d)
        s = self.repr(s)

        # (b, a, d), (b, a, 1) -> (b, a, d)
        x = self.attn(s, a)

        # (b, a, d) -> (b, a, 2)
        x = self.out(x)
        
        mu, std = torch.chunk(x, chunks=2, dim=-1)
        log_std = nn.functional.softplus(std)
        
        return mu, log_std