import torch
import torch.nn as nn

from config.dsac import MIN_LOG_STD, MAX_LOG_STD

# class Actor(nn.Module):
#     def __init__(self, input_dim):
#         super(Actor, self).__init__()
#         self.lsre_cann = LSRE_CANN(input_dim)
#         self.input = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU())
#         self.hidden = nn.Sequential(nn.Linear(256, 256), nn.GELU())
#         self.output = nn.Sequential(nn.Linear(256, 2))

#     def forward(self, s):
#         ''' ### Forward pass of Actor
#         Args:
#             s (torch.Tensor): State tensor of shape (batch_dim, asset_dim, window_size, feature_dim)
#         Returns:
#             mu (torch.Tensor): Mean tensor of shape (batch_dim, asset_dim, 1)
#             std (torch.Tensor): Standard deviation tensor of shape (batch_dim, asset_dim, 1)
#         '''
#         x = self.attn(s)
#         x = x[:, -1, :]
#         x = self.input(x)
#         x = self.hidden(x)
#         x = self.output(x)
#         mu, log_std = torch.chunk(x, chunks=2, dim=-1)
#         std = torch.exp(torch.clamp(log_std, MIN_LOG_STD, MAX_LOG_STD))
#         return mu, std
    
#--------------------------------------------------------------------------------------------------------------

from config.dsac import HIDDEN_DIM, MIN_LOG_STD, MAX_LOG_STD
from config.lsre_cann import LATENT_DIM
from net.lsre_cann import LSRE_CANN
from net.attention import QuickGELU

class LSRE_CANN_Actor(nn.Module):
    def __init__(self, cfg):
        super(LSRE_CANN_Actor, self).__init__()
        self.repr = LSRE_CANN(cfg)

        self.mu_out = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            QuickGELU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        
        self.std_out = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            QuickGELU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, s):
        ''' ### Forward pass of Actor
        Args:
            s (torch.Tensor): State tensor of shape (batch_dim, asset_dim, window_dim, feature_dim)
        Returns:
            mu (torch.Tensor): Mean tensor of shape (batch_dim, asset_dim, 1)
            std (torch.Tensor): Standard deviation tensor of shape (batch_dim, asset_dim, 1)
        '''
        # (b, a, w, f) -> (b, a, d)
        x = self.repr(s)

        # (b, a, d) -> (b, a, 1)
        mu = self.mu_out(x)

        # (b, a, d) -> (b, a, 1)
        log_std = self.std_out(x)
        std = torch.exp(torch.clamp(log_std, MIN_LOG_STD, MAX_LOG_STD))

        return mu, std