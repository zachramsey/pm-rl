
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        
        self.input = nn.Sequential(nn.Linear(cfg["latent_dim"]+1, 256), nn.GELU())
        self.hidden1 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.hidden2 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.hidden3 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.output1 = nn.Sequential(nn.Linear(256, 2), nn.GELU())

    def forward(self, s, a):
        ''' ### Forward pass of Critic
        Args:
            s (torch.Tensor): State tensor of shape (batch_dim, asset_dim, latent_dim)
            a (torch.Tensor): Action tensor of shape (batch_dim, asset_dim, 1)
        Returns:
            mu (torch.Tensor): Q-value tensor of shape (batch_dim, asset_dim, 1)
            log_std (torch.Tensor): Standard deviation tensor of shape (batch_dim, asset_dim)
        '''
        x = torch.cat([s, a], dim=-1)
        x = self.input(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.output1(x)
        mu, std = torch.chunk(x, chunks=2, dim=-1)
        log_std = nn.functional.softplus(std)
        return mu, log_std

#--------------------------------------------------------------------------------------------------------------

from srl.lsre_cann import LSRE_CANN
from srl.attention import AttentionBlock

class LSRE_CANN_Critic(nn.Module):
    def __init__(self, cfg):
        super(LSRE_CANN_Critic, self).__init__()
        
        self.repr = LSRE_CANN(cfg)
        self.attn = AttentionBlock(cfg["num_cross_heads"], cfg["cross_head_dim"], cfg["latent_dim"], 1)
        self.out = nn.Linear(cfg["latent_dim"], 2)

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