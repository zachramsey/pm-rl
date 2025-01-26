
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.min_log_std = cfg["min_log_std"]
        self.max_log_std = cfg["max_log_std"]

        self.input = nn.Sequential(nn.Linear(cfg["latent_dim"], 256), nn.GELU())
        self.hidden1 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.hidden2 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.hidden3 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.output1 = nn.Sequential(nn.Linear(256, 2))

    def forward(self, s):
        ''' ### Forward pass of Actor
        Args:
            s (torch.Tensor): State tensor of shape (batch_dim, asset_dim, latent_dim)
        Returns:
            mu (torch.Tensor): Mean tensor of shape (batch_dim, asset_dim, 1)
            std (torch.Tensor): Standard deviation tensor of shape (batch_dim, asset_dim, 1)
        '''
        x = self.input(s)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.output1(x)
        mu, log_std = torch.chunk(x, chunks=2, dim=-1)
        std = torch.exp(torch.clamp(log_std, self.min_log_std, self.max_log_std))
        return mu, std
    
#--------------------------------------------------------------------------------------------------------------

from srl.lsre_cann import LSRE_CANN

class LSRE_CANN_Actor(nn.Module):
    def __init__(self, cfg):
        super(LSRE_CANN_Actor, self).__init__()
        
        self.min_log_std = cfg["min_log_std"]
        self.max_log_std = cfg["max_log_std"]
        
        self.repr = LSRE_CANN(cfg)
        self.out = nn.Linear(cfg["latent_dim"], 2)

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

        # (b, a, d) -> (b, a, 2)
        x = self.out(x)

        mu, log_std = torch.chunk(x, chunks=2, dim=-1)
        std = torch.exp(torch.clamp(log_std, self.min_log_std, self.max_log_std))

        return mu, std