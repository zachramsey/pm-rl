
import os
import torch
import torch.nn as nn
from torch.optim import Adam

from config.pg import *
from config.base import LOG_DIR, COMISSION, RISK_FREE_RATE, REWARD, REWARD_SCALE, EPSILON

if POLICY == "LSRE-CANN":
    from net.lsre_cann import LSRE_CANN as Policy

class PG:
    def __init__(self, feat_dim):
        self.policy = Policy(feat_dim)
        self.optim = Adam(self.policy.parameters(), lr=LR)

        self.info = {
            "epoch": [],
            "loss": []
        }

    def training_mode(self, mode):
        if mode:
            self.policy.train()
        else:
            self.policy.eval()

    def act(self, s):
        ''' ### Act based on the policy network
        Args:
            s (torch.Tensor): State tensor of shape (asset_dim, window_size, feat_dim)
        Returns:
            a (torch.Tensor): Action tensor of shape (asset_dim, 1)
        '''
        with torch.no_grad():
            a = self.policy(s)
        return a

    def _reward(self, a, _v, _a, p):
        ''' ### Calculate the loss for a single step
        Args:
            a (torch.Tensor): New portfolio weights (batch_size, asset_dim, 1)
            _a (torch.Tensor): Prev portfolio weights (batch_size, asset_dim, 1)
            p (torch.Tensor): Relative price changes (batch_size, asset_dim, 1)
            _v (torch.Tensor): Previous portfolio value (batch_size, 1, 1)
        Returns:
            loss (torch.Tensor): Loss tensor ()
        '''
        
        # Normalize the weights
        if not torch.isclose(torch.sum(a), torch.tensor(1.0), atol=1e-6) or torch.min(a) < 0:
            a = torch.softmax(a, dim=1)

        # Calculate the transaction remainder factor
        # Derived from: github.com/ZhengyaoJiang/PGPortfolio/blob/master/pgportfolio/tools/trade.py
        if COMISSION > 0:
            mu_last = 1
            mu = 1 - 2*COMISSION + COMISSION**2
            while abs(mu - mu_last) > 1e-10:
                mu_last = mu
                numer = 1 - (COMISSION * _a[:, 0]) - (2*COMISSION - COMISSION**2) * torch.sum(torch.maximum(_a[:, 1:] - mu * a[:, 1:], ))
                denom = 1 - COMISSION * a[:, 0]
                mu = numer / denom
            _v = mu * _v

        # Calculate the new portfolio value
        portfolio = _v * (a * p)                        # (batch_size, asset_dim, 1)
        v = torch.sum(portfolio, dim=1, keepdim=True)   # (batch_size, 1, 1)

        # Calculate the returns for the current day
        ret = v / _v

        # Calculate the reward for the current day
        if REWARD == "returns":
            r = torch.mean(ret * REWARD_SCALE)
        elif REWARD == "log_returns":
            r = torch.mean(torch.log(ret) * REWARD_SCALE)
        elif REWARD == "sharpe_ratio":
            r = torch.mean(ret) / torch.std(ret) * REWARD_SCALE

        return r

        
    def update(self, epoch, s, a, r, _v, _a, p):
        ''' ### Update the policy network
        Args:
            epoch (int): Current epoch
            s (torch.Tensor): State (batch_size, asset_dim, window_size, feat_dim)
            a (torch.Tensor): Action (batch_size, asset_dim, 1)
            r (torch.Tensor): Reward (batch_size, 1, 1)
            _v (torch.Tensor): Previous value (batch_size, 1, 1)
            _a (torch.Tensor): Previous action (batch_size, asset_dim, 1)
            p (torch.Tensor): Target (batch_size, asset_dim, 1)
        '''
        self.optim.zero_grad()
        a = self.policy(s)
        if torch.rand(1) < EPSILON:
            a = torch.rand_like(a)
            a = torch.softmax(a, dim=1)
        loss = -self._reward(a, _v, _a, p)
        loss.backward()
        self.optim.step()

        self.info["epoch"].append(epoch)
        self.info["loss"].append(loss.item())

    def log_info(self):
        # Check that logs directory exists
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        # Write training information to log file
        with open(LOG_DIR + "latest.log", "a") as f:
            f.write("="*50 + "\n")
            for key, value in self.info.items():
                if key == "epoch":
                    f.write(f"Epoch {value[-1]}\n")
                    f.write("-"*50 + "\n")
                    f.write("Training Step:\n")
                else:
                    f.write(f"{key}: {value[-1]}\n")
            f.write("-"*50 + "\n")