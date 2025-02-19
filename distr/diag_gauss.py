
import torch
import torch.nn as nn
from torch.distributions import Normal

# Derived from: github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py
class DiagGaussianDistribution:
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim, log_std_init=0.0):
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions, log_std):
        action_std = torch.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions):
        log_prob = self.distribution.log_prob(actions)
        return self.sum_independent_dims(log_prob)

    def entropy(self):
        return self.sum_independent_dims(self.distribution.entropy())

    def sample(self):
        return self.distribution.rsample()

    def mode(self):
        return self.distribution.mean
    
    def get_actions(self, deterministic=False):
        if deterministic:
            return self.mode()
        return self.sample()

    def actions_from_params(self, mean_actions, log_std, deterministic=False):
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions, log_std):
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob
    
    def sum_independent_dims(self, tensor):
        if len(tensor.shape) > 1:
            tensor = tensor.sum(dim=1)
        else:
            tensor = tensor.sum()
        return tensor