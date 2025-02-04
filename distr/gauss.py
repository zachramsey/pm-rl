import torch

class GaussDistribution:
    def __init__(self, mean, std):
        self.mean = mean
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(mean, std),
            reinterpreted_batch_ndims=1,
        )
        # self.act_high_lim = torch.tensor([1.0])
        # self.act_low_lim = torch.tensor([-1.0])

    def sample(self):
        action = self.gauss_distribution.sample()
        log_prob = self.gauss_distribution.log_prob(action).unsqueeze(-1)
        return action, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        log_prob = self.gauss_distribution.log_prob(action).unsqueeze(-1)
        return action, log_prob

    def log_prob(self, action) -> torch.Tensor:
        log_prob = self.gauss_distribution.log_prob(action)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return self.mean
        # return torch.clamp(self.mean, self.act_low_lim, self.act_high_lim)

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )