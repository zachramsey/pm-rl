
from config.base import REWARD, REWARD_SCALE, RISK_FREE_RATE
import numpy as np
import torch

class Reward:
    def __init__(self, env):
        self.env = env
        self.reward_map = {
            "returns": self.returns,
            "log_returns": self.log_returns,
            "sharpe_ratio": self.sharpe_ratio
        }

    def get_reward(self):
        if REWARD == "returns": return REWARD_SCALE * self.returns()
        elif REWARD == "log_returns": return REWARD_SCALE * self.log_returns()
        elif REWARD == "sharpe_ratio": return REWARD_SCALE * self.sharpe_ratio()

    def returns(self):
        return self.env.info["values"][-1] / self.env.info["values"][-2]

    def log_returns(self):
        return np.log(self.env.info["values"][-1] / self.env.info["values"][-2])

    def sharpe_ratio(self):
        values = np.array(self.env.info["values"])
        excess_returns = values[1:] / values[:-1]
        mean_returns = np.mean(excess_returns)
        std_returns = np.std(excess_returns, ddof=1)
        return (mean_returns - RISK_FREE_RATE) / std_returns
