from config.base import INITIAL_CASH, COMISSION

import torch
import numpy as np
from env.sim.weight_buffer import ActionBuffer
from env.reward import Reward

class TradingEnv:
    def __init__(self):
        self.value = INITIAL_CASH         # Initialize the portfolio value
        self.weights = ActionBuffer()    # Buffer for storing the weights
        self.reward = Reward(self)       # Reward function

        self.info = {
            "values": [INITIAL_CASH],
            "actions": [self.weights.get_last().flatten()],
            "rewards": [],
            "returns": []
        }
        

    def reset(self, features):
        """ Reset the environment to the initial state.
        Args:
            features (torch.Tensor): Features for the first day
        Returns:
            features (torch.Tensor): Updated features for the first day
        """
        self.value = INITIAL_CASH   # Reset the portfolio value
        self.weights.reset()        # Reset the weights buffer

        action = self.weights.get_all()                 # Get the action from the buffer
        features[:, :, -1] = torch.from_numpy(action)   # Replace the last column with the weights

        self.info = {
            "values": [INITIAL_CASH],
            "actions": [self.weights.get_last().flatten()],
            "rewards": [],
            "returns": []
        }

        return features


    def step(self, action, features, rel_prices):
        """ Execute a trading action and return new state, reward, and other info.
        Args:
            action (torch.Tensor): Desired portfolio weights
            features (torch.Tensor): Features for the current day
            rel_prices (torch.Tensor): Price relative vector for the current day
        Returns:
            features (torch.Tensor): Updated features for the next day
            reward (float): Reward for the current day
        """
        action = action.flatten().cpu().numpy()
        rel_prices = rel_prices.flatten().cpu().numpy()

        # Normalize the weights
        if np.isclose(np.sum(action), 1, atol=1e-6) and np.min(action) >= 0:
            w = action
        else:
            numer = np.exp(action)
            w = numer / np.sum(numer)
        
        # Get the portfolio weights before the action
        w_last = self.weights.get_last()

        # Calculate the transaction remainder factor
        # Derived from: github.com/ZhengyaoJiang/PGPortfolio/blob/master/pgportfolio/tools/trade.py
        if COMISSION > 0:
            mu_last = 1
            mu = 1 - 2*COMISSION + COMISSION**2
            while abs(mu - mu_last) > 1e-10:
                mu_last = mu
                numer = 1 - (COMISSION * w_last[0]) - (2*COMISSION - COMISSION**2) * np.sum(np.maximum(w_last[1:] - mu * w[1:], ))
                denom = 1 - COMISSION * w[0]
                mu = numer / denom
            self.value = mu * self.value

        # Calculate the new portfolio value
        portfolio = self.value * (w * rel_prices)
        value = np.sum(portfolio)
        self.info["values"].append(value)

        # Calculate final portfolio weights
        w = portfolio / value
        self.weights.update(w)
        self.info["actions"].append(w)

        # Calculate the returns for the current day
        ret = value / self.value
        self.value = value
        self.info["returns"].append(ret)

        # Calculate the reward for the current day
        r = self.reward.get_reward()
        self.info["rewards"].append(r)

        # Replace the last column of the features with the weights
        features[:, :, -1] = torch.from_numpy(self.weights.get_all())
        
        return torch.tensor(r), features
    
    
    def log_info(self, path):
        """ Save the environment info to a file.
        Args:
            path (str): Path to the file
        """
        with open(path, "a") as f:
            #{str(self.info["action"][-1]):<20}
            f.write(f"{str(self.info["value"][-1]):<20}{str(self.info["reward"][-1]):<20}\n")
