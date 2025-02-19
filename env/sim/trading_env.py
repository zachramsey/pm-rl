from config.base import INITIAL_CASH, COMISSION, REWARD_SCALE, REWARD, RISK_FREE_RATE

import torch
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
            "rewards": [0],
            "returns": [0]
        }
        

    def reset(self, features):
        """ Reset the environment to the initial state.
        Args:
            features (torch.Tensor): Features for the first day
        Returns:
            features (torch.Tensor): Updated features for the first day
        """
        self.value = INITIAL_CASH       # Reset the portfolio value
        self.weights.reset()            # Reset the weights buffer

        action = self.weights.get_all() # Get the action from the buffer
        features[:, :, -1] = action     # Replace the last column with the weights

        self.info = {
            "values": [INITIAL_CASH],
            "actions": [self.weights.get_last().flatten()],
            "rewards": [0],
            "returns": [0]
        }

        return features


    def step(self, action, features, prices):
        """ Execute a trading action and return new state, reward, and other info.
        Args:
            action (torch.Tensor): Desired portfolio weights
            features (torch.Tensor): Features for the current day
            prices (torch.Tensor): Price relative vector for the current day
        Returns:
            features (torch.Tensor): Updated features for the next day
            reward (float): Reward for the current day
        """
        w = action.flatten()
        prices = prices.flatten()

        # Normalize the weights
        if not torch.isclose(torch.sum(w), torch.tensor(1.0), atol=1e-6) and torch.min(action) < 0:
            numer = torch.exp(w)
            w = numer / torch.sum(numer)
        
        # Get the portfolio weights before the action
        w_last = self.weights.get_last()

        # Calculate the transaction remainder factor
        # Derived from: github.com/ZhengyaoJiang/PGPortfolio/blob/master/pgportfolio/tools/trade.py
        if COMISSION > 0:
            mu_last = 1
            mu = 1 - 2*COMISSION + COMISSION**2
            while abs(mu - mu_last) > 1e-10:
                mu_last = mu
                numer = 1 - (COMISSION * w_last[0]) - (2*COMISSION - COMISSION**2) * torch.sum(torch.maximum(w_last[1:] - mu * w[1:], ))
                denom = 1 - COMISSION * w[0]
                mu = numer / denom
            self.value = mu * self.value

        # Calculate the new portfolio value
        portfolio = self.value * (w * prices)
        value = torch.sum(portfolio)
        self.info["values"].append(value)

        # Calculate final portfolio weights
        w = portfolio / value
        self.weights.update(w)
        self.info["actions"].append(w.detach().flatten().cpu().numpy())

        # Calculate the returns for the current day
        ret = value / self.value
        self.value = value.detach()
        self.info["returns"].append(ret.detach().cpu().numpy())

        # Calculate the reward for the current day
        # if REWARD == "returns":
        #     r = ret * REWARD_SCALE
        # elif REWARD == "log_returns":
        #     r = torch.log(ret) * REWARD_SCALE
        # elif REWARD == "sharpe_ratio":
        #     r = ret / torch.std(ret) * REWARD_SCALE
        r = torch.log(ret) * REWARD_SCALE
        self.info["rewards"].append(r.detach().cpu().numpy())

        # Replace the last column of the features with the weights
        features[:, :, -1] = self.weights.get_all()
        
        return r, features
    
    
    def log_info(self, path):
        """ Save the environment info to a file.
        Args:
            path (str): Path to the file
        """
        with open(path, "a") as f:
            #{str(self.info["action"][-1]):<20}
            f.write(f"{str(self.info["value"][-1]):<20}{str(self.info["reward"][-1]):<20}\n")
