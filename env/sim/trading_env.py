from config.base import INITIAL_CASH, NUM_INVESTED, BUY_PENALTY, SELL_PENALTY

import torch
from env.sim.weight_buffer import ActionBuffer

class TradingEnv:
    def __init__(self):
        self.value = INITIAL_CASH         # Initialize the portfolio value
        self.weights = ActionBuffer()    # Buffer for storing the weights

        self.info = {
            "actions": [],
            "values": [],
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
        self.value = INITIAL_CASH     # Reset the portfolio value
        self.weights.reset()            # Reset the weights buffer

        self.info = {
            "actions": [],
            "values": [],
            "rewards": [],
            "returns": []
        }

        action = self.weights.get_all() # Get the action from the buffer
        features[:, :, -1] = action     # Replace the last column with the weights

        return features


    # Derived from: github.com/ZhengyaoJiang/PGPortfolio/blob/master/pgportfolio/tools/trade.py
    def _calc_remainder(self, w, w_targ):
        """ Calculate the transaction remainder factor -- the percentage of the portfolio that remains after transaction fees.
        Args:
            w (torch.Tensor): Current portfolio weights
            w_targ (torch.Tensor): Desired new portfolio weights
        Returns:
            float: Transaction remainder factor
        """
        mu0 = torch.tensor(1.0, device=w.device)
        mu1 = torch.tensor(1.0 - SELL_PENALTY - BUY_PENALTY + SELL_PENALTY * BUY_PENALTY, device=w.device)

        while torch.abs(mu1 - mu0) > 1e-10:
            mu0 = mu1
            denom = 1 - BUY_PENALTY * w_targ[0]
            coeff = 1 - (BUY_PENALTY * w[0]) - (SELL_PENALTY + BUY_PENALTY - (SELL_PENALTY * BUY_PENALTY))
            mu1 = (coeff * torch.sum(torch.maximum(w[1:] - mu1 * w_targ[1:], torch.tensor(0.0, device=w.device)))) / denom

        return mu1.item()


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
        action = action.flatten()
        rel_prices = rel_prices.flatten()
        action = action + torch.min(action)

        # Keep top k weights and set the rest to zero
        min_topk = torch.topk(action, NUM_INVESTED).values[-1]
        action = torch.where(action >= min_topk, action, torch.tensor(0.0, device=action.device))

        # # Set all weights below 0.5 to zero
        # action = torch.where(action >= 0.5, action, torch.tensor(0.0, device=action.device))

        # action = torch.clamp(action, 0.0)   # Clip the weights to be non-negative

        # Normalize the weights
        action = action / torch.sum(action)
        
        # Get the portfolio weights before the action
        w = self.weights.get_last()

        # # Calculate the transaction remainder factor
        # mu = self._calc_remainder(w, action)

        # Calculate the new portfolio value
        value = self.value * torch.dot(w, rel_prices)

        # Calculate the reward for the current day
        reward = torch.log(value / self.value) * 100
        returns = value / self.value

        # Update the portfolio value
        self.value = value

        # Update the weights buffer with the new weights
        self.weights.update(action)

        # Replace the last column of the features with the weights
        features[:, :, -1] = self.weights.get_all()

        # Store info about current environment step
        self.info["actions"].append(action.cpu().numpy())
        self.info["values"].append(self.value.item())
        self.info["rewards"].append(reward.item())
        self.info["returns"].append(returns.item())

        return reward, features
    
        # pv_after_commission = calculate_pv_after_commission(omega, self._last_omega, self._commission_rate)
        # delta_value = pv_after_commission * np.dot(omega, future_price)
        # value = self.value * portfolio_change
        # weights = pv_after_commission * omega * future_price / portfolio_change

    
    def log_info(self, path):
        """ Save the environment info to a file.
        Args:
            path (str): Path to the file
        """
        with open(path, "a") as f:
            #{str(self.info["action"][-1]):<20}
            f.write(f"{str(self.info["value"][-1]):<20}{str(self.info["reward"][-1]):<20}\n")
