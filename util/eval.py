
import numpy as np

class Metrics:
    def __init__(self, env, cfg):
        self.log_file = cfg["log_dir"] + "latest.log"
        self.env = env

    def sharpe(self):
        ''' ### Calculate Annualized Sharpe ratio
        Args:
            returns (np.ndarray): Returns of the portfolio
        Returns:
            sharpe (float): Sharpe ratio
        '''
        returns = np.array(self.env.info["returns"])
        excess_returns = returns - 0.04
        mu = np.mean(excess_returns)
        std = np.std(excess_returns, ddof=1)
        return (mu / std) * np.sqrt(252)

    def mdd(self):
        ''' ### Calculate Maximum Drawdown
        Args:
            returns (np.ndarray): Returns of the portfolio
        Returns:
            mdd (float): Maximum Draw
        '''
        values = np.array(self.env.info["values"])
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return np.max(drawdown) * 100

    def average_turnover(self):
        ''' ### Calculate Average Turnover
        Args:
            weights (np.ndarray): Weights of the portfolio
        Returns:
            avg_turnover (float): Average Turnover
        '''
        weights = np.array(self.env.info["actions"])
        turnover = 0
        for i in range(1, len(weights)):
            turnover += np.sum(np.abs(weights[i] - weights[i-1]))
        return turnover / (len(weights) - 1)
    
    def write(self):
        ''' ### Write metrics to the log file'''
        with open(self.log_file, "a") as f:
            f.write(f"Evaluation Step:\n")
            f.write(f"     Sharpe Ratio: {self.sharpe()}\n")
            f.write(f"Maximum Draw-Down: {self.mdd()}\n")
            f.write(f" Average Turnover: {self.average_turnover()}\n")
            f.write(f"      Final Value: {self.env.info["values"][-1]:.2f}\n\n")
