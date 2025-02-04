
import os
from config.base import LOG_DIR, REPORT_DIR, RISK_FREE_RATE
import quantstats as qs
import numpy as np
import pandas as pd

class Metrics:
    def __init__(self, env, dates):
        self.log_file = LOG_DIR + "latest.log"
        self.env = env
        self.dates = dates

    def sharpe(self):
        returns = self.env.info["returns"]
        returns = pd.DataFrame(returns, index=self.dates[-len(returns):])
        sharpe = qs.stats.sharpe(returns, rf=RISK_FREE_RATE)
        return sharpe.to_numpy()[0]
    
    def sortino(self):
        returns = self.env.info["returns"]
        returns = pd.DataFrame(returns, index=self.dates[-len(returns):])
        sortino = qs.stats.sortino(returns, rf=RISK_FREE_RATE)
        return sortino.to_numpy()[0]

    def mdd(self):
        values = self.env.info["values"]
        values = pd.DataFrame(values, index=self.dates[-len(values):])
        mdd = qs.stats.max_drawdown(values)
        return mdd.to_numpy()[0]

    def average_turnover(self):
        weights = np.array(self.env.info["actions"])
        turnover = 0
        for i in range(1, len(weights)):
            turnover += np.sum(np.abs(weights[i] - weights[i-1]))
        return turnover / (len(weights) - 1)
    
    def write(self):
        ''' ### Write metrics to the log file'''
        # report_loc = REPORT_DIR + "latest.html"
        # if not os.path.exists(REPORT_DIR):
        #     os.makedirs(REPORT_DIR)
        # qs.reports.html(returns=returns, benchmark="SPY", rf=RISK_FREE_RATE, download_filename=report_loc)
        with open(self.log_file, "a") as f:
            f.write(f"Evaluation Step:\n")
            f.write(f"     Sharpe Ratio: {self.sharpe()}\n")
            f.write(f"Maximum Draw-Down: {self.mdd()}\n")
            f.write(f" Average Turnover: {self.average_turnover()}\n")
            f.write(f"      Final Value: {self.env.info["values"][-1]:.2f}\n\n")
