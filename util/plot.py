import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime as dt

from config.base import PLOT_DIR, NUM_ASSETS, TRAINER

class Visualizer:
    def __init__(self, agent, env, train_dates, eval_dates):
        self.agent = agent
        self.env = env
        self.train_dates = train_dates
        self.eval_dates = eval_dates

    def plot(self):
        self.plot_update_info()
        # self.animate_portfolio()

    def plot_update_info(self):
        lo = lambda x, y: [a - b for a, b in zip(x, y)]
        hi = lambda x, y: [a + b for a, b in zip(x, y)]

        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)
            
        if TRAINER == "Off-Policy":
            e, q1, q2, q1_std, q2_std, q1_mean_std, q2_mean_std, \
            q1_loss, q2_loss, critic_loss, actor_loss, entropy, alpha = self.agent.info.values()
            step = np.arange(len(q1))
            
            plt.figure(figsize=(8, 10))

            q = np.where(q1 <= q2, q1, q2)
            std = np.where(q1 <= q2, q1_std, q2_std)
            plt.subplot(2, 1, 1)
            plt.plot(step, q)
            plt.fill_between(step, lo(q, std), hi(q, std), alpha=0.2)
            plt.xlabel("Step")
            plt.ylabel("Q")
            plt.title("Minimum Q Value and Standard Deviation")

            plt.subplot(2, 1, 2)
            plt.plot(step, critic_loss, label="Critic")
            plt.plot(step, actor_loss, label="Actor")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Actor & Critic Loss")
            plt.legend()
        elif TRAINER == "On-Policy":
            plt.figure(figsize=(8, 10))
            
            e, loss = self.agent.info.values()
            step = np.arange(len(loss))
            plt.subplot(2, 1, 1)
            plt.plot(step, loss)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Policy Loss")

            vals, acts, rews, rets = self.env.info.values()
            step = np.arange(len(vals))
            plt.subplot(2, 1, 2)
            plt.plot(step, vals, label="Value")
            plt.plot(step, rews, label="Reward")
            plt.plot(step, rets, label="Return")
            plt.xlabel("Step")
            plt.title("Backtesting Metrics")

        plt.savefig(PLOT_DIR + "latest.png")
        plt.close()

    def animate_portfolio(self):
        weights = np.array(self.env.info["actions"])
        values = np.array(self.env.info["values"])

        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)

        # Setup the figure and axes
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, NUM_ASSETS - 0.5)
        ax.set_ylim(0, 1, auto=True, emit=True)
        ax.set_xticks(range(NUM_ASSETS))
        ax.set_xlabel("Stock Symbol")
        ax.set_ylabel("Weight")

        # Initialize bars and text
        bars = ax.bar(range(NUM_ASSETS), weights[0], color="b")
        date_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)
        value_text = ax.text(0.05, 0.8, "", transform=ax.transAxes)

        # Update function for animation
        def update(frame):
            for bar, height in zip(bars, weights[frame]):
                bar.set_height(height)
            date_text.set_text(f"Date: {self.eval_dates[frame].date()}")
            value_text.set_text(f"Value: {values[frame]:.2f}")
            return bars, date_text, value_text

        ani = FuncAnimation(fig, update, frames=len(self.eval_dates), interval=100, blit=False)
        ani.save(os.path.join(PLOT_DIR, f"portfolio_{dt.now().strftime("%y-%m-%d_%H-%M-%S")}.mp4"), fps=30)
