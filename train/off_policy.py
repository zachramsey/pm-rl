
import sys
import random
import numpy as np
import torch

from config.base import SEED, MODEL, NUM_EPOCHS, SEED_EPOCHS, BATCH_SIZE, UPDATE_STEPS, PLOT_FREQ, EVAL_FREQ, SEED_PRINT_FREQ, COLLECT_PRINT_FREQ, TRAIN_PRINT_FREQ, EVAL_PRINT_FREQ, LOG_DIR

from loader.data_loader import StockDataLoader
from replay.traj_buffer import ReplayBuffer
from env.sim.trading_env import TradingEnv

if MODEL == "DSAC":
    from agent.dsac.dsac import DSAC as Agent
elif MODEL == "TD3":
    from agent.td3.td3 import TD3 as Agent
elif MODEL == "DreamerV3":
    from agent.dreamer.dreamer import DreamerV3 as Agent

from util.eval import Metrics
from util.plot import Visualizer
from util.util import print_inline_every

class Train:
    def __init__(self):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        self.data = StockDataLoader()

        self.train_dl = self.data.get_train_data()
        self.train_dates = self.data.get_train_dates()
        self.train_len = self.data.get_train_len()

        self.eval_dl = self.data.get_eval_data()
        self.eval_dates = self.data.get_eval_dates()
        self.eval_len = self.data.get_eval_len()
        
        self.buffer = ReplayBuffer(self.data)
        self.env = TradingEnv()
        self.agent = Agent(self.data.get_num_features())

        self.metrics = Metrics(self.env, self.eval_dates)
        self.visualizer = Visualizer(self.agent, self.env, self.train_dates, self.eval_dates)

        self.s = None

    def train(self):
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch}:")
            if epoch < SEED_EPOCHS or len(self.buffer) < BATCH_SIZE:
                self._collect_rand(epoch)
            else:
                self._collect(epoch)
                self._update_off_policy(epoch)
                self._evaluate_off_policy(epoch)

                if epoch % PLOT_FREQ == 0:
                    self.visualizer.plot()

    def _collect_rand(self, epoch):
        for step, (feat, targ) in enumerate(self.train_dl):
            feat = feat.squeeze(0)
            if step == 0:
                self.s = self.env.reset(feat)
            else:
                a = self.agent.act(self.s, is_random=True)
                r, s_ = self.env.step(a, feat, targ)
                self.buffer.add(epoch, step, a, r)
                self.s = s_
            print_inline_every(step, SEED_PRINT_FREQ, self.train_len, 
                f" Seeding | Step: {step} | Date: {self.train_dates[step]} | Value: {self.env.value:.2f}")

    def _collect(self, epoch):
        with torch.no_grad():
            self.agent.eval()
            for step, (feat, targ) in enumerate(self.train_dl):
                feat = feat.squeeze(0)
                if step == 0:
                    self.s = self.env.reset(feat)
                else:
                    a = self.agent.act(self.s)
                    r, s_ = self.env.step(a, feat, targ)
                    self.buffer.add(epoch, step, a, r)
                    self.s = s_
                print_inline_every(step, COLLECT_PRINT_FREQ, self.train_len, 
                    f"Interact | Step: {step} | Date: {self.train_dates[step]} | Value: {self.env.value:.2f}")

    def _update_off_policy(self, epoch):
        self.agent.train()
        for step in range(UPDATE_STEPS):
            s, a, r, s_ = self.buffer.sample()
            self.agent.update(epoch, step, s, a, r, s_)
            print_inline_every(step, TRAIN_PRINT_FREQ, UPDATE_STEPS, 
                f"  Update | Step: {step} | Actor Loss: {self.agent.info["actor_loss"][-1]:.6f} | Critic Loss: {self.agent.info["critic_loss"][-1]:.6f}")
        self.agent.log_info(LOG_DIR + "latest.log")

    def _evaluate_off_policy(self, epoch):
        if epoch % EVAL_FREQ == 0:
            self.agent.eval()
            with torch.no_grad():
                for step, (feat, targ) in enumerate(self.eval_dl):
                    feat = feat.squeeze(0)
                    if step == 0:
                        self.s = self.env.reset(feat)
                    else:
                        a = self.agent.act(self.s, is_deterministic=True)
                        _, s_ = self.env.step(a, feat, targ)
                        self.s = s_
                    print_inline_every(step, EVAL_PRINT_FREQ, self.eval_len, 
                        f"Evaluate | Step: {step} | Date: {self.eval_dates[step]} | Value: {self.env.value:.2f}")
                self.metrics.write()
