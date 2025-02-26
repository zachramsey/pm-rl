
import sys
import random
import numpy as np
import torch

from config.base import ALGORITHM, SEED, NUM_EPOCHS, BATCH_SIZE
from config.base import ROLLOUT_PRINT_FREQ, TRAIN_PRINT_FREQ, EVAL_PRINT_FREQ
from config.base import PLOT_FREQ, EVAL_FREQ

from data.data_loader import StockDataLoader
from replay.rollout_buffer import RolloutBuffer

from env.sim.trading_env import TradingEnv

if ALGORITHM == "PG":
    from agent.pg.pg import PG as Agent

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
        self.train_dl = self.data.train_dl
        self.eval_dl = self.data.test_dl
        
        self.buffer = RolloutBuffer(self.data.num_features, self.data.train_len, self.data.train_prices)
        self.env = TradingEnv()
        self.agent = Agent(self.data.num_features)

        self.metrics = Metrics(self.env, self.data.test_dates)
        self.visualizer = Visualizer(self.agent, self.env, self.data.train_dates, self.data.test_dates)

        self.s = None

    def train(self):
        self._evaluate(0)
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch}:")
            self._rollout(epoch)
            self._update(epoch)
            self._evaluate(epoch)

            if epoch % PLOT_FREQ == 0:
                self.visualizer.plot()

    def _rollout(self, epoch):
        self.agent.training_mode(False)
        self.buffer.reset()
        rew_tot = 0
        for step, (datetime, prices, data) in enumerate(self.train_dl):
            if step == 0:
                self.s = self.env.reset(data)
            else:
                a = self.agent.act(self.s)
                r, s_ = self.env.step(a, data, prices)
                self.buffer.add(self.s, a, self.env.value, r)
                rew_tot += r
                self.s = s_
            print_inline_every(step, ROLLOUT_PRINT_FREQ, len(self.train_dl),
                f"  Rollout | Step: {step}  | Date: {datetime.strftime('%d/%m/%y')} | Reward: {rew_tot:.6f} | Value: {self.env.value:.2f}")

    def _update(self, epoch):
        self.agent.training_mode(True)
        for i, (s, a, r, _v, _a, p) in enumerate(self.buffer.sample_random()):
            self.agent.update(i, s, a, r, _v, _a, p)
            print_inline_every(i, TRAIN_PRINT_FREQ, len(self.train_dl) // BATCH_SIZE, 
                f"  Update | Step: {i} | Loss: {self.agent.info["loss"][-1]:.12f}")
        sys.stdout.write("\033[F\033[K")
        print(f"  Update | Step: {i} | Loss: {np.average(self.agent.info["loss"]):.12f}")
        self.agent.log_info()

    def _evaluate(self, epoch):
        if epoch % EVAL_FREQ == 0:
            self.agent.training_mode(False)
            with torch.no_grad():
                rew_tot = 0
                for step, (datetime, prices, data) in enumerate(self.eval_dl):
                    if step == 0:
                        self.s = self.env.reset(data)
                    else:
                        a = self.agent.act(self.s)
                        r, s_ = self.env.step(a, data, prices)
                        rew_tot += r
                        self.s = s_
                    print_inline_every(step, EVAL_PRINT_FREQ, len(self.eval_dl),
                        f"Evaluate | Step: {step} | Date: {datetime.strftime('%d/%m/%y')}  | Reward: {rew_tot:.6f} | Value: {self.env.value:.2f}")
                self.metrics.write()
