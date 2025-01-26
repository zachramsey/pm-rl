import torch

from preprocess.data_loader import StockDataLoader
from env.trading_env import TradingEnv
from srl.srl import SRL
from agent.dsac import DSAC

from utils.buffer import ReplayBuffer
from utils.metrics import Metrics
from utils.visualize import Visualizer
from utils.utils import print_inline_every

class TrainOffPolicy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.s = None

        data = StockDataLoader(cfg)
        self.train_dl = data.get_train_data()
        self.eval_dl = data.get_eval_data()
        self.train_dates = self.train_dl.dataset.dates
        self.eval_dates = self.eval_dl.dataset.dates
        
        self.buffer = ReplayBuffer(self.train_dl, self.cfg)
        self.env = TradingEnv(self.cfg)
        self.agent = DSAC(self.cfg)

        self.metrics = Metrics(self.env, self.cfg)
        self.visualizer = Visualizer(self.agent, self.env, self.train_dates, self.eval_dates, self.cfg)

    def train(self):
        for epoch in range(self.cfg["epochs"]):
            print(f"\nEpoch {epoch}:")
            self._interact(epoch)
            self._update(epoch)
            self._evaluate(epoch)
            if epoch % self.cfg["plot_freq"] == 0 and epoch > 0:
                self.visualizer.plot()

    def _interact(self, epoch):
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
                print_inline_every(step, self.cfg["interact_print_freq"], self.cfg["train_len"], 
                    f"Interact | Step: {step} | Date: {self.train_dates[step].date()} | Value: {self.env.value:.2f}")

    def _update(self, epoch):
        self.agent.train()
        for step in range(self.cfg["update_steps"]):
            s, a, r, s_ = self.buffer.sample()
            self.agent.update(epoch, step, s, a, r, s_)
            print_inline_every(step, self.cfg["update_print_freq"], self.cfg["update_steps"], 
                f"  Update | Step: {step} | Actor Loss: {self.agent.info["actor_loss"][-1]:.6f} | Critic Loss: {self.agent.info["critic_loss"][-1]:.6f}")
        self.agent.log_info(self.cfg["log_dir"] + "latest.log")

    def _evaluate(self, epoch):
        if epoch % self.cfg["eval_freq"] == 0:
            with torch.no_grad():
                self.agent.eval()
                for step, (feat, targ) in enumerate(self.eval_dl):
                    feat = feat.squeeze(0)
                    if step == 0:
                        self.s = self.env.reset(feat)
                    else:
                        a = self.agent.act(self.s, is_deterministic=True)
                        _, s_ = self.env.step(a, feat, targ)
                        self.s = s_
                    print_inline_every(step, self.cfg["eval_print_freq"], self.cfg["eval_len"], 
                        f"Evaluate | Step: {step} | Date: {self.eval_dates[step].date()} | Value: {self.env.value:.2f}")
                self.metrics.write()
