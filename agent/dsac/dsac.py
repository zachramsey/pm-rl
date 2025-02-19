'''
Derived from: github.com/Jingliang-Duan/DSAC-v2/blob/main/dsac_v2.py
'''
import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal

from config.base import BATCH_SIZE, NUM_ASSETS, LOG_DIR
from config.dsac import GAMMA, TAU, TAU_B, DELAY_UPDATE, CLIPPING_RANGE, GRAD_BIAS, STD_BIAS, ACTOR_LR, CRITIC_LR, ALPHA_LR

from distr.tanh_gauss import TanhGaussDistribution
from distr.gauss import GaussDistribution
# from agent.dsac.value import Critic
# from agent.dsac.policy import Actor
from agent.dsac.value import LSRE_CANN_Critic as Critic
from agent.dsac.policy import LSRE_CANN_Actor as Actor

class DSAC:
    def __init__(self, feat_dim):
        # Networks (Q1, Q2, Pi)
        self.critic1 = Critic(feat_dim)
        self.critic2 = Critic(feat_dim)
        self.actor = Actor(feat_dim)

        # Target Networks
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        self.actor_target = deepcopy(self.actor)

        # Perform soft update (Polyak update) on target networks
        for p in self.critic1_target.parameters(): p.requires_grad = False
        for p in self.critic2_target.parameters(): p.requires_grad = False
        for p in self.actor_target.parameters(): p.requires_grad = False

        # Parameterized policy entropy temperature (Alpha)
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.entropy_target = -NUM_ASSETS

        # Optimizers
        self.critic1_opt = Adam(self.critic1.parameters(), lr=CRITIC_LR)
        self.critic2_opt = Adam(self.critic2.parameters(), lr=CRITIC_LR)
        self.actor_opt = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.alpha_opt = Adam([self.log_alpha], lr=ALPHA_LR)

        # Moving Averages
        self.ema_std1 = -1.0
        self.ema_std2 = -1.0

        # Collect training information
        self.info = {
            "epoch": [],
            "q1": [],
            "q2": [],
            "q1_std": [],
            "q2_std": [],
            "q1_mean_std": [],
            "q2_mean_std": [],
            "q1_loss": [],
            "q2_loss": [],
            "critic_loss": [],
            "actor_loss": [],
            "entropy": [],
            "alpha": [],
        }


    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()


    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()


    def act(self, s, is_random=False, is_deterministic=False):
        if is_random:
            act = torch.randn(NUM_ASSETS, 1)
        else:
            act, std = self.actor(s)
            act_dist = GaussDistribution(act, std)
            act = act_dist.mode() if is_deterministic else act_dist.sample()[0]
        return act.detach()
    

    def _critic_objective(self, s, a, r, s_next):
        # Calculate Q-values for current state
        q1, q1_std = self.critic1(s, a)
        q2, q2_std = self.critic2(s, a)

        # Calculate clipping bounds and gradient scalars
        if self.ema_std1 == -1.0: self.ema_std1 = torch.mean(q1_std.detach())
        else: self.ema_std1 = (1 - TAU_B) * self.ema_std1 + TAU_B * torch.mean(q1_std.detach())

        if self.ema_std2 == -1.0: self.ema_std2 = torch.mean(q2_std.detach())
        else: self.ema_std2 = (1 - TAU_B) * self.ema_std2 + TAU_B * torch.mean(q2_std.detach())

        # Get action for next state from target policy
        logits_next_mean, logits_next_std = self.actor_target(s_next)
        act_dist = GaussDistribution(logits_next_mean, logits_next_std)
        a_next, log_prob_a_next = act_dist.rsample()

        # Determine minimum Q-value function
        q1_next, q1_next_std = self.critic1_target(s_next, a_next)
        z = Normal(torch.zeros_like(q1_next), torch.ones_like(q1_next_std)).sample()
        z = torch.clamp(z, -CLIPPING_RANGE, CLIPPING_RANGE)
        q1_next_sample = q1_next + torch.mul(z, q1_next_std)

        q2_next, q2_next_std = self.critic2_target(s_next, a_next)
        z = Normal(torch.zeros_like(q2_next), torch.ones_like(q2_next_std)).sample()
        z = torch.clamp(z, -CLIPPING_RANGE, CLIPPING_RANGE)
        q2_next_sample = q2_next + torch.mul(z, q2_next_std)

        q_next = torch.min(q1_next_sample, q2_next_sample)
        q_next_sample = torch.where(q1_next < q2_next, q1_next_sample, q2_next_sample)

        # Entropy temperature
        alpha = torch.exp(self.log_alpha).item()

        # Target Q-values
        q1_targ = (r + GAMMA * (q_next.detach() - alpha * log_prob_a_next))
        q1_targ_sample = (r + GAMMA * (q_next_sample.detach() - alpha * log_prob_a_next.detach()))
        td_bound1 = CLIPPING_RANGE * self.ema_std1.detach()
        q1_targ_bound = q1.detach() + torch.clamp(q1_targ_sample - q1.detach(), -td_bound1, td_bound1)

        q2_targ = (r + GAMMA * (q_next.detach() - alpha * log_prob_a_next))
        q2_targ_sample = (r + GAMMA * (q_next_sample.detach() - alpha * log_prob_a_next.detach()))
        td_bound2 = CLIPPING_RANGE * self.ema_std2.detach()
        q2_targ_bound = q2.detach() + torch.clamp(q2_targ_sample - q2.detach(), -td_bound2, td_bound2)

        # Lower bounded standard deviations
        q1_std_detach = torch.clamp(q1_std, min=0.0).detach()
        q2_std_detach = torch.clamp(q2_std, min=0.0).detach()

        # Critic 1 Loss
        grad_mean = -((q1_targ.detach() - q1).detach() / (torch.pow(q1_std_detach, 2) + STD_BIAS)) * q1
        grad_std = -((torch.pow(q1.detach() - q1_targ_bound.detach(), 2) - torch.pow(q1_std_detach, 2)) 
                    /(torch.pow(q1_std_detach, 3) + STD_BIAS)) * q1_std
        q1_loss = (torch.pow(self.ema_std1, 2) + GRAD_BIAS) * torch.mean(grad_mean + grad_std)

        # Critic 2 Loss
        grad_mean = -((q2_targ.detach() - q2).detach() / (torch.pow(q2_std_detach, 2) + STD_BIAS)) * q2
        grad_std = -((torch.pow(q2.detach() - q2_targ_bound.detach(), 2) - torch.pow(q2_std_detach, 2)) 
                    /(torch.pow(q2_std_detach, 3) + STD_BIAS)) * q2_std
        q2_loss = (torch.pow(self.ema_std2, 2) + GRAD_BIAS) * torch.mean(grad_mean + grad_std)

        return (q1_loss+q2_loss, q1_loss.detach(), q2_loss.detach(), 
                torch.mean(q1.detach()), torch.mean(q2.detach()), 
                torch.mean(q1_std.detach()), torch.mean(q2_std.detach()))
    

    def _actor_objective(self, s, act_new, log_prob_new):
        q1, std1 = self.critic1(s, act_new)
        q2, std2 = self.critic2(s, act_new)
        q_min = torch.min(q1, q2)
        std_min = torch.where(q1 < q2, std1, std2)
        policy_loss = torch.mean(torch.exp(self.log_alpha).item() * log_prob_new - (q_min / std_min))
        # policy_loss = torch.mean(torch.exp(self.log_alpha).item() * log_prob_new - torch.min(q1, q2))
        entropy = -torch.mean(log_prob_new.detach())
        return policy_loss, entropy


    def update(self, epoch, step, s, a, r, s_next):
        # Get action for current state
        logits_mean, logits_std = self.actor(s)
        # policy_mean = torch.mean(torch.tanh(logits_mean)).item()
        # policy_std = torch.mean(logits_std).item()

        act_dist = GaussDistribution(logits_mean, logits_std)
        act_new, log_prob_new = act_dist.rsample()
        act_new = act_new.expand(BATCH_SIZE, -1, -1)

        # Update Critic
        self.critic1_opt.zero_grad()#set_to_none=True)
        self.critic2_opt.zero_grad()
        value_loss, q1_loss, q2_loss, q1, q2, q1_std, q2_std = self._critic_objective(s, a, r, s_next)
        value_loss.backward()

        # Freeze critic networks
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        # Update Actor
        self.actor_opt.zero_grad()
        policy_loss, entropy = self._actor_objective(s, act_new, log_prob_new)
        policy_loss.backward()

        # Unfreeze networks
        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # Update temperature
        self.alpha_opt.zero_grad()
        temperature_loss = -self.log_alpha * torch.mean(log_prob_new.detach() + self.entropy_target)
        temperature_loss.backward()

        # Optimize critics
        self.critic1_opt.step()
        self.critic2_opt.step()

        # Delayed update
        if step % DELAY_UPDATE == 0:
            self.actor_opt.step()       # Optimize actor network
            self.alpha_opt.step()       # Optimize temperature network

            # Perform soft update on target networks
            with torch.no_grad():
                polyak = 1 - TAU

                for p, p_targ in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters(),):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

        # Collect training information
        self.info["epoch"].append(epoch)
        self.info["q1"].append(q1.item())
        self.info["q2"].append(q2.item())
        self.info["q1_std"].append(q1_std.item())
        self.info["q2_std"].append(q2_std.item())
        self.info["q1_mean_std"].append(self.ema_std1)
        self.info["q2_mean_std"].append(self.ema_std2)
        self.info["q1_loss"].append(q1_loss.item())
        self.info["q2_loss"].append(q2_loss.item())
        self.info["critic_loss"].append(value_loss.item())
        self.info["actor_loss"].append(policy_loss.item())
        self.info["entropy"].append(entropy.item())
        self.info["alpha"].append(torch.exp(self.log_alpha).item())

    def log_info(self):
        # Check that logs directory exists
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        # Write training information to log file
        with open(LOG_DIR + "latest.log", "a") as f:
            f.write("="*50 + "\n")
            for key, value in self.info.items():
                if key == "epoch":
                    f.write(f"Epoch {value[-1]}\n")
                    f.write("-"*50 + "\n")
                    f.write("Training Step:\n")
                else:
                    f.write(f"{key}: {value[-1]}\n")
            f.write("-"*50 + "\n")
