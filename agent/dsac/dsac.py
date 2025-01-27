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
        self.q1_std_bound = 0.0
        self.q2_std_bound = 0.0
        self.q1_grad_scalar = 0.0
        self.q2_grad_scalar = 0.0

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
            act_dist = TanhGaussDistribution(act, std)
            act = act_dist.mode() if is_deterministic else act_dist.sample()[0]
        return act.detach()
    

    def _critic_objective(self, s, a, r, s_next):
        # Calculate Q-values for current state
        q1, q1_std = self.critic1(s, a)
        q2, q2_std = self.critic2(s, a)

        # Calculate clipping bounds and gradient scalars
        self.q1_std_bound = TAU_B * CLIPPING_RANGE * torch.mean(q1_std.detach()) + (1 - TAU_B) * self.q1_std_bound
        self.q2_std_bound = TAU_B * CLIPPING_RANGE * torch.mean(q2_std.detach()) + (1 - TAU_B) * self.q2_std_bound
        
        self.q1_grad_scalar = TAU_B * torch.mean(torch.pow(q1_std.detach(), 2)) + (1 - TAU_B) * self.q1_grad_scalar
        self.q2_grad_scalar = TAU_B * torch.mean(torch.pow(q2_std.detach(), 2)) + (1 - TAU_B) * self.q2_grad_scalar

        # Get action for next state from target policy
        logits_next_mean, logits_next_std = self.actor_target(s_next)
        act_dist = TanhGaussDistribution(logits_next_mean, logits_next_std)
        a_next, log_prob_a_next = act_dist.rsample()

        # Determine minimum Q-value function
        q1_next, q1_next_std = self.critic1_target(s_next, a_next)
        q2_next, q2_next_std = self.critic2_target(s_next, a_next)
        q_next = torch.min(q1_next, q2_next)
        q_next_std = torch.where(q1_next > q2_next, q1_next_std, q2_next_std)

        # Entropy temperature
        alpha = torch.exp(self.log_alpha)

        # Target Q-value
        q_targ = (r + GAMMA * (q_next - alpha * log_prob_a_next)).detach()

        # Target returns
        z = Normal(q_next, torch.pow(q_next_std, 2)).sample()
        ret_targ = (r + GAMMA * (z - alpha * log_prob_a_next)).detach()

        # Critic 1 Loss
        std_detach = torch.clamp(q1_std, min=0.0).detach()
        grad_mean = -((q_targ - q1).detach() / (torch.pow(std_detach, 2) + STD_BIAS)) * q1
        
        ret_targ_bound = torch.clamp(ret_targ, q1-self.q1_std_bound, q1+self.q1_std_bound).detach()
        grad_std = -((torch.pow(q1.detach() - ret_targ_bound, 2) - torch.pow(std_detach, 2)) 
                    /(torch.pow(std_detach, 3) + STD_BIAS)) * q1_std
        
        q1_loss = (self.q1_grad_scalar + GRAD_BIAS) * torch.mean(grad_mean + grad_std)

        # Critic 2 Loss
        std_detach = torch.clamp(q2_std, min=0.0).detach()
        grad_mean = -((q_targ - q2).detach() / (torch.pow(std_detach, 2) + STD_BIAS)) * q2
        
        ret_targ_bound = torch.clamp(ret_targ, q2-self.q2_std_bound, q2+self.q2_std_bound).detach()
        grad_std = -((torch.pow(q2.detach() - ret_targ_bound, 2) - torch.pow(std_detach, 2)) 
                    /(torch.pow(std_detach, 3) + STD_BIAS)) * q2_std
        
        q2_loss = (self.q2_grad_scalar + GRAD_BIAS) * torch.mean(grad_mean + grad_std)

        return (q1_loss+q2_loss, q1_loss.detach(), q2_loss.detach(), 
                torch.mean(q1.detach()), torch.mean(q2.detach()), 
                torch.mean(q1_std.detach()), torch.mean(q2_std.detach()))
    

    def _actor_objective(self, s, act_new, log_prob_new):
        q1, std1 = self.critic1(s, act_new)
        q2, std2 = self.critic2(s, act_new)
        q_min = torch.min(q1, q2)
        std_min = torch.where(q1 < q2, std1, std2)
        policy_loss = torch.mean(torch.exp(self.log_alpha) * log_prob_new - (q_min / std_min))
        entropy = -torch.mean(log_prob_new.detach())
        return policy_loss, entropy


    def update(self, epoch, step, s, a, r, s_next):
        # Get action for current state
        logits_mean, logits_std = self.actor(s)
        policy_mean = torch.mean(torch.tanh(logits_mean), dim=0, keepdim=True)
        policy_std = torch.mean(logits_std, dim=0, keepdim=True)
        act_dist = TanhGaussDistribution(policy_mean, policy_std)
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
        self.info["q1_mean_std"].append(torch.sqrt(self.q1_grad_scalar).item())
        self.info["q2_mean_std"].append(torch.sqrt(self.q2_grad_scalar).item())
        self.info["q1_loss"].append(q1_loss.item())
        self.info["q2_loss"].append(q2_loss.item())
        self.info["critic_loss"].append(value_loss.item())
        self.info["actor_loss"].append(policy_loss.item())
        self.info["entropy"].append(entropy.item())
        self.info["alpha"].append(torch.exp(self.log_alpha).item())

    def log_info(self, log_file):
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
