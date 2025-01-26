import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import copy
from collections import deque

from cross_attn.actor import Actor
from cross_attn.critic import Critic
from replay_buffer import PrioritizedReplayBuffer

#================================================================================
class TD3:
    '''### Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent'''
    def __init__(self, n_stocks, n_features, 
                       n_hidden, p_dropout,
                       max_action, actor_lr, critic_lr, 
                       batch_size, buffer_size, n_step, 
                       alpha, beta, beta_incr, gamma, tau):
        
        # Set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the actor and critic networks
        self.actor = Actor(n_features+1, n_hidden, p_dropout, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(n_stocks+1, n_features+1, n_hidden, p_dropout).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Initialize the replay buffer
        # container = torch.empty((n_stocks, n_features+1))
        # storage = TensorStorage(container, buffer_size)
        # self.replay_buffer = PrioritizedReplayBuffer(alpha, beta, storage=storage, batch_size=batch_size)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha, beta, beta_incr)

        # Set the hyperparameters
        self.max_action = max_action                # Maximum action value
        self.batch_size = batch_size                # Batch size for training

        self.gamma = gamma                       # Discount factor
        self.tau = tau                              # Target network update rate
        self.grad_clip = 1.0                        # Gradient clipping parameter
        self.policy_noise = 0.2 * max_action        # Noise added to target policy during critic update
        self.noise_clip = 0.5 * max_action          # Range to clip target policy noise
        self.policy_freq = 2                        # Delayed policy update frequency
        
        self.total_it = 0                           # Total iterations
        self.n_step = n_step                        # Number of steps for n-step return
        self.n_step_buffer = deque(maxlen=n_step)   # Buffer for n-step return

    #----------------------------------------
    def select_action(self, state, deterministic=False):
        '''### Select an action from the policy
        Args:
            state (np.array): The input state (n_stocks, n_features)
            deterministic (bool): Whether to select the action deterministically
        Returns:
            np.array: The selected action (n_stocks)
        '''
        with torch.no_grad():
            actions = self.actor(state, deterministic)

        # if not deterministic:
        #     noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        #     actions = (actions + noise).clamp(-self.max_action, self.max_action).squeeze(0)

        return actions
    

    #----------------------------------------
    def add_experience(self, state, action, reward, next_state, done):
        '''### Add a new transition to the replay buffer
        Args:
            state (np.array): The current state (n_stocks, n_features)
            action (np.array): The action taken (n_stocks)
            reward (float): The reward received
            next_state (np.array): The next state (n_stocks, n_features)
            done (bool): The termination signal
        '''
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Add the n-step transition to the replay buffer
        if len(self.n_step_buffer) < self.n_step:
            return

        # Compute the n-step return
        reward, next_state, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, s_next, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (s_next, d) if d else (next_state, done)

        state, action = self.n_step_buffer[0][:2]

        state = state.cpu().numpy()
        action = action.cpu().numpy()
        next_state = next_state.cpu().numpy()

        # Add the n-step transition to the replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

    #----------------------------------------
    def train(self):
        '''### Train the agent'''
        self.total_it += 1

        if len(self.replay_buffer) < 90000:
            return

        with torch.no_grad():
            # Sample a batch of transitions from the replay buffer
            samples, idxs, weights = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = samples
            
            states = torch.FloatTensor(states).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
            dones = torch.FloatTensor(dones).reshape(-1, 1).to(self.device)
            weights = torch.FloatTensor(weights).reshape(-1, 1).to(self.device)

            # Select actions according to policy and add clipped noise
            next_actions = self.actor_target(next_states)
            # noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute critic loss
        td_error = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        critic_err = (td_error * weights).mean()
        self.replay_buffer.update_priorities(idxs, td_error.cpu().data.numpy().flatten())

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_err.backward()
        clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        self.actor.reset_noise()
        self.actor_target.reset_noise()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
