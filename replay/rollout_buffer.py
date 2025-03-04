
from config.base import WINDOW_SIZE, NUM_ASSETS, INITIAL_CASH, BATCH_SIZE

import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, num_features, train_len, train_prices):
        self.feat_dim = num_features
        self.step_offset = WINDOW_SIZE - 1
        self.epoch_len = train_len - self.step_offset
        self.prices = np.array(train_prices.transpose(0, 1)[WINDOW_SIZE-1:].unsqueeze(-1))

        self.init_act = torch.zeros((NUM_ASSETS, 1))
        self.init_act[0] = 1

        self.s = np.zeros((self.epoch_len+1, NUM_ASSETS, WINDOW_SIZE, self.feat_dim))
        self.a = np.zeros((self.epoch_len+1, NUM_ASSETS, 1))
        self.v = np.zeros((self.epoch_len+1, 1, 1))
        self.r = np.zeros((self.epoch_len+1, 1, 1))

        self.s[0] = np.zeros((NUM_ASSETS, WINDOW_SIZE, self.feat_dim))
        self.a[0] = self.init_act
        self.v[0] = torch.tensor([[INITIAL_CASH]])
        self.r[0] = torch.tensor([[0]])

        self.step = 1

    def reset(self):
        """ Reset the buffer for a new epoch """
        self.s = np.zeros((self.epoch_len+1, NUM_ASSETS, WINDOW_SIZE, self.feat_dim))
        self.a = np.zeros((self.epoch_len+1, NUM_ASSETS, 1))
        self.v = np.zeros((self.epoch_len+1, 1, 1))
        self.r = np.zeros((self.epoch_len+1, 1, 1))

        self.s[0] = np.zeros((NUM_ASSETS, WINDOW_SIZE, self.feat_dim))
        self.a[0] = self.init_act
        self.v[0] = torch.tensor([[float(INITIAL_CASH)]])
        self.r[0] = torch.tensor([[0.0]])

        self.step = 1

    def add(self, s, a, v, r):
        """ Add an experience to the replay buffer
        Args:
            s (torch.Tensor): Observation tensor of shape (asset_dim, window_size, feat_dim)
            a (torch.Tensor): Action tensor of shape (1, asset_dim)
            v (torch.Tensor): Value tensor of shape (1,)
            r (torch.Tensor): Reward tensor of shape (1,)
        """
        if self.step > self.step_offset:
            step = self.step - self.step_offset
            self.s[step] = np.array(s)
            self.a[step] = np.array(a)
            self.v[step] = np.array(v)
            self.r[step] = np.array(r)
        self.step += 1

    def sample(self):
        """ Sample a batch of trajectories from the replay buffer

        Returns:
            Iterator : An iterator over the batched trajectories\n
            **s_batch** (torch.Tensor, [batch_size, num_assets, window_size, feat_dim]): Observations tensor  
            **a_batch** (torch.Tensor, [batch_size, num_assets, 1]): Actions tensor  
            **r_batch** (torch.Tensor, [batch_size, 1, 1]): Rewards tensor  
            **_v_batch** (torch.Tensor, [batch_size, 1, 1]): Previous values tensor  
            **_a_batch** (torch.Tensor, [batch_size, num_assets, 1]): Previous actions tensor  
            **p_batch** (torch.Tensor, [batch_size, num_assets, 1]): Relative price changes tensor  
        """
        batch_size = self.epoch_len if BATCH_SIZE == -1 else BATCH_SIZE
        num_batchs = self.epoch_len // batch_size

        s_batch = torch.zeros((num_batchs, batch_size, NUM_ASSETS, WINDOW_SIZE, self.feat_dim))
        a_batch = torch.zeros((num_batchs, batch_size, NUM_ASSETS, 1))
        r_batch = torch.zeros((num_batchs, batch_size, 1, 1))
        _v_batch = torch.zeros((num_batchs, batch_size, 1, 1))
        _a_batch = torch.zeros((num_batchs, batch_size, NUM_ASSETS, 1))
        p_batch = torch.zeros((num_batchs, batch_size, NUM_ASSETS, 1))

        step = 0
        curr_idx = 1
        while curr_idx < self.epoch_len and self.epoch_len - curr_idx >= batch_size:
            s = self.s[curr_idx:curr_idx+batch_size]        # (batch_size, num_assets, window_size, feat_dim)
            a = self.a[curr_idx:curr_idx+batch_size]        # (batch_size, num_assets, 1)
            r = self.r[curr_idx:curr_idx+batch_size]        # (batch_size, 1, 1)
            _v = self.v[curr_idx-1:curr_idx+batch_size-1]   # (batch_size, 1, 1)
            _a = self.a[curr_idx-1:curr_idx+batch_size-1]   # (batch_size, num_assets, 1)
            p = self.prices[curr_idx:curr_idx+batch_size]   # (batch_size, num_assets, 1)

            s_batch[step] = torch.tensor(s, dtype=torch.float32)
            a_batch[step] = torch.tensor(a, dtype=torch.float32)
            r_batch[step] = torch.tensor(r, dtype=torch.float32)
            _v_batch[step] = torch.tensor(_v, dtype=torch.float32)
            _a_batch[step] = torch.tensor(_a, dtype=torch.float32)
            p_batch[step] = torch.tensor(p, dtype=torch.float32)

            step += 1
            curr_idx += batch_size

        return iter(zip(s_batch, a_batch, r_batch, _v_batch, _a_batch, p_batch))

    def sample_random(self):
        """ Sample a random batch of trajectories from the replay buffer

        Returns:
            Iterator : An iterator over the batched trajectories\n
            **s_batch** (torch.Tensor, [batch_size, num_assets, window_size, feat_dim]): Observations tensor  
            **a_batch** (torch.Tensor, [batch_size, num_assets, 1]): Actions tensor  
            **r_batch** (torch.Tensor, [batch_size, 1, 1]): Rewards tensor  
            **_v_batch** (torch.Tensor, [batch_size, 1, 1]): Previous values tensor  
            **_a_batch** (torch.Tensor, [batch_size, num_assets, 1]): Previous actions tensor  
            **p_batch** (torch.Tensor, [batch_size, num_assets, 1]): Relative price changes tensor  
        """
        batch_size = self.epoch_len if BATCH_SIZE == -1 else BATCH_SIZE
        num_batchs = (self.epoch_len-1) // batch_size

        s_batch = torch.zeros((num_batchs, batch_size, NUM_ASSETS, WINDOW_SIZE, self.feat_dim))
        a_batch = torch.zeros((num_batchs, batch_size, NUM_ASSETS, 1))
        r_batch = torch.zeros((num_batchs, batch_size, 1, 1))
        _v_batch = torch.zeros((num_batchs, batch_size, 1, 1))
        _a_batch = torch.zeros((num_batchs, batch_size, NUM_ASSETS, 1))
        p_batch = torch.zeros((num_batchs, batch_size, NUM_ASSETS, 1))

        idxs = np.random.choice(np.arange(1, self.epoch_len), (num_batchs, batch_size), replace=False)

        for step, idx in enumerate(idxs):
            s = self.s[idx]        # (batch_size, num_assets, window_size, feat_dim)
            a = self.a[idx]        # (batch_size, num_assets, 1)
            r = self.r[idx]        # (batch_size, 1, 1)
            _v = self.v[idx-1]     # (batch_size, 1, 1)
            _a = self.a[idx-1]     # (batch_size, num_assets, 1)
            p = self.prices[idx]   # (batch_size, num_assets, 1)

            s_batch[step] = torch.tensor(s, dtype=torch.float32)
            a_batch[step] = torch.tensor(a, dtype=torch.float32)
            r_batch[step] = torch.tensor(r, dtype=torch.float32)
            _v_batch[step] = torch.tensor(_v, dtype=torch.float32)
            _a_batch[step] = torch.tensor(_a, dtype=torch.float32)
            p_batch[step] = torch.tensor(p, dtype=torch.float32)

        return iter(zip(s_batch, a_batch, r_batch, _v_batch, _a_batch, p_batch))