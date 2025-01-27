from config.base import BUFFER_SIZE, PERCENT_LATEST, BATCH_SIZE, WINDOW_SIZE, NUM_ASSETS

import torch
from loader.data_loader import StockDataLoader

class ReplayBuffer:
    def __init__(self, data: StockDataLoader):
        self.dataset = data.get_train_data().dataset

        self.feat_dim = data.get_num_features()
        self.step_offset = 2 * (WINDOW_SIZE - 1)
        self.epoch_len = data.get_train_len() - self.step_offset
        self.num_epochs = BUFFER_SIZE // self.epoch_len
        self.num_epochs_last = PERCENT_LATEST * BATCH_SIZE

        self.buffer = {
            "i": torch.zeros((self.num_epochs, self.epoch_len, 1)),
            "a": torch.zeros((self.num_epochs, self.epoch_len, NUM_ASSETS)),
            "r": torch.zeros((self.num_epochs, self.epoch_len, 1, 1))
        }

    def add(self, e, i, a, r):
        """ Add an experience to the replay buffer
        Args:
            e (int): Epoch number
            i (int): Step number
            a (torch.Tensor): Action tensor of shape (asset_dim, 1)
            r (torch.Tensor): Reward tensor of shape (1,)
        """
        if i < WINDOW_SIZE - 1: return
        epoch = e % self.num_epochs
        step = i - self.step_offset
        self.buffer["i"][epoch, step] = torch.tensor(i)             # (1,)
        self.buffer["a"][epoch, step] = a.reshape(NUM_ASSETS)   # (asset_dim,)
        self.buffer["r"][epoch, step] = r.reshape(1)                # (1,)
        
    def sample(self):
        """ Sample a batch of trajectories from the replay buffer
        Returns:
            s (torch.Tensor): State tensor of shape (asset_dim, window_size, feat_dim)
            a (torch.Tensor): Action tensor of shape (asset_dim, window_size, 1)
            r (torch.Tensor): Reward tensor of shape (1, window_size)
            s_ (torch.Tensor): Next state tensor of shape (asset_dim, window_size, feat_dim)
        """
        epochs_last = torch.tensor([self.num_epochs-1]*int(self.num_epochs_last))
        epochs_rand = torch.randint(0, self.num_epochs, (BATCH_SIZE - int(self.num_epochs_last),))
        epochs = torch.cat((epochs_last, epochs_rand), dim=0)
        starts = torch.randint(0, self.epoch_len - WINDOW_SIZE - 1, (BATCH_SIZE,))
        ends = starts + WINDOW_SIZE

        batch_s = torch.zeros((BATCH_SIZE, NUM_ASSETS, WINDOW_SIZE, self.feat_dim))
        batch_a = torch.zeros((BATCH_SIZE, NUM_ASSETS, 1))
        batch_r = torch.zeros((BATCH_SIZE, 1, 1))
        batch_s_ = torch.zeros((BATCH_SIZE, NUM_ASSETS, WINDOW_SIZE, self.feat_dim))

        for b in range(BATCH_SIZE):
            a = self.buffer["a"][epochs[b], starts[b]:ends[b]+1]    # (window_size+1, asset_dim)
            r = self.buffer["r"][epochs[b], ends[b]-1]              # (1,)
            
            a = a.transpose(0, 1)                               # (asset_dim, window_size+1)
            r = r.unsqueeze(-1)                                 # (1, 1)
            
            i = self.buffer["i"][epochs[b], ends[b]-1].long()   # Step number
            s = self.dataset[i][0]                         # (asset_dim, window_size, feat_dim)
            s_ = self.dataset[i+1][0]                      # (asset_dim, window_size, feat_dim)

            s[..., -1] = a.squeeze(-1)[:, :-1]                  # Replace the last column with the action
            s_[..., -1] = a.squeeze(-1)[:, 1:]                  # Replace the last column with the action

            a = a[:, -1].unsqueeze(-1)                          # (asset_dim, 1)

            batch_s[b] = s
            batch_a[b] = a
            batch_r[b] = r
            batch_s_[b] = s_

        return batch_s, batch_a, batch_r, batch_s_
    