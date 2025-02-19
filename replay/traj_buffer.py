from config.base import BUFFER_SIZE, INCLUDE_LAST, BATCH_SIZE, WINDOW_SIZE, NUM_ASSETS

import torch
from loader.data_loader import StockDataLoader

class ReplayBuffer:
    def __init__(self, data: StockDataLoader):
        self.dataset = data.get_train_data().dataset

        self.feat_dim = data.get_num_features()
        self.step_offset = 2 * (WINDOW_SIZE - 1)
        self.epoch_len = data.get_train_len() - self.step_offset
        self.max_epoch = BUFFER_SIZE // self.epoch_len

        self.curr_epoch = 0
        self.full = False
        self.buffer = {
            "i": torch.zeros((self.max_epoch, self.epoch_len, 1)),
            "a": torch.zeros((self.max_epoch, self.epoch_len, NUM_ASSETS)),
            "r": torch.zeros((self.max_epoch, self.epoch_len, 1, 1))
        }

    def __len__(self):
        return self.max_epoch if self.full else self.curr_epoch

    def add(self, e, i, a, r):
        """ Add an experience to the replay buffer
        Args:
            e (int): Epoch number
            i (int): Step number
            a (torch.Tensor): Action tensor of shape (asset_dim, 1)
            r (torch.Tensor): Reward tensor of shape (1,)
        """
        if i < WINDOW_SIZE - 1: return
        self.curr_epoch = int(e % self.max_epoch)
        step = i - self.step_offset

        self.buffer["i"][self.curr_epoch, step] = torch.tensor(i)             # (1,)
        self.buffer["a"][self.curr_epoch, step] = a.reshape(NUM_ASSETS)   # (asset_dim,)
        self.buffer["r"][self.curr_epoch, step] = r.reshape(1)                # (1,)

        if not self.full and self.curr_epoch == self.max_epoch - 1:
            self.full = True
        
    def sample(self):
        """ Sample a batch of trajectories from the replay buffer
        Returns:
            s (torch.Tensor): State tensor of shape (asset_dim, window_size, feat_dim)
            a (torch.Tensor): Action tensor of shape (asset_dim, window_size, 1)
            r (torch.Tensor): Reward tensor of shape (1, window_size)
            s_ (torch.Tensor): Next state tensor of shape (asset_dim, window_size, feat_dim)
        """
        choose_epochs = list(range(self.max_epoch if self.full else self.curr_epoch))
        if not INCLUDE_LAST:
            choose_epochs.remove(self.curr_epoch)

        epochs_rand = torch.randperm(len(choose_epochs))[:BATCH_SIZE-INCLUDE_LAST]
        epochs = torch.cat([torch.tensor([self.curr_epoch]), epochs_rand]) if INCLUDE_LAST else epochs_rand
        
        start = torch.randint(0, self.epoch_len - WINDOW_SIZE - 1, (1,))
        end = start + WINDOW_SIZE

        batch_s = torch.zeros((BATCH_SIZE, NUM_ASSETS, WINDOW_SIZE, self.feat_dim))
        batch_a = torch.zeros((BATCH_SIZE, NUM_ASSETS, 1))
        batch_r = torch.zeros((BATCH_SIZE, 1, 1))
        batch_s_ = torch.zeros((BATCH_SIZE, NUM_ASSETS, WINDOW_SIZE, self.feat_dim))

        for b in range(BATCH_SIZE):
            a = self.buffer["a"][epochs[b], start:end+1]    # (window_size+1, asset_dim)
            r = self.buffer["r"][epochs[b], end-1]          # (1,)
            
            a = a.transpose(0, 1)                           # (asset_dim, window_size+1)
            r = r.unsqueeze(-1)                             # (1, 1)
            
            i = self.buffer["i"][epochs[b], end-1].long()   # Step number
            s = self.dataset[i][0]                          # (asset_dim, window_size, feat_dim)
            s_ = self.dataset[i+1][0]                       # (asset_dim, window_size, feat_dim)

            s[..., -1] = a.squeeze(-1)[:, :-1]              # Replace the last column with the action
            s_[..., -1] = a.squeeze(-1)[:, 1:]              # Replace the last column with the action

            a = a[:, -1].unsqueeze(-1)                      # (asset_dim, 1)

            batch_s[b] = s
            batch_a[b] = a
            batch_r[b] = r
            batch_s_[b] = s_

        return batch_s, batch_a, batch_r, batch_s_
    