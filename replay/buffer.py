import torch

class ReplayBuffer:
    def __init__(self, train_dl, cfg):
        self.data = train_dl
        self.cfg = cfg
        self.step_offset = 2 * (cfg["window_size"] - 1)
        self.epoch_len = len(train_dl.dataset) - self.step_offset
        self.num_epochs = cfg["capacity"] // self.epoch_len
        self.batch_size = cfg["batch_size"]
        self.num_epochs_last = cfg["perc_last"] * self.batch_size
        self.asset_dim = cfg["asset_dim"]
        self.window_size = cfg["window_size"]
        self.feat_dim = cfg["feat_dim"]

        self.buffer = {
            "i": torch.zeros((self.num_epochs, self.epoch_len, 1)),
            "a": torch.zeros((self.num_epochs, self.epoch_len, self.asset_dim)),
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
        if i < self.cfg["window_size"] - 1: return
        epoch = e % self.num_epochs
        step = i - self.step_offset
        self.buffer["i"][epoch, step] = torch.tensor(i)             # (1,)
        self.buffer["a"][epoch, step] = a.reshape(self.asset_dim)   # (asset_dim,)
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
        epochs_rand = torch.randint(0, self.num_epochs, (self.batch_size - int(self.num_epochs_last),))
        epochs = torch.cat((epochs_last, epochs_rand), dim=0)
        starts = torch.randint(0, self.epoch_len - self.window_size - 1, (self.batch_size,))
        ends = starts + self.window_size

        batch_s = torch.zeros((self.batch_size, self.asset_dim, self.window_size, self.feat_dim))
        batch_a = torch.zeros((self.batch_size, self.asset_dim, 1))
        batch_r = torch.zeros((self.batch_size, 1, 1))
        batch_s_ = torch.zeros((self.batch_size, self.asset_dim, self.window_size, self.feat_dim))

        for b in range(self.batch_size):
            a = self.buffer["a"][epochs[b], starts[b]:ends[b]+1]    # (window_size+1, asset_dim)
            r = self.buffer["r"][epochs[b], ends[b]-1]              # (1,)
            
            a = a.transpose(0, 1)                               # (asset_dim, window_size+1)
            r = r.unsqueeze(-1)                                 # (1, 1)
            
            i = self.buffer["i"][epochs[b], ends[b]-1].long()   # Step number
            s = self.data.dataset[i][0]                         # (asset_dim, window_size, feat_dim)
            s_ = self.data.dataset[i+1][0]                      # (asset_dim, window_size, feat_dim)

            s[..., -1] = a.squeeze(-1)[:, :-1]                  # Replace the last column with the action
            s_[..., -1] = a.squeeze(-1)[:, 1:]                  # Replace the last column with the action

            a = a[:, -1].unsqueeze(-1)                          # (asset_dim, 1)

            batch_s[b] = s
            batch_a[b] = a
            batch_r[b] = r
            batch_s_[b] = s_

        return batch_s, batch_a, batch_r, batch_s_
    