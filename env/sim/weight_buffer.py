from config.base import WINDOW_SIZE, NUM_ASSETS

import numpy as np

class ActionBuffer:
    def __init__(self):
        """ Initialize the action buffer """
        self.buffer = np.zeros((WINDOW_SIZE, NUM_ASSETS))  # Buffer to store actions
        self.buffer[0, 0] = 1                                   # Init first action as all cash
        self.idx = 1                                            # Pointer to curr position in the buffer
        self.is_full = False                                    # Track if the buffer is full

    def update(self, action):
        """ Update the buffer with the latest action.
        Args:
            action (torch.Tensor): A tensor of shape (act_dim,) representing the action.
        """
        if action.shape != (NUM_ASSETS,):
            raise ValueError(f"Action must have shape ({NUM_ASSETS},), got {action.shape}")
        
        self.buffer[self.idx] = action              # Insert action at the current buffer position
        self.idx = (self.idx + 1) % WINDOW_SIZE     # Increment the buffer pointer
        
        # Mark buffer as full after the first full cycle
        if self.idx == 0:
            self.is_full = True

    def get_last(self):
        """ Retrieve the last action from the buffer """
        return self.buffer[(self.idx - 1) % WINDOW_SIZE]

    def get_all(self):
        """ Retrieve the buffer in shape (act_dim, window_size) with the most recent actions
            placed at the end of the buffer. Empty slots at beginning are padded with zeros.
        Returns:
            torch.Tensor: Buffer of shape (act_dim, window).
        """
        if self.is_full:
            buffer = self.buffer
        else:
            padding = np.zeros((WINDOW_SIZE - self.idx, NUM_ASSETS))
            buffer = np.concat((padding, self.buffer[:self.idx]), axis=0)

        return buffer.T

    def reset(self):
        """ Reset the buffer to the initial state """
        self.buffer = np.zeros((WINDOW_SIZE, NUM_ASSETS))
        self.buffer[0, 0] = 1
        self.idx = 1
        self.is_full = False