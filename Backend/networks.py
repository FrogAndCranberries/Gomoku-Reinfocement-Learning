import numpy as np
import torch as t
import torch.nn as nn

class Q_net(nn.Module):
    """
    A convolutional network that takes a game board and returns the probabilities for the best next move.
    """
    def __init__(self, channels, kernel_sizes):

        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_sizes[0], 
                      padding=(kernel_sizes[0] - 1) // 2, padding_mode="zeros"),
            nn.Softmax2d()
        )

    def forward(self, input:t.Tensor):
        return self.network.forward(input)
