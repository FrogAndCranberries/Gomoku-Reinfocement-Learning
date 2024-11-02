import numpy as np
import torch as t
import torch.nn as nn

class Q_net(nn.Module):
    """
    A convolutional network that takes a game board and returns the probabilities for the best next move.
    """
    def __init__(self, board_size, channels = [1,4,8,16,1], kernel_sizes = [5, 5, 4, 3]):

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_sizes[0], padding_mode="zeros"),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernel_sizes[1], padding_mode="zeros"),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=kernel_sizes[2], padding_mode="zeros"),
            nn.Conv2d(in_channels=channels[3], out_channels=channels[4], kernel_size=kernel_sizes[3], padding_mode="zeros"),
            nn.Softmax2d()
        )

        super.init()

    def forward(self, input):
        return self.network.forward(input=input)
