import numpy as np
import torch as t
import torch.nn as nn
from itertools import chain
from typing import Iterable

class Q_net(nn.Module):
    """
    A convolutional network that takes a game board and returns the probabilities for the best next move.
    """
    def __init__(self, channels:Iterable[int], kernel_sizes:Iterable[int]) -> None:

        super().__init__()

        if channels[0] != 3 or channels[-1] != 1:
            raise ValueError(f"Q net must have 3 input and 1 output channel, not {channels} channels.")

        # Generate alternating Conv2d and ReLU layers using passed channel and kernel sizes
        layers = list(chain.from_iterable([
            [nn.Conv2d(in_channels=channels[index], out_channels=channels[index + 1], kernel_size=kernel, 
                padding=(kernel - 1) // 2, padding_mode="zeros"),
                nn.ReLU()]
            for index, kernel in enumerate(kernel_sizes)]))
        
        # Replace last ReLU with a 2d Softmax and create a sequential module
        layers.pop()

        layers.append(nn.Softmax2d())

        self.network = nn.Sequential(layers)

    def forward(self, input:t.Tensor):
        return self.network.forward(input)
