import numpy as np
import torch as t
import torch.nn as nn
import einops
from itertools import chain
from typing import Iterable

class SoftmaxPlanar(nn.Module):
    """
    Apply softmax over all values within one channel.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x:t.Tensor) -> t.Tensor:
        last_dim = x.shape[-1]
        x_1d = einops.rearrange(x, "... h w -> ... (h w)")
        x_softmaxed = nn.functional.softmax(x_1d, dim=-1)
        x_2d = einops.rearrange(x_softmaxed, "... (h w) -> ... h w", w=last_dim)
        return x_2d


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
        layers.append(SoftmaxPlanar())

        self.network = nn.Sequential(*layers)

    def forward(self, x:t.Tensor) -> t.Tensor:
        return self.network.forward(x)
