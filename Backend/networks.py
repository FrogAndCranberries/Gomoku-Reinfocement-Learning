from torch import Tensor
import torch.nn as nn
import einops
from typing import Iterable

class SoftmaxPlanar(nn.Module):
    """
    Apply softmax over all values in planes along the last two dimension.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x:Tensor) -> Tensor:
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



        layers = []
        for index, kernel_size in enumerate(kernel_sizes):

            padding_left = (kernel_size - 1) // 2
            padding_right = kernel_size // 2
            padding_top = (kernel_size - 1) // 2
            padding_bottom = kernel_size // 2

            layers.append(nn.ZeroPad2d(padding=(padding_left, padding_right, padding_top, padding_bottom)))
            layers.append(nn.Conv2d(in_channels=channels[index], 
                                    out_channels=channels[index + 1], 
                                    kernel_size=kernel_size, 
                                    padding=0))
                                    
            layers.append(nn.ReLU())
        
        layers.pop()
        layers.append(SoftmaxPlanar())

        self.network = nn.Sequential(*layers)

    def forward(self, x:Tensor) -> Tensor:
        return self.network.forward(x)
