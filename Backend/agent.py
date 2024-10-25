import numpy as np
import torch as t
from typing import Tuple



class Agent:
    def __init__(self, side) -> None:
        if side != -1 and side != 1:
            raise ValueError(f"Side must be 1 or -1, not {side}.")
        self.side = side

    def move(self, observation) -> Tuple[int, int]:
        pass
