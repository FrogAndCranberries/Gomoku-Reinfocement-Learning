import numpy as np
import torch as t
from typing import Tuple
from networks import Q_net
from collections import deque
from collections import namedtuple

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminated"))

class Replay_buffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append(Transition(state, action, reward, next_state, terminated))

    def pop(self):
        return self.buffer.pop()


class Player_agent_DQN:
    def __init__(self, board_size, connect, side, 
                 channels = [1,4,8,16,1], kernel_sizes = [5, 5, 4, 3], 
                 buffer_size = 100_000, epsilon = 0.05, gamma = 0.95, lr = 0.003, batch_size = 32) -> None:
        

        if side != -1 and side != 1:
            raise ValueError(f"Side must be 1 or -1, not {side}.")
        

        self.side = side
        self.board_size = board_size
        self.connect = connect


        self.value_network = Q_net(channels=channels, kernel_sizes=kernel_sizes)
        self.target = Q_net(channels=channels, kernel_sizes=kernel_sizes)
        self.buffer = Replay_buffer(buffer_size)
        self.optimizer = t.optim.Adam(params=self.value_network.parameters, lr=lr, maximize=True)

        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

    def move(self, observation) -> Tuple[int, int]:
        pass

