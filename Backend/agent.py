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
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size


        self.value_network = Q_net(channels=channels, kernel_sizes=kernel_sizes)
        self.target_network = Q_net(channels=channels, kernel_sizes=kernel_sizes)
        self.sync()


        self.rng = np.random.default_rng()
        self.buffer = Replay_buffer(buffer_size)
        self.optimizer = t.optim.Adam(params=self.value_network.parameters, lr=lr, maximize=True)

    
    def sync(self):
        self.target_network.load_state_dict(self.value_network.state_dict())


    def greedy_policy(self, observation):
        board = observation["board"]
        if not np.all(board.shape == self.board_size):
            raise ValueError(f"Observed board dimensions {board.shape} do not correspont to board size {self.board_size}.")
        
        with t.no_grad():
            state = t.tensor(board, dtype=t.float32)
            values = self.value_network(state)
            max_action = t.unravel_index(t.argmax(values), values.shape)
            return max_action
    
    def epsilon_greedy_policy(self, observation):
        board = observation["board"]
        if not np.all(observation.shape == self.board_size):
            raise ValueError(f"Observation dimensions {board.shape} do not correspont to board size {self.board_size}.")
        
        if self.rng.random() < self.epsilon:
            indices = self.rng.integers(0, self.board_size, 2)
            return indices
        else:
            indices = self.greedy_policy(observation)
            return indices


