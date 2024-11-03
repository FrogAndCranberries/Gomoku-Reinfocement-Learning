import numpy as np
import torch as t
from typing import Tuple
from networks import Q_net

class Player_agent_DQN:
    def __init__(self, board_size, connect, side, 
                 channels = [1,4,8,16,1], kernel_sizes = [5, 5, 4, 3], 
                 epsilon = 0.05, lr = 0.003) -> None:
        

        if side != -1 and side != 1:
            raise ValueError(f"Side must be 1 or -1, not {side}.")
        

        self.side = side
        self.board_size = board_size
        self.connect = connect
        self.epsilon = epsilon


        self.value_network = Q_net(channels=channels, kernel_sizes=kernel_sizes)
        self.target_network = Q_net(channels=channels, kernel_sizes=kernel_sizes)
        self.sync()


        self.rng = np.random.default_rng()
        self.optimizer = t.optim.Adam(params=self.value_network.parameters, lr=lr, maximize=True)

    
    def sync(self):
        """
        Copies the parameters of the value network to the target network.
        """
        self.target_network.load_state_dict(self.value_network.state_dict())


    def greedy_policy(self, observation):
        """
        Returns the coordinates for the best move according to the value network
        """

        # Flip board symbols if the agent plays as the -1 symbol and check input compatibility
        board_standardized = observation.board * self.side
        if not np.all(board_standardized.shape == self.board_size):
            raise ValueError(f"Observed board dimensions {board_standardized.shape} do not correspont to board size {self.board_size}.")
        
        # Pass board state through value network and get coordinates of the highest value move
        with t.no_grad():
            state = t.tensor(board_standardized, dtype=t.float32)
            values = self.value_network(state)
            max_action = t.unravel_index(t.argmax(values), values.shape)
            return max_action
    
    def epsilon_greedy_policy(self, observation):
        """
        Returns the coordinates for a random move with probability self.epsilon and 
        the best move according to the value network otherwise.
        """
        if self.rng.random() < self.epsilon:
            indices = self.rng.integers(0, self.board_size, 2)
            return indices
        else:
            indices = self.greedy_policy(observation)
            return indices

    def side_as_char(self):
        if self.side == 1:
            return "X"
        else:
            return "O"

