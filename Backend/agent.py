import numpy as np
import torch as t
from typing import Tuple
from networks import Q_net
import einops

class Player_agent_DQN:
    def __init__(self, board_size, connect, 
                 channels, kernel_sizes, side = 1,
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
        self.optimizer = t.optim.Adam(params=self.value_network.parameters(), lr=lr, maximize=True)

    
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
        if not board_standardized.shape[0] == self.board_size:
            raise ValueError(f"Observed board dimensions {board_standardized.shape} do not correspont to board size {self.board_size}.")
        
        # Pass board state through value network and get coordinates of the highest value move
        with t.no_grad():
            state = t.tensor(board_standardized[np.newaxis,np.newaxis,...], dtype=t.float32)
            action_values = np.asarray(self.value_network(state))
            action_values2d = einops.rearrange(action_values, "b c w h -> (b c w) h")
            max_action = np.unravel_index(np.argmax(action_values2d), action_values2d.shape)
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

