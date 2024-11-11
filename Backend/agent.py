from dataclasses import dataclass
import numpy as np
import torch as t
from torch import Tensor
from networks import Q_net
from game import Observation
from typing import Sequence
import einops

@dataclass(slots=True)
class AgentConfig:
    board_size: int
    connect: int
    channels: Sequence[int]
    kernel_sizes: Sequence[int]
    player_side: int = 1
    epsilon: float = 0.2
    epsilon_multiplier_per_nn_sync: float = 0.99
    lr: float = 0.003

class PlayerAgentDQN:
    def __init__(self, cfg: AgentConfig) -> None:
    
        if cfg.player_side != -1 and cfg.player_side != 1:
            raise ValueError(f"Side must be 1 or -1, not {cfg.player_side}.")
        
        self.side = cfg.player_side
        self.board_size = cfg.board_size
        self.connect = cfg.connect
        self.epsilon = cfg.epsilon
        self.action_space_size = cfg.board_size ** 2
        self.epsilon_multiplier_per_nn_sync = cfg.epsilon_multiplier_per_nn_sync

        self.value_network = Q_net(channels=cfg.channels, kernel_sizes=cfg.kernel_sizes)
        self.target_network = Q_net(channels=cfg.channels, kernel_sizes=cfg.kernel_sizes)
        self.sync_networks()

        self.rng = np.random.default_rng()
        self.optimizer = t.optim.Adam(params=self.value_network.parameters(), lr=cfg.lr)

    
    def sync_networks(self) -> None:
        """
        Copies the parameters of the value network to the target network, unlinking the target parameters from torch grad computation.
        """
        self.target_network.load_state_dict(self.value_network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False


    def greedy_policy(self, observation: Observation) -> int:
        """
        Returns the coordinates for the best move according to the value network
        """

        input = self._observation_to_nn_input(observation)

        # Pass board state through value network and get coordinates of the highest value move
        with t.inference_mode():
            state = t.tensor(input, dtype=t.float32)
            action_values = np.asarray(self.value_network(state))
            action_values2d = einops.rearrange(action_values, "... w h -> (... w) h")
            max_action = np.argmax(action_values2d)
            return max_action

    def epsilon_greedy_policy(self, observation: Observation) -> int:
        """
        Returns the coordinates for a random move with probability self.epsilon and 
        the best move according to the value network otherwise.
        """
        if self.rng.random() < self.epsilon:
            action = self.rng.integers(low=0, high=self.action_space_size, size=1)[0]
            return action
        else:
            action = self.greedy_policy(observation)
            return action

    def get_player_side_as_char(self) -> str:
        if self.side == 1:
            return "X"
        else:
            return "O"
        
    def decay_epsilon(self) -> None:
        self.epsilon *= self.epsilon_multiplier_per_nn_sync
        
    def _observation_to_nn_input(self, observation: Observation) -> Tensor:
        
        if self.side == 1:
            input = observation.board
        else:
            input = observation.board[(1,0,2),...]
        
        return input

        

