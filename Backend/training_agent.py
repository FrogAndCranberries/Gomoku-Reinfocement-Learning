from game import *
from agent import PlayerAgentDQN, AgentConfig
from collections import namedtuple
from collections import deque
import matplotlib.pyplot as plt
import einops
import random
import time
import torch as t
from torch import Tensor
import numpy as np
from tqdm import tqdm
from opponent import *
from dataclasses import dataclass

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminated"))
Reward_values = namedtuple("Reward_values", ("standard", "invalid", "win", "loss", "draw"))

class ReplayBuffer:
    def __init__(self, size: int) -> None:
        self.buffer = deque(maxlen=size)

    def push(self, transition: Transition) -> None:
        """
        Append a transition to the buffer.
        """
        self.buffer.append(transition)

    def sample(self, size: int) -> list[Transition]:
        """
        Sample N transitions from the buffer.
        """
        return random.sample(self.buffer, size)
    
@dataclass(slots=True)
class TrainingConfig:
    opponent_type: str = "random"
    first_player: Side = Side.X
    reward_values: Reward_values = Reward_values(0, -10, 100, -100, 1)
    buffer_size: int = 200_000
    batch_size: int = 32
    gamma: float = 0.95
    lr = 0.003


class TrainingAgent:
    def __init__(self, player_agent: PlayerAgentDQN, cfg: TrainingConfig) -> None:
                
        self.agent: PlayerAgentDQN = player_agent
        self.reward_values: Reward_values = cfg.reward_values
        self.batch_size: int = cfg.batch_size
        self.gamma: float = cfg.gamma
        self.replay_buffer: ReplayBuffer = ReplayBuffer(cfg.buffer_size)
        self.game: Game = Game(
            board_size=player_agent.board_size, 
            connect=player_agent.connect, 
            first_player=cfg.first_player, 
            opponent_type=cfg.opponent_type)
        self.losses: list = []
        self.optimizer = t.optim.Adam(params=self.agent.value_network.parameters(), lr=cfg.lr)

    def run_training_loop(self, 
                          interaction_steps: int = 10, 
                          loops: int = 50_000, 
                          switch_sides: bool = True, 
                          side_switch_period: int = 5_000, 
                          sync_period: int = 1_000) -> list[int]:
        """
        The player agent interacts with the environment and updates its network weights in a training loop, 
        periodically syncing value and target network weights.
        """
        self.losses = []
        self._add_experience_to_buffers()

        for i in tqdm(range(loops)):

            loss = self._update_value_network_weights()
            self.losses.append(loss)
            self._play_game(interaction_steps)

            if i % sync_period == 0:
                self.agent.sync_networks()
                self.agent.decay_epsilon()
            
            if switch_sides and i % side_switch_period == 0:
                self.game.first_player = game.first_player.switch

        return self.losses

    def plot_losses(self) -> None:
        """
        Plots the losses accumulated in the training loop.
        """
        # Smooth out the losses
        smoothed_losses = np.convolve(np.array(self.losses), 
                                      np.ones(len(self.losses) // 50) / (len(self.losses) // 50), 
                                      mode='valid')
        plt.plot(smoothed_losses)
        plt.xlabel('Update step')
        plt.ylabel('Loss')
        plt.xlim(0, len(smoothed_losses))
        plt.ylim(0, 1.1*max(smoothed_losses))
        plt.show()

    def evaluate(self, episodes: int = 50, turn_limit: int = 100) -> tuple[float, float]:
        """
        Record rewards from several agent games and return their mean and variance.
        """
        rewards = []
        
        for _ in range(episodes):
            rewards.append(self._get_episode_reward(turn_limit))
        
        mean = np.mean(rewards)
        std = np.std(rewards)
        return mean, std

    def _update_value_network_weights(self) -> float:
        """
        Makes a single gradient step update on weights of player agent's value network 
        by calculating loss on a sample batch from replay buffer.
        Returns this iteration's loss.
        """

        states, actions, rewards, next_states, terminated = self._sample_batch_from_buffer()

        # This is like letting the target network "peek" one step into the future for hopefully more accurate value prediction
        target_values = self._calculate_target_values(rewards, next_states, terminated)

        Q_values = self.agent.value_network(states)
        action_indices = np.unravel_index(actions, self.game.board.shape)
        chosen_action_Q_values = Q_values[np.arange(Q_values.shape[0]),0,*action_indices]

        # Get MSE loss between evaluation of actions taken and 
        loss = t.nn.functional.mse_loss(chosen_action_Q_values, target_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def _play_game(self, total_turns: int) -> None:
        game = self.game
        game.reset()
        if game.first_player != self.agent.side:
            game.opponent_move()

        for _ in range(total_turns):

            initial_obs = game.obs
            move = self.agent.epsilon_greedy_policy(initial_obs)
            game.take_turns(move)
            reward = self._determine_reward()

            self.replay_buffer.push(Transition(initial_obs.board, move, reward, game.obs.board, game.obs.terminated))

            if game.obs.terminated:
                game.reset()
                if game.first_player != self.agent.side:
                    game.opponent_move()



    def _get_episode_reward(self, turn_limit: int) -> float:
        episode_reward = 0
        game = self.game
        game.reset()
        if game.first_player != self.agent.side:
            game.opponent_move()

        for _ in range(turn_limit):

            initial_obs = game.obs
            move = self.agent.epsilon_greedy_policy(initial_obs)
            game.take_turns(move)
            episode_reward += self._determine_reward()

            if game.obs.terminated:
                break
        
        return episode_reward


    def _add_experience_to_buffers(self) -> None:
        self._play_game(self.batch_size * 100)
        self.agent.side = self.agent.side.switch
        self._play_game(self.batch_size * 100)
        self.agent.side = self.agent.side.switch
    

    def _print_endstate(self) -> None:

        match self.game.obs.endstate:

            case MoveResult.STANDARD:
                pass

            case MoveResult.INVALID:
                pass

            case MoveResult.WON_X:
                if self.agent.side == Side.X:
                    print(f"Game won on turn {self.game.turn}.")
                else:
                    print(f"Game lost on turn {self.game.turn}.")

            case MoveResult.WON_O:
                if self.agent.side == Side.O:
                    print(f"Game won on turn {self.game.turn}.")
                else:
                    print(f"Game lost on turn {self.game.turn}.")

            case MoveResult.DRAW:
                print(f"Game drawn on turn {self.game.turn}.")

    def _calculate_target_values(self, rewards: Tensor, next_states: Tensor, terminated: Tensor) -> Tensor:
        next_state_values = self.agent.target_network(next_states)
        max_next_values =  einops.reduce(next_state_values, "b c i j -> b", "max")
        max_next_values[terminated] = 0
        target_values = rewards + self.gamma * max_next_values
        return target_values

    def _sample_batch_from_buffer(self) -> tuple[Tensor, ...]:

        batch = self.replay_buffer.sample(self.batch_size)

        states      = t.tensor(np.array([transition.state for transition in batch]), dtype=t.float32)
        actions     = t.tensor(np.array([transition.action for transition in batch]), dtype=t.int32)
        rewards     = t.tensor(np.array([transition.reward for transition in batch]), dtype=t.float32)
        next_states = t.tensor(np.array([transition.next_state for transition in batch]), dtype=t.float32)
        terminated  = t.tensor(np.array([transition.terminated for transition in batch]), dtype=t.bool)

        return states,actions,rewards,next_states,terminated


    def _determine_reward(self) -> float:

        match self.game.last_move_result:

            case MoveResult.STANDARD:
                return self.reward_values.standard
            
            case MoveResult.INVALID:
                return self.reward_values.invalid

            case MoveResult.WON_X:
                if self.agent.side == Side.X:
                    return self.reward_values.win
                else:
                    return self.reward_values.loss

            case MoveResult.WON_O:
                if self.agent.side == Side.O:
                    return self.reward_values.win
                else:
                    return self.reward_values.loss

            case MoveResult.DRAW:
                return self.reward_values.draw
            

if __name__ == "__main__":

    # agent = Player_agent_DQN(board_size=3, connect=3, player_side=-1,channels=[3,4,8,1], kernel_sizes=[3,3,3])
    # agent.value_network.load_state_dict(t.load("backend/models/value_network_3x3_3layer.pth"))
    # game = Game(3,3,1)
    # obs = game.reset()
    # game.print_board()
    # print("\n")
    # while True:

    #     move = input()
    #     move = list(map(lambda x: int(x), move.split(' ')))
    #     print(move)
    #     obs = game.make_move(move)
    #     game.print_board()
    #     print("\n")

    #     agent_move = agent.greedy_policy(obs)
    #     if not game.is_move_valid(agent_move):
    #         agent_move = input()
    #         agent_move = list(map(lambda x: int(x), agent_move.split(' ')))

    #     obs = game.make_move(agent_move)

    #     game.print_board()
    #     print("\n")



    agentCfg = AgentConfig(
        board_size=5, connect=4, player_side=Side.X, channels=[3,4,8,1], kernel_sizes=[4,4,4]
    )

    trainingCfg = TrainingConfig(
        opponent_type="random",
        first_player=Side.X
    )

    agent = PlayerAgentDQN(agentCfg)
    # agent.value_network.load_state_dict(t.load("backend/models/value_network_3x3_3layer.pth"))
    ta = TrainingAgent(player_agent=agent, cfg = trainingCfg)
    result = ta.evaluate()
    print(result)
    ta.run_training_loop(interaction_steps=10, loops=10_000, switch_sides=True, side_switch_period=3_000)
    ta.plot_losses()
    print(ta.evaluate())
    t.save(ta.agent.value_network.state_dict(), "backend/models/value_network_4x4_3layer.pth")
    # ta.visualise(turn_limit=25, delay = 1)