from game import *
from agent import Player_agent_DQN
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

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminated"))
Reward_values = namedtuple("Reward_values", ("valid", "invalid", "win", "loss", "draw"))

class Replay_buffer:
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
    

class Training_agent:
    def __init__(self, player_agent: Player_agent_DQN, 
                 size: int, 
                 connect: int, 
                 opponent_type: str = "random",
                 first_player: int = 1,
                 reward_values: Reward_values = Reward_values(0, -10, 100, -100, 1),
                 buffer_size: int = 200_000, 
                 batch_size: int = 32, 
                 gamma: float = 0.95) -> None:
                
        # Assign values
        self.agent: Player_agent_DQN = player_agent
        self.reward_values: Reward_values = reward_values
        self.batch_size: int = batch_size
        self.gamma: float = gamma
        self.replay_buffer_X: Replay_buffer = Replay_buffer(buffer_size)
        self.replay_buffer_O: Replay_buffer = Replay_buffer(buffer_size)
        self.game: Game = Game(size, connect, first_player)
        self.opponent: Opponent = self._create_opponent(opponent_type)
        self.losses: list = []

    def run_training_loop(self, interaction_steps: int = 10, 
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
            if self.agent.side == self.game.first_player:
                self._play_game_as_first_player(interaction_steps)
            else:
                self._play_game_as_second_player(interaction_steps)

            if i % sync_period == 0:
                self.agent.sync_networks()
                self.agent.decay_epsilon()
            
            if switch_sides and i % side_switch_period == 0:
                self.agent.side *= -1

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

        self.agent.optimizer.zero_grad()

        loss.backward()

        self.agent.optimizer.step()

        return loss.item()

    def _play_game_as_first_player(self, total_turns: int) -> None:
        """
        The agent interacts with the game for N steps with epsilon greedy policy, saving the transitions into the replay buffer.
        """
        game = self.game
        game.reset()

        for _ in range(total_turns):
            
            initial_obs = game.obs

            agent_move = self.agent.epsilon_greedy_policy(game.obs)

            if not self.game.is_move_valid(agent_move):
                reward = self._determine_reward(valid_move=False)
                self._push_to_buffer(game.obs.board, agent_move, reward, game.obs.board, game.obs.terminated)
                continue
            
            else:
                self.game.make_move(agent_move)

                if not game.obs.terminated:
                    self._make_opponent_move()
                
                reward = self._determine_reward(valid_move=True)

                self._push_to_buffer(initial_obs.board, agent_move, reward, game.obs.board, initial_obs.terminated)

            if game.obs.terminated:
                game.reset()


    def _play_game_as_second_player(self, total_turns: int) -> None:
        """
        The agent interacts with the game for N steps with epsilon greedy policy, saving the transitions into the replay buffer.
        """
        game = self.game
        game.reset()
        self._make_opponent_move()

        for _ in range(total_turns):

            initial_obs = game.obs

            agent_move = self.agent.epsilon_greedy_policy(game.obs)

            if not self.game.is_move_valid(agent_move):
                reward = self._determine_reward(valid_move=False)
                self._push_to_buffer(game.obs.board, agent_move, reward, game.obs.board, game.obs.terminated)
                continue
            
            else:

                self.game.make_move(agent_move)

                if not game.obs.terminated:
                    self._make_opponent_move()
                
                reward = self._determine_reward(valid_move=True)

                self._push_to_buffer(initial_obs.board, agent_move, reward, game.obs.board, initial_obs.terminated)
                        
            if game.obs.terminated:
                game.reset()
                self._make_opponent_move()



    def _get_episode_reward(self, turn_limit: int) -> float:
        episode_reward = 0
        game = self.game
        game.reset()

        if self.agent.side != game.next_player:
            self._make_opponent_move()

        for _ in range(turn_limit):
            if game.obs.terminated:
                break

            agent_move = self.agent.epsilon_greedy_policy(game.obs)

            if not self.game.is_move_valid(agent_move):
                episode_reward += self._determine_reward(valid_move=False)
                continue
            
            else:
                self.game.make_move(agent_move)
                if not game.obs.terminated:
                    self._make_opponent_move()
                
                episode_reward += self._determine_reward(valid_move=True)
        
        return episode_reward


    def _add_experience_to_buffers(self) -> None:
        self._play_game_as_first_player(self.batch_size * 100)
        self.agent.side *= -1
        self._play_game_as_first_player(self.batch_size * 100)
        self.agent.side *= -1
    

    def _print_endstate(self) -> None:

        match self.game.obs.endstate:

            case Endstate.NONE:
                pass

            case Endstate.WON_X:
                if self.agent.side == 1:
                    print(f"Game won on turn {self.game.turn}.")
                else:
                    print(f"Game lost on turn {self.game.turn}.")

            case Endstate.WON_O:
                if self.agent.side == 1:
                    print(f"Game lost on turn {self.game.turn}.")
                else:
                    print(f"Game won on turn {self.game.turn}.")

            case Endstate.DRAW:
                print(f"Game drawn on turn {self.game.turn}.")

    def _push_to_buffer(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated: bool) -> None:

        if agent.side == 1:
            self.replay_buffer_X.push(Transition(state, action, reward, next_state, terminated))
        else:
            state = state[(1,0,2),...]
            next_state = next_state[(1,0,2),...]
            self.replay_buffer_O.push(Transition(state, action, reward, next_state, terminated))

    def _calculate_target_values(self, rewards: Tensor, next_states: Tensor, terminated: Tensor) -> Tensor:
        next_state_values = self.agent.target_network(next_states)
        max_next_values =  einops.reduce(next_state_values, "b c i j -> b", "max")
        max_next_values[terminated] = 0
        target_values = rewards + self.gamma * max_next_values
        return target_values

    def _sample_batch_from_buffer(self) -> tuple[Tensor, ...]:

        if self.agent.side == 1:
            batch = self.replay_buffer_X.sample(self.batch_size)
        else:
            batch = self.replay_buffer_O.sample(self.batch_size)

        states      = t.tensor(np.array([transition.state for transition in batch]), dtype=t.float32)
        actions     = t.tensor(np.array([transition.action for transition in batch]), dtype=t.int32)
        rewards     = t.tensor(np.array([transition.reward for transition in batch]), dtype=t.float32)
        next_states = t.tensor(np.array([transition.next_state for transition in batch]), dtype=t.float32)
        terminated  = t.tensor(np.array([transition.terminated for transition in batch]), dtype=t.bool)

        return states,actions,rewards,next_states,terminated

    def _make_opponent_move(self) -> None:
            opponent_move = self.opponent.get_move()
            self.game.make_move(opponent_move)

    def _determine_reward(self, valid_move=True) -> float:

        if not valid_move:
            return self.reward_values.invalid

        match self.game.endstate:

            case Endstate.NONE:
                return self.reward_values.valid

            case Endstate.WON_X:
                if self.agent.side == 1:
                    return self.reward_values.win
                else:
                    return self.reward_values.loss

            case Endstate.WON_O:
                if self.agent.side == 1:
                    return self.reward_values.loss
                else:
                    return self.reward_values.win

            case Endstate.DRAW:
                return self.reward_values.draw
            
    def _create_opponent(self, opponent_type) -> Opponent:
        
        opponent_types = {
            "random": RandomOpponent,
            "central": CentralOpponent,
            "random_central": RandomCentralOpponent
        }

        opponent = opponent_types.get(opponent_type)

        if opponent is None:
            raise KeyError(f"Opponent type {opponent_type} not implemented.")
        else:
            return opponent(game=self.game)



if __name__ == "__main__":

    # agent = Player_agent_DQN(board_size=3, connect=3, player_side=-1,channels=[3,4,8,1], kernel_sizes=[3,3,3])
    # agent.value_network.load_state_dict(t.load("value_network_3x3_3layer.pth"))
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





    agent = Player_agent_DQN(board_size=5, connect=4, player_side=1,channels=[3,4,8,1], kernel_sizes=[4,4,4])
    # agent.value_network.load_state_dict(t.load("value_network_3x3_3layer.pth"))
    ta = Training_agent(player_agent=agent, size=5, connect=4, opponent_type="random_central", buffer_size=50_000)
    result = ta.evaluate()
    print(result)
    ta.run_training_loop(interaction_steps=10, loops=10_000, switch_sides=True, side_switch_period=3_000)
    ta.plot_losses()
    print(ta.evaluate())
    t.save(ta.agent.value_network.state_dict(), "value_network_4x4_3layer.pth")
    # ta.visualise(turn_limit=25, delay = 1)