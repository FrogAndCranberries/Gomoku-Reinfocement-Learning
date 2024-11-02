from game import Game, Endstate, Observation
from agent import Player_agent_DQN
from collections import namedtuple
from collections import deque
import einops
import random
import torch as t

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminated"))
Reward_values = namedtuple("Reward_values", ("valid", "invalid", "win", "loss", "draw"))

class Replay_buffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, terminated):
        """
        Append a transition to the buffer.
        """
        self.buffer.append(Transition(state, action, reward, next_state, terminated))

    def sample(self, size):
        """
        Sample N transitions from the buffer.
        """
        return random.sample(self.buffer, size)
    

class Training_agent:
    def __init__(self, player_agent:Player_agent_DQN, size:int, connect:int, 
                 opponent_type = "random", reward_values = Reward_values(1,-10,1000,-1000,10),
                 buffer_size = 200_000, batch_size = 32, gamma = 0.95) -> None:
        
        # Check opponent type is valid.
        allowed_opponent_types = ("random", "central", "random_central")
        if opponent_type not in allowed_opponent_types:
            raise ValueError(f"Opponent type '{opponent_type}' not in {allowed_opponent_types}.")
        
        # Assign values
        self.agent = player_agent
        self.opponent_type = opponent_type
        self.reward_values = reward_values
        self.batch_size = batch_size
        self.gamma = gamma

        self.buffer = Replay_buffer(buffer_size)
        self.game = Game(size, connect)

    def interact(self, steps:int) -> None:
        """
        The agent interacts with the game for N steps with epsilon greedy policy, saving the transitions into the replay buffer.
        """

        # Reset environment and ensure it's agent's turn
        obs = self.game.reset()

        if self.agent.side != self.game.next_turn:
            obs = self.opponent_move()

        for _ in range(steps):
            
            # Reset game if terminated and get agent to play
            if obs.terminated:
                obs = self.game.reset()

            move = self.agent.epsilon_greedy_policy(obs)

            # If move is not valid, do not advance the game and just record the transition with a punishment for invalid move
            if not self.game.is_move_valid(move):
                reward = self.reward_values.invalid
                self.buffer.push(obs.board, move, reward, obs.board, terminated=obs.terminated)
                continue
            
            # Else play the move, if the game has not terminated let opponent play, then get a reward according to game end state
            next_obs = self.game.play(move)
            if not next_obs.terminated:
                next_obs = self.opponent_move()
            
            match next_obs.endstate:

                case Endstate.NONE:
                    reward = self.reward_values.valid

                case Endstate.WON_1:
                    if self.agent.side == 1:
                        reward = self.reward_values.win
                    else:
                        reward = self.reward_values.loss

                case Endstate.LOST_1:
                    if self.agent.side == 1:
                        reward = self.reward_values.loss
                    else:
                        reward = self.reward_values.win

                case Endstate.DRAW:
                    reward = self.reward_values.draw

            # Save transition and update observation
            self.buffer.push(obs.board, move, reward, next_obs.board, next_obs.terminated)

            obs = next_obs
    
    def update_weights(self) -> float:
        """
        Makes a single gradient step update on weights of player agent's value network 
        by calculating loss on a sample batch from replay buffer.
        Returns this iteration's loss.
        """

        # Get a sample batch from replay buffer and parse into separate tensors
        batch = self.buffer.sample(self.batch_size)

        states      = t.tensor([transition["state"] for transition in batch], dtype=t.int8)
        actions     = t.tensor([transition["action"] for transition in batch], dtype=t.int16)
        rewards     = t.tensor([transition["reward"] for transition in batch], dtype=t.float32)
        next_states = t.tensor([transition["next_state"] for transition in batch], dtype=t.int8)
        terminated  = t.tensor([transition["terminated"] for transition in batch], dtype=t.bool)

        # Get target values by summing rewards and target network's evaluation of next states
        # This is like letting the target network "peek" one step into the future for a hopefully more accurate evaluation
        next_state_values = self.agent.target_network(next_states)
        max_next_values =  einops.reduce(next_state_values, "b i j -> b", "max")
        max_next_values[terminated] = 0
        target_values = rewards + self.gamma * max_next_values

        # Get current value network's evaluation of the taken actions
        Q_values = self.agent.value_network(states)
        chosen_action_Q_values = Q_values[t.arange(Q_values.size[0]), actions]

        # Get MSE loss and make a gradient step on value network's parameters
        loss = t.nn.functional.mse_loss(chosen_action_Q_values, target_values)

        self.agent.optimizer.zero_grad()

        loss.backward()

        self.agent.optimizer.step()

        return loss.item()

    def opponent_move(self) -> Observation:
        """
        Calls the opponent move method of self.game corresponding to self.opponent_type.
        Returns a new observation.
        """
        match self.opponent_type:

            case "random":  
                return self.game.random_move()
            
            case "central":
                return self.game.most_central_move()
            
            case "random_central":
                return self.game.random_central_move()
            
            case _:
                raise NotImplementedError(f"Opponent type {self.opponent_type} not implemented.")

    def training_loop(self):
        pass

