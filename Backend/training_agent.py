from game import Game, Endstate, Observation
from agent import Player_agent_DQN
from collections import namedtuple

Reward_values = namedtuple("Reward_values", ("valid", "invalid", "win", "loss", "draw"))

class Training_agent:
    def __init__(self, player_agent:Player_agent_DQN, size:int, connect:int, 
                 opponent_type="random", reward_values = Reward_values(1,-10,1000,-1000,10)) -> None:
        self.game = Game(size, connect)
        self.agent = player_agent
        allowed_opponent_types = ("random", "central", "random_central")
        if opponent_type not in allowed_opponent_types:
            raise ValueError(f"Opponent type '{opponent_type}' not in {allowed_opponent_types}.")
        self.opponent_type = opponent_type
        self.reward_values = reward_values

    def interact(self, steps:int) -> None:
        obs = self.game.reset()

        if self.agent.side != self.game.next_turn:
            obs = self.opponent_move()

        for _ in range(steps):
            
            if obs.terminated:
                obs = self.game.reset()

            move = self.agent.epsilon_greedy_policy(obs)

            if not self.game.is_move_valid(move):
                reward = self.reward_values.invalid
                self.agent.buffer.push(obs.board, move, reward, obs.board, terminated=obs.terminated)
                continue

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

            self.agent.buffer.push(obs.board, move, reward, next_obs.board, next_obs.terminated)

            obs = next_obs
                
    

    def opponent_move(self) -> Observation:
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

