from game import Game
from agent import Player_agent_DQN

class Training_agent:
    def __init__(self, player_agent:Player_agent_DQN, size:int, connect:int, opponent_type="random") -> None:
        self.game = Game(size, connect)
        self.agent = player_agent
        allowed_opponent_types = ("random")
        if opponent_type not in allowed_opponent_types:
            raise ValueError(f"Opponent type '{opponent_type}' not in {allowed_opponent_types}.")
        self.opponent_type = opponent_type

    def training_loop():
        pass

