from game import Game
from agent import Player_agent_DQN
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

