from abc import ABC, abstractmethod
from game import Game
import numpy as np


class Opponent(ABC):
    
    def __init__(self):
        self.rng = np.random.default_rng()

    @abstractmethod
    def get_move(self, game: Game):
        pass
    
    def manhattan_distance_from_center(self, game: Game, index: int | np.ndarray) -> float:
        board_size = game.board_size
        i = index // board_size
        j = index % board_size

        return np.abs(i - board_size // 2) + np.abs(j - board_size // 2)
    
class RandomOpponent(Opponent):

    def get_move(self, game: Game) -> int:
        """
        Returns a random valid move for the next player.
        """
        valid_moves = game.get_valid_moves()
        random_move = self.rng.choice(valid_moves, size=1)
        return random_move
    
class CentralOpponent(Opponent):

    def get_move(self, game: Game) -> int:
        """
        Returns the most central valid move for the next player.
        """
        valid_moves = game.get_valid_moves()
        distances_from_center = self.manhattan_distance_from_center(valid_moves)
        most_central_move = valid_moves[np.argmin(distances_from_center)]
        return most_central_move

class RandomCentralOpponent(Opponent):

    def get_move(self, game: Game) -> int:
        """
        Returns a random valid move with higher probability near the board center for the next player.
        """

        # Get valid moves and order by distance from center
        valid_moves = game.get_valid_moves()
        distances_from_center = self.manhattan_distance_from_center(valid_moves)
        ordered_indices = np.argsort(distances_from_center)
        ordered_valid_moves = valid_moves[ordered_indices]

        # Get decaying probability distribution
        weights = (valid_moves.shape[0] - np.arange(valid_moves.shape[0])) ** 2
        distribution = weights / weights.sum()

        random_central_move = self.rng.choice(ordered_valid_moves, size=1, p=distribution)
        return random_central_move