import numpy as np
from enum import Enum
from scipy.signal import convolve2d
from typing import Dict, Sequence
from collections import namedtuple


# Gomoku game instance with taking turns and evaluating win conditions

class Endstate(Enum):
    NONE = 0
    WON_1 = 1
    LOST_1 = 2
    DRAW = 3

Observation = namedtuple("Observation", ("board", "terminated", "endstate"))

class Game:
    def __init__(self, size:int, connect:int, first_player:int = 1, seed = None):
        if connect > size:
            raise ValueError(f"Can't play connect-{connect} on board with size {size}")
        self.size = size
        self.connect = connect
        self.board = np.zeros((size, size))
        self.turn = 0
        self.first_player = first_player
        self.next_turn = first_player
        self.terminated = False
        self.endstate = Endstate.NONE
        self.rng = np.random.default_rng(seed)
        self.flat_gaussian = np.exp(np.indices((size, size), dtype=np.float32))

    def play(self, coords:Sequence[int]) -> Observation:
        i,j = coords
        # Not needed while checks are done in interact fnc for training
        # if not self.is_move_valid(i,j):
        #     raise IndexError(f"Tried to play at invalid field ({i}, {j}).")
        
        self.board[i,j] = self.next_turn
        self.next_turn *= -1
        self.turn += 1
        self.evaluate()
        return Observation(self.board, self.terminated, self.endstate)
    
    def get_valid_moves(self) -> np.ndarray:
        valid_moves = np.stack(np.where(self.board == 0)).T
        return valid_moves
        
    def is_move_valid(self, coords) -> bool:
        i,j = coords
        return i >= 0 and j >= 0 and i < self.size and j < self.size and self.board[i,j] == 0

    def evaluate(self) -> None:
        horizontal = np.ones((1,self.connect))
        vertical = np.ones((self.connect,1))
        diagonal = np.eye(self.connect)
        antidiagonal = np.eye(self.connect)[::-1,:]

        win_horizontal = convolve2d(self.board, horizontal, mode="valid", boundary="fill", fillvalue=0)
        win_vertical = convolve2d(self.board, vertical, mode="valid", boundary="fill", fillvalue=0)
        win_diagonal = convolve2d(self.board, diagonal, mode="valid", boundary="fill", fillvalue=0)
        win_antidiagonal = convolve2d(self.board, antidiagonal, mode="valid", boundary="fill", fillvalue=0)

        won_1 = np.any((win_horizontal >= self.connect, 
                        win_vertical >= self.connect,
                        win_diagonal >= self.connect,
                        win_antidiagonal >= self.connect))
        
        
        lost_1 = np.any((win_horizontal <= -self.connect, 
                        win_vertical <= -self.connect,
                        win_diagonal <= -self.connect,
                        win_antidiagonal <= -self.connect))
        
        if won_1 and lost_1:
            raise ValueError(f"Somehow both players won on board {self.board}.")
        
        if won_1:
            self.terminated = True
            self.endstate = Endstate.WON_1
            return

        if lost_1:
            self.terminated = True
            self.endstate = Endstate.LOST_1
            return

        board_full = ~np.isin(0,self.board)

        if board_full:
            self.terminated = True
            self.endstate = Endstate.DRAW
        return
    
        # print(win_antidiagonal, win_diagonal, win_vertical, win_horizontal)
    
    def reset(self) -> Observation:
        self.board = np.zeros(self.size, self.size)
        self.turn = 0
        self.next_turn = self.first_player
        self.terminated = False
        self.endstate = Endstate.NONE
        return Observation(self.board, self.terminated, self.endstate)

    def random_move(self) -> Observation:
        valid_moves = self.get_valid_moves()
        random_move = self.rng.choice(valid_moves, size=1, axis = 0)
        result = self.play(random_move)
        return result

    def most_central_move(self) -> Observation:
        valid_moves = self.get_valid_moves()
        moves_by_center_distance = np.abs(valid_moves - self.size // 2)
        central_move = np.argmin(moves_by_center_distance)
        result = self.play(valid_moves[central_move])
        return result
    
    def random_central_move(self) -> Observation:
        valid_moves = self.get_valid_moves()
        weights = (valid_moves.shape[0] - np.arange(valid_moves.shape[0])) ** 2
        distribution = weights / weights.sum
        random_central_move = self.rng.choice(valid_moves, size=1, axis = 0, p=distribution)
        result = self.play(random_central_move)
        return result

    def get_observation(self) -> Observation:
        return Observation(self.board, self.terminated, self.endstate)


if __name__ == "__main__":
    game = Game(3, 2)
    print(game.board)
    game.board[0:2,0:3] = np.ones(3)
    # game.board[2:7,4] = np.ones((5)) * -1
    game.get_valid_moves()
    game.random_move()