import numpy as np
from enum import Enum
from scipy.signal import convolve2d
from typing import Dict, Sequence
from collections import namedtuple
from sys import stdout

class Endstate(Enum):
    NONE = 0
    WON_1 = 1
    LOST_1 = 2
    DRAW = 3

Observation = namedtuple("Observation", ("board", "terminated", "endstate"))

class Game:
    """
    Gomoku game instance
    """
    def __init__(self, size:int, connect:int, first_player:int = 1, seed = None):

        if connect > size:
            raise ValueError(f"Can't play connect-{connect} on board with size {size}")
        self.size = size
        self.connect = connect
        self.board = np.zeros((size, size), dtype=np.int8)
        self.turn = 0
        self.first_player = first_player
        self.next_player = first_player
        self.terminated = False
        self.endstate = Endstate.NONE
        self.rng = np.random.default_rng(seed)

    def play(self, coords:Sequence[int]) -> Observation:
        """
        Next player plays at the passed coordinates, the game is evaluated and new observation returned.
        """
        i,j = coords

        # Check shouldn't be needed while checks are done in interact fnc for training
        # Can be removed for performance if no errors are seen
        if not self.is_move_valid(coords):
            raise IndexError(f"Tried to play at invalid field ({i}, {j}).")
        
        # Update the symbol on the board, flip next_player's symbol and increment turn counter
        self.board[i,j] = self.next_player
        self.next_player *= -1
        self.turn += 1

        # Evaluate game and return observation
        self.evaluate()
        return Observation(self.board, self.terminated, self.endstate)
    
    def get_valid_moves(self) -> np.ndarray:
        """
        Returns the coordinates for all valid moves on the game board.
        """
        valid_moves = np.stack(np.where(self.board == 0)).T
        return valid_moves
        
    def is_move_valid(self, coords) -> bool:
        """
        Returns whether there is a free playing square at the passed coordinates.
        """
        i,j = coords
        return i >= 0 and j >= 0 and i < self.size and j < self.size and self.board[i,j] == 0

    def evaluate(self) -> None:
        """
        Evaluates whether any player has won, lost or drawn and if the game has terminated,
        and updates the self.endstate and self.terminnated fields.
        """

        # Create check kernels
        horizontal = np.ones((1,self.connect))
        vertical = np.ones((self.connect,1))
        diagonal = np.eye(self.connect)
        antidiagonal = np.eye(self.connect)[::-1,:]

        # Check for winning sequences
        win_horizontal = convolve2d(self.board, horizontal, mode="valid", boundary="fill", fillvalue=0)
        win_vertical = convolve2d(self.board, vertical, mode="valid", boundary="fill", fillvalue=0)
        win_diagonal = convolve2d(self.board, diagonal, mode="valid", boundary="fill", fillvalue=0)
        win_antidiagonal = convolve2d(self.board, antidiagonal, mode="valid", boundary="fill", fillvalue=0)

        won_1 = np.any((np.any(win_horizontal >= self.connect), 
                        np.any(win_vertical >= self.connect),
                        np.any(win_diagonal >= self.connect),
                        np.any(win_antidiagonal >= self.connect)))
        
        
        lost_1 = np.any((np.any(win_horizontal <= -self.connect), 
                        np.any(win_vertical <= -self.connect),
                        np.any(win_diagonal <= -self.connect),
                        np.any(win_antidiagonal <= -self.connect)))
        
        if won_1 and lost_1:
            raise ValueError(f"Somehow both players won on board {self.board}.")
        
        # Update terminated and endstate fields
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
        """
        Reset the game to initial state and return a new observation.
        """
        self.board = np.zeros(self.size, self.size)
        self.turn = 0
        self.next_player = self.first_player
        self.terminated = False
        self.endstate = Endstate.NONE
        return Observation(self.board, self.terminated, self.endstate)

    def random_move(self) -> Observation:
        """
        Play a random valid move for the next player and return a new observation.
        """
        valid_moves = self.get_valid_moves()
        random_move = self.rng.choice(valid_moves, size=1, axis = 0).flatten()
        result = self.play(random_move)
        return result

    def most_central_move(self) -> Observation:
        """
        Play the most central valid move for the next player and return a new observation.
        """
        valid_moves = self.get_valid_moves()
        moves_by_center_distance = np.sum(np.abs(valid_moves - self.size // 2), axis=1)
        central_move_index = np.argmin(moves_by_center_distance)
        central_move = valid_moves[central_move_index, :].flatten()
        result = self.play(central_move)
        return result
    
    def random_central_move(self) -> Observation:
        """
        Play a random valid move with higher probability near the board center for the next player and return a new observation.
        """

        # Get valid moves and order by distance from center
        valid_moves = self.get_valid_moves()
        moves_by_center_distance = np.sum(np.abs(valid_moves - self.size // 2), axis=1)
        ordered_indices = np.argsort(moves_by_center_distance)
        ordered_valid_moves = valid_moves[ordered_indices, :]

        # Get decaying probability distribution
        weights = (valid_moves.shape[0] - np.arange(valid_moves.shape[0])) ** 2
        distribution = weights / weights.sum()

        # Sample a move
        random_central_move = self.rng.choice(ordered_valid_moves, size=1, axis = 0, p=distribution).flatten()
        result = self.play(random_central_move)

        return result

    def get_observation(self) -> Observation:
        """
        Returns an observation of the current game state.
        """
        return Observation(self.board, self.terminated, self.endstate)
    
    def print_board(self, fill = '.'):
        """
        Prints out the game board to console.
        """
        char_board = np.full(self.board.shape, fill)
        char_board[self.board == 1] = 'X'
        char_board[self.board == -1] = 'O'
        string_board = '\n'.join([' '.join(line) for line in char_board.tolist()])
        print(string_board)
        print('\n')

    def clear_printed_board(self):
        for _ in self.size:
            stdout.write("\033[F")
            stdout.write("\033[K")
        stdout.flush()




if __name__ == "__main__":

    # Testing
    game = Game(5, 3)
    # print(game.board)
    # game.board[0:2,0:3] = np.ones(3)
    # game.board[2:7,4] = np.ones((5)) * -1
    for i in range(20):
        obs = game.random_central_move()
        game.print_board()
        if obs.terminated: break