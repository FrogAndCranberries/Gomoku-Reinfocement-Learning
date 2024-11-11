import numpy as np
from enum import Enum
from scipy.signal import convolve2d
from collections import namedtuple
from sys import stdout

class Endstate(Enum):
    NONE = 0
    WON_X = 1
    WON_O = 2
    DRAW = 3

Observation = namedtuple("Observation", ("board", "terminated", "endstate"))

class Game:
    """
    A connenct-N game instance.
    """

    def __init__(self, board_size: int, connect: int, first_player: int = 1) -> None:
        """
        Creates a game instance of connect-N on board with side length board_size where the first player is X for first_player = 1 and 
        O for first_player = -1.
        """
        if first_player != 1 and first_player != -1:
            raise ValueError(f"First player's symbol must be 1 for X or -1 for O.")
        if connect > board_size:
            raise ValueError(f"Can't play connect-{connect} on board with size {board_size}.")
        
        self.board_size: int = board_size
        self.max_index: int = board_size ** 2 - 1
        self.connect: int = connect
        self.board: np.ndarray = np.zeros((board_size, board_size), dtype=np.int8)
        self.turn: int = 0
        self.first_player: int = first_player
        self.next_player: int = first_player
        self.terminated: bool = False
        self.endstate: Endstate = Endstate.NONE
        self.win_check_kernels: dict = self._initialise_kernels()
        self.valid_moves: list = list(range(board_size ** 2))

    def reset(self) -> None:
        """
        Reset the game to initial state and return a new observation.
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.turn = 0
        self.next_player = self.first_player
        self.terminated = False
        self.endstate = Endstate.NONE
        self.valid_moves = list(range(self.board_size ** 2))

    def make_move(self, index: int) -> None:
        """
        Next player plays at the passed flat index and the game result is evaluated.
        """
        i,j = np.unravel_index(index, self.board.shape)

        assert self.is_move_valid(index)
        
        self.board[i,j] = self.next_player
        self.valid_moves.remove(index)
        self._advance_turn()
        self._evaluate()

    @property
    def obs(self) -> Observation:
        """
        Returns an observation of the current game state.
        """
        board_3_channel = np.stack((self.board == 1, self.board == -1, self.board == 0))
        board_3_channel = board_3_channel.astype(np.int8)
        
        return Observation(board_3_channel, self.terminated, self.endstate)
    
    def get_valid_moves(self) -> np.ndarray:
        """
        Returns an array of flat indices for all empty fields on the game board.
        """
        if self.terminated:
            return np.empty(1)
        return np.array(self.valid_moves)
    
    def is_move_valid(self, index: int) -> bool:
        """
        Returns whether there is a free playing square at the passed flat index.
        """
        return index in self.valid_moves
    
    def _advance_turn(self) -> None:
        """
        Flips the next player symbol between 1 and -1 and increments turn.
        """
        self.next_player *= -1
        self.turn += 1
        
    def _evaluate(self) -> None:
        """
        Updates self.endstate and self.terminated fields.
        """
        won_X, won_O = self._find_winning_sequences()
        
        assert not (won_X and won_O)
        
        # Update terminated and endstate fields
        if won_X:
            self.terminated = True
            self.endstate = Endstate.WON_X
            return

        if won_O:
            self.terminated = True
            self.endstate = Endstate.WON_O
            return

        if self._is_draw(won_X, won_O):
            self.terminated = True
            self.endstate = Endstate.DRAW
            return
        
    def _is_draw(self, won_X: bool, won_O: bool) -> None:
        board_is_full = ~np.isin(0,self.board)
        return not won_X and not won_O and board_is_full
    
    def _find_winning_sequences(self) -> tuple:
        """
        Returns two bools whether there are any X and any O winning sequences
        """
        wins_X = []
        wins_O = []

        for kernel in self.win_check_kernels.values():
            symbol_sequence_in_kernel_dir = convolve2d(self.board, kernel, mode="valid")
            wins_X.append(np.any(symbol_sequence_in_kernel_dir >= self.connect))
            wins_O.append(np.any(symbol_sequence_in_kernel_dir <= -self.connect))

        won_X = np.any(np.asarray(wins_X))
        won_O = np.any(np.asarray(wins_O))

        return won_X, won_O
    
    def _initialise_kernels(self) -> np.recarray:

        kernels = {}
        kernels["horizontal"] = np.ones((1,self.connect), dtype=np.int8)
        kernels["vertical"] = np.ones((self.connect,1), dtype=np.int8)
        kernels["diagonal"] = np.eye(self.connect, dtype=np.int8)
        kernels["antidiagonal"] = np.fliplr(np.eye(self.connect, dtype=np.int8))

        return kernels
    
class GameUtils:

    @staticmethod
    def print_board(game: Game, fill: str = '.') -> None:
        """
        Prints out the game board to console.
        """
        board_as_chars = np.full(game.board.shape, fill)
        board_as_chars[game.board == 1] = 'X'
        board_as_chars[game.board == -1] = 'O'
        board_as_string = '\n'.join([' '.join(line) for line in board_as_chars.tolist()])
        print(board_as_string)

    @staticmethod
    def clear_printed_board(game:Game = None):
        """
        Deletes a board printed by print_board.
        """
        for _ in range(game.board_size):
            stdout.write("\033[F")
            stdout.write("\033[K")
        stdout.flush()


if __name__ == "__main__":

    # Testing
    game = Game(10, 3)
    # print(game.board)
    game.board[0:2,0:3] = np.ones(3)
    game.board[2:7,4] = np.ones((5)) * -1
    game.get_observation()

    # time.sleep(1)
    # print("test")
    # for i in range(20):
    #     obs = game.random_central_move()
    #     game.print_board()
        
    #     time.sleep(0.3)
    #     game.clear_printed_board()
    #     if obs.terminated: break