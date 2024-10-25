import numpy as np
from enum import Enum
from scipy.signal import convolve2d

# Gomoku game instance with taking turns and evaluating win conditions

class Endstate(Enum):
    NONE = 0
    WON_1 = 1
    LOST_1 = 2
    DRAW = 3


class Game:
    def __init__(self, size:int, connect:int):
        if connect > size:
            raise ValueError(f"Can't play connect-{connect} on board with size {size}")
        self.size = size
        self.connect = connect
        self.board = np.zeros((size, size))
        self.turn = 0
        self.next_turn = 1
        self.terminated = False
        self.endstate = Endstate.NONE

    def play(self, x:int, y:int):
        if not self.is_move_valid(x,y):
            raise IndexError(f"Tried to play at invalid field ({x}, {y}).")
        self.board[x,y] = self.next_turn
        self.next_turn *= -1
        self.turn += 1
        self.evaluate()
        return {"board": self.board, "terminated": self.terminated, "endstate": self.endstate}
    
    def get_valid_moves(self) -> np.ndarray:
        valid_moves = np.stack(np.where(self.board == 0)).T
        return(valid_moves)
        
    def is_move_valid(self, x:int, y:int) -> bool:
        return x >= 0 and y >= 0 and x < self.size and y < self.size and self.board[x,y] == 0

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
    

if __name__ == "__main__":
    game = Game(3, 2)
    print(game.board)
    # game.board[1,1:3] = np.ones(2)
    # game.board[2:7,4] = np.ones((5)) * -1
    game.get_valid_moves()