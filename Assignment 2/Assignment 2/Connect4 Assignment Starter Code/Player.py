import random

import numpy as np

# Global Constants
MAX_DEPTH = 30


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        alpha = -np.inf
        beta = np.inf
        depth = 0

        player = self.player_number
        opponent = self.get_opponent(player)

        frame = board.copy()
        moves = self.get_valid_cols(frame)
        select_moves = []
        for col in moves:
            row = self.get_valid_row(frame, col)
            frame[row][col] = player
            depth += 1
            alpha = max(alpha, self.alphabeta_min_value(frame, alpha, beta, player, opponent, depth))
            select_moves.append([alpha, col])
            frame[row][col] = 0

        best = max(select_moves, key=lambda x: x[0])
        """print("Alpha", alpha, "Beta", beta)
        print("AB Select Moves", select_moves)
        print("AB Best", best)"""
        return best[1]

    def alphabeta_max_value(self, board, alpha, beta, player, opponent, depth):
        """
        Alpha Beta Search Max-Value function
        Args:
            board: A copy of the current game board
            alpha: The best path to root for MAX
            beta: The best path to root for MIN
            player: Current player playing, i.e. MAX player
            opponent: The other player, i.e. MIN player
            depth: Recursion depth control variable

        Returns:
            An estimated maximum score given the current freeze-frame of the game board

        """
        max_val = -np.inf
        frame = board.copy()
        valid_moves = self.get_valid_cols(frame)

        if depth >= MAX_DEPTH or len(valid_moves) == 0:
            return self.evaluation_function(frame)

        for col in valid_moves:
            row = self.get_valid_row(frame, col)
            frame[row][col] = player
            depth += 1

            min_val = self.alphabeta_min_value(frame, alpha, beta, player, opponent, depth)
            max_val = max(max_val, min_val)

            if max_val >= beta:
                return max_val

            alpha = max(alpha, max_val)
            frame[row][col] = 0

        return max_val

    def alphabeta_min_value(self, board, alpha, beta, player, opponent, depth):
        """
        Alpha Beta Search Min-Value function
        Args:
            board: A copy of the current game board
            alpha: The best path to root for MAX
            beta: The best path to root for MIN
            player: Current player playing, i.e. MAX player
            opponent: The other player, i.e. MIN player
            depth: Recursion depth control variable

        Returns:
            An estimated minimum score given the current freeze-frame of the game board

        """
        min_val = np.inf
        frame = board.copy()
        valid_moves = self.get_valid_cols(frame)

        if depth >= MAX_DEPTH or len(valid_moves) == 0:
            return self.evaluation_function(frame)

        for col in valid_moves:
            row = self.get_valid_row(frame, col)
            frame[row][col] = opponent
            depth += 1

            max_val = self.alphabeta_max_value(frame, alpha, beta, player, opponent, depth)
            min_val = min(min_val, max_val)

            if min_val <= alpha:
                return min_val

            beta = min(min_val, beta)
            frame[row][col] = 0

        return min_val

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        alpha = -np.inf
        player = self.player_number
        opponent = self.get_opponent(player)
        depth = 0

        frame = board.copy()
        select_moves = self.exp_value(frame, alpha, player, opponent, depth)
        best = max(select_moves, key=lambda x: x[0])
        """print("Alpha", alpha)
        print("Expectimax Select Moves", select_moves)
        print("Expectimax Best", best)"""
        return best[1]

    def exp_value(self, board, alpha, player, opponent, depth):
        """
        Expectimax value function. Drives the max and min functions.
        Args:
            board: The current game board
            alpha: MAX path to root
            player: MAX player
            opponent: MIN player
            depth: Recursion depth control variable

        Returns:
            Estimated best column to drop piece into
        """
        select_moves = []
        frame = board.copy()
        moves = self.get_valid_cols(frame)
        for col in moves:
            row = self.get_valid_row(frame, col)
            frame[row][col] = player
            depth += 1
            alpha = max(alpha, self.expmax_value(frame, player, opponent, depth))
            select_moves.append([alpha, col])
            frame[row][col] = 0

        return select_moves

    def expmax_value(self, board, player, opponent, depth):
        """
        Expectimax maximum value function. Similar to alphabeta_max_value.
        Args:
            board: A freeze-frame of the current game board
            player: MAX player
            opponent: MIN player
            depth: Recursion depth control variable

        Returns:
            An estimated value of this path to root
        """
        max_val = -np.inf
        frame = board.copy()
        moves = self.get_valid_cols(frame)
        if depth >= MAX_DEPTH or len(moves) == 0:
            return self.evaluation_function(frame)

        for col in moves:
            row = self.get_valid_row(frame, col)
            frame[row][col] = player
            depth += 1
            min_val = self.expmin_value(frame, player, opponent, depth)
            max_val = max(max_val, min_val)
            frame[row][col] = 0

        return max_val

    def expmin_value(self, board, player, opponent, depth):
        """
        Expectimax minimum value function
        Args:
            board: A freeze-frame of the current game board
            player: Current player playing, i.e. MAX player
            opponent: The evil player, i.e. MIN player
            depth: Recursion depth control variable

        Returns:
            An estimated value of this path to root
        """
        min_val = 0
        frame = board.copy()
        moves = self.get_valid_cols(frame)
        if depth >= MAX_DEPTH or len(moves) == 0:
            return self.evaluation_function(frame)

        prob = 1 / len(moves)
        for col in moves:
            row = self.get_valid_row(frame, col)
            frame[row][col] = opponent
            depth += 1
            max_val = self.expmax_value(frame, player, opponent, depth)
            min_val += max_val * prob
            frame[row][col] = 0

        return min_val

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that
        represents the evaluation function for the current player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        player = self.player_number
        opponent = self.get_opponent(player)
        score = 0

        """
        Following code has been inspired by 
        https://github.com/KeithGalli/Connect4-Python
        and associated YouTube video 
        https://www.youtube.com/watch?v=MMLtza3CZFM
        """
        # Check for horizontal wins
        for row in range(0, 6):
            row_arr = [int(i) for i in list(board[row,:])]
            for col in range(0, 7-3):
                frame = row_arr[col:col+4]
                score += self.eval_frame(frame, player, opponent)
        # Check for vertical wins
        for col in range(0, 7):
            col_arr = [int(i) for i in list(board[:, col])]
            for row in range(0, 6-3):
                frame = col_arr[row:row+4]
                score += self.eval_frame(frame, player, opponent)
        # Check for ↗ wins
        for row in range(0, 6-3):
            for col in range(0, 7-3):
                frame = [board[row+i][col+i] for i in range(4)]
                score += self.eval_frame(frame, player, opponent)
        # Check for ↘ wins
        for row in range(0, 6-3):
            for col in range(0, 7-3):
                frame = [board[row+3-i][col+1] for i in range(4)]
                score += self.eval_frame(frame, player, opponent)

        #print("Score", score)
        return score

    def eval_frame(self, frame, player, opponent):
        """
        Count up the score on this freeze-frame of the game board for
        player given.
        Args:
            frame: Freeze-frame of game board
            player: MAX player
            opponent: MIN player

        Returns:
            An estimated score

        """
        score = 0

        if frame.count(player) == 4:
            score += 100
        elif frame.count(player) == 3 and frame.count(0) == 1:
            score += 69
        elif frame.count(player) == 2 and frame.count(0) == 2:
            score += 30
        elif frame.count(player) == 1 and frame.count(0) == 3:
            score += 5

        if frame.count(opponent) == 4:
            score -= 95
        elif frame.count(opponent) == 3 and frame.count(0) == 1:
            score -= 80
        elif frame.count(opponent) == 2 and frame.count(0) == 2:
            score -= 30
        elif frame.count(opponent) == 1 and frame.count(0) == 3:
            score -= 5

        return score

    """
    Helper Functions
    """

    def get_opponent(self, player):
        """
        Gives you the player number of your opponent
        Args:
            player: current player number

        Returns:
            the other player number available, i.e. opponent's player number
        """
        if player == 1:
            return 2
        else:
            return 1

    def get_valid_cols(self, board):
        """
        Given board, return columns that are valid to
        drop new pieces into
        Args:
            board: The game board

        Returns:
            cols: Valid columns to drop new pieces into

        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        return valid_cols

    def get_valid_row(self, board, col):
        """
        Simulates the piece dropping down into the board
        Args:
            board: A freeze-frame of the game board
            col: The column to drop game piece into

        Returns:
            The row which the piece will end up on
        """
        for row in range(0, 6, 1):
            if board[row][col] == 0:
                return row


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move
