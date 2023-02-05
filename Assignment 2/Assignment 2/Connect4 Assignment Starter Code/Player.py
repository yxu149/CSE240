import random

import numpy as np

# Global Constants
EMPTY = 0
DEPTH = 20
ROW_MAX = 6
COLUMN_MAX = 7


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
        alpha = 0
        beta = 0

        return self.minimax_value(board, DEPTH, alpha, beta, True)[0]

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
        alpha = 0
        beta = 0

        return self.expectimax_value(board, DEPTH, alpha, beta, True)[0]

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
        player = self.get_player_num()
        if player == 1:
            opponent = 2
        else:
            opponent = 1

        return self.eval_frame(board, player, opponent)

    """
    Helper Functions
    """

    def get_player_num(self):
        return self.player_number

    def minimax_value(self, state, depth, alpha, beta, is_max_player):
        player = self.get_player_num()
        if player == 1:
            opponent = 2
        else:
            opponent = 1

        if depth == 0 or self.is_terminal_node(state, player) or self.is_terminal_node(state, opponent):
            if self.is_terminal_node(state, player) or self.is_terminal_node(state, opponent):
                if self.is_winning_move(state, player):
                    return None, 9999
                elif self.is_winning_move(state, opponent):
                    return None, -9999
                else:
                    return None, 0

        if is_max_player:
            val = -np.inf
            valid_cols = self.get_valid_cols(state)
            move = random.choice(valid_cols)
            for col in valid_cols:
                row = self.get_row(state, col, player)
                new_state = state.copy()
                new_state[row][col] = player
                new_val = self.minimax_value(new_state, depth - 1, alpha, beta, False)[1]
                if new_val > val:
                    val = new_val
                    move = col
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return move, val
        else:
            val = np.inf
            valid_cols = self.get_valid_cols(state)
            move = random.choice(valid_cols)
            for col in valid_cols:
                row = self.get_row(state, col, opponent)
                new_state = state.copy()
                new_state[row][col] = opponent
                new_val = self.minimax_value(new_state, depth - 1, alpha, beta, True)[1]
                if new_val < val:
                    val = new_val
                    move = col
                beta = min(beta, val)
                if alpha >= beta:
                    break
            return move, val

    def expectimax_value(self, state, depth, alpha, beta, is_max_player):
        player = self.get_player_num()
        if player == 1:
            opponent = 2
        else:
            opponent = 1

        if depth == 0 or self.is_terminal_node(state, player) or self.is_terminal_node(state, opponent):
            if self.is_terminal_node(state, player) or self.is_terminal_node(state, opponent):
                if self.is_winning_move(state, player):
                    return None, 9999
                elif self.is_winning_move(state, opponent):
                    return None, -9999
                else:
                    return None, 0

        if is_max_player:
            val = -np.inf
            valid_cols = self.get_valid_cols(state)
            move = random.choice(valid_cols)
            for col in valid_cols:
                row = self.get_row(state, col, player)
                new_state = state.copy()
                new_state[row][col] = player
                new_val = self.minimax_value(new_state, depth - 1, alpha, beta, False)[1]
                if new_val > val:
                    val = new_val
                    move = col
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return move, val
        else:
            val = 0
            valid_cols = self.get_valid_cols(state)
            move = random.choice(valid_cols)
            for col in valid_cols:
                row = self.get_row(state, col, opponent)
                new_state = state.copy()
                new_state[row][col] = opponent
                prob = 1/len(valid_cols)

                new_val = self.minimax_value(new_state, depth - 1, alpha, beta, True)[1] * prob
                if new_val < val:
                    val = new_val
                    move = col
                beta = min(beta, val)
                if alpha >= beta:
                    break
            return move, val


    def get_valid_cols(self, board):
        """
        Given board, return columns that are valid to
        drop new pieces into
        Args:
            board: The game board

        Returns:
            cols: Valid columns to drop new pieces into

        """
        cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                cols.append(i)

        return cols

    def is_winning_move(self, board, player):
        """
        Local estimator of whether this move is the
        one that will win player the game or not.
        Intentional written differently from the
        game_complete function in ConnectFour.py
        to mimic human thinking behaviour
        Args:
            board: The game board
            player: The player playing when function
            is called

        Returns:
            a boolean value of whether player given
            will be making a winning move or not

        """
        # Check for horizontal wins
        for c in range(COLUMN_MAX - 3):
            for r in range(ROW_MAX):
                if board[r][c] == player \
                        and board[r][c + 1] == player \
                        and board[r][c + 2] == player \
                        and board[r][c + 3] == player:
                    return True
        # Check for vertical wins
        for c in range(COLUMN_MAX):
            for r in range(ROW_MAX - 3):
                if board[r][c] == player \
                        and board[r + 1][c] == player \
                        and board[r + 2][c] == player \
                        and board[r + 3][c] == player:
                    return True
        # Check for ↗ wins
        for c in range(COLUMN_MAX - 3):
            for r in range(ROW_MAX - 3):
                if board[r][c] == player \
                        and board[r + 1][c + 1] == player \
                        and board[r + 2][c + 2] == player \
                        and board[r + 3][c + 3] == player:
                    return True
        # Check for ↘ wins
        for c in range(COLUMN_MAX - 3):
            for r in range(ROW_MAX - 3):
                if board[r][c] == player \
                        and board[r - 1][c + 1] == player \
                        and board[r - 2][c + 2] == player \
                        and board[r - 3][c + 3] == player:
                    return True

    def is_terminal_node(self, board, player):
        """
        Is the last node of branch. No more successors.
        Args:
            board: The game board
            player: The player playing when function
            is called

        Returns:
            a boolean value of whether this node
            has further successors
        """
        return self.is_winning_move(board, player) or len(self.get_valid_cols(board)) == 0

    def get_row(self, board, col, player):
        """
        Simulate a piece dropping into the board
        Args:
            board: The game board
            col: Column to drop piece into
            player: The player dropping the piece

        Returns:
            The row number this dropped piece will end up
        """
        for row in range(0, 6, 1): 
            if board[row][col] == 0: 
                return row

        return None

    def eval_frame(self, state, player, opponent):
        score = 0
        if self.is_winning_move(state, player):
            score += 10000
        if state.count(player) == 4:
            score += 3000
        elif state.count(player) == 3 and state.count(0) == 1:
            score += 1500
        elif state.count(player) == 2 and state.count(0) == 2:
            score += 1000
        if state.count(opponent) == 3 and state.count(0) == 1:
            score -= 2000
        return score


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
