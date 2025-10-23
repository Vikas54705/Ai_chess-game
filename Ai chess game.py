# chess_ai_game_ml.py (Final Combined Version)
import pygame
import sys
import copy
import time
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load trained ML model and encoder
try:
    clf = load("model/move_predictor.pkl")
    le = load("model/move_encoder.pkl")
except:
    clf, le = None, None

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 640, 640
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smart Chess AI (ML + Minimax)")

WHITE = (245, 245, 220)
BROWN = (139, 69, 19)
BLUE = (0, 0, 255)
GREEN = (34, 139, 34)
FONT = pygame.font.SysFont('Segoe UI Symbol', 48)
clock = pygame.time.Clock()

pieces = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
}

# Piece mapping for ML
piece_map = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
}

def fen_to_board(fen):
    board_part = fen.split()[0]
    board = []
    for char in board_part:
        if char == '/': continue
        elif char.isdigit(): board.extend([0]*int(char))
        else: board.append(piece_map.get(char, 0))
    return board

# Chess logic
# (all your existing helper functions go here: init_board, is_white, get_moves, etc.)
# For brevity, we'll just write placeholders here:
from chess_helpers import *  # Assume all helper functions moved here

# ML prediction
import chess

def get_ml_prediction(board):
    if clf is None or le is None:
        return "(ML model not loaded)"
    fen = board_to_fen(board)
    features = pd.DataFrame([fen_to_board(fen)])
    pred_idx = clf.predict(features)[0]
    move = le.inverse_transform([pred_idx])[0]
    return move

# Evaluate move quality

def evaluate_move_quality(before_board, after_board, is_white_turn):
    before_score = evaluate(before_board)
    after_score = evaluate(after_board)
    if not is_white_turn:
        before_score *= -1
        after_score *= -1
    delta = after_score - before_score
    if delta >= 2:
        return "Brilliant Move!"
    elif delta <= -2:
        return "Blunder!"
    elif delta < 0:
        return "Inaccuracy"
    else:
        return "Good Move"

# Drawing and UI
# (your draw_board logic goes here)

# Game loop
board = init_board()
selected = None
legal_moves = []
running = True
player_turn = True

while running:
    draw_board()
    clock.tick(30)

    if not get_all_valid_moves(board, player_turn):
        print("Checkmate!" if is_in_check(board, player_turn) else "Stalemate!")
        pygame.time.delay(4000)
        running = False
        continue

    if not player_turn:
        start_time = time.time()
        _, move = alphabeta(board, 2, -float('inf'), float('inf'), False)
        print(f"AI move: {move}, took {time.time() - start_time:.2f}s")
        print("ML Prediction:", get_ml_prediction(board))
        if move:
            make_move(board, move[0], move[1])
        player_turn = True
        continue

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            r, c = y // SQUARE_SIZE, x // SQUARE_SIZE
            if selected:
                if (r, c) in legal_moves:
                    before_board = copy.deepcopy(board)
                    make_move(board, selected, (r, c))
                    feedback = evaluate_move_quality(before_board, board, True)
                    print(f"Player Move Feedback: {feedback}")
                    selected = None
                    legal_moves = []
                    player_turn = False
                else:
                    selected = None
                    legal_moves = []
            elif board[r][c] != ' ' and is_white(board[r][c]):
                selected = (r, c)
                moves = get_moves(board, r, c)
                legal_moves = []
                for move in moves:
                    temp = copy.deepcopy(board)
                    make_move(temp, selected, move)
                    if not is_in_check(temp, True):
                        legal_moves.append(move)
