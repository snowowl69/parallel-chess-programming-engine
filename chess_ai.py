import chess
import random
import torch
import time
import logging
import concurrent.futures

try:
    from mcts import MCTS
    from neural_network import load_model
    mcts_engine = None
except ImportError:
    mcts_engine = None

logger = logging.getLogger(__name__)

# Basic material evaluation values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Position bonus tables
PAWN_BONUS = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_BONUS = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

def evaluate_board(board):
    """
    Evaluates the board using material and positional bonuses.
    Returns a score from White's perspective.
    """
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -99999
        else:
            return 99999
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
        return 0

    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
            
        value = PIECE_VALUES.get(piece.piece_type, 0)
        
        # Positional bonuses
        if piece.piece_type == chess.PAWN:
            pos_bonus = PAWN_BONUS[sq ^ 56] if piece.color == chess.WHITE else PAWN_BONUS[sq]
            value += pos_bonus
        elif piece.piece_type == chess.KNIGHT:
            pos_bonus = KNIGHT_BONUS[sq ^ 56] if piece.color == chess.WHITE else KNIGHT_BONUS[sq]
            value += pos_bonus
            
        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value
            
    return score

def order_moves(board, moves):
    """Simple move ordering for alpha-beta pruning."""
    def move_score(move):
        score = 0
        if board.is_capture(move):
            score += 10
            # Capturing with a weaker piece is better
            attacker = board.piece_at(move.from_square)
            victim = board.piece_at(move.to_square)
            if attacker and victim:
                score += PIECE_VALUES.get(victim.piece_type, 0) - PIECE_VALUES.get(attacker.piece_type, 0)
        if move.promotion:
            score += 50
        return score

    return sorted(moves, key=move_score, reverse=True)

def minimax(board, depth, alpha, beta, is_maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    moves = list(board.legal_moves)
    moves = order_moves(board, moves)

    if is_maximizing:
        max_eval = float('-inf')
        for move in moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def get_minimax_move(board, depth):
    moves = list(board.legal_moves)
    if not moves:
        return None

    moves = order_moves(board, moves)
    best_move = None
    is_maximizing = board.turn == chess.WHITE
    best_eval = float('-inf') if is_maximizing else float('inf')

    def eval_move(move):
        b = board.copy()
        b.push(move)
        eval_score = minimax(b, depth - 1, float('-inf'), float('inf'), not is_maximizing)
        return move, eval_score

    # Evaluate top-level branches in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(eval_move, moves)

    for move, eval_score in results:
        if is_maximizing and eval_score > best_eval:
            best_eval = eval_score
            best_move = move
        elif not is_maximizing and eval_score < best_eval:
            best_eval = eval_score
            best_move = move

    # Fallback to random if something fails
    if best_move is None and moves:
        best_move = random.choice(moves)
        
    return best_move

def initialize_mcts():
    """Initializes the MCTS engine with the latest model."""
    global mcts_engine
    if mcts_engine is not None:
        return True
        
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Find latest checkpoint
        import os
        import glob
        checkpoint_files = glob.glob("models/checkpoints/chess_model_epoch_*.pth")
        if not checkpoint_files:
            return False
            
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        model = load_model(latest_checkpoint, device)
        model.eval()
        mcts_engine = MCTS(model, device=device)
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MCTS: {e}")
        return False

def get_best_move(board, difficulty=2):
    """
    Returns the best move based on difficulty setting.
    1: Fast (Minimax Depth 2)
    2: Medium (Minimax Depth 3)
    3: Hard (MCTS or Minimax Depth 4 if MCTS unavailable)
    """
    start_time = time.time()
    move = None
    
    if difficulty == 3 and initialize_mcts():
        # Use MCTS for hard difficulty
        logger.info("Using MCTS for move calculation")
        mcts_engine.num_simulations = 400
        move = mcts_engine.get_best_move(board, temperature=0.0)
    
    if move is None:
        # Fallback to Minimax
        depth = 2 if difficulty == 1 else 3 if difficulty == 2 else 4
        logger.info(f"Using Minimax depth {depth} for move calculation")
        move = get_minimax_move(board, depth=depth)
        
    elapsed = time.time() - start_time
    logger.info(f"AI move calculation took {elapsed:.2f} seconds")
    return move
