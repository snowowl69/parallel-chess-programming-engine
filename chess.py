import tkinter as tk
from tkinter import simpledialog
import chess as chess_game
import random
import torch
import threading
import time
import logging
from datetime import datetime
import pygame
import platform


log_filename = f"chess_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')


def get_cpu_info():
    try:
        return platform.processor() or "CPU"
    except:
        return "CPU"

def get_gpu_info():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No CUDA-compatible GPU available"

CPU_NAME = get_cpu_info()
GPU_NAME = get_gpu_info()
CUDA_VERSION = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "N/A"


print("Initializing evaluation engine...")


def evaluate_board_global(board, device):
    piece_values = torch.tensor([1, 3, 3, 5, 9], device=device, dtype=torch.float32)
    eval_score = torch.tensor(0, device=device, dtype=torch.float32)
    for piece, value in zip([chess_game.PAWN, chess_game.KNIGHT, chess_game.BISHOP, chess_game.ROOK, chess_game.QUEEN], piece_values):
        eval_score += len(board.pieces(piece, chess_game.WHITE)) * value
        eval_score -= len(board.pieces(piece, chess_game.BLACK)) * value
    return eval_score.item()

start_cpu = time.time()
dummy_board = chess_game.Board()
cpu_score = evaluate_board_global(dummy_board, torch.device("cpu"))
cpu_time = time.time() - start_cpu

device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_gpu = time.time()
gpu_score = evaluate_board_global(dummy_board, device_gpu)
gpu_time = time.time() - start_gpu

print("="*60)
print("🧠 Evaluation Engine Benchmark:")
print("="*60)
print(f"CPU Evaluation:")
print(f"   • Device          : {CPU_NAME} (CPU)")
print(f"   • Precision       : {torch.float32}")
print(f"   • Eval Score      : {'+' if cpu_score >= 0 else ''}{cpu_score:.2f}")
print(f"   • Time Taken      : {cpu_time:.6f} seconds")
print(f"   • Eval Logic      : Material balance (P=1, N/B=3, R=5, Q=9)")
print("-"*60)
print(f"GPU Evaluation:")
if torch.cuda.is_available():
    print(f"   • Device          : {GPU_NAME} ({CUDA_VERSION})")
    print(f"   • Precision       : {torch.float32}")
    print(f"   • Eval Score      : {'+' if gpu_score >= 0 else ''}{gpu_score:.2f}")
    print(f"   • Time Taken      : {gpu_time:.6f} seconds")
    print(f"   • Memory Used     : {torch.cuda.memory_allocated(0)/1e6:.2f} MB")
else:
    print(f"   • Device          : No CUDA-compatible GPU available.")
    print(f"   • Precision       : {torch.float32}")
    print(f"   • Eval Score      : {'+' if gpu_score >= 0 else ''}{gpu_score:.2f} (CPU fallback)")
    print(f"   • Time Taken      : {gpu_time:.6f} seconds")
print(f"   • Eval Logic      : Material balance (P=1, N/B=3, R=5, Q=9)")
print("="*60)


device_cpu = torch.device("cpu")


pygame.init()
try:
    pygame.mixer.init()
    move_sound = pygame.mixer.Sound("move.wav")
    move_sound.set_volume(0.6)
    print("[INFO] Move sound loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not initialize sound or load move.wav: {e}")
    move_sound = None

try:
    capture_sound = pygame.mixer.Sound("capture.wav")
    capture_sound.set_volume(0.7)
    print("[INFO] Capture sound loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load capture sound.wav: {e}")
    capture_sound = None

UNICODE_PIECES = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

class ChessGUI:
    def __init__(self, master, mode="engine", player_color=chess_game.WHITE):
        self.master = master
        self.master.title("Chess Game")
        self.board = chess_game.Board()
        self.mode = mode
        self.player_color = player_color
        self.selected_square = None
        self.drag_start_square = None
        self.move_history = []
        self.move_display_history = []
        self.canvas = tk.Canvas(master, width=640, height=640)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_drop)


        self.status = tk.Label(master, text="", font=("Arial", 12))
        self.status.pack()

        self.history_label = tk.Label(master, text="", font=("Courier", 12))
        self.history_label.pack()

        self.square_size = 80
        self.draw_board()
        self.update_board()

        self.time_left = [0, 0]
        self.clock_labels = [tk.Label(master, font=("Arial", 12)) for _ in range(2)]
        self.clock_labels[0].pack()
        self.clock_labels[1].pack()
        self.timer_running = True

        if self.mode in ["manual", "vs_ai"]:
            self.set_timer_dialog()
            self.update_clock()

        if self.mode == "vs_ai" and self.board.turn != self.player_color:
            self.master.after(1000, self.ai_move)
        elif self.mode == "engine":
            self.master.after(1000, self.play_engine_game)

    def format_move(self, move):
        piece = self.board.piece_at(move.from_square)
        if piece is None:
            return move.uci()
        
        
        moving_piece_symbol = UNICODE_PIECES[self.board.piece_at(move.from_square).symbol()]
        to_sq_name = chess_game.square_name(move.to_square)
        
        promotion_text = ""
        if move.promotion:
          
            promoted_piece_char = chess_game.piece_symbol(move.promotion)
            if self.board.turn == chess_game.BLACK:
                promoted_piece_char = promoted_piece_char.lower()
            promotion_symbol = UNICODE_PIECES[promoted_piece_char]
            promotion_text = f" → {promotion_symbol}"
            
        return f"{moving_piece_symbol} to {to_sq_name}{promotion_text}"


    def set_timer_dialog(self):
        time_choice = simpledialog.askstring("Choose Timer", "Select time: 10 / 5 / 3 minutes")
        if time_choice not in ["10", "5", "3"]:
            print("Invalid time selected. Defaulting to 5 minutes.")
            time_choice = "5"
        total_time = int(time_choice) * 60
        self.time_left = [total_time, total_time]
        self.update_clock()

    def update_clock(self):
        if self.mode not in ["manual", "vs_ai"] or not self.timer_running:
            return

        if self.mode == "vs_ai":
            current_player_idx = 0 if self.board.turn == self.player_color else 1
        else:
            current_player_idx = 0 if self.board.turn == chess_game.WHITE else 1

        self.time_left[current_player_idx] -= 1
        
        if self.time_left[current_player_idx] <= 0:
            self.timer_running = False
            if self.mode == "vs_ai":
                winner = "AI" if self.board.turn == self.player_color else "Player"
                result = f"⏰ Time's up! {winner} wins!"
            else:
                winner = "White" if current_player_idx == 1 else "Black"
                result = f"⏰ Time's up! {winner} wins!"
            
            logging.info(result)
            self.status.config(text=result)
            return

        for i in range(2):
            mins, secs = divmod(self.time_left[i], 60)
            label = f"{'White' if i == 0 else 'Black'}: {mins:02}:{secs:02}"
            self.clock_labels[i].config(text=label)

        self.master.after(1000, self.update_clock)

    def draw_board(self):
        colors = ["#f0d9b5", "#b58863"]
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size
                y1 = rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                color = colors[(rank + file) % 2]
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tags="square")

    def update_board(self):
        self.canvas.delete("piece")
        for square in chess_game.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                file = chess_game.square_file(square)
                rank = chess_game.square_rank(square)
            
                if self.player_color == chess_game.WHITE:
                    display_rank = 7 - rank
                    display_file = file
                else:
                    display_rank = rank
                    display_file = 7 - file
                
                x = display_file * self.square_size + 40
                y = display_rank * self.square_size + 40
                symbol = UNICODE_PIECES[piece.symbol()]
                self.canvas.create_text(x, y, text=symbol, font=("Arial", 32), tags="piece")
        

        display_history = " | ".join(self.move_display_history[-8:])
        self.history_label.config(text=display_history)

    def get_square_under_mouse(self, event):
        col = event.x // self.square_size
        row = event.y // self.square_size
        if self.player_color == chess_game.WHITE:
            row = 7 - row
        else:
            col = 7 - col
        return chess_game.square(col, row)

    def on_click(self, event):
        if self.mode == "engine" or (self.mode == "vs_ai" and self.board.turn != self.player_color):
            return
        self.drag_start_square = self.get_square_under_mouse(event)

    def on_drag(self, event):
        pass

    def on_drop(self, event):
        if self.mode == "engine" or (self.mode == "vs_ai" and self.board.turn != self.player_color):
            return

        from_sq = self.drag_start_square
        to_sq = self.get_square_under_mouse(event)

        if from_sq is None or to_sq is None:
            return


        move_to_make = None
        piece_on_from_sq = self.board.piece_at(from_sq)

        if piece_on_from_sq and piece_on_from_sq.piece_type == chess_game.PAWN:
            if (self.board.turn == chess_game.WHITE and chess_game.square_rank(to_sq) == 7) or \
               (self.board.turn == chess_game.BLACK and chess_game.square_rank(to_sq) == 0):
                
                promo_move = chess_game.Move(from_sq, to_sq, promotion=chess_game.QUEEN)
                if promo_move in self.board.legal_moves:
                    move_to_make = promo_move
                else:
                    pass
        
        if move_to_make is None:
            move_to_make = chess_game.Move(from_sq, to_sq)

        if move_to_make in self.board.legal_moves:

            is_capture = self.board.piece_at(move_to_make.to_square) is not None or \
                         self.board.is_en_passant(move_to_make)
            

            formatted_move = self.format_move(move_to_make)
            

            self.board.push(move_to_make)
            move_uci = move_to_make.uci()
            self.move_history.append(move_uci)
            self.move_display_history.append(formatted_move)
            

            if is_capture and capture_sound:
                pygame.mixer.Sound.play(capture_sound)
            elif move_sound:
                pygame.mixer.Sound.play(move_sound)
                
            self.update_board()
            self.update_clock()
            logging.info(f"Player plays: {formatted_move} ({move_uci})")
            self.evaluate_parallel()
            
            if self.mode == "vs_ai" and not self.board.is_game_over():
                self.master.after(500, self.ai_move)
            

            turn_text = "White" if self.board.turn == chess_game.WHITE else "Black"
            self.status.config(text=f"Last move: {formatted_move} | {turn_text} to play")

        self.drag_start_square = None

    def evaluate_board(self, board_to_eval, device):
        piece_values = torch.tensor([1, 3, 3, 5, 9], device=device, dtype=torch.float32)
        eval_score = torch.tensor(0, device=device, dtype=torch.float32)
        for piece_type, value in zip([chess_game.PAWN, chess_game.KNIGHT, chess_game.BISHOP, chess_game.ROOK, chess_game.QUEEN], piece_values):
            eval_score += len(board_to_eval.pieces(piece_type, chess_game.WHITE)) * value
            eval_score -= len(board_to_eval.pieces(piece_type, chess_game.BLACK)) * value
        return eval_score.item()

    def evaluate_parallel(self):
        if self.board.is_game_over():
            return
            
        result = {}

        def run_eval(name, device):
            start = time.time()
            score = self.evaluate_board(self.board, device)
            duration = time.time() - start
            result[name] = (score, duration)

        t1 = threading.Thread(target=run_eval, args=("CPU", device_cpu))
        t2 = threading.Thread(target=run_eval, args=("GPU", device_gpu))
        t1.start()
        t2.start()
        t1.join()
        t2.join()


        benchmark_text = (
            f"🧠 Evaluation Engine Benchmark:\n"
            f"──────────────────────────────────────────────────────\n"
            f"CPU Evaluation:\n"
            f"   • Device          : {CPU_NAME} (CPU)\n"
            f"   • Precision       : torch.float32\n"
            f"   • Eval Score      : {'+' if result['CPU'][0] >= 0 else ''}{result['CPU'][0]:.2f}\n"
            f"   • Time Taken      : {result['CPU'][1]:.6f} seconds\n"
            f"   • Eval Logic      : Material balance (P=1, N/B=3, R=5, Q=9)\n"
            f"GPU Evaluation:\n"
        )
        
        if torch.cuda.is_available():
            benchmark_text += (
                f"   • Device          : {GPU_NAME} ({CUDA_VERSION})\n"
                f"   • Precision       : torch.float32\n"
                f"   • Eval Score      : {'+' if result['GPU'][0] >= 0 else ''}{result['GPU'][0]:.2f}\n"
                f"   • Time Taken      : {result['GPU'][1]:.6f} seconds\n"
                f"   • Memory Used     : {torch.cuda.memory_allocated(0)/1e6:.2f} MB\n"
            )
        else:
            benchmark_text += (
                f"   • Device          : No CUDA-compatible GPU available\n"
                f"   • Precision       : torch.float32\n"
                f"   • Eval Score      : {'+' if result['GPU'][0] >= 0 else ''}{result['GPU'][0]:.2f} (CPU fallback)\n"
                f"   • Time Taken      : {result['GPU'][1]:.6f} seconds\n"
            )
        
        benchmark_text += "──────────────────────────────────────────────────────"
        

        logging.info(benchmark_text)
        print(benchmark_text)
        
        if len(self.move_display_history) > 0:
            turn_text = "White" if self.board.turn == chess_game.WHITE else "Black"
            last_move = self.move_display_history[-1]
            self.status.config(text=f"Last move: {last_move} | {turn_text} to play")

    def get_best_move(self, depth):
        def minimax(board, depth, is_max_player):
            if depth == 0 or board.is_game_over():
                return self.evaluate_board(board, device_cpu), None

            best_value = float('-inf') if is_max_player else float('inf')
            best_move = None

            legal_moves = list(board.legal_moves)
            random.shuffle(legal_moves)

            for move in legal_moves:
                board.push(move)
                val, _ = minimax(board, depth - 1, not is_max_player)
                board.pop()

                if is_max_player:
                    if val > best_value:
                        best_value = val
                        best_move = move
                else:
                    if val < best_value:
                        best_value = val
                        best_move = move
            
            return best_value, best_move

        _, best_move = minimax(self.board.copy(), depth, self.board.turn == chess_game.WHITE)
        return best_move

    def play_engine_game(self):
        if self.board.is_game_over():
            result = self.board.result()
            self.status.config(text=f"Game Over! Result: {result}")
            logging.info(f"Game Over! Result: {result}")
            return

        self.evaluate_parallel()
        

        current_player_color_name = "White" if self.board.turn == chess_game.WHITE else "Black"
        self.status.config(text=f"Thinking... {current_player_color_name}'s turn")

        move = self.get_best_move(depth=2)
        if not move:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
            else:
                self.status.config(text="No legal moves. Game might be over.")
                return
        

        formatted_move = self.format_move(move)
        
        is_capture = self.board.piece_at(move.to_square) is not None or self.board.is_en_passant(move)
        
        self.board.push(move)
        self.move_history.append(move.uci())
        self.move_display_history.append(formatted_move)
        
        logging.info(f"{current_player_color_name} plays: {formatted_move} ({move.uci()})")
        
        if is_capture and capture_sound:
            pygame.mixer.Sound.play(capture_sound)
        elif move_sound:
            pygame.mixer.Sound.play(move_sound)
            
        self.update_board()
        
        next_turn_text = "White" if self.board.turn == chess_game.WHITE else "Black"
        self.status.config(text=f"Last move: {formatted_move} | {next_turn_text} to play")
        
        self.master.after(1000, self.play_engine_game)

    def ai_move(self):
        if self.board.is_game_over():
            result = self.board.result()
            self.status.config(text=f"Game Over! Result: {result}")
            logging.info(f"Game Over! Result: {result}")
            return

        self.evaluate_parallel()
        
        if self.board.turn == self.player_color:
            return
        
        ai_color_name = "White" if self.board.turn == chess_game.WHITE else "Black"
        self.status.config(text=f"AI ({ai_color_name}) is thinking...")

        move = self.get_best_move(depth=2)
        if not move:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
            else:
                self.status.config(text="No legal moves for AI. Game might be over.")
                return

        formatted_move = self.format_move(move)
        
        is_capture = self.board.piece_at(move.to_square) is not None or self.board.is_en_passant(move)
        
        self.board.push(move)
        self.move_history.append(move.uci())
        self.move_display_history.append(formatted_move)
        
        logging.info(f"AI plays: {formatted_move} ({move.uci()})")
        
        if is_capture and capture_sound:
            pygame.mixer.Sound.play(capture_sound)
        elif move_sound:
            pygame.mixer.Sound.play(move_sound)
            
        self.update_board()
        self.update_clock()
        
        player_turn_text = "White" if self.board.turn == chess_game.WHITE else "Black"
        self.status.config(text=f"Last move: {formatted_move} | {player_turn_text} to play")


def run_gui():
    root = tk.Tk()
    root.withdraw()

    mode = simpledialog.askstring("Choose Mode", "Enter mode:\nmanual / vs_ai / engine")
    if not mode:
        root.destroy()
        return
    mode = mode.lower()

    if mode not in ["manual", "vs_ai", "engine"]:
        print("Invalid mode selected. Exiting.")
        root.destroy()
        return

    player_color = chess_game.WHITE
    if mode in ["manual", "vs_ai"]:
        color_input = simpledialog.askstring("Choose Color", "Play as (white/black)?")
        if not color_input:
            root.destroy()
            return
        color_input = color_input.lower()
        if color_input not in ["white", "black"]:
            print("Invalid color selected. Exiting.")
            root.destroy()
            return
        player_color = chess_game.WHITE if color_input == "white" else chess_game.BLACK

    root.deiconify()
    gui = ChessGUI(root, mode=mode, player_color=player_color)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
