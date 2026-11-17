import tkinter as tk
from tkinter import simpledialog
import chess as chess_game
import random
import torch
import threading
import time
import logging
from datetime import datetime
import logging
from datetime import datetime
import pygame
import platform

from chess_ai import get_best_move, evaluate_board

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
def evaluate_board_global(board, device):
    piece_values = torch.tensor([1, 3, 3, 5, 9], device=device, dtype=torch.float32)
    eval_score = torch.tensor(0, device=device, dtype=torch.float32)
    for piece, value in zip([chess_game.PAWN, chess_game.KNIGHT, chess_game.BISHOP, chess_game.ROOK, chess_game.QUEEN], piece_values):
        eval_score += len(board.pieces(piece, chess_game.WHITE)) * value
        eval_score -= len(board.pieces(piece, chess_game.BLACK)) * value
    return eval_score.item()
import numpy as np
pygame.init()
try:
    pygame.mixer.init()
    def make_beep(freq, duration):
        sample_rate = 44100
        n_samples = int(round(duration * sample_rate))
        buf = np.zeros((n_samples, 2), dtype=np.int16)
        max_sample = int(32767 * 0.1)  # 10% volume
        t = np.linspace(0, duration, n_samples, False)
        wave = np.sin(freq * t * 2 * np.pi)
        sound_wave = np.int16(wave * max_sample)
        buf[:, 0] = sound_wave
        buf[:, 1] = sound_wave
        return pygame.sndarray.make_sound(buf)
    move_sound = make_beep(440, 0.1)
    capture_sound = make_beep(880, 0.15)
except Exception:
    move_sound = None
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
        if self.board.is_game_over():
            self.timer_running = False
            result = self.board.result()
            if result == "1-0":
                winner = "White wins!"
            elif result == "0-1":
                winner = "Black wins!"
            else:
                winner = "Draw!"
            self.status.config(text=f"Game Over! {winner}")
            return
        current_player_idx = 0 if self.board.turn == chess_game.WHITE else 1
        self.time_left[current_player_idx] -= 1
        if self.time_left[current_player_idx] <= 0:
            self.timer_running = False
            winner = "Black wins!" if current_player_idx == 0 else "White wins!"
            self.status.config(text=f"Time's up! {winner}")
            logging.info(f"Time's up! {winner}")
            return
        for i in range(2):
            mins, secs = divmod(max(0, self.time_left[i]), 60)
            color = 'White' if i == 0 else 'Black'
            label = f"{color}: {mins:02}:{secs:02}"
            self.clock_labels[i].config(
                text=label,
                fg='green' if self.board.turn == (i == 0) else 'black'
            )
        for i in range(2):
            mins, secs = divmod(self.time_left[i], 60)
            color = 'White' if i == 0 else 'Black'
            label = f"{color}: {mins:02}:{secs:02}"
            self.clock_labels[i].config(text=label, fg='green' if self.board.turn == (i == 0) else 'black')
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
        self.canvas.delete("highlight")
        clicked_square = self.get_square_under_mouse(event)
        if clicked_square is not None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.drag_start_square = clicked_square
                self.highlight_legal_moves(clicked_square)
            else:
                self.drag_start_square = None
    def highlight_legal_moves(self, from_square):
        for move in self.board.legal_moves:
            if move.from_square == from_square:
                file = chess_game.square_file(move.to_square)
                rank = chess_game.square_rank(move.to_square)
                if self.player_color == chess_game.WHITE:
                    display_rank = 7 - rank
                    display_file = file
                else:
                    display_rank = rank
                    display_file = 7 - file
                x1 = display_file * self.square_size
                y1 = display_rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill="#00ff00",
                    stipple="gray50",
                    tags="highlight"
                )
    def on_drag(self, event):
        pass
    def check_game_over(self):
        if self.board.is_game_over():
            self.timer_running = False
            result = self.board.result()
            if result == "1-0":
                winner = "White wins!"
            elif result == "0-1":
                winner = "Black wins!"
            else:
                winner = "Draw!"
            self.status.config(text=f"Game Over! {winner}")
            logging.info(f"Game Over! {winner}")
            return True
        return False
    def on_drop(self, event):
        if self.mode == "engine" or (self.mode == "vs_ai" and self.board.turn != self.player_color):
            return
        self.canvas.delete("highlight")
        from_sq = self.drag_start_square
        to_sq = self.get_square_under_mouse(event)
        if from_sq is None or to_sq is None:
            return
        move_to_make = None
        piece_on_from_sq = self.board.piece_at(from_sq)
        if piece_on_from_sq and piece_on_from_sq.piece_type == chess_game.PAWN:
            if (self.board.turn == chess_game.WHITE and chess_game.square_rank(to_sq) == 7) or \
               (self.board.turn == chess_game.BLACK and chess_game.square_rank(to_sq) == 0):
                promotion_piece = self.show_promotion_dialog()
                promo_move = chess_game.Move(from_sq, to_sq, promotion=promotion_piece)
                if promo_move in self.board.legal_moves:
                    move_to_make = promo_move
                else:
                    promo_move = chess_game.Move(from_sq, to_sq, promotion=chess_game.QUEEN)
                    if promo_move in self.board.legal_moves:
                        move_to_make = promo_move
        if move_to_make is None:
            move_to_make = chess_game.Move(from_sq, to_sq)
        if move_to_make in self.board.legal_moves:
            is_capture = self.board.piece_at(move_to_make.to_square) is not None or \
                         self.board.is_en_passant(move_to_make)
            formatted_move = self.format_move(move_to_make)
            if is_capture and capture_sound:
                capture_sound.play()
            elif move_sound:
                move_sound.play()
            self.board.push(move_to_make)
            move_uci = move_to_make.uci()
            self.move_history.append(move_uci)
            self.move_display_history.append(formatted_move)
            self.update_board()
            self.update_clock()
            logging.info(f"Player plays: {formatted_move} ({move_uci})")
            if not self.check_game_over():
                self.evaluate_parallel()
                if self.mode == "vs_ai":
                    self.master.after(500, self.ai_move)
            turn_text = "White" if self.board.turn == chess_game.WHITE else "Black"
            self.status.config(text=f"Last move: {formatted_move} | {turn_text} to play")
        self.drag_start_square = None
    def evaluate_board(self, board_to_eval, device=None):
        return evaluate_board(board_to_eval)
        
    def evaluate_parallel(self):
        if self.board.is_game_over():
            return
        start = time.time()
        score = self.evaluate_board(self.board)
        duration = time.time() - start
        print(f"Position evaluation: {score:.2f} (computed in {duration:.3f}s)")
        logging.info("Evaluation completed")
        if len(self.move_display_history) > 0:
            turn_text = "White" if self.board.turn == chess_game.WHITE else "Black"
            last_move = self.move_display_history[-1]
            self.status.config(text=f"Last move: {last_move} | {turn_text} to play")
            
    def get_best_move(self, depth):
        # We delegate to our new shared AI module
        return get_best_move(self.board.copy(), difficulty=2)
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
        self.update_board()
        next_turn_text = "White" if self.board.turn == chess_game.WHITE else "Black"
        self.status.config(text=f"Last move: {formatted_move} | {next_turn_text} to play")
        self.master.after(1000, self.play_engine_game)
    def ai_move(self):
        if self.check_game_over():
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
        self.update_board()
        self.update_clock()
        player_turn_text = "White" if self.board.turn == chess_game.WHITE else "Black"
        self.status.config(text=f"Last move: {formatted_move} | {player_turn_text} to play")
    def show_promotion_dialog(self):
        from tkinter import messagebox
        pieces = {
            'Queen (Recommended)': chess_game.QUEEN,
            'Rook': chess_game.ROOK,
            'Bishop': chess_game.BISHOP,
            'Knight': chess_game.KNIGHT
        }
        choice = simpledialog.askstring(
            "Pawn Promotion",
            "Choose promotion piece:\n1 = Queen\n2 = Rook\n3 = Bishop\n4 = Knight\nEnter number (1-4):",
            initialvalue="1"
        )
        if choice == "2":
            return chess_game.ROOK
        elif choice == "3":
            return chess_game.BISHOP
        elif choice == "4":
            return chess_game.KNIGHT
        else:
            return chess_game.QUEEN
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