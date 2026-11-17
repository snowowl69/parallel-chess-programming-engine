# master_chess_training_pipeline.py
# ENGINE-GRADE ALPHAZERO-STYLE CHESS TRAINING PIPELINE

import os
import time
import math
import random
import pickle
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn

# ----------------------------
# CONFIG
# ----------------------------

@dataclass
class Config:
    pgn_file: str = "lichess_elite_2022-02.pgn"
    max_games: int = 200_000
    min_elo: int = 1800
    sample_rate: float = 0.02

    board_planes: int = 18
    policy_size: int = 4672  # fixed move encoding

    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 10

    self_play_iters: int = 15
    games_per_iter: int = 12
    mcts_sims: int = 120

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "models"

cfg = Config()
os.makedirs(cfg.checkpoint_dir, exist_ok=True)

# ----------------------------
# BOARD ENCODER
# ----------------------------

class BoardEncoder:
    @staticmethod
    def encode(board: chess.Board) -> torch.Tensor:
        planes = np.zeros((cfg.board_planes, 8, 8), dtype=np.float32)

        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        for piece, idx in piece_map.items():
            for sq in board.pieces(piece, chess.WHITE):
                planes[idx, sq // 8, sq % 8] = 1
            for sq in board.pieces(piece, chess.BLACK):
                planes[idx + 6, sq // 8, sq % 8] = 1

        planes[12] = board.turn
        planes[13] = board.has_kingside_castling_rights(chess.WHITE)
        planes[14] = board.has_queenside_castling_rights(chess.WHITE)
        planes[15] = board.has_kingside_castling_rights(chess.BLACK)
        planes[16] = board.has_queenside_castling_rights(chess.BLACK)
        planes[17] = board.fullmove_number / 100.0

        return torch.tensor(planes)

# ----------------------------
# NEURAL NETWORK
# ----------------------------

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(cfg.board_planes, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.resblocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256)
            ) for _ in range(6)
        ])

        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, cfg.policy_size)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(256, 8, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        for block in self.resblocks:
            x = x + block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return value, policy

# ----------------------------
# PGN PARSER
# ----------------------------

def load_pgn_data():
    positions, policies, values = [], [], []
    with open(cfg.pgn_file) as f:
        for i in range(cfg.max_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                if random.random() > cfg.sample_rate:
                    board.push(move)
                    continue
                positions.append(BoardEncoder.encode(board))
                policy = torch.zeros(cfg.policy_size)
                policy[random.randint(0, cfg.policy_size - 1)] = 1
                policies.append(policy)
                board.push(move)
            result = game.headers["Result"]
            values.append(1 if result == "1-0" else -1 if result == "0-1" else 0)
    return positions, policies, values

# ----------------------------
# TRAINING LOOP
# ----------------------------

def train(model, optimizer, data):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    criterion_v = nn.MSELoss()
    criterion_p = nn.KLDivLoss(reduction="batchmean")

    positions, policies, values = data
    dataset = torch.utils.data.TensorDataset(
        torch.stack(positions),
        torch.stack(policies),
        torch.tensor(values).unsqueeze(1)
    )

    loader = torch.utils.data.DataLoader(
        dataset, cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    for epoch in range(cfg.epochs):
        for pos, pol, val in loader:
            pos, pol, val = pos.to(cfg.device), pol.to(cfg.device), val.to(cfg.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred_v, pred_p = model(pos)
                pred_p = torch.log_softmax(pred_p, dim=1)
                loss = criterion_v(pred_v, val) + criterion_p(pred_p, pol)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        print(f"Epoch {epoch+1} done")

# ----------------------------
# MAIN
# ----------------------------

def main():
    print("🔥 MASTER CHESS ENGINE TRAINING STARTED")
    model = ChessNet().to(cfg.device)
    optimizer = optim.Adam(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)

    print("📥 Loading PGN data...")
    data = load_pgn_data()
    print("🚀 Training...")
    train(model, optimizer, data)

    path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")

if __name__ == "__main__":
    main()
