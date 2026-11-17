import os
import sys
import time
import json
import pickle
import logging
import threading
from datetime import datetime
from typing import Iterator, List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch
import chess
import chess.pgn
from io import StringIO
from neural_network import ChessNet, BoardEncoder, MoveEncoder, TrainingData, create_network, save_model, load_model
from training_monitor import TrainingDatabase, TrainingMetrics
@dataclass
class ParsingConfig:
    min_rating: int = 1500
    max_rating: int = 3000
    min_game_length: int = 10
    max_game_length: int = 200
    time_controls: List[str] = None
    sample_rate: float = 0.1
    positions_per_game: int = 5
    batch_size: int = 1000
@dataclass
class ProcessingStats:
    games_read: int = 0
    games_filtered: int = 0
    positions_extracted: int = 0
    processing_time: float = 0.0
    current_file_position: int = 0
    total_file_size: int = 0
class SmartPGNParser:
    def __init__(self, config: ParsingConfig):
        self.config = config
        self.stats = ProcessingStats()
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('database_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.progress_file = "parsing_progress.json"
    def save_progress(self, file_position: int, stats: ProcessingStats):
        progress = {
            'file_position': file_position,
            'stats': {
                'games_read': stats.games_read,
                'games_filtered': stats.games_filtered,
                'positions_extracted': stats.positions_extracted,
                'processing_time': stats.processing_time
            },
            'timestamp': datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    def load_progress(self) -> Tuple[int, ProcessingStats]:
        if not os.path.exists(self.progress_file):
            return 0, ProcessingStats()
        try:
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            stats = ProcessingStats()
            stats.games_read = progress['stats']['games_read']
            stats.games_filtered = progress['stats']['games_filtered']
            stats.positions_extracted = progress['stats']['positions_extracted']
            stats.processing_time = progress['stats']['processing_time']
            file_position = progress['file_position']
            self.logger.info(f"Resuming from position {file_position:,} ({progress['timestamp']})")
            return file_position, stats
        except Exception as e:
            self.logger.warning(f"Could not load progress: {e}")
            return 0, ProcessingStats()
    def filter_game(self, game: chess.pgn.Game) -> bool:
        try:
            white_elo = game.headers.get("WhiteElo")
            black_elo = game.headers.get("BlackElo")
            if white_elo and black_elo:
                try:
                    white_rating = int(white_elo)
                    black_rating = int(black_elo)
                    if (white_rating < self.config.min_rating or white_rating > self.config.max_rating or
                        black_rating < self.config.min_rating or black_rating > self.config.max_rating):
                        return False
                except ValueError:
                    return False
            if self.config.time_controls:
                time_control = game.headers.get("TimeControl", "")
                if not any(tc in time_control for tc in self.config.time_controls):
                    return False
            result = game.headers.get("Result", "*")
            if result == "*":
                return False
            return True
        except Exception:
            return False
    def extract_positions(self, game: chess.pgn.Game) -> List[Tuple[torch.Tensor, np.ndarray, float]]:
        positions = []
        board = game.board()
        moves = list(game.mainline_moves())
        if len(moves) < self.config.min_game_length or len(moves) > self.config.max_game_length:
            return positions
        result = game.headers.get("Result", "*")
        if result == "1-0":
            game_value = 1.0
        elif result == "0-1":
            game_value = -1.0
        else:
            game_value = 0.0
        num_positions = min(self.config.positions_per_game, len(moves))
        if num_positions <= 0:
            return positions
        position_indices = np.linspace(0, len(moves) - 1, num_positions, dtype=int)
        for i, move_idx in enumerate(position_indices):
            try:
                temp_board = game.board()
                for move in moves[:move_idx]:
                    temp_board.push(move)
                board_tensor = self.board_encoder.board_to_tensor(temp_board)
                position_value = game_value if temp_board.turn == chess.WHITE else -game_value
                next_move = moves[move_idx]
                policy = np.zeros(4096, dtype=np.float32)
                policy_index = self.move_encoder.move_to_index(next_move)
                policy[policy_index] = 1.0
                board_tensor_np = board_tensor.numpy() if torch.is_tensor(board_tensor) else board_tensor
                for k in range(4):
                    rotated = np.rot90(board_tensor_np, k, axes=(1, 2)).copy()
                    positions.append((torch.tensor(rotated), policy.copy(), position_value))
                    flipped = np.flip(rotated, axis=2).copy()
                    positions.append((torch.tensor(flipped), policy.copy(), position_value))
            except Exception as e:
                self.logger.debug(f"Error extracting position {move_idx}: {e}")
                continue
        return positions
    def process_database(self, pgn_file_path: str) -> TrainingData:
        self.logger.info(f"Starting database processing: {pgn_file_path}")
        file_size = os.path.getsize(pgn_file_path)
        self.stats.total_file_size = file_size
        self.logger.info(f"File size: {file_size / (1024**3):.2f} GB")
        start_position, self.stats = self.load_progress()
        training_data = TrainingData()
        start_time = time.time()
        last_save_time = start_time
        save_interval = 300
        try:
            with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                if start_position > 0:
                    pgn_file.seek(start_position)
                while True:
                    current_position = pgn_file.tell()
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break
                        self.stats.games_read += 1
                        if np.random.random() > self.config.sample_rate:
                            continue
                        if not self.filter_game(game):
                            continue
                        self.stats.games_filtered += 1
                        positions = self.extract_positions(game)
                        for board_tensor, policy, value in positions:
                            training_data.add_sample(board_tensor, policy, value)
                            self.stats.positions_extracted += 1
                        if self.stats.games_read % 1000 == 0:
                            progress = (current_position / file_size) * 100
                            elapsed = time.time() - start_time
                            games_per_sec = self.stats.games_read / elapsed if elapsed > 0 else 0
                            self.logger.info(
                                f"Progress: {progress:.1f}% | "
                                f"Games: {self.stats.games_read:,} | "
                                f"Filtered: {self.stats.games_filtered:,} | "
                                f"Positions: {self.stats.positions_extracted:,} | "
                                f"Speed: {games_per_sec:.1f} games/sec"
                            )
                        current_time = time.time()
                        if current_time - last_save_time > save_interval:
                            self.stats.processing_time = current_time - start_time
                            self.save_progress(current_position, self.stats)
                            last_save_time = current_time
                    except Exception as e:
                        self.logger.warning(f"Error processing game at position {current_position}: {e}")
                        continue
        except Exception as e:
            self.logger.error(f"Error reading file: {e}")
            raise
        total_time = time.time() - start_time
        self.stats.processing_time = total_time
        self.logger.info("="*60)
        self.logger.info("DATABASE PROCESSING COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Total games read: {self.stats.games_read:,}")
        self.logger.info(f"Games after filtering: {self.stats.games_filtered:,}")
        self.logger.info(f"Training positions extracted: {self.stats.positions_extracted:,}")
        self.logger.info(f"Processing time: {total_time:.2f} seconds")
        self.logger.info(f"Average speed: {self.stats.games_read / total_time:.1f} games/sec")
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
        return training_data
def process_lichess_database(pgn_file: str, config: Optional[ParsingConfig] = None) -> TrainingData:
    if config is None:
        config = ParsingConfig()
    parser = SmartPGNParser(config)
    return parser.process_database(pgn_file)
if __name__ == "__main__":
    print("🗃️  Chess Database Parser")
    print("="*50)
    pgn_file = "lichess_db_standard_rated_2017-10.pgn"
    if not os.path.exists(pgn_file):
        print(f"❌ PGN file not found: {pgn_file}")
        print("Please ensure the Lichess database file is in the current directory.")
        sys.exit(1)
    config = ParsingConfig(
        min_rating=1500,
        sample_rate=0.05,
        positions_per_game=3,
        batch_size=1000
    )
    print(f"📁 Processing: {pgn_file}")
    print(f"⚙️  Configuration:")
    print(f"   - Min rating: {config.min_rating}")
    print(f"   - Sample rate: {config.sample_rate*100:.1f}%")
    print(f"   - Positions per game: {config.positions_per_game}")
    try:
        training_data = process_lichess_database(pgn_file, config)
        save_path = "processed_training_data.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"✅ Training data saved to: {save_path}")
        print(f"📊 Total positions: {len(training_data.positions):,}")
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()