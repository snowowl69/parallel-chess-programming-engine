import os
import time
import json
import random
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import chess
import chess.pgn
try:
    from neural_network import ChessNet, BoardEncoder, TrainingData, create_network, save_model, load_model
    from mcts import MCTS, MCTSNode
except ImportError:
    print("Warning: Some modules not found. Make sure neural_network.py and mcts.py are in the same directory.")
@dataclass
class TrainingConfig:
    num_self_play_games: int = 100
    mcts_simulations: int = 800
    temperature_threshold: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 10
    num_res_blocks: int = 10
    num_hidden: int = 256
    num_iterations: int = 100
    save_interval: int = 10
    data_buffer_size: int = 10000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
@dataclass
class GameResult:
    game_id: str
    moves: List[str]
    positions: List[str]
    mcts_policies: List[np.ndarray]
    result: str
    game_length: int
    total_time: float
    white_player: str = "SelfPlay"
    black_player: str = "SelfPlay"
class SelfPlayEngine:
    def __init__(self, neural_network: ChessNet, config: TrainingConfig):
        self.network = neural_network
        self.config = config
        self.mcts = MCTS(neural_network, config.device)
        self.mcts.num_simulations = config.mcts_simulations
        self.games_played = 0
        self.total_positions = 0
    def play_game(self, game_id: Optional[str] = None) -> GameResult:
        if game_id is None:
            game_id = f"selfplay_{self.games_played}_{int(time.time())}"
        start_time = time.time()
        board = chess.Board()
        moves = []
        positions = []
        mcts_policies = []
        move_count = 0
        while not board.is_game_over() and move_count < 200:
            positions.append(board.fen())
            temperature = 1.0 if move_count < self.config.temperature_threshold else 0.0
            position_tensor, action_probs = self.mcts.get_training_data(board)
            mcts_policies.append(action_probs)
            move = self.mcts.get_best_move(board, temperature=temperature)
            if move == chess.Move.null():
                break
            moves.append(move.uci())
            board.push(move)
            move_count += 1
        result = board.result()
        total_time = time.time() - start_time
        self.games_played += 1
        self.total_positions += len(positions)
        return GameResult(
            game_id=game_id,
            moves=moves,
            positions=positions,
            mcts_policies=mcts_policies,
            result=result,
            game_length=len(moves),
            total_time=total_time
        )
    def play_multiple_games(self, num_games: int) -> List[GameResult]:
        games = []
        print(f"Playing {num_games} self-play games...")
        for i in range(num_games):
            if i % 10 == 0:
                print(f"Game {i+1}/{num_games}")
            game = self.play_game()
            games.append(game)
        return games
class TrainingDataProcessor:
    def __init__(self, config: TrainingConfig):
        self.config = config
    def process_games(self, games: List[GameResult]) -> TrainingData:
        training_data = TrainingData()
        for game in games:
            game_value = self._get_game_value(game.result)
            for i, (position_fen, mcts_policy) in enumerate(zip(game.positions, game.mcts_policies)):
                board = chess.Board(position_fen)
                position_tensor = BoardEncoder.board_to_tensor(board)
                value = game_value if board.turn == chess.WHITE else -game_value
                training_data.add_sample(position_tensor, mcts_policy, value)
        return training_data
    def _get_game_value(self, result: str) -> float:
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0
class NeuralNetworkTrainer:
    def __init__(self, config: TrainingConfig, network: ChessNet):
        self.config = config
        self.device = torch.device(config.device)
        self.optimizer = optim.Adam(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    def train_network(self, network: ChessNet, training_data: TrainingData) -> Dict[str, float]:
        network.train()
        value_criterion = nn.MSELoss()
        policy_criterion = nn.KLDivLoss(reduction='batchmean')
        positions, policies, values = training_data.get_batch(len(training_data.positions))
        dataset = TensorDataset(
            positions.to(self.device),
            policies.to(self.device),
            values.to(self.device).unsqueeze(1)
        )
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        num_batches = 0
        for epoch in range(self.config.epochs_per_iteration):
            for batch_positions, batch_policies, batch_values in dataloader:
                self.optimizer.zero_grad()
                pred_values, pred_policies = network(batch_positions)
                value_loss = value_criterion(pred_values, batch_values)
                policy_loss = policy_criterion(pred_policies, batch_policies)
                total_batch_loss = value_loss + policy_loss
                total_batch_loss.backward()
                self.optimizer.step()
                total_loss += total_batch_loss.item()
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                num_batches += 1
        metrics = {
            'total_loss': total_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'learning_rate': self.config.learning_rate
        }
        return metrics
class TrainingManager:
    def __init__(self, config: TrainingConfig, model_dir: str = "models"):
        self.config = config
        self.model_dir = model_dir
        self.device = torch.device(config.device)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(f"{model_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{model_dir}/training_data", exist_ok=True)
        self._setup_logging()
        self.network = create_network(self.device)
        self.selfplay_engine = SelfPlayEngine(self.network, config)
        self.data_processor = TrainingDataProcessor(config)
        self.trainer = NeuralNetworkTrainer(config, self.network)
        self.training_history = []
        self.iteration = 0
    def _setup_logging(self):
        log_file = f"{self.model_dir}/training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    def run_training_iteration(self) -> Dict[str, any]:
        start_time = time.time()
        self.iteration += 1
        self.logger.info(f"Starting training iteration {self.iteration}")
        self.logger.info(f"Playing {self.config.num_self_play_games} self-play games...")
        games = self.selfplay_engine.play_multiple_games(self.config.num_self_play_games)
        self.logger.info("Processing training data...")
        training_data = self.data_processor.process_games(games)
        self.logger.info("Training neural network...")
        training_metrics = self.trainer.train_network(self.network, training_data)
        iteration_time = time.time() - start_time
        iteration_results = {
            'iteration': self.iteration,
            'games_played': len(games),
            'training_samples': len(training_data.positions),
            'iteration_time': iteration_time,
            'training_metrics': training_metrics,
            'average_game_length': np.mean([g.game_length for g in games]),
            'game_results': {
                'white_wins': sum(1 for g in games if g.result == "1-0"),
                'black_wins': sum(1 for g in games if g.result == "0-1"),
                'draws': sum(1 for g in games if g.result == "1/2-1/2")
            }
        }
        self.training_history.append(iteration_results)
        if self.iteration % self.config.save_interval == 0:
            self._save_checkpoint()
        self._save_training_data(games, training_data)
        self.logger.info(f"Iteration {self.iteration} completed in {iteration_time:.2f}s")
        self.logger.info(f"Training loss: {training_metrics['total_loss']:.4f}")
        return iteration_results
    def run_full_training(self):
        self.logger.info(f"Starting full training process - {self.config.num_iterations} iterations")
        self.logger.info(f"Configuration: {asdict(self.config)}")
        try:
            for i in range(self.config.num_iterations):
                iteration_results = self.run_training_iteration()
                print(f"\nIteration {i+1}/{self.config.num_iterations} Summary:")
                print(f"  Games played: {iteration_results['games_played']}")
                print(f"  Training samples: {iteration_results['training_samples']}")
                print(f"  Training loss: {iteration_results['training_metrics']['total_loss']:.4f}")
                print(f"  Average game length: {iteration_results['average_game_length']:.1f}")
                print(f"  Time: {iteration_results['iteration_time']:.1f}s")
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self._save_checkpoint()
            self.logger.info("Training completed")
    def _save_checkpoint(self):
        checkpoint_path = f"{self.model_dir}/checkpoints/model_iter_{self.iteration}.pt"
        save_model(self.network, checkpoint_path)
        history_path = f"{self.model_dir}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    def _save_training_data(self, games: List[GameResult], training_data: TrainingData):
        pgn_path = f"{self.model_dir}/training_data/games_iter_{self.iteration}.pgn"
        with open(pgn_path, 'w') as f:
            for game in games:
                pgn_game = chess.pgn.Game()
                pgn_game.headers["Event"] = "Self-Play Training"
                pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
                pgn_game.headers["White"] = "ChessAI"
                pgn_game.headers["Black"] = "ChessAI"
                pgn_game.headers["Result"] = game.result
                board = chess.Board()
                node = pgn_game
                for move_uci in game.moves:
                    move = chess.Move.from_uci(move_uci)
                    node = node.add_variation(move)
                    board.push(move)
                print(pgn_game, file=f)
                print("", file=f)
        data_path = f"{self.model_dir}/training_data/training_data_iter_{self.iteration}.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'positions': training_data.positions,
                'policies': training_data.policies,
                'values': training_data.values
            }, f)
def create_default_config() -> TrainingConfig:
    return TrainingConfig(
        num_self_play_games=25,
        mcts_simulations=400,
        num_iterations=50,
        batch_size=32,
        learning_rate=0.001,
        epochs_per_iteration=5
    )
if __name__ == "__main__":
    print("🚀 Starting AlphaZero-style Self-Play Training!")
    print("=" * 50)
    config = create_default_config()
    print(f"Device: {config.device}")
    print(f"Games per iteration: {config.num_self_play_games}")
    print(f"MCTS simulations: {config.mcts_simulations}")
    print(f"Total iterations: {config.num_iterations}")
    trainer = TrainingManager(config)
    try:
        trainer.run_full_training()
        print("\n🎉 Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise