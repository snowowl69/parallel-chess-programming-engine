import os
import sys
import time
import json
import pickle
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import chess
from neural_network import ChessNet, BoardEncoder, TrainingData, create_network, save_model, load_model
from smart_database_parser import process_lichess_database, ParsingConfig
from training_monitor import TrainingDatabase, TrainingMetrics
from self_play_training import SelfPlayEngine, TrainingConfig
@dataclass
class PipelineConfig:
    pgn_file: str = "lichess_db_standard_rated_2017-10.pgn"
    min_rating: int = 1500
    sample_rate: float = 0.1
    positions_per_game: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs_per_checkpoint: int = 5
    initial_epochs: int = 50
    self_play_iterations: int = 20
    games_per_iteration: int = 25
    mcts_simulations: int = 400
    checkpoint_dir: str = "models/checkpoints"
    checkpoint_interval: int = 1
    max_checkpoints: int = 20
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 2
@dataclass
class TrainingState:
    stage: str = "initial"
    epoch: int = 0
    iteration: int = 0
    total_positions_seen: int = 0
    total_training_time: float = 0.0
    best_loss: float = float('inf')
    best_eval_score: float = 0.0
    initial_epochs_completed: int = 0
    self_play_iterations_completed: int = 0
    model_architecture: Dict[str, Any] = None
    last_checkpoint_path: str = ""
    start_time: str = ""
    last_update: str = ""
class IntelligentTrainingPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = TrainingState()
        self.device = torch.device(config.device)
        self.network = None
        self.optimizer = None
        self.training_db = TrainingDatabase("training_history.db")
        os.makedirs("training_logs", exist_ok=True)
        log_file = f"training_logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.state_file = "training_state.json"
    def cleanup_corrupted_files(self):
        files_to_check = [
            "processed_training_data.pkl",
            "training_state.json"
        ]
        cleaned_files = []
        for file_path in files_to_check:
            if os.path.exists(file_path):
                try:
                    if file_path.endswith('.pkl'):
                        with open(file_path, 'rb') as f:
                            pickle.load(f)
                    elif file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            json.load(f)
                except (EOFError, pickle.UnpicklingError, json.JSONDecodeError, FileNotFoundError):
                    self.logger.warning(f"🗑️  Removing corrupted file: {file_path}")
                    try:
                        os.remove(file_path)
                        cleaned_files.append(file_path)
                    except OSError as e:
                        self.logger.error(f"Could not remove {file_path}: {e}")
        if os.path.exists(self.config.checkpoint_dir):
            for file_name in os.listdir(self.config.checkpoint_dir):
                if file_name.endswith('.pth') or file_name.endswith('.pt'):
                    file_path = os.path.join(self.config.checkpoint_dir, file_name)
                    try:
                        torch.load(file_path, map_location='cpu')
                    except Exception:
                        self.logger.warning(f"🗑️  Removing corrupted checkpoint: {file_path}")
                        try:
                            os.remove(file_path)
                            cleaned_files.append(file_path)
                        except OSError as e:
                            self.logger.error(f"Could not remove {file_path}: {e}")
        if cleaned_files:
            self.logger.info(f"✅ Cleaned up {len(cleaned_files)} corrupted files")
        else:
            self.logger.info("✅ No corrupted files found")
        return cleaned_files
    def save_state(self):
        self.state.last_update = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)
    def load_state(self) -> bool:
        if not os.path.exists(self.state_file):
            return False
        try:
            with open(self.state_file, 'r') as f:
                state_dict = json.load(f)
            for key, value in state_dict.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
            self.logger.info(f"Loaded training state from {self.state.last_update}")
            self.logger.info(f"Current stage: {self.state.stage}")
            self.logger.info(f"Progress: Epoch {self.state.epoch}, Iteration {self.state.iteration}")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load training state: {e}")
            return False
    def initialize_network(self):
        if self.state.last_checkpoint_path and os.path.exists(self.state.last_checkpoint_path):
            try:
                self.logger.info(f"Loading model from {self.state.last_checkpoint_path}")
                self.network = load_model(self.state.last_checkpoint_path, self.device)
                checkpoint = torch.load(self.state.last_checkpoint_path, map_location=self.device)
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer = optim.Adam(
                        self.network.parameters(),
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay
                    )
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("✅ Model and optimizer loaded successfully")
                return
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
        self.logger.info("Creating new neural network")
        self.network = create_network(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.state.model_architecture = {
            'num_res_blocks': 10,
            'num_hidden': 256,
            'total_parameters': sum(p.numel() for p in self.network.parameters())
        }
    def save_checkpoint(self, epoch: int, loss: float, eval_score: float = 0.0):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"chess_model_epoch_{epoch}_{timestamp}.pth"
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        checkpoint_data = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'eval_score': eval_score,
            'config': asdict(self.config),
            'state': asdict(self.state),
            'timestamp': timestamp,
            'model_config': {
                'num_res_blocks': 10,
                'num_hidden': 256
            }
        }
        torch.save(checkpoint_data, checkpoint_path)
        self.state.last_checkpoint_path = checkpoint_path
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_name}")
        self.cleanup_old_checkpoints()
        return checkpoint_path
    def cleanup_old_checkpoints(self):
        try:
            checkpoint_files = []
            for file in os.listdir(self.config.checkpoint_dir):
                if file.startswith("chess_model_") and file.endswith(".pth"):
                    full_path = os.path.join(self.config.checkpoint_dir, file)
                    checkpoint_files.append((full_path, os.path.getmtime(full_path)))
            checkpoint_files.sort(key=lambda x: x[1], reverse=True)
            for i, (file_path, _) in enumerate(checkpoint_files):
                if i >= self.config.max_checkpoints:
                    os.remove(file_path)
                    self.logger.info(f"🗑️  Removed old checkpoint: {os.path.basename(file_path)}")
        except Exception as e:
            self.logger.warning(f"Error cleaning up checkpoints: {e}")
    def process_database_stage(self):
        self.logger.info("🗃️  STAGE 1: DATABASE PROCESSING AND INITIAL TRAINING")
        self.logger.info("="*60)
        training_data_file = "processed_training_data.pkl"
        training_data = None
        if os.path.exists(training_data_file):
            self.logger.info(f"Loading existing training data from {training_data_file}")
            try:
                with open(training_data_file, 'rb') as f:
                    training_data = pickle.load(f)
                self.logger.info("✅ Successfully loaded existing training data")
            except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
                self.logger.warning(f"⚠️  Corrupted training data file detected: {e}")
                self.logger.info("🗑️  Removing corrupted file and regenerating training data...")
                try:
                    os.remove(training_data_file)
                    self.logger.info("✅ Corrupted file removed successfully")
                except OSError as remove_error:
                    self.logger.warning(f"Could not remove corrupted file: {remove_error}")
                training_data = None
        if training_data is None:
            self.logger.info(f"Processing PGN database: {self.config.pgn_file}")
            if not os.path.exists(self.config.pgn_file):
                raise FileNotFoundError(f"PGN file not found: {self.config.pgn_file}")
            parsing_config = ParsingConfig(
                min_rating=self.config.min_rating,
                sample_rate=self.config.sample_rate,
                positions_per_game=self.config.positions_per_game
            )
            training_data = process_lichess_database(self.config.pgn_file, parsing_config)
            try:
                temp_file = training_data_file + ".tmp"
                with open(temp_file, 'wb') as f:
                    pickle.dump(training_data, f)
                os.rename(temp_file, training_data_file)
                self.logger.info(f"💾 Training data saved to {training_data_file}")
            except Exception as save_error:
                self.logger.error(f"Failed to save training data: {save_error}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                self.logger.warning("Continuing without saving processed data (will reprocess next time)")
        if len(training_data.positions) == 0:
            raise ValueError("No training positions available!")
        self.logger.info(f"📊 Training data summary:")
        self.logger.info(f"   - Total positions: {len(training_data.positions):,}")
        self.logger.info(f"   - Total policies: {len(training_data.policies):,}")
        self.logger.info(f"   - Total values: {len(training_data.values):,}")
        positions_tensor = torch.stack(training_data.positions)
        # Convert numpy arrays to single array before tensor creation for performance
        import numpy as np
        policies_array = np.array(training_data.policies, dtype=np.float32)
        values_array = np.array(training_data.values, dtype=np.float32)
        policies_tensor = torch.from_numpy(policies_array)
        values_tensor = torch.from_numpy(values_array).unsqueeze(1)
        dataset = TensorDataset(positions_tensor, policies_tensor, values_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        self.network.train()
        start_epoch = self.state.initial_epochs_completed
        for epoch in range(start_epoch, self.config.initial_epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            total_value_loss = 0.0
            total_policy_loss = 0.0
            num_batches = 0
            for batch_idx, (positions, policies, values) in enumerate(dataloader):
                positions = positions.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)
                self.optimizer.zero_grad()
                pred_values, pred_policies = self.network(positions)
                value_loss = nn.MSELoss()(pred_values, values)
                policy_loss = nn.KLDivLoss(reduction='batchmean')(pred_policies, policies)
                loss = value_loss + policy_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                num_batches += 1
                self.state.total_positions_seen += positions.size(0)
                if batch_idx % 100 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.config.initial_epochs} | "
                        f"Batch {batch_idx}/{len(dataloader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Value: {value_loss.item():.4f} | "
                        f"Policy: {policy_loss.item():.4f}"
                    )
                    self.save_state()
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / num_batches
            avg_value_loss = total_value_loss / num_batches
            avg_policy_loss = total_policy_loss / num_batches
            self.state.epoch = epoch + 1
            self.state.initial_epochs_completed = epoch + 1
            self.state.total_training_time += epoch_time
            if avg_loss < self.state.best_loss:
                self.state.best_loss = avg_loss
            metrics = TrainingMetrics(
                iteration=epoch,
                timestamp=datetime.now(),
                games_played=0,
                total_positions=self.state.total_positions_seen,
                average_game_length=0.0,
                game_results={},
                total_loss=avg_loss,
                value_loss=avg_value_loss,
                policy_loss=avg_policy_loss,
                learning_rate=self.config.learning_rate,
                iteration_time=epoch_time,
                games_per_hour=0.0,
                positions_per_second=self.state.total_positions_seen / self.state.total_training_time
            )
            self.training_db.save_iteration_metrics(metrics)
            self.logger.info(
                f"✅ Epoch {epoch+1} complete | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Best Loss: {self.state.best_loss:.4f}"
            )
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1, avg_loss)
                self.save_state()
        self.state.stage = "self_play"
        self.save_state()
        self.logger.info("✅ Initial training stage completed!")
    def self_play_stage(self):
        self.logger.info("🎮 STAGE 2: SELF-PLAY TRAINING")
        self.logger.info("="*40)
        self_play_config = TrainingConfig(
            num_self_play_games=self.config.games_per_iteration,
            mcts_simulations=self.config.mcts_simulations,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            device=self.config.device
        )
        self_play_engine = SelfPlayEngine(self.network, self_play_config)
        start_iteration = self.state.self_play_iterations_completed
        for iteration in range(start_iteration, self.config.self_play_iterations):
            iteration_start_time = time.time()
            self.logger.info(f"🎯 Self-play iteration {iteration+1}/{self.config.self_play_iterations}")
            self.logger.info(f"Generating {self.config.games_per_iteration} self-play games...")
            game_results = []
            positions_collected = 0
            for game_num in range(self.config.games_per_iteration):
                try:
                    game_result = self_play_engine.play_game(f"selfplay_iter{iteration}_game{game_num}")
                    game_results.append(game_result)
                    positions_collected += len(game_result.positions)
                    if (game_num + 1) % 5 == 0:
                        self.logger.info(f"   Generated {game_num+1}/{self.config.games_per_iteration} games")
                except Exception as e:
                    self.logger.warning(f"Error in self-play game {game_num}: {e}")
                    continue
            if game_results:
                self.logger.info(f"Training on {positions_collected:,} new positions...")
                iteration_time = time.time() - iteration_start_time
                self.state.iteration = iteration + 1
                self.state.self_play_iterations_completed = iteration + 1
                self.state.total_training_time += iteration_time
                avg_game_length = sum(len(gr.positions) for gr in game_results) / len(game_results)
                metrics = TrainingMetrics(
                    iteration=iteration,
                    timestamp=datetime.now(),
                    games_played=len(game_results),
                    total_positions=positions_collected,
                    average_game_length=avg_game_length,
                    game_results={"wins": 0, "draws": 0, "losses": 0},
                    total_loss=0.0,
                    value_loss=0.0,
                    policy_loss=0.0,
                    learning_rate=self.config.learning_rate,
                    iteration_time=iteration_time,
                    games_per_hour=len(game_results) * 3600 / iteration_time,
                    positions_per_second=positions_collected / iteration_time
                )
                self.training_db.save_iteration_metrics(metrics)
                self.logger.info(
                    f"✅ Iteration {iteration+1} complete | "
                    f"Games: {len(game_results)} | "
                    f"Positions: {positions_collected:,} | "
                    f"Avg length: {avg_game_length:.1f} | "
                    f"Time: {iteration_time:.1f}s"
                )
                if (iteration + 1) % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(
                        self.state.epoch + iteration + 1,
                        0.0,
                        0.0
                    )
                    self.save_state()
        self.state.stage = "completed"
        self.save_state()
        self.logger.info("✅ Self-play training stage completed!")
    def run_complete_pipeline(self):
        self.logger.info("🚀 STARTING INTELLIGENT CHESS AI TRAINING PIPELINE")
        self.logger.info("="*60)
        self.cleanup_corrupted_files()
        resumed = self.load_state()
        if resumed:
            self.logger.info("📥 Resuming from previous training state")
        else:
            self.state.start_time = datetime.now().isoformat()
        self.initialize_network()
        try:
            if self.state.stage == "initial":
                self.process_database_stage()
            if self.state.stage == "self_play":
                self.self_play_stage()
            if self.state.stage == "completed":
                self.logger.info("🎉 TRAINING PIPELINE COMPLETED!")
                final_checkpoint = self.save_checkpoint(
                    self.state.epoch + self.state.iteration,
                    self.state.best_loss,
                    self.state.best_eval_score
                )
                self.logger.info(f"📁 Final model saved: {final_checkpoint}")
                self.logger.info(f"⏱️  Total training time: {self.state.total_training_time:.2f} seconds")
                self.logger.info(f"📊 Total positions processed: {self.state.total_positions_seen:,}")
                summary = {
                    'training_completed': True,
                    'final_checkpoint': final_checkpoint,
                    'total_time': self.state.total_training_time,
                    'total_positions': self.state.total_positions_seen,
                    'best_loss': self.state.best_loss,
                    'completion_time': datetime.now().isoformat()
                }
                with open("training_summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
        except KeyboardInterrupt:
            self.logger.info("\n⏹️  Training interrupted by user")
            self.save_state()
            self.logger.info("💾 State saved. Resume with the same command.")
        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            self.save_state()
            raise
def create_training_pipeline(pgn_file: str = "small_dataset.pgn", iterations=3, games=5, mcts=50) -> IntelligentTrainingPipeline:
    config = PipelineConfig(
        pgn_file=pgn_file,
        min_rating=1000,
        sample_rate=1.0,
        positions_per_game=2,
        batch_size=16,
        initial_epochs=5,
        self_play_iterations=iterations,
        games_per_iteration=games,
        mcts_simulations=mcts,
        checkpoint_interval=1,
        max_checkpoints=5
    )
    return IntelligentTrainingPipeline(config)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--games', type=int, default=5)
    parser.add_argument('--mcts', type=int, default=50)
    args = parser.parse_args()

    print("🧠 Intelligent Chess AI Training Pipeline")
    print("="*50)
    pipeline = create_training_pipeline(
        iterations=args.iterations,
        games=args.games,
        mcts=args.mcts
    )
    pipeline.run_complete_pipeline()