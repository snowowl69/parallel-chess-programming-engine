import os
import time
import json
import random
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import chess
import chess.engine
import chess.pgn
try:
    from neural_network import ChessNet, load_model
    from mcts import MCTS
    from training_monitor import TrainingDatabase
except ImportError:
    print("Warning: Some modules not found")
@dataclass
class GameResult:
    game_id: str
    white_player: str
    black_player: str
    result: str
    moves: List[str]
    game_length: int
    time_control: str
    termination_reason: str
    timestamp: datetime
@dataclass
class EvaluationResults:
    test_name: str
    our_model: str
    opponent: str
    total_games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    estimated_elo_diff: float
    average_game_length: float
    total_time: float
    games: List[GameResult]
class ChessEngineInterface:
    def __init__(self, engine_path: str, time_limit: float = 1.0, depth: int = 10):
        self.engine_path = engine_path
        self.time_limit = time_limit
        self.depth = depth
        self.engine = None
    def start_engine(self):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            print(f"✅ Engine started: {self.engine_path}")
        except Exception as e:
            print(f"❌ Failed to start engine: {e}")
            self.engine = None
    def stop_engine(self):
        if self.engine:
            self.engine.quit()
            self.engine = None
    def get_move(self, board: chess.Board) -> chess.Move:
        if not self.engine:
            raise RuntimeError("Engine not started")
        limit = chess.engine.Limit(time=self.time_limit, depth=self.depth)
        result = self.engine.play(board, limit)
        return result.move
    def configure_strength(self, elo: int):
        if self.engine:
            try:
                self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
                print(f"Engine strength set to {elo} Elo")
            except:
                print("Engine doesn't support strength configuration")
class ModelEvaluator:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = load_model(model_path, device)
        self.model.eval()
        self.mcts = MCTS(self.model, device)
        self.mcts.num_simulations = 400
    def play_against_engine(self, engine: ChessEngineInterface,
                          num_games: int = 10,
                          time_control: str = "5+0") -> List[GameResult]:
        games = []
        print(f"🎯 Playing {num_games} games against engine...")
        for i in range(num_games):
            our_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            print(f"Game {i+1}/{num_games} - Playing as {'White' if our_color == chess.WHITE else 'Black'}")
            game_result = self._play_single_game_vs_engine(
                engine, our_color, f"eval_game_{i+1}", time_control
            )
            games.append(game_result)
            wins = sum(1 for g in games if self._game_result_for_us(g, our_color) == 1.0)
            print(f"   Result: {game_result.result} | Our score: {wins}/{len(games)}")
        return games
    def _play_single_game_vs_engine(self, engine: ChessEngineInterface,
                                   our_color: chess.Color,
                                   game_id: str,
                                   time_control: str) -> GameResult:
        board = chess.Board()
        moves = []
        start_time = time.time()
        our_name = "ChessAI"
        engine_name = "Engine"
        while not board.is_game_over() and len(moves) < 200:
            if board.turn == our_color:
                try:
                    move = self.mcts.get_best_move(board, temperature=0.0)
                    if move == chess.Move.null():
                        break
                except Exception as e:
                    print(f"Error getting our move: {e}")
                    break
            else:
                try:
                    move = engine.get_move(board)
                except Exception as e:
                    print(f"Error getting engine move: {e}")
                    break
            moves.append(move.uci())
            board.push(move)
        result = board.result()
        if board.is_checkmate():
            termination = "checkmate"
        elif board.is_stalemate():
            termination = "stalemate"
        elif board.is_insufficient_material():
            termination = "insufficient material"
        elif board.is_seventyfive_moves():
            termination = "75-move rule"
        elif board.is_fivefold_repetition():
            termination = "5-fold repetition"
        elif len(moves) >= 200:
            termination = "move limit"
            result = "1/2-1/2"
        else:
            termination = "unknown"
        total_time = time.time() - start_time
        return GameResult(
            game_id=game_id,
            white_player=our_name if our_color == chess.WHITE else engine_name,
            black_player=engine_name if our_color == chess.WHITE else our_name,
            result=result,
            moves=moves,
            game_length=len(moves),
            time_control=time_control,
            termination_reason=termination,
            timestamp=datetime.now()
        )
    def play_against_previous_version(self, old_model_path: str,
                                    num_games: int = 20) -> List[GameResult]:
        print(f"🕐 Loading previous model version...")
        old_model = load_model(old_model_path, self.device)
        old_model.eval()
        old_mcts = MCTS(old_model, self.device)
        old_mcts.num_simulations = self.mcts.num_simulations
        games = []
        print(f"🎯 Playing {num_games} games against previous version...")
        for i in range(num_games):
            new_plays_white = i % 2 == 0
            game_result = self._play_model_vs_model(
                self.mcts if new_plays_white else old_mcts,
                old_mcts if new_plays_white else self.mcts,
                "NewModel" if new_plays_white else "OldModel",
                "OldModel" if new_plays_white else "NewModel",
                f"version_test_{i+1}"
            )
            games.append(game_result)
            new_score = 0
            for g in games:
                if g.white_player == "NewModel":
                    if g.result == "1-0":
                        new_score += 1
                    elif g.result == "1/2-1/2":
                        new_score += 0.5
                else:
                    if g.result == "0-1":
                        new_score += 1
                    elif g.result == "1/2-1/2":
                        new_score += 0.5
            print(f"Game {i+1}/{num_games} - Result: {game_result.result} | "
                  f"New model score: {new_score}/{len(games)}")
        return games
    def _play_model_vs_model(self, white_mcts: MCTS, black_mcts: MCTS,
                           white_name: str, black_name: str,
                           game_id: str) -> GameResult:
        board = chess.Board()
        moves = []
        start_time = time.time()
        while not board.is_game_over() and len(moves) < 200:
            current_mcts = white_mcts if board.turn == chess.WHITE else black_mcts
            try:
                move = current_mcts.get_best_move(board, temperature=0.0)
                if move == chess.Move.null():
                    break
                moves.append(move.uci())
                board.push(move)
            except Exception as e:
                print(f"Error in model vs model game: {e}")
                break
        result = board.result()
        total_time = time.time() - start_time
        return GameResult(
            game_id=game_id,
            white_player=white_name,
            black_player=black_name,
            result=result,
            moves=moves,
            game_length=len(moves),
            time_control="unlimited",
            termination_reason="game_over" if board.is_game_over() else "move_limit",
            timestamp=datetime.now()
        )
    def _game_result_for_us(self, game: GameResult, our_color: chess.Color) -> float:
        if game.result == "1/2-1/2":
            return 0.5
        elif (game.result == "1-0" and our_color == chess.WHITE) or \
             (game.result == "0-1" and our_color == chess.BLACK):
            return 1.0
        else:
            return 0.0
class EloEstimator:
    @staticmethod
    def calculate_elo_diff(wins: int, draws: int, losses: int) -> float:
        total_games = wins + draws + losses
        if total_games == 0:
            return 0.0
        score = (wins + 0.5 * draws) / total_games
        score = max(0.001, min(0.999, score))
        elo_diff = 400 * np.log10(score / (1 - score))
        return elo_diff
    @staticmethod
    def calculate_confidence_interval(wins: int, draws: int, losses: int,
                                    confidence: float = 0.95) -> Tuple[float, float]:
        total_games = wins + draws + losses
        if total_games < 10:
            return (float('-inf'), float('inf'))
        score = (wins + 0.5 * draws) / total_games
        variance = score * (1 - score) / total_games
        std_error = np.sqrt(variance)
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)
        lower_score = max(0.001, score - z * std_error)
        upper_score = min(0.999, score + z * std_error)
        lower_elo = 400 * np.log10(lower_score / (1 - lower_score))
        upper_elo = 400 * np.log10(upper_score / (1 - upper_score))
        return (lower_elo, upper_elo)
class EvaluationManager:
    def __init__(self, model_path: str, results_dir: str = "evaluation_results"):
        self.model_path = model_path
        self.results_dir = results_dir
        self.evaluator = ModelEvaluator(model_path)
        os.makedirs(results_dir, exist_ok=True)
    def run_comprehensive_evaluation(self) -> Dict[str, EvaluationResults]:
        results = {}
        print("🏆 Starting Comprehensive Model Evaluation")
        print("=" * 50)
        print("\n📊 Test 1: Against Weak Engine (1200 Elo)")
        try:
            weak_engine = self._setup_stockfish(elo=1200, time_limit=0.5)
            if weak_engine:
                games = self.evaluator.play_against_engine(weak_engine, num_games=10)
                results['weak_engine'] = self._analyze_games(
                    games, "WeakEngine_1200", "Our model vs weak engine"
                )
                weak_engine.stop_engine()
        except Exception as e:
            print(f"❌ Weak engine test failed: {e}")
        print("\n📊 Test 2: Against Medium Engine (1600 Elo)")
        try:
            medium_engine = self._setup_stockfish(elo=1600, time_limit=1.0)
            if medium_engine:
                games = self.evaluator.play_against_engine(medium_engine, num_games=10)
                results['medium_engine'] = self._analyze_games(
                    games, "MediumEngine_1600", "Our model vs medium engine"
                )
                medium_engine.stop_engine()
        except Exception as e:
            print(f"❌ Medium engine test failed: {e}")
        print("\n📊 Test 3: Against Previous Version")
        previous_models = self._find_previous_models()
        if previous_models:
            try:
                old_model_path = previous_models[0]
                games = self.evaluator.play_against_previous_version(old_model_path, num_games=20)
                results['previous_version'] = self._analyze_games(
                    games, "PreviousVersion", "Current vs previous version"
                )
            except Exception as e:
                print(f"❌ Previous version test failed: {e}")
        else:
            print("⚠️  No previous model versions found")
        self._save_evaluation_results(results)
        self._print_evaluation_summary(results)
        return results
    def _setup_stockfish(self, elo: int, time_limit: float) -> Optional[ChessEngineInterface]:
        stockfish_paths = [
            "stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "stockfish.exe",
            r"C:\Program Files\Stockfish\stockfish.exe"
        ]
        for path in stockfish_paths:
            try:
                engine = ChessEngineInterface(path, time_limit=time_limit)
                engine.start_engine()
                if engine.engine:
                    engine.configure_strength(elo)
                    return engine
            except:
                continue
        print(f"⚠️  Stockfish not found. Install Stockfish for engine testing.")
        return None
    def _find_previous_models(self) -> List[str]:
        model_dir = os.path.dirname(self.model_path)
        if not model_dir:
            model_dir = "models/checkpoints"
        if not os.path.exists(model_dir):
            return []
        model_files = []
        for filename in os.listdir(model_dir):
            if filename.endswith('.pt') and filename != os.path.basename(self.model_path):
                model_files.append(os.path.join(model_dir, filename))
        model_files.sort(key=os.path.getmtime, reverse=True)
        return model_files[:3]
    def _analyze_games(self, games: List[GameResult], opponent: str,
                      description: str) -> EvaluationResults:
        total_games = len(games)
        wins = 0
        losses = 0
        draws = 0
        for game in games:
            if game.result == "1/2-1/2":
                draws += 1
            elif (game.result == "1-0" and game.white_player.startswith("ChessAI")) or \
                 (game.result == "0-1" and game.black_player.startswith("ChessAI")):
                wins += 1
            else:
                losses += 1
        win_rate = wins / total_games if total_games > 0 else 0.0
        elo_diff = EloEstimator.calculate_elo_diff(wins, draws, losses)
        avg_game_length = np.mean([game.game_length for game in games]) if games else 0.0
        total_time = sum(game.timestamp.timestamp() for game in games) if games else 0.0
        results = EvaluationResults(
            test_name=description,
            our_model=self.model_path,
            opponent=opponent,
            total_games=total_games,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=win_rate,
            estimated_elo_diff=elo_diff,
            average_game_length=avg_game_length,
            total_time=total_time,
            games=games
        )
        return results
    def _save_evaluation_results(self, results: Dict[str, EvaluationResults]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.results_dir, f"evaluation_summary_{timestamp}.json")
        summary = {}
        for test_name, result in results.items():
            summary[test_name] = {
                'total_games': result.total_games,
                'wins': result.wins,
                'losses': result.losses,
                'draws': result.draws,
                'win_rate': result.win_rate,
                'estimated_elo_diff': result.estimated_elo_diff,
                'average_game_length': result.average_game_length
            }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        for test_name, result in results.items():
            pgn_path = os.path.join(self.results_dir, f"{test_name}_{timestamp}.pgn")
            with open(pgn_path, 'w') as f:
                for game in result.games:
                    pgn_game = chess.pgn.Game()
                    pgn_game.headers["Event"] = f"Evaluation: {result.test_name}"
                    pgn_game.headers["Date"] = game.timestamp.strftime("%Y.%m.%d")
                    pgn_game.headers["White"] = game.white_player
                    pgn_game.headers["Black"] = game.black_player
                    pgn_game.headers["Result"] = game.result
                    pgn_game.headers["TimeControl"] = game.time_control
                    pgn_game.headers["Termination"] = game.termination_reason
                    board = chess.Board()
                    node = pgn_game
                    for move_uci in game.moves:
                        try:
                            move = chess.Move.from_uci(move_uci)
                            node = node.add_variation(move)
                            board.push(move)
                        except:
                            break
                    print(pgn_game, file=f)
                    print("", file=f)
        print(f"💾 Evaluation results saved to {self.results_dir}")
    def _print_evaluation_summary(self, results: Dict[str, EvaluationResults]):
        print("\n" + "="*60)
        print("🏆 EVALUATION SUMMARY")
        print("="*60)
        for test_name, result in results.items():
            print(f"\n📊 {result.test_name}")
            print(f"   Opponent: {result.opponent}")
            print(f"   Games: {result.total_games}")
            print(f"   Record: {result.wins}W-{result.losses}L-{result.draws}D")
            print(f"   Win Rate: {result.win_rate:.1%}")
            print(f"   Estimated Elo Diff: {result.estimated_elo_diff:+.0f}")
            if result.estimated_elo_diff > 0:
                print(f"   📈 Model is stronger!")
            elif result.estimated_elo_diff < -50:
                print(f"   📉 Model needs improvement")
            else:
                print(f"   ⚖️  Roughly equal strength")
        print("\n🎯 Overall Assessment:")
        avg_elo_diff = np.mean([r.estimated_elo_diff for r in results.values()])
        print(f"   Average Elo Difference: {avg_elo_diff:+.0f}")
        if avg_elo_diff > 100:
            print("   🌟 Excellent performance!")
        elif avg_elo_diff > 50:
            print("   ✅ Good performance")
        elif avg_elo_diff > 0:
            print("   📈 Slight improvement")
        else:
            print("   ⚠️  Needs more training")
def run_quick_evaluation(model_path: str):
    print("🚀 Quick Model Evaluation")
    print("=" * 30)
    try:
        evaluator = ModelEvaluator(model_path)
        print("Testing against random player...")
        class RandomPlayer:
            @staticmethod
            def get_move(board):
                legal_moves = list(board.legal_moves)
                return random.choice(legal_moves) if legal_moves else chess.Move.null()
        wins = 0
        games = 5
        for i in range(games):
            board = chess.Board()
            our_turn = i % 2 == 0
            while not board.is_game_over() and len(board.move_stack) < 100:
                if (board.turn == chess.WHITE) == our_turn:
                    move = evaluator.mcts.get_best_move(board, temperature=0.0)
                else:
                    move = RandomPlayer.get_move(board)
                if move == chess.Move.null():
                    break
                board.push(move)
            result = board.result()
            if (result == "1-0" and our_turn) or (result == "0-1" and not our_turn):
                wins += 1
            print(f"Game {i+1}: {result} ({'Win' if ((result == '1-0' and our_turn) or (result == '0-1' and not our_turn)) else 'Loss/Draw'})")
        print(f"\nScore vs Random: {wins}/{games} ({wins/games:.1%})")
        if wins >= games * 0.8:
            print("✅ Model appears to be working well!")
        else:
            print("⚠️  Model may need more training")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
if __name__ == "__main__":
    print("🧪 Model Evaluation and Testing System")
    print("=" * 50)
    model_path = "models/test_model.pt"
    if os.path.exists(model_path):
        run_quick_evaluation(model_path)
    else:
        print("⚠️  No model found for evaluation")
        print("Train a model first using the self-play training system")
    print("\n✅ Evaluation system ready!")
    print("\nFeatures:")
    print("- Test against external engines (Stockfish)")
    print("- Compare with previous model versions")
    print("- Elo rating estimation")
    print("- Comprehensive game analysis")
    print("- PGN export for detailed review")