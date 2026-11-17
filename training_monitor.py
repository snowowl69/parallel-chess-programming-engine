import os
import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
@dataclass
class TrainingMetrics:
    iteration: int
    timestamp: datetime
    games_played: int
    total_positions: int
    average_game_length: float
    game_results: Dict[str, int]
    total_loss: float
    value_loss: float
    policy_loss: float
    learning_rate: float
    iteration_time: float
    games_per_hour: float
    positions_per_second: float
    eval_score: Optional[float] = None
    eval_games_won: Optional[int] = None
    eval_games_total: Optional[int] = None
class TrainingDatabase:
    def __init__(self, db_path: str = "training_history.db"):
        self.db_path = db_path
        self._init_database()
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_iterations (
                    iteration INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    games_played INTEGER,
                    total_positions INTEGER,
                    average_game_length REAL,
                    white_wins INTEGER,
                    black_wins INTEGER,
                    draws INTEGER,
                    total_loss REAL,
                    value_loss REAL,
                    policy_loss REAL,
                    learning_rate REAL,
                    iteration_time REAL,
                    games_per_hour REAL,
                    positions_per_second REAL,
                    eval_score REAL,
                    eval_games_won INTEGER,
                    eval_games_total INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_results (
                    game_id TEXT PRIMARY KEY,
                    iteration INTEGER,
                    timestamp TEXT,
                    result TEXT,
                    game_length INTEGER,
                    total_time REAL,
                    white_player TEXT,
                    black_player TEXT,
                    pgn_data TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    iteration INTEGER,
                    timestamp TEXT,
                    model_path TEXT,
                    model_size_mb REAL,
                    validation_score REAL
                )
            """)
            conn.commit()
    
    def save_metrics(self, metrics: TrainingMetrics):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO training_iterations
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.iteration,
                metrics.timestamp.isoformat(),
                metrics.games_played,
                metrics.total_positions,
                metrics.average_game_length,
                metrics.game_results.get('white', 0),
                metrics.game_results.get('black', 0),
                metrics.game_results.get('draw', 0),
                metrics.total_loss,
                metrics.value_loss,
                metrics.policy_loss,
                metrics.learning_rate,
                metrics.iteration_time,
                metrics.games_per_hour,
                metrics.positions_per_second,
                metrics.eval_score,
                metrics.eval_games_won,
                metrics.eval_games_total
            ))
            conn.commit()

    def save_iteration_metrics(self, metrics: TrainingMetrics):
        """Alias for save_metrics to match pipeline usage."""
        self.save_metrics(metrics)
    
    def get_all_metrics(self) -> List[TrainingMetrics]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM training_iterations ORDER BY iteration")
            rows = cursor.fetchall()
            return [self._row_to_metrics(row) for row in rows]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total_iterations,
                    SUM(games_played) as total_games,
                    SUM(total_positions) as total_positions,
                    AVG(total_loss) as avg_loss,
                    MIN(total_loss) as best_loss,
                    SUM(iteration_time) as total_time_hours
                FROM training_iterations
            """)
            row = cursor.fetchone()
            return {
                'total_iterations': row[0] or 0,
                'total_games': row[1] or 0,
                'total_positions': row[2] or 0,
                'avg_loss': row[3] or 0,
                'best_loss': row[4] or 0,
                'total_time_hours': (row[5] or 0) / 3600
            }
    
    def _row_to_metrics(self, row) -> TrainingMetrics:
        return TrainingMetrics(
            iteration=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            games_played=row[2],
            total_positions=row[3],
            average_game_length=row[4],
            game_results={'white': row[5], 'black': row[6], 'draw': row[7]},
            total_loss=row[8],
            value_loss=row[9],
            policy_loss=row[10],
            learning_rate=row[11],
            iteration_time=row[12],
            games_per_hour=row[13],
            positions_per_second=row[14],
            eval_score=row[15],
            eval_games_won=row[16],
            eval_games_total=row[17]
        )

def generate_html_report(summary: Dict[str, Any], metrics_list: List[TrainingMetrics]) -> str:
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Chess AI Training Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px;
                 background-color: #ecf0f1; border-radius: 5px; }}
        .plot {{ text-align: center; margin: 30px 0; }}
        img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Chess AI Training Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <h2>📊 Training Summary</h2>
    <div class="metric">
        <strong>Total Iterations:</strong><br>
        {summary.get('total_iterations', 0)}
    </div>
    <div class="metric">
        <strong>Total Games:</strong><br>
        {summary.get('total_games', 0):,}
    </div>
    <div class="metric">
        <strong>Total Positions:</strong><br>
        {summary.get('total_positions', 0):,}
    </div>
    <div class="metric">
        <strong>Best Loss:</strong><br>
        {summary.get('best_loss', 0):.4f}
    </div>
    <div class="metric">
        <strong>Training Time:</strong><br>
        {summary.get('total_time_hours', 0):.1f} hours
    </div>
    <h2>📈 Training Progress</h2>
    <div class="plot">
        <h3>Training Overview</h3>
        <img src="plots/overview.png" alt="Training Overview">
    </div>
    <div class="plot">
        <h3>Loss Convergence</h3>
        <img src="plots/loss_convergence.png" alt="Loss Convergence">
    </div>
    <div class="plot">
        <h3>Training Efficiency</h3>
        <img src="plots/efficiency.png" alt="Training Efficiency">
    </div>
    <h2>📋 Detailed Metrics</h2>
    <table>
        <tr>
            <th>Iteration</th>
            <th>Games</th>
            <th>Total Loss</th>
            <th>Value Loss</th>
            <th>Policy Loss</th>
            <th>Avg Game Length</th>
            <th>Time (s)</th>
        </tr>
        {table_rows}
    </table>
    <footer style="margin-top: 50px; text-align: center; color: #666;">
        <p>Generated by Chess AI Training Monitor</p>
    </footer>
</body>
</html>"""
    
    table_rows = ""
    for metrics in metrics_list[-20:]:  # Last 20 iterations
        table_rows += f"""
        <tr>
            <td>{metrics.iteration}</td>
            <td>{metrics.games_played}</td>
            <td>{metrics.total_loss:.4f}</td>
            <td>{metrics.value_loss:.4f}</td>
            <td>{metrics.policy_loss:.4f}</td>
            <td>{metrics.average_game_length:.1f}</td>
            <td>{metrics.iteration_time:.1f}</td>
        </tr>"""
    
    return html_template.format(table_rows=table_rows, summary=summary)


class TrainingVisualizer:
    def __init__(self, db: TrainingDatabase):
        self.db = db
        self.fig = None
        self.axes = None
    
    def create_dashboard(self, save_path: str = "plots"):
        os.makedirs(save_path, exist_ok=True)
        metrics = self.db.get_all_metrics()
        
        if not metrics:
            print("⚠️ No training data available")
            return
        
        # Overview plot
        self._plot_overview(metrics, os.path.join(save_path, "overview.png"))
        
        # Loss convergence
        self._plot_loss_convergence(metrics, os.path.join(save_path, "loss_convergence.png"))
        
        # Efficiency metrics
        self._plot_efficiency(metrics, os.path.join(save_path, "efficiency.png"))
        
        print(f"✅ Dashboard saved to {save_path}/")
    
    def _plot_overview(self, metrics: List[TrainingMetrics], filepath: str):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Overview', fontsize=16, fontweight='bold')
        
        iterations = [m.iteration for m in metrics]
        
        # Games played
        axes[0, 0].plot(iterations, [m.games_played for m in metrics], 'b-', linewidth=2)
        axes[0, 0].set_title('Games per Iteration')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Games')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Total loss
        axes[0, 1].plot(iterations, [m.total_loss for m in metrics], 'r-', linewidth=2)
        axes[0, 1].set_title('Total Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Game results
        white_wins = [m.game_results.get('white', 0) for m in metrics]
        black_wins = [m.game_results.get('black', 0) for m in metrics]
        draws = [m.game_results.get('draw', 0) for m in metrics]
        
        axes[1, 0].bar(iterations, white_wins, label='White', alpha=0.7)
        axes[1, 0].bar(iterations, black_wins, label='Black', alpha=0.7, bottom=white_wins)
        axes[1, 0].bar(iterations, draws, label='Draw', alpha=0.7, 
                      bottom=[w+b for w, b in zip(white_wins, black_wins)])
        axes[1, 0].set_title('Game Results Distribution')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Games')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Positions per second
        axes[1, 1].plot(iterations, [m.positions_per_second for m in metrics], 'g-', linewidth=2)
        axes[1, 1].set_title('Training Speed')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Positions/Second')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_loss_convergence(self, metrics: List[TrainingMetrics], filepath: str):
        fig, ax = plt.subplots(figsize=(12, 6))
        iterations = [m.iteration for m in metrics]
        
        ax.plot(iterations, [m.total_loss for m in metrics], label='Total Loss', linewidth=2)
        ax.plot(iterations, [m.value_loss for m in metrics], label='Value Loss', linewidth=2)
        ax.plot(iterations, [m.policy_loss for m in metrics], label='Policy Loss', linewidth=2)
        
        ax.set_title('Loss Convergence', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency(self, metrics: List[TrainingMetrics], filepath: str):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        iterations = [m.iteration for m in metrics]
        
        # Games per hour
        axes[0].plot(iterations, [m.games_per_hour for m in metrics], 'purple', linewidth=2)
        axes[0].set_title('Games per Hour')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Games/Hour')
        axes[0].grid(True, alpha=0.3)
        
        # Average game length
        axes[1].plot(iterations, [m.average_game_length for m in metrics], 'orange', linewidth=2)
        axes[1].set_title('Average Game Length')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Moves')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Example usage
    db = TrainingDatabase("training_history.db")
    visualizer = TrainingVisualizer(db)
    visualizer.create_dashboard()
    
    summary = db.get_summary_stats()
    metrics = db.get_all_metrics()
    
    html_report = generate_html_report(summary, metrics)
    with open("training_report.html", "w") as f:
        f.write(html_report)
    
    print("✅ Training report generated successfully!")
