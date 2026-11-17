import chess
import numpy as np
import math
import random
from typing import Dict, List, Optional, Tuple
from neural_network import ChessNet, BoardEncoder, MoveEncoder
import torch
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, parent_move=None, prior_prob=0.0):
        self.board = board.copy()
        self.parent = parent
        self.parent_move = parent_move
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.is_expanded = False
        self.value_estimate = 0.0
        self.policy_probs = {}
    def is_leaf(self) -> bool:
        return not self.is_expanded
    def is_terminal(self) -> bool:
        return self.board.is_game_over()
    def get_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    def select_child(self, c_puct=1.0) -> 'MCTSNode':
        best_score = -float('inf')
        best_child = None
        sqrt_parent_visits = math.sqrt(self.visit_count)
        for move, child in self.children.items():
            q_value = child.get_value()
            u_value = c_puct * child.prior_prob * sqrt_parent_visits / (1 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    def expand(self, value_estimate: float, policy_probs: Dict[chess.Move, float]):
        self.is_expanded = True
        self.value_estimate = value_estimate
        self.policy_probs = policy_probs
        for move in self.board.legal_moves:
            prior_prob = policy_probs.get(move, 1e-8)
            child_board = self.board.copy()
            child_board.push(move)
            child_node = MCTSNode(
                board=child_board,
                parent=self,
                parent_move=move,
                prior_prob=prior_prob
            )
            self.children[move] = child_node
    def backup(self, value: float):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)
    def get_action_probs(self, temperature=1.0) -> Dict[chess.Move, float]:
        if not self.children:
            return {}
        if temperature == 0:
            best_move = max(self.children.keys(),
                          key=lambda m: self.children[m].visit_count)
            return {move: (1.0 if move == best_move else 0.0)
                   for move in self.children.keys()}
        visit_counts = np.array([self.children[move].visit_count
                               for move in self.children.keys()])
        if temperature != 1.0:
            visit_counts = visit_counts ** (1.0 / temperature)
        probs = visit_counts / visit_counts.sum()
        return {move: prob for move, prob in
               zip(self.children.keys(), probs)}
class MCTS:
    def __init__(self, neural_network: ChessNet, device='cpu'):
        self.neural_network = neural_network
        self.device = device
        self.move_encoder = MoveEncoder()
        self.c_puct = 1.0
        self.num_simulations = 800
    def search(self, board: chess.Board, num_simulations: Optional[int] = None) -> MCTSNode:
        if num_simulations is None:
            num_simulations = self.num_simulations
        root = MCTSNode(board)
        for _ in range(num_simulations):
            self._simulate(root)
        return root
    def _simulate(self, root: MCTSNode):
        path = []
        current = root
        while not current.is_leaf() and not current.is_terminal():
            current = current.select_child(self.c_puct)
            path.append(current)
        if current.is_terminal():
            result = current.board.result()
            if result == "1-0":
                value = 1.0 if current.board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                value = -1.0 if current.board.turn == chess.WHITE else 1.0
            else:
                value = 0.0
        else:
            value, policy_probs = self._evaluate_position(current.board)
            if not current.is_terminal():
                current.expand(value, policy_probs)
        current.backup(value)
    def _evaluate_position(self, board: chess.Board) -> Tuple[float, Dict[chess.Move, float]]:
        board_tensor = BoardEncoder.board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value_logit, policy_logits = self.neural_network(board_tensor)
            value = value_logit.item()
            policy_probs_array = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        policy_probs = {}
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            move_index = self.move_encoder.move_to_index(move)
            if move_index < len(policy_probs_array):
                policy_probs[move] = policy_probs_array[move_index]
            else:
                policy_probs[move] = 1e-8
        total_prob = sum(policy_probs.values())
        if total_prob > 0:
            policy_probs = {move: prob / total_prob
                          for move, prob in policy_probs.items()}
        else:
            uniform_prob = 1.0 / len(legal_moves)
            policy_probs = {move: uniform_prob for move in legal_moves}
        return value, policy_probs
    def get_best_move(self, board: chess.Board, temperature=0.0) -> chess.Move:
        root = self.search(board)
        if not root.children:
            legal_moves = list(board.legal_moves)
            return random.choice(legal_moves) if legal_moves else chess.Move.null()
        action_probs = root.get_action_probs(temperature)
        if temperature == 0.0:
            return max(action_probs.keys(), key=lambda m: action_probs[m])
        else:
            moves = list(action_probs.keys())
            probs = list(action_probs.values())
            return np.random.choice(moves, p=probs)
    def get_training_data(self, board: chess.Board) -> Tuple[torch.Tensor, np.ndarray]:
        root = self.search(board)
        position_tensor = BoardEncoder.board_to_tensor(board)
        action_probs_dict = root.get_action_probs(temperature=1.0)
        action_probs = np.zeros(4096)
        for move, prob in action_probs_dict.items():
            move_index = self.move_encoder.move_to_index(move)
            if move_index < 4096:
                action_probs[move_index] = prob
        return position_tensor, action_probs
class MCTSStats:
    def __init__(self):
        self.reset()
    def reset(self):
        self.total_simulations = 0
        self.total_time = 0.0
        self.positions_evaluated = 0
        self.average_depth = 0.0
    def update(self, simulations: int, time_taken: float, max_depth: int):
        self.total_simulations += simulations
        self.total_time += time_taken
        self.positions_evaluated += 1
        self.average_depth = ((self.average_depth * (self.positions_evaluated - 1) + max_depth)
                             / self.positions_evaluated)
    def get_summary(self) -> str:
        if self.positions_evaluated == 0:
            return "No MCTS data collected yet."
        avg_time = self.total_time / self.positions_evaluated
        avg_sims = self.total_simulations / self.positions_evaluated
        return f"""MCTS Performance Summary:
        - Positions evaluated: {self.positions_evaluated}
        - Average simulations per position: {avg_sims:.0f}
        - Average time per position: {avg_time:.2f}s
        - Average search depth: {self.average_depth:.1f}
        - Simulations per second: {avg_sims/avg_time:.0f}"""
if __name__ == "__main__":
    print("Testing MCTS...")
    from neural_network import create_network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = create_network(device)
    mcts = MCTS(network, device)
    mcts.num_simulations = 100
    board = chess.Board()
    print(f"Starting position: {board.fen()}")
    best_move = mcts.get_best_move(board)
    print(f"MCTS best move: {best_move}")
    position_tensor, action_probs = mcts.get_training_data(board)
    print(f"Position tensor shape: {position_tensor.shape}")
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Non-zero action probabilities: {np.count_nonzero(action_probs)}")
    print("MCTS implementation ready!")