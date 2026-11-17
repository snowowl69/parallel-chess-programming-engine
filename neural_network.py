import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from typing import Tuple, List
class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=10, num_hidden=256):
        super(ChessNet, self).__init__()
        self.input_layer = nn.Conv2d(18, num_hidden, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(num_hidden)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_hidden) for _ in range(num_res_blocks)
        ])
        self.value_conv = nn.Conv2d(num_hidden, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.policy_conv = nn.Conv2d(num_hidden, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(128, 4096)
    def forward(self, x):
        x = F.relu(self.input_bn(self.input_layer(x)))
        for res_block in self.res_blocks:
            x = res_block(x)
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)
        return value, policy
class ResidualBlock(nn.Module):
    def __init__(self, num_hidden):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
class BoardEncoder:
    @staticmethod
    def board_to_tensor(board: chess.Board) -> torch.Tensor:
        tensor = torch.zeros(18, 8, 8, dtype=torch.float32)
        piece_to_plane = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                plane = piece_to_plane[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane += 6
                tensor[plane, row, col] = 1.0
        
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[15, :, :] = 1.0
        
        if board.turn == chess.WHITE:
            tensor[16, :, :] = 1.0
            
        tensor[17, :, :] = board.halfmove_clock / 100.0

        return tensor
    @staticmethod
    def board_to_batch(boards: List[chess.Board]) -> torch.Tensor:
        batch = torch.stack([
            BoardEncoder.board_to_tensor(board) for board in boards
        ])
        return batch
class MoveEncoder:
    def __init__(self):
        self.move_to_index_dict = {}
        self.index_to_move_dict = {}
        self._create_move_mapping()
    def _create_move_mapping(self):
        index = 0
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq != to_sq:
                    self.move_to_index_dict[(from_sq, to_sq)] = index
                    self.index_to_move_dict[index] = (from_sq, to_sq)
                    index += 1
    def move_to_index(self, move: chess.Move) -> int:
        key = (move.from_square, move.to_square)
        return self.move_to_index_dict.get(key, 0)
    def index_to_move(self, index: int) -> chess.Move:
        if index in self.index_to_move_dict:
            from_sq, to_sq = self.index_to_move_dict[index]
            return chess.Move(from_sq, to_sq)
        return chess.Move.null()
class TrainingData:
    def __init__(self):
        self.positions = []
        self.policies = []
        self.values = []
    def add_sample(self, position: torch.Tensor, policy: np.ndarray, value: float):
        self.positions.append(position)
        self.policies.append(policy)
        self.values.append(value)
    def clear(self):
        self.positions.clear()
        self.policies.clear()
        self.values.clear()
    def get_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.positions) < batch_size:
            batch_size = len(self.positions)
        indices = np.random.choice(len(self.positions), batch_size, replace=False)
        batch_positions = torch.stack([self.positions[i] for i in indices])
        batch_policies = torch.tensor([self.policies[i] for i in indices], dtype=torch.float32)
        batch_values = torch.tensor([self.values[i] for i in indices], dtype=torch.float32)
        return batch_positions, batch_policies, batch_values
def create_network(device='cpu'):
    network = ChessNet(num_res_blocks=10, num_hidden=256)
    network.to(device)
    return network
def save_model(model, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_res_blocks': 10,
            'num_hidden': 256
        }
    }, filepath)
    print(f"Model saved to {filepath}")
def load_model(filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    model = ChessNet(
        num_res_blocks=checkpoint['model_config']['num_res_blocks'],
        num_hidden=checkpoint['model_config']['num_hidden']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model
if __name__ == "__main__":
    print("Testing ChessNet...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = create_network(device)
    batch_size = 4
    test_input = torch.randn(batch_size, 18, 8, 8).to(device)
    with torch.no_grad():
        value_out, policy_out = net(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Value output shape: {value_out.shape}")
    print(f"Policy output shape: {policy_out.shape}")
    print(f"Value range: {value_out.min().item():.3f} to {value_out.max().item():.3f}")
    print("\nTesting board encoding...")
    board = chess.Board()
    encoded = BoardEncoder.board_to_tensor(board)
    print(f"Encoded board shape: {encoded.shape}")
    print(f"Non-zero elements: {torch.count_nonzero(encoded)}")
    print("\nChessNet ready for training!")