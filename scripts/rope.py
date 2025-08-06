import torch
import torch.nn as nn
from einops import rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        freq_indices = torch.arange(0, d_k // 2, dtype = torch.float32,device = device)
        freqs = 1 / (theta ** (2 * freq_indices / d_k))
        positions = torch.arange(max_seq_len, dtype = torch.float32, device = device)
        angles = torch.outer(positions, freqs)
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        self.register_buffer('cos_vals', cos_vals, persistent = False)
        self.register_buffer('sin_vals', sin_vals, persistent = False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        #*batch_dim, seq_len, d_k =original_shape
        cos = self.cos_vals[token_positions]
        sin = self.sin_vals[token_positions]
        #x_pairs = x.view(*batch_dim,seq_len, d_k // 2, 2)
        x_pairs = rearrange(x, '... seq_len (d_pair pair) -> ... seq_len d_pair pair', pair = 2)
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        #rotated_pairs = torch.stack([rotated_x1, rotated_x2], dim = -1)
        #rotated_x = rotated_pairs.view(*original_shape)
        rotated_x = rearrange([rotated_x1, rotated_x2], 'pair ... seq_len d_pair -> ... seq_len (d_pair pair)')
        return rotated_x