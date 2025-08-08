import torch
import torch.nn as nn
import torch.nn.init as init
from scripts.scaled_dot_product_attention import scaled_dot_product_attention
from scripts.rope import RotaryPositionalEmbedding
from einops import rearrange, einsum

class Multihead_Self_Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: int = 10000, max_seq_len: int = 1024, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.W_q = nn.Parameter(torch.Tensor(d_model, d_model, device = device))
        self.W_k = nn.Parameter(torch.Tensor(d_model, d_model, device = device))
        self.W_v = nn.Parameter(torch.Tensor(d_model, d_model, device = device))
        self.W_o = nn.Parameter(torch.Tensor(d_model, d_model, device = device))
        self.init_weights_()
        self.RoPE = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device = self.device)

    def init_weights_(self):
        init.kaiming_uniform_(self.W_q)
        init.kaiming_uniform_(self.W_k)
        init.kaiming_uniform_(self.W_v)
        init.kaiming_uniform_(self.W_o)

    def forward_without_rope(self, x: torch.Tensor) -> torch.Tensor:
        Q = einsum(x, self.W_q, "... s d_in, d_out d_in-> ... s d_out")
        K = einsum(x, self.W_k, "... s d_in, d_out d_in -> ... s d_out")
        V = einsum(x, self.W_v, "... s d_in, d_out d_in -> ... s d_out")
        Q = rearrange(Q, "... s (h d_k) -> ... h s d_k", h = self.num_heads)
        K = rearrange(K, "... s (h d_k) -> ... h s d_k", h = self.num_heads)
        V = rearrange(V, "... s (h d_k) -> ... h s d_k", h = self.num_heads)
        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype = bool, device = self.device), diagonal = 0)
        Multiheads = scaled_dot_product_attention(Q, K, V, causal_mask)
        Multiheads = rearrange(Multiheads, "... h s d_k -> ... s (h d_k)")
        return einsum(Multiheads, self.W_o, "... s d_in, d_out d_in -> ... s d_out")
    
    def forward_w_rope(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        Q = einsum(x, self.W_q, "... s d_in, d_out d_in -> ... s d_out")
        K = einsum(x, self.W_k, "... s d_in, d_out d_in -> ... s d_out")
        V = einsum(x, self.W_v, "... s d_in, d_out d_in -> ... s d_out")
        Q = rearrange(Q, "... s (h d_k) -> ... h s d_k", h = self.num_heads)
        K = rearrange(K, "... s (h d_k) -> ... h s d_k", h = self.num_heads)
        V = rearrange(V, "... s (h d_k) -> ... h s d_k", h = self.num_heads)
        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype = bool, device = self.device), diagonal = 0)
        rotated_Q = self.RoPE.forward(Q, token_positions)
        rotated_K = self.RoPE.forward(K, token_positions)
        Multiheads = scaled_dot_product_attention(rotated_Q, rotated_K, V, causal_mask)
        Multiheads = rearrange(Multiheads, "... h s d_k -> ... s (h d_k)")
        return einsum(Multiheads, self.W_o, "... s d_in, d_out d_in -> ... s d_out")
# --- IMPORTANT NOTE on Weight Matrix Convention ---
#
# 1. The PyTorch Convention: Standard PyTorch layers (like nn.Linear) and most pre-trained
#    models store weight matrices with the shape: (output_features, input_features).
#
# 2. The Required Operation: To correctly use these weights, the forward pass operation
#    must be mathematically equivalent to `input @ weight.T` (input multiplied by the
#    weight's transpose).
#
# 3. The `einsum` Implementation: The string `"... s d_in, d_out d_in -> ... s d_out"`
#    correctly implements this `input @ weight.T` operation. It maps the `d_in` from the
#    input `x` to the second dimension of the weight matrix (`d_in` in the `d_out, d_in`
#    part), which correctly performs the linear transformation.
#
# Using the more intuitive `"... s d_in, d_in d_out -> ... s d_out"` would assume a weight
# shape of (input_features, output_features), leading to incorrect results when using
# standard PyTorch weights. This was the root cause of the previous test failures.