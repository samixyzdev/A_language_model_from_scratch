import torch
from einops import einsum
from scripts.softmax import softmax

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, masking: torch.Tensor = None) -> torch.Tensor:
    d_k = query.shape[-1]
    dot_product = einsum(query, key, "... n d_k, ... m d_k -> ... n m") / torch.sqrt(torch.tensor(d_k))
    if masking is not None:
        dot_product = torch.where(masking, dot_product, -torch.inf)
    attention_weights = softmax(dot_product, dim_i = -1)
    return einsum(attention_weights, value, "... n m, ... m d_v -> ... n d_v")
