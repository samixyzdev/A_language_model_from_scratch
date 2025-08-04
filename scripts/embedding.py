import torch
import torch.nn as nn
import torch.nn.init as init

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device = device, dtype = dtype))
        init.xavier_uniform_(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

"""
super().__init__() in Python is a method call that invokes the constructor of the parent class.
In the context of PyTorch, when a class inherits from torch.nn.Module
"""
"""
torch.nn.init.xavier_uniform_: This function modifies the tensor you pass to it directly, without returning a new one. 
It's the standard practice for initializing an existing nn.Parameter.

torch.nn.init.xavier_uniform: This function returns a new tensor with the values initialized according to the Xavier uniform distribution. 
It does not modify any input tensor.
"""
"""
Advanced indexing (self.weight[token_ids]) in PyTorch is an optimized lookup operation. 
It uses the integer token_ids as indices to directly retrieve corresponding vectors from the self.weight matrix. 
This is far more efficient than the conceptual alternative of one-hot encoding followed by matrix multiplication, which would involve many wasteful zero multiplications.
"""