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
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        pass

"""
super().__init__() in Python is a method call that invokes the constructor of the parent class.
In the context of PyTorch, when a class inherits from torch.nn.Module
"""