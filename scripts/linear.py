import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty((in_features, out_features), device = device, dtype = dtype))
        self.init_weight()

    def init_weight(self):
        mean = 0
        var = 2 / (self.in_features + self.out_features)
        std = torch.sqrt(torch.tensor(var))
        init.trunc_normal_(self.weight, mean = mean, std = std, a = -3, b = 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight
