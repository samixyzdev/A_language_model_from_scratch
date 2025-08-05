import torch
import torch.nn as nn
import torch.nn.init as init

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones((d_model, ), device = device, dtype = dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x.pow(2), -1, keepdim = True) + self.eps)
        normalized_x = x / rms
        result = normalized_x * self.weight
        return result.to(in_dtype)