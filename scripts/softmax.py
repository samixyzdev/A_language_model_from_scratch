import torch

def softmax(x: torch.Tensor, dim_i: int) -> torch.Tensor:
    largest_entry = torch.max(x, dim = dim_i, keepdim = True)[0]
    exp = torch.exp(x - largest_entry)
    sum_of_exp = torch.sum(exp, dim_i, keepdim = True)
    return  exp / sum_of_exp
"""
By adding [0] at the end of the torch.max call, you are telling PyTorch to use the maximum values tensor and not the entire tuple object. 
"""