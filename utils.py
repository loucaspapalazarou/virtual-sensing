import torch

def preprocess_tensor(t: torch.Tensor, degrees_of_freedom: int) -> torch.Tensor:
    return t.view(-1, t.size(1) // degrees_of_freedom, degrees_of_freedom)