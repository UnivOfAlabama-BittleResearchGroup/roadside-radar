import torch

def check_gpu_available():
    return torch.cuda.is_available()
