import torch
import torch.nn as nn


@torch.no_grad()
def update_momentum_encoder(student: nn.Module, teacher: nn.Module, m: float) -> None:
    """Update teacher parameters as EMA of student: param_t = m * param_t + (1 - m) * param_s."""
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(m).add_(param_s.data, alpha=(1 - m))
