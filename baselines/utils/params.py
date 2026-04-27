import torch
from prettytable import PrettyTable

def count_parameters(
    model: torch.nn.Module
) -> tuple[PrettyTable, int]:
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Source: https://stackoverflow.com/a/62508086
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    return table, total_params
