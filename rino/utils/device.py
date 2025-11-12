import torch


def get_available_device() -> torch.device:
    """Get the available device for PyTorch.
    Priority: CUDA -> MPS -> CPU.

    Returns:
        The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
