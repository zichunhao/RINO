import torch


def check_bf16_support(device: torch.device) -> bool:
    """Check whether bfloat16 is supported on the given device.

    - CUDA: requires compute capability >= 8.0 (Ampere+) and PyTorch bf16 support.
    - CPU: supported on modern PyTorch builds.
    - MPS / other: not supported.
    """
    if device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            return False
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16, device=device)
            return True
        except Exception:
            return False
    elif device.type == "cpu":
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16, device=device)
            return True
        except Exception:
            return False
    else:
        return False


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
