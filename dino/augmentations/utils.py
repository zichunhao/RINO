import torch
import sys

eps = sys.float_info.epsilon


def normalize_axis(axis: torch.Tensor) -> torch.Tensor:
    """
    Normalize the spatial components of 4D vectors, ignoring the temporal (first) component.

    Args:
        axis: Vectors to normalize with shape (N, 4), where the first component is temporal
              and the last three components represent the spatial direction.

    Returns:
        Vectors with shape (N, 4) where the spatial components are normalized and the
        temporal component is set to 0.
    """
    # Extract spatial components (last 3 elements)
    spatial = axis[..., 1:]
    # Normalize only the spatial components
    norm = torch.norm(spatial, dim=1, keepdim=True) + eps
    normalized_spatial = spatial / norm

    # Create output tensor with zero temporal component
    result = torch.zeros_like(axis)
    result[..., 1:] = normalized_spatial
    return result
