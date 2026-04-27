import torch

from .boost import get_random_boost_matrices, get_random_boost_matrices_axis
from .rotation import get_random_rotation_matrices, get_rotation_matrix_axis


def get_random_lorentz_matrices(
    N: int,
    sigma: float = 0.3,
    sample_mode: str = "beta",
    beta_min: float = -0.99,
    beta_max: float = 0.99,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Generate random proper, orthochronous Lorentz transformation matrices
    using R' @ B @ R

    Args:
        N: The number of random matrices to generate.
        eps: A small value to numerically stabilize the computation. Default is 1e-6.
        sigma: Standard deviation for the random distribution of the Lorentz factor gamma. Default is 0.3.
            If sigma=0, just returns random rotation matrices.
        device: The device to put the tensors on. If None, uses the default device.
        dtype: The data type of the tensors. If None, uses the default dtype.

    Returns:
        A tuple of two tensors:
            - A tensor of shape (N, 4, 4) representing the Lorentz transformation matrices.
    """
    if sigma == 0:
        # return only one set of random rotation matrices
        return get_random_rotation_matrices(N, device=device, dtype=dtype)
    R1 = get_random_rotation_matrices(N, device=device, dtype=dtype)
    B = get_random_boost_matrices(
        N,
        sigma=sigma,
        sample_mode=sample_mode,
        beta_min=beta_min,
        beta_max=beta_max,
        device=device,
        dtype=dtype,
    )
    R2 = get_random_rotation_matrices(N, device=device, dtype=dtype)

    return torch.bmm(torch.bmm(R2, B), R1)


def get_random_lorentz_matrices_axis(
    axis: torch.Tensor,
    sigma: float = 0.3,
    sample_mode: str = "beta",
    beta_min: float = -0.99,
    beta_max: float = 0.99,
    R_min: float | None = None,
    R_max: float | None = None,
    forward_only: bool = False,
) -> torch.Tensor:
    """
    Generate random proper, orthochronous Lorentz transformation matrices
    along a specified axis using B @ R decomposition (or R @ B, since they commute
    when using the same axis).

    Args:
        axis: The axis tensor with shape (N, 4). The first (temporal) component
              is ignored, and only the last three components (spatial) are used to define
              both the boost and rotation direction. The spatial components will be
              normalized if not already.
        sigma: Standard deviation for the random distribution of the Lorentz factor gamma.
               Default is 0.3. If sigma=0, just returns random rotation matrices.
        sample_mode: The mode to sample the beta values. Choices: 'gamma' or 'beta'.
            If 'gamma', samples gamma values from 1 + |N(0, sigma)| and then computes beta (with 50% probability of sign flip).
            If 'beta', samples beta values from |N(0, sigma)|.
            Default is 'beta'.
        beta_min: The minimum value for the beta parameter. Default is -0.99.
        beta_max: The maximum value for the beta parameter. Default is 0.99.
        R_min: Minimum allowed jet radius. If None, beta_min is set to -1.
        R_max: Maximum allowed jet radius. If None, beta_max is set to 1.
        forward_only: If True, only return forward transformations (beta > 0) when sampling gamma.
            If sampling beta, this parameter is ignored.
            Default is False.

    Returns:
        A tensor of shape (N, 4, 4) representing the Lorentz transformation matrices.
        Each matrix represents a combination of a rotation and a boost along the
        specified axis.
    """
    N = axis.shape[0]
    device, dtype = axis.device, axis.dtype

    if sigma == 0:
        # Return only random rotation matrices along the axis
        return get_rotation_matrix_axis(
            theta=2 * torch.pi * torch.rand(N, device=device, dtype=dtype), axis=axis
        )

    # Generate rotation and boost matrices along the same axis
    R = get_rotation_matrix_axis(
        theta=2 * torch.pi * torch.rand(N, device=device, dtype=dtype), axis=axis
    )
    B = get_random_boost_matrices_axis(
        axis,
        sigma=sigma,
        beta_min=beta_min,
        beta_max=beta_max,
        R_min=R_min,
        R_max=R_max,
        sample_mode=sample_mode,
        forward_only=forward_only,
    )

    # Combine the transformations: B @ R (or R @ B, they commute)
    return torch.bmm(B, R)
