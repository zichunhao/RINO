import torch
from .utils import normalize_axis


def get_rotation_matrix(thetas: torch.Tensor) -> torch.Tensor:
    """
    Get the 4D rotation matrices for rotations around x, y, and z axes.

    Args:
        thetas: Rotation angles (in radians) with shape (N, 3) for x, y, and z axes.

    Returns:
        A tensor of shape (N, 4, 4) representing the rotation matrices.
    """
    theta_x, theta_y, theta_z = thetas.unbind(-1)
    Rx = get_rotation_matrix_x(theta_x)
    Ry = get_rotation_matrix_y(theta_y)
    Rz = get_rotation_matrix_z(theta_z)

    return torch.bmm(torch.bmm(Rz, Ry), Rx)


def get_rotation_matrix_x(theta: torch.Tensor) -> torch.Tensor:
    """
    Get the 4D rotation matrices for rotations around the x-axis.

    Args:
        theta: The rotation angles in radians, shape (N,).

    Returns:
        A tensor of shape (N, 4, 4) representing the rotation matrices.
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)

    return torch.stack(
        [
            torch.stack([ones, zeros, zeros, zeros], dim=1),
            torch.stack([zeros, ones, zeros, zeros], dim=1),
            torch.stack([zeros, zeros, c, -s], dim=1),
            torch.stack([zeros, zeros, s, c], dim=1),
        ],
        dim=1,
    )


def get_rotation_matrix_y(theta: torch.Tensor) -> torch.Tensor:
    """
    Get the 4D rotation matrices for rotations around the y-axis.

    Args:
        theta: The rotation angles in radians, shape (N,).

    Returns:
        A tensor of shape (N, 4, 4) representing the rotation matrices.
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)

    return torch.stack(
        [
            torch.stack([ones, zeros, zeros, zeros], dim=1),
            torch.stack([zeros, c, zeros, s], dim=1),
            torch.stack([zeros, zeros, ones, zeros], dim=1),
            torch.stack([zeros, -s, zeros, c], dim=1),
        ],
        dim=1,
    )


def get_rotation_matrix_z(theta: torch.Tensor) -> torch.Tensor:
    """
    Get the 4D rotation matrices for rotations around the z-axis.

    Args:
        theta: The rotation angles in radians, shape (N,).

    Returns:
        A tensor of shape (N, 4, 4) representing the rotation matrices.
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)

    return torch.stack(
        [
            torch.stack([ones, zeros, zeros, zeros], dim=1),
            torch.stack([zeros, c, -s, zeros], dim=1),
            torch.stack([zeros, s, c, zeros], dim=1),
            torch.stack([zeros, zeros, zeros, ones], dim=1),
        ],
        dim=1,
    )


def get_random_rotation_matrices(
    N: int, device: torch.device | None = None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    Get random rotation matrices.

    Args:
        N: Number of random rotation matrices to generate.
        device: The device to put the tensors on. If None, uses the default device.
        dtype: The data type of the tensors. If None, uses the default dtype.

    Returns:
        Random rotation matrices of shape (N, 4, 4).
    """
    if device is None:
        device = torch.get_default_device()
    if dtype is None:
        dtype = torch.get_default_dtype()

    # all directions -> Euler angles
    # [0, pi) for theta_x, [-pi, pi) for theta_y, [0, pi) for theta_z
    theta_x = torch.rand(N, device=device, dtype=dtype) * torch.pi
    theta_y = torch.pi - (torch.rand(N, device=device, dtype=dtype) * 2 * torch.pi)
    theta_z = torch.rand(N, device=device, dtype=dtype) * torch.pi

    thetas = torch.stack([theta_x, theta_y, theta_z], dim=1)
    return get_rotation_matrix(thetas)


def get_random_rotation_z_matrices(
    N: int, device: torch.device | None = None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    Get random rotation matrices about the z axis.

    Args:
        N: Number of random rotation matrices to generate.
        device: The device to put the tensors on. If None, uses the default device.
        dtype: The data type of the tensors. If None, uses the default dtype.

    Returns:
        Random rotation matrices of shape (N, 4, 4).
    """
    theta = torch.rand(N, device=device, dtype=dtype) * 2 * torch.pi
    return get_rotation_matrix_z(theta)


def get_rotation_matrix_axis(theta: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Get 4D rotation matrices for rotations around arbitrary spatial axes, ignoring
    the temporal component of the axis vectors.

    Args:
        theta: Rotation angles in radians with shape (N,)
        axis: Rotation axes with shape (N, 4). The first component (temporal) is ignored,
              and only the last three components (spatial) are used to define the rotation axis.
              The spatial components will be normalized if not already.

    Returns:
        Rotation matrices with shape (N, 4, 4) representing spatial rotations in 4D spacetime.
        These matrices will have 1 in the [0,0] position to preserve the temporal component,
        with the remaining 3x3 block representing the spatial rotation.
    """
    # Normalize the axis vectors (ignoring temporal component)
    u = normalize_axis(axis)

    # Get batch size
    N = theta.shape[0]

    # Get the spatial components of the normalized axis
    # Shape: (N, 3)
    u_spatial = u[..., 1:]

    # Expand dimensions for broadcasting
    # Shape: (N, 3, 1)
    u_spatial = u_spatial.unsqueeze(-1)
    # Shape: (N, 1, 3)
    u_spatial_T = u_spatial.transpose(-2, -1)

    # Construct the spatial cross-product matrix (antisymmetric matrix) for each axis
    # Shape: (N, 3, 3)
    K_spatial = torch.zeros((N, 3, 3), device=theta.device, dtype=theta.dtype)

    # Fill the spatial antisymmetric matrix
    K_spatial[..., 0, 1] = -u_spatial[:, 2, 0]  # xy plane
    K_spatial[..., 0, 2] = u_spatial[:, 1, 0]  # xz plane
    K_spatial[..., 1, 2] = -u_spatial[:, 0, 0]  # yz plane

    K_spatial[..., 1, 0] = u_spatial[:, 2, 0]
    K_spatial[..., 2, 0] = -u_spatial[:, 1, 0]
    K_spatial[..., 2, 1] = u_spatial[:, 0, 0]

    # Compute spatial rotation using Rodrigues formula
    sin_theta = torch.sin(theta).view(N, 1, 1)
    cos_theta = torch.cos(theta).view(N, 1, 1)

    # Shape: (N, 3, 3)
    outer_product_spatial = torch.matmul(u_spatial, u_spatial_T)
    I_spatial = torch.eye(3, device=theta.device, dtype=theta.dtype).expand(N, 3, 3)

    # Compute the spatial rotation block
    R_spatial = (
        I_spatial
        + sin_theta * K_spatial
        + (1 - cos_theta) * (outer_product_spatial - I_spatial)
    )

    # Create the full 4D rotation matrix
    R = torch.zeros((N, 4, 4), device=theta.device, dtype=theta.dtype)
    R[:, 0, 0] = 1.0  # Set temporal component to 1
    R[:, 1:, 1:] = R_spatial  # Set spatial block

    return R


def get_random_rotation_matrices_axis(axis: torch.Tensor) -> torch.Tensor:
    """
    Get random rotation matrices around arbitrary spatial axes.

    Args:
        axis: Rotation axes with shape (N, 4). The first component (temporal) is ignored,
            and only the last three components (spatial) are used to define the rotation axis.
            The spatial components will be normalized if not already.

    Returns:
        Rotation matrices with shape (N, 4, 4) representing spatial rotations in 4D spacetime.
        These matrices will have 1 in the [0,0] position to preserve the temporal component,
        with the remaining 3x3 block representing the spatial rotation.
    """
    N = axis.shape[0]
    theta = torch.rand(N, device=axis.device, dtype=axis.dtype) * 2 * torch.pi
    return get_rotation_matrix_axis(theta, axis)
