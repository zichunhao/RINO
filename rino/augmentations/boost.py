import torch
from .utils import normalize_axis
import sys

eps = sys.float_info.epsilon


def get_gamma(beta: torch.Tensor) -> torch.Tensor:
    """
    Get the Lorentz factor gamma for boosts specified by beta.

    Args:
        beta: The velocity v / c, shape (N, 3) or (3,).
    Returns:
        gamma: The Lorentz factor, shape (N,) or scalar.
    Raises:
        ValueError: If any beta magnitude >= 1.
    """
    beta_mag = torch.norm(beta, dim=-1)
    if torch.any(beta_mag >= 1):
        raise ValueError(
            f"We require that 0 <= beta < 1 for all inputs. Found max |beta| = {beta_mag.max().item()}"
        )

    return 1 / torch.sqrt(1 - torch.sum(beta**2, dim=-1))


def get_boost_matrix(betas: torch.Tensor) -> torch.Tensor:
    """
    Get the Lorentz boost matrices for boosts specified by betas.

    Args:
        betas: The velocities v / c, shape (N, 3).

    Returns:
        A tensor of shape (N, 4, 4) representing the Lorentz boost matrices.
    """

    beta = torch.norm(betas, dim=1)
    gamma = get_gamma(betas)

    beta_x, beta_y, beta_z = betas.unbind(1)

    gamma_00 = gamma
    gamma_11 = 1 + (gamma - 1) * beta_x**2 / (beta + eps) ** 2
    gamma_22 = 1 + (gamma - 1) * beta_y**2 / (beta + eps) ** 2
    gamma_33 = 1 + (gamma - 1) * beta_z**2 / (beta + eps) ** 2
    gamma_01 = -gamma * beta_x
    gamma_02 = -gamma * beta_y
    gamma_03 = -gamma * beta_z
    gamma_12 = (gamma - 1) * beta_x * beta_y / (beta + eps) ** 2
    gamma_13 = (gamma - 1) * beta_x * beta_z / (beta + eps) ** 2
    gamma_23 = (gamma - 1) * beta_y * beta_z / (beta + eps) ** 2

    # Stack the components
    boost_matrices = torch.stack(
        [
            torch.stack([gamma_00, gamma_01, gamma_02, gamma_03], dim=1),
            torch.stack([gamma_01, gamma_11, gamma_12, gamma_13], dim=1),
            torch.stack([gamma_02, gamma_12, gamma_22, gamma_23], dim=1),
            torch.stack([gamma_03, gamma_13, gamma_23, gamma_33], dim=1),
        ],
        dim=1,
    )

    return boost_matrices


def get_random_boost_matrices(
    N: int,
    sigma: float = 0.1,
    beta_min: float = -0.99,
    beta_max: float = 0.99,
    sample_mode: str = "gamma",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Get random boost matrices.

    Args:
        N: Number of random boost matrices to generate.
        eps: A small value to numerically stabilize the computation. Default is 1e-6.
        sigma: Standard deviation for the random distribution. Default is 0.1.
            If sigma=0, returns the identity matrix.
        device: The device to put the tensors on. If None, uses the default device.
        dtype: The data type of the tensors. If None, uses the default dtype.

    Returns:
        Random boost matrices of shape (N, 4, 4).
    """
    if device is None:
        device = torch.get_default_device()
    if dtype is None:
        dtype = torch.get_default_dtype()

    if sigma < 0:
        raise ValueError(f"Expected sigma >= 0. Got: sigma={sigma}")
    if sigma == 0:
        return torch.eye(4, device=device, dtype=dtype).repeat(N, 1, 1)

    # uniform random directions
    n = torch.rand(N, 3, device=device, dtype=dtype)
    n_hat = n / (torch.norm(n, dim=1, keepdim=True) + eps)
    # random beta values
    beta_mag = sample_betas(
        N=N,
        sigma=sigma,
        sample_mode=sample_mode,
        beta_min=beta_min,
        beta_max=beta_max,
        forward_only=False,
        dtype=dtype,
        device=device,
    )
    betas = beta_mag.unsqueeze(1) * n_hat

    return get_boost_matrix(betas)


def get_boost_matrix_axis(beta: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Get the Lorentz boost matrix for a boost specified by beta along the axis.

    Args:
        beta: The velocity magnitude v/c with shape (N,), must be < 1
        axis: The axis to boost along with shape (N, 4). The first (temporal) component
              is ignored, and only the last three components (spatial) are used to define
              the boost direction. The spatial components will be normalized if not already.

    Returns:
        A tensor of shape (N, 4, 4) representing the Lorentz boost matrix.
    """
    # Normalize the axis (ignoring temporal component) using the imported function
    n_hat = normalize_axis(axis)[..., 1:]  # Only keep spatial components

    # final beta vectors
    betas = beta.unsqueeze(1) * n_hat  # Shape: (N, 3)

    # Use the existing get_boost_matrix function to compute the result
    return get_boost_matrix(betas)


def get_random_boost_matrices_axis(
    axis: torch.Tensor,
    sigma: float = 0.1,
    beta_min: float = -0.99,
    beta_max: float = 0.99,
    R_min: float | None = None,
    R_max: float | None = None,
    sample_mode: str = "beta",
    forward_only: bool = False,
) -> torch.Tensor:
    """
    Get random boost matrices along the axis.

    Args:
        axis: The axis to boost along with shape (N, 4). The first (temporal) component
              is energy (E), and the last three components (px,py,pz) define the
              boost direction.
        sigma: Standard deviation for the random distribution. Default is 0.1.
            If sigma=0, returns the identity matrix.
        beta_min: Minimum beta magnitude. Default is -0.99.
        beta_max: Maximum beta magnitude. Default is 0.99.
        R_min: Minimum allowed jet radius. If None, beta_min is set to -1.
        R_max: Maximum allowed jet radius. If None, beta_max is set to 1.
        sample_mode: The mode to sample the beta values. Choices: 'gamma' or 'beta'.
            If 'gamma', samples gamma values from 1 + |N(0, sigma)| and then computes beta.
            If 'beta', samples beta values from |N(0, sigma)|.
            Default is 'beta'.
        forward_only: If True, only return forward transformations (beta > 0) when sampling gamma.
            If sampling beta, this parameter is ignored.
            Default is False.

    Returns:
        Random boost matrices of shape (N, 4, 4).
    """
    N = axis.shape[0]
    if sigma < 0:
        raise ValueError(f"Expected sigma >= 0. Got: sigma={sigma}")
    elif sigma == 0:
        return torch.eye(4, device=axis.device, dtype=axis.dtype).expand(N, 4, 4)

    beta_min_dR, beta_max_dR = constrain_dR(jet_p4=axis, R_min=R_min, R_max=R_max)

    beta_min_final = torch.maximum(
        torch.tensor(beta_min, device=axis.device, dtype=axis.dtype), beta_min_dR
    )
    beta_min_final = torch.clamp(beta_min_final, min=-1 + eps, max=1 - eps)

    beta_max_final = torch.minimum(
        torch.tensor(beta_max, device=axis.device, dtype=axis.dtype), beta_max_dR
    )
    beta_max_final = torch.clamp(beta_max_final, min=-1 + eps, max=1 - eps)

    # Do not boost if beta_min > beta_max
    is_valid = beta_min_final <= beta_max_final
    beta_max_final[~is_valid] = eps
    beta_min_final[~is_valid] = -eps

    beta = sample_betas(
        N=N,
        sigma=sigma,
        sample_mode=sample_mode,
        dtype=axis.dtype,
        device=axis.device,
        beta_min=beta_min_final,
        beta_max=beta_max_final,
        forward_only=forward_only,
    )

    return get_boost_matrix_axis(beta, axis)


def sample_betas(
    N: int,
    sigma: float,
    sample_mode: str = "beta",
    beta_min: float | torch.Tensor = -0.99,
    beta_max: float | torch.Tensor = 0.99,
    forward_only: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Sample beta values for Lorentz boosts.

    Args:
        N: Number of beta values to sample.
        sigma: Standard deviation for the random distribution.
        sample_mode: The mode to sample the beta values. Choices: 'gamma' or 'beta'.
            If 'gamma', samples gamma values from 1 + |N(0, sigma)| and then computes beta (with 50% probability of sign flip).
            If 'beta', samples beta values from |N(0, sigma)|.
            Default is 'beta'.
        beta_min: Minimum beta magnitude. Default is -0.99.
        beta_max: Maximum beta magnitude. Default is 0.99.
        dtype: The data type of the tensors. If None, uses the default dtype.
        device: The device to put the tensors on. If None, uses the default device.
        forward_only: If True, only return forward transformations (beta > 0) when sampling gamma.
            If sampling beta, this parameter is ignored.
            Default is False.

    Returns:
        A tensor of shape (N,) containing the beta values.
    """
    if device is None:
        device = torch.get_default_device()
    if dtype is None:
        dtype = torch.get_default_dtype()

    sample_mode = sample_mode.lower()
    if sample_mode not in ["gamma", "beta"]:
        raise ValueError(
            f"Expected sample_mode to be 'gamma' or 'beta'. Got: {sample_mode}"
        )

    if sample_mode == "gamma":
        gamma = 1.0 + sigma * torch.abs(torch.randn(N, device=device, dtype=dtype))
        # boosting jets in the forward direction = boosting the lab frame backwards
        beta = -torch.sqrt(1 - 1 / gamma**2)
        if not forward_only:
            # flip beta sign with 50% probability
            flip = torch.rand(N, device=device, dtype=dtype) < 0.5
            beta[flip] *= -1
    else:
        beta = sigma * torch.randn(N, device=device, dtype=dtype)
    
    beta = torch.clamp(beta, min=beta_min, max=beta_max)

    return beta


def constrain_dR(
    jet_p4: torch.Tensor, R_min: float | None = None, R_max: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constrain boost parameters (beta) to keep jet radius within [R_min, R_max].
    Uses the approximation R ~ 2M/pT where M is jet mass and pT is transverse momentum.

    Note that pT' = γ pT |β E/|p3| + 1|.
    We are not boosting the jet to reverse direction, so
    pT' = γ pT (β E / |p3| + 1). Then,
    the constraint R_min <= R' <= R_max can be rewritten as:
    ω_min <= γ(βρ + 1) <= ω_max
    where ω = R/R', ρ = E/|p3|, and γ = 1/sqrt(1-β²)

    Args:
        jet_p4: Jet four-momenta, shape (N, 4) where components are (E, px, py, pz)
        R_min: Minimum allowed jet radius. If None, beta_min is set to -1.
        R_max: Maximum allowed jet radius. If None, beta_max is set to 1.

    Returns:
        beta_min: Lower bound on boost parameter β of each jet
        beta_max: Upper bound on boost parameter β of each jet
    """

    def get_beta_sol(omega: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Solve γ(βρ + 1) = ω for β"""
        numer = -rho + omega * torch.sqrt(rho**2 + omega**2 - 1 + eps)
        denom = rho**2 + omega**2 + eps
        return numer / denom

    # Extract kinematic variables
    E = jet_p4[..., 0]
    p3 = jet_p4[..., 1:]
    p3_norm = torch.norm(p3, dim=-1)
    pT = torch.sqrt(jet_p4[..., 1] ** 2 + jet_p4[..., 2] ** 2)

    # Calculate mass and rho = E/|p3|
    M = torch.sqrt(E**2 - p3_norm**2)
    rho = E / (p3_norm + eps)

    # Current jet radius and target ratios
    R = 2 * M / (pT + eps)

    # Get the effective omega bounds
    if R_max is not None:
        omega_min = R / R_max
        beta_min = get_beta_sol(omega_min, rho)
    else:
        beta_min = (-1 + eps) * torch.ones_like(R)

    if R_min is not None:
        omega_max = R / R_min
        beta_max = get_beta_sol(omega_max, rho)
    else:
        beta_max = (1 - eps) * torch.ones_like(R)

    return beta_min, beta_max
