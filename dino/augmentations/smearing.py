import torch


def smear(
    features: torch.Tensor,
    smear_factor: float,
    indices_ignore: list[int] | None = None,
    allow_sign_change: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Apply Gaussian smearing to features, with the option to skip specific feature indices.
    The result is p * (1 + ε), where ε ~ N(0, σ).

    Args:
        features: The features to smear, with shape [N, ..., D]
        smear_factor: The standard deviation of the Gaussian noise
        allow_sign_change: Whether to allow the smeared features to change sign.
            If False, 1 + ε is clamped to be eps. Defaults to True.
        eps: A small value to prevent the smeared features from being zero.
        indices_ignore: List of indices in the last dimension (D) where smearing should not be applied.
            These features will remain unchanged. Defaults to None.

    Returns:
        torch.Tensor: Selectively smeared features
    """
    # Generate noise of the same shape as input
    noise = torch.randn_like(features) * smear_factor
    factor = 1 + noise

    if not allow_sign_change:
        factor = torch.clamp_min(factor, eps) * torch.sign(factor)

    if indices_ignore is not None:
        # No smearing for specific features
        mask = torch.ones_like(factor)
        mask[..., indices_ignore] = 0

        factor = factor * mask + (1 - mask)

    return features * factor
