import torch
from utils.logger import LOGGER


def random_masking(
    particles: torch.Tensor,
    mask: torch.Tensor | None = None,
    mask_prob: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random masking to the particles.

    Args:
        particles (torch.Tensor): The particles to be masked.
        mask (torch.Tensor, optional): The mask to be applied. Defaults to None.
            Note: 1 indicates the particle is masked, 0 indicates the particle is not masked.
            This is the PyTorch convention in transformers.
        mask_prob (float, optional): The probability of masking. Defaults to 0.1.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the masked particles and the updated mask.

    Raises:
        ValueError: If the provided mask shape doesn't match the particles shape.
    """
    if mask_prob <= 0:
        return particles, mask
    if mask_prob >= 1:
        LOGGER.warning("Mask probability is 1. All particles will be masked.")
        return torch.zeros_like(particles), torch.ones_like(particles).bool()

    num_jets, num_particles, _ = particles.shape
    if (mask is not None) and (mask.shape != (num_jets, num_particles)):
        raise ValueError(
            f"Mask shape must match particles shape. "
            f"Found: {mask.shape}, Expected: {(num_jets, num_particles)}"
        )

    # Generate random mask (1 for masked, 0 for not masked)
    rand_mask = (
        torch.rand(num_jets, num_particles, device=particles.device) <= mask_prob
    ).bool()

    if mask is not None:
        # Combine with existing mask (use OR operation since 1 means masked)
        mask = torch.logical_or(mask, rand_mask)
        mask[:, 0] = mask[:, 0] & ~torch.all(
            mask, dim=1
        )  # Ensure at least one particle is unmasked
    else:
        mask = rand_mask

    # Apply mask (invert the mask for multiplication since 0 should keep the particle)
    particles = particles * (~mask.bool()).unsqueeze(-1).float()

    return particles, mask
