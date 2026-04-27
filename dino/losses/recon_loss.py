"""
Masked particle reconstruction loss (MAE-style).

Predicts raw particle features at masked positions from the backbone's
particle-level output. No teacher needed — the target is the input itself.

Config example
--------------
.. code-block:: yaml

    recon:
      weight: 1.0
      weight_warmup:          # optional
        start_value: 0.0
        start_epoch: 0
        end_epoch: 10
        warmup_scheduler: cosine
      mask_ratio: 0.25
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warmup_schedule import WarmupSchedule
from utils.logger import LOGGER


class ReconLoss(nn.Module):
    """Masked particle reconstruction loss with optional weight warmup.

    Parameters
    ----------
    mask_ratio:
        Fraction of valid particles to mask per sample.
    weight:
        Target (final) loss weight.
    weight_warmup:
        Optional dict controlling the weight schedule.
    """

    def __init__(
        self,
        mask_ratio: float = 0.25,
        weight: float = 1.0,
        weight_warmup: dict | None = None,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self._weight_schedule = WarmupSchedule.from_config(weight, weight_warmup)
        self.current_epoch: int = 0
        self.current_step: int = 0

        if self._weight_schedule._start_value != self._weight_schedule.end_value:
            LOGGER.info(
                f"ReconLoss weight warmup: {self._weight_schedule._start_value} -> "
                f"{self._weight_schedule.end_value} "
                f"(epochs {self._weight_schedule.start_epoch}-{self._weight_schedule.end_epoch}, "
                f"{self._weight_schedule.warmup_scheduler})"
            )

    # ------------------------------------------------------------------
    # schedule helpers
    # ------------------------------------------------------------------

    def _counter_for(self, schedule: WarmupSchedule) -> int:
        return self.current_step if schedule.mode == "step" else self.current_epoch

    def get_current_weight(self) -> float:
        return self._weight_schedule.get_value(self._counter_for(self._weight_schedule))

    def step_step(self) -> None:
        self.current_step += 1

    def step_epoch(self) -> None:
        self.current_epoch += 1

    def resume_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        LOGGER.info(
            f"ReconLoss resumed at epoch {epoch}: weight={self.get_current_weight():.4f}"
        )

    def resume_step(self, step: int) -> None:
        self.current_step = step

    # ------------------------------------------------------------------
    # particle masking (same as iBOT)
    # ------------------------------------------------------------------

    def create_particle_mask(
        self, mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Randomly mask a fraction of valid particles per sample.

        Returns a boolean mask where True = masked (to be predicted).
        """
        _, N = mask.shape
        valid_counts = mask.sum(dim=1, keepdim=True).float()
        num_to_mask = (valid_counts * self.mask_ratio).clamp(min=1).long()

        rand = torch.where(
            mask,
            torch.rand_like(mask, dtype=torch.float32),
            torch.full_like(mask, float("-inf"), dtype=torch.float32),
        )
        sorted_idx = rand.argsort(dim=1, descending=True)
        cols = torch.arange(N, device=device).unsqueeze(0)
        mask_in_sorted = cols < num_to_mask

        particle_mask = torch.zeros_like(mask)
        particle_mask.scatter_(dim=1, index=sorted_idx, src=mask_in_sorted)
        return particle_mask & mask

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        particle_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        return_components: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Compute MSE reconstruction loss on masked particles.

        Parameters
        ----------
        predicted:
            Decoder output, shape ``(B, N, part_dim)``.
        target:
            Original particle features, shape ``(B, N, part_dim)``.
        particle_mask:
            Boolean mask — True = masked (to be predicted).
        valid_mask:
            Boolean mask — True = real particle (not padding).
        """
        combined_mask = valid_mask & particle_mask

        if not combined_mask.any():
            result = torch.tensor(0.0, device=predicted.device)
            if return_components:
                return result, {"recon_loss": 0.0, "masked_ratio": 0.0}
            return result

        # MSE on masked positions only
        pred_masked = predicted[combined_mask]   # (M, part_dim)
        tgt_masked = target[combined_mask]       # (M, part_dim)
        raw_loss = F.mse_loss(pred_masked, tgt_masked)

        weighted_loss = raw_loss * self.get_current_weight()

        if return_components:
            num_valid = int(valid_mask.sum().item())
            num_masked = int(combined_mask.sum().item())
            masked_ratio = num_masked / num_valid if num_valid > 0 else 0.0
            return weighted_loss, {
                "recon_loss": raw_loss.detach().cpu().item(),
                "masked_ratio": masked_ratio,
            }
        return weighted_loss
