"""
Gram Loss Implementation based on Meta's DINOv3
Adapted for particle physics jet data

The Gram loss encourages the student to match the pairwise similarity structure
(correlation matrix) of the teacher's representations at the particle/patch level.

In DINOv3, by default:
- img_level=False: Gram matrix computed over entire batch of patches/particles
- This encourages learning cross-image/cross-jet particle relationships
"""

from __future__ import annotations

import math
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import LOGGER

from .warmup_schedule import WarmupSchedule


class _GramCore(nn.Module):
    """
    Implementation of the Gram loss from DINOv3.

    The Gram loss computes the MSE between the Gram matrices (pairwise similarities)
    of student and teacher particle-level feature representations.

    By default (img_level=False), computes Gram matrix over all particles in the batch,
    encouraging the model to learn cross-jet particle relationships.
    """

    def __init__(
        self,
        apply_norm: bool = True,
        img_level: bool = False,  # DINOv3 default: False (batch-level)
        remove_neg: bool = False,  # DINOv3 default: False
        remove_only_teacher_neg: bool = False,
        tokens_used: Literal["all", "masked", "unmasked"] = "all",
        total_weight: float = 1.0,
        warmup_epochs: int = 0,
        warmup_scheduler: Literal["linear", "cosine"] = "linear",
    ):
        """
        Args:
            apply_norm: Whether to L2-normalize features before computing Gram matrix.
                       When True, similarities are cosine similarities in [-1, 1].
                       DINOv3 default: True
            img_level: If True, compute Gram matrix per jet/image (B, N, dim).
                      If False (default), compute over entire batch (B*N, dim).
                      DINOv3 default: False
            remove_neg: If True, clamp negative similarities to 0 for both student and teacher.
                       DINOv3 default: False
            remove_only_teacher_neg: If True, only clamp teacher negatives and corresponding
                                    student negatives. Mutually exclusive with remove_neg.
                                    DINOv3 default: False
            tokens_used: Which tokens/particles to use for Gram loss computation.
                        - "all": Use all valid particles (default)
                        - "masked": Only use masked particles (from iBOT)
                        - "unmasked": Only use unmasked particles
            total_weight: Weight multiplier for the loss term. DINOv3 default: 1.0
            warmup_epochs: Number of epochs to warm up the loss weight from 0 to total_weight.
            warmup_scheduler: Type of warmup scheduler ('linear' or 'cosine').
        """
        super().__init__()

        # Loss function
        self.mse_loss = nn.MSELoss()

        # Parameters
        self.apply_norm = apply_norm
        self.img_level = img_level
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg
        self.tokens_used = tokens_used
        self.total_weight = total_weight

        # Validate mutually exclusive options
        if self.remove_neg and self.remove_only_teacher_neg:
            raise ValueError(
                "remove_neg and remove_only_teacher_neg are mutually exclusive. "
                "Set only one to True."
            )

        # Warmup setup
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        if warmup_epochs > 0:
            if warmup_scheduler == "linear":
                self.weight_schedule = torch.linspace(0.0, total_weight, warmup_epochs)
            elif warmup_scheduler == "cosine":
                iters = torch.arange(warmup_epochs)
                self.weight_schedule = (
                    0.5
                    * total_weight
                    * (1 - torch.cos(math.pi * iters / warmup_epochs))
                )
            else:
                raise ValueError(f"Unknown warmup scheduler: {warmup_scheduler}")
            LOGGER.info(
                f"Gram loss warmup enabled: 0 -> {total_weight} "
                f"over {warmup_epochs} epochs using {warmup_scheduler} schedule"
            )
        else:
            self.weight_schedule = None

    def get_current_weight(self) -> float:
        """Get the current loss weight based on warmup schedule."""
        if self.weight_schedule is None or self.current_epoch >= self.warmup_epochs:
            return self.total_weight
        else:
            return float(self.weight_schedule[self.current_epoch])

    def step_epoch(self):
        """Increment the epoch counter for weight warmup."""
        if self.weight_schedule is not None and self.current_epoch < self.warmup_epochs:
            current_weight = self.get_current_weight()
            LOGGER.info(
                f"Epoch {self.current_epoch}: Gram loss weight = {current_weight:.4f}"
            )
        self.current_epoch += 1

    def resume_epoch(self, epoch: int):
        """Resume from a specific epoch."""
        self.current_epoch = epoch
        if self.weight_schedule is not None and self.current_epoch < self.warmup_epochs:
            current_weight = self.get_current_weight()
            LOGGER.info(
                f"Resumed at epoch {self.current_epoch}: Gram loss weight = {current_weight:.4f}"
            )

    def forward(
        self,
        student_particle_features: torch.Tensor,
        teacher_particle_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        particle_mask: torch.Tensor | None = None,
        return_components: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the Gram loss between student and teacher particle-level features.

        Args:
            student_particle_features: Student particle features [B, N, D]
            teacher_particle_features: Teacher particle features [B, N, D]
            valid_mask: Boolean mask for valid particles [B, N].
                       True = valid particle, False = padding.
            particle_mask: Boolean mask for iBOT masked particles [B, N].
                          True = masked particle. Only used when tokens_used != "all".
            return_components: Whether to return loss components dict.

        Returns:
            loss or (loss, components_dict)
        """
        # Validate input dimensions for img_level mode
        if self.img_level:
            assert (
                len(student_particle_features.shape) == 3
            ), f"Expected 3D tensor (B, N, D), got {student_particle_features.shape}"
            assert (
                len(teacher_particle_features.shape) == 3
            ), f"Expected 3D tensor (B, N, D), got {teacher_particle_features.shape}"

        # Float casting
        student_feats = student_particle_features.float()
        teacher_feats = teacher_particle_features.float()

        # Filter particles based on tokens_used
        if self.tokens_used != "all" and valid_mask is not None:
            if particle_mask is None:
                LOGGER.warning(
                    f"tokens_used='{self.tokens_used}' but particle_mask is None. "
                    "Using all valid particles."
                )
                combined_mask = valid_mask
            else:
                if self.tokens_used == "masked":
                    # Only use particles that are both valid AND masked
                    combined_mask = valid_mask & particle_mask
                elif self.tokens_used == "unmasked":
                    # Only use particles that are valid AND NOT masked
                    combined_mask = valid_mask & ~particle_mask
                else:
                    raise ValueError(f"Unknown tokens_used: {self.tokens_used}")
        elif valid_mask is not None:
            combined_mask = valid_mask
        else:
            combined_mask = None

        # Extract valid particles if mask is provided
        if combined_mask is not None:
            # Flatten and select valid particles: (B, N, D) -> (num_valid, D)
            student_feats = student_feats[combined_mask]
            teacher_feats = teacher_feats[combined_mask]

            # Check if we have any valid particles
            if student_feats.shape[0] == 0:
                zero_loss = torch.tensor(0.0, device=student_particle_features.device)
                if return_components:
                    return zero_loss, {"gram_loss": 0.0}
                return zero_loss
        elif not self.img_level and len(teacher_feats.shape) == 3:
            # Flatten (B, N, D) into (B*N, D) for batch-level computation
            student_feats = student_feats.flatten(0, 1)
            teacher_feats = teacher_feats.flatten(0, 1)

        # Normalize features if requested
        if self.apply_norm:
            teacher_feats = F.normalize(teacher_feats, dim=-1)
            student_feats = F.normalize(student_feats, dim=-1)

        # Compute Gram matrices (pairwise similarities)
        # Shape: (num_particles, num_particles) or (B, N, N) if img_level=True
        teacher_gram = torch.matmul(teacher_feats, teacher_feats.transpose(-1, -2))
        student_gram = torch.matmul(student_feats, student_feats.transpose(-1, -2))

        # Handle negative values
        if self.remove_neg:
            teacher_gram = teacher_gram.clamp(min=0.0)
            student_gram = student_gram.clamp(min=0.0)
        elif self.remove_only_teacher_neg:
            # Create mask for where teacher is negative
            teacher_neg_mask = teacher_gram < 0
            teacher_gram = teacher_gram.clamp(min=0.0)
            # Zero out student values where teacher was negative
            student_gram = student_gram.masked_fill(teacher_neg_mask, 0.0)

        # Compute MSE loss
        loss = self.mse_loss(student_gram, teacher_gram)

        # Apply weight (with potential warmup)
        current_weight = self.get_current_weight()
        weighted_loss = loss * current_weight

        if return_components:
            components = {
                "gram_loss": loss.detach().cpu().item(),
            }
            return weighted_loss, components

        return weighted_loss


"""
Gram Loss - weight-warmup wrapper.

This file is a thin adapter around whatever existing GramLoss implementation
lives in your codebase.  It adds:

* ``weight`` / ``weight_warmup`` schedule (via ``WarmupSchedule``)
* ``step_epoch()`` / ``resume_epoch()`` for curriculum control
* ``get_current_weight()`` for transparent weight inspection

The underlying gram computation is unchanged; ``total_weight`` is no longer
accepted as a constructor argument - use ``weight`` instead.

Config example
--------------
.. code-block:: yaml

    gram:
      weight: 2.0
      weight_warmup:          # optional
        start_value: 0.0
        start_epoch: 0
        end_epoch: 10
        warmup_scheduler: cosine
      apply_norm: true
      img_level: false
      remove_neg: false
      remove_only_teacher_neg: false
      tokens_used: "all"
"""


class GramLoss(nn.Module):
    """Gram-matrix self-distillation loss with optional weight warmup.

    All keyword arguments other than ``weight`` and ``weight_warmup`` are
    forwarded verbatim to the underlying ``_GramLossCore`` (your existing
    implementation).  Rename the import below if your class has a different
    name.

    Parameters
    ----------
    weight:
        Target (final) weight for this loss term.
    weight_warmup:
        Optional dict controlling the weight schedule.
        Keys: ``start_value``, ``start_epoch``, ``end_epoch``,
        ``warmup_scheduler`` ("cosine" | "linear").
    **gram_kwargs:
        Forwarded to the underlying gram loss implementation
        (``apply_norm``, ``img_level``, ``remove_neg``, etc.)
    """

    def __init__(
        self,
        weight: float = 1.0,
        weight_warmup: dict | None = None,
        **gram_kwargs,
    ) -> None:
        super().__init__()

        # --- import underlying implementation ----------------------------
        # Force total_weight=1.0 so _GramCore never scales the loss itself;
        # all weighting is done here via _weight_schedule.
        gram_kwargs.pop("total_weight", None)
        self._core = _GramCore(total_weight=1.0, **gram_kwargs)

        # --- weight schedule ---------------------------------------------
        self._weight_schedule = WarmupSchedule.from_config(weight, weight_warmup)
        self.current_epoch: int = 0
        self.current_step: int = 0

        if self._weight_schedule._start_value != self._weight_schedule.end_value:
            LOGGER.info(
                f"GramLoss weight warmup: {self._weight_schedule._start_value} -> {self._weight_schedule.end_value} "
                f"(epochs {self._weight_schedule.start_epoch}-{self._weight_schedule.end_epoch}, "
                f"{self._weight_schedule.warmup_scheduler})"
            )

    # ------------------------------------------------------------------
    # schedule helpers
    # ------------------------------------------------------------------

    def _counter_for(self, schedule: WarmupSchedule) -> int:
        """Return step or epoch counter depending on schedule mode."""
        return self.current_step if schedule.mode == "step" else self.current_epoch

    def get_current_weight(self) -> float:
        return self._weight_schedule.get_value(self._counter_for(self._weight_schedule))

    def step_step(self) -> None:
        """Advance internal step counter (call once per optimizer step)."""
        self.current_step += 1

    def step_epoch(self) -> None:
        self.current_epoch += 1

    def resume_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        LOGGER.info(
            f"GramLoss resumed at epoch {epoch}: weight={self.get_current_weight():.4f}"
        )

    def resume_step(self, step: int) -> None:
        self.current_step = step

    # ------------------------------------------------------------------
    # forward - delegates to core, then applies current weight
    # ------------------------------------------------------------------

    def forward(
        self,
        student_particle_features: torch.Tensor,
        teacher_particle_features: torch.Tensor,
        valid_mask: torch.Tensor,
        particle_mask: torch.Tensor | None = None,
        return_components: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Compute the Gram loss.

        Parameters
        ----------
        student_particle_features, teacher_particle_features:
            Per-particle backbone features - shape ``(B, N, D)``.
        valid_mask:
            Boolean validity mask - shape ``(B, N)``.
        particle_mask:
            Optional iBOT-style particle mask (passed through to core if used).
        return_components:
            If True, also return a dict of scalar components.
        """
        result = self._core(
            student_particle_features=student_particle_features,
            teacher_particle_features=teacher_particle_features,
            valid_mask=valid_mask,
            particle_mask=particle_mask,
            return_components=return_components,
        )

        if return_components:
            raw_loss, components = result
        else:
            raw_loss = result
            components = {}

        weighted_loss = raw_loss * self.get_current_weight()

        if return_components:
            return weighted_loss, components
        return weighted_loss
