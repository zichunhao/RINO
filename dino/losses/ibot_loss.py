"""
iBOT Loss.

Config example
--------------
.. code-block:: yaml

    ibot:
      weight: 1.0
      weight_warmup:          # optional
        start_value: 0.0
        start_epoch: 0
        end_epoch: 10
        warmup_scheduler: cosine
      mask_ratio: 0.2
      teacher_temp: 0.07
      temp_warmup:            # optional
        start_value: 0.04
        start_epoch: 0
        end_epoch: 20
        warmup_scheduler: cosine
      student_temp: 0.10
      enable_sinkhorn_knopp: true
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from .warmup_schedule import WarmupSchedule
from utils.logger import LOGGER

# try:
#     from xformers.ops import cross_entropy

#     def lossfunc(t: torch.Tensor, s: torch.Tensor, temp: float) -> torch.Tensor:
#         s = s.float()
#         t = t.float()
#         if s.ndim == 2:
#             return -cross_entropy(
#                 s.unsqueeze(0), t.unsqueeze(0), temp, bw_inplace=True
#             ).squeeze(0)
#         return -cross_entropy(s, t, temp, bw_inplace=True)

# except ImportError:
#     LOGGER.info("xformers not available, using standard PyTorch cross-entropy for iBOT")

#     def lossfunc(t: torch.Tensor, s: torch.Tensor, temp: float) -> torch.Tensor:
#         return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)


def lossfunc(t: torch.Tensor, s: torch.Tensor, temp: float) -> torch.Tensor:
    return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)


class iBOTLoss(nn.Module):
    """iBOT masked-prediction loss with flexible weight and temperature schedules.

    Parameters
    ----------
    out_dim:
        Output dimensionality of the iBOT head.
    mask_ratio:
        Fraction of valid particles to mask per sample.
    weight:
        Target (final) weight for this loss term.
    weight_warmup:
        Optional dict controlling how ``weight`` is warmed up.
        Keys: ``start_value``, ``start_epoch``, ``end_epoch``,
        ``warmup_scheduler`` ("cosine" | "linear").
    teacher_temp:
        Target (final) teacher temperature.
    temp_warmup:
        Optional dict controlling how ``teacher_temp`` is warmed up.
        Same keys as ``weight_warmup``.
    student_temp:
        Student temperature (fixed).
    center_momentum:
        EMA momentum for the centering buffer.
    enable_sinkhorn_knopp:
        Use Sinkhorn-Knopp instead of softmax + centering for the teacher.
    sinkhorn_knopp_niters:
        Number of Sinkhorn-Knopp iterations.
    """

    def __init__(
        self,
        out_dim: int,
        mask_ratio: float = 0.3,
        weight: float = 1.0,
        weight_warmup: dict | None = None,
        teacher_temp: float = 0.07,
        temp_warmup: dict | None = None,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        enable_sinkhorn_knopp: bool = False,
        sinkhorn_knopp_niters: int = 3,
        mask_method: Literal["iter", "vector"] = "vector",
    ) -> None:
        super().__init__()

        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.mask_ratio = mask_ratio
        if not enable_sinkhorn_knopp:
            LOGGER.warning(
                "iBOTLoss: enable_sinkhorn_knopp=False — the softmax+centering path's "
                "center EMA is not all-reduced in distributed training. "
                "Forcing enable_sinkhorn_knopp=True."
            )
        self.enable_sinkhorn_knopp = True
        self.sinkhorn_knopp_niters = sinkhorn_knopp_niters
        self.mask_vectorized = "vector" in mask_method.lower()

        # center shape: (1, 1, out_dim) - broadcasts over (batch, particles, dim)
        self.register_buffer("center", torch.zeros(1, 1, out_dim))
        self.updated = True
        self.async_batch_center: torch.Tensor | None = None

        # --- schedules ---------------------------------------------------
        self._weight_schedule = WarmupSchedule.from_config(weight, weight_warmup)
        self._temp_schedule = WarmupSchedule.from_config(teacher_temp, temp_warmup)

        self.current_epoch: int = 0
        self.current_step: int = 0

        self._log_schedule("weight", self._weight_schedule)
        self._log_schedule("teacher_temp", self._temp_schedule)

    # ------------------------------------------------------------------
    # schedule helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_schedule(name: str, schedule: WarmupSchedule) -> None:
        if schedule._start_value != schedule.end_value:
            unit = "steps" if schedule.mode == "step" else "epochs"
            LOGGER.info(
                f"iBOTLoss {name} warmup: {schedule._start_value} â†’ {schedule.end_value} "
                f"({unit} {schedule._start_idx}-{schedule._end_idx}, "
                f"{schedule.warmup_scheduler})"
            )

    def _counter_for(self, schedule: WarmupSchedule) -> int:
        """Return the appropriate counter (step or epoch) for *schedule*."""
        return self.current_step if schedule.mode == "step" else self.current_epoch

    def get_current_weight(self) -> float:
        return self._weight_schedule.get_value(self._counter_for(self._weight_schedule))

    def get_current_teacher_temp(self) -> float:
        return self._temp_schedule.get_value(self._counter_for(self._temp_schedule))

    def get_next_teacher_temp(self) -> float:
        return self._temp_schedule.get_value(self._counter_for(self._temp_schedule) + 1)

    def step_epoch(self) -> None:
        """Advance epoch counter (call once per epoch, after the epoch)."""
        self.current_epoch += 1

    def step_step(self) -> None:
        """Advance internal step counter (call once per optimizer step)."""
        self.current_step += 1

    def resume_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        LOGGER.info(
            f"iBOTLoss resumed at epoch {epoch}: "
            f"weight={self.get_current_weight():.4f}, "
            f"teacher_temp={self.get_current_teacher_temp():.4f}"
        )

    def resume_step(self, step: int) -> None:
        """Resume training from *step*."""
        self.current_step = step

    # ------------------------------------------------------------------
    # particle masking
    # ------------------------------------------------------------------
    def create_particle_mask(
        self, mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if self.mask_vectorized:
            return self._create_particle_mask_vector(mask, device)
        else:
            return self._create_particle_mask_iter(mask, device)

    def _create_particle_mask_iter(
        self, mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Randomly mask a fraction of valid particles per sample.

        Parameters
        ----------
        mask:
            Boolean validity mask - shape ``(batch_size, num_particles)``.
        device:
            Target device for the output tensor.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape ``(batch_size, num_particles)``; ``True``
            means the particle is masked (to be predicted).
        """
        batch_size, num_particles = mask.shape
        particle_mask = torch.zeros_like(mask, dtype=torch.bool, device=device)

        for i in range(batch_size):
            valid_indices = mask[i].nonzero(as_tuple=True)[0]
            num_valid = len(valid_indices)
            # Need at least 2 valid particles to mask any (always keep ≥1 unmasked).
            if num_valid >= 2:
                num_to_mask = max(1, min(num_valid - 1, int(num_valid * self.mask_ratio)))
                chosen = valid_indices[
                    torch.randperm(num_valid, device=device)[:num_to_mask]
                ]
                particle_mask[i, chosen] = True

        return particle_mask

    def _create_particle_mask_vector(
        self, mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        _, N = mask.shape
        valid_counts = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        valid_long = valid_counts.long()
        # Mask between 1 and (valid - 1) so at least one unmasked particle remains.
        # When valid_counts <= 1, the upper bound is 0 → no masking for that sample.
        num_to_mask = (valid_counts * self.mask_ratio).long()
        num_to_mask = torch.minimum(
            num_to_mask.clamp(min=1),
            (valid_long - 1).clamp(min=0),
        )  # (B, 1)

        # Random scores only for valid positions; invalid → -inf so they rank last
        rand = torch.where(
            mask,
            torch.rand_like(mask, dtype=torch.float32),
            torch.full_like(mask, float("-inf"), dtype=torch.float32),
        )

        # 1. Single argsort: get indices of the largest random values
        sorted_idx = rand.argsort(dim=1, descending=True)  # (B, N)

        # 2. Create a boolean mask in the "sorted" domain
        # cols shape (1, N) compares against num_to_mask shape (B, 1) -> broadcasts to (B, N)
        cols = torch.arange(N, device=device).unsqueeze(0)
        mask_in_sorted = cols < num_to_mask

        # 3. Scatter the True values back to their original tensor positions
        particle_mask = torch.zeros_like(mask)
        particle_mask.scatter_(dim=1, index=sorted_idx, src=mask_in_sorted)

        return particle_mask & mask  # never mask padding

    # ------------------------------------------------------------------
    # teacher normalization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens: torch.Tensor) -> None:
        """Accumulate an async center update from teacher tokens.

        Parameters
        ----------
        teacher_patch_tokens:
            Shape ``(batch_size, num_particles, out_dim)``.
        """
        self.updated = False
        self.async_batch_center = torch.sum(
            teacher_patch_tokens.mean(1), dim=0, keepdim=True
        )

    @torch.no_grad()
    def apply_center_update(self) -> None:
        """Flush the accumulated center update (EMA step)."""
        if not self.updated:
            # In a distributed run you would all-reduce async_batch_center here.
            _t = self.async_batch_center  # type: ignore[assignment]
            self.center = self.center * self.center_momentum + _t.unsqueeze(0) * (
                1.0 - self.center_momentum
            )
            self.updated = True

    @torch.no_grad()
    def softmax_center_teacher(
        self, teacher_patch_tokens: torch.Tensor
    ) -> torch.Tensor:
        self.apply_center_update()
        current_temp = self.get_current_teacher_temp()
        return F.softmax((teacher_patch_tokens - self.center) / current_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self, teacher_output: torch.Tensor, n_masked_patches: int
    ) -> torch.Tensor:
        current_temp = self.get_current_teacher_temp()
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / current_temp).t()  # [K, B]
        B = n_masked_patches
        K = Q.shape[0]

        Q = Q / Q.sum()

        for _ in range(self.sinkhorn_knopp_niters):
            row_sums = Q.sum(dim=1, keepdim=True)
            row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
            Q = Q / row_sums / K

            col_sums = Q.sum(dim=0, keepdim=True)
            col_sums = torch.where(col_sums == 0, torch.ones_like(col_sums), col_sums)
            Q = Q / col_sums / B

        return (Q * B).t()

    # ------------------------------------------------------------------
    # forward (masked)
    # ------------------------------------------------------------------

    def forward_masked(
        self,
        student_particle_predictions: torch.Tensor,
        teacher_particle_predictions: torch.Tensor,
        particle_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        return_components: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Compute iBOT loss using only masked particles (recommended path).

        Parameters
        ----------
        student_particle_predictions:
            Shape ``(batch_size, num_particles, out_dim)``.
        teacher_particle_predictions:
            Shape ``(batch_size, num_particles, out_dim)``.
        particle_mask:
            Boolean mask - which particles are masked for prediction.
            Shape ``(batch_size, num_particles)``.
        valid_mask:
            Boolean mask - which particles are real (not padding).
            Shape ``(batch_size, num_particles)``.
        return_components:
            If True, also return a dict of scalar components.
        """
        combined_mask = valid_mask & particle_mask

        if not combined_mask.any():
            result = torch.tensor(0.0, device=student_particle_predictions.device)
            if return_components:
                return result, {"ibot_loss": 0.0, "masked_ratio": 0.0}
            return result

        batch_size = student_particle_predictions.shape[0]
        n_masked_patches = int(combined_mask.sum().item())

        # Extract only masked positions
        student_masked = student_particle_predictions[combined_mask]  # [M, D]
        teacher_masked = teacher_particle_predictions[combined_mask]  # [M, D]

        # Teacher normalization
        if self.enable_sinkhorn_knopp:
            teacher_out = self.sinkhorn_knopp_teacher(teacher_masked, n_masked_patches)
        else:
            self.update_center(teacher_particle_predictions)
            teacher_out = (
                self.softmax_center_teacher(teacher_masked.unsqueeze(0).unsqueeze(0))
                .squeeze(0)
                .squeeze(0)
            )

        # Per-particle loss
        loss = lossfunc(teacher_out, student_masked, self.student_temp)

        # Per-sample weighting: 1 / num_masked_in_sample
        masks_weight = (
            (1.0 / combined_mask.sum(-1).clamp(min=1.0))
            .unsqueeze(-1)
            .expand_as(combined_mask)[combined_mask]
        )
        loss = -(loss * masks_weight).sum() / batch_size

        weighted_loss = loss * self.get_current_weight()

        if return_components:
            num_valid = int(valid_mask.sum().item())
            masked_ratio = n_masked_patches / num_valid if num_valid > 0 else 0.0
            return weighted_loss, {
                "ibot_loss": loss.detach().cpu().item(),
                "masked_ratio": masked_ratio,
            }
        return weighted_loss

    # ------------------------------------------------------------------
    # forward (non-masked, kept for compatibility)
    # ------------------------------------------------------------------

    def forward(
        self,
        student_particle_predictions: torch.Tensor,
        teacher_particle_predictions: torch.Tensor,
        particle_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        return_components: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Compute iBOT loss over all particles (non-masked path, for compatibility).

        Loss is computed for all positions but only masked+valid particles
        contribute to the average.
        """
        combined_mask = valid_mask & particle_mask

        if not combined_mask.any():
            result = torch.tensor(0.0, device=student_particle_predictions.device)
            if return_components:
                return result, {"ibot_loss": 0.0}
            return result

        self.update_center(teacher_particle_predictions)
        teacher_out = self.softmax_center_teacher(teacher_particle_predictions)

        loss = torch.sum(
            teacher_out
            * F.log_softmax(student_particle_predictions / self.student_temp, dim=-1),
            dim=-1,
        )
        loss = torch.sum(loss * combined_mask.float(), dim=-1) / combined_mask.sum(
            dim=-1
        ).clamp(min=1.0)
        loss = -loss.mean()

        weighted_loss = loss * self.get_current_weight()

        if return_components:
            return weighted_loss, {"ibot_loss": loss.detach().cpu().item()}
        return weighted_loss
