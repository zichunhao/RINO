"""
DINO Loss — refactored.

Changes vs original
-------------------
* ``total_weight`` / ``warmup_teacher_temp`` / ``warmup_teacher_temp_epochs`` /
  ``warmup_scheduler`` replaced by ``weight`` + optional ``weight_warmup`` block
  and ``teacher_temp`` + optional ``temp_warmup`` block (both driven by
  ``WarmupSchedule``).
* KoLeo regularization fully removed — now lives in ``KoLeoLoss``.
* ``get_current_weight()`` / ``get_current_teacher_temp()`` are the single
  source-of-truth for the effective values at the current epoch.
* ``step_epoch()`` / ``resume_epoch()`` advance **both** schedules.

Config example
--------------
.. code-block:: yaml

    dino:
      weight: 1.0
      weight_warmup:          # optional
        start_value: 0.0
        start_epoch: 0
        end_epoch: 10
        warmup_scheduler: cosine
      teacher_temp: 0.07
      temp_warmup:            # optional
        start_value: 0.04
        start_epoch: 0
        end_epoch: 20
        warmup_scheduler: cosine
      student_temp: 0.10
      enable_sinkhorn_knopp: true
      sinkhorn_knopp_niters: 3
      skip_same_view: true
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from .warmup_schedule import WarmupSchedule
from utils.logger import LOGGER


class DINOLoss(nn.Module):
    """DINO self-distillation loss with flexible weight and temperature schedules.

    Parameters
    ----------
    out_dim:
        Dimensionality of the DINO projection head output.
    num_local_views:
        Number of local (cropped) views produced by the augmenter.
    num_global_views:
        Number of global views produced by the augmenter.
    weight:
        Target (final) weight for this loss term.
    weight_warmup:
        Optional dict controlling how ``weight`` is warmed up over epochs.
        Keys: ``start_value``, ``start_epoch``, ``end_epoch``,
        ``warmup_scheduler`` ("cosine" | "linear").
    teacher_temp:
        Target (final) teacher softmax temperature.
    temp_warmup:
        Optional dict controlling how ``teacher_temp`` is warmed up.
        Same keys as ``weight_warmup``.
    student_temp:
        Student softmax temperature (fixed).
    center_momentum:
        EMA momentum for the teacher centering buffer.
    skip_same_view:
        If True, skip the loss term when student and teacher indices match.
    enable_sinkhorn_knopp:
        Use Sinkhorn-Knopp normalization instead of softmax + centering for
        the teacher.
    sinkhorn_knopp_niters:
        Number of Sinkhorn-Knopp iterations.
    """

    def __init__(
        self,
        out_dim: int,
        num_local_views: int = 2,
        num_global_views: int = 2,
        weight: float = 1.0,
        weight_warmup: dict | None = None,
        teacher_temp: float = 0.07,
        temp_warmup: dict | None = None,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        skip_same_view: bool = True,
        enable_sinkhorn_knopp: bool = False,
        sinkhorn_knopp_niters: int = 3,
    ) -> None:
        super().__init__()

        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.num_local_views = num_local_views
        self.num_global_views = num_global_views
        self.skip_same_view = skip_same_view
        self.enable_sinkhorn_knopp = enable_sinkhorn_knopp
        self.sinkhorn_knopp_niters = sinkhorn_knopp_niters

        self.register_buffer("center", torch.zeros(1, out_dim))
        self.training = False

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
                f"DINOLoss {name} warmup: {schedule._start_value} â†’ {schedule.end_value} "
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
        """Advance internal epoch counter (call once per epoch, after the epoch)."""
        self.current_epoch += 1

    def step_step(self) -> None:
        """Advance internal step counter (call once per optimizer step)."""
        self.current_step += 1

    def resume_epoch(self, epoch: int) -> None:
        """Resume training from *epoch*."""
        self.current_epoch = epoch
        LOGGER.info(
            f"DINOLoss resumed at epoch {epoch}: "
            f"weight={self.get_current_weight():.4f}, "
            f"teacher_temp={self.get_current_teacher_temp():.4f}"
        )

    def resume_step(self, step: int) -> None:
        """Resume training from *step*."""
        self.current_step = step

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def compute_sum_loss(
        self,
        student_out: tuple[torch.Tensor, ...],
        teacher_out: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=student_out[0].device)
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if self.skip_same_view and v == iq:
                    continue
                try:
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                except Exception as e:
                    LOGGER.error(
                        f"Error computing loss for teacher view {iq} and student view {v}: {e}"
                    )
                    LOGGER.error(
                        f"Teacher shape: {q.shape}, Student shape: {student_out[v].shape}"
                    )
                    raise
                total_loss = total_loss + loss.mean()
                n_loss_terms += 1

        return total_loss / n_loss_terms if n_loss_terms > 0 else total_loss

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_backbone_output: torch.Tensor | None = None,
        teacher_backbone_output: torch.Tensor | None = None,
        return_components: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Compute the DINO loss.

        Parameters
        ----------
        student_output:
            Concatenated student head outputs - shape ``(N * num_views, out_dim)``.
        teacher_output:
            Concatenated teacher head outputs - shape ``(N * num_global_views, out_dim)``.
        student_backbone_output:
            Unused here; kept for API symmetry with callers.
        teacher_backbone_output:
            Unused here; kept for API symmetry with callers.
        return_components:
            If True, also return a ``dict`` of scalar loss components.
        """
        # student: temperature scaling + chunk per view
        student_out = (student_output / self.student_temp).chunk(
            self.num_local_views + self.num_global_views
        )

        # teacher: normalize + chunk per global view
        if self.enable_sinkhorn_knopp:
            teacher_norm = self.sinkhorn_knopp_teacher(teacher_output)
        else:
            teacher_norm = self.softmax_center_teacher(teacher_output)
        teacher_out = teacher_norm.detach().chunk(self.num_global_views)

        raw_loss = self.compute_sum_loss(student_out, teacher_out)
        weighted_loss = raw_loss * self.get_current_weight()

        if return_components:
            return weighted_loss, {"dino_loss": raw_loss.detach().cpu().item()}
        return weighted_loss

    # ------------------------------------------------------------------
    # teacher normalization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply_center_update(self, teacher_output: torch.Tensor) -> None:
        if not self.training:
            return
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1.0 - self.center_momentum
        )

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output: torch.Tensor) -> torch.Tensor:
        current_temp = self.get_current_teacher_temp()
        out = F.softmax((teacher_output - self.center) / current_temp, dim=-1)
        self.apply_center_update(teacher_output)
        return out

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output: torch.Tensor) -> torch.Tensor:
        """Sinkhorn-Knopp normalization (no centering)."""
        current_temp = self.get_current_teacher_temp()
        Q = torch.exp(teacher_output / current_temp).t()  # [K, B]
        B = Q.shape[1]
        K = Q.shape[0]

        sum_Q = Q.sum()
        if sum_Q == 0:
            return torch.ones_like(Q.t()) / K
        Q = Q / sum_Q

        for _ in range(self.sinkhorn_knopp_niters):
            row_sums = Q.sum(dim=1, keepdim=True)
            row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
            Q = Q / row_sums / K

            col_sums = Q.sum(dim=0, keepdim=True)
            col_sums = torch.where(col_sums == 0, torch.ones_like(col_sums), col_sums)
            Q = Q / col_sums / B

        return (Q * B).t()
