"""
Koleo regularizer for DINOv2 modified from
https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py

KoLeo Loss - standalone wrapper.

Wraps the existing ``KoLeoLoss`` regulariser and adds:
* ``weight`` / ``weight_warmup`` schedule (via ``WarmupSchedule``)
* ``step_epoch()`` / ``resume_epoch()`` for curriculum control
* ``get_current_weight()`` for transparent weight inspection

Config example
--------------
.. code-block:: yaml

    koleo:
      weight: 0.5
      weight_warmup:          # optional
        start_value: 0.0
        start_epoch: 0
        end_epoch: 10
        warmup_scheduler: cosine
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warmup_schedule import WarmupSchedule
from utils.logger import LOGGER


class _KoLeoLossCore(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Args:
            student_output (BxD): backbone output of student
        """
        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(student_output)  # noqa: E741
        distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
        loss = -torch.log(distances + eps).mean()
        return loss


class KoLeoLoss(nn.Module):
    """Standalone KoLeo entropy-maximisation loss with optional weight warmup.

    Parameters
    ----------
    weight:
        Target (final) loss weight.
    weight_warmup:
        Optional dict controlling the weight schedule.
        Keys: ``start_value``, ``start_epoch``, ``end_epoch``,
        ``warmup_scheduler`` ("cosine" | "linear").
    """

    def __init__(
        self,
        weight: float = 1.0,
        weight_warmup: dict | None = None,
    ) -> None:
        super().__init__()

        self._regularizer = _KoLeoLossCore()
        self._weight_schedule = WarmupSchedule.from_config(weight, weight_warmup)
        self.current_epoch: int = 0
        self.current_step: int = 0

        if self._weight_schedule._start_value != self._weight_schedule.end_value:
            LOGGER.info(
                f"KoLeoLoss weight warmup: {self._weight_schedule._start_value} -> {self._weight_schedule.end_value} "
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
        """Advance epoch counter (call once per epoch, after the epoch)."""
        self.current_epoch += 1

    def resume_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        LOGGER.info(
            f"KoLeoLoss resumed at epoch {epoch}: weight={self.get_current_weight():.4f}"
        )

    def resume_step(self, step: int) -> None:
        self.current_step = step

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        student_global_reps: list[torch.Tensor],
        return_components: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Compute KoLeo loss over all global-view student backbone representations.

        Parameters
        ----------
        student_global_reps:
            List of tensors, one per global view, each shape ``(N, d_model)``.
        return_components:
            If True, also return a dict of scalar components.
        """
        num_views = len(student_global_reps)
        raw_loss = (
            sum(self._regularizer(rep) for rep in student_global_reps) / num_views
        )

        weighted_loss = raw_loss * self.get_current_weight()

        if return_components:
            return weighted_loss, {"koleo_loss": raw_loss.detach().cpu().item()}
        return weighted_loss
