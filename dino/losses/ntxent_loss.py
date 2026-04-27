"""
NT-Xent (Normalized Temperature-scaled Cross-Entropy) contrastive loss.

Based on the SimCLR / JetCLR formulation. Wrapped with ``WarmupSchedule``
for consistent weight scheduling with other PARCEL losses.

Config example
--------------
.. code-block:: yaml

    ntxent:
      temperature: 0.1
      weight: 1.0
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


class NTXentLoss(nn.Module):
    """NT-Xent contrastive loss with optional weight warmup.

    Supports the standard 2-view case (SimCLR) as well as multi-view:
    given *V* views each of batch size *B*, every view of the same sample
    is a positive pair and all other samples are negatives.

    Parameters
    ----------
    temperature:
        Softmax temperature for the similarity logits.
    weight:
        Target (final) loss weight.
    weight_warmup:
        Optional dict controlling the weight schedule.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        weight: float = 1.0,
        weight_warmup: dict | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self._weight_schedule = WarmupSchedule.from_config(weight, weight_warmup)
        self.current_epoch: int = 0
        self.current_step: int = 0

        if self._weight_schedule._start_value != self._weight_schedule.end_value:
            LOGGER.info(
                f"NTXentLoss weight warmup: {self._weight_schedule._start_value} -> "
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
            f"NTXentLoss resumed at epoch {epoch}: weight={self.get_current_weight():.4f}"
        )

    def resume_step(self, step: int) -> None:
        self.current_step = step

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        views: list[torch.Tensor],
        return_components: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Compute NT-Xent loss over multiple views.

        Parameters
        ----------
        views:
            List of *V* tensors, each ``(B, D)`` — projection-head outputs
            from *V* augmented views of the same batch.
        return_components:
            If True, also return a dict of scalar components.

        Returns
        -------
        Weighted loss (and optionally a components dict).
        """
        V = len(views)
        B = views[0].shape[0]

        # Normalize and stack: (V*B, D)
        z = torch.cat([F.normalize(v, dim=1) for v in views], dim=0)

        # (V*B, V*B) cosine similarity matrix
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity
        sim.fill_diagonal_(float("-inf"))

        # For each row i (view v, sample s), positives are all other views
        # of sample s, i.e. columns {v'*B + s : v' != v}.
        # We use cross_entropy over all V-1 positives by averaging the loss
        # across each positive pair (symmetric).
        if V == 2:
            # Fast path: standard SimCLR/JetCLR formulation
            labels = torch.cat(
                [torch.arange(B, 2 * B, device=z.device),
                 torch.arange(B, device=z.device)],
            )
            raw_loss = F.cross_entropy(sim, labels)
        else:
            # Multi-view: for each anchor, average log-softmax over all positives
            total = V * B
            log_softmax = F.log_softmax(sim, dim=1)

            loss = torch.tensor(0.0, device=z.device)
            num_pairs = 0
            for v1 in range(V):
                for v2 in range(V):
                    if v1 == v2:
                        continue
                    # Rows from view v1, positive columns from view v2
                    rows = torch.arange(v1 * B, (v1 + 1) * B, device=z.device)
                    cols = torch.arange(v2 * B, (v2 + 1) * B, device=z.device)
                    loss = loss - log_softmax[rows, cols].mean()
                    num_pairs += 1
            raw_loss = loss / num_pairs

        weighted_loss = raw_loss * self.get_current_weight()

        if return_components:
            return weighted_loss, {"ntxent_loss": raw_loss.detach().cpu().item()}
        return weighted_loss
