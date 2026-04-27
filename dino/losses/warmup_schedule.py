"""
WarmupSchedule: a lightweight config-driven schedule for a scalar value.

Behaviour
---------
- counter < start          → constant at start_value
- start <= counter < end   → ramp from start_value → end_value
- counter >= end           → constant at end_value

Both 'linear' and 'cosine' ramp shapes are supported.
The schedule can operate in epoch mode (default) or step mode.

Step mode is activated by setting ``start_step`` / ``end_step`` in the config:

.. code-block:: yaml

    temp_warmup:
      start_value: 0.04
      start_step: 0
      end_step: 5000
      warmup_scheduler: linear
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class WarmupSchedule:
    """Scalar warmup schedule with a flat prefix and a flat suffix.

    Parameters
    ----------
    end_value:
        The target value reached (and held) at/after the end counter.
    start_value:
        The value held before the start counter (also the starting point of
        the ramp).  Defaults to ``end_value`` (i.e. no warmup).
    start_epoch:
        Epoch at which the ramp begins (epoch mode).  Defaults to 0.
    end_epoch:
        Epoch at which the ramp finishes (epoch mode).  Must be >= ``start_epoch``.
    warmup_scheduler:
        Shape of the ramp: ``"linear"`` or ``"cosine"``.
    start_step:
        Step at which the ramp begins (step mode).  Setting this (or
        ``end_step``) switches the schedule to step mode.
    end_step:
        Step at which the ramp finishes (step mode).
    """

    end_value: float
    start_value: float | None = None  # None → same as end_value (no ramp)
    start_epoch: int = 0
    end_epoch: int = 0
    warmup_scheduler: Literal["linear", "cosine"] = "cosine"
    start_step: int | None = None
    end_step: int | None = None

    # resolved after __post_init__
    _start_value: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        start = self._start_idx
        end = self._end_idx
        if end < start:
            raise ValueError(
                f"end ({end}) must be >= start ({start}) in {self.mode} mode"
            )
        self._start_value = (
            self.end_value if self.start_value is None else self.start_value
        )

    # ------------------------------------------------------------------
    # mode helpers
    # ------------------------------------------------------------------

    @property
    def mode(self) -> Literal["epoch", "step"]:
        """Return ``"step"`` when step-based fields are set, else ``"epoch"``."""
        return (
            "step"
            if (self.start_step is not None or self.end_step is not None)
            else "epoch"
        )

    @property
    def _start_idx(self) -> int:
        if self.mode == "step":
            return self.start_step if self.start_step is not None else 0
        return self.start_epoch

    @property
    def _end_idx(self) -> int:
        if self.mode == "step":
            return self.end_step if self.end_step is not None else 0
        return self.end_epoch

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_value(self, counter: int) -> float:
        """Return the schedule value for *counter* (epoch or step, per ``mode``)."""
        start = self._start_idx
        end = self._end_idx

        if counter < start:
            return self._start_value

        ramp_length = end - start
        if ramp_length == 0 or counter >= end:
            return self.end_value

        # fractional progress through ramp: 0 → 1
        t = (counter - start) / ramp_length

        if self.warmup_scheduler == "linear":
            factor = t
        else:  # cosine
            factor = 0.5 * (1.0 - math.cos(math.pi * t))

        return self._start_value + factor * (self.end_value - self._start_value)

    # ------------------------------------------------------------------
    # convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, end_value: float, cfg: dict | None) -> "WarmupSchedule":
        """Build from an optional config dict.

        If *cfg* is ``None`` the schedule is a constant equal to *end_value*.

        Epoch mode keys (all optional):
            start_value, start_epoch, end_epoch, warmup_scheduler
        Step mode keys (activates step mode when present):
            start_step, end_step  (plus start_value, warmup_scheduler)
        """
        if cfg is None:
            return cls(end_value=end_value)
        return cls(
            end_value=end_value,
            start_value=cfg.get("start_value", None),
            start_epoch=cfg.get("start_epoch", 0),
            end_epoch=cfg.get("end_epoch", 0),
            warmup_scheduler=cfg.get("warmup_scheduler", "cosine"),
            start_step=cfg.get("start_step", None),
            end_step=cfg.get("end_step", None),
        )

    @classmethod
    def constant(cls, value: float) -> "WarmupSchedule":
        """Return a constant schedule (no warmup at all)."""
        return cls(end_value=value)
