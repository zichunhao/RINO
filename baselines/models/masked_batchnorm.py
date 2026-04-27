from pathlib import Path
import torch
from torch import nn
from utils.logger import setup_logger

LOGGER = setup_logger(name=Path(__file__).stem, log_level="DEBUG")
EPS = 1e-6


class MaskedBatchNorm1d(nn.Module):
    """
    Batch normalization for 1D data with mask support.
    Only computes statistics over unmasked values.

    Args:
        num_features: Number of features
        momentum: BatchNorm momentum (default: None)
        affine: Whether to use learnable affine parameters (default: False)
        track_running_stats: Whether to track running statistics (default: True)
    """

    def __init__(
        self,
        num_features: int,
        momentum: float | None = None,
        affine: bool = False,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass with optional mask.

        Args:
            x: Input tensor (batch_size, num_features, seq_len) or (batch_size, seq_len, num_features)
            mask: Boolean mask (batch_size, seq_len) where True indicates padding

        Returns:
            Normalized tensor of same shape as input
        """
        if x.dim() not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D input tensor, got {x.dim()}D")

        # Handle both (B,C,L) and (B,L,C) formats
        need_transpose = x.dim() == 3 and x.shape[1] != self.num_features
        if need_transpose:
            x = x.transpose(1, 2)

        if x.dim() == 2:
            x = x.unsqueeze(2)
            if mask is not None:
                mask = mask.unsqueeze(2)

        # Calculate batch statistics
        if mask is not None:
            # Create feature mask from sequence mask
            mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)
            valid_values = ~mask

            # Compute mean only over valid values
            n = valid_values.sum(dim=2, keepdim=True).clamp_min(1)
            mean = (x * valid_values.float()).sum(dim=2, keepdim=True) / n

            # Compute variance only over valid values
            var = ((x - mean) * valid_values.float()).pow(2).sum(
                dim=2, keepdim=True
            ) / n
        else:
            mean = x.mean(dim=2, keepdim=True)
            var = x.var(dim=2, keepdim=True, unbiased=False)

        if self.track_running_stats:
            if self.training:
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked + 1)
                else:
                    exponential_average_factor = self.momentum

                self.running_mean = (
                    1 - exponential_average_factor
                ) * self.running_mean + exponential_average_factor * mean.mean(0).squeeze()
                self.running_var = (
                    1 - exponential_average_factor
                ) * self.running_var + exponential_average_factor * var.mean(0).squeeze()
                self.num_batches_tracked += 1

                # Use running statistics for normalization
                mean_norm = mean
                var_norm = var
            else:
                mean_norm = self.running_mean.view(1, -1, 1)
                var_norm = self.running_var.view(1, -1, 1)
        else:
            mean_norm = mean
            var_norm = var

        # Normalize
        x = (x - mean_norm) / torch.sqrt(var_norm + EPS)

        if self.affine:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

        # Restore original shape
        if need_transpose:
            x = x.transpose(1, 2)
        if x.dim() == 3 and x.size(2) == 1:
            x = x.squeeze(2)

        return x
