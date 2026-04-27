from __future__ import annotations

import numpy as np
from pyparsing import Literal
import torch
import torch.nn as nn
import sys
from pathlib import Path

EPS = sys.float_info.epsilon


class JetCLRTransformer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_mask: bool = False,
        use_continuous_mask: bool = True,
        batch_first: bool = True,
        layer_norm: bool = False,  # false by default in JetCLR
        pooling: Literal["sum", "mean"] = "mean",
        **kwargs,
    ):
        super().__init__()

        # Validate mask settings
        if use_mask and use_continuous_mask:
            raise ValueError(
                "Cannot use both use_mask and use_continuous_mask simultaneously"
            )

        # Define hyperparameters
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.use_mask = use_mask
        self.use_continuous_mask = use_continuous_mask

        # Define subnetworks
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            activation=activation,
            **kwargs,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            enable_nested_tensor=False,
            mask_check=False,
        )
        self.pooling = pooling.lower()
        if self.pooling not in ["sum", "mean"]:
            raise ValueError("pooling must be either 'sum' or 'mean'")
        self.norm = nn.LayerNorm(d_model) if layer_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        **ignore,
    ) -> torch.Tensor:
        """
        Input shape: (batch_size, n_constit, input_dim)
        With batch_first=True, no need to transpose
        """
        # Make a copy
        x = x + 0.0
        pT = x[:, :, 0]

        # Create mask if needed
        if self.use_mask:
            pT_zero = x[:, :, 0] == 0
            mask = self.make_mask(pT_zero).to(x.device)
        elif self.use_continuous_mask:
            pT = x[:, :, 0]
            mask = self.make_continuous_mask(pT).to(x.device)
        elif mask is not None:
            mask = mask.to(x.device)
        else:
            mask = None

        # (batch_size, n_constit, d_model)
        x = self.embedding(x)
        x = self.transformer(x, mask=mask)

        if self.use_mask:
            # Set masked constituents to zero
            x[pT_zero] = 0
        elif self.use_continuous_mask:
            # Scale x by pT for IR safety
            x *= pT[:, :, None]

        # (batch_size, d_model)
        x = self.norm(x)
        if self.pooling == "sum":
            return torch.sum(x, dim=1), x
        elif self.pooling == "mean":
            return self._masked_mean(x, pT > 0), x

    def make_mask(self, pT_zero):
        """
        Input: batch of bools of whether pT=0, shape (batch_size, n_constit)
        Output: mask for transformer, shape (batch_size * nhead, n_constit, n_constit)
        """
        n_constit = pT_zero.size(1)
        pT_zero = torch.repeat_interleave(pT_zero, self.nhead, axis=0)
        pT_zero = torch.repeat_interleave(pT_zero[:, None], n_constit, axis=1)
        mask = torch.zeros(pT_zero.size(0), n_constit, n_constit)
        mask[pT_zero] = -np.inf
        return mask

    def make_continuous_mask(self, pT):
        n_constit = pT.size(1)
        # log_pT = torch.log(pT + 1e-16)
        # log_pT = torch.log(pT.clamp(min=1e-4))
        log_pT = torch.log(pT + 1e-16).clamp(min=-10.0)

        # Normalize: Subtract the max so the largest bias is 0 (Softmax stability trick)
        # This prevents the values from becoming too large and causing overflow
        # log_pT = log_pT - log_pT.max(dim=1, keepdim=True)[0]

        mask = log_pT.unsqueeze(1).repeat(1, n_constit, 1)
        mask = torch.repeat_interleave(mask, self.nhead, dim=0)

        mask = 0.5 * mask

        return mask

    def _masked_mean(self, x, mask):
        mask = mask.unsqueeze(-1)
        masked_x = x * mask
        mean = masked_x.sum(dim=1) / (mask.sum(dim=1) + EPS)
        return mean

    def train(self, mode=True):
        super().train(mode)
        # Keep transformer layers in train mode always to avoid PyTorch 2.x
        # fast path bug that produces NaN with custom attention masks
        for layer in self.transformer.layers:
            layer.train(True)
        return self

    def eval(self):
        return self.train(False)
