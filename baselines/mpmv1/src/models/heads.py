"""Classification head modules for MPMv1 FineTuner finetuning.

These match the ``finaliser`` interface that
``src.models.bert.FineTuner`` expects::

    forward(nodes, mask, ctxt=None, attn_bias=None, attn_mask=None) -> logits

``nodes`` is the per-token backbone output ``(B, N, D)``, ``mask`` is the
``(B, N)`` validity mask.  Heads mean-pool over the mask and project to the
target output dim.
"""

import torch as T
from torch import nn

from mattstools.mattstools.modules import DenseNetwork


class MLPPoolHead(nn.Module):
    """Mean-pool the masked token features and run a dense stack.

    Mirrors DINO's mlp-vanilla head: hidden_dims=[256, 128], ReLU,
    BatchNorm, dropout 0.3, followed by a final linear to ``outp_dim``.
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        edge_dim: int | None = None,   # accepted for finaliser API
        ctxt_dim: int | None = None,   # accepted for finaliser API
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        # DenseNetwork does not pool; we pool manually and feed the vector.
        self.net = DenseNetwork(
            inpt_dim=inpt_dim,
            outp_dim=outp_dim,
            hddn_dim=list(hidden_dims),
            num_blocks=len(hidden_dims),
            act_h="relu",
            nrm="batch" if use_batchnorm else "none",
            drp=dropout,
        )

    def forward(
        self,
        nodes: T.Tensor,
        mask: T.BoolTensor,
        ctxt: T.Tensor | None = None,
        attn_bias: T.Tensor | None = None,
        attn_mask: T.Tensor | None = None,
    ) -> T.Tensor:
        # Masked mean-pool over the token axis.
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        pooled = (nodes * mask.unsqueeze(-1)).sum(dim=-2) / denom
        return self.net(pooled)
