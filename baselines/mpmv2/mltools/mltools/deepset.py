"""Code for a simple deep set."""

import torch as T
from torch import nn

from .mlp import MLP


class DeepSet(nn.Module):
    """A deep set network that can provide attention pooling."""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        feat_net_kwargs: dict | None = None,
        post_net_kwargs: dict | None = None,
        add_multiplicity: bool = False,
    ) -> None:
        super().__init__()

        # Dict default arguments
        feat_net_kwargs = feat_net_kwargs or {}
        post_net_kwargs = post_net_kwargs or {}

        # Attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim + add_multiplicity
        self.add_multiplicity = add_multiplicity

        # Submodules
        self.feat_net = MLP(
            inpt_dim=self.inpt_dim,
            ctxt_dim=self.ctxt_dim,
            **feat_net_kwargs,  # Will define the outp_dim
        )
        self.post_net = MLP(
            inpt_dim=self.feat_net.outp_dim,
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_dim,
            **post_net_kwargs,
        )

    def forward(
        self,
        inpt: T.Tensor,
        mask: T.BoolTensor | None,
        ctxt: T.Tensor | None = None,
    ) -> T.Tensor:
        """Forward pass for deep set."""
        # Combine the context with the multiplicity
        if self.add_multiplicity:
            mult = mask.sum(dim=-1, keepdim=True).float()
            ctxt = mult if ctxt is None else T.cat([ctxt, mult], dim=-1)

        # Pass the values through the feature network
        feat_outs = self.feat_net(inpt, ctxt)

        # Average pool using the mask
        if mask is not None:
            feat_outs = feat_outs * mask.unsqueeze(-1)
            feat_outs = feat_outs.sum(dim=-2) / mask.sum(dim=-1)
        else:
            feat_outs = feat_outs.mean(dim=-2)

        # Pass the pooled information through post network and return
        return self.post_net(feat_outs, ctxt)
