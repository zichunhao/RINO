"""Adapter wrapping JetTransformerEncoder for use inside MPMv1's Bert model.

MPMv1's Bert expects an encoder with the interface:
    encoder(nodes, mask, ctxt=high) -> (B, N, outp_dim)

JetTransformerEncoder instead:
    forward(particles, mask, jets) -> (rep, particles_out)
    where particles_out is (B, N, d_model)

This adapter bypasses JTE's embedding layer and pooling, using only its
transformer backbone, final norm, and register tokens — the same pattern
as MPMv2's jte_wrapper.py.  Bert's own node_norm handles input normalization,
and a final linear projection maps d_model -> outp_dim (n_classes).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Allow importing from baselines/models/ regardless of the working directory.
_BASELINES_DIR = Path(__file__).resolve().parents[3]  # mpmv1/src/models -> baselines/
if str(_BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINES_DIR))

from models.jet_transformer_encoder import JetTransformerEncoder  # noqa: E402


class JTEAdapter(nn.Module):
    """Wrap JetTransformerEncoder to match FullTransformerEncoder interface.

    Parameters
    ----------
    jte:
        An already-constructed :class:`JetTransformerEncoder`.
    outp_dim:
        Output dimension per token (typically n_classes for masked prediction).
    """

    def __init__(self, jte: JetTransformerEncoder, outp_dim: int) -> None:
        super().__init__()
        self.jte = jte
        self.model_dim = jte.d_model
        self.outp_proj = nn.Linear(jte.d_model, outp_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        ctxt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run pre-normalized tokens through JTE's transformer backbone.

        Parameters
        ----------
        x:
            ``(B, N, node_dim)`` — already normalized by Bert's node_norm.
        mask:
            ``(B, N)`` — True = valid token (mattstools convention).
        ctxt:
            Ignored. Included for interface compatibility.

        Returns
        -------
        ``(B, N, outp_dim)`` — per-token logits/embeddings.
        """
        # Convert from mattstools convention (True=valid) to PyTorch attn
        # convention (True=padding).
        padding_mask = ~mask
        batch_size = x.shape[0]

        # Embed through JTE's particle projection (node_dim -> d_model)
        embedded = self.jte.part_norm(self.jte.particle_embedding(x))

        # Prepend register tokens when present.
        if self.jte.register_tokens is not None:
            regs = self.jte.register_tokens.expand(batch_size, -1, -1)
            embedded = torch.cat([regs, embedded], dim=1)
            padding_mask = F.pad(
                padding_mask, (self.jte.num_registers, 0), value=False
            )

        # Run transformer backbone
        out = self.jte.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        out = self.jte.final_norm(out)

        # Strip register tokens
        if self.jte.register_tokens is not None:
            out = out[:, self.jte.num_registers :]

        # Project to output dimension
        return self.outp_proj(out)


def build_jte_adapter(
    backbone_config: dict, inpt_dim: int, outp_dim: int
) -> JTEAdapter:
    """Instantiate a JTEAdapter from a config dict.

    Parameters
    ----------
    backbone_config:
        Dict with JetTransformerEncoder constructor keyword arguments.
    inpt_dim:
        Particle feature dimension (node_dim from data).
    outp_dim:
        Output dimension (n_classes for MPM masked prediction).
    """
    cfg = {k: v for k, v in backbone_config.items() if k != "type"}
    jte = JetTransformerEncoder(part_dim=inpt_dim, **cfg)
    return JTEAdapter(jte, outp_dim)
