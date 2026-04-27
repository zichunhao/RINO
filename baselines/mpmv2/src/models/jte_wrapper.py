"""Adapter wrapping JetTransformerEncoder for use inside MaskedParticleModelling.

MPM uses the mltools Transformer interface:
  - encoder(x, mask, ctxt=None)  where mask is True=valid, x is pre-embedded
  - encoder.dim                  → hidden dimension
  - encoder.num_registers        → number of prepended register tokens
  - encoder.get_combined_mask(mask) → mask extended to cover register tokens

JetTransformerEncoder instead:
  - takes raw particle features and does its own embedding
  - returns a (rep, particles_out) tuple with optional pooling

This wrapper bypasses JTE's embedding layer and pooling, using only its
transformer backbone, final norm, and register tokens.  MPM's own csts_emb
/ csts_id_emb layers handle the embedding as before.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Allow importing from baselines/models/ regardless of the working directory
_BASELINES_DIR = Path(__file__).parents[3]  # .../baselines/mpm/src/models → baselines/
if str(_BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINES_DIR))

from models.jet_transformer_encoder import JetTransformerEncoder  # noqa: E402


class JetTransformerMLToolsWrapper(nn.Module):
    """Wrap JetTransformerEncoder to expose the mltools Transformer interface.

    Only the transformer backbone, final norm, and register tokens from JTE
    are used.  The embedding and pooling layers of JTE are intentionally
    bypassed — MPM supplies pre-embedded tokens and reads per-token outputs.

    Parameters
    ----------
    jte:
        An already-constructed :class:`JetTransformerEncoder` instance.
        ``part_dim`` is irrelevant here (the embedding is unused), but JTE
        must have been built with ``pooling`` set to any valid value.
    """

    def __init__(self, jte: JetTransformerEncoder) -> None:
        super().__init__()
        self.jte = jte

    # ------------------------------------------------------------------
    # mltools-compatible properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Hidden dimension of the transformer."""
        return self.jte.d_model

    @property
    def outp_dim(self) -> int:
        """Output dimension — same as ``dim`` since we expose per-token features."""
        return self.jte.d_model

    @property
    def num_registers(self) -> int:
        """Number of register tokens prepended to the sequence."""
        return self.jte.num_registers

    def get_combined_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Extend a valid-particle mask to cover prepended register tokens.

        Parameters
        ----------
        mask:
            ``(B, N)`` bool tensor — True = valid particle (mltools convention).

        Returns
        -------
        ``(B, num_registers + N)`` bool tensor, same convention.
        """
        if self.jte.num_registers == 0:
            return mask
        prefix = mask.new_ones(mask.shape[0], self.jte.num_registers)
        return torch.cat([prefix, mask], dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        ctxt: torch.Tensor | None = None,
        causal: bool = False,  # accepted for API compatibility, unused
    ) -> torch.Tensor:
        """Run pre-embedded tokens through JTE's transformer backbone.

        Parameters
        ----------
        x:
            ``(B, N, d_model)`` — already-embedded particle tokens.
        mask:
            ``(B, N)`` — True = valid token (mltools convention).
        ctxt:
            Ignored.  JTE does not use jet-level context at the backbone
            level (context can be handled via ``cls_jet`` pooling inside
            JTE, but that is not used in the MPM setting).
        causal:
            Ignored.  Included for :class:`JetBackbone` compatibility.

        Returns
        -------
        ``(B, num_registers + N, d_model)`` — per-token embeddings after
        the transformer stack and final normalisation.
        """
        # Convert from mltools convention (True=valid) to PyTorch attn
        # convention (True=ignored / padding).
        padding_mask = ~mask
        batch_size = x.shape[0]

        # Prepend register tokens when present.
        if self.jte.register_tokens is not None:
            regs = self.jte.register_tokens.expand(batch_size, -1, -1)
            x = torch.cat([regs, x], dim=1)
            padding_mask = F.pad(
                padding_mask, (self.jte.num_registers, 0), value=False
            )

        x = self.jte.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.jte.final_norm(x)
        return x


def build_jte_wrapper(encoder_config: dict, csts_dim: int) -> "JetTransformerMLToolsWrapper":
    """Instantiate a :class:`JetTransformerMLToolsWrapper` from a config dict.

    The config dict should contain JTE constructor keyword arguments.  The
    special key ``"type"`` (used to identify this encoder variant) is stripped
    before forwarding to :class:`JetTransformerEncoder`.

    Parameters
    ----------
    encoder_config:
        Dict with keys matching :class:`JetTransformerEncoder` parameters.
        ``part_dim`` is set automatically from *csts_dim*; do not include it.
    csts_dim:
        Particle feature dimension (from the data sample).
    """
    cfg = {k: v for k, v in encoder_config.items() if k != "type"}
    jte = JetTransformerEncoder(part_dim=csts_dim, **cfg)
    return JetTransformerMLToolsWrapper(jte)
