"""Causal transformer decoder for autoregressive token prediction.

Architecturally identical to :class:`JetTransformerEncoder` (same building
blocks: SwiGLU/GELU FFN, RMSNorm/LayerNorm, LayerScale, register tokens)
but with two key differences:

1. **Causal attention mask** — each position can only attend to itself and
   earlier positions in the sequence.
2. **Token embedding** — input is discrete VQ-VAE code indices, mapped through
   a learned embedding table (instead of linear projection from continuous
   features).

Used by OmniJet-α for autoregressive next-token prediction on VQ-tokenized
jet constituents (sorted by descending pT).
"""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .jet_transformer_encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
    _make_norm,
)
from utils.logger import LOGGER


class JetTransformerDecoder(nn.Module):
    """Causal transformer decoder for autoregressive jet constituent modeling.

    Parameters
    ----------
    vocab_size:
        Number of discrete token classes (VQ codebook size + special tokens).
    d_model:
        Internal transformer dimension.
    nhead:
        Number of attention heads.
    num_layers:
        Number of transformer layers.
    activation:
        FFN activation. ``"SwiGLU"`` (default) or ``"GELU"``.
    norm:
        Normalization layer. ``"LayerNorm"`` or ``"RMSNorm"``.
    layer_scale_init:
        Initial value for LayerScale gamma. ``None`` disables LayerScale.
    num_registers:
        Number of learnable register tokens prepended to the sequence.
    mlp_ratio:
        Feedforward hidden dim multiplier.
    qkv_bias:
        Whether to include bias in QKV projections.
    attention_dropout:
        Dropout probability inside attention.
    norm_eps:
        Epsilon for normalization layers.
    apply_final_norm:
        Apply a normalization layer after the transformer stack.
    apply_embedding_norm:
        Apply a normalization layer to token embeddings before the transformer.
    max_seq_len:
        Maximum sequence length (for causal mask buffer).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        activation: Literal["GELU", "SwiGLU"] = "SwiGLU",
        norm: Literal["LayerNorm", "RMSNorm"] = "RMSNorm",
        layer_scale_init: float | None = 0.01,
        num_registers: int = 0,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        apply_final_norm: bool = True,
        apply_embedding_norm: bool = True,
        max_seq_len: int = 256,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_registers = num_registers

        dim_feedforward = int(mlp_ratio) * d_model

        LOGGER.info(f"Decoder vocab_size: {vocab_size}")
        LOGGER.info(f"Decoder d_model: {d_model}, nhead: {nhead}, layers: {num_layers}")
        LOGGER.info(f"FFN activation: {activation} (dim_ff={dim_feedforward})")

        # ---- token embedding ------------------------------------------------ #
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embed_norm = (
            _make_norm(norm, d_model, norm_eps)
            if apply_embedding_norm
            else nn.Identity()
        )

        # ---- register tokens ------------------------------------------------ #
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_registers, d_model)
            )
            nn.init.normal_(self.register_tokens, std=0.02)
            LOGGER.info(f"Using {num_registers} register tokens.")
        else:
            self.register_tokens = None

        # ---- transformer backbone ------------------------------------------- #
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            norm=norm,
            layer_scale_init=layer_scale_init,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout,
            norm_eps=norm_eps,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # ---- final norm ----------------------------------------------------- #
        self.final_norm = (
            _make_norm(norm, d_model, norm_eps) if apply_final_norm else nn.Identity()
        )

        # ---- causal mask buffer --------------------------------------------- #
        # Upper-triangular True mask: position i cannot attend to j > i.
        # nn.MultiheadAttention treats True as "block this position".
        causal = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: ``(B, N)`` – integer token indices.
            mask: ``(B, N)`` – **True = valid token**, False = padding.

        Returns:
            ``(B, N, d_model)`` – per-position embeddings (excludes registers).
        """
        batch_size, seq_len = token_ids.shape

        # Convert to PyTorch padding convention: True = masked/ignored
        padding_mask = ~mask

        # Embed tokens
        x = self.embed_norm(self.token_embedding(token_ids))

        # Prepend register tokens
        if self.register_tokens is not None:
            regs = self.register_tokens.expand(batch_size, -1, -1)
            x = torch.cat([regs, x], dim=1)
            padding_mask = F.pad(padding_mask, (self.num_registers, 0), value=False)

        total_len = x.shape[1]

        # Build causal mask for the full sequence (registers + tokens).
        # Registers can attend to each other but not to future token positions.
        causal = self.causal_mask[:total_len, :total_len]

        # Run transformer with causal + padding masks
        x = self.transformer(
            x, src_key_padding_mask=padding_mask, attn_mask=causal
        )
        x = self.final_norm(x)

        # Strip register tokens — return only token positions
        if self.register_tokens is not None:
            x = x[:, self.num_registers :]

        return x

    # ---------------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
