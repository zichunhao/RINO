import copy
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .positional_encoding import (
    PositionalEncoding,
    RelativePositionalBias,
    ScaleConditioning,
    ScaleProjection,
)
from utils.logger import LOGGER

EPS = 1e-5

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Gated bilinear feedforward network with Swish activation."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.lin1 = nn.Linear(d_model, 2 * dim_feedforward)
        self.lin2 = nn.Linear(dim_feedforward, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(self.drop(F.silu(x1) * x2))


class GELUMLP(nn.Module):
    """Standard two-layer MLP with GELU activation."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.lin1 = nn.Linear(d_model, dim_feedforward)
        self.lin2 = nn.Linear(dim_feedforward, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.drop(F.gelu(self.lin1(x))))


class LayerScale(nn.Module):
    """Per-channel learnable scale to stabilize deep residuals."""

    def __init__(self, d_model: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((d_model,), float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no mean-centering)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.normalized_shape = (dim,)  # compatibility shim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _make_norm(
    norm: Literal["LayerNorm", "RMSNorm"], d_model: int, eps: float
) -> nn.Module:
    if norm is None or norm.lower().startswith("layer"):
        return nn.LayerNorm(d_model, eps=eps)
    if norm.lower().startswith("rms"):
        return RMSNorm(d_model, eps=eps)
    raise ValueError(f"Unknown norm: {norm!r}. Choose 'LayerNorm' or 'RMSNorm'.")


# ---------------------------------------------------------------------------
# Transformer encoder layer
# ---------------------------------------------------------------------------


class TransformerEncoderLayer(nn.Module):
    """
    Pre-norm transformer encoder layer with configurable FFN and normalization.

    Supports:
    - FFN: SwiGLU or GELU MLP
    - Norm: LayerNorm or RMSNorm
    - Optional LayerScale on attention and FFN sublayers
    - Optional QKV bias
    - Separate attention dropout
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: Literal["GELU", "SwiGLU"] = "SwiGLU",
        norm: Literal["LayerNorm", "RMSNorm"] | None = "LayerNorm",
        layer_scale_init: float | None = None,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=attention_dropout,
            bias=qkv_bias,
            batch_first=True,
        )

        if activation == "SwiGLU":
            self.ffn = SwiGLU(d_model, dim_feedforward)
        elif activation == "GELU":
            self.ffn = GELUMLP(d_model, dim_feedforward)
        else:
            raise ValueError(
                f"Unknown activation: {activation!r}. Choose 'SwiGLU' or 'GELU'."
            )

        self.norm1 = _make_norm(norm, d_model, norm_eps)
        self.norm2 = _make_norm(norm, d_model, norm_eps)

        self.ls1 = (
            LayerScale(d_model, layer_scale_init)
            if layer_scale_init is not None
            else nn.Identity()
        )
        self.ls2 = (
            LayerScale(d_model, layer_scale_init)
            if layer_scale_init is not None
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nx = self.norm1(x)
        kpm = src_key_padding_mask
        if kpm is not None and attn_bias is not None and kpm.dtype == torch.bool:
            # MHA deprecates mismatched bool key_padding_mask + float attn_mask.
            # Convert to a float additive mask (0 = keep, -inf = ignore).
            kpm = torch.zeros_like(kpm, dtype=attn_bias.dtype).masked_fill_(
                kpm, float("-inf")
            )
        attn_out, _ = self.self_attn(
            nx,
            nx,
            nx,
            key_padding_mask=kpm,
            attn_mask=attn_bias,
            need_weights=False,
        )
        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask, attn_bias=attn_bias)
        return x


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


class AttentionPooling(nn.Module):
    """Single-query attention pooling with learnable query vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # Zero-init → starts as uniform attention (mean of all tokens)
        self.q = nn.Parameter(torch.zeros(1, d_model))
        self.scale = d_model**-0.5

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            attention_mask: (B, N), True = valid token
        Returns:
            (B, D)
        """
        B = x.shape[0]
        q = self.w_q(self.q).unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)
        k = self.w_k(x)
        v = self.w_v(x)

        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, 1, N)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1), float("-inf"))

        return torch.bmm(F.softmax(scores, dim=-1), v).squeeze(1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


_POOLING_OPTIONS = (
    "mean",
    "weighted_mean",
    "attn_pool",
    "cls_token",
    "cls_jet",
    "cls_token_concat_mean",
    "cls_jet_concat_mean",
    "last_token",
)


class JetTransformerEncoder(nn.Module):
    """
    Transformer encoder for jet tagging / representation learning.

    Args:
        part_dim: Dimension of input particle-level features.
        d_model: Internal transformer dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        pooling: Pooling strategy. One of:
            - ``"mean"`` – masked mean over particle tokens.
            - ``"weighted_mean"`` – learnable soft-weighted mean.
            - ``"attn_pool"`` – attention pooling with a learnable query.
            - ``"cls_token"`` – prepend a learnable class token.
            - ``"cls_jet"`` – initialize the class token from projected jet features.
            - ``"cls_token_concat_mean"`` / ``"cls_jet_concat_mean"`` – concatenate
              the class token with the mean-pooled particle representation
              (doubles output dimension to ``2 * d_model``).
            - ``"last_token"`` – use the last valid (non-padding) particle token.
              Natural representation for autoregressive/causal models (e.g. OmniJet).
        activation: FFN activation. ``"SwiGLU"`` (default) or ``"GELU"``.
        norm: Normalization layer. ``"LayerNorm"`` (default), ``"RMSNorm"``, or
            ``None`` (falls back to LayerNorm).
        layer_scale_init: Initial value for LayerScale gamma. ``None`` disables LayerScale.
        jet_dim: Dimension of jet-level features; required for ``"cls_jet"`` pooling.
        mlp_ratio: Feedforward hidden dim multiplier (``dim_feedforward = mlp_ratio * d_model``).
        qkv_bias: Whether to include bias in QKV projections.
        attention_dropout: Dropout probability inside attention.
        norm_eps: Epsilon for normalization layers.
        apply_final_norm: Apply a normalization layer after the transformer stack.
        apply_embedding_norm: Apply a normalization layer to particle (and jet) embeddings
            before the transformer.
        use_penultimate_layer: Use second-to-last layer output instead of the final layer.
            Not recommended for pre-training.
    """

    def __init__(
        self,
        part_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        jet_dim: int | None = None,
        pooling: Literal[
            "mean",
            "weighted_mean",
            "attn_pool",
            "cls_token",
            "cls_jet",
            "cls_token_concat_mean",
            "cls_jet_concat_mean",
            "last_token",
        ] = "cls_jet",
        activation: Literal["GELU", "SwiGLU"] = "SwiGLU",
        norm: Literal["LayerNorm", "RMSNorm"] = "LayerNorm",
        layer_scale_init: float | None = 1.0,
        num_registers: int = 0,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        apply_final_norm: bool = True,
        apply_embedding_norm: bool = False,
        pos_encoding_kwargs: dict[str, any] | None = None,
        rel_pos_bias_kwargs: dict[str, any] | None = None,
        scale_conditioning_kwargs: dict[str, any] | None = None,
        scale_projection_kwargs: dict[str, any] | None = None,
        use_penultimate_layer: bool = False,
    ):
        super().__init__()

        # ---- validation -------------------------------------------------- #
        if d_model % nhead != 0:
            LOGGER.warning(f"d_model ({d_model}) is not divisible by nhead ({nhead}).")

        pooling = pooling.lower()
        if pooling not in _POOLING_OPTIONS:
            raise ValueError(
                f"pooling must be one of {_POOLING_OPTIONS}. Got: {pooling!r}."
            )

        if "cls_jet" in pooling and not (jet_dim and jet_dim > 0):
            raise ValueError(
                "'cls_jet' pooling requires jet_dim to be specified and > 0."
            )

        # ---- state -------------------------------------------------------- #
        self.pooling = pooling
        self.d_model = d_model
        self.has_jet_input = bool(jet_dim and jet_dim > 0)
        self.uses_cls_token = "cls_token" in pooling or "cls_jet" in pooling
        self.rep_dim = 2 * d_model if "concat_mean" in pooling else d_model

        self.use_penultimate_layer = use_penultimate_layer
        if use_penultimate_layer and pooling in ("weighted_mean", "attn_pool"):
            LOGGER.error(
                "use_penultimate_layer is incompatible with 'weighted_mean'/'attn_pool'. Disabling."
            )
            self.use_penultimate_layer = False
        if use_penultimate_layer and num_layers < 2:
            LOGGER.error(
                "use_penultimate_layer requires at least 2 encoder layers. Disabling."
            )
            self.use_penultimate_layer = False
        if self.use_penultimate_layer:
            LOGGER.warning(
                "Using penultimate layer output. Not recommended for pre-training."
            )

        dim_feedforward = int(mlp_ratio) * d_model

        # ---- logging ----------------------------------------------------- #
        LOGGER.info(f"Particle feature dimension: {part_dim}")
        LOGGER.info(f"Model dimension: {d_model}")
        LOGGER.info(f"Pooling strategy: {pooling}")
        LOGGER.info(
            f"FFN activation: {activation}  (dim_feedforward={dim_feedforward})"
        )
        LOGGER.info(f"Normalization: {norm or 'LayerNorm'} (eps={norm_eps})")
        LOGGER.info(f"LayerScale init: {layer_scale_init}")
        if "concat_mean" in pooling:
            LOGGER.warning(
                f"'{pooling}' doubles the output dimension to {self.rep_dim}."
            )

        # ---- embeddings -------------------------------------------------- #
        self.particle_embedding = nn.Linear(part_dim, d_model)
        self.part_norm = (
            _make_norm(norm, d_model, norm_eps)
            if apply_embedding_norm
            else nn.Identity()
        )

        if "cls_jet" in pooling:
            self.jet_embedding = nn.Linear(jet_dim, d_model)
            self.jet_norm = (
                _make_norm(norm, d_model, norm_eps)
                if apply_embedding_norm
                else nn.Identity()
            )
        else:
            self.jet_embedding = None
            self.jet_norm = nn.Identity()

        if "cls_token" in pooling:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.num_registers = num_registers
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_registers, d_model)
            )
            LOGGER.info(f"Using {self.num_registers} register tokens.")
            nn.init.normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

        # ---- pooling network (non-cls strategies) ------------------------- #
        if pooling == "attn_pool":
            self.pooling_network = AttentionPooling(d_model)
        elif pooling == "weighted_mean":
            self.pooling_network = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.pooling_network = None

        # ---- positional encoding ------------------------------------------- #
        # injected post-embedding, pre-transformer
        if pos_encoding_kwargs is not None:
            pos_encoding_kwargs["out_features"] = (
                d_model  # ensure PE output matches model dim
            )
            self.pos_encoding = PositionalEncoding(**pos_encoding_kwargs)
        else:
            self.pos_encoding = None

        # ---- relative positional bias -------------------------------------- #
        # pairwise ΔR bias added to attention logits at every layer
        if rel_pos_bias_kwargs is not None:
            self.rel_pos_bias = RelativePositionalBias(
                nhead=nhead, **rel_pos_bias_kwargs
            )
        else:
            self.rel_pos_bias = None

        # ---- backbone-internal scale token --------------------------------- #
        # Prepended after CLS: [CLS, scale_token, registers..., particles...]
        # Gives all attention layers direct access to the RG scale (nprongs).
        if scale_conditioning_kwargs is not None:
            self.scale_conditioning = ScaleConditioning(
                d_model=d_model, **scale_conditioning_kwargs
            )
            LOGGER.info("Backbone-internal scale token enabled.")
        else:
            self.scale_conditioning = None

        # ---- backbone-internal scale projection ------------------------------ #
        # Applied after pooling: projects out the learned scale-dependent
        # subspace so that rep is scale-invariant by construction.
        if scale_projection_kwargs is not None:
            self.scale_projection = ScaleProjection(
                d_model=d_model, **scale_projection_kwargs
            )
            LOGGER.info("Backbone-internal scale projection enabled.")
        else:
            self.scale_projection = None

        # number of non-particle prefix tokens (CLS + scale_token + registers);
        # used to zero-pad the RPE bias matrix and slice particles_out
        self.num_prefix = (
            int(self.uses_cls_token)
            + int(self.scale_conditioning is not None)
            + num_registers
        )

        # ---- transformer backbone ---------------------------------------- #
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
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        # ---- final norm -------------------------------------------------- #
        self.final_norm = (
            _make_norm(norm, d_model, norm_eps) if apply_final_norm else nn.Identity()
        )

    # ---------------------------------------------------------------------- #
    # Forward helpers
    # ---------------------------------------------------------------------- #

    def _masked_mean(
        self, embeddings: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean over valid (non-padded) tokens. padding_mask: True = ignored (PyTorch convention)."""
        valid = (~padding_mask).unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(EPS)

    def _pool(self, output: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Apply the chosen pooling strategy to the transformer output."""
        if self.pooling in ("cls_token", "cls_jet"):
            return output[:, 0]

        if self.pooling in ("cls_token_concat_mean", "cls_jet_concat_mean"):
            cls_rep = output[:, 0]
            # skip all prefix tokens (CLS + scale_token + registers) to get particle tokens
            particles_out = output[:, self.num_prefix :]
            particle_mask = padding_mask[:, self.num_prefix :]
            mean_rep = self._masked_mean(particles_out, particle_mask)
            return torch.cat([cls_rep, mean_rep], dim=-1)

        if self.pooling == "mean":
            return self._masked_mean(output, padding_mask)

        if self.pooling == "weighted_mean":
            logits = (
                self.pooling_network(output)
                .squeeze(-1)
                .masked_fill(padding_mask, float("-inf"))
            )
            return (output * F.softmax(logits, dim=1).unsqueeze(-1)).sum(dim=1)

        if self.pooling == "attn_pool":
            return self.pooling_network(output, attention_mask=~padding_mask)

        if self.pooling == "last_token":
            # Use last valid particle token (skip prefix tokens: CLS/registers/scale).
            # Designed for autoregressive models (e.g. OmniJet) where the last token
            # has attended to the full sequence via causal masking.
            particles_out = output[:, self.num_prefix :]
            particle_padding = padding_mask[:, self.num_prefix :]  # True = padding
            valid_counts = (~particle_padding).long().sum(dim=1)   # (B,)
            last_idx = (valid_counts - 1).clamp(min=0)             # (B,)
            return particles_out[
                torch.arange(particles_out.shape[0], device=output.device), last_idx
            ]

        raise ValueError(f"Unknown pooling: {self.pooling!r}")

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor,
        jets: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        nprongs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            particles: ``(B, N, part_dim)`` – particle feature matrix.
            mask: ``(B, N)`` – **True = valid particle**, False = padding.
            jets: ``(B, jet_dim)`` – jet-level features (required for ``cls_jet``).
            coords: unmasked particles for PE/RPE; falls back to ``particles``.
            nprongs: ``(B,)`` – RG scale (number of subjets/particles); used by
                the backbone-internal scale token. Inferred from ``mask`` if None.

        Returns:
            rep: ``(B, rep_dim)`` – pooled jet representation.
            particles_out: ``(B, N, d_model)`` – per-particle embeddings after the transformer.
        """
        if mask is None:
            mask = torch.ones(
                particles.shape[:2], dtype=torch.bool, device=particles.device
            )

        # Convert to PyTorch padding convention: True = masked/ignored
        padding_mask = ~mask
        batch_size = particles.shape[0]

        # Embed particles
        x = self.part_norm(self.particle_embedding(particles.to(self.device)))

        # Inject positional encoding (pre-transformer, post-embedding)
        if self.pos_encoding is not None:
            # Use coords if provided (iBOT case: particles may be masked),
            # otherwise fall back to particles (coords are intact)
            pe_source = coords if coords is not None else particles
            x = x + self.pos_encoding(
                pe_source.to(self.device), mask=mask.to(self.device)
            )

        # Compute relative positional bias (B, nhead, N, N) from pairwise ΔR
        attn_bias = None
        if self.rel_pos_bias is not None:
            pe_source = coords if coords is not None else particles
            rpe_coords = pe_source[..., self.rel_pos_bias.input_indices].to(self.device)
            particle_bias = self.rel_pos_bias(rpe_coords)  # (B, nhead, N, N)
            B_rpe, H, N_rpe, _ = particle_bias.shape
            if self.num_prefix > 0:
                # Pad with zero bias for CLS / register tokens
                total = N_rpe + self.num_prefix
                padded = particle_bias.new_zeros(B_rpe, H, total, total)
                padded[:, :, self.num_prefix :, self.num_prefix :] = particle_bias
                particle_bias = padded
            # MHA expects (B*nhead, S, S)
            attn_bias = particle_bias.reshape(
                B_rpe * H, particle_bias.shape[2], particle_bias.shape[3]
            )

        # Build prefix: [cls_ (optional)] [scale_token (optional)] [registers (optional)]
        prefix_parts = []
        if self.uses_cls_token:
            if "cls_jet" in self.pooling:
                if jets is None:
                    raise ValueError("jets must be provided for 'cls_jet' pooling.")
                cls_ = self.jet_norm(
                    self.jet_embedding(jets.to(self.device))
                ).unsqueeze(1)
            else:
                cls_ = self.cls_token.expand(batch_size, -1, -1)
            prefix_parts.append(cls_)

        if self.scale_conditioning is not None:
            if nprongs is None:
                nprongs = mask.sum(dim=-1).float()
            log_n = torch.log(nprongs.clamp(min=1).to(self.device))
            scale_token = self.scale_conditioning(log_n).unsqueeze(1)  # (B, 1, d_model)
            prefix_parts.append(scale_token)

        if self.register_tokens is not None:
            prefix_parts.append(self.register_tokens.expand(batch_size, -1, -1))

        if prefix_parts:
            x = torch.cat([*prefix_parts, x], dim=1)
            num_prefix = sum(p.shape[1] for p in prefix_parts)
            padding_mask = F.pad(padding_mask, (num_prefix, 0), value=False)

        # Run transformer (optionally stop at penultimate layer)
        if self.use_penultimate_layer:
            for layer in self.transformer_encoder.layers[:-1]:
                x = layer(x, src_key_padding_mask=padding_mask, attn_bias=attn_bias)
        else:
            x = self.transformer_encoder(
                x, src_key_padding_mask=padding_mask, attn_bias=attn_bias
            )
        x = self.final_norm(x)

        # Separate prefix tokens (CLS + scale_token + registers) from particle tokens
        if self.num_prefix > 0:
            particles_out = x[:, self.num_prefix :]
        else:
            particles_out = x

        rep = self._pool(x, padding_mask)

        # Project out scale-dependent subspace if configured
        if self.scale_projection is not None:
            if nprongs is None:
                nprongs = mask.sum(dim=-1).float()
            log_n = torch.log(nprongs.clamp(min=1).to(self.device))
            rep = self.scale_projection(rep, log_n)

        return rep, particles_out

    # ---------------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------------- #

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def freeze_backbone(self) -> None:
        """Freeze transformer encoder weights."""
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze transformer encoder weights."""
        for param in self.transformer_encoder.parameters():
            param.requires_grad = True

    def freeze_embedding(self) -> None:
        """Freeze all embedding layers."""
        modules = [self.particle_embedding, self.part_norm]
        if self.jet_embedding is not None:
            modules += [self.jet_embedding, self.jet_norm]
        for m in modules:
            for param in m.parameters():
                param.requires_grad = False

    def unfreeze_embedding(self) -> None:
        """Unfreeze all embedding layers."""
        modules = [self.particle_embedding, self.part_norm]
        if self.jet_embedding is not None:
            modules += [self.jet_embedding, self.jet_norm]
        for m in modules:
            for param in m.parameters():
                param.requires_grad = True
