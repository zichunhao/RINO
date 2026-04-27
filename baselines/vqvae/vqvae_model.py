"""Shared VQ-VAE for tokenizing jet constituents.

Tokenizes per-particle continuous features (7D RINO kinematics) into discrete
codebook indices. Used as a preprocessing utility by both MPMv1 (masked
prediction) and OmniJet-alpha (autoregressive prediction).

Architecture: small transformer encoder/decoder (3 layers, 128 dim, 8 heads)
with VectorQuant from vqtorch (512 codes, latent_dim=16).
"""

import torch
import torch.nn as nn
from vqtorch.nn import VectorQuant


class SharedVQVAE(nn.Module):
    """Small VQ-VAE for per-particle tokenization.

    Parameters
    ----------
    input_dim:
        Number of continuous input features per particle (default 7 for RINO).
    latent_dim:
        Dimension of the latent space before quantization.
    hidden_dim:
        Internal transformer dimension.
    num_heads:
        Number of attention heads in the transformer layers.
    num_layers:
        Number of transformer encoder/decoder layers.
    num_codes:
        Codebook size (number of discrete tokens).
    alpha:
        Weight for the VQ commitment loss relative to reconstruction loss.
    vq_kwargs:
        Additional keyword arguments for :class:`VectorQuant`.
    """

    def __init__(
        self,
        input_dim: int = 7,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        num_codes: int = 512,
        alpha: float = 10.0,
        vq_kwargs: dict | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_codes = num_codes
        self.alpha = alpha

        # ---- Encoder ---- #
        self.enc_proj_in = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.enc_proj_out = nn.Linear(hidden_dim, latent_dim)

        # ---- Quantizer ---- #
        _vq_kwargs = dict(
            feature_size=latent_dim,
            num_codes=num_codes,
            beta=0.9,
            kmeans_init=True,
            norm=None,
            cb_norm=None,
            affine_lr=2.0,
            sync_nu=2.0,
            replace_freq=10,
        )
        if vq_kwargs:
            _vq_kwargs.update(vq_kwargs)
        self.quantizer = VectorQuant(**_vq_kwargs)

        # ---- Decoder ---- #
        self.dec_proj_in = nn.Linear(latent_dim, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.dec_proj_out = nn.Linear(hidden_dim, input_dim)

        # ---- Reconstruction loss ---- #
        self.recon_loss_fn = nn.L1Loss(reduction="none")

    def encode(self, particles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode particles to latent space.

        Args:
            particles: ``(B, N, input_dim)`` continuous features.
            mask: ``(B, N)`` True = valid particle.

        Returns:
            ``(B, N, latent_dim)`` latent embeddings.
        """
        padding_mask = ~mask
        x = self.enc_proj_in(particles)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.enc_proj_out(x)

    def decode(self, z_q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents back to particle features.

        Args:
            z_q: ``(B, N, latent_dim)`` quantized latent embeddings.
            mask: ``(B, N)`` True = valid particle.

        Returns:
            ``(B, N, input_dim)`` reconstructed features.
        """
        padding_mask = ~mask
        x = self.dec_proj_in(z_q)
        x = self.decoder(x, src_key_padding_mask=padding_mask)
        return self.dec_proj_out(x)

    def forward(
        self, particles: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Full forward: encode → quantize → decode.

        Args:
            particles: ``(B, N, input_dim)``
            mask: ``(B, N)`` True = valid particle.

        Returns:
            rec_particles: ``(B, N, input_dim)`` reconstructed features.
            z_q: ``(B, N, latent_dim)`` quantized latent embeddings.
            vq_dict: dict with keys ``"loss"``, ``"q"`` (code indices), etc.
        """
        latents = self.encode(particles, mask)

        # VectorQuant expects (*, feature_size); operate only on valid particles
        # to avoid polluting the codebook with padding values.
        # Cast to fp32 for vqtorch (its k-means init doesn't support bf16).
        valid_latents = latents[mask].unsqueeze(-1).float()
        z_q_flat, vq_dict = self.quantizer(valid_latents)
        z_q = torch.zeros_like(latents)
        z_q[mask] = z_q_flat.squeeze(-1).to(latents.dtype)

        rec_particles = self.decode(z_q, mask)
        return rec_particles, z_q, vq_dict

    def compute_loss(
        self, particles: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute total VQ-VAE loss.

        Returns:
            total_loss: scalar
            metrics: dict with rec_loss, vq_loss, total_loss
        """
        rec_particles, _z_q, vq_dict = self.forward(particles, mask)

        # Reconstruction loss: L1, averaged over valid particles
        rec_loss = self.recon_loss_fn(particles, rec_particles)
        rec_loss[~mask] = 0.0
        rec_loss = rec_loss.sum((1, 2)) / mask.sum(1).clamp_min(1)
        rec_loss = rec_loss.mean()

        # VQ commitment loss: averaged over valid particles
        vq_loss_flat = vq_dict["loss"]
        vq_loss_full = torch.zeros_like(particles[..., 0])
        vq_loss_full[mask] = vq_loss_flat
        vq_loss = (vq_loss_full.sum(1) / mask.sum(1).clamp_min(1)).mean()

        total_loss = rec_loss + self.alpha * vq_loss

        return total_loss, {
            "rec_loss": rec_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "total_loss": total_loss.detach(),
        }

    @torch.no_grad()
    def tokenize(self, particles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Tokenize particles into discrete codebook indices.

        Args:
            particles: ``(B, N, input_dim)``
            mask: ``(B, N)`` True = valid particle.

        Returns:
            ``(B, N)`` integer code indices. Padded positions are set to -1.
        """
        latents = self.encode(particles, mask)
        z_q_flat, vq_dict = self.quantizer(latents[mask].unsqueeze(-1).float())

        # Extract code indices
        code_indices = torch.full(
            mask.shape, fill_value=-1, dtype=torch.long, device=particles.device
        )
        # vqtorch's ``q`` can come back as (n_valid,), (n_valid, 1), or even
        # (n_valid, 1, 1) depending on input shape.  Flatten unconditionally.
        code_indices[mask] = vq_dict["q"].reshape(-1).to(torch.long)
        return code_indices
