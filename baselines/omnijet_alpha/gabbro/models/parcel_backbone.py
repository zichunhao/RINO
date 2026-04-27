"""PARCEL-compatible autoregressive backbone for OmniJet-alpha.

Uses JetTransformerDecoder (causal attention) with the same architecture
parameters as RINO's JetTransformerEncoder for apples-to-apples comparison.
Online tokenization via a frozen shared VQ-VAE.
"""

import sys
from pathlib import Path

import awkward as ak
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow importing from baselines/models/
_BASELINES_DIR = Path(__file__).resolve().parents[3]  # omnijet_alpha/gabbro/models -> baselines/
if str(_BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINES_DIR))

from models.jet_transformer_decoder import JetTransformerDecoder  # noqa: E402


class NextTokenPredictionHead(nn.Module):
    """Linear head mapping d_model -> vocab_size."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PARCELNextTokenPrediction(L.LightningModule):
    """Autoregressive next-token prediction using JetTransformerDecoder.

    Two-stage pipeline:
    1. A frozen shared VQ-VAE tokenizes continuous RINO features into discrete codes.
    2. This module predicts next tokens autoregressively using a causal decoder.

    Parameters
    ----------
    decoder_kwargs:
        Keyword arguments for :class:`JetTransformerDecoder`.
    vqvae_ckpt_path:
        Path to the shared VQ-VAE checkpoint. If None, assumes pre-tokenized input.
    start_token_id:
        Token ID for the start-of-sequence token (default: num_codes).
    stop_token_id:
        Token ID for the end-of-sequence token (default: num_codes + 1).
    lr:
        Learning rate.
    weight_decay:
        Weight decay for AdamW.
    """

    def __init__(
        self,
        decoder_kwargs: dict | None = None,
        vqvae_ckpt_path: str | None = None,
        start_token_id: int | None = None,
        stop_token_id: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        _decoder_kwargs = decoder_kwargs or {}
        self.vocab_size = _decoder_kwargs.get("vocab_size", 514)
        self.num_codes = self.vocab_size - 2  # codes + start + stop

        self.start_token_id = start_token_id if start_token_id is not None else self.num_codes
        self.stop_token_id = stop_token_id if stop_token_id is not None else self.num_codes + 1

        # Build decoder
        self.decoder = JetTransformerDecoder(**_decoder_kwargs)

        # Build prediction head
        self.head = NextTokenPredictionHead(self.decoder.d_model, self.vocab_size)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer params
        self.lr = lr
        self.weight_decay = weight_decay

        # Load frozen VQ-VAE if provided
        self.vqvae = None
        if vqvae_ckpt_path and vqvae_ckpt_path != "None":
            self._load_vqvae(vqvae_ckpt_path)

    def _load_vqvae(self, ckpt_path: str):
        """Load and freeze the shared VQ-VAE for online tokenization."""
        # Import here to avoid circular deps
        sys.path.insert(0, str(_BASELINES_DIR))
        from vqvae import SharedVQVAELightning  # noqa: E402

        # Use Lightning's ``load_from_checkpoint`` so the model is built
        # from the saved hparams (not default ctor args — those would cause
        # size mismatches if the trained ckpt used different dims).
        # weights_only=False because Lightning ckpts contain non-tensor
        # hparams (AttributeDict) that torch 2.6+'s default rejects.
        vqvae_module = SharedVQVAELightning.load_from_checkpoint(
            ckpt_path, map_location="cpu", weights_only=False
        )
        self.vqvae = vqvae_module.model
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def _tokenize_batch(self, particles: torch.Tensor, mask: torch.Tensor):
        """Run frozen VQ-VAE to get per-particle code indices.

        Returns
        -------
        token_ids: (B, N) long tensor with code indices.
            Padded positions are filled with stop_token_id.
        """
        with torch.no_grad():
            code_indices = self.vqvae.tokenize(particles, mask)  # (B, N), -1 for pad

        # Replace -1 (padding) with stop_token_id
        code_indices[code_indices < 0] = self.stop_token_id
        return code_indices

    def _prepare_autoregressive(self, code_indices: torch.Tensor, mask: torch.Tensor):
        """Build input/target pairs for next-token prediction.

        Prepends start token, appends stop token after last valid particle.

        Returns
        -------
        input_ids: (B, N) — tokens 0..T-1 (start + codes)
        target_ids: (B, N) — tokens 1..T (codes + stop)
        ar_mask: (B, N) — True for valid positions
        """
        B, N = code_indices.shape
        device = code_indices.device

        # Prepend start token, making sequence length N+1
        start = torch.full((B, 1), self.start_token_id, dtype=torch.long, device=device)
        stop = torch.full((B, 1), self.stop_token_id, dtype=torch.long, device=device)

        # full sequence: [start, code_0, code_1, ..., code_{N-1}, stop]
        full_seq = torch.cat([start, code_indices, stop], dim=1)  # (B, N+2)

        # Mask for the full sequence: start is valid, then mask, then stop for last valid
        full_mask = torch.cat([
            mask.new_ones(B, 1),  # start token
            mask,
            mask.new_zeros(B, 1),  # stop token (may or may not be in range)
        ], dim=1)  # (B, N+2)

        # For each jet, mark the position right after the last valid particle as valid (stop)
        n_valid = mask.sum(dim=1)  # (B,)
        stop_pos = n_valid + 1  # +1 for start token
        for i in range(B):
            if stop_pos[i] < full_mask.shape[1]:
                full_mask[i, stop_pos[i]] = True

        # Truncate to max_seq_len of decoder
        max_len = min(full_seq.shape[1] - 1, self.decoder.causal_mask.shape[0])
        input_ids = full_seq[:, :max_len]
        target_ids = full_seq[:, 1 : max_len + 1]
        ar_mask = full_mask[:, :max_len]

        return input_ids, target_ids, ar_mask

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder + head.

        Args:
            token_ids: (B, N) integer token indices.
            mask: (B, N) True = valid position.

        Returns:
            logits: (B, N, vocab_size)
        """
        hidden = self.decoder(token_ids, mask)  # (B, N, d_model)
        return self.head(hidden)  # (B, N, vocab_size)

    def model_step(self, batch):
        """Perform a single model step.

        If VQ-VAE is loaded, tokenizes continuous features on-the-fly.
        Otherwise expects pre-tokenized input in batch["part_features"].
        """
        if self.vqvae is not None:
            # Online tokenization from RINO features
            particles = batch["sequence"]  # (B, N, 7)
            mask = batch["mask"]  # (B, N)

            code_indices = self._tokenize_batch(particles, mask)
            input_ids, target_ids, ar_mask = self._prepare_autoregressive(
                code_indices, mask
            )
        else:
            # Pre-tokenized input (original OmniJet-alpha format)
            X = batch["part_features"].squeeze().long()
            input_ids = X[:, :, 0]
            target_ids = X[:, :, 1]
            ar_mask = batch["part_mask"]

        logits = self.forward(input_ids, ar_mask)

        # Flatten for cross-entropy
        B, T, C = logits.shape
        # reshape (not view) because ``target_ids`` is a slice of the
        # prepended-start/appended-stop sequence and may be non-contiguous.
        loss = self.criterion(
            logits.reshape(B * T, C), target_ids.reshape(B * T)
        )

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def generate_batch(self, batch_size: int, max_len: int = 128):
        """Autoregressively generate a batch of token sequences."""
        device = next(self.parameters()).device
        idx = torch.full((batch_size, 1), self.start_token_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            mask = torch.ones_like(idx, dtype=torch.bool)
            logits = self.forward(idx, mask)
            next_logits = logits[:, -1, :]
            # Exclude start token from sampling
            probs = F.softmax(next_logits[:, 1:], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) + 1
            idx = torch.cat([idx, next_token], dim=1)

            # Stop if all sequences have generated a stop token
            if (idx == self.stop_token_id).any(dim=1).all():
                break

        # Trim to stop token per jet
        result = []
        for seq in idx.cpu().numpy():
            stop_pos = np.where(seq == self.stop_token_id)[0]
            if len(stop_pos) > 0:
                result.append(seq[: stop_pos[0]])
            else:
                result.append(seq)
        return ak.Array(result)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
