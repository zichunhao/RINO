"""PARCEL-compatible classification backbone for OmniJet-alpha finetuning.

Loads a pretrained ``PARCELNextTokenPrediction`` (JetTransformerDecoder)
backbone, uses the frozen shared VQ-VAE to tokenize continuous RINO features,
runs the decoder in causal mode, and classifies via either a summation head
(linear) or a dense MLP head (matching DINO's mlp-vanilla: hidden_dims=[256,128],
ReLU, BatchNorm, dropout 0.3).

Vanilla finetune: no freezing — the entire model (backbone + head + VQ-VAE keys
whose requires_grad is off) is trainable end-to-end.
"""

import logging
import sys
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow importing from baselines/models/ and baselines/vqvae/
_BASELINES_DIR = Path(__file__).resolve().parents[3]  # omnijet_alpha/gabbro/models -> baselines/
if str(_BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINES_DIR))

from models.jet_transformer_decoder import JetTransformerDecoder  # noqa: E402

log = logging.getLogger(__name__)


class LinearSumHead(nn.Module):
    """Mean/sum-pooled linear classification head (matches OmniJet-α's default)."""

    def __init__(self, d_model: int, n_out: int) -> None:
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, n_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e = F.relu(self.embed(x))
        pooled = (e * mask.unsqueeze(-1)).sum(dim=1)
        return self.proj(pooled)


class MLPClassificationHead(nn.Module):
    """Mean-pooled + DenseNetwork head matching DINO's mlp-vanilla variant.

    Architecture: mean-pool → [Linear(d, 256) → BN → ReLU → Dropout(0.3)
    → Linear(256, 128) → BN → ReLU → Dropout(0.3)] → Linear(128, n_out).
    """

    def __init__(
        self,
        d_model: int,
        n_out: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_model
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.net(pooled)


class PARCELClassificationLightning(L.LightningModule):
    """Classification wrapper around a pretrained PARCEL (JetTransformerDecoder) backbone.

    Parameters
    ----------
    decoder_kwargs:
        Kwargs passed to :class:`JetTransformerDecoder`. Must match the pretraining run.
    vqvae_ckpt_path:
        Path to the shared VQ-VAE Lightning checkpoint (same one used at pretraining).
    backbone_weights_path:
        Path to the pretrained ``PARCELNextTokenPrediction`` checkpoint. If ``None``
        or ``"None"``, the decoder is randomly initialised.
    class_head_type:
        ``"summation"`` (linear) or ``"mlp"``.
    n_out_nodes:
        Number of output classes. For binary top-vs-QCD use 2.
    lr, weight_decay:
        AdamW optimiser.
    """

    def __init__(
        self,
        decoder_kwargs: dict,
        vqvae_ckpt_path: str | None = None,
        backbone_weights_path: str | None = None,
        class_head_type: str = "summation",
        n_out_nodes: int = 2,
        lr: float = 1.0e-3,
        weight_decay: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = decoder_kwargs.get("vocab_size", 514)
        # Same convention as PARCELNextTokenPrediction
        self.num_codes = self.vocab_size - 2
        self.start_token_id = self.num_codes
        self.stop_token_id = self.num_codes + 1

        # Build decoder (must match pretraining kwargs)
        self.decoder = JetTransformerDecoder(**decoder_kwargs)

        # Classification head
        if class_head_type == "summation":
            self.head = LinearSumHead(self.decoder.d_model, n_out_nodes)
        elif class_head_type == "mlp":
            self.head = MLPClassificationHead(self.decoder.d_model, n_out_nodes)
        else:
            raise ValueError(f"Invalid class_head_type: {class_head_type}")

        self.criterion = nn.CrossEntropyLoss()
        self.n_out_nodes = n_out_nodes

        # Frozen VQ-VAE for online tokenisation
        self.vqvae = None
        if vqvae_ckpt_path and vqvae_ckpt_path != "None":
            self._load_vqvae(vqvae_ckpt_path)

        # Pretrained backbone weights
        if backbone_weights_path and backbone_weights_path != "None":
            self._load_backbone_weights(backbone_weights_path)

    # ------------------------------------------------------------------ #
    # Loading helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_vqvae(self, ckpt_path: str) -> None:
        sys.path.insert(0, str(_BASELINES_DIR))
        from vqvae import SharedVQVAELightning  # noqa: E402

        log.info(f"Loading frozen VQ-VAE from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        vqvae_module = SharedVQVAELightning()
        vqvae_module.load_state_dict(ckpt["state_dict"])
        self.vqvae = vqvae_module.model
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

    def _load_backbone_weights(self, ckpt_path: str) -> None:
        """Load decoder weights from a PARCELNextTokenPrediction Lightning ckpt.

        The pretraining Lightning module stores the backbone under the
        ``decoder.*`` prefix in its state dict; we strip that prefix and load
        into ``self.decoder`` with ``strict=False`` so extra keys (head, vqvae)
        are ignored.
        """
        log.info(f"Loading pretrained backbone weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        decoder_sd: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("decoder."):
                new_key = k[len("decoder."):]
                decoder_sd[new_key] = v

        if not decoder_sd:
            log.warning(
                "No 'decoder.*' keys found in backbone checkpoint — decoder will "
                "stay randomly initialised. Check that the checkpoint is a "
                "PARCELNextTokenPrediction ckpt."
            )
            return

        missing, unexpected = self.decoder.load_state_dict(decoder_sd, strict=False)
        log.info(
            f"Loaded decoder weights: {len(decoder_sd)} keys transferred, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
        if missing:
            log.warning(f"Missing decoder keys (first 5): {missing[:5]}")
        if unexpected:
            log.warning(f"Unexpected decoder keys (first 5): {unexpected[:5]}")

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def _tokenize(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the frozen VQ-VAE to get per-particle code indices."""
        with torch.no_grad():
            code_ids = self.vqvae.tokenize(sequence, mask)  # (B, N), -1 for pad
        code_ids = code_ids.clone()
        code_ids[code_ids < 0] = self.stop_token_id
        return code_ids

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Tokenize + decode + classify.

        Args:
            sequence: ``(B, N, 7)`` — RINO-normalized particle features.
            mask: ``(B, N)`` — True = valid particle.
        Returns:
            ``(B, n_out_nodes)`` logits.
        """
        if self.vqvae is None:
            raise RuntimeError(
                "vqvae_ckpt_path must be provided for classification finetuning "
                "because the decoder expects token IDs as input."
            )

        code_ids = self._tokenize(sequence, mask)  # (B, N)

        # Prepend a start token (mirrors the pretraining input format)
        B = code_ids.shape[0]
        device = code_ids.device
        start = torch.full((B, 1), self.start_token_id, dtype=torch.long, device=device)
        token_ids = torch.cat([start, code_ids], dim=1)
        start_mask = mask.new_ones(B, 1)
        full_mask = torch.cat([start_mask, mask], dim=1)

        # Clamp to the decoder's max_seq_len (causal mask buffer defines the limit).
        max_len = self.decoder.causal_mask.shape[0]
        token_ids = token_ids[:, :max_len]
        full_mask = full_mask[:, :max_len]

        features = self.decoder(token_ids, full_mask)  # (B, N, d_model)
        return self.head(features, full_mask)

    # ------------------------------------------------------------------ #
    # Training loop                                                       #
    # ------------------------------------------------------------------ #

    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(batch["sequence"], batch["mask"])
        labels = batch["labels"].long().view(-1)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, logits, labels = self._shared_step(batch)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, logits, labels = self._shared_step(batch)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, logits, labels = self._shared_step(batch)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> dict:
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
