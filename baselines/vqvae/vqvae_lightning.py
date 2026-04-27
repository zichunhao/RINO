"""PyTorch Lightning wrapper for shared VQ-VAE training."""

import torch
from pytorch_lightning import LightningModule

from .vqvae_model import SharedVQVAE


class SharedVQVAELightning(LightningModule):
    """Lightning module for training the shared VQ-VAE.

    Parameters
    ----------
    model_kwargs:
        Keyword arguments for :class:`SharedVQVAE`.
    optimizer_kwargs:
        Keyword arguments for the optimizer (default: AdamW).
    lr:
        Learning rate.
    weight_decay:
        Weight decay for AdamW.
    warmup_steps:
        Number of linear warmup steps before cosine decay.
    """

    def __init__(
        self,
        model_kwargs: dict | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 5000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SharedVQVAE(**(model_kwargs or {}))
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

    def forward(self, particles, mask):
        return self.model(particles, mask)

    def _shared_step(self, batch, prefix: str):
        particles = batch["sequence"]  # (B, N, 7)
        mask = batch["mask"]  # (B, N)

        total_loss, metrics = self.model.compute_loss(particles, mask)

        for key, val in metrics.items():
            self.log(f"{prefix}/{key}", val, prog_bar=(key == "total_loss"))

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

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
