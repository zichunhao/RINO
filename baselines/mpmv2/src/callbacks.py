import sys
import logging
from pytorch_lightning.callbacks import TQDMProgressBar, Callback

log = logging.getLogger(__name__)


class StderrProgressBar(TQDMProgressBar):
    """Redirects progress bar to stderr so it doesn't pollute log files."""

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.file = sys.stderr
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.file = sys.stderr
        return bar


class EpochLossPrinter(Callback):
    """Prints train and validation loss to the log after each epoch."""

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train/total_loss")
        if loss is not None:
            log.info(f"Epoch {trainer.current_epoch} | train/total_loss: {loss:.4f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("valid/total_loss")
        if loss is not None:
            log.info(f"Epoch {trainer.current_epoch} | valid/total_loss: {loss:.4f}")
