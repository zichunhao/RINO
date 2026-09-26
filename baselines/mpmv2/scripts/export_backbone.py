"""Export the pretrained JetBackbone from an MPMv2 pretraining checkpoint.

Classifier (src.models.classifier.Classifier) loads its backbone via
``torch.load(backbone_path)``, expecting a pickled ``JetBackbone`` instance.
This script loads a Lightning checkpoint produced by ``scripts/train.py
experiment=pretrain_rino``, extracts the ``csts_emb``, ``csts_id_emb``, and
``encoder`` submodules, wraps them in a ``JetBackbone``, and pickles it.

Usage:
    python scripts/export_backbone.py \
        --ckpt <path/to/pretrain_run>/checkpoints/ \
        --output <path/to/pretrain_run>/backbone.pkl
"""

import argparse
import inspect
import logging
from pathlib import Path

import rootutils
import torch as T

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from src.models.mpm import MaskedParticleModelling as MPM  # noqa: E402
from src.models.utils import JetBackbone  # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _resolve_ckpt(ckpt: str) -> Path:
    """Accept a file path or a directory; if a directory, pick the highest-step
    ``{step}.ckpt`` produced by the default pretrain callback."""
    p = Path(ckpt)
    if p.is_file():
        return p
    if p.is_dir():
        candidates = [c for c in p.glob("*.ckpt") if c.stem.isdigit()]
        if not candidates:
            raise FileNotFoundError(f"No numeric-step .ckpt files under {p}")
        return max(candidates, key=lambda c: int(c.stem))
    raise FileNotFoundError(f"Neither a file nor a directory: {p}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to a Lightning .ckpt file, OR a directory containing step-named ckpts",
    )
    parser.add_argument("--output", required=True, help="Output path for pickled JetBackbone")
    args = parser.parse_args()

    ckpt_path = _resolve_ckpt(args.ckpt)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Resolved checkpoint: {ckpt_path}")

    log.info(f"Loading MPM Lightning checkpoint from {ckpt_path}")
    # The checkpoint pickles the model hyperparameters (functools.partial
    # objects), so it needs a full unpickle; Lightning versions that expose
    # weights_only otherwise defer to torch's weights-only default.
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(MPM.load_from_checkpoint).parameters:
        load_kwargs["weights_only"] = False
    model = MPM.load_from_checkpoint(str(ckpt_path), **load_kwargs)
    model.eval()

    log.info("Assembling JetBackbone from pretrained submodules")
    backbone = JetBackbone(
        csts_emb=model.csts_emb,
        csts_id_emb=model.csts_id_emb,  # may be None if use_id=false
        encoder=model.encoder,
    )
    backbone.eval()

    log.info(f"Saving pickled JetBackbone to {out_path}")
    T.save(backbone, str(out_path))
    log.info("Done")


if __name__ == "__main__":
    main()
