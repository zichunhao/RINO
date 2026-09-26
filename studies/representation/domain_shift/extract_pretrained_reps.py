#!/usr/bin/env python3
"""Extract pretrained (pre-finetuning) jet representations.

Runs a frozen pretrained encoder over inference splits (by default the JetNet
and JetClass test sets) and saves the pooled representations in the same
layout as ``dino/dino_inference.py``:

    <output-dir>/output_<split>_<tag>-0.pt
        rep:   (N, d) float32 pooled representations
        label: (N,)   jet labels (plus any other ``aux`` field of the dataloader)

``compute_domain_shift.py`` reads these files exactly like the finetuned
inference outputs, which gives the pretrained-representation domain-shift
table (App. "Domain Shift in Representation Space").

Encoders (``--model-type``, see ``representation_utils.py``):
    dino      DINO/iBOT pretraining checkpoint; the EMA teacher backbone is used.
    jetclr    JetCLR / JetCLR-scale pretraining checkpoint.
    backbone  ``{"backbone": state_dict}`` from baselines/scripts/convert_checkpoint.py
              (MPMv1, MPMv2, JetCLR, OmniJet-alpha).

Split definitions are read from ``inference.dataloader.<split>`` of
``--data-config`` (default: ``--model-config``); any RINO pretraining config
defines ``test_jetnet`` and ``test_jetclass``.

Usage:
    # RINO (teacher backbone, native CLS pooling)
    python studies/representation/domain_shift/extract_pretrained_reps.py \
        --model-type dino --model-config configs/dino/<pretrain>.yaml \
        --load-epoch best --output-dir experiments/<run>/pretrained-reps

    # Converted baseline backbone (mean pooling, shared baseline architecture)
    python studies/representation/domain_shift/extract_pretrained_reps.py \
        --model-type backbone \
        --model-config studies/representation/baseline-backbone.yaml \
        --checkpoint <path/to/backbone.pt> \
        --data-config configs/dino/<pretrain>.yaml \
        --output-dir experiments/<run>/pretrained-reps
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from representation_utils import (  # noqa: E402
    LOGGER,
    MODEL_TYPES,
    POOLINGS,
    autocast,
    batch_inputs,
    build_dataloader,
    configure_logger,
    encode,
    get_part_dim,
    load_config,
    load_encoder,
    resolve_checkpoint,
    resolve_device,
)

MODE = "inference"


@torch.no_grad()
def extract_split(
    backbone: torch.nn.Module,
    dataloader,
    device: torch.device,
    pooling: str,
    use_bf16: bool,
    max_jets: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pooled representations and ``aux`` fields for every jet of a dataloader.

    Args:
        max_jets: Stop after this many jets (in loader order); ``None`` reads all.

    Returns:
        ``{"rep": (N, d) float32, <aux key>: (N, ...)}``.
    """
    reps: list[torch.Tensor] = []
    aux: dict[str, list[torch.Tensor]] = {}
    n_seen = 0
    for batch in tqdm(dataloader, desc="Extracting", unit="batch"):
        particles, jets, mask = batch_inputs(batch, device)
        take = particles.shape[0]
        if max_jets is not None:
            take = min(take, max_jets - n_seen)
        with autocast(device, use_bf16):
            rep = encode(backbone, particles, mask, jets, pooling=pooling)
        reps.append(rep[:take].float().cpu())
        for key, val in batch["aux"].items():
            aux.setdefault(key, []).append(torch.as_tensor(val)[:take].cpu())
        n_seen += take
        if max_jets is not None and n_seen >= max_jets:
            break

    if not reps:
        raise RuntimeError("The dataloader returned no batches")
    results = {"rep": torch.cat(reps, dim=0)}
    results.update({key: torch.cat(vals, dim=0) for key, vals in aux.items()})
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pretrained jet representations for the domain-shift study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-type", required=True, choices=MODEL_TYPES)
    parser.add_argument(
        "--model-config",
        required=True,
        help="Pretraining config (dino/jetclr) or a config with models.backbone (backbone).",
    )
    ckpt = parser.add_mutually_exclusive_group()
    ckpt.add_argument("--checkpoint", default=None, help="Checkpoint file.")
    ckpt.add_argument(
        "--load-epoch",
        default=None,
        help="Epoch (int or 'best') resolved via training.checkpoints_dir of --model-config.",
    )
    parser.add_argument(
        "--data-config",
        default=None,
        help="Config whose inference.dataloader defines the splits (default: --model-config).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["test_jetnet", "test_jetclass"]
    )
    parser.add_argument("--pooling", default="native", choices=POOLINGS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tag", default="pretrained", help="Output files are output_<split>_<tag>-0.pt."
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override the split batch size.")
    parser.add_argument(
        "--max-jets",
        type=int,
        default=None,
        help="Read at most this many jets per split, in loader order (default: all).",
    )
    parser.add_argument(
        "--allow-missing-keys",
        action="store_true",
        help="backbone type: keep the initialisation of tensors absent from the "
        "checkpoint (e.g. the input layer of a converted OmniJet-alpha backbone).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seeds any freshly initialised layer.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 autocast on CUDA.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing outputs.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logger(LOGGER, log_file=args.log_file, log_level=args.log_level)
    if args.model_type != "backbone" and args.allow_missing_keys:
        raise ValueError("--allow-missing-keys only applies to --model-type backbone")

    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config) if args.data_config else model_config
    if args.checkpoint is None and args.load_epoch is None:
        if args.model_type == "backbone":
            raise ValueError("--model-type backbone requires --checkpoint")
        args.load_epoch = "best"
    checkpoint = resolve_checkpoint(model_config, args.checkpoint, args.load_epoch)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for split in args.splits:
        out_path = output_dir / f"output_{split}_{args.tag}-0.pt"
        if out_path.exists() and not args.overwrite:
            LOGGER.info(f"{out_path} exists; skipping (use --overwrite to recompute)")
            continue
        todo.append((split, out_path))
    if not todo:
        return

    device = resolve_device(args.device)
    use_bf16 = bool(model_config.get("use_bf16", False)) and not args.no_bf16
    torch.manual_seed(args.seed)
    part_dim = get_part_dim(data_config, MODE, todo[0][0])
    backbone, _head = load_encoder(
        model_type=args.model_type,
        model_config=model_config,
        checkpoint=checkpoint,
        part_dim=part_dim,
        device=device,
        allow_missing_keys=args.allow_missing_keys,
    )
    LOGGER.info(f"Loaded {args.model_type} encoder from {checkpoint} (pooling={args.pooling})")

    for split, out_path in todo:
        dataloader = build_dataloader(data_config, MODE, split, args.batch_size)
        results = extract_split(
            backbone, dataloader, device, args.pooling, use_bf16, args.max_jets
        )
        torch.save(results, out_path)
        LOGGER.info(f"Saved {tuple(results['rep'].shape)} representations to {out_path}")


if __name__ == "__main__":
    main()
