#!/usr/bin/env python3
"""Cosine similarity of jet representations across kT clustering scales.

Every jet is encoded once per scale: the unclustered constituents (``ALL``)
and each exclusive-kT subjet view (``subjet2`` ... ``subjet16``) stored by the
clustered JetClass dataset. The pooled representations are L2-normalised, and
for every pair of scales the per-jet cosine similarity is summarised over jets
(mean, std, median, min, max). The mean is what the scale-structure and
swapped-roles figures show (App. "Scale Structure").

With ``--head-space`` (``dino`` and ``jetclr`` models) the same statistics are
also computed after the projection head applied to the native pooled
representation: in the head output (the 2048-d DINO prototype scores for RINO)
and in the L2-normalised bottleneck that feeds the head's last layer.

Outputs in ``--output-dir``:
    cosine_similarity_stats.json
        scales                           scale names, in --scales order
        pairwise_cosine                  {"<a>_vs_<b>": {mean, std, median, min, max}}
        head_output_pairwise_cosine      (--head-space) same, head output
        head_bottleneck_pairwise_cosine  (--head-space) same, head bottleneck
      ``pairwise_cosine`` uses the schema of
      ``dino/diagnostics/scale_invariance_analysis.py``.
    scale_embeddings.pt  (--save-embeddings) {scale: (N, d) tensor, "labels": (N,)}

Usage:
    # RINO backbone and head (teacher; data = the config's clustered val split)
    python studies/representation/scale_structure/compute_scale_cosines.py \
        --model-type dino --model-config configs/dino/<pretrain>.yaml \
        --load-epoch best --head-space \
        --output-dir experiments/<run>/scale_structure

    # Converted baseline backbone, same jets
    python studies/representation/scale_structure/compute_scale_cosines.py \
        --model-type backbone \
        --model-config studies/representation/baseline-backbone.yaml \
        --checkpoint <path/to/backbone.pt> \
        --data-config configs/dino/<pretrain>.yaml \
        --output-dir experiments/<run>/scale_structure
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
    normalize,
    resolve_checkpoint,
    resolve_device,
)

MODE = "training"  # the clustered views are defined by the training dataloaders
DEFAULT_SCALES = ["ALL", "subjet2", "subjet3", "subjet4", "subjet6", "subjet8", "subjet16"]


def _view_inputs(
    batch: dict,
    scale: str,
    full: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(particles, jets, mask)`` of one scale; ``ALL`` is the unclustered jet."""
    if scale == "ALL":
        return full
    views = batch.get("views") or {}
    if scale not in views:
        raise KeyError(f"View '{scale}' not in the batch; available: {sorted(views)}")
    view = views[scale]
    particles = torch.as_tensor(np.asarray(view["features"]), dtype=torch.float32).to(device)
    mask = torch.as_tensor(np.asarray(view["mask"]), dtype=torch.bool).to(device)
    jets = full[1]
    if "jets" in view:
        jets = torch.as_tensor(np.asarray(view["jets"]), dtype=torch.float32).to(device)
    return particles, jets, mask


@torch.no_grad()
def extract_scale_embeddings(
    backbone: torch.nn.Module,
    dataloader,
    device: torch.device,
    scales: list[str],
    num_jets: int,
    pooling: str,
    use_bf16: bool,
) -> dict[str, torch.Tensor]:
    """Pooled representation of the same ``num_jets`` jets at every scale.

    Returns:
        ``{scale: (N, d) float32}`` plus ``"labels": (N,)`` when available.
    """
    embeddings: dict[str, list[torch.Tensor]] = {s: [] for s in scales}
    labels: list[torch.Tensor] = []
    n_seen = 0
    with tqdm(total=num_jets, desc="Extracting", unit="jet") as pbar:
        for batch in dataloader:
            full = batch_inputs(batch, device)
            take = min(full[0].shape[0], num_jets - n_seen)
            for scale in scales:
                particles, jets, mask = _view_inputs(batch, scale, full, device)
                with autocast(device, use_bf16):
                    rep = encode(
                        backbone, particles[:take], mask[:take], jets[:take], pooling
                    )
                embeddings[scale].append(rep.float().cpu())
            label = (batch.get("aux") or {}).get("label")
            if label is not None:
                labels.append(torch.as_tensor(np.asarray(label))[:take])
            n_seen += take
            pbar.update(take)
            if n_seen >= num_jets:
                break

    if n_seen < num_jets:
        LOGGER.warning(f"Split exhausted after {n_seen} jets (requested {num_jets})")
    result = {s: torch.cat(v, dim=0) for s, v in embeddings.items()}
    if labels:
        result["labels"] = torch.cat(labels, dim=0)
    return result


def pairwise_cosine_stats(
    embeddings: dict[str, torch.Tensor], scales: list[str]
) -> dict[str, dict[str, float]]:
    """Per-jet cosine statistics for every pair of scales, keyed ``"<a>_vs_<b>"``."""
    normed = {s: normalize(embeddings[s]) for s in scales}
    stats = {}
    for i, a in enumerate(scales):
        for b in scales[i + 1 :]:
            cos = (normed[a] * normed[b]).sum(dim=-1)
            stats[f"{a}_vs_{b}"] = {
                "mean": float(cos.mean()),
                "std": float(cos.std()),
                "median": float(cos.median()),
                "min": float(cos.min()),
                "max": float(cos.max()),
            }
    return stats


@torch.no_grad()
def head_space_embeddings(
    head: torch.nn.Module,
    embeddings: dict[str, torch.Tensor],
    scales: list[str],
    device: torch.device,
    chunk_size: int = 4096,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Project the per-scale representations through the head (float32).

    Returns:
        ``(output, bottleneck)``: the head output per scale and, for a
        ``DINOHead``, the L2-normalised bottleneck that feeds its last layer
        (``None`` for heads without that structure).
    """
    has_bottleneck = hasattr(head, "layers") and hasattr(head, "last_layer")
    head = head.float().to(device).eval()
    output: dict[str, torch.Tensor] = {}
    bottleneck: dict[str, torch.Tensor] = {}
    for scale in scales:
        outs, bots = [], []
        for x in embeddings[scale].split(chunk_size):
            x = x.float().to(device)
            outs.append(head(x).float().cpu())
            if has_bottleneck:
                h = x
                for layer in head.layers:
                    h = layer(h)
                if getattr(head, "l2_norm", False):
                    h = torch.nn.functional.normalize(h, dim=-1)
                bots.append(h.float().cpu())
        output[scale] = torch.cat(outs, dim=0)
        if has_bottleneck:
            bottleneck[scale] = torch.cat(bots, dim=0)
    return output, (bottleneck if has_bottleneck else None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-scale cosine similarity of pretrained jet representations.",
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
        help="Config whose training.dataloader.<split> reads the clustered JetClass "
        "data with subjet views (default: --model-config).",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--scales", nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--num-jets", type=int, default=10000)
    parser.add_argument("--pooling", default="native", choices=POOLINGS)
    parser.add_argument(
        "--head-space",
        action="store_true",
        help="Also compute the statistics after the projection head (dino/jetclr).",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--allow-missing-keys",
        action="store_true",
        help="backbone type: keep the initialisation of tensors absent from the checkpoint.",
    )
    parser.add_argument("--save-embeddings", action="store_true", help="Also save scale_embeddings.pt.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42, help="Seeds any freshly initialised layer.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 autocast on CUDA.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logger(LOGGER, log_file=args.log_file, log_level=args.log_level)
    if args.head_space and args.model_type == "backbone":
        raise ValueError("--head-space needs a model with a head (dino or jetclr)")
    if args.head_space and args.pooling != "native":
        raise ValueError("--head-space applies the head to the native pooled representation")
    if args.model_type != "backbone" and args.allow_missing_keys:
        raise ValueError("--allow-missing-keys only applies to --model-type backbone")

    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config) if args.data_config else model_config
    if args.checkpoint is None and args.load_epoch is None:
        if args.model_type == "backbone":
            raise ValueError("--model-type backbone requires --checkpoint")
        args.load_epoch = "best"
    checkpoint = resolve_checkpoint(model_config, args.checkpoint, args.load_epoch)

    device = resolve_device(args.device)
    use_bf16 = bool(model_config.get("use_bf16", False)) and not args.no_bf16
    torch.manual_seed(args.seed)
    backbone, head = load_encoder(
        model_type=args.model_type,
        model_config=model_config,
        checkpoint=checkpoint,
        part_dim=get_part_dim(data_config, MODE, args.split),
        device=device,
        allow_missing_keys=args.allow_missing_keys,
    )

    dataloader = build_dataloader(data_config, MODE, args.split, args.batch_size)
    embeddings = extract_scale_embeddings(
        backbone, dataloader, device, args.scales, args.num_jets, args.pooling, use_bf16
    )
    n_jets = int(embeddings[args.scales[0]].shape[0])

    stats = {
        "model_type": args.model_type,
        "model_config": args.model_config,
        "checkpoint": args.checkpoint,
        "load_epoch": args.load_epoch,
        "data_config": args.data_config or args.model_config,
        "split": args.split,
        "num_jets": n_jets,
        "pooling": args.pooling,
        "d_model": int(embeddings[args.scales[0]].shape[1]),
        "scales": list(args.scales),
        "pairwise_cosine": pairwise_cosine_stats(embeddings, args.scales),
    }
    if args.head_space:
        output, bottleneck = head_space_embeddings(head, embeddings, args.scales, device)
        stats["head_output_dim"] = int(output[args.scales[0]].shape[1])
        stats["head_output_pairwise_cosine"] = pairwise_cosine_stats(output, args.scales)
        if bottleneck is not None:
            stats["head_bottleneck_dim"] = int(bottleneck[args.scales[0]].shape[1])
            stats["head_bottleneck_pairwise_cosine"] = pairwise_cosine_stats(
                bottleneck, args.scales
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "cosine_similarity_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    LOGGER.info(f"Saved {stats_path}")
    if args.save_embeddings:
        torch.save(embeddings, output_dir / "scale_embeddings.pt")
        LOGGER.info(f"Saved {output_dir / 'scale_embeddings.pt'}")

    for key in ("pairwise_cosine", "head_output_pairwise_cosine"):
        if key in stats:
            means = [v["mean"] for v in stats[key].values()]
            LOGGER.info(
                f"{key}: {n_jets} jets, cross-scale mean cosine in "
                f"[{min(means):.3f}, {max(means):.3f}]"
            )


if __name__ == "__main__":
    main()
