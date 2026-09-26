"""Convert pretrained baseline checkpoints to the format expected by
dino/classification_train.py (i.e. {"backbone": <JetTransformerEncoder state_dict>}).

Supported baselines: mpmv1, mpmv2, jetclr, omnijet.

What each converter does (details in its docstring):
    mpmv1    MPMv1's input standardisation (Bert.node_norm) is folded into
             particle_embedding, so raw features give the pretrained embedding.
    mpmv2    MPMv2's own csts_emb replaces particle_embedding; the (trained)
             embedding norm part_norm is kept.
    jetclr   Already a JetTransformerEncoder state_dict; validated only.
    omnijet  The causal decoder's layers are renamed to encoder layers; its VQ
             token-embedding table is dropped, and particle_embedding is left
             to the finetuning model's random initialisation.

Usage:
    python baselines/scripts/convert_checkpoint.py \
        --model mpmv1 \
        --input experiments/mpmv1/.../best_001.ckpt \
        --output experiments/mpmv1/.../backbone.pt

    python baselines/scripts/convert_checkpoint.py \
        --model mpmv2 \
        --input experiments/mpmv2/.../last.ckpt \
        --output experiments/mpmv2/.../backbone.pt

    python baselines/scripts/convert_checkpoint.py \
        --model jetclr \
        --input experiments/jetclr/.../model_ep95.pt \
        --output experiments/jetclr/.../backbone.pt

    python baselines/scripts/convert_checkpoint.py \
        --model omnijet \
        --input experiments/omnijet_alpha/.../last.ckpt \
        --output experiments/omnijet_alpha/.../backbone.pt
"""

import argparse
import sys
from pathlib import Path
from collections import OrderedDict

import torch

# ---------------------------------------------------------------------------
# Reference JTE config shared by all RINO baselines
# ---------------------------------------------------------------------------
BASELINE_JTE_CONFIG = dict(
    part_dim=7,
    d_model=256,
    nhead=16,
    num_layers=8,
    pooling="mean",
    activation="SwiGLU",
    norm="RMSNorm",
    layer_scale_init=0.01,
    num_registers=4,
    apply_final_norm=True,
    apply_embedding_norm=True,
)


def _build_reference_model():
    """Build a fresh JetTransformerEncoder with the baseline config for validation."""
    # Import from baselines/models/ (same copy used by all baselines)
    baselines_dir = Path(__file__).resolve().parents[1]
    if str(baselines_dir) not in sys.path:
        sys.path.insert(0, str(baselines_dir))
    from models.jet_transformer_encoder import JetTransformerEncoder

    return JetTransformerEncoder(**BASELINE_JTE_CONFIG)


# ---------------------------------------------------------------------------
# MPMv1
# ---------------------------------------------------------------------------

MPMV1_NORM_KEYS = ("node_norm.means", "node_norm.vars")
MPMV1_NORM_EPS = 1e-8  # same epsilon as NewIterativeNormLayer.forward


def convert_mpmv1(ckpt_path: Path) -> OrderedDict:
    """Extract JTE backbone weights from an MPMv1 Lightning checkpoint.

    MPMv1 wraps JTE inside JTEAdapter, so trained weights live at
    ``state_dict["encoder.jte.*"]``.  The particle_embedding IS trained
    because JTEAdapter calls ``jte.particle_embedding()`` in forward.

    In pretraining, ``Bert.forward`` standardises the constituents with its
    input normaliser ``node_norm`` before JTEAdapter embeds them.  The
    finetuned JetTransformerEncoder has no such layer, so the normaliser is
    folded into ``particle_embedding`` (see :func:`fold_mpmv1_node_norm`).
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in raw:
        raise KeyError("MPMv1 checkpoint missing 'state_dict' key")
    sd = raw["state_dict"]

    prefix = "encoder.jte."
    backbone_sd = OrderedDict()
    dropped = []
    for k, v in sd.items():
        if k.startswith(prefix):
            backbone_sd[k[len(prefix):]] = v
        elif k in MPMV1_NORM_KEYS:
            dropped.append(k + " (folded into particle_embedding)")
        else:
            dropped.append(k)

    if not backbone_sd:
        raise ValueError(f"No keys with prefix '{prefix}' found in checkpoint")

    fold_mpmv1_node_norm(backbone_sd, sd)

    _print_summary("mpmv1", backbone_sd, dropped)
    return backbone_sd


def fold_mpmv1_node_norm(backbone_sd: OrderedDict, mpm_sd: dict) -> None:
    """Fold MPMv1's input normaliser into ``particle_embedding`` (in place).

    MPMv1 pretrains on ``particle_embedding(node_norm(x))`` with
    ``node_norm(x) = (x - m) / s``, where ``m = node_norm.means`` and
    ``s = sqrt(node_norm.vars) + 1e-8`` are the running statistics frozen in
    the checkpoint.  Since both maps are affine,

        W' = W / s,    b' = b - W' m

    gives ``W' x + b' == W node_norm(x) + b``, so the converted backbone
    reproduces the pretrained embedding when fed the raw features used at
    finetuning.  (``node_norm`` leaves padded slots untouched; those slots are
    masked out of attention and pooling, so their values do not matter.)

    Parameters
    ----------
    backbone_sd:
        Extracted JTE state_dict; its ``particle_embedding.*`` entries are
        replaced by the folded weights.
    mpm_sd:
        The full MPMv1 Lightning ``state_dict`` holding ``node_norm.*``.
    """
    missing = [k for k in MPMV1_NORM_KEYS if k not in mpm_sd]
    if missing:
        raise KeyError(f"MPMv1 checkpoint missing input normaliser key(s): {missing}")

    # Fold in float64 and cast back, to keep the fold exact to float32 precision.
    mean = mpm_sd["node_norm.means"].double().reshape(-1)
    scale = mpm_sd["node_norm.vars"].double().reshape(-1).sqrt() + MPMV1_NORM_EPS
    weight = backbone_sd["particle_embedding.weight"]
    bias = backbone_sd["particle_embedding.bias"]
    if weight.shape[1] != mean.numel():
        raise ValueError(
            f"node_norm has {mean.numel()} features but particle_embedding "
            f"expects {weight.shape[1]}"
        )

    folded_weight = weight.double() / scale[None, :]
    folded_bias = bias.double() - folded_weight @ mean
    backbone_sd["particle_embedding.weight"] = folded_weight.to(weight.dtype)
    backbone_sd["particle_embedding.bias"] = folded_bias.to(bias.dtype)

    n_seen = mpm_sd.get("node_norm.n")
    n_seen = f"{int(n_seen)} constituents" if n_seen is not None else "unknown n"
    print(f"  Folded node_norm ({n_seen}) into particle_embedding:")
    print(f"    mean: {[round(v, 4) for v in mean.tolist()]}")
    print(f"    std:  {[round(v, 4) for v in scale.tolist()]}")


# ---------------------------------------------------------------------------
# MPMv2
# ---------------------------------------------------------------------------

def convert_mpmv2(ckpt_path: Path) -> OrderedDict:
    """Extract JTE backbone weights from an MPMv2 Lightning checkpoint.

    MPMv2's wrapper bypasses JTE's particle_embedding (uses its own csts_emb
    nn.Linear with the same shape).  The JTE particle_embedding weights stored
    in the checkpoint are UNTRAINED (random init).  We replace them with
    csts_emb weights.

    The JTE embedding norm ``part_norm`` is kept as is.  The MPMv2 wrapper
    (``JetTransformerMLToolsWrapper.forward``) applies ``jte.part_norm`` to the
    embedded tokens before the register tokens and transformer, matching
    ``JetTransformerEncoder.forward``, so ``part_norm`` is trained during
    pretraining and the converted backbone (finetuned with
    ``apply_embedding_norm: true``) computes the pretrained function.  A
    checkpoint whose ``part_norm.weight`` is still at its initial value of
    ones comes from a pretraining run that did not apply the norm; a warning
    is printed in that case.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in raw:
        raise KeyError("MPMv2 checkpoint missing 'state_dict' key")
    sd = raw["state_dict"]

    prefix = "encoder.jte."
    backbone_sd = OrderedDict()
    dropped = []

    # Collect csts_emb weights for particle_embedding replacement
    csts_emb_weight = None
    csts_emb_bias = None

    for k, v in sd.items():
        if k == "csts_emb.weight":
            csts_emb_weight = v
            dropped.append(k + " (-> particle_embedding.weight)")
        elif k == "csts_emb.bias":
            csts_emb_bias = v
            dropped.append(k + " (-> particle_embedding.bias)")
        elif k.startswith(prefix):
            backbone_sd[k[len(prefix):]] = v
        else:
            dropped.append(k)

    if not backbone_sd:
        raise ValueError(f"No keys with prefix '{prefix}' found in checkpoint")

    # Replace untrained particle_embedding with csts_emb
    if csts_emb_weight is None:
        raise ValueError("MPMv2 checkpoint missing 'csts_emb.weight'")
    if csts_emb_bias is None:
        raise ValueError("MPMv2 checkpoint missing 'csts_emb.bias'")

    old_pe_w = backbone_sd.get("particle_embedding.weight")
    old_pe_b = backbone_sd.get("particle_embedding.bias")

    backbone_sd["particle_embedding.weight"] = csts_emb_weight
    backbone_sd["particle_embedding.bias"] = csts_emb_bias

    # Sanity check shapes
    if old_pe_w is not None and old_pe_w.shape != csts_emb_weight.shape:
        print(
            f"  WARNING: shape mismatch: particle_embedding.weight {old_pe_w.shape} "
            f"vs csts_emb.weight {csts_emb_weight.shape}"
        )
    if old_pe_b is not None and old_pe_b.shape != csts_emb_bias.shape:
        print(
            f"  WARNING: shape mismatch: particle_embedding.bias {old_pe_b.shape} "
            f"vs csts_emb.bias {csts_emb_bias.shape}"
        )

    print(f"  Replaced particle_embedding with csts_emb weights "
          f"(weight: {csts_emb_weight.shape}, bias: {csts_emb_bias.shape})")

    # An RMSNorm/LayerNorm weight that never received a gradient is still all
    # ones; finetuning would then apply a norm the pretrained layers never saw.
    part_norm_weight = backbone_sd.get("part_norm.weight")
    if part_norm_weight is not None and torch.all(part_norm_weight == 1):
        print(
            "  WARNING: part_norm.weight is still at its initial value (all ones), "
            "so this checkpoint was pretrained without the JTE embedding norm. "
            "Re-pretrain with the current MPMv2 wrapper, or finetune with "
            "apply_embedding_norm: false."
        )

    # The particle-ID embedding has no counterpart in JetTransformerEncoder.
    if any(k.startswith("csts_id_emb.") for k in sd):
        print(
            "  WARNING: checkpoint was pretrained with use_id=True; the "
            "csts_id_emb particle-ID embedding is dropped by the conversion."
        )

    _print_summary("mpmv2", backbone_sd, dropped)
    return backbone_sd


# ---------------------------------------------------------------------------
# JetCLR
# ---------------------------------------------------------------------------

def convert_jetclr(ckpt_path: Path) -> OrderedDict:
    """Validate a JetCLR checkpoint (already in the correct format)."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "backbone" not in raw:
        raise KeyError(
            f"JetCLR checkpoint missing 'backbone' key. "
            f"Found keys: {list(raw.keys())}"
        )
    backbone_sd = OrderedDict(raw["backbone"])
    print(f"  JetCLR checkpoint already in correct format ({len(backbone_sd)} keys)")
    _print_summary("jetclr", backbone_sd, dropped=[])
    return backbone_sd


# ---------------------------------------------------------------------------
# OmniJet-alpha
# ---------------------------------------------------------------------------

def convert_omnijet(ckpt_path: Path) -> OrderedDict:
    """Extract JTE backbone weights from an OmniJet-alpha Lightning checkpoint.

    OmniJet-alpha pretrains a ``JetTransformerDecoder`` (stored under
    ``state_dict["decoder.*"]``) on VQ-VAE token ids.  The decoder is built
    from the same TransformerEncoder/TransformerEncoderLayer classes as
    ``JetTransformerEncoder``, so its layer, register-token and norm weights
    load directly once the top-level module names are mapped:

        decoder.embed_norm.*       -> part_norm.*
        decoder.transformer.*      -> transformer_encoder.*
        decoder.register_tokens    -> register_tokens      (if present)
        decoder.final_norm.*       -> final_norm.*
        decoder.token_embedding.*  -> dropped
        decoder.causal_mask        -> dropped (buffer)
        head.*, vqvae.*            -> dropped (pretraining head, frozen tokenizer)

    The converted backbone differs from the pretrained decoder in two ways:

    * Input layer.  The decoder embeds discrete token ids with an
      ``nn.Embedding(vocab_size, d_model)`` table, while the encoder embeds
      continuous constituent features with ``particle_embedding =
      nn.Linear(part_dim, d_model)``.  There is no weight mapping between the
      two, so the token table is dropped and the converted checkpoint has no
      ``particle_embedding.*`` entries.  The finetuning loader
      (``load_state_dict(strict=False)``) keeps that layer at its random
      initialisation.
    * Attention.  The decoder was pretrained with a causal mask; the encoder
      that loads these weights attends bidirectionally.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in raw:
        raise KeyError("OmniJet checkpoint missing 'state_dict' key")
    sd = raw["state_dict"]

    prefix = "decoder."
    # Decoder module name -> JetTransformerEncoder module name (prefix match).
    renames = {
        "embed_norm.": "part_norm.",
        "transformer.": "transformer_encoder.",
    }
    backbone_sd = OrderedDict()
    dropped = []

    for k, v in sd.items():
        if not k.startswith(prefix):
            dropped.append(k)
            continue
        dk = k[len(prefix):]

        if dk == "causal_mask":
            dropped.append(k + " (causal-mask buffer)")
        elif dk.startswith("token_embedding."):
            dropped.append(k + " (VQ token table; no continuous-input counterpart)")
        else:
            for old, new in renames.items():
                if dk.startswith(old):
                    dk = new + dk[len(old):]
                    break
            # register_tokens and final_norm.* keep their names.
            backbone_sd[dk] = v

    if not backbone_sd:
        raise ValueError(f"No keys with prefix '{prefix}' found in checkpoint")

    print(
        "  particle_embedding.* is not in the checkpoint: the finetuning model "
        "keeps its random initialisation for the continuous input layer."
    )

    _print_summary("omnijet", backbone_sd, dropped)
    return backbone_sd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_summary(model_name, backbone_sd, dropped):
    print(f"\n{'=' * 60}")
    print(f"  {model_name.upper()} Conversion Summary")
    print(f"{'=' * 60}")
    print(f"  Extracted keys: {len(backbone_sd)}")
    print(f"  Dropped keys:   {len(dropped)}")

    # Show key tensor shapes
    print(f"\n  Key tensor shapes:")
    highlight_keys = [
        "particle_embedding.weight",
        "particle_embedding.bias",
        "part_norm.weight",
        "register_tokens",
        "transformer_encoder.layers.0.self_attn.in_proj_weight",
        "final_norm.weight",
    ]
    for k in highlight_keys:
        if k in backbone_sd:
            print(f"    {k}: {list(backbone_sd[k].shape)}")

    if dropped:
        print(f"\n  Dropped keys (first 15):")
        for k in dropped[:15]:
            print(f"    {k}")
        if len(dropped) > 15:
            print(f"    ... and {len(dropped) - 15} more")
    print()


def _verify(backbone_sd):
    """Load the converted state_dict into a fresh JTE to catch shape mismatches."""
    print("  Verifying against fresh JetTransformerEncoder ... ", end="", flush=True)
    model = _build_reference_model()
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(backbone_sd.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        print(f"\n  WARNING: {len(missing)} missing key(s): {sorted(missing)[:10]}")
    if unexpected:
        print(f"\n  WARNING: {len(unexpected)} unexpected key(s): {sorted(unexpected)[:10]}")

    # Filter to only matching keys for load test
    filtered = OrderedDict(
        {k: v for k, v in backbone_sd.items() if k in model_keys}
    )
    model.load_state_dict(filtered, strict=False)

    # Also verify shapes for matched keys
    shape_mismatches = []
    ref_sd = model.state_dict()
    for k in filtered:
        if ref_sd[k].shape != filtered[k].shape:
            shape_mismatches.append(
                f"    {k}: expected {list(ref_sd[k].shape)}, got {list(filtered[k].shape)}"
            )
    if shape_mismatches:
        print(f"\n  SHAPE MISMATCHES:")
        for m in shape_mismatches:
            print(m)
        raise RuntimeError("Shape mismatches detected; conversion is incorrect")

    if not missing and not unexpected:
        print("OK")
    else:
        print("OK (with warnings above)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CONVERTERS = {
    "mpmv1": convert_mpmv1,
    "mpmv2": convert_mpmv2,
    "jetclr": convert_jetclr,
    "omnijet": convert_omnijet,
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert baseline checkpoints to RINO finetune format"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(CONVERTERS.keys()),
        help="Baseline model type",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input checkpoint",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to save converted checkpoint",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification against a fresh JetTransformerEncoder",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Converting {args.model} checkpoint:")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")

    convert_fn = CONVERTERS[args.model]
    backbone_sd = convert_fn(args.input)

    # Verify against a fresh model
    if not args.skip_verify:
        _verify(backbone_sd)

    # Save in the expected format
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": backbone_sd}, args.output)
    print(f"  Saved to {args.output}")
    print(f"  File size: {args.output.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
