"""Batch processing for JetCLR+iBOT+KoLeo training.

Single-model contrastive learning:
- **NT-Xent** on CLS tokens across different kT clustering views
- **iBOT** with stop-gradient: unmasked view output is the target,
  masked view output is the prediction (cross-entropy with pT rank embedding)
- **KoLeo** entropy maximization on backbone representations
"""

import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from augmentations import Augmenter
from utils.logger import LOGGER

module_dir: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = module_dir.parent


@contextmanager
def maybe_autocast(device: torch.device, use_bf16: bool):
    """Context manager that applies bfloat16 autocast when use_bf16 is True."""
    if use_bf16:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            yield
    else:
        yield


def _get_nprongs(view_key: str, mask: torch.Tensor) -> torch.Tensor:
    """Return the RG scale for a view as a (B,) float tensor."""
    if view_key == "ALL":
        return mask.sum(dim=-1).float()
    n = int(view_key.replace("subjet", ""))
    return torch.full(
        (mask.shape[0],), float(n), dtype=torch.float32, device=mask.device
    )


def _select_cluster_views(
    nprongs_choices: list, num_views: int, exclude_keys: set[str] | None = None
) -> list[str]:
    """Select cluster views based on nprongs choices."""
    exclude_keys = exclude_keys or set()
    available = [
        n
        for n in nprongs_choices
        if ("ALL" if n == "ALL" else f"subjet{n}") not in exclude_keys
    ]
    if num_views < len(available):
        selected_nprongs = random.sample(available, num_views)
    elif num_views == len(available):
        selected_nprongs = available
    else:
        raise ValueError(
            f"Cannot generate {num_views} views: only {len(available)} choices remain "
            f"after excluding {exclude_keys} from {nprongs_choices}"
        )

    final_selected_nprongs = []
    for nprong in selected_nprongs:
        if isinstance(nprong, list):
            final_selected_nprongs.append(random.choice(nprong))
        else:
            final_selected_nprongs.append(nprong)

    return [
        f"subjet{nprong}" if nprong != "ALL" else "ALL"
        for nprong in final_selected_nprongs
    ]


def _select_smear_views(smear_choices: list, num_views: int) -> list[str]:
    """Select smear views based on smear choices."""
    if num_views > len(smear_choices):
        raise ValueError(
            f"Cannot generate {num_views} views with only {len(smear_choices)} smear choices"
        )
    return random.sample(smear_choices, num_views)


def _get_view_data(
    view_key: str,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract particle features, jets, and mask for a named view key."""
    if view_key in ("ALL", "no_smear"):
        return particles, jets, mask

    views = batch["views"]
    if view_key not in views:
        raise KeyError(f"View key {view_key} not found in batch views")

    view_data = views[view_key]
    view_particles = view_data["features"].to(dtype=torch.float32, device=device)
    view_mask = view_data["mask"].to(dtype=torch.bool, device=device)
    view_jets = (
        view_data["jets"].to(dtype=torch.float32, device=device)
        if "jets" in view_data
        else jets.to(dtype=torch.float32, device=device)
    )
    return view_particles, view_jets, view_mask


def process_batch(
    backbone: nn.Module,
    projection_head: nn.Module,
    ibot_head: nn.Module | None,
    ibot_pos_embedding: nn.Module | None,
    recon_head: nn.Module | None,
    ntxent_loss: nn.Module,
    ibot_loss: nn.Module | None,
    recon_loss: nn.Module | None,
    koleo_loss: nn.Module | None,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    augmenter: Augmenter,
    optimizer: torch.optim.Optimizer | None = None,
    accelerator: Accelerator | None = None,
    use_bf16: bool = False,
    truncate_seq: bool = True,
) -> dict[str, float]:
    """Process a batch through the JetCLR training objective.

    Single-model architecture:
    - NT-Xent contrastive loss on CLS tokens across kT clustering views
    - Optional iBOT masked prediction using stop-gradient on the unmasked view
    - Optional reconstruction loss (MAE-style) on masked particle features
    - Optional KoLeo entropy maximization on backbone representations

    Returns dict of scalar metrics.
    """
    _device = accelerator.device if accelerator is not None else device

    particles = batch["sequence"].to(dtype=torch.float32, device=_device)
    mask = batch["mask"].to(dtype=torch.bool, device=_device)
    jets = batch["class_"].to(dtype=torch.float32, device=_device)

    if truncate_seq:
        max_len = int(mask.sum(dim=1).max().item())
        particles = particles[:, :max_len]
        mask = mask[:, :max_len]

    batch_size = particles.shape[0]
    augmentation_params = config["augmentation_params"]

    # ------------------------------------------------------------------ #
    # Collect all views (global + local) for contrastive loss              #
    # ------------------------------------------------------------------ #
    all_view_reps: list[torch.Tensor] = []          # CLS reps for NT-Xent
    all_view_proj_outputs: list[torch.Tensor] = []  # projection head outputs for NT-Xent
    all_backbone_reps: list[torch.Tensor] = []      # for KoLeo (global views only)
    # Track which view gave us the full-jet ("ALL") particle features
    cached_full_particle_features: torch.Tensor | None = None

    global_view_count = 0

    for view_config in augmentation_params["global"]:
        view_reps, view_projs, particle_feats = _process_view_config(
            backbone=backbone,
            projection_head=projection_head,
            particles=particles,
            jets=jets,
            mask=mask,
            view_config=view_config,
            batch=batch,
            augmenter=augmenter,
            device=_device,
            use_bf16=use_bf16,
        )
        all_view_reps.extend(view_reps)
        all_view_proj_outputs.extend(view_projs)
        all_backbone_reps.extend(view_reps)  # global views for KoLeo
        global_view_count += len(view_reps)
        if particle_feats is not None and cached_full_particle_features is None:
            cached_full_particle_features = particle_feats

    for view_config in augmentation_params.get("local", []):
        view_reps, view_projs, _ = _process_view_config(
            backbone=backbone,
            projection_head=projection_head,
            particles=particles,
            jets=jets,
            mask=mask,
            view_config=view_config,
            batch=batch,
            augmenter=augmenter,
            device=_device,
            use_bf16=use_bf16,
        )
        all_view_reps.extend(view_reps)
        all_view_proj_outputs.extend(view_projs)

    # ------------------------------------------------------------------ #
    # NT-Xent contrastive loss on projection head outputs                  #
    # ------------------------------------------------------------------ #
    with maybe_autocast(_device, use_bf16):
        ntxent_loss_value, ntxent_components = ntxent_loss(
            views=all_view_proj_outputs,
            return_components=True,
        )

    total_loss = ntxent_loss_value
    loss_components = ntxent_components

    # ------------------------------------------------------------------ #
    # KoLeo loss (global views only)                                       #
    # ------------------------------------------------------------------ #
    if koleo_loss is not None and global_view_count > 0:
        global_reps = all_backbone_reps[:global_view_count]
        with maybe_autocast(_device, use_bf16):
            koleo_loss_value, koleo_components = koleo_loss(
                student_global_reps=global_reps,
                return_components=True,
            )
        total_loss = total_loss + koleo_loss_value
        loss_components.update(koleo_components)

    # ------------------------------------------------------------------ #
    # iBOT loss (stop-gradient masked prediction)                          #
    # ------------------------------------------------------------------ #
    if ibot_loss is not None and ibot_head is not None:
        ibot_loss_value, ibot_components = _compute_ibot_loss(
            backbone=backbone,
            ibot_head=ibot_head,
            ibot_pos_embedding=ibot_pos_embedding,
            ibot_loss=ibot_loss,
            particles=particles,
            jets=jets,
            mask=mask,
            cached_full_particle_features=cached_full_particle_features,
            device=_device,
            use_bf16=use_bf16,
        )
        total_loss = total_loss + ibot_loss_value
        loss_components.update(ibot_components)

    # ------------------------------------------------------------------ #
    # Reconstruction loss (MAE-style masked prediction)                    #
    # ------------------------------------------------------------------ #
    if recon_loss is not None and recon_head is not None:
        recon_loss_value, recon_components = _compute_recon_loss(
            backbone=backbone,
            recon_head=recon_head,
            recon_loss=recon_loss,
            particles=particles,
            jets=jets,
            mask=mask,
            device=_device,
            use_bf16=use_bf16,
        )
        total_loss = total_loss + recon_loss_value
        loss_components.update(recon_components)

    # ------------------------------------------------------------------ #
    # Optimization step                                                    #
    # ------------------------------------------------------------------ #
    if optimizer is not None:
        optimizer.zero_grad()

        if accelerator is not None:
            accelerator.backward(total_loss)
        else:
            total_loss.backward()

        grad_clip = config["training"].get("grad_clip", None)
        if grad_clip is not None and grad_clip > 0:
            clip_fn = (
                accelerator.clip_grad_norm_
                if accelerator is not None
                else torch.nn.utils.clip_grad_norm_
            )
            clip_fn(backbone.parameters(), max_norm=grad_clip)
            clip_fn(projection_head.parameters(), max_norm=grad_clip)
            if ibot_head is not None:
                clip_fn(ibot_head.parameters(), max_norm=grad_clip)
            if ibot_pos_embedding is not None:
                clip_fn(ibot_pos_embedding.parameters(), max_norm=grad_clip)
            if recon_head is not None:
                clip_fn(recon_head.parameters(), max_norm=grad_clip)

        optimizer.step()

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #
    loss_dict: dict[str, float] = {
        "loss": total_loss.item(),
        "batch_size": batch_size,
    }
    loss_dict.update(loss_components)
    return loss_dict


def _process_view_config(
    backbone: nn.Module,
    projection_head: nn.Module,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    view_config: dict,
    batch: dict[str, torch.Tensor],
    augmenter: Augmenter,
    device: torch.device,
    use_bf16: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor | None]:
    """Process a single view configuration through backbone + projection head.

    Returns (reps, proj_outputs, full_particle_features).
    full_particle_features is set only for the "ALL" view.
    """
    is_precomputed = any(
        comp["type"].lower() in ["cluster", "smear"]
        for comp in view_config["components"]
    )

    if is_precomputed:
        return _process_precomputed_views(
            backbone, projection_head, particles, jets, mask,
            view_config, batch, device, use_bf16,
        )
    else:
        return _process_computed_views(
            backbone, projection_head, particles, jets, mask,
            view_config, augmenter, device, use_bf16,
        )


def _process_precomputed_views(
    backbone: nn.Module,
    projection_head: nn.Module,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    view_config: dict,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    use_bf16: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor | None]:
    """Process pre-computed (cluster/smear) views."""
    if "views" not in batch:
        raise ValueError(
            "Pre-computed view augmentation requested but no views provided in batch"
        )

    view_component = next(
        comp for comp in view_config["components"]
        if comp["type"].lower() in ["cluster", "smear"]
    )
    view_type = view_component["type"].lower()

    if view_type == "cluster":
        view_choices = view_component["args"]["nprongs_choices"]
        selected_views = _select_cluster_views(view_choices, view_config["num"])
    elif view_type == "smear":
        view_choices = view_component["args"]["smear_choices"]
        selected_views = _select_smear_views(view_choices, view_config["num"])
    else:
        raise ValueError(f"Unsupported pre-computed view type: {view_type}")

    reps = []
    proj_outputs = []
    captured_particle_features = None

    for view_key in selected_views:
        view_particles, view_jets, view_mask = _get_view_data(
            view_key, particles, jets, mask, batch, device,
        )
        view_nprongs = _get_nprongs(view_key, view_mask)

        with maybe_autocast(device, use_bf16):
            rep, particle_features = backbone(
                particles=view_particles,
                jets=view_jets.to(device),
                mask=view_mask,
                nprongs=view_nprongs.to(device),
            )
            proj_out = projection_head(rep)

        reps.append(rep)
        proj_outputs.append(proj_out)

        if view_key == "ALL" and captured_particle_features is None:
            captured_particle_features = particle_features

    return reps, proj_outputs, captured_particle_features


def _process_computed_views(
    backbone: nn.Module,
    projection_head: nn.Module,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    view_config: dict,
    augmenter: Augmenter,
    device: torch.device,
    use_bf16: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], None]:
    """Process on-the-fly augmented views."""
    reps = []
    proj_outputs = []

    for _ in range(view_config["num"]):
        aug_parts, aug_jets, aug_mask = augmenter(
            particles.clone(), jets.clone(), mask.clone(),
            view_config["components"],
        )
        view_nprongs = aug_mask.sum(dim=-1).float()

        with maybe_autocast(device, use_bf16):
            rep, _ = backbone(
                particles=aug_parts, jets=aug_jets,
                mask=aug_mask, nprongs=view_nprongs,
            )
            proj_out = projection_head(rep)

        reps.append(rep)
        proj_outputs.append(proj_out)

    return reps, proj_outputs, None


def _compute_ibot_loss(
    backbone: nn.Module,
    ibot_head: nn.Module,
    ibot_pos_embedding: nn.Module | None,
    ibot_loss: nn.Module,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    cached_full_particle_features: torch.Tensor | None,
    device: torch.device,
    use_bf16: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute iBOT loss with stop-gradient.

    The unmasked (clean) view's particle-level features serve as the target
    (with stop-gradient applied). The masked view's features are the student
    predictions. Cross-entropy is computed between the two through the iBOT head.

    This follows the original iBOT paper's formulation where masking is treated
    as a view transformation, and the model must predict masked token
    representations from the unmasked context.
    """
    # 1. Get target: unmasked particle features (stop-gradient)
    if cached_full_particle_features is not None:
        target_particle_features = cached_full_particle_features.detach()
    else:
        with torch.no_grad():
            with maybe_autocast(device, use_bf16):
                _, target_particle_features = backbone(
                    particles=particles, jets=jets, mask=mask,
                    nprongs=mask.sum(dim=-1).float(),
                )

    # 2. Create particle mask for iBOT
    particle_mask = ibot_loss.create_particle_mask(mask, device)

    # 3. Forward pass with masked particles
    masked_particles = particles.clone()
    masked_particles[particle_mask] = 0.0

    # Pass unmasked particles as coords so PE sees true positions
    backbone_module = getattr(backbone, "module", backbone)
    coords = (
        particles
        if (
            backbone_module.pos_encoding is not None
            or backbone_module.rel_pos_bias is not None
        )
        else None
    )

    with maybe_autocast(device, use_bf16):
        _, student_particle_features = backbone(
            particles=masked_particles, jets=jets, mask=mask,
            coords=coords, nprongs=mask.sum(dim=-1).float(),
        )

    # 4. Add positional embedding (pT rank embedding) if configured
    pos_emb = None
    if ibot_pos_embedding is not None:
        emb_module = getattr(ibot_pos_embedding, "module", ibot_pos_embedding)
        with maybe_autocast(device, use_bf16):
            indices = emb_module.input_indices
            pe_input = particles if not indices else particles[..., indices]
            pos_emb = ibot_pos_embedding(pe_input)

    student_pf = student_particle_features
    if pos_emb is not None:
        student_pf = student_particle_features + pos_emb

    target_pf = target_particle_features
    if pos_emb is not None:
        target_pf = target_particle_features + pos_emb

    # 5. Project through iBOT head
    B, N, D = student_pf.shape
    with maybe_autocast(device, use_bf16):
        student_ibot_pred = ibot_head(
            student_pf.reshape(-1, D)
        ).reshape(B, N, -1)

    with torch.no_grad():
        with maybe_autocast(device, use_bf16):
            target_ibot_pred = ibot_head(
                target_pf.reshape(-1, D)
            ).reshape(B, N, -1)

    # 6. iBOT loss (cross-entropy between masked student and stop-grad target)
    with maybe_autocast(device, use_bf16):
        ibot_loss_value, ibot_components = ibot_loss.forward_masked(
            student_particle_predictions=student_ibot_pred,
            teacher_particle_predictions=target_ibot_pred,
            particle_mask=particle_mask,
            valid_mask=mask,
            return_components=True,
        )

    return ibot_loss_value, ibot_components


def _compute_recon_loss(
    backbone: nn.Module,
    recon_head: nn.Module,
    recon_loss: nn.Module,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    use_bf16: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute MAE-style reconstruction loss on masked particles.

    Mask particles, run through backbone, decode with recon_head, and
    compute MSE against the original particle features. No teacher needed.
    """
    # 1. Create particle mask
    particle_mask = recon_loss.create_particle_mask(mask, device)

    # 2. Forward pass with masked particles
    masked_particles = particles.clone()
    masked_particles[particle_mask] = 0.0

    # Pass unmasked particles as coords so PE sees true positions
    backbone_module = getattr(backbone, "module", backbone)
    coords = (
        particles
        if (
            backbone_module.pos_encoding is not None
            or backbone_module.rel_pos_bias is not None
        )
        else None
    )

    with maybe_autocast(device, use_bf16):
        _, student_particle_features = backbone(
            particles=masked_particles, jets=jets, mask=mask,
            coords=coords, nprongs=mask.sum(dim=-1).float(),
        )

    # 3. Decode: map d_model -> part_dim
    B, N, D = student_particle_features.shape
    with maybe_autocast(device, use_bf16):
        predicted = recon_head(
            student_particle_features.reshape(-1, D)
        ).reshape(B, N, -1)

    # 4. MSE loss on masked positions against original input features
    with maybe_autocast(device, use_bf16):
        recon_loss_value, recon_components = recon_loss(
            predicted=predicted,
            target=particles,
            particle_mask=particle_mask,
            valid_mask=mask,
            return_components=True,
        )

    return recon_loss_value, recon_components
