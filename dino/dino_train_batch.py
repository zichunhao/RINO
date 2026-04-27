import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from utils.logger import LOGGER
import torch
import torch.nn as nn
from torch import optim
from accelerate import Accelerator
from augmentations import Augmenter
from augmentations.pre_loading import apply_jetclr_view, apply_lorentz_view

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


def process_view(
    student_backbone: nn.Module,
    student_dino_head: nn.Module,
    teacher_backbone: nn.Module | None,
    teacher_dino_head: nn.Module | None,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    view_config: dict,
    batch: dict[str, torch.Tensor],
    augmenter: Augmenter,
    accelerator: Accelerator | None = None,
    use_bf16: bool = False,
    dino_scale_embedding: nn.Module | None = None,
    exclude_view_keys: set[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[str]]:
    """Process a single view configuration and return student and teacher outputs.

    Args:
        student_backbone: Student backbone model
        student_dino_head: Student DINO head
        teacher_backbone: Teacher backbone model (None for local views)
        teacher_dino_head: Teacher DINO head (None for local views)
        particles: Input particle features
        jets: Input jet features
        mask: Input mask
        view_config: Configuration for this view
        batch: Full batch dictionary
        augmenter: Data augmentation module
        accelerator: Optional Accelerator instance to handle distributed training
        use_bf16: Whether to use bfloat16 autocast for forward passes
        exclude_view_keys: View keys to exclude from selection (e.g. teacher-selected
            views passed to local view calls to prevent trivial same-input pairs).

    Returns:
        tuple of (student_outputs, teacher_outputs, captured_teacher_particles, selected_view_keys)
        teacher_outputs and captured_teacher_particles are None for local views.
        captured_teacher_particles is set only when an "ALL" view was processed.
        selected_view_keys is the list of view keys actually used for this view config.
    """
    if any(
        comp["type"].lower() in ["cluster", "smear", "jetclr", "lorentz"]
        for comp in view_config["components"]
    ):
        return process_precomputed_view(
            student_backbone=student_backbone,
            student_dino_head=student_dino_head,
            teacher_backbone=teacher_backbone,
            teacher_dino_head=teacher_dino_head,
            particles=particles,
            jets=jets,
            mask=mask,
            view_config=view_config,
            batch=batch,
            accelerator=accelerator,
            use_bf16=use_bf16,
            dino_scale_embedding=dino_scale_embedding,
            exclude_view_keys=exclude_view_keys,
        )
    else:
        outputs = process_computed_view(
            student_backbone=student_backbone,
            student_dino_head=student_dino_head,
            teacher_backbone=teacher_backbone,
            teacher_dino_head=teacher_dino_head,
            particles=particles,
            jets=jets,
            mask=mask,
            view_config=view_config,
            augmenter=augmenter,
            use_bf16=use_bf16,
            dino_scale_embedding=dino_scale_embedding,
        )
        # On-the-fly views don't use named keys; return empty list.
        return (*outputs, [])


def process_precomputed_view(
    student_backbone: nn.Module,
    student_dino_head: nn.Module,
    teacher_backbone: nn.Module | None,
    teacher_dino_head: nn.Module | None,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    view_config: dict,
    batch: dict[str, torch.Tensor],
    accelerator: Accelerator | None = None,
    use_bf16: bool = False,
    dino_scale_embedding: nn.Module | None = None,
    exclude_view_keys: set[str] | None = None,
) -> tuple[
    list[torch.Tensor], list[torch.Tensor] | None, torch.Tensor | None, list[str]
]:
    """Process views that use pre-computed data from the dataloader (cluster, smear, etc.).

    Returns a four-tuple: (student_outputs, teacher_outputs, captured_teacher_particles,
    selected_view_keys).
    captured_teacher_particles is the teacher backbone's per-particle features from the
    "ALL" view (i.e. the full unmasked jet), or None if no "ALL" view was processed.
    The caller can reuse this to skip a redundant clean teacher forward pass for iBOT.
    selected_view_keys is the list of view keys actually processed; callers can pass this
    as exclude_view_keys to subsequent local-view calls to prevent trivial same-input pairs.
    """
    if "views" not in batch:
        raise ValueError(
            "Pre-computed view augmentation requested but no views provided in batch"
        )

    views = batch["views"]
    student_dino_outputs = []
    student_dino_reps = []
    teacher_dino_outputs = []
    teacher_dino_reps = []
    captured_teacher_particles: torch.Tensor | None = None

    view_component = next(
        comp
        for comp in view_config["components"]
        if comp["type"].lower() in ["cluster", "smear", "jetclr", "lorentz"]
    )
    view_type = view_component["type"].lower()

    if view_type == "cluster":
        view_choices = view_component["args"]["nprongs_choices"]
        selected_views = _select_cluster_views(
            view_choices, view_config["num"], exclude_keys=exclude_view_keys
        )
    elif view_type == "smear":
        view_choices = view_component["args"]["smear_choices"]
        selected_views = _select_smear_views(view_choices, view_config["num"])
    elif view_type in ("jetclr", "lorentz"):
        # Each such view is re-sampled independently; keys are synthetic labels.
        selected_views = [f"{view_type}_{i}" for i in range(view_config["num"])]
    else:
        raise ValueError(f"Unsupported pre-computed view type: {view_type}")

    device = accelerator.device if accelerator is not None else particles.device

    for view_key in selected_views:
        if view_key in ["ALL", "no_smear"]:
            view_particles = particles
            view_mask = mask
            view_jets = jets
        elif view_type in ("jetclr", "lorentz"):
            if "raw" not in views:
                raise KeyError(
                    "jetclr/lorentz views require batch['views']['raw'] — use a "
                    "dataloader config that exposes raw (E, px, py, pz)"
                )
            raw_view = views["raw"]
            target_device = (
                accelerator.device if accelerator is not None else particles.device
            )
            raw_p4 = raw_view["features"].to(
                dtype=torch.float32, non_blocking=True, device=target_device
            )
            raw_jet_p4 = raw_view["jets"].to(
                dtype=torch.float32, non_blocking=True, device=target_device
            )
            raw_mask = raw_view["mask"].to(
                dtype=torch.bool, non_blocking=True, device=target_device
            )
            aug_args = view_component.get("args", {}) or {}
            if view_type == "jetclr":
                view_particles = apply_jetclr_view(raw_p4, raw_jet_p4, raw_mask, aug_args)
            else:
                view_particles = apply_lorentz_view(raw_p4, raw_jet_p4, raw_mask, aug_args)
            view_mask = raw_mask
            view_jets = jets
            # Stand-in view_data dict shape so the downstream code path below is
            # uniform; we short-circuit before it reads view_data below.
            view_data = None
        else:
            if view_key not in views:
                raise KeyError(f"View key {view_key} not found in batch views")

            view_data = views[view_key]

            if accelerator is None:
                view_particles = (
                    view_data["features"]
                    .to(dtype=torch.float32)
                    .to(non_blocking=True, device=particles.device)
                )
                view_mask = view_data["mask"].to(
                    dtype=torch.bool, non_blocking=True, device=mask.device
                )
                if "jets" in view_data:
                    view_jets = view_data["jets"].to(
                        dtype=torch.float32, non_blocking=True, device=jets.device
                    )
                else:
                    view_jets = jets
            else:
                view_particles = view_data["features"].to(
                    dtype=torch.float32, non_blocking=True, device=accelerator.device
                )
                view_mask = view_data["mask"].to(
                    dtype=torch.bool, non_blocking=True, device=accelerator.device
                )
                if "jets" in view_data:
                    view_jets = view_data["jets"].to(
                        dtype=torch.float32,
                        non_blocking=True,
                        device=accelerator.device,
                    )
                else:
                    view_jets = jets.to(
                        dtype=torch.float32,
                        non_blocking=True,
                        device=accelerator.device,
                    )

        view_nprongs = _get_nprongs(view_key, view_mask)

        with maybe_autocast(device, use_bf16):
            student_rep, student_particle_features = student_backbone(
                particles=view_particles,
                jets=view_jets.to(device),
                mask=view_mask,
                nprongs=view_nprongs.to(device),
            )
            if dino_scale_embedding is not None:
                log_n = torch.log(view_nprongs.clamp(min=1).to(student_rep.device))
                student_rep = student_rep + dino_scale_embedding(log_n)
            student_dino_output = student_dino_head(student_rep)
        student_dino_outputs.append(student_dino_output)
        student_dino_reps.append(student_rep)

        if teacher_backbone is not None and teacher_dino_head is not None:
            with torch.no_grad():
                with maybe_autocast(device, use_bf16):
                    teacher_rep, teacher_pf = teacher_backbone(
                        particles=view_particles,
                        jets=view_jets.to(device),
                        mask=view_mask,
                        nprongs=view_nprongs.to(device),
                    )
                    if dino_scale_embedding is not None:
                        log_n = torch.log(
                            view_nprongs.clamp(min=1).to(teacher_rep.device)
                        )
                        teacher_rep = teacher_rep + dino_scale_embedding(log_n)
                    teacher_dino_output = teacher_dino_head(teacher_rep)
                teacher_dino_outputs.append(teacher_dino_output)
                teacher_dino_reps.append(teacher_rep)
                # Cache particle-level features from the full-jet ("ALL") view so the
                # caller can skip a redundant clean teacher pass for iBOT.
                if view_key == "ALL" and captured_teacher_particles is None:
                    captured_teacher_particles = teacher_pf

    return (
        (student_dino_outputs, student_dino_reps),
        (
            (teacher_dino_outputs, teacher_dino_reps)
            if teacher_backbone is not None
            else None
        ),
        captured_teacher_particles,
        selected_views,
    )


def _get_nprongs(view_key: str, mask: torch.Tensor) -> torch.Tensor:
    """Return the RG scale for a view as a (B,) float tensor.

    For "ALL" views the scale is the actual particle count per jet (inferred
    from the validity mask).  For subjet views it is the hardcoded requested
    clustering level parsed from the view key, e.g. "subjet8" → 8.

    The hardcoded value is used for subjet views (rather than the filled count)
    because the RG scale should reflect the *requested* clustering level, not
    the effective number of non-empty subjets.  An all-padded subjet slot in a
    subjet16 view does not change the fact that the coarse-graining was
    performed at the 16-prong level.
    """
    if view_key == "ALL":
        return mask.sum(dim=-1).float()
    if view_key.startswith(("jetclr", "lorentz")):
        # Augmentation-only views don't coarse-grain; report the actual particle count.
        return mask.sum(dim=-1).float()
    n = int(view_key.replace("subjet", ""))
    return torch.full(
        (mask.shape[0],), float(n), dtype=torch.float32, device=mask.device
    )


def _select_cluster_views(
    nprongs_choices: list, num_views: int, exclude_keys: set[str] | None = None
) -> list[str]:
    """Select cluster views based on nprongs choices.

    Args:
        nprongs_choices: List of nprong values (int, "ALL", or nested list for random choice).
        num_views: Number of views to select.
        exclude_keys: View keys to exclude from selection (e.g. "ALL", "subjet6"). Used to
            prevent trivial same-input pairs when teacher-selected global views overlap with
            the local view pool.
    """
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


def process_computed_view(
    student_backbone: nn.Module,
    student_dino_head: nn.Module,
    teacher_backbone: nn.Module | None,
    teacher_dino_head: nn.Module | None,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    view_config: dict,
    augmenter: Augmenter,
    use_bf16: bool = False,
    dino_scale_embedding: nn.Module | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None, None]:
    """Process views using runtime augmentation through the Augmenter."""
    student_dino_outputs = []
    student_dino_reps = []
    teacher_dino_outputs = []
    teacher_dino_reps = []

    device = particles.device

    for _ in range(view_config["num"]):
        augmented_parts, augmented_jets, augmented_mask = augmenter(
            particles.clone(),
            jets.clone(),
            mask.clone(),
            view_config["components"],
        )

        view_nprongs_computed = augmented_mask.sum(dim=-1).float()
        with maybe_autocast(device, use_bf16):
            student_rep, student_particle_features = student_backbone(
                particles=augmented_parts,
                jets=augmented_jets,
                mask=augmented_mask,
                nprongs=view_nprongs_computed,
            )
            if dino_scale_embedding is not None:
                log_n = torch.log(
                    view_nprongs_computed.clamp(min=1).to(student_rep.device)
                )
                student_rep = student_rep + dino_scale_embedding(log_n)
            student_dino_output = student_dino_head(student_rep)
        student_dino_outputs.append(student_dino_output)
        student_dino_reps.append(student_rep)

        if teacher_backbone is not None and teacher_dino_head is not None:
            with torch.no_grad():
                with maybe_autocast(device, use_bf16):
                    teacher_rep, _ = teacher_backbone(
                        particles=augmented_parts,
                        jets=augmented_jets,
                        mask=augmented_mask,
                        nprongs=view_nprongs_computed,
                    )
                    if dino_scale_embedding is not None:
                        log_n = torch.log(
                            view_nprongs_computed.clamp(min=1).to(teacher_rep.device)
                        )
                        teacher_rep = teacher_rep + dino_scale_embedding(log_n)
                    teacher_dino_output = teacher_dino_head(teacher_rep)
                teacher_dino_outputs.append(teacher_dino_output)
                teacher_dino_reps.append(teacher_rep)

    return (
        (student_dino_outputs, student_dino_reps),
        (
            (teacher_dino_outputs, teacher_dino_reps)
            if teacher_backbone is not None
            else None
        ),
        None,
    )


def process_batch(
    student_backbone: nn.Module,
    student_dino_head: nn.Module,
    student_ibot_head: nn.Module,
    teacher_backbone: nn.Module,
    teacher_dino_head: nn.Module,
    teacher_ibot_head: nn.Module,
    ibot_pos_embedding: nn.Module | None,
    dino_loss: nn.Module,
    ibot_loss: nn.Module | None,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    augmenter: Augmenter,
    optimizer: optim.Optimizer | None = None,
    accelerator: Accelerator | None = None,
    gram_loss: nn.Module | None = None,
    koleo_loss: nn.Module | None = None,
    use_bf16: bool = False,
    truncate_seq: bool = True,
    dino_scale_embedding: nn.Module | None = None,
    step_index: int | None = None,
) -> dict[str, float]:
    """Process a batch through the full DINO/iBOT/Gram/KoLeo training objective.

    Each loss's effective weight is read from ``loss.get_current_weight()`` so
    that weight-warmup schedules are honoured transparently.

    Args:
        student_backbone: Student backbone model.
        student_dino_head: Student DINO projection head.
        student_ibot_head: Student iBOT projection head.
        teacher_backbone: Teacher backbone model (EMA of student).
        teacher_dino_head: Teacher DINO projection head.
        teacher_ibot_head: Teacher iBOT projection head.
        ibot_pos_embedding: Legacy external positional embedding for iBOT. Mutually
            exclusive with backbone-internal ``pos_encoding``. Prefer configuring
            ``pos_encoding_kwargs`` in the backbone instead.
        dino_loss: DINOLoss instance.
        ibot_loss: iBOTLoss instance, or None to disable.
        batch: Batch dict from the dataloader.
        device: Target device.
        config: Full training config dict.
        augmenter: Augmentation module.
        optimizer: If provided, a backward + optimizer step is performed.
        accelerator: Optional HuggingFace Accelerate instance.
        gram_loss: GramLoss instance, or None to disable.
        koleo_loss: KoLeoLoss instance, or None to disable.
        use_bf16: Enable bfloat16 autocast for forward passes.

    Returns:
        Dict of scalar metrics (loss components + batch_size).
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

    particle_mask = None
    if ibot_loss is not None:
        particle_mask = ibot_loss.create_particle_mask(mask, _device)

    augmentation_params = config["augmentation_params"]

    # Symmetric DINO: swap global/local views on odd steps so that
    # the distillation direction alternates (UV->IR and IR->UV).
    # Requires num_global_views == num_local_views.
    if augmentation_params.get("symmetric", False) and step_index is not None:
        if step_index % 2 == 1:
            augmentation_params = dict(augmentation_params)
            augmentation_params["global"] = config["augmentation_params"]["local"]
            augmentation_params["local"] = config["augmentation_params"]["global"]

    # ------------------------------------------------------------------ #
    # DINO view processing                                                 #
    # ------------------------------------------------------------------ #
    all_student_dino_outputs: list[torch.Tensor] = []
    all_student_dino_reps: list[torch.Tensor] = []
    all_teacher_dino_outputs: list[torch.Tensor] = []
    all_teacher_dino_reps: list[torch.Tensor] = []
    all_teacher_argmax: list[torch.Tensor] = []
    cached_teacher_particle_features: torch.Tensor | None = None
    # Accumulate view keys selected by the teacher across all global view configs.
    # These are passed as exclusions to local view selection to prevent trivial
    # same-input pairs (e.g. teacher-global "ALL" vs student-local "ALL").
    teacher_selected_view_keys: set[str] = set()

    for view_config in augmentation_params["global"]:
        (
            (student_dino_outputs, student_dino_reps),
            (
                teacher_dino_outputs,
                teacher_dino_reps,
            ),
            t_particles,
            selected_keys,
        ) = process_view(
            student_backbone=student_backbone,
            student_dino_head=student_dino_head,
            teacher_backbone=teacher_backbone,
            teacher_dino_head=teacher_dino_head,
            particles=particles,
            jets=jets,
            mask=mask,
            view_config=view_config,
            batch=batch,
            augmenter=augmenter,
            accelerator=accelerator,
            use_bf16=use_bf16,
            dino_scale_embedding=dino_scale_embedding,
        )
        all_student_dino_outputs.extend(student_dino_outputs)
        all_student_dino_reps.extend(student_dino_reps)
        all_teacher_dino_outputs.extend(teacher_dino_outputs)
        all_teacher_dino_reps.extend(teacher_dino_reps)
        for teacher_rep in teacher_dino_reps:
            all_teacher_argmax.append(torch.argmax(teacher_rep, dim=-1))
        if t_particles is not None and cached_teacher_particle_features is None:
            cached_teacher_particle_features = t_particles
        teacher_selected_view_keys.update(selected_keys)

    for view_config in augmentation_params["local"]:
        (student_dino_outputs, _), _, _, _ = process_view(
            student_backbone=student_backbone,
            student_dino_head=student_dino_head,
            teacher_backbone=None,
            teacher_dino_head=None,
            particles=particles,
            jets=jets,
            mask=mask,
            view_config=view_config,
            batch=batch,
            augmenter=augmenter,
            accelerator=accelerator,
            use_bf16=use_bf16,
            dino_scale_embedding=dino_scale_embedding,
            exclude_view_keys=teacher_selected_view_keys,
        )
        all_student_dino_outputs.extend(student_dino_outputs)

    student_dino_output = torch.cat(all_student_dino_outputs)
    teacher_dino_output = torch.cat(all_teacher_dino_outputs)
    all_student_dino_reps_cat = torch.cat(all_student_dino_reps)
    all_teacher_dino_reps_cat = (
        torch.cat(all_teacher_dino_reps) if all_teacher_dino_reps else None
    )

    # ------------------------------------------------------------------ #
    # DINO loss                                                            #
    # ------------------------------------------------------------------ #
    with maybe_autocast(_device, use_bf16):
        dino_loss_value, dino_loss_components = dino_loss(
            student_output=student_dino_output,
            teacher_output=teacher_dino_output,
            student_backbone_output=all_student_dino_reps_cat,
            teacher_backbone_output=all_teacher_dino_reps_cat,
            return_components=True,
        )

    total_loss = dino_loss_value
    loss_components = dino_loss_components

    # ------------------------------------------------------------------ #
    # KoLeo loss (global views of student backbone)                        #
    # ------------------------------------------------------------------ #
    if koleo_loss is not None:
        num_global_views = dino_loss.num_global_views
        samples_per_view = all_student_dino_reps_cat.shape[0] // (
            dino_loss.num_local_views + num_global_views
        )
        global_reps = all_student_dino_reps_cat[: samples_per_view * num_global_views]
        global_view_chunks = list(global_reps.chunk(num_global_views))

        with maybe_autocast(_device, use_bf16):
            koleo_loss_value, koleo_loss_components = koleo_loss(
                student_global_reps=global_view_chunks,
                return_components=True,
            )
        total_loss = total_loss + koleo_loss_value
        loss_components.update(koleo_loss_components)

    # ------------------------------------------------------------------ #
    # Shared clean-particle forward pass (Gram + iBOT both need it)       #
    # ------------------------------------------------------------------ #
    need_clean_pass = (gram_loss is not None) or (
        ibot_loss is not None
        and student_ibot_head is not None
        and teacher_ibot_head is not None
    )

    student_particle_features = None
    teacher_particle_features = None

    if need_clean_pass:
        if cached_teacher_particle_features is not None:
            # Reuse teacher particle features captured from the "ALL" global view —
            # avoids a redundant full-jet teacher forward pass.
            teacher_particle_features = cached_teacher_particle_features
        else:
            with torch.no_grad():
                with maybe_autocast(_device, use_bf16):
                    _, teacher_particle_features = teacher_backbone(
                        particles=particles,
                        jets=jets,
                        mask=mask,
                        nprongs=mask.sum(dim=-1).float(),
                    )

        if gram_loss is not None:
            with maybe_autocast(_device, use_bf16):
                _, student_particle_features = student_backbone(
                    particles=particles,
                    jets=jets,
                    mask=mask,
                    nprongs=mask.sum(dim=-1).float(),
                )

    # ------------------------------------------------------------------ #
    # Gram loss                                                            #
    # ------------------------------------------------------------------ #
    # Gram loss (DINOv3-style: clean input projected through iBOT head)   #
    # ------------------------------------------------------------------ #
    if gram_loss is not None:
        if student_particle_features is None:
            raise RuntimeError(
                "student_particle_features is None before gram_loss forward pass - "
                "the student clean-particle forward pass was not executed."
            )
        if student_ibot_head is None or teacher_ibot_head is None:
            raise RuntimeError(
                "DINOv3-style Gram loss requires iBOT heads, but student_ibot_head "
                "or teacher_ibot_head is None."
            )
        B_g, N_g, D_g = student_particle_features.shape
        with maybe_autocast(_device, use_bf16):
            student_gram_feats = student_ibot_head(
                student_particle_features.reshape(-1, D_g)
            ).reshape(B_g, N_g, -1)
        with torch.no_grad():
            with maybe_autocast(_device, use_bf16):
                teacher_gram_feats = teacher_ibot_head(
                    teacher_particle_features.reshape(-1, D_g)
                ).reshape(B_g, N_g, -1)
        with maybe_autocast(_device, use_bf16):
            gram_loss_value, gram_loss_components = gram_loss(
                student_particle_features=student_gram_feats,
                teacher_particle_features=teacher_gram_feats,
                valid_mask=mask,
                particle_mask=particle_mask,
                return_components=True,
            )
        total_loss = total_loss + gram_loss_value
        loss_components.update(gram_loss_components)

    # ------------------------------------------------------------------ #
    # iBOT loss                                                            #
    # ------------------------------------------------------------------ #
    if (
        ibot_loss is not None
        and student_ibot_head is not None
        and teacher_ibot_head is not None
        and particle_mask is not None
    ):
        # Sanity check: ibot_pos_embedding is the old external PE path.
        # It must not be used together with backbone-internal pos_encoding.
        student_backbone_module = getattr(student_backbone, "module", student_backbone)
        if (
            ibot_pos_embedding is not None
            and student_backbone_module.pos_encoding is not None
        ):
            raise ValueError(
                "Cannot use both ibot_pos_embedding (external) and backbone pos_encoding "
                "(internal) simultaneously — PE would be applied twice."
            )

        masked_particles = particles.clone()
        masked_particles[particle_mask] = 0.0

        # Pass unmasked particles as coords so the backbone's internal PE and
        # relative positional bias see true positions at zeroed-out particles.
        coords = (
            particles
            if (
                student_backbone_module.pos_encoding is not None
                or student_backbone_module.rel_pos_bias is not None
            )
            else None
        )

        with maybe_autocast(_device, use_bf16):
            _, student_particle_features_ibot = student_backbone(
                particles=masked_particles,
                jets=jets,
                mask=mask,
                coords=coords,
                nprongs=mask.sum(dim=-1).float(),
            )

        # External PE path (legacy, only active when backbone has no pos_encoding)
        pos_emb = None
        if ibot_pos_embedding is not None:
            emb_module = getattr(ibot_pos_embedding, "module", ibot_pos_embedding)
            with maybe_autocast(_device, use_bf16):
                indices = emb_module.input_indices
                pe_input = particles if not indices else particles[..., indices]
                pos_emb = ibot_pos_embedding(pe_input)

        student_pf = student_particle_features_ibot
        if pos_emb is not None:
            student_pf = student_particle_features_ibot + pos_emb

        with maybe_autocast(_device, use_bf16):
            batch_size_ibot, num_particles, feature_dim = student_pf.shape
            student_ibot_pred = student_ibot_head(
                student_pf.reshape(-1, feature_dim)
            ).reshape(batch_size_ibot, num_particles, -1)

        with torch.no_grad():
            teacher_pf = teacher_particle_features
            if pos_emb is not None:
                teacher_pf = teacher_particle_features + pos_emb

            with maybe_autocast(_device, use_bf16):
                teacher_ibot_pred = teacher_ibot_head(
                    teacher_pf.reshape(-1, feature_dim)
                ).reshape(batch_size_ibot, num_particles, -1)

        with maybe_autocast(_device, use_bf16):
            ibot_loss_value, ibot_loss_components = ibot_loss.forward_masked(
                student_particle_predictions=student_ibot_pred,
                teacher_particle_predictions=teacher_ibot_pred,
                particle_mask=particle_mask,
                valid_mask=mask,
                return_components=True,
            )

        total_loss = total_loss + ibot_loss_value
        loss_components.update(ibot_loss_components)

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
            clip_fn(student_backbone.parameters(), max_norm=grad_clip)
            clip_fn(student_dino_head.parameters(), max_norm=grad_clip)
            if student_ibot_head is not None:
                clip_fn(student_ibot_head.parameters(), max_norm=grad_clip)
            if ibot_pos_embedding is not None:
                clip_fn(ibot_pos_embedding.parameters(), max_norm=grad_clip)

        optimizer.step()

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #
    all_teacher_argmax_cat = torch.cat(all_teacher_argmax)
    unique, _ = torch.unique(all_teacher_argmax_cat, return_counts=True)
    num_unique = len(unique) / all_teacher_argmax_cat.shape[-1]

    loss_dict: dict[str, float] = {
        "loss": total_loss.item(),
        "batch_size": batch_size,
    }
    loss_dict.update(loss_components)
    return loss_dict
