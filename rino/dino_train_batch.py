import random
from pathlib import Path
from typing import Any
from utils.logger import LOGGER
import torch
import torch.nn as nn
from torch import optim
from accelerate import Accelerator
from augmentations import Augmenter

module_dir: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = module_dir.parent
# torch.set_float32_matmul_precision("high")


def process_view(
    student: nn.Module,
    teacher: nn.Module | None,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    view_config: dict,
    batch: dict[str, torch.Tensor],
    part_batch_norm: nn.Module | None,
    augmenter: Augmenter,
    accelerator: Accelerator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Process a single view configuration and return student and teacher outputs.

    Args:
        student: Student model
        teacher: Teacher model (None for local views)
        particles: Input particle features
        jets: Input jet features
        mask: Input mask
        view_config: Configuration for this view
        batch: Full batch dictionary
        part_batch_norm: Optional batch normalization module
        augmenter: Data augmentation module
        accelerator: Optional Accelerator instance to handle distributed training

    Returns:
        tuple of (student_output, teacher_output)
        teacher_output will be None for local views
    """
    # Check if this view uses pre-computed data from dataloader
    if any(comp["type"].lower() in ["cluster", "smear"] for comp in view_config["components"]):
        return process_precomputed_view(
            student=student,
            teacher=teacher,
            particles=particles,
            jets=jets,
            mask=mask,
            view_config=view_config,
            batch=batch,
            part_batch_norm=part_batch_norm,
            accelerator=accelerator,
        )
    else:
        return process_computed_view(
            student=student,
            teacher=teacher,
            particles=particles,
            jets=jets,
            mask=mask,
            view_config=view_config,
            augmenter=augmenter,
        )


def process_precomputed_view(
    student: nn.Module,
    teacher: nn.Module | None,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    view_config: dict,
    batch: dict[str, torch.Tensor],
    part_batch_norm: nn.Module | None,
    accelerator: Accelerator | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    """Process views that use pre-computed data from the dataloader (cluster, smear, etc.)."""
    if "views" not in batch:
        raise ValueError(
            "Pre-computed view augmentation requested but no views provided in batch"
        )

    views = batch["views"]
    student_outputs = []
    student_reps = []
    teacher_outputs = []
    teacher_reps = []

    # Get the view component configuration
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

    for view_key in selected_views:
        # Handle the special "ALL" case for cluster views
        if view_key in ["ALL", "no_smear"]:
            view_particles = particles
            view_mask = mask
            view_jets = jets
        else:
            # Get pre-computed view data
            if view_key not in views:
                raise KeyError(f"View key {view_key} not found in batch views")

            view_data = views[view_key]
            
            # Convert to tensors and move to appropriate device
            if accelerator is None:
                # Manual device handling when not using Accelerate
                view_particles = torch.tensor(
                    view_data["features"], dtype=torch.float32
                ).to(particles.device, non_blocking=True)
                view_mask = torch.tensor(view_data["mask"], dtype=torch.bool).to(
                    mask.device, non_blocking=True
                )
                # Handle jets - use smeared jets if available, otherwise use original
                if "jets" in view_data:
                    view_jets = torch.tensor(view_data["jets"], dtype=torch.float32).to(
                        jets.device, non_blocking=True
                    )
                else:
                    view_jets = jets
            else:
                view_particles = torch.tensor(
                    view_data["features"], dtype=torch.float32
                ).to(accelerator.device)
                view_mask = torch.tensor(view_data["mask"], dtype=torch.bool).to(
                    accelerator.device
                )
                # Handle jets - use smeared jets if available, otherwise use original
                if "jets" in view_data:
                    view_jets = torch.tensor(view_data["jets"], dtype=torch.float32).to(
                        accelerator.device
                    )
                else:
                    view_jets = jets.to(accelerator.device)

            # Apply batch normalization if provided
            if part_batch_norm is not None:
                view_particles = part_batch_norm(view_particles, mask=view_mask)

        # Determine target device
        if accelerator is not None:
            device = accelerator.device
        else:
            device = particles.device

        # Get student output
        student_output, student_rep = student(
            particles=view_particles,
            jets=(view_jets.clone()).to(device),
            mask=view_mask,
        )
        student_outputs.append(student_output)
        student_reps.append(student_rep)

        # Get teacher output if provided
        if teacher is not None:
            with torch.no_grad():
                teacher_output, teacher_rep = teacher(
                    particles=view_particles,
                    jets=(view_jets.clone()).to(device),
                    mask=view_mask,
                )
                teacher_outputs.append(teacher_output)
                teacher_rep.detach()
                teacher_reps.append(teacher_rep)

    return (student_outputs, student_reps), (
        (teacher_outputs, teacher_reps) if teacher is not None else None
    )


def _select_cluster_views(nprongs_choices: list, num_views: int) -> list[str]:
    """Select cluster views based on nprongs choices."""
    if num_views > len(nprongs_choices):
        raise ValueError(
            f"Cannot generate {num_views} views with only {len(nprongs_choices)} nprongs choices"
        )
    
    selected_nprongs = random.sample(nprongs_choices, num_views)
    return [f"subjet{nprong}" if nprong != "ALL" else "ALL" for nprong in selected_nprongs]


def _select_smear_views(smear_choices: list, num_views: int) -> list[str]:
    """Select smear views based on smear choices."""
    if num_views > len(smear_choices):
        raise ValueError(
            f"Cannot generate {num_views} views with only {len(smear_choices)} smear choices"
        )
    
    return random.sample(smear_choices, num_views)

def process_computed_view(
    student: nn.Module,
    teacher: nn.Module | None,
    particles: torch.Tensor,
    jets: torch.Tensor,
    mask: torch.Tensor,
    view_config: dict,
    augmenter: Augmenter,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    """Process views using runtime augmentation through the Augmenter."""
    student_outputs = []
    student_reps = []
    teacher_outputs = []
    teacher_reps = []

    for _ in range(view_config["num"]):
        augmented_parts, augmented_jets, augmented_mask = augmenter(
            particles.clone(),
            jets.clone(),
            mask.clone(),
            view_config["components"],
        )

        # Get student output
        student_output, student_rep = student(
            particles=augmented_parts,
            jets=augmented_jets,
            mask=augmented_mask,
        )
        student_outputs.append(student_output)
        student_reps.append(student_rep)

        # Get teacher output if provided
        if teacher is not None:
            with torch.no_grad():
                teacher_output, teacher_rep = teacher(
                    particles=augmented_parts,
                    jets=augmented_jets,
                    mask=augmented_mask,
                )
                teacher_outputs.append(teacher_output)
                teacher_rep.detach()
                teacher_reps.append(teacher_rep)

    return (student_outputs, student_reps), (
        (teacher_outputs, teacher_reps) if teacher is not None else None
    )


def process_batch(
    student: nn.Module,
    teacher: nn.Module,
    dino_loss: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    augmenter: Augmenter,
    part_batch_norm: nn.Module | None = None,
    optimizer: optim.Optimizer | None = None,
    accelerator: Accelerator | None = None,
) -> dict[str, float]:
    """Process a batch of data through the DINO model.

    This function has been refactored to be more modular and maintainable.
    """
    # Prepare input tensors
    if accelerator is None:
        # Manual device handling when not using Accelerate
        particles = torch.tensor(batch["sequence"], dtype=torch.float32).to(
            device, non_blocking=True
        )
        mask = torch.tensor(batch["mask"], dtype=torch.bool).to(
            device, non_blocking=True
        )
        jets = torch.tensor(batch["class_"], dtype=torch.float32).to(
            device, non_blocking=True
        )
    else:
        # need to send data to the correct device when creating tensors
        particles = torch.tensor(batch["sequence"], dtype=torch.float32).to(
            accelerator.device
        )
        mask = torch.tensor(batch["mask"], dtype=torch.bool).to(accelerator.device)
        jets = torch.tensor(batch["class_"], dtype=torch.float32).to(accelerator.device)

    batch_size = particles.shape[0]

    if part_batch_norm is not None:
        particles = part_batch_norm(particles, mask=mask)

    augmentation_params = config["augmentation_params"]

    # Process global views
    all_student_outputs = []
    all_student_reps = []
    all_teacher_outputs = []
    all_teacher_reps = []
    all_teacher_argmax = []

    # Handle global views
    for view_config in augmentation_params["global"]:
        (student_outputs, student_reps), (teacher_outputs, teacher_reps) = process_view(
            student=student,
            teacher=teacher,
            particles=particles,
            jets=jets,
            mask=mask,
            view_config=view_config,
            batch=batch,
            part_batch_norm=part_batch_norm,
            augmenter=augmenter,
            accelerator=accelerator,
        )
        all_student_outputs.extend(student_outputs)
        all_student_reps.extend(student_reps)
        all_teacher_outputs.extend(teacher_outputs)
        all_teacher_reps.extend(teacher_reps)
        for teacher_rep in teacher_reps:
            teacher_rep_argmax = torch.argmax(teacher_rep, dim=-1)
            all_teacher_argmax.append(teacher_rep_argmax)

    # Handle local views
    for view_config in augmentation_params["local"]:
        (student_outputs, _), _ = process_view(
            student=student,
            teacher=None,  # No teacher for local views
            particles=particles,
            jets=jets,
            mask=mask,
            view_config=view_config,
            batch=batch,
            part_batch_norm=part_batch_norm,
            augmenter=augmenter,
            accelerator=accelerator,
        )
        all_student_outputs.extend(student_outputs)

    # Concatenate all outputs
    student_output = torch.cat(all_student_outputs)
    teacher_output = torch.cat(all_teacher_outputs)
    all_student_reps = torch.cat(all_student_reps)
    all_teacher_reps = torch.cat(all_teacher_reps) if all_teacher_reps else None

    # Compute loss
    loss, loss_components = dino_loss(
        student_output=student_output,
        teacher_output=teacher_output,
        student_backbone_output=all_student_reps,
        teacher_backbone_output=all_teacher_reps,
        return_components=True,
    )

    # Optimization step if optimizer is provided
    if optimizer is not None:
        optimizer.zero_grad()

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

        # Gradient clipping
        grad_clip = config["training"].get("grad_clip", None)
        if grad_clip is not None and grad_clip > 0:
            if accelerator is not None:
                # Use accelerator's clip_grad_norm_ for proper gradient clipping in distributed settings
                accelerator.clip_grad_norm_(student.parameters(), max_norm=grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=grad_clip)

        optimizer.step()

    # Compute metrics
    all_teacher_argmax = torch.cat(all_teacher_argmax)
    # unique dominating dimension
    unique, counts = torch.unique(all_teacher_argmax, return_counts=True)
    num_unique = len(unique) / all_teacher_argmax.shape[-1]

    loss_dict = {
        "loss": loss.item(),
        # "dom_unique": num_unique,
        "batch_size": batch_size,
    }
    for key, value in loss_components.items():
        loss_dict[key] = value

    # # Sort counts in descending order to get actual top counts
    # sorted_counts, _ = torch.sort(counts, descending=True)

    # for i in range(3):
    #     if len(sorted_counts) > i:
    #         loss_dict[f"dom_top{i+1}"] = sorted_counts[i] / sum(counts)

    return loss_dict