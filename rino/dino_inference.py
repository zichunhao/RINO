import argparse
from pathlib import Path
import yaml
import os
import torch
from tqdm import tqdm

from utils.device import get_available_device
from utils.producers import get_models, get_models_finetune, get_dataloader_and_config, get_config
from utils.logger import LOGGER, configure_logger

module_dir: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = module_dir.parent
torch.set_float32_matmul_precision("high")


def load_model(config: dict, device: torch.device):
    LOGGER.debug(f"Infer part_dim from dataloader config")
    dataloader_config = get_config(config=config, mode="inference")

    particle_features = dataloader_config.outputs.sequence
    part_dim = len(particle_features)
    # jet_features = dataloader_config.outputs.class_
    # jet_dim = len(jet_features)
    
    has_new_head = "head_params" in config
    if not has_new_head:
        # For DINO, we only need the teacher model for inference
        LOGGER.info("Loading DINO teacher model for inference")
        _, model, part_batch_norm = get_models(
            part_dim=part_dim,
            config=config,
            mode="inference",
            device=device,
        )
    else:
        # For classification heads, we load the teacher and the head
        LOGGER.info("Loading finetuned DINO model with new head for inference")
        model, part_batch_norm = get_models_finetune(
            part_dim=part_dim,
            config=config,
            mode="inference",
            device=device,
            train_head=False,  # No new head for inference
        )
    return model, part_batch_norm, has_new_head


def load_embedding_weights(model, embedding_tag: str, config: dict, device: torch.device):
    """
    Load embedding weights for a specific dataset based on the embedding_tag.
    
    Args:
        model: The model to load embeddings into
        embedding_tag: Tag indicating which embeddings to load (e.g., 'jetclass', 'toptagging')
        config: Configuration dictionary
        device: Device to load weights to
    """
    # Get embedding path from config
    embedding_paths = config["inference"].get("embeddings", {})
    embedding_path = embedding_paths.get(embedding_tag)
    
    if not embedding_path:
        LOGGER.info(f"No specific embedding path for {embedding_tag}, using default checkpoint embeddings")
        return
    
    # Replace placeholders in path
    embedding_path = embedding_path.replace("PROJECT_ROOT", str(PROJECT_ROOT))
    embedding_path = embedding_path.replace("JOBNAME", config.get("name", ""))
    
    if not Path(embedding_path).exists():
        LOGGER.warning(f"Embedding path {embedding_path} does not exist, using default checkpoint embeddings")
        return
    
    try:
        LOGGER.info(f"Loading embedding weights from {embedding_path} for {embedding_tag}")
        
        # Extract weights
        state_dict = torch.load(embedding_path, map_location=device)
        teacher_state = state_dict["teacher"]
        
        # Get the right model and load embeddings
        # Handle potential wrapping (module., etc.)
        target_model = model
        
        # If it's ModelWithNewHead, get backbone
        if hasattr(model, 'backbone'):
            target_model = target_model.backbone
        
        # Handle potential DDP/module wrapping
        if hasattr(target_model, 'module'):
            target_model = target_model.module
            
        # Load embedding weights
        target_model.load_embedding_weights(teacher_state)
        
        LOGGER.info(f"Successfully loaded embedding weights for {embedding_tag}")
        
    except Exception as e:
        LOGGER.error(f"Failed to load embedding weights from {embedding_path}: {e}")
        LOGGER.info(f"Continuing with default checkpoint embeddings for {embedding_tag}")


@torch.no_grad()
def inference(
    config: dict, 
    device: torch.device, 
    include_head_output: bool = False, 
    include_input: bool = False
):
    LOGGER.info(f"PyTorch version: {torch.__version__}")
    LOGGER.info(f"CUDA version: {torch.version.cuda}")
    LOGGER.info(f"cuDNN version: {torch.backends.cudnn.version()}")
    LOGGER.info(f"Inference with config: {config}")

    # Load the teacher model for inference
    model, part_batch_norm, has_new_head = load_model(config=config, device=device)
    model.eval()
    if part_batch_norm is not None:
        part_batch_norm.eval()

    splits = config["inference"].get("splits", ["train", "val", "test"])
    LOGGER.info(f"Splits to process: {splits}")

    for split in splits:
        LOGGER.info(f"Processing split: {split}")
        
        # Get dataloader and check for embedding_tag
        dataloader, dataloader_config = get_dataloader_and_config(
            config=config,
            mode="inference",
            split=split,
        )
        
        # Load appropriate embedding weights based on embedding_tag
        split_config = config["inference"]["dataloader"][split]
        embedding_tag = split_config.get("embedding_tag")
        
        if embedding_tag:
            LOGGER.info(f"Split {split} uses embedding_tag: {embedding_tag}")
            load_embedding_weights(model, embedding_tag, config, device)
        else:
            LOGGER.info(f"Split {split} has no embedding_tag, using default embeddings")

        results = {
            "rep": [],  # Store representations from teacher model
        }
        
        if has_new_head:
            if not include_head_output:
                LOGGER.warning(
                    "--include-head-output is on for finetuned models with new head; logits will be saved."
                )
            results["logits"] = []
        else:
            if include_head_output:
                results["proj"] = []
            else:
                LOGGER.info(
                    "Only saving representations from the teacher model; not the head's output"
                )
        
        if include_input:
            LOGGER.info("Storing inputs (particles and jets) in results")
            results["jets"] = []  # Store jets
            results["particles"] = []  # Store particles
            results["mask"] = []  # Store mask for particles

        batches_per_file = config["inference"].get("batches_per_file")
        current_batch = 0
        file_idx = 0

        with tqdm(dataloader, desc=f"Inference {split}", unit="batch") as pbar:
            for batch in pbar:
                particles = torch.tensor(batch["sequence"], dtype=torch.float32).to(
                    device
                )
                jets = torch.tensor(batch["class_"], dtype=torch.float32).to(device)
                mask = torch.tensor(batch["mask"], dtype=torch.bool).to(device)
                if part_batch_norm is not None:
                    particles = part_batch_norm(particles, mask=mask)
                    
                if include_input:
                    results["jets"].append(jets.cpu())
                    results["particles"].append(particles.cpu())
                    results["mask"].append(mask.cpu())

                # Get representation from teacher model (no head)
                if not has_new_head:
                    proj, rep = model(particles=particles, jets=jets, mask=mask)
                    results["rep"].append(rep.cpu())
                    if include_head_output:
                        results["proj"].append(proj.cpu())
                else:
                    logits, rep = model(
                        particles=particles,
                        jets=jets,
                        mask=mask,
                        include_dino_head=False,
                    )
                    results["rep"].append(rep.cpu())
                    results["logits"].append(logits.cpu())
                    
                # Store auxiliary information
                aux = batch["aux"]
                for aux_key, aux_val in aux.items():
                    if aux_key not in results:
                        results[aux_key] = []
                    results[aux_key].append(torch.tensor(aux_val).cpu())

                if batches_per_file and current_batch >= batches_per_file:
                    save_batches(
                        results=results, config=config, split=split, file_idx=file_idx
                    )
                    current_batch = 0
                    file_idx += 1
                    results = {k: [] for k in results.keys()}

                current_batch += 1

        # Save the remaining batches or save all
        if len(next(iter(results.values()))) != 0:
            save_batches(results=results, config=config, split=split, file_idx=file_idx)

        # free memory
        del dataloader
        torch.cuda.empty_cache()


def save_batches(
    results: dict[str, list[torch.Tensor]],
    config: dict,
    split: str,
    file_idx: int | None,
):
    job_name = config.get("name", "")
    epoch = config["inference"]["load_epoch"]

    output_dir = config["inference"]["output_dir"]
    output_dir = output_dir.replace("PROJECT_ROOT", str(PROJECT_ROOT))
    output_dir = output_dir.replace("JOBNAME", job_name)
    output_dir = output_dir.replace("EPOCHNUM", str(epoch))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine results
    for key in results:
        try:
            results[key] = torch.cat(results[key], dim=0)
        except RuntimeError as e:
            # Handle size mismatch by padding tensors
            tensor_list = results[key]
            
            if len(tensor_list) == 0:
                continue

            max_sizes = list(tensor_list[0].shape)
            for tensor in tensor_list[1:]:
                for i in range(1, len(tensor.shape)):  # Skip dim 0 (batch dimension)
                    max_sizes[i] = max(max_sizes[i], tensor.shape[i])
            
            # Pad tensors to match max sizes
            padded_tensors = []
            for tensor in tensor_list:
                # Calculate padding needed for each dimension
                padding = []
                for i in range(len(tensor.shape) - 1, 0, -1):  # Reverse order for F.pad
                    diff = max_sizes[i] - tensor.shape[i]
                    padding.extend([0, diff])  # [left_pad, right_pad] for each dim
                
                if any(p > 0 for p in padding):
                    padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
                    padded_tensors.append(padded_tensor)
                else:
                    padded_tensors.append(tensor)
            
            results[key] = torch.cat(padded_tensors, dim=0)

    if "label" in results and "logits" in results:
        logits = results["logits"]
        labels = results["label"]
        if "jetclass" in split:
            # remove leptonic top (label 9)
            mask = (labels != 9)
            mask_logits = logits[mask]
            mask_labels = labels[mask]
        else:
            mask_logits = logits
            mask_labels = labels

        # Calculate accuracy
        if mask_logits.shape[-1] == 1:
            # Binary classification
            preds = (mask_logits[:, 0] > 0).long()
        else:
            # Multi-class classification
            preds = mask_logits.argmax(dim=-1)

        acc = (preds == mask_labels).float().mean().item()
        LOGGER.info(f"Accuracy for {split}, file {file_idx}: {acc:.4f}")
            

    # Save results
    output_filename = config["inference"]["output_filename"]
    output_filename = output_filename.replace("SPLIT", split)
    output_filename = output_filename.replace("JOBNAME", job_name)
    output_filename = output_filename.replace("EPOCHNUM", str(epoch))

    if file_idx is not None:
        # Add file index before the file extension
        name, ext = os.path.splitext(output_filename)
        output_filename = f"{name}-{file_idx}{ext}"

    output_path = output_dir / output_filename
    torch.save(results, output_path)
    LOGGER.info(f"Results for {split} (batch {file_idx}) saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for DINO model")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--include-head-output",
        action="store_true",
        help="Also include the head's output in the results. Always true for finetuned models.",
    )
    parser.add_argument(
        "--include-input",
        action="store_true",
        help="Include the original inputs (particles and jets) in the output.",
    )
    parser.add_argument(
        "-lv",
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "-lf",
        "--log-file",
        type=str,
        default=None,
        help="Path to the log file. If not specified, logs will be written to stdout.",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    configure_logger(
        logger=LOGGER,
        name="DINO Inference",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    device = config.get("device", None)
    if device is None:
        device = get_available_device()
    else:
        device = torch.device(device)

    inference(config, device, include_head_output=args.include_head_output, include_input=args.include_input)