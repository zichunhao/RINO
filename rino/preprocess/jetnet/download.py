from pathlib import Path
import argparse
import os

import numpy as np
import h5py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_jetnet_dataset(
    download_dir: str | Path | None, 
    jet_type: str,
    num_particles: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from jetnet.datasets import JetNet
    except ImportError:
        # pip install jetnet
        logger.error("JetNet package not found. Trying to install it...")
        os.system("pip install jetnet")
        from jetnet.datasets import JetNet
    
    particle_data, jet_data = JetNet.getData(
        jet_type=[jet_type], 
        data_dir=str(download_dir), 
        download=(download_dir is not None),
        num_particles=num_particles,
    )
    return particle_data, jet_data


def get_jetclass_format(
    particle_data: np.ndarray, 
    jet_data: np.ndarray
) -> dict[str, np.ndarray]:
    # jet data: (label, pt, eta, phi, mass, nparticles)
    jet_label = jet_data[:, 0].astype(int)
    jet_pt = jet_data[:, 1]
    jet_eta = jet_data[:, 2]
    # randomly generated phi from -pi to pi
    jet_phi = np.random.uniform(-np.pi, np.pi, size=jet_data.shape[0])
    jet_mass = jet_data[:, 3]
    jet_nparticles = jet_data[:, 4]

    jet_energy = np.sqrt(jet_mass**2 + jet_pt**2 * (np.cosh(jet_eta)**2))
    
    # particle data: (deta, dphi, dpt, mask)
    part_deta = particle_data[:, :, 0]
    part_dphi = particle_data[:, :, 1]
    part_dpt = particle_data[:, :, 2]
    part_mask = particle_data[:, :, 3]

    part_pt = jet_pt[:, None] * part_dpt
    part_eta = jet_eta[:, None] + part_deta
    part_phi = jet_phi[:, None] + part_dphi
    part_phi = (part_phi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    part_px = part_pt * np.cos(part_phi)
    part_py = part_pt * np.sin(part_phi)
    part_pz = part_pt * np.sinh(part_eta)
    part_energy = np.sqrt(part_px**2 + part_py**2 + part_pz**2)
    
    return {
        "part_energy": part_energy,
        "part_px": part_px,
        "part_py": part_py,
        "part_pz": part_pz,
        "part_deta": part_deta,
        "part_dphi": part_dphi,
        "part_is_real": part_mask,
        "jet_energy": jet_energy,
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "jet_phi": jet_phi,
        "jet_nparticles": jet_nparticles,
        "jet_label": jet_label,
    }


def process_single_jettype(
    jet_type: str,
    data_dir: Path,
    download_dir: Path,
    splits: tuple[int, int, int] = (70, 15, 15),
    num_particles: int = 150
) -> None:
    """
    Process a single jet type and save it in JetClass format.
    
    Args:
        jet_type: Single jet type ("g", "q", "t", "w", or "z")
        data_dir: Directory to save the processed data
        download_dir: Directory for temporary downloads
        splits: Tuple of (train, val, test) percentages that sum to 100
    """
    
    logging.info(f"Processing jet type: {jet_type}")
    
    # Download and process the dataset to JetClass format
    logging.info(f"Downloading JetNet dataset for {jet_type} to {download_dir}...")
    particle_data, jet_data = get_jetnet_dataset(
        download_dir=download_dir, 
        jet_type=jet_type, 
        num_particles=num_particles
    )
    logging.info(f"Processing {jet_type} dataset to JetClass format...")
    data = get_jetclass_format(particle_data, jet_data)
    
    # Train, validation, and test splits
    split_train, split_val, split_test = tuple(splits)
    split_train = split_train / 100
    split_val = split_val / 100
    split_test = split_test / 100

    njets = len(data["jet_pt"])
    
    shuffle_indices = np.random.permutation(njets)
    
    train_indices = shuffle_indices[: int(split_train * njets)]
    val_indices = shuffle_indices[int(split_train * njets) : int((split_train + split_val) * njets)]
    test_indices = shuffle_indices[int((split_train + split_val) * njets) :]
    
    train_data = {key: value[train_indices] for key, value in data.items()}
    val_data = {key: value[val_indices] for key, value in data.items()}
    test_data = {key: value[test_indices] for key, value in data.items()}
    
    logging.info(f"Dataset split for {jet_type} - train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Save the datasets to HDF5 files with jet type prefix
    for split_label, split_data in zip(
        ["train", "val", "test"], 
        [train_data, val_data, test_data]
    ):
        output_path = data_dir / f"{jet_type}_{split_label}.h5"
        logging.info(f"Saving {jet_type} {split_label} dataset to {output_path}")
        with h5py.File(output_path, "w") as f:
            for key, value in split_data.items():
                f.create_dataset(key, data=value)
    
    logging.info(f"Completed processing jet type: {jet_type}")


def combine_and_shuffle_splits(
    data_dir: Path,
    jet_types: list[str],
) -> None:
    """
    Combine all jet types for each split and shuffle them.
    
    Args:
        data_dir: Directory containing the individual jet type files
        jet_types: List of jet types to combine
    """
    
    logging.info("Combining and shuffling splits across all jet types...")
    
    # Create combined jet type name
    combined_name = "".join(sorted(jet_types))
    
    for split in ["train", "val", "test"]:
        combined_data = {}
        total_samples = 0
        
        # First pass: determine total size and initialize arrays
        for jet_type in jet_types:
            file_path = data_dir / f"{jet_type}_{split}.h5"
            if not file_path.exists():
                logging.warning(f"File {file_path} not found, skipping...")
                continue
                
            with h5py.File(file_path, "r") as f:
                n_samples = f["jet_pt"].shape[0]
                total_samples += n_samples
                
                # Initialize combined arrays on first jet type
                if not combined_data:
                    for key in f.keys():
                        shape = list(f[key].shape)
                        shape[0] = 0  # Will be extended
                        combined_data[key] = []
        
        if total_samples == 0:
            logging.warning(f"No data found for {split} split")
            continue
            
        # Second pass: load and combine data
        logging.info(f"Combining {len(jet_types)} jet types for {split} split ({total_samples} samples total)")
        
        for jet_type in jet_types:
            file_path = data_dir / f"{jet_type}_{split}.h5"
            if not file_path.exists():
                continue
                
            with h5py.File(file_path, "r") as f:
                for key in f.keys():
                    combined_data[key].append(f[key][:])
        
        # Concatenate all data
        for key in combined_data.keys():
            combined_data[key] = np.concatenate(combined_data[key], axis=0)
        
        shuffle_indices = np.random.permutation(total_samples)
        
        for key in combined_data.keys():
            combined_data[key] = combined_data[key][shuffle_indices]
        
        # Save the combined and shuffled data with concatenated jet type names
        output_path = data_dir / f"{combined_name}_{split}.h5"
        logging.info(f"Saving combined {split} dataset to {output_path}")
        with h5py.File(output_path, "w") as f:
            for key, value in combined_data.items():
                f.create_dataset(key, data=value)
    
    logging.info("Finished combining and shuffling splits")


def process_jetnet_dataset(
    data_dir: str | Path,
    splits: tuple[int, int, int] = (70, 15, 15),
    num_particles: int = 150,
    seed: int | None = None,
    cleanup: bool = False,
    jet_types: list[str] = None,
    together: bool = False,
) -> None:
    """
    Process the JetNet dataset and save it in JetClass format.
    
    Args:
        data_dir: Directory to save the processed data
        splits: Tuple of (train, val, test) percentages that sum to 100
        seed: Random seed for reproducibility
        cleanup: If True, remove the download directory before processing
        jet_types: List of jet types to process. Defaults to ["g", "q", "t", "w", "z"]
        together: If True, combine all jet types for each split and shuffle
    """
    
    if jet_types is None:
        jet_types = ["g", "q", "t", "w", "z"]
    
    if seed is not None:
        logging.info(f"Setting random seed to {seed}.")
        np.random.seed(seed)
    
    data_dir = Path(data_dir)
    download_dir = data_dir / "jetnet_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Validate splits
    if len(splits) != 3:
        raise ValueError("Splits must be a tuple of three integers (train, val, test).")
    
    if sum(splits) != 100:
        raise ValueError("Splits must be in percentages that sum to 100.")
    
    logging.info(f"Split into train:validation:test = {splits[0]}:{splits[1]}:{splits[2]}.")
    logging.info(f"Processing jet types: {jet_types}")
    if together:
        logging.info("Will combine and shuffle all jet types for each split")
    
    # Process each jet type separately
    for jet_type in jet_types:
        try:
            process_single_jettype(
                jet_type=jet_type,
                data_dir=data_dir,
                download_dir=download_dir,
                splits=splits,
                num_particles=num_particles,
            )
        except Exception as e:
            logging.error(f"Failed to process jet type {jet_type}: {str(e)}")
            continue
    
    # Combine and shuffle if requested
    if together:
        combine_and_shuffle_splits(
            data_dir=data_dir,
            jet_types=jet_types,
        )
    
    # Cleanup if requested
    if cleanup:
        import shutil
        logging.info(f"Cleaning up download directory: {download_dir}")
        shutil.rmtree(download_dir, ignore_errors=True)
    
    logging.info("Dataset processing completed successfully!")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Process JetNet dataset and convert to JetClass format with separate files per jet type",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--download-dir",
        type=str,
        required=True,
        help="Directory to save the processed dataset"
    )
    
    parser.add_argument(
        "--clean-up",
        action="store_true",
        help="Remove the download directory after processing"
    )

    parser.add_argument(
        "--splits",
        type=int,
        nargs=3,
        default=[70, 15, 15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/validation/test split percentages (must sum to 100)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--jet-types",
        type=str,
        nargs="+",
        default=["g", "q", "t", "w", "z"],
        choices=["g", "q", "t", "w", "z"],
        help="Jet types to process"
    )

    parser.add_argument(
        "-n",
        "--num-particles",
        type=int,
        choices=(30, 150),
        default=150,
        help="Number of particles per jet"
    )

    parser.add_argument(
        "--together",
        action="store_true",
        help="Combine all jet types for each split and shuffle them"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate splits
    if sum(args.splits) != 100:
        parser.error(f"Splits must sum to 100, got {sum(args.splits)}")
    
    if any(split <= 0 for split in args.splits):
        parser.error("All splits must be positive")
    
    # Process the dataset
    process_jetnet_dataset(
        data_dir=args.download_dir,
        splits=tuple(args.splits),
        num_particles=args.num_particles,
        seed=args.seed,
        cleanup=args.clean_up,
        jet_types=args.jet_types,
        together=args.together,
    )


if __name__ == "__main__":
    main()