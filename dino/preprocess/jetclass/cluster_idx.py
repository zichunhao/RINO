#!/usr/bin/env python3
import argparse
from pathlib import Path
import uproot
import fastjet
import awkward as ak
import numpy as np
import vector
import logging
from tqdm import tqdm
import concurrent.futures
from typing import List, Tuple
import torch
import traceback
import h5py
import random

EPS = np.finfo(float).eps


def apply_smearing(arr: ak.Array, smear_factor: float) -> ak.Array:
    if smear_factor <= 0:
        return arr

    # Generate epsilon ~ N(0, smear_factor)
    epsilon = np.random.normal(0, smear_factor, size=ak.to_numpy(ak.flatten(arr)).shape)
    epsilon = ak.unflatten(epsilon, ak.num(arr))
    return arr * (1 + epsilon)


def get_pfcands_vector(
    events: uproot.TTree,
    pad_particles: int | None = None,
    smear_factor: float = -1,
) -> tuple[vector.backends.awkward.MomentumArray4D, dict]:
    """Get particle four-vectors from events, with optional padding and smearing.

    Args:
        events: Input events from ROOT file
        pad_particles: Number of particles to pad each event to
        smear_factor: If >0, apply Gaussian smearing to energy and momentum components

    Returns:
        Tuple of (momentum four-vectors, particle info dictionary with 'part_' prefix)
    """
    part_keys = [k for k in events.keys() if k.startswith("part_")]
    if not part_keys:
        raise ValueError("No particle keys found in events")

    # Read all arrays at once
    arrays = events.arrays(part_keys)
    info = {key.replace("part_", ""): arrays[key] for key in part_keys}

    if smear_factor > 0:
        # Apply smearing to energy and momentum components
        for component in ["energy", "px", "py", "pz"]:
            if component in info:
                info[component] = apply_smearing(info[component], smear_factor)
            else:
                logging.error(
                    f"Component '{component}' not found in particle data, skipping smearing"
                )

    # 1 if particle is real, 0 if it is a padding particle
    first_array = arrays[part_keys[0]]
    info["is_real"] = ak.ones_like(first_array, dtype=np.int32)

    # Pad particles to the same length
    if pad_particles is not None and pad_particles > 0:
        current_length = ak.num(first_array)
        pad_length = pad_particles - current_length
        padding = ak.Array([ak.Array([0] * n) for n in pad_length])
        info = {
            key: ak.concatenate([val, padding], axis=-1) for key, val in info.items()
        }

    info["idx"] = ak.local_index(info["energy"])

    # Create vector and convert to desired format
    vector_data = vector.zip(info).to_ptphietamass()

    # Return info with 'part_' prefix for saving
    part_info = {f"part_{key}": val for key, val in info.items()}

    return vector_data, part_info


def get_clustering(
    pfcands_vector: vector.backends.awkward.MomentumArray4D,
    dR: float = 0.8,
    algorithm: str = "kt",
) -> fastjet.ClusterSequence:
    """Creates and returns a FastJet ClusterSequence object.

    Args:
        pfcands_vector: Input particle four-vectors
        dR: Jet radius parameter
        algorithm: Jet clustering algorithm ('kt', 'anti-kt', or 'cambridge')

    Returns:
        fastjet.ClusterSequence: The clustering sequence object
    """
    if algorithm.lower() == "kt":
        algorithm = fastjet.kt_algorithm
    elif algorithm.lower() == "cambridge":
        algorithm = fastjet.cambridge_algorithm
    elif algorithm.lower() in ("anti-kt", "antikt"):
        algorithm = fastjet.antikt_algorithm
    else:
        raise ValueError(
            f"Invalid algorithm: {algorithm}. Choose 'kt', 'anti-kt', or 'cambridge'."
        )

    jet_def = fastjet.JetDefinition(algorithm, dR)
    return fastjet.ClusterSequence(pfcands_vector, jet_def)


def create_subjet_masks(
    pfcands_vector: vector.backends.awkward.MomentumArray4D,
    clustering: fastjet.ClusterSequence,
    num_prongs: int,
) -> dict[str, ak.Array]:
    """Create membership masks for each subjet.

    Args:
        pfcands_vector: Input particle four-vectors
        clustering: FastJet clustering sequence
        num_prongs: Number of subjets to extract

    Returns:
        Dictionary with mask arrays for each subjet
    """
    subjet_mask = {}
    kt_subjets = clustering.exclusive_jets_constituents(num_prongs)

    # for i in range(num_prongs):
    #     mask = ak.zeros_like(pfcands_vector.idx, dtype=bool)
    #     subjet_indices = kt_subjets.idx[:, i]
    #     mask = ak.any(pfcands_vector.idx == subjet_indices[:, None], axis=-1)
    #     mask = ak.values_astype(mask, np.int32)  # Convert bool to int32
    #     subjet_mask[f"subjet_{num_prongs}prong_idx{i}_mask"] = mask

    # return subjet_mask
    subjet_assignment = ak.full_like(pfcands_vector.idx, -1, dtype=np.int32)

    for i in range(num_prongs):
        subjet_indices = kt_subjets.idx[:, i]
        mask = ak.any(pfcands_vector.idx == subjet_indices[:, None], axis=-1)
        subjet_assignment = ak.where(mask, i, subjet_assignment)

    subjet_mask[f"subjet_{num_prongs}prong_idx"] = subjet_assignment
    return subjet_mask


def recalculate_jet_quantities(part_info: dict, smear_factor: float = -1):
    """Recalculate jet-level quantities from smeared particle data.

    Args:
        part_info: Dictionary with particle data (with 'part_' prefix)
        smear_factor: If >0, recalculate jet quantities from smeared particles

    Returns:
        Dictionary with recalculated jet quantities
    """
    if smear_factor <= 0:
        return {}

    # Get particle arrays
    part_px = part_info["part_px"]
    part_py = part_info["part_py"]
    part_pz = part_info["part_pz"]
    part_energy = part_info["part_energy"]

    # Calculate jet-level quantities by summing over particles
    jet_px = ak.sum(part_px, axis=-1)
    jet_py = ak.sum(part_py, axis=-1)
    jet_pz = ak.sum(part_pz, axis=-1)
    jet_energy = ak.sum(part_energy, axis=-1)

    # Calculate derived quantities
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)
    jet_phi = np.arctan2(jet_py, jet_px)
    jet_eta = np.arcsinh(jet_pz / (jet_pt + EPS))

    return {
        "jet_px": jet_px,
        "jet_py": jet_py,
        "jet_pz": jet_pz,
        "jet_energy": jet_energy,
        "jet_pt": jet_pt,
        "jet_phi": jet_phi,
        "jet_eta": jet_eta,
    }


def pad_val(
    arr: ak.Array, max_length: int, axis: int, clip: bool = True, fill_val: float = 0
) -> ak.Array:
    """Pad an array to a fixed length with zeros."""
    return ak.fill_none(ak.pad_none(arr, max_length, axis=axis, clip=clip), fill_val)


def save_as_tensor(
    events,
    part_info: dict,
    subjet_masks: dict[str, ak.Array],
    output_path: Path,
    smear_factor: float = -1,
) -> None:
    tensor_dict = {}
    original_events = events.arrays()

    # Recalculate jet quantities if smearing was applied
    recalc_jet_vars = recalculate_jet_quantities(part_info, smear_factor)

    # get max particles from smeared data
    max_particles = int(ak.max(ak.count(part_info["part_energy"], axis=-1)))
    part_energy = part_info["part_energy"]
    part_energy = pad_val(part_energy, max_particles, axis=-1, clip=True, fill_val=0.0)
    part_energy = part_energy.to_numpy()
    part_is_real = torch.Tensor((part_energy != 0).astype(int))
    tensor_dict["part_is_real"] = part_is_real

    # Use smeared particle arrays (highest priority)
    for key, arr in part_info.items():
        if key.startswith("part_"):
            padded_arr = pad_val(arr, max_particles, axis=-1, clip=True, fill_val=0.0)
            tensor_arr = torch.Tensor(padded_arr.to_numpy())
        else:
            continue

        if "is_real" in key:
            tensor_arr = tensor_arr.to(torch.bool)
        tensor_dict[key] = tensor_arr

    # Add recalculated jet quantities (second priority)
    for key, arr in recalc_jet_vars.items():
        if key not in tensor_dict:  # Avoid duplicates
            tensor_arr = torch.Tensor(arr.to_numpy())
            if "is_real" in key:
                tensor_arr = tensor_arr.to(torch.bool)
            tensor_dict[key] = tensor_arr

    # Add non-particle arrays from original events (lowest priority)
    for key in original_events.fields:
        if (
            not key.startswith("part_")
            and not key.startswith("npart_")
            and key not in tensor_dict
        ):  # Avoid duplicates

            if isinstance(original_events[key], ak.Array):
                arr = torch.Tensor(original_events[key].to_numpy())
            else:
                arr = torch.tensor(original_events[key])

            if "is_real" in key:
                arr = arr.to(torch.bool)
            tensor_dict[key] = arr

    # Add subjet masks
    for name, arr in subjet_masks.items():
        if name not in tensor_dict:
            # Pad masks to max_particles
            padded_arr = pad_val(arr, max_particles, axis=-1, clip=True, fill_val=False)
            tensor_arr = torch.Tensor(padded_arr.to_numpy()).to(torch.bool)
            tensor_dict[name] = tensor_arr

    # Change extension to .pt
    output_path = output_path.with_suffix(".pt")

    # Save as PyTorch tensor file
    torch.save(tensor_dict, output_path)
    print(f"Saved tensor data to {output_path}")


def save_as_array(
    events,
    part_info: dict,
    subjet_masks: dict[str, ak.Array],
    output_path: Path,
    smear_factor: float = -1,
) -> None:
    """Save events and subjet masks data as HDF5 format."""
    original_events = events.arrays()

    # Recalculate jet quantities if smearing was applied
    recalc_jet_vars = recalculate_jet_quantities(part_info, smear_factor)

    # get max particles from smeared data
    max_particles = int(ak.max(ak.count(part_info["part_energy"], axis=-1)))
    part_energy = part_info["part_energy"]
    part_energy = pad_val(part_energy, max_particles, axis=-1, clip=True, fill_val=0.0)
    part_energy = part_energy.to_numpy()
    part_is_real = (part_energy != 0).astype(int)

    # Change extension to .h5
    output_path = output_path.with_suffix(".h5")

    # Save as HDF5 file
    with h5py.File(output_path, "w", locking=False) as f:
        # Keep track of written keys to avoid duplicates
        written_keys = set()

        # Save particle is_real array
        f.create_dataset("part_is_real", data=part_is_real, compression="gzip")
        written_keys.add("part_is_real")

        # Save smeared particle arrays (highest priority)
        for key, arr in part_info.items():
            if key.startswith("part_") and key not in written_keys:
                padded_arr = pad_val(
                    arr, max_particles, axis=-1, clip=True, fill_val=0.0
                )
                arr_numpy = padded_arr.to_numpy()
                f.create_dataset(key, data=arr_numpy, compression="gzip")
                written_keys.add(key)

        # Save recalculated jet quantities (second priority)
        for key, arr in recalc_jet_vars.items():
            if key not in written_keys:
                arr_numpy = arr.to_numpy()
                f.create_dataset(key, data=arr_numpy, compression="gzip")
                written_keys.add(key)

        # Save remaining non-particle arrays from original events (lowest priority)
        for key in original_events.fields:
            if (
                not key.startswith("part_")
                and not key.startswith("npart_")
                and key not in written_keys
            ):

                if isinstance(original_events[key], ak.Array):
                    arr_numpy = original_events[key].to_numpy()
                else:
                    arr_numpy = np.array(original_events[key])
                f.create_dataset(key, data=arr_numpy, compression="gzip")
                written_keys.add(key)

        # Add subjet masks
        for name, arr in subjet_masks.items():
            if name not in written_keys:
                # Pad masks to max_particles
                padded_arr = pad_val(
                    arr, max_particles, axis=-1, clip=True, fill_val=False
                )
                arr_numpy = padded_arr.to_numpy().astype(bool)
                f.create_dataset(name, data=arr_numpy, compression="gzip")
                written_keys.add(name)

        # Add metadata
        f.attrs["format_version"] = "1.0"
        f.attrs["description"] = (
            "Processed JetClass dataset with kt-clustered subjet masks"
        )
        if smear_factor > 0:
            f.attrs["smearing_applied"] = True
            f.attrs["smear_factor"] = smear_factor

    print(f"Saved HDF5 data to {output_path}")


def save_as_root(
    events,
    part_info: dict,
    subjet_masks: dict[str, ak.Array],
    output_path: Path,
    smear_factor: float = -1,
) -> None:
    # Save as ROOT file
    output_path = output_path.with_suffix(".root")
    with uproot.recreate(str(output_path)) as output_file:
        # Start with original data
        original_arrays = events.arrays()

        # Recalculate jet quantities if smearing was applied
        recalc_jet_vars = recalculate_jet_quantities(part_info, smear_factor)

        # Create dictionary for writing
        arrays_to_write = {}

        # Add smeared particle data
        for key, arr in part_info.items():
            arrays_to_write[key] = arr

        # Add original non-particle data (with potential jet quantity replacements)
        for field in original_arrays.fields:
            if field.startswith("part_") or field.startswith("npart_"):
                continue  # Skip, already handled above
            elif field in recalc_jet_vars:
                arrays_to_write[field] = recalc_jet_vars[field]
            else:
                arrays_to_write[field] = original_arrays[field]

        # Add subjet masks
        for name, arr in subjet_masks.items():
            arrays_to_write[name] = arr

        # Write tree with combined data
        output_file["tree"] = arrays_to_write


def process_file(
    args: Tuple[Path, Path, float, str, List[int], int, str, float],
) -> None:
    """Process a single ROOT file and save the results.

    Args:
        args: Tuple containing (input_path, output_path, dR, algorithm, num_prongs_list, pad_particles, output_format, smear_factor)
    """
    (
        input_path,
        output_path,
        dR,
        algorithm,
        num_prongs_list,
        pad_particles,
        output_format,
        smear_factor,
    ) = args

    logging.info(f"Processing {input_path}")

    try:
        with uproot.open(input_path) as file:
            events = file["tree"]

        # Get particle vectors and smeared particle info
        pfcands_vector, part_info = get_pfcands_vector(
            events, pad_particles=pad_particles, smear_factor=smear_factor
        )
        clustering = get_clustering(pfcands_vector, dR=dR, algorithm=algorithm)

        # Create subjet masks
        subjet_masks = {}
        for num_prongs in num_prongs_list:
            masks = create_subjet_masks(pfcands_vector, clustering, num_prongs)
            subjet_masks.update(masks)

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if "hdf5" in output_format:
            save_as_array(events, part_info, subjet_masks, output_path, smear_factor)
        elif "pt" in output_format:
            save_as_tensor(events, part_info, subjet_masks, output_path, smear_factor)
        else:
            save_as_root(events, part_info, subjet_masks, output_path, smear_factor)

        logging.info(f"Successfully processed {input_path}")
        return True

    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")
        logging.debug(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process ROOT files with kt subjet masks"
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        type=str,
        required=True,
        help="Input directory containing ROOT files",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--dr", type=float, default=0.8, help="Jet radius parameter (default: 0.8)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="kt",
        choices=["kt", "anti-kt", "antikt", "cambridge"],
        help="Jet clustering algorithm (default: kt)",
    )
    parser.add_argument(
        "--nums-prongs",
        type=int,
        nargs="+",
        default=[],
        help="List of number of prongs (default: empty)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads to use (default: number of CPU cores)",
    )
    parser.add_argument(
        "--pad-particles",
        type=int,
        default=-1,
        help="Number of particles to pad each jet to. If -1, uses max(prong_list). (default: -1)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="hdf5",
        choices=["hdf5", "pt", "root"],
        help="Format to save the output files (default: hdf5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--smear-factor",
        type=float,
        default=-1,
        help="Apply Gaussian smearing to energy/momentum if >0 (default: -1, no smearing)",
    )
    parser.add_argument(
        "--log",
        dest="log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all ROOT files in input directory
    root_files = list(input_dir.glob("*.root"))

    if not root_files:
        logging.warning(f"No ROOT files found in {input_dir}")
        return

    # Prepare arguments for process_file
    if args.pad_particles == -1:
        if len(args.nums_prongs) == 0:
            pad_particles = -1
        else:
            pad_particles = max(args.nums_prongs)
    else:
        pad_particles = args.pad_particles

    logging.info(f"Found {len(root_files)} ROOT files to process")
    logging.info(
        f"Using parameters: dR={args.dr}, algorithm={args.algorithm}, "
        f"nums_prongs={args.nums_prongs}, pad_particles={pad_particles}, "
        f"threads={args.threads}, "
        f"output_format={args.output_format}, smear_factor={args.smear_factor}"
    )

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        logging.info(f"Using random seed: {args.seed}")

    process_args = [
        (
            input_path,
            output_dir / input_path.name,
            args.dr,
            args.algorithm,
            args.nums_prongs,
            pad_particles,
            args.output_format,
            args.smear_factor,
        )
        for input_path in root_files
    ]

    # Process files using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_file, arg) for arg in process_args]

        # Show progress bar
        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                try:
                    success = future.result()
                    if not success:
                        logging.warning("A file failed to process")
                except Exception as e:
                    logging.error(f"Unexpected error during processing: {str(e)}")


if __name__ == "__main__":
    main()
