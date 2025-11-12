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

def apply_smearing(
    arr: ak.Array, smear_factor: float
) -> ak.Array:
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
        for component in ['energy', 'px', 'py', 'pz']:
            if component in info:
                info[component] = apply_smearing(info[component], smear_factor)
            else:
                logging.error(f"Component '{component}' not found in particle data, skipping smearing")

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


def subjet_logical_or(arr):
    return ak.values_astype(ak.any(arr, axis=-1), np.int32)


def subjet_logical_and(arr):
    return ak.values_astype(ak.all(arr, axis=-1), np.int32)


def subjet_zeros_like(arr):
    return ak.zeros_like(arr, np.int32)


def jetclass_format(kt_subjets_consts, pid_aggregate: str = "or"):
    # Create real particle mask
    real_mask = kt_subjets_consts.is_real

    # Count real particles
    real_sum = ak.sum(real_mask, axis=-1)
    has_real = real_sum > 0

    # 4-momentum (set to 0 if no real particles)
    subjet_px = ak.where(has_real, ak.sum(kt_subjets_consts.px, axis=-1), 0.0)
    subjet_py = ak.where(has_real, ak.sum(kt_subjets_consts.py, axis=-1), 0.0)
    subjet_pz = ak.where(has_real, ak.sum(kt_subjets_consts.pz, axis=-1), 0.0)
    subjet_energy = ak.where(has_real, ak.sum(kt_subjets_consts.energy, axis=-1), 0.0)

    # Jet level quantities (set to 0 if no real particles)
    jet_px = ak.sum(subjet_px, axis=-1)
    jet_py = ak.sum(subjet_py, axis=-1)
    jet_pz = ak.sum(subjet_pz, axis=-1)

    # Calculate phi and eta with protection against empty subjets
    pt = np.sqrt(subjet_px**2 + subjet_py**2)
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)

    jet_phi = ak.where(has_real, np.arctan2(jet_py, jet_px), 0.0)
    jet_eta = ak.where(jet_pt > 0, np.arcsinh(jet_pz / jet_pt), 0.0)

    subjet_phi = ak.where(has_real, np.arctan2(subjet_py, subjet_px), 0.0)
    subjet_eta = ak.where(pt > 0, np.arcsinh(subjet_pz / pt), 0.0)

    # Calculate delta quantities
    subjet_deta = ak.where(has_real, jet_eta - subjet_eta, 0.0)
    subjet_dphi = ak.where(has_real, jet_phi - subjet_phi, 0.0)
    subjet_dphi = ak.where(has_real, (subjet_dphi + np.pi) % (2 * np.pi) - np.pi, 0.0)

    # For subjets with no real particles, set values to 0
    # d0val, dzval: average of real particles
    subjet_d0val = ak.where(
        has_real, ak.sum(kt_subjets_consts.d0val, axis=-1) / real_sum, 0.0
    )
    subjet_dzval = ak.where(
        has_real, ak.sum(kt_subjets_consts.dzval, axis=-1) / real_sum, 0.0
    )

    # part_d0err, part_dzerr: aggregate real particles
    subjet_d0err = ak.where(
        has_real,
        np.sqrt(ak.sum((kt_subjets_consts.d0err) ** 2, axis=-1)) / real_sum,
        0.0,
    )
    subjet_dzerr = ak.where(
        has_real,
        np.sqrt(ak.sum((kt_subjets_consts.dzerr) ** 2, axis=-1)) / real_sum,
        0.0,
    )

    # charge: sum of real particles (already correctly handles padded particles)
    subjet_charge = ak.sum(kt_subjets_consts.charge, axis=-1)

    # is_real: logical OR of constituents (already correctly handles padded particles)
    subjet_is_real = subjet_logical_or(kt_subjets_consts.is_real)

    # PIDs: logical OR of all PIDs (only for real particles)
    pid_aggregate = pid_aggregate.lower()
    if "or" in pid_aggregate:
        aggregate = subjet_logical_or
    elif "and" in pid_aggregate:
        aggregate = subjet_logical_and
    elif "zero" in pid_aggregate:
        aggregate = subjet_zeros_like
    elif "sum" in pid_aggregate:
        aggregate = lambda x: ak.sum(x, axis=-1)
    else:
        raise ValueError(
            f"Invalid PID aggregation method: {pid_aggregate}. Choose 'or', 'and', 'zero', or 'sum'."
        )

    subjet_isChargedHadron = aggregate(kt_subjets_consts.isChargedHadron * real_mask)
    subjet_isNeutralHadron = aggregate(kt_subjets_consts.isNeutralHadron * real_mask)
    subjet_isPhoton = aggregate(kt_subjets_consts.isPhoton * real_mask)
    subjet_isElectron = aggregate(kt_subjets_consts.isElectron * real_mask)
    subjet_isMuon = aggregate(kt_subjets_consts.isMuon * real_mask)

    return {
        "px": subjet_px,
        "py": subjet_py,
        "pz": subjet_pz,
        "energy": subjet_energy,
        "deta": subjet_deta,
        "dphi": subjet_dphi,
        "d0val": subjet_d0val,
        "d0err": subjet_d0err,
        "dzval": subjet_dzval,
        "dzerr": subjet_dzerr,
        "charge": subjet_charge,
        "is_real": subjet_is_real,
        "isChargedHadron": subjet_isChargedHadron,
        "isNeutralHadron": subjet_isNeutralHadron,
        "isPhoton": subjet_isPhoton,
        "isElectron": subjet_isElectron,
        "isMuon": subjet_isMuon,
    }


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


def pad_val(arr: ak.Array, max_length: int, axis: int, clip: bool = True, fill_val: float = 0) -> ak.Array:
    """Pad an array to a fixed length with zeros."""
    return ak.fill_none(ak.pad_none(arr, max_length, axis=axis, clip=clip), fill_val)


def save_as_tensor(events, part_info: dict, subjets: dict[str, ak.Array], output_path: Path, smear_factor: float = -1) -> None:
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
        if (not key.startswith("part_") and 
            not key.startswith("npart_") and 
            key not in tensor_dict):  # Avoid duplicates
            
            if isinstance(original_events[key], ak.Array):
                arr = torch.Tensor(original_events[key].to_numpy())
            else:
                arr = torch.tensor(original_events[key])

            if "is_real" in key:
                arr = arr.to(torch.bool)
            tensor_dict[key] = arr

    # Add subjet arrays
    for name, arr in subjets.items():
        if not name.startswith("nsubjet") and name not in tensor_dict:
            arr = torch.Tensor(arr.to_numpy())
            if "is_real" in name:
                arr = arr.to(torch.bool)
            tensor_dict[name] = arr

    # Change extension to .pt
    output_path = output_path.with_suffix(".pt")

    # Save as PyTorch tensor file
    torch.save(tensor_dict, output_path)
    print(f"Saved tensor data to {output_path}")


def save_as_array(events, part_info: dict, subjets: dict[str, ak.Array], output_path: Path, smear_factor: float = -1) -> None:
    """Save events and subjets data as HDF5 format."""
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
    with h5py.File(output_path, "w") as f:
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
            if (not key.startswith("part_") and not key.startswith("npart_") and key not in written_keys):
                
                if isinstance(original_events[key], ak.Array):
                    arr_numpy = original_events[key].to_numpy()
                else:
                    arr_numpy = np.array(original_events[key])
                f.create_dataset(key, data=arr_numpy, compression="gzip")
                written_keys.add(key)

        # Add subjet arrays
        for name, arr in subjets.items():
            if not name.startswith("nsubjet") and name not in written_keys:
                arr_numpy = arr.to_numpy()
                f.create_dataset(name, data=arr_numpy, compression="gzip")
                written_keys.add(name)

        # Add metadata
        f.attrs["format_version"] = "1.0"
        f.attrs["description"] = "Processed JetClass dataset with kt-clustered subjets"
        if smear_factor > 0:
            f.attrs["smearing_applied"] = True
            f.attrs["smear_factor"] = smear_factor

    print(f"Saved HDF5 data to {output_path}")
def save_as_root(events, part_info: dict, subjets: dict[str, ak.Array], output_path: Path, smear_factor: float = -1) -> None:
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

        # Add subjet data
        for name, arr in subjets.items():
            arrays_to_write[name] = arr

        # Write tree with combined data
        output_file["tree"] = arrays_to_write


def process_file(
    args: Tuple[Path, Path, float, str, List[int], str, int, str, float],
) -> None:
    """Process a single ROOT file and save the results.

    Args:
        args: Tuple containing (input_path, output_path, dR, algorithm, num_prongs_list, pid_aggregate, pad_particles, output_format, smear_factor)
    """
    (
        input_path,
        output_path,
        dR,
        algorithm,
        num_prongs_list,
        pid_aggregate,
        pad_particles,
        output_format,
        smear_factor,
    ) = args

    logging.info(f"Processing {input_path}")

    try:
        with uproot.open(input_path) as file:
            events = file["tree"]
        subjets = {}

        # Get particle vectors and smeared particle info
        pfcands_vector, part_info = get_pfcands_vector(events, pad_particles=pad_particles, smear_factor=smear_factor)
        clustering = get_clustering(pfcands_vector, dR=dR, algorithm=algorithm)

        for num_prongs in num_prongs_list:
            subjets_consts = clustering.exclusive_jets_constituents(num_prongs)
            subjets_jetclass = jetclass_format(
                subjets_consts, pid_aggregate=pid_aggregate
            )
            for k, v in subjets_jetclass.items():
                subjets[f"subjet{num_prongs}_{k}"] = v

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if "hdf5" in output_format:
            save_as_array(events, part_info, subjets, output_path, smear_factor)
        elif "pt" in output_format:
            save_as_tensor(events, part_info, subjets, output_path, smear_factor)
        else:
            save_as_root(events, part_info, subjets, output_path, smear_factor)

        logging.info(f"Successfully processed {input_path}")
        return True

    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")
        logging.debug(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Process ROOT files with kt subjets")
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
        nargs="+",  # This allows multiple integer arguments
        default=[], 
        help="List of number of prongs (default: empty)",
    )
    parser.add_argument(
        "--pid-aggregate",
        type=str,
        default="or",
        choices=["or", "and", "zero", "sum"],
        help="Method to aggregate PID information (default: or)",
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
        f"pid_aggregate={args.pid_aggregate}, threads={args.threads}, "
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
            args.pid_aggregate,
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