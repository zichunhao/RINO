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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_events_root(path: Path, tree_name: str = "tree"):
    """Load events from a ROOT file.

    Returns:
        part_info: dict of jagged awkward arrays, keys with 'part_' prefix
        original_arrays: dict of flat (per-jet) arrays for all non-particle branches
    """
    with uproot.open(path) as f:
        tree = f[tree_name]
        arrays = tree.arrays(library="ak")

    part_keys = [k for k in arrays.fields if k.startswith("part_")]
    other_keys = [
        k
        for k in arrays.fields
        if not k.startswith("part_") and not k.startswith("npart_")
    ]
    part_info = {k: arrays[k] for k in part_keys}
    original_arrays = {k: arrays[k] for k in other_keys}
    return part_info, original_arrays


def load_events_hdf5(path: Path):
    """Load events from an HDF5 file.

    Particle arrays are stored padded (shape N×max_p). Real particles are
    identified by part_energy != 0 and stripped back to jagged arrays so the
    rest of the pipeline (fastjet clustering) works identically to ROOT input.

    Returns:
        part_info: dict of jagged awkward arrays, keys with 'part_' prefix
        original_arrays: dict of flat (per-jet) arrays for all non-particle keys
    """
    with h5py.File(path, "r") as f:
        data = {k: f[k][:] for k in f.keys()}

    part_keys = [k for k in data if k.startswith("part_")]
    other_keys = [
        k for k in data if not k.startswith("part_") and not k.startswith("npart_")
    ]

    # Build real-particle mask from part_energy != 0
    real_mask = ak.Array(data["part_energy"] != 0)  # (N, max_p) bool
    part_info = {k: ak.Array(data[k])[real_mask] for k in part_keys}
    original_arrays = {k: ak.Array(data[k]) for k in other_keys}
    return part_info, original_arrays


def get_pfcands_vector(
    part_info: dict,
    pad_particles: int | None = None,
) -> vector.backends.awkward.MomentumArray4D:
    """Build a 4-vector array from a part_info dict.

    Args:
        part_info: dict of jagged awkward arrays with 'part_' prefix stripped
        pad_particles: if > 0, pad each event to this many particles

    Returns:
        Momentum four-vectors (ptphietamass)
    """
    info = dict(part_info)  # shallow copy; keys already have 'part_' stripped

    # Mark real particles before any padding
    first_key = next(iter(info))
    info["is_real"] = ak.ones_like(info[first_key], dtype=np.int32)

    if pad_particles is not None and pad_particles > 0:
        info = {
            key: ak.fill_none(ak.pad_none(val, pad_particles, axis=1, clip=False), 0)
            for key, val in info.items()
        }

    return vector.zip(info).to_ptphietamass(), info


def get_clustering(
    pfcands_vector: vector.backends.awkward.MomentumArray4D,
    dR: float = 0.8,
    algorithm: str = "kt",
) -> fastjet.ClusterSequence:
    if algorithm.lower() == "kt":
        algorithm = fastjet.kt_algorithm
    elif algorithm.lower() in ("cambridge", "ca"):
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
    real_mask = kt_subjets_consts.is_real
    real_sum = ak.sum(real_mask, axis=-1)
    has_real = real_sum > 0

    subjet_px = ak.where(has_real, ak.sum(kt_subjets_consts.px, axis=-1), 0.0)
    subjet_py = ak.where(has_real, ak.sum(kt_subjets_consts.py, axis=-1), 0.0)
    subjet_pz = ak.where(has_real, ak.sum(kt_subjets_consts.pz, axis=-1), 0.0)
    subjet_energy = ak.where(has_real, ak.sum(kt_subjets_consts.energy, axis=-1), 0.0)

    jet_px = ak.sum(subjet_px, axis=-1)
    jet_py = ak.sum(subjet_py, axis=-1)
    jet_pz = ak.sum(subjet_pz, axis=-1)

    pt = np.sqrt(subjet_px**2 + subjet_py**2)
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)

    jet_phi = ak.where(has_real, np.arctan2(jet_py, jet_px), 0.0)
    jet_eta = ak.where(jet_pt > 0, np.arcsinh(jet_pz / jet_pt), 0.0)

    subjet_phi = ak.where(has_real, np.arctan2(subjet_py, subjet_px), 0.0)
    subjet_eta = ak.where(pt > 0, np.arcsinh(subjet_pz / pt), 0.0)

    subjet_deta = ak.where(has_real, subjet_eta - jet_eta, 0.0)
    subjet_dphi = ak.where(has_real, subjet_phi - jet_phi, 0.0)
    subjet_dphi = ak.where(has_real, (subjet_dphi + np.pi) % (2 * np.pi) - np.pi, 0.0)

    subjet_d0val = ak.where(
        has_real, ak.sum(kt_subjets_consts.d0val, axis=-1) / real_sum, 0.0
    )
    subjet_dzval = ak.where(
        has_real, ak.sum(kt_subjets_consts.dzval, axis=-1) / real_sum, 0.0
    )
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

    subjet_charge = ak.sum(kt_subjets_consts.charge, axis=-1)
    subjet_is_real = subjet_logical_or(kt_subjets_consts.is_real)

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


def pad_val(
    arr: ak.Array, max_length: int, axis: int, clip: bool = True, fill_val: float = 0
) -> ak.Array:
    return ak.fill_none(ak.pad_none(arr, max_length, axis=axis, clip=clip), fill_val)


def sort_by_pt(
    part_info: dict, subjets: dict, num_prongs_list: list
) -> tuple[dict, dict]:
    """Sort particles and subjets within each event by pT descending.

    Padded particles (px = py = 0, hence pT = 0) fall naturally to the end.

    Args:
        part_info: dict of awkward arrays keyed as 'part_*'.
        subjets: dict of awkward arrays keyed as 'subjet{N}_*'.
        num_prongs_list: list of prong counts present in subjets.

    Returns:
        (sorted_part_info, sorted_subjets)
    """
    pt = np.sqrt(part_info["part_px"] ** 2 + part_info["part_py"] ** 2)
    part_sort_idx = ak.argsort(pt, axis=1, ascending=False)
    sorted_parts = {k: v[part_sort_idx] for k, v in part_info.items()}

    sorted_subjets = {}
    for num_prongs in num_prongs_list:
        prefix = f"subjet{num_prongs}_"
        pt_sj = np.sqrt(subjets[f"{prefix}px"] ** 2 + subjets[f"{prefix}py"] ** 2)
        sj_sort_idx = ak.argsort(pt_sj, axis=1, ascending=False)
        for k, v in subjets.items():
            if k.startswith(prefix):
                sorted_subjets[k] = v[sj_sort_idx]
    for k, v in subjets.items():
        if k not in sorted_subjets:
            sorted_subjets[k] = v

    return sorted_parts, sorted_subjets


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------

KINEMATICS_KEYS = {"px", "py", "pz", "energy", "deta", "dphi", "is_real"}


def _resolve_max_particles(part_info: dict, max_particles: int) -> int:
    if max_particles > 1:
        return max_particles
    return int(ak.max(ak.count(part_info["part_energy"], axis=-1)))


def save_as_tensor(
    part_info: dict,
    original_arrays: dict,
    subjets: dict[str, ak.Array],
    output_path: Path,
    max_particles: int = -1,
) -> None:
    tensor_dict = {}
    max_p = _resolve_max_particles(part_info, max_particles)

    part_energy = part_info["part_energy"]
    part_energy = pad_val(part_energy, max_p, axis=-1, clip=True, fill_val=0.0)
    part_energy = part_energy.to_numpy()
    part_is_real = torch.Tensor((part_energy != 0).astype(int))
    tensor_dict["part_is_real"] = part_is_real

    for key, arr in part_info.items():
        if key.startswith("part_"):
            padded_arr = pad_val(arr, max_p, axis=-1, clip=True, fill_val=0.0)
            tensor_arr = torch.Tensor(padded_arr.to_numpy())
            if "is_real" in key:
                tensor_arr = tensor_arr.to(torch.bool)
            tensor_dict[key] = tensor_arr

    for key, arr in original_arrays.items():
        if key not in tensor_dict:
            if isinstance(arr, ak.Array):
                tensor_arr = torch.Tensor(arr.to_numpy())
            else:
                tensor_arr = torch.tensor(arr)
            if "is_real" in key:
                tensor_arr = tensor_arr.to(torch.bool)
            tensor_dict[key] = tensor_arr

    for name, arr in subjets.items():
        if not name.startswith("nsubjet") and name not in tensor_dict:
            arr = torch.Tensor(arr.to_numpy())
            if "is_real" in name:
                arr = arr.to(torch.bool)
            tensor_dict[name] = arr

    output_path = output_path.with_suffix(".pt")
    torch.save(tensor_dict, output_path)
    print(f"Saved tensor data to {output_path}")


def save_as_array(
    part_info: dict,
    original_arrays: dict,
    subjets: dict[str, ak.Array],
    output_path: Path,
    max_particles: int = -1,
) -> None:
    max_p = _resolve_max_particles(part_info, max_particles)
    output_path = output_path.with_suffix(".h5")

    with h5py.File(output_path, "w") as f:
        written_keys = set()

        part_energy = part_info["part_energy"]
        part_energy = pad_val(part_energy, max_p, axis=-1, clip=True, fill_val=0.0)
        part_is_real = (part_energy.to_numpy() != 0).astype(int)
        f.create_dataset("part_is_real", data=part_is_real, compression="gzip")
        written_keys.add("part_is_real")

        for key, arr in part_info.items():
            if key.startswith("part_") and key not in written_keys:
                padded_arr = pad_val(arr, max_p, axis=-1, clip=True, fill_val=0.0)
                f.create_dataset(key, data=padded_arr.to_numpy(), compression="gzip")
                written_keys.add(key)

        for key, arr in original_arrays.items():
            if key not in written_keys:
                if isinstance(arr, ak.Array):
                    arr_numpy = arr.to_numpy()
                else:
                    arr_numpy = np.array(arr)
                f.create_dataset(key, data=arr_numpy, compression="gzip")
                written_keys.add(key)

        for name, arr in subjets.items():
            if not name.startswith("nsubjet") and name not in written_keys:
                f.create_dataset(name, data=arr.to_numpy(), compression="gzip")
                written_keys.add(name)

        f.attrs["format_version"] = "1.0"
        f.attrs["description"] = "Processed JetClass dataset with kt-clustered subjets"

    print(f"Saved HDF5 data to {output_path}")


def save_as_root(
    part_info: dict,
    original_arrays: dict,
    subjets: dict[str, ak.Array],
    output_path: Path,
) -> None:
    output_path = output_path.with_suffix(".root")
    arrays_to_write = {}
    arrays_to_write.update(part_info)
    arrays_to_write.update(original_arrays)
    arrays_to_write.update(subjets)

    with uproot.recreate(str(output_path)) as output_file:
        output_file["tree"] = arrays_to_write


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def process_file(
    args: Tuple,
) -> None:
    (
        input_path,
        output_path,
        dR,
        algorithm,
        num_prongs_list,
        pid_aggregate,
        pad_particles,
        output_format,
        kinematics_only,
        max_particles,
        sort_by_pt_flag,
    ) = args

    logging.info(f"Processing {input_path}")

    try:
        # Load events (ROOT or HDF5)
        suffix = Path(input_path).suffix.lower()
        if suffix in (".h5", ".hdf5"):
            raw_part_info, original_arrays = load_events_hdf5(input_path)
        else:
            raw_part_info, original_arrays = load_events_root(input_path)

        # Strip 'part_' prefix for vector construction
        stripped = {k.replace("part_", ""): v for k, v in raw_part_info.items()}

        pfcands_vector, padded_stripped = get_pfcands_vector(stripped, pad_particles)
        # Re-attach 'part_' prefix
        part_info = {f"part_{k}": v for k, v in padded_stripped.items()}

        clustering = get_clustering(pfcands_vector, dR=dR, algorithm=algorithm)

        subjets = {}
        for num_prongs in num_prongs_list:
            subjets_consts = clustering.exclusive_jets_constituents(num_prongs)
            subjets_jetclass = jetclass_format(
                subjets_consts, pid_aggregate=pid_aggregate
            )
            for k, v in subjets_jetclass.items():
                subjets[f"subjet{num_prongs}_{k}"] = v

        if kinematics_only:
            subjets = {
                k: v
                for k, v in subjets.items()
                if any(k.endswith(f"_{kk}") for kk in KINEMATICS_KEYS)
            }

        if sort_by_pt_flag:
            part_info, subjets = sort_by_pt(part_info, subjets, num_prongs_list)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if "hdf5" in output_format:
            save_as_array(
                part_info, original_arrays, subjets, output_path, max_particles
            )
        elif "pt" in output_format:
            save_as_tensor(
                part_info, original_arrays, subjets, output_path, max_particles
            )
        else:
            save_as_root(part_info, original_arrays, subjets, output_path)

        logging.info(f"Successfully processed {input_path}")
        return True

    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")
        logging.debug(traceback.format_exc())
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Process ROOT/HDF5 files with kt subjets"
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        type=str,
        required=True,
        help="Input directory containing ROOT or HDF5 files",
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
        choices=["kt", "anti-kt", "antikt", "cambridge", "ca"],
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
        "--pid-aggregate",
        type=str,
        default="sum",
        choices=["or", "and", "zero", "sum"],
        help="Method to aggregate PID information (default: sum)",
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
        help="Number of particles to pad each jet to before clustering. "
        "If -1, uses max(prong_list). (default: -1)",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=-1,
        help="Maximum number of particles per event in the output. "
        "If > 1, use this value; otherwise compute from data. (default: -1)",
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
        "--subjet-kinematics-only",
        action="store_true",
        dest="kinematics_only",
        help="If set, only (px, py, pz, energy, deta, dphi, is_real) are saved "
        "per subjet; all particle fields are always kept",
    )
    parser.add_argument(
        "--sort-by-pt",
        action="store_true",
        dest="sort_by_pt",
        help="If set, sort particles and subjets within each event by pT descending "
        "before saving. Padded particles (pT=0) fall naturally to the end.",
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

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Accept both ROOT and HDF5 input files
    input_files = (
        list(input_dir.glob("*.root"))
        + list(input_dir.glob("*.h5"))
        + list(input_dir.glob("*.hdf5"))
    )

    if not input_files:
        logging.warning(f"No ROOT or HDF5 files found in {input_dir}")
        return

    if args.pad_particles == -1:
        pad_particles = max(args.nums_prongs) if args.nums_prongs else -1
    else:
        pad_particles = args.pad_particles

    logging.info(f"Found {len(input_files)} input files to process")
    logging.info(
        f"Using parameters: dR={args.dr}, algorithm={args.algorithm}, "
        f"nums_prongs={args.nums_prongs}, pad_particles={pad_particles}, "
        f"max_particles={args.max_particles}, pid_aggregate={args.pid_aggregate}, "
        f"threads={args.threads}, output_format={args.output_format}, "
        f"kinematics_only={args.kinematics_only}, sort_by_pt={args.sort_by_pt}"
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
            args.kinematics_only,
            args.max_particles,
            args.sort_by_pt,
        )
        for input_path in input_files
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_file, arg) for arg in process_args]

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
