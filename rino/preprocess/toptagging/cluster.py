#!/usr/bin/env python3
import argparse
from pathlib import Path
import fastjet
import numpy as np
import logging
import traceback
from tqdm import tqdm
import concurrent.futures
import pandas as pd
import awkward as ak
import vector
import torch
import h5py

EPS = np.finfo(float).eps


def load_parquet_data(
    input_path: Path, num_particles: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(input_path)
    num_events = len(df)

    p4 = np.zeros((num_events, num_particles, 4), dtype=np.float64)

    for particle in range(num_particles):
        e_col = f"E_{particle}"
        px_col = f"PX_{particle}"
        py_col = f"PY_{particle}"
        pz_col = f"PZ_{particle}"

        if e_col in df.columns:
            p4[:, particle, 0] = df[e_col].values
            p4[:, particle, 1] = df[px_col].values
            p4[:, particle, 2] = df[py_col].values
            p4[:, particle, 3] = df[pz_col].values

    labels = (
        df["is_signal_new"].values
        if "is_signal_new" in df.columns
        else np.zeros(num_events)
    )

    return p4, labels


def shuffle_data(
    p4: np.ndarray, labels: np.ndarray, seed: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle the data using a random permutation."""
    if seed is not None:
        np.random.seed(seed)
    
    num_events = p4.shape[0]
    indices = np.random.permutation(num_events)
    
    return p4[indices], labels[indices]


def split_array(arr: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    return [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]


def cart_to_cyl(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    eta = np.arcsinh(pz / (pt + EPS))
    return pt, eta, phi


def p4_to_awkward(p4: np.ndarray) -> tuple[ak.Array, ak.Array]:
    energy = p4[:, :, 0]
    px = p4[:, :, 1]
    py = p4[:, :, 2]
    pz = p4[:, :, 3]

    is_real = (energy > 0).astype(np.int32)

    ak_energy = ak.Array(energy)
    ak_px = ak.Array(px)
    ak_py = ak.Array(py)
    ak_pz = ak.Array(pz)
    ak_is_real = ak.Array(is_real)

    components = {
        "energy": ak_energy,
        "px": ak_px,
        "py": ak_py,
        "pz": ak_pz,
        "is_real": ak_is_real,
    }

    particles_vector = vector.zip(components)

    return particles_vector, ak_is_real


def get_clustering(
    particles_vector: vector.backends.awkward.MomentumArray4D,
    dR: float = 0.8,
    algorithm: str = "kt",
) -> fastjet.ClusterSequence:
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
    return fastjet.ClusterSequence(particles_vector, jet_def)


def subjet_logical_or(arr: ak.Array) -> ak.Array:
    return ak.values_astype(ak.any(arr, axis=-1), np.int32)


def process_subjets(
    subjets_consts: ak.Array, jet_vectors: dict[str, np.ndarray]
) -> dict[str, ak.Array]:
    real_mask = subjets_consts.is_real

    real_sum = ak.sum(real_mask, axis=-1)
    has_real = real_sum > 0

    # Calculate basic 4-vector components
    subjet_energy = ak.where(has_real, ak.sum(subjets_consts.energy, axis=-1), 0.0)
    subjet_px = ak.where(has_real, ak.sum(subjets_consts.px, axis=-1), 0.0)
    subjet_py = ak.where(has_real, ak.sum(subjets_consts.py, axis=-1), 0.0)
    subjet_pz = ak.where(has_real, ak.sum(subjets_consts.pz, axis=-1), 0.0)

    pt = ak.where(has_real, np.sqrt(subjet_px**2 + subjet_py**2), 0.0)
    phi = ak.where(has_real, np.arctan2(subjet_py, subjet_px), 0.0)
    eta = ak.where(has_real & (pt > 0), np.arcsinh(subjet_pz / (pt + EPS)), 0.0)

    # Delta quantities
    jet_eta = jet_vectors["eta"]
    jet_phi = jet_vectors["phi"]
    jet_pt = jet_vectors["pt"]
    jet_energy = jet_vectors["energy"]

    deta = ak.where(has_real, eta - jet_eta, 0.0)
    dphi = ak.where(has_real, phi - jet_phi, 0.0)
    dphi = ak.where(has_real, (dphi + np.pi) % (2 * np.pi) - np.pi, 0.0)

    delta_R = ak.where(has_real, np.sqrt(deta**2 + dphi**2), 0.0)

    # Relative quantities
    rel_pt = ak.where(has_real, pt / (jet_pt + EPS), 0.0)
    rel_energy = ak.where(has_real, subjet_energy / (jet_energy + EPS), 0.0)

    is_real = subjet_logical_or(real_mask)

    return {
        "energy": subjet_energy,
        "px": subjet_px,
        "py": subjet_py,
        "pz": subjet_pz,
        "pt": pt,
        "deta": deta,
        "dphi": dphi,
        "delta_R": delta_R,
        "rel_pt": rel_pt,
        "rel_energy": rel_energy,
        "is_real": is_real,
    }


def save_as_tensor(data_dict: dict, output_path: Path):
    output_path = output_path.with_suffix(".pt")
    tensors_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            tensors_dict[key] = torch.tensor(value)
        elif isinstance(value, ak.Array):
            tensors_dict[key] = torch.tensor(ak.to_numpy(value))
    torch.save(tensors_dict, str(output_path))

def save_as_array(data_dict: dict, output_path: Path):
    output_path = output_path.with_suffix(".h5")
    with h5py.File(output_path, "w") as f:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            elif isinstance(value, ak.Array):
                f.create_dataset(key, data=ak.to_numpy(value))
            else:
                logging.warning(f"Unsupported type for key {key}: {type(value)}")

        f.attrs["format_version"] = "1.0"
        f.attrs["description"] = "Processed TopTagging dataset with kt-clustered subjets"


def process_chunk(
    chunk_data: tuple[int, np.ndarray, np.ndarray],
    dR: float,
    algorithm: str,
    num_prongs_list: list[int],
) -> dict:
    chunk_id, p4_chunk, labels_chunk = chunk_data

    # Jet-level calculations
    jet_p4 = np.sum(p4_chunk, axis=1)
    jet_energy = jet_p4[:, 0]
    jet_pt, jet_eta, jet_phi = cart_to_cyl(jet_p4[:, 1], jet_p4[:, 2], jet_p4[:, 3])

    particles_vector, is_real = p4_to_awkward(p4_chunk)
    clustering = get_clustering(particles_vector, dR=dR, algorithm=algorithm)

    # Initialize output dict
    chunk_results = {
        "label": labels_chunk,
        "jet_energy": jet_energy,
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "jet_phi": jet_phi,
    }

    jet_vectors = {
        "energy": jet_energy,
        "px": jet_p4[:, 1],
        "py": jet_p4[:, 2],
        "pz": jet_p4[:, 3],
        "pt": jet_pt,
        "eta": jet_eta,
        "phi": jet_phi,
    }

    # Particle data
    chunk_results.update(
        {
            "part_energy": ak.Array([row[:, 0] for row in p4_chunk]),
            "part_px": ak.Array([row[:, 1] for row in p4_chunk]),
            "part_py": ak.Array([row[:, 2] for row in p4_chunk]),
            "part_pz": ak.Array([row[:, 3] for row in p4_chunk]),
            "part_is_real": is_real,
        }
    )

    part_pt, part_eta, part_phi = cart_to_cyl(
        ak.to_numpy(chunk_results["part_px"]),
        ak.to_numpy(chunk_results["part_py"]),
        ak.to_numpy(chunk_results["part_pz"]),
    )

    # Relative eta and phi
    part_deta = part_eta - jet_eta.reshape(-1, 1)
    part_dphi = part_phi - jet_phi.reshape(-1, 1)
    part_dphi = (part_dphi + np.pi) % (2 * np.pi) - np.pi
    part_delta_R = np.sqrt(part_deta**2 + part_dphi**2)

    # Relative pt and energy
    part_rel_pt = part_pt / (jet_pt.reshape(-1, 1) + EPS)
    part_energy = ak.to_numpy(chunk_results["part_energy"])
    part_rel_energy = part_energy / (jet_energy.reshape(-1, 1) + EPS)

    chunk_results.update(
        {
            "part_deta": ak.Array(part_deta),
            "part_dphi": ak.Array(part_dphi),
            "part_delta_R": ak.Array(part_delta_R),
            "part_rel_pt": ak.Array(part_rel_pt),
            "part_rel_energy": ak.Array(part_rel_energy),
        }
    )

    # Process subjets
    for num_prongs in num_prongs_list:
        if num_prongs > 0:
            subjets_consts = clustering.exclusive_jets_constituents(num_prongs)
            subjet_data = process_subjets(subjets_consts, jet_vectors)

            for k, v in subjet_data.items():
                chunk_results[f"subjet{num_prongs}_{k}"] = v

    # Convert awkward arrays to numpy for merging
    for key in list(chunk_results.keys()):
        if isinstance(chunk_results[key], ak.Array):
            try:
                chunk_results[key] = ak.to_numpy(chunk_results[key])
            except Exception as e:
                logging.warning(
                    f"Error converting {key} to numpy: {str(e)}\n{traceback.format_exc()}"
                )

    return chunk_results


def merge_chunk_results(chunk_results: list[dict]) -> dict:
    merged_results = {}

    common_keys = set(chunk_results[0].keys())
    for chunk in chunk_results[1:]:
        common_keys = common_keys.intersection(set(chunk.keys()))

    # Merge all chunks for common keys
    for key in common_keys:
        try:
            # Make sure all arrays have the same shape in the last dimensions
            shapes = [
                chunk[key].shape[1:] if len(chunk[key].shape) > 1 else ()
                for chunk in chunk_results
            ]
            if all(shape == shapes[0] for shape in shapes):
                merged_results[key] = np.concatenate(
                    [chunk[key] for chunk in chunk_results]
                )
            else:
                logging.warning(
                    f"Skipping key {key} due to inconsistent shapes: {shapes}"
                )
        except Exception as e:
            logging.warning(
                f"Error merging chunks for key {key}: {str(e)}\n{traceback.format_exc()}"
            )

    return merged_results


def process_file(
    args: tuple[Path, Path, float, str, list[int], int, int, int, str, bool, int],
) -> None:
    (
        input_path,
        output_dir,
        dR,
        algorithm,
        num_prongs_list,
        num_particles,
        num_threads,
        chunk_size,
        output_format,
        shuffle,
        seed,
    ) = args

    logging.info(f"Processing {input_path}")

    try:
        # Load data
        p4, labels = load_parquet_data(input_path, num_particles=num_particles)
        
        # Shuffle data if requested
        if shuffle:
            logging.info(f"Shuffling data with seed: {seed}")
            p4, labels = shuffle_data(p4, labels, seed=seed)

        p4_chunks = split_array(p4, chunk_size=chunk_size)
        label_chunks = split_array(labels, chunk_size=chunk_size)

        logging.info(
            f"Processing {len(p4_chunks)} chunks with up to {num_threads} parallel threads"
        )

        chunk_data = [(i, p4_chunks[i], label_chunks[i]) for i in range(len(p4_chunks))]

        chunk_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(process_chunk, chunk, dR, algorithm, num_prongs_list): i
                for i, chunk in enumerate(chunk_data)
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Processing chunks from {input_path.name}",
            ):
                chunk_idx = futures[future]
                try:
                    result = future.result()
                    chunk_results.append(result)
                    logging.debug(
                        f"Completed chunk {chunk_idx} ({len(chunk_results)}/{len(chunk_data)})"
                    )
                except Exception as e:
                    logging.error(
                        f"Error processing chunk {chunk_idx}: {str(e)}\n{traceback.format_exc()}"
                    )

        # Merge results from all chunks
        output_dir.mkdir(parents=True, exist_ok=True)
        pt_path = output_dir / f"{input_path.stem}.pt"
        output_path = output_dir / f"{input_path.stem}.{output_format}"
        try:
            merged_results = merge_chunk_results(chunk_results)
            if "hdf5" in output_format:
                save_as_array(merged_results, output_path)
            elif "pt" in output_format:
                save_as_tensor(merged_results, output_path)
            else:
                logging.error(
                    f"Unsupported output format: {output_format}. Supported formats are 'hdf5' and 'pt'. Using 'hdf5' as default."
                )
                save_as_array(merged_results, output_path)
            logging.info(f"Successfully processed {input_path}")
        except Exception as e:
            # save raw data if merging fails
            logging.error(
                f"Error merging chunk results: {str(e)}\n{traceback.format_exc()}"
            )
            torch.save(chunk_results, pt_path)
            logging.error(f"Saved raw data to {pt_path}")

        return True

    except Exception as e:
        logging.error(
            f"Error processing {input_path}: {str(e)}\n{traceback.format_exc()}"
        )
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process TopTagging dataset with kt subjets"
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        type=str,
        required=True,
        help="Input directory containing parquet files",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory for processed files",
    )
    parser.add_argument("--dr", type=float, default=0.8, help="Jet radius parameter")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="kt",
        choices=["kt", "anti-kt", "antikt", "cambridge"],
        help="Jet clustering algorithm",
    )
    parser.add_argument(
        "--nums-prongs",
        type=int,
        nargs="+",
        default=[],
        help="List of number of prongs",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=200,
        help="Number of particles per jet",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of worker threads for parallel processing",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Fixed number of events per processing chunk",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="hdf5",
        choices=["hdf5", "pt"],
        help="Format for output files (hdf5 or pt, default: hdf5)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the data before processing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")

    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(input_dir.glob("*.parquet"))

    if not parquet_files:
        logging.warning(f"No parquet files found in {input_dir}")
        return

    logging.info(f"Found {len(parquet_files)} parquet files to process")

    process_args = [
        (
            input_path,
            output_dir,
            args.dr,
            args.algorithm,
            args.nums_prongs,
            args.num_particles,
            args.threads,
            args.chunk_size,
            args.output_format,
            args.shuffle,
            args.seed,
        )
        for input_path in parquet_files
    ]

    # Process files sequentially, but with internal parallelism
    for arg in tqdm(process_args, desc="Processing files"):
        try:
            success = process_file(arg)
            if not success:
                logging.warning(f"Failed to process {arg[0]}")
        except Exception as e:
            logging.error(
                f"Unexpected error processing {arg[0]}: {str(e)}\n{traceback.format_exc()}"
            )


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"args: {args}")
    main(args)