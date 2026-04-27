import argparse

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from src.setup.root_utils import (
    common_particle_class,
    lifetime_signing,
    read_jetclass_file,
)

_EPS = 1e-8


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert JetClass files to a more usable format"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/srv/beegfs/scratch/groups/rodem/anomalous_jets/data/JetClass/Pythia/",
        help="The path to the JetClass files",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        default="/srv/fast/share/rodem/JetClassH5/",
        help="The path to save the converted files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.root",
        help="Glob pattern for selecting ROOT files within each split "
        "subfolder (e.g. 'ZJetsToNuNu_*.root' for QCD only).",
    )
    return parser.parse_args()


def main() -> None:
    """Convert the JetClass root files to a more usable HDF format.

    The output features are the following:
    Independant continuous (7 dimensional vector)
    - 0: pt
    - 1: deta
    - 2: dphi
    - 3: d0val
    - 4: d0err
    - 5: dzval
    - 6: dzerr
    Independant categorical (single int representing following classes)
    - 0: isPhoton
    - 1: isHadron_Neg
    - 2: isHadron_Neutral
    - 3: isHadron_Pos
    - 4: isElectron_Neg
    - 5: isElectron_Pos
    - 6: isMuon_Neg
    - 7: isMuon_Pos
    """
    # Get the arguments
    args = get_args()

    # Make sure the destination path exists
    dest_path = Path(args.dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Get all of the root files in the source path
    source_path = Path(args.source_path)
    subfolders = [x for x in source_path.iterdir() if x.is_dir()]

    # Loop over the subfolders
    for subfolder in subfolders:
        print(f"Processing {subfolder.name}")

        # Copy the subfolder to the destination path
        dest_folder = dest_path / subfolder.name

        # Make the folder
        Path(dest_folder).mkdir(parents=True, exist_ok=True)
        files = list(subfolder.glob(args.pattern))

        # Sort the files based the number in the name
        files = sorted(files, key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

        # Loop through the files in the subfolder and load the information
        for file in tqdm(files):
            # Define the destination file
            dest_file = Path(dest_folder) / file.name.replace(".root", ".h5")

            # Skip if the file already exists
            # if Path(dest_file).exists():
            # continue

            # Load the data from the file
            jets, csts, labels = read_jetclass_file(file, num_particles=183)

            # Get the pt from the px and py
            pt = np.linalg.norm(csts[..., :2], axis=-1, keepdims=True)

            # Split the csts into the different groups of information
            sel_csts = np.concatenate([pt, csts[..., 2:8]], axis=-1)

            # Switch to lifetime signing convention for the impact parameters
            d0, z0 = lifetime_signing(
                d0=sel_csts[..., 3],
                z0=sel_csts[..., 5],
                tracks=sel_csts[..., :3],
                jets=jets[..., :3],
                is_centered=True,
            )
            sel_csts[..., 3] = d0
            sel_csts[..., 5] = z0

            # Clip eta and phi to the actual jet radius
            sel_csts[..., 1:3] = np.clip(sel_csts[..., 1:3], -0.8, 0.8)

            # Convert the particle class information to the common format
            csts_id = common_particle_class(
                charge=csts[..., -6],
                isPhoton=csts[..., -5].astype(bool),
                isHadron=csts[..., -4].astype(bool) | csts[..., -3].astype(bool),
                isElectron=csts[..., -2].astype(bool),
                isMuon=csts[..., -1].astype(bool),
            )

            # The jet features need the number of constituents
            mask = sel_csts[..., 0] > 0
            num_csts = np.sum(mask, axis=-1, keepdims=True)
            jets = np.concatenate([jets, num_csts], axis=-1)

            # Save the data to an HDF file
            with h5py.File(dest_file, "w") as f:
                f.create_dataset("csts", data=sel_csts)
                f.create_dataset("csts_id", data=csts_id)
                f.create_dataset("jets", data=jets)
                f.create_dataset("labels", data=labels)
                f.create_dataset("mask", data=mask)


def main_simplified_kinematics() -> None:
    """Convert JetClass root files to HDF format with simplified kinematics.

    Particle features are normalised following get_simplified_kinematics in
    dino/dataloader/processed/processors.py. No particle IDs or impact
    parameters are stored.

    The output csts dataset has 9 dimensions:
    - 0: norm_part_energy  = part_energy / 25
    - 1: norm_part_px      = part_px / 25
    - 2: norm_part_py      = part_py / 25
    - 3: norm_part_pz      = part_pz / 25
    - 4: norm_log_pt       = (log(pt + eps) - 1.5) / 1.5
    - 5: norm_log_energy   = (log(energy + eps) - 1.5) / 1.5
    - 6: norm_delta_R      = (sqrt(deta^2 + dphi^2) - 0.3) / 0.2
    - 7: norm_deta         = deta / 0.3
    - 8: norm_dphi         = dphi / 0.3
    """
    args = get_args()

    dest_path = Path(args.dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source_path)
    subfolders = [x for x in source_path.iterdir() if x.is_dir()]

    for subfolder in subfolders:
        print(f"Processing {subfolder.name}")

        dest_folder = dest_path / subfolder.name
        Path(dest_folder).mkdir(parents=True, exist_ok=True)
        files = list(subfolder.glob(args.pattern))

        files = sorted(files, key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

        for file in tqdm(files):
            dest_file = Path(dest_folder) / file.name.replace(".root", ".h5")

            # Skip if already produced (for resume after partial runs).
            if dest_file.exists() and dest_file.stat().st_size > 0:
                continue

            jets, csts, labels = read_jetclass_file(
                file,
                num_particles=183,
                particle_features=[
                    "part_energy",
                    "part_px",
                    "part_py",
                    "part_pz",
                    "part_deta",
                    "part_dphi",
                ],
            )

            # Unpack columns: [energy, px, py, pz, deta, dphi]
            part_energy = csts[..., 0]
            part_px     = csts[..., 1]
            part_py     = csts[..., 2]
            part_pz     = csts[..., 3]
            part_deta   = csts[..., 4]
            part_dphi   = csts[..., 5]

            part_pt = np.sqrt(part_px**2 + part_py**2)
            mask = part_pt > 0

            # normalize_part_p4: mean=0, std=25
            norm_energy = part_energy / 25.0
            norm_px     = part_px     / 25.0
            norm_py     = part_py     / 25.0
            norm_pz     = part_pz     / 25.0

            # log-normalised kinematics
            norm_log_pt     = (np.log(part_pt     + _EPS) - 1.5) / 1.5
            norm_log_energy = (np.log(part_energy + _EPS) - 1.5) / 1.5

            # angular features
            delta_R      = np.sqrt(part_deta**2 + part_dphi**2)
            norm_delta_R = (delta_R   - 0.3) / 0.2
            norm_deta    = part_deta  / 0.3
            norm_dphi    = part_dphi  / 0.3

            sel_csts = np.stack(
                [
                    norm_energy,
                    norm_px,
                    norm_py,
                    norm_pz,
                    norm_log_pt,
                    norm_log_energy,
                    norm_delta_R,
                    norm_deta,
                    norm_dphi,
                ],
                axis=-1,
            )

            num_csts = np.sum(mask, axis=-1, keepdims=True)
            jets = np.concatenate([jets, num_csts], axis=-1)

            with h5py.File(dest_file, "w") as f:
                f.create_dataset("csts", data=sel_csts)
                f.create_dataset("jets", data=jets)
                f.create_dataset("labels", data=labels)
                f.create_dataset("mask", data=mask)


def main_rino_kinematics() -> None:
    """Convert JetClass root files to HDF format with RINO kinematics.

    Particle features are normalized following get_rino_kinematics in
    dino/dataloader/jetclass/processors.py. Uses the same normalization
    constants for apples-to-apples comparison with RINO (DINO+iBOT).

    The output csts dataset has 7 dimensions:
    - 0: norm_log_pt           = (log(pt + eps) - 1.7) / 1.8
    - 1: norm_log_energy       = (log(energy + eps) - 2.0) / 1.8
    - 2: norm_log_rel_pt       = (log(pt / jet_pt + eps) + 4.7) / 1.8
    - 3: norm_log_rel_energy   = (log(energy / jet_energy + eps) + 4.7) / 1.8
    - 4: norm_delta_R          = (sqrt(deta^2 + dphi^2) - 0.14) / 0.25
    - 5: norm_deta             = deta / 0.14
    - 6: norm_dphi             = dphi / 0.14

    The jets dataset includes 4 jet-level features:
    - 0-3: original jet features (pt, eta, phi, mass)
    - 4: num_constituents
    - 5: norm_log_jet_energy   = (log(jet_energy + eps) - 6.7) / 0.5
    - 6: norm_log_jet_pt       = (log(jet_pt + eps) - 6.4) / 0.25
    - 7: norm_jet_eta          = jet_eta / 1.3
    - 8: norm_jet_phi          = jet_phi / pi
    """
    args = get_args()

    dest_path = Path(args.dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source_path)
    subfolders = [x for x in source_path.iterdir() if x.is_dir()]

    # RINO normalization constants (from dino/dataloader/jetclass/processors.py)
    import math

    NORM = {
        "log_pt": {"mean": 1.7, "std": 1.8},
        "log_energy": {"mean": 2.0, "std": 1.8},
        "log_rel_pt": {"mean": -4.7, "std": 1.8},
        "log_rel_energy": {"mean": -4.7, "std": 1.8},
        "deta": {"mean": 0.0, "std": 0.14},
        "dphi": {"mean": 0.0, "std": 0.14},
        "delta_R": {"mean": 0.14, "std": 0.25},
        "jet_log_energy": {"mean": 6.7, "std": 0.5},
        "jet_log_pt": {"mean": 6.4, "std": 0.25},
        "jet_eta": {"mean": 0.0, "std": 1.3},
        "jet_phi": {"mean": 0.0, "std": math.pi},
    }

    def _norm(x, key):
        return (x - NORM[key]["mean"]) / NORM[key]["std"]

    for subfolder in subfolders:
        print(f"Processing {subfolder.name}")

        dest_folder = dest_path / subfolder.name
        Path(dest_folder).mkdir(parents=True, exist_ok=True)
        files = list(subfolder.glob(args.pattern))
        files = sorted(files, key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

        for file in tqdm(files):
            dest_file = Path(dest_folder) / file.name.replace(".root", ".h5")

            # Skip if already produced (for resume after partial runs).
            if dest_file.exists() and dest_file.stat().st_size > 0:
                continue

            jets, csts, labels = read_jetclass_file(
                file,
                num_particles=183,
                particle_features=[
                    "part_energy",
                    "part_px",
                    "part_py",
                    "part_pz",
                    "part_deta",
                    "part_dphi",
                ],
                jet_features=[
                    "jet_pt",
                    "jet_eta",
                    "jet_phi",
                    "jet_energy",
                ],
            )

            # Unpack particle columns: [energy, px, py, pz, deta, dphi]
            part_energy = csts[..., 0]
            part_px = csts[..., 1]
            part_py = csts[..., 2]
            part_deta = csts[..., 4]
            part_dphi = csts[..., 5]

            part_pt = np.sqrt(part_px**2 + part_py**2)
            mask = part_pt > 0

            # Jet kinematics: [jet_pt, jet_eta, jet_phi, jet_energy]
            jet_pt = jets[..., 0:1]
            jet_eta = jets[..., 1:2]
            jet_phi = jets[..., 2:3]
            jet_energy = jets[..., 3:4]

            # 7 RINO particle features
            f0 = _norm(np.log(part_pt + _EPS), "log_pt")
            f1 = _norm(np.log(part_energy + _EPS), "log_energy")
            f2 = _norm(np.log(part_pt / (jet_pt + _EPS) + _EPS), "log_rel_pt")
            f3 = _norm(np.log(part_energy / (jet_energy + _EPS) + _EPS), "log_rel_energy")
            delta_R = np.sqrt(part_deta**2 + part_dphi**2)
            f4 = _norm(delta_R, "delta_R")
            f5 = _norm(part_deta, "deta")
            f6 = _norm(part_dphi, "dphi")

            sel_csts = np.stack([f0, f1, f2, f3, f4, f5, f6], axis=-1)
            sel_csts[~mask] = 0.0

            # Jet-level features
            num_csts = np.sum(mask, axis=-1, keepdims=True)
            norm_jet = np.concatenate(
                [
                    jets,
                    num_csts,
                    _norm(np.log(jet_energy + _EPS), "jet_log_energy"),
                    _norm(np.log(jet_pt + _EPS), "jet_log_pt"),
                    _norm(jet_eta, "jet_eta"),
                    _norm(jet_phi, "jet_phi"),
                ],
                axis=-1,
            )

            with h5py.File(dest_file, "w") as f:
                f.create_dataset("csts", data=sel_csts)
                f.create_dataset("jets", data=norm_jet)
                f.create_dataset("labels", data=labels)
                f.create_dataset("mask", data=mask)


if __name__ == "__main__":
    main_rino_kinematics()
