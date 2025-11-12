import torch
from matplotlib import pyplot as plt
import json
import argparse
import math

from pathlib import Path
import logging
from sklearn.metrics import roc_curve, auc


logging.basicConfig(level=logging.INFO)

FEATURE_LABELS = [
    "norm_part_px",
    "norm_part_py",
    "norm_part_pz",
    "norm_part_energy",
    "log_part_pt",
    "log_part_energy",
    "log_rel_part_pt",
    "log_rel_part_energy",
    "part_delta_R",
    "part_deta",
    "part_dphi",
]

JET_FEATURE_LABELS = [
    "nparticles",
    "energy", 
    "pt",
    "eta",
    "phi"
]


def unnormalize(x, means, std_devs):
    """Unnormalize a tensor."""
    return x * std_devs + means

def untransform_jets(jets):
    norm_nparticles, norm_log_energy, norm_log_pt, norm_eta, norm_phi = jets.unbind(axis=-1)
    nparticles = unnormalize(norm_nparticles, 40.0, 15.0)
    log_energy = unnormalize(norm_log_energy, 7.0, 1.0)
    log_pt = unnormalize(norm_log_pt, 6.5, 0.5)
    eta = unnormalize(norm_eta, 0.0, 2.0)
    phi = unnormalize(norm_phi, 0.0, math.pi)
    
    energy = torch.exp(log_energy)
    pt = torch.exp(log_pt)
    return nparticles, energy, pt, eta, phi

def unnormalize_dR_deta_dphi(norm_delta_R, norm_deta, norm_dphi):
    part_delta_R = unnormalize(norm_delta_R, 0.3, 0.2)
    part_deta = unnormalize(norm_deta, 0.0, 0.3)
    part_dphi = unnormalize(norm_dphi, 0.0, 0.3)
    return part_delta_R, part_deta, part_dphi
    

def unnormalize_particles(particles):
    (
        norm_part_px,
        norm_part_py,
        norm_part_pz,
        norm_part_energy,
        norm_log_part_pt,
        norm_log_part_energy,
        norm_log_rel_part_pt,
        norm_log_rel_part_energy,
        norm_part_delta_R,
        norm_part_deta,
        norm_part_dphi,
    ) = particles.unbind(axis=-1)
    
    part_px = unnormalize(norm_part_px, 0.0, 25.0)
    part_py = unnormalize(norm_part_py, 0.0, 25.0)
    part_pz = unnormalize(norm_part_pz, 0.0, 25.0)
    part_energy = unnormalize(norm_part_energy, 0.0, 25.0)
    log_part_pt = unnormalize(norm_log_part_pt, 1.5, 1.5)
    log_part_energy = unnormalize(norm_log_part_energy, 1.5, 1.5)
    log_rel_part_pt = unnormalize(norm_log_rel_part_pt, -4.5, 1.5)
    log_rel_part_energy = unnormalize(norm_log_rel_part_energy, -5, 1.5)
    part_delta_R, part_deta, part_dphi = unnormalize_dR_deta_dphi(
        norm_part_delta_R, norm_part_deta, norm_part_dphi
    )

    return (
        part_px,
        part_py,
        part_pz,
        part_energy,
        log_part_pt,
        log_part_energy,
        log_rel_part_pt,
        log_rel_part_energy,
        part_delta_R,
        part_deta,
        part_dphi
    )
    

def create_kinematic_mask(
    data, 
    jet_pt_bounds: tuple[float, float] = (500, 650)
) -> torch.Tensor:
    jets = data["jets"]
    _, _, jet_pt, _, _ = untransform_jets(jets)
    
    particles = data["particles"]
    (
        norm_part_px,
        norm_part_py,
        norm_part_pz,
        norm_part_energy,
        norm_log_part_pt,
        norm_log_part_energy,
        norm_log_rel_part_pt,
        norm_log_rel_part_energy,
        norm_part_delta_R,
        norm_part_deta,
        norm_part_dphi,
    ) = particles.unbind(axis=-1)
    
    part_delta_R, part_deta, part_dphi = unnormalize_dR_deta_dphi(
        norm_part_delta_R, norm_part_deta, norm_part_dphi
    )

    jet_mask = (jet_pt > jet_pt_bounds[0]) & (jet_pt < jet_pt_bounds[1])
    
    # Get the actual particle mask to only consider real particles
    real_particle_mask = data["mask"]
    
    # Particle kinematic constraints
    particle_kinematic_mask = (
        (part_delta_R < 0.8 * 2)
        & (torch.abs(part_deta) < 0.8 * 2)
        & (torch.abs(part_dphi) < 0.8 * 2)
    )
    
    # Apply kinematic constraints only to real particles
    # For padded particles (where real_particle_mask is False), we don't care about constraints
    particle_constraints_satisfied = particle_kinematic_mask | (~real_particle_mask)
    
    # Keep events where ALL real particles satisfy the kinematic constraints
    all_particles_pass = particle_constraints_satisfied.all(axis=-1)
    
    return jet_mask & all_particles_pass
    
def analyze(
    data_dir: Path | str,
    data_tags: list[str],
    data_splits: list[str] = ("train", "val", "test"),
    jet_pt_min: float = -math.inf,
    jet_pt_max: float = math.inf,
):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    
    # Data label mapping
    data_label_dict = {
        "jetclass": "JetClass",
        "toptagging": "TopTagging",
        "jetnet": "JetNet",
        "jetnet30": "JetNet30",
    }
    
    metric_dict = {}
    
    # make plots of input distributions
    for data_split in data_splits:
        logging.info(f"Analyzing data split: {data_split}")
        
        # Build paths for each data tag
        paths = []
        for data_tag in data_tags:
            path = data_dir / f"output_{data_split}_{data_tag}_best-0.pt"
            paths.append(path)
        
        logging.info(f"Loading data from {[str(p) for p in paths]}...")
        
        # Load existing data files
        datasets = []
        valid_data_tags = []
        for path, data_tag in zip(paths, data_tags):
            if path.exists():
                data = torch.load(path, map_location=torch.device("cpu"))
                datasets.append(data)
                valid_data_tags.append(data_tag)
                logging.info(f"Loaded data from {path}")
            else:
                logging.warning(f"Data file {path} not found. Skipping {data_tag}...")
        
        if not datasets:
            logging.error(f"No data files found for {data_split}. Skipping...")
            continue

        fig_dir = data_dir / f"figures_pt{jet_pt_min}to{jet_pt_max}/{data_split}"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Create kinematic masks for all datasets
        kinematic_masks = []
        for data in datasets:
            if "particles" in data and "jets" in data:
                plot = True
                mask = create_kinematic_mask(data, jet_pt_bounds=(jet_pt_min, jet_pt_max))
            else:
                plot = False
                logging.warning(f"No particles or jets found in data for {data_split}. Skipping kinematic mask creation.")
                mask = torch.ones_like(data["label"], dtype=torch.bool)
            kinematic_masks.append(mask)

        for data_tag, mask in zip(valid_data_tags, kinematic_masks):
            fraction_passing = mask.float().mean().item()
            data_label = data_label_dict.get(data_tag.lower(), data_tag)
            logging.info(f"Fraction of jets passing kinematic cuts for {data_split} - {data_label}: {fraction_passing:.2f}")

        logging.info(f"Compute Accuracy and AUC for {data_split}...")
        plt.figure(figsize=(10, 5))
        
        for data_tag, data, kinematic_mask in zip(valid_data_tags, datasets, kinematic_masks):
            data_label = data_label_dict.get(data_tag.lower(), data_tag)
            
            # if nothing left after kinematic cuts, skip
            if not kinematic_mask.any():
                logging.warning(f"No jets left after kinematic cuts for {data_label} in {data_split}. Skipping...")
                continue
            
            logits = data["logits"]
            labels = data["label"]
            
            filtered_logits = logits[(labels != 9) & kinematic_mask]  
            filtered_labels = labels[(labels != 9) & kinematic_mask]
            if data_tag.lower() == "jetclass":  # Only apply to JetClass
                filtered_labels[filtered_labels == 8] = 1

            pred = (filtered_logits[:, 0] > 0).long()
            acc = (pred == filtered_labels).float().mean().item()
            
            # make roc curve for Top vs QCD
            fpr, tpr, thresholds = roc_curve(
                filtered_labels.cpu().numpy(), 
                filtered_logits[:, 0].detach().cpu().numpy(), 
                pos_label=1
            )
            roc_auc = auc(fpr, tpr)
            
            metric_dict[data_label] = {
                "roc_auc": roc_auc,
                "acc": acc
            }

            plt.plot(tpr, fpr, label=f"{data_label} (area = {roc_auc:.2f})")
        
        plt.xlabel(r"$\epsilon_\mathrm{s}$")
        plt.ylabel(r"$\epsilon_\mathrm{b}$")
        plt.yscale("log")

        plt.grid()
        plt.legend()
        plt.ylim(1e-4, 1)

        plt.tight_layout()
        plt.savefig(fig_dir / "roc_curve.pdf", bbox_inches="tight")
        plt.close()

        
        logging.info(f"Accuracy and AUC for {data_split}: {metric_dict}")
        # save metrics to a JSON file
        with open(fig_dir / f"metrics.json", "w") as f:
            json.dump(metric_dict, f, indent=4)

        # Plot jet features (unnormalized)
        if plot:
            logging.info(f"Plotting jet features for {data_split}...")
            for i, jet_feature_name in enumerate(JET_FEATURE_LABELS):
                logging.info(f"Plotting jet feature: {jet_feature_name}...")
                plt.figure(figsize=(10, 5))

                for data_tag, data, kinematic_mask in zip(valid_data_tags, datasets, kinematic_masks):
                    data_label = data_label_dict.get(data_tag.lower(), data_tag)
                    
                    # Get unnormalized jet features
                    jets = data["jets"]
                    nparticles, energy, pt, eta, phi = untransform_jets(jets)
                    jet_features = torch.stack([nparticles, energy, pt, eta, phi], dim=-1)
                    
                    for signal_type in ("QCD", "Top"):
                        if signal_type == "QCD":
                            target_label = 0
                        else:
                            if data_tag.lower() == "jetclass":
                                target_label = 8
                            else:
                                target_label = 1

                        # Choose the right label in the selected phase space
                        sample_mask = (data["label"] == target_label) & kinematic_mask

                        jet_feature_values = jet_features[sample_mask, i].cpu().numpy()
                        
                        plt.hist(
                            jet_feature_values,
                            bins=100, 
                            density=True,
                            histtype='step',
                            label=f"{data_label} {signal_type}",
                            alpha=0.8
                        )
                
                plt.legend()
                plt.xlabel(jet_feature_name)
                plt.ylabel("Density")
                plt.grid()
                plt.tight_layout()
                plt.savefig(fig_dir / f"jet_{jet_feature_name}.pdf", bbox_inches="tight")
                plt.close()

            # Plot particle features (with proper masking to avoid -2, -3 values)
            logging.info(f"Plotting particle features for {data_split}...")
            for i in range(len(FEATURE_LABELS)):
                logging.info(f"Plotting {FEATURE_LABELS[i]}...")
                plt.figure(figsize=(10, 5))

                for data_tag, data, kinematic_mask in zip(valid_data_tags, datasets, kinematic_masks):
                    data_label = data_label_dict.get(data_tag.lower(), data_tag)
                    
                    for signal_type in ("QCD", "Top"):
                        if signal_type == "QCD":
                            target_label = 0
                        else:
                            if data_tag.lower() == "jetclass":
                                target_label = 8
                            else:
                                target_label = 1

                        # Choose the right label in the selected phase space
                        sample_mask = (data["label"] == target_label) & kinematic_mask

                        # Get particle features and particle mask
                        particle_features = data["particles"][sample_mask][..., i]  # [N_selected_jets, N_particles]
                        particle_mask = data["mask"][sample_mask]  # [N_selected_jets, N_particles]
                        
                        # Apply particle mask to filter out invalid particles (where mask is False)
                        # This should remove the -2, -3 values which correspond to masked/padded particles
                        valid_features = particle_features[particle_mask].cpu().numpy()
                        
                        # Additional filtering to remove any remaining invalid values
                        # (in case there are still some edge cases)
                        if FEATURE_LABELS[i] in ["part_deta", "part_dphi"]:
                            # For delta eta and delta phi, remove extreme values that look like padding
                            valid_features = valid_features[(valid_features > -1.5) & (valid_features < 1.5)]
                        elif FEATURE_LABELS[i] == "part_delta_R":
                            # For delta R, remove negative values and very large values
                            valid_features = valid_features[(valid_features >= 0) & (valid_features < 2.0)]
                        
                        if len(valid_features) > 0:  # Only plot if we have valid data
                            plt.hist(
                                valid_features,
                                bins=200, 
                                density=True,
                                histtype='step',
                                label=f"{data_label} {signal_type}",
                                alpha=0.8
                            )
                
                plt.legend()
                plt.xlabel(FEATURE_LABELS[i])
                plt.ylabel("Density")
                plt.yscale("log")
                plt.grid()
                plt.tight_layout()
                plt.savefig(fig_dir / f"particle_{FEATURE_LABELS[i]}.pdf", bbox_inches="tight")
                plt.close()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze inference results.")
    parser.add_argument(
        "-d",
        "--data-dir", 
        type=str, 
        help="Path to the directory containing the inference data."
    )
    parser.add_argument(
        "--data-tags",
        nargs="+",
        default=["jetclass", "toptagging", "jetnet", "jetnet30"],
        choices=["jetclass", "toptagging", "jetnet", "jetnet30"],
        help="List of data tags (e.g. jetclass, toptagging, jetnet) to analyze."
    )
    parser.add_argument(
        "--data-splits", 
        nargs="+", 
        default=["test"], 
        choices=["train", "val", "test"],
        help="List of data splits (e.g. train, val, test) to analyze."
    )
    parser.add_argument(
        "--jet-pt-min",
        type=float,
        default=-math.inf,
        help="Minimum jet pT for filtering."
    )
    parser.add_argument(
        "--jet-pt-max",
        type=float,
        default=math.inf,
        help="Maximum jet pT for filtering."
    )

    args = parser.parse_args()
    
    analyze(args.data_dir, args.data_tags, args.data_splits, args.jet_pt_min, args.jet_pt_max)