#!/usr/bin/env python3
"""Scale invariance analysis for DINO/iBOT pretrained models.

For each jet, passes it through the model at every available coarse-graining
scale (ALL, subjet2, subjet3, ..., subjet16) and collects the CLS-token
representation. Then computes pairwise cosine similarity across scales to
test whether the representation is invariant under the RG flow.

Usage:
    python dino/scale_invariance_analysis.py \
        --config configs/dino/<pretrain>.yaml \
        --load-epoch best \
        --num-jets 10000 \
        --output-dir experiments/<JOBNAME>/scale_invariance

The script produces:
    - cosine_similarity_matrix.pdf  — heatmap of mean pairwise cosine sim
    - cosine_similarity_stats.json  — full statistics per scale pair
    - per_jet_variance.pdf          — histogram of per-jet embedding variance
    - scale_embeddings.pt           — raw embeddings for further analysis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure project root is on sys.path
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))

from utils.producers import get_models, get_dataloader_and_config, get_config
from utils.ckpt import process_placeholder, get_checkpoints_path
from utils.logger import LOGGER, configure_logger
from models import AssembledModel
from dino_train import check_bf16_support

torch.set_float32_matmul_precision("high")

# All scales present in the jetclass-clustered data
ALL_SCALES = ["ALL", "subjet2", "subjet3", "subjet4", "subjet6", "subjet8", "subjet16"]


def load_teacher_backbone(config: dict, device: torch.device) -> AssembledModel:
    """Load the teacher backbone for representation extraction."""
    # Get part_dim from the training dataloader config
    train_dl = config["training"]["dataloader"]["train"]
    preprocessed = train_dl.get("preprocessed", False)
    dataloader_config = get_config(
        config=config, mode="training", preprocessed=preprocessed
    )
    part_dim = len(dataloader_config.outputs.sequence)

    (
        _student,
        (teacher_backbone, _teacher_dino_head, _teacher_ibot_head),
        _ibot_pe,
        _scale_emb,
    ) = get_models(
        part_dim=part_dim,
        config=config,
        mode="inference",
        device=device,
    )
    model = AssembledModel(
        backbone=teacher_backbone,
        embedding=None,
        heads=None,  # We only need representations
    )
    model.eval()
    return model


def _masked_mean_particles(particles_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean over particle tokens (excluding CLS/registers).

    Args:
        particles_out: (B, N, d_model) — particle tokens only (prefix already stripped).
        mask: (B, N) — True = valid particle, False = padding.
    """
    masked = particles_out * mask.unsqueeze(-1).float()
    return masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)


@torch.no_grad()
def extract_scale_embeddings(
    model: AssembledModel,
    dataloader,
    device: torch.device,
    num_jets: int,
    use_bf16: bool = True,
    cat_mean_pool: bool = False,
    pooling_override: str | None = None,
) -> dict[str, torch.Tensor]:
    """Extract embeddings at each scale for the same jets.

    Args:
        pooling_override: If "mean", compute mean over particle tokens instead
            of using the model's default pooling (CLS). If None, uses the
            model's native pooling (CLS token for DINO-pretrained models).

    Returns:
        dict mapping scale name -> (N, d_model) tensor of embeddings.
    """
    use_mean = pooling_override == "mean"
    pool_label = "mean-particle" if use_mean else "CLS"
    LOGGER.info(f"Pooling mode: {pool_label}")

    embeddings = {scale: [] for scale in ALL_SCALES}
    labels = []
    n_collected = 0

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        if n_collected >= num_jets:
            break

        particles = torch.tensor(batch["sequence"], dtype=torch.float32).to(device)
        jets = torch.tensor(batch["class_"], dtype=torch.float32).to(device)
        mask = torch.tensor(batch["mask"], dtype=torch.bool).to(device)
        views = batch.get("views", {})
        batch_labels = batch["aux"].get("label")

        bs = particles.shape[0]
        take = min(bs, num_jets - n_collected)

        # ALL = raw particle view
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            if use_mean:
                # Get both pooled rep and particle tokens
                rep_cls, particles_out = model.backbone(
                    model.embedding(particles) if not isinstance(model.embedding, torch.nn.Identity) else particles,
                    mask=mask, jets=jets,
                )
                rep_all = _masked_mean_particles(particles_out, mask)
            else:
                rep_all = model(
                    particles=particles, jets=jets, mask=mask,
                    rep_only=True, cat_mean_pool=cat_mean_pool,
                )
        embeddings["ALL"].append(rep_all[:take].float().cpu())

        # Subjet views
        for scale in ALL_SCALES:
            if scale == "ALL":
                continue
            if scale not in views:
                LOGGER.warning(f"Scale {scale} not found in batch views, skipping")
                continue

            view_data = views[scale]
            view_particles = view_data["features"].to(dtype=torch.float32, device=device)
            view_mask = view_data["mask"].to(dtype=torch.bool, device=device)
            view_jets = jets
            if "jets" in view_data:
                view_jets = view_data["jets"].to(dtype=torch.float32, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                if use_mean:
                    _, particles_out = model.backbone(
                        model.embedding(view_particles) if not isinstance(model.embedding, torch.nn.Identity) else view_particles,
                        mask=view_mask, jets=view_jets,
                    )
                    rep = _masked_mean_particles(particles_out, view_mask)
                else:
                    rep = model(
                        particles=view_particles, jets=view_jets, mask=view_mask,
                        rep_only=True, cat_mean_pool=cat_mean_pool,
                    )
            embeddings[scale].append(rep[:take].float().cpu())

        if batch_labels is not None:
            labels.append(torch.tensor(batch_labels[:take]))

        n_collected += take

    # Concatenate
    result = {}
    for scale in ALL_SCALES:
        if embeddings[scale]:
            result[scale] = torch.cat(embeddings[scale], dim=0)
    if labels:
        result["labels"] = torch.cat(labels, dim=0)

    return result


def compute_invariance_metrics(
    embeddings: dict[str, torch.Tensor],
) -> dict:
    """Compute pairwise cosine similarity and per-jet variance across scales.

    Returns dict with:
        - pairwise_cosine: {(s1, s2): {mean, std, median, min, max}}
        - per_jet_mean_cosine: (N,) mean cosine sim for each jet across all scale pairs
        - per_jet_std: (N,) std of embeddings across scales for each jet
        - scale_names: list of scale names used
    """
    scales = [s for s in ALL_SCALES if s in embeddings]
    n_jets = embeddings[scales[0]].shape[0]

    # Normalize embeddings for cosine similarity
    normed = {s: F.normalize(embeddings[s], dim=-1) for s in scales}

    # Pairwise cosine similarity
    pairwise = {}
    for i, s1 in enumerate(scales):
        for j, s2 in enumerate(scales):
            if j <= i:
                continue
            cos_sim = (normed[s1] * normed[s2]).sum(dim=-1)  # (N,)
            pairwise[(s1, s2)] = {
                "mean": float(cos_sim.mean()),
                "std": float(cos_sim.std()),
                "median": float(cos_sim.median()),
                "min": float(cos_sim.min()),
                "max": float(cos_sim.max()),
            }

    # Per-jet statistics: stack all scale embeddings and measure variance
    stacked = torch.stack([normed[s] for s in scales], dim=1)  # (N, S, d)
    per_jet_mean_emb = stacked.mean(dim=1, keepdim=True)  # (N, 1, d)
    per_jet_cos_to_mean = (
        F.normalize(stacked, dim=-1) * F.normalize(per_jet_mean_emb.expand_as(stacked), dim=-1)
    ).sum(dim=-1)  # (N, S)
    per_jet_mean_cosine = per_jet_cos_to_mean.mean(dim=1)  # (N,)

    # Embedding variance across scales (L2 distance from centroid)
    per_jet_var = ((stacked - per_jet_mean_emb) ** 2).sum(dim=-1).mean(dim=1)  # (N,)

    # Per-scale breakdown: (N, S) cosine and (N, S) L2 variance per scale
    per_jet_var_per_scale = ((stacked - per_jet_mean_emb) ** 2).sum(dim=-1)  # (N, S)

    return {
        "pairwise_cosine": pairwise,
        "per_jet_mean_cosine": per_jet_mean_cosine,
        "per_jet_var": per_jet_var,
        "per_jet_cos_per_scale": per_jet_cos_to_mean,  # (N, S)
        "per_jet_var_per_scale": per_jet_var_per_scale,  # (N, S)
        "scale_names": scales,
    }


def compute_label_stratified_metrics(
    embeddings: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict:
    """Compute invariance metrics stratified by jet class label."""
    unique_labels = labels.unique().tolist()
    # JetClass label names (indices 0-9)
    label_names = {
        0: "QCD", 1: "Hbb", 2: "Hcc", 3: "Hgg", 4: "H4q",
        5: "Hqql", 6: "Zqq", 7: "Wqq", 8: "Tbqq", 9: "Tbl",
    }
    results = {}
    for lbl in unique_labels:
        mask = labels == lbl
        sub_emb = {s: embeddings[s][mask] for s in embeddings if s in ALL_SCALES}
        if not sub_emb:
            continue
        metrics = compute_invariance_metrics(sub_emb)
        name = label_names.get(int(lbl), f"class_{int(lbl)}")
        results[name] = {
            "n_jets": int(mask.sum()),
            "mean_cosine_to_centroid": float(metrics["per_jet_mean_cosine"].mean()),
            "mean_variance": float(metrics["per_jet_var"].mean()),
        }
    return results


def plot_cosine_matrix(
    metrics: dict,
    output_path: Path,
    title: str = "",
    unseen_scales: list[str] | None = None,
):
    """Plot pairwise cosine similarity heatmap.

    Args:
        unseen_scales: If provided, color held-out scale tick labels in crimson.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available, skipping plot")
        return

    scales = metrics["scale_names"]
    n = len(scales)
    matrix = np.eye(n)

    for i, s1 in enumerate(scales):
        for j, s2 in enumerate(scales):
            if j <= i:
                continue
            val = metrics["pairwise_cosine"][(s1, s2)]["mean"]
            matrix[i, j] = val
            matrix[j, i] = val

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] < 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(scales, rotation=45, ha="right")
    ax.set_yticklabels(scales)

    # Color held-out scale labels in crimson
    if unseen_scales:
        for i, s in enumerate(scales):
            if s in unseen_scales:
                ax.get_xticklabels()[i].set_color("crimson")
                ax.get_xticklabels()[i].set_fontweight("bold")
                ax.get_yticklabels()[i].set_color("crimson")
                ax.get_yticklabels()[i].set_fontweight("bold")

    fig.colorbar(im, ax=ax, label="Mean Cosine Similarity")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved cosine similarity matrix to {output_path}")


def compute_heldout_metrics(
    embeddings: dict[str, torch.Tensor],
    seen_scales: list[str],
    unseen_scales: list[str],
) -> dict:
    """Compare CLS similarity between seen and unseen (heldout) scales.

    For the heldout models, training used a subset of scales and validation
    used unseen interpolation scales. This function quantifies whether the
    representation smoothly interpolates to unseen scales.

    Args:
        embeddings: dict mapping scale name -> (N, d_model) tensor
        seen_scales: scales used during training (e.g. ["subjet2", "subjet4", "subjet8", "subjet16"])
        unseen_scales: scales held out during training (e.g. ["subjet3", "subjet6"])

    Returns:
        dict with:
        - seen_seen_cosine: mean cosine between pairs of seen scales
        - unseen_unseen_cosine: mean cosine between pairs of unseen scales
        - seen_unseen_cosine: mean cosine between seen and unseen scales
        - interpolation_gap: seen_seen - seen_unseen (smaller = better interpolation)
        - per_unseen: {scale: mean cosine to all seen scales}
    """
    available_seen = [s for s in seen_scales if s in embeddings]
    available_unseen = [s for s in unseen_scales if s in embeddings]

    if not available_seen or not available_unseen:
        return {"error": "Not enough scales available for heldout analysis"}

    normed = {s: F.normalize(embeddings[s], dim=-1) for s in available_seen + available_unseen}

    def mean_pairwise_cosine(scales_a, scales_b):
        """Mean cosine similarity across all (a, b) pairs."""
        cos_vals = []
        for sa in scales_a:
            for sb in scales_b:
                if sa == sb:
                    continue
                cos = (normed[sa] * normed[sb]).sum(dim=-1).mean().item()
                cos_vals.append(cos)
        return float(np.mean(cos_vals)) if cos_vals else 0.0

    seen_seen = mean_pairwise_cosine(available_seen, available_seen)
    unseen_unseen = mean_pairwise_cosine(available_unseen, available_unseen)
    seen_unseen = mean_pairwise_cosine(available_seen, available_unseen)

    # Per-unseen scale: how well does each unseen scale match the seen centroid?
    per_unseen = {}
    for us in available_unseen:
        cos_to_seen = []
        for ss in available_seen:
            cos = (normed[us] * normed[ss]).sum(dim=-1).mean().item()
            cos_to_seen.append(cos)
        per_unseen[us] = {
            "mean_cosine_to_seen": float(np.mean(cos_to_seen)),
            "min_cosine_to_seen": float(np.min(cos_to_seen)),
            "max_cosine_to_seen": float(np.max(cos_to_seen)),
        }

    return {
        "seen_scales": available_seen,
        "unseen_scales": available_unseen,
        "seen_seen_cosine": seen_seen,
        "unseen_unseen_cosine": unseen_unseen,
        "seen_unseen_cosine": seen_unseen,
        "interpolation_gap": seen_seen - seen_unseen,
        "per_unseen": per_unseen,
    }


# Heldout scale definitions (from design-notes.md)
HELDOUT_CONFIGS = {
    # Row 9: mixed heldout — train {2,4,8,16}, val {3,6}
    "heldout-ptrank": {
        "seen": ["subjet2", "subjet4", "subjet8", "subjet16"],
        "unseen": ["subjet3", "subjet6"],
    },
    # Row 11: g6l2 heldout — train {8,16}/{2,4,ALL}, val {3,6}
    "heldout-g6l2-ptrank": {
        "seen": ["subjet2", "subjet4", "subjet8", "subjet16", "ALL"],
        "unseen": ["subjet3", "subjet6"],
    },
    # Row 12: g6l2 heldout pbin
    "heldout-g6l2-pbin": {
        "seen": ["subjet2", "subjet4", "subjet8", "subjet16", "ALL"],
        "unseen": ["subjet3", "subjet6"],
    },
}


def plot_per_jet_histogram(metrics: dict, output_path: Path, **kwargs):
    """Plot per-jet consistency and variance as separate PDFs (step histograms)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available, skipping plot")
        return

    cos_vals = metrics["per_jet_mean_cosine"].numpy()
    var_vals = metrics["per_jet_var"].numpy()
    output_dir = output_path.parent

    # 1. Per-jet consistency (cosine to centroid) — always overall, no seen/unseen split
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(cos_vals, bins=80, histtype='step', linewidth=1.5, alpha=0.8)
    ax.axvline(cos_vals.mean(), color="red", linestyle="--",
               label=f"mean = {cos_vals.mean():.4f}")
    ax.set_xlabel("Mean Cosine Similarity to Scale Centroid")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    consistency_path = output_dir / "per_jet_consistency.pdf"
    fig.savefig(consistency_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved per-jet consistency to {consistency_path}")

    # 2. Per-jet variance — always overall
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(var_vals, bins=80, histtype='step', linewidth=1.5, alpha=0.8, color='orange')
    ax.axvline(var_vals.mean(), color="red", linestyle="--",
               label=f"mean = {var_vals.mean():.4f}")
    ax.set_xlabel("Per-Jet Embedding Variance Across Scales")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    variance_path = output_dir / "per_jet_variance.pdf"
    fig.savefig(variance_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Saved per-jet variance to {variance_path}")


def plot_pca_scales(
    metrics: dict,
    output_dir: Path,
    unseen_scales: list[str] | None = None,
    n_jets_plot: int = 2000,
):
    """Plot PCA projections of embeddings colored by scale.

    Produces:
        - pca_scales.pdf: First 3 PCA components as corner plot (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        LOGGER.warning("matplotlib/sklearn not available, skipping PCA plot")
        return

    if "per_jet_cos_per_scale" not in metrics:
        LOGGER.warning("No per-scale data in metrics, skipping PCA plot")
        return

    scales = metrics["scale_names"]
    cos_per_scale = metrics["per_jet_cos_per_scale"]  # (N, S)
    N, S = cos_per_scale.shape

    # We need the raw embeddings — reconstruct from the stacked normed embeddings
    # Actually we need the embeddings dict. Let's check if it was passed through.
    # Since metrics doesn't carry raw embeddings, we'll use a different approach:
    # The caller should pass embeddings separately. For now, skip if not available.
    LOGGER.warning("PCA plot requires raw embeddings — use plot_pca_from_embeddings()")
    return


def plot_pca_from_embeddings(
    embeddings: dict[str, torch.Tensor],
    output_dir: Path,
    unseen_scales: list[str] | None = None,
    n_jets_plot: int = 2000,
    n_components: int = 5,
):
    """Plot PCA corner plots of embeddings.

    Produces:
        1. pca_by_scale.pdf — corner plot colored by scale (diagonal = per-scale histograms)
        2. pca_seen_vs_unseen.pdf — (heldout only) corner plot, seen vs unseen coloring

    Same PCA transformation for both plots.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        LOGGER.warning("matplotlib/sklearn not available, skipping PCA plot")
        return

    scales = [s for s in ALL_SCALES if s in embeddings and isinstance(embeddings[s], torch.Tensor)]
    if len(scales) < 2:
        return

    # Subsample jets
    N = embeddings[scales[0]].shape[0]
    idx = np.random.RandomState(42).choice(N, min(n_jets_plot, N), replace=False)

    # Stack all scales: (N_sub * S, d)
    all_embs = []
    all_scale_labels = []
    for s in scales:
        emb = F.normalize(embeddings[s][idx], dim=-1).numpy()
        all_embs.append(emb)
        all_scale_labels.extend([s] * len(idx))
    all_embs = np.concatenate(all_embs, axis=0)
    all_scale_labels = np.array(all_scale_labels)

    # Single PCA fit
    nc = min(n_components, all_embs.shape[1])
    pca = PCA(n_components=nc)
    projected = pca.fit_transform(all_embs)
    var_explained = pca.explained_variance_ratio_

    unseen_set = set(unseen_scales or [])

    # Color map for scales
    seen_list = [s for s in scales if s not in unseen_set]
    cmap_vals = plt.cm.viridis(np.linspace(0.15, 0.85, len(seen_list)))
    scale_colors = {}
    ci = 0
    for s in scales:
        if s in unseen_set:
            scale_colors[s] = "crimson"
        else:
            scale_colors[s] = cmap_vals[ci]
            ci += 1

    def _corner_plot(color_fn, label_fn, marker_fn, out_path, legend_title=None):
        """Draw an NxN corner plot: lower-tri = scatter, diagonal = step histogram."""
        fig, axes = plt.subplots(nc, nc, figsize=(3.5 * nc, 3.5 * nc))
        legend_handles = []
        legend_labels_seen = set()

        for i in range(nc):
            for j in range(nc):
                ax = axes[i][j]
                if j > i:
                    # Upper triangle — hide
                    ax.set_visible(False)
                elif i == j:
                    # Diagonal — histograms
                    groups = {}
                    for k, lbl in enumerate(all_scale_labels):
                        g = label_fn(lbl)
                        groups.setdefault(g, []).append(projected[k, i])
                    for g in sorted(groups.keys()):
                        c = color_fn(g)
                        ax.hist(groups[g], bins=50, histtype='step',
                                linewidth=1.2, color=c, alpha=0.8,
                                density=True)
                    ax.set_ylabel("Density")
                else:
                    # Lower triangle — scatter
                    groups = {}
                    for k, lbl in enumerate(all_scale_labels):
                        g = label_fn(lbl)
                        groups.setdefault(g, []).append(k)
                    for g in sorted(groups.keys()):
                        idxs = groups[g]
                        c = color_fn(g)
                        m = marker_fn(g)
                        h = ax.scatter(projected[idxs, j], projected[idxs, i],
                                       c=[c], s=4, alpha=0.4, marker=m,
                                       edgecolors="none", label=g)
                        if g not in legend_labels_seen:
                            legend_handles.append(h)
                            legend_labels_seen.add(g)

                # Axis labels on edges only
                if i == nc - 1:
                    ax.set_xlabel(f"PC{j+1} ({var_explained[j]:.1%})")
                else:
                    ax.set_xticklabels([])
                if j == 0 and i != 0:
                    ax.set_ylabel(f"PC{i+1} ({var_explained[i]:.1%})")
                elif j != 0:
                    ax.set_yticklabels([])

        # Legend at bottom
        fig.legend(
            handles=legend_handles,
            labels=sorted(legend_labels_seen),
            loc="lower center",
            ncol=min(len(legend_labels_seen), 7),
            bbox_to_anchor=(0.5, -0.02),
            fontsize=9,
            markerscale=3,
            frameon=True,
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.08)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info(f"Saved PCA corner plot to {out_path}")

    # 1. By scale
    _corner_plot(
        color_fn=lambda g: scale_colors.get(g, "gray"),
        label_fn=lambda lbl: lbl,
        marker_fn=lambda g: "x" if g in unseen_set else "o",
        out_path=output_dir / "pca_by_scale.pdf",
    )

    # 2. Seen vs unseen (heldout only)
    if unseen_scales:
        _corner_plot(
            color_fn=lambda g: "crimson" if g == "held-out" else "steelblue",
            label_fn=lambda lbl: "held-out" if lbl in unseen_set else "seen",
            marker_fn=lambda g: "x" if g == "held-out" else "o",
            out_path=output_dir / "pca_seen_vs_unseen.pdf",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze scale invariance of pretrained DINO/iBOT representations"
    )
    parser.add_argument(
        "--config", "-c", required=False, default=None,
        help="Path to pretrain config YAML",
    )
    parser.add_argument(
        "--load-epoch", default="best",
        help="Checkpoint epoch to load (int or 'best')",
    )
    parser.add_argument(
        "--num-jets", type=int, default=10000,
        help="Number of jets to analyze (default: 10000)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory (default: experiments/<JOBNAME>/scale_invariance)",
    )
    parser.add_argument(
        "--split", default="val",
        help="Which data split to use (default: val)",
    )
    parser.add_argument(
        "--cat-mean-pool", action="store_true",
        help="Concatenate mean-pooled features with CLS token",
    )
    parser.add_argument(
        "--pooling", default=None, choices=["cls", "mean"],
        help="Override pooling: 'cls' (default, use model's CLS token), "
             "'mean' (mean of particle tokens, excluding CLS/registers)",
    )
    parser.add_argument(
        "--heldout", default=None,
        choices=list(HELDOUT_CONFIGS.keys()),
        help="Run heldout interpolation analysis (specify which heldout config)",
    )
    parser.add_argument(
        "--plot-only", default=None,
        help="Skip inference; regenerate plots from saved data at this directory "
             "(must contain scale_embeddings.pt or cosine_similarity_stats.json)",
    )
    parser.add_argument(
        "--log-file", default=None,
        help="Log file path",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()

    configure_logger(LOGGER, log_file=args.log_file, log_level=args.log_level)

    # --plot-only mode: regenerate plots from saved data, no inference needed
    if args.plot_only is None and args.config is None:
        parser.error("--config is required when not using --plot-only")

    if args.plot_only:
        plot_dir = Path(args.plot_only)
        LOGGER.info(f"Plot-only mode: loading from {plot_dir}")

        embeddings_path = plot_dir / "scale_embeddings.pt"
        stats_path = plot_dir / "cosine_similarity_stats.json"

        if embeddings_path.exists():
            LOGGER.info(f"Loading embeddings from {embeddings_path}")
            embeddings = torch.load(embeddings_path, map_location="cpu")
            metrics = compute_invariance_metrics(embeddings)
        elif stats_path.exists():
            LOGGER.info(f"No embeddings found; loading stats from {stats_path}")
            with open(stats_path) as f:
                stats_json = json.load(f)
            # Reconstruct metrics dict from JSON for heatmap plotting
            scales = stats_json["scales"]
            pairwise = {}
            for key, val in stats_json["pairwise_cosine"].items():
                s1, s2 = key.split("_vs_")
                pairwise[(s1, s2)] = val
            metrics = {"pairwise_cosine": pairwise, "scale_names": scales}
            # Can only plot heatmap from JSON (no per-jet histograms without embeddings)
            plot_cosine_matrix(metrics, plot_dir / "cosine_similarity_matrix.pdf")
            LOGGER.info("Regenerated cosine_similarity_matrix.pdf from JSON (no histograms without embeddings)")
            sys.exit(0)
        else:
            LOGGER.error(f"Neither {embeddings_path} nor {stats_path} found")
            sys.exit(1)

        # Auto-detect heldout config from directory name
        unseen = None
        dir_name = plot_dir.name if plot_dir.name not in ("cls", "mean") else plot_dir.parent.name
        for heldout_key, heldout_cfg in HELDOUT_CONFIGS.items():
            if heldout_key.replace("-", "") in dir_name.replace("-", ""):
                unseen = heldout_cfg["unseen"]
                LOGGER.info(f"Auto-detected heldout config '{heldout_key}': unseen={unseen}")
                break

        # Regenerate all plots from embeddings
        plot_cosine_matrix(metrics, plot_dir / "cosine_similarity_matrix.pdf",
                           unseen_scales=unseen)
        plot_per_jet_histogram(metrics, plot_dir / "per_jet_variance.pdf",
                               unseen_scales=unseen)
        plot_pca_from_embeddings(embeddings, plot_dir, unseen_scales=unseen)
        LOGGER.info("Plot-only mode complete.")
        sys.exit(0)

    # Load config
    import yaml
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Resolve PROJECT_ROOT placeholders in all string values
    def resolve_placeholders(obj, config, epoch_num=None):
        if isinstance(obj, str):
            return process_placeholder(s=obj, config=config, epoch_num=epoch_num)
        elif isinstance(obj, dict):
            return {k: resolve_placeholders(v, config, epoch_num) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_placeholders(v, config, epoch_num) for v in obj]
        return obj
    config = resolve_placeholders(config, config)

    # Set load epoch
    load_epoch = args.load_epoch
    if load_epoch != "best":
        load_epoch = int(load_epoch)
    config["inference"] = config.get("inference", {})
    config["inference"]["load_epoch"] = load_epoch

    # Device
    device_str = config.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    # BF16
    want_bf16 = config.get("use_bf16", False)
    use_bf16 = want_bf16 and check_bf16_support(device)

    # Load model
    LOGGER.info(f"Loading teacher backbone from epoch: {load_epoch}")
    model = load_teacher_backbone(config, device)
    model = model.to(device)

    # Output directory
    job_name = config.get("name", "unknown")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        pool_suffix = f"_{args.pooling}" if args.pooling and args.pooling != "cls" else ""
        output_dir = Path(f"experiments/{job_name}/scale_invariance{pool_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Output directory: {output_dir}")

    # Load data — use the training dataloader config (has clustered views)
    train_dl_config = config["training"]["dataloader"]
    split_config = train_dl_config.get(args.split, train_dl_config.get("train"))
    # Override batch size for analysis
    if "kwargs" in split_config:
        split_config["kwargs"]["batch_size"] = min(
            split_config["kwargs"].get("batch_size", 500), 500
        )

    # Disable shuffling for reproducible analysis
    config.setdefault("shuffle", {})["training"] = False

    dataloader, _ = get_dataloader_and_config(
        config=config,
        mode="training",
        split=args.split,
    )

    # Extract embeddings at each scale
    LOGGER.info(f"Extracting embeddings for {args.num_jets} jets across {len(ALL_SCALES)} scales")
    pooling_override = args.pooling if args.pooling != "cls" else None
    embeddings = extract_scale_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        num_jets=args.num_jets,
        use_bf16=use_bf16,
        cat_mean_pool=args.cat_mean_pool,
        pooling_override=pooling_override,
    )

    available_scales = [s for s in ALL_SCALES if s in embeddings]
    n_jets = embeddings[available_scales[0]].shape[0]
    d_model = embeddings[available_scales[0]].shape[1]
    LOGGER.info(
        f"Collected {n_jets} jets, {len(available_scales)} scales, d_model={d_model}"
    )

    # Compute invariance metrics
    metrics = compute_invariance_metrics(embeddings)

    # Print summary
    LOGGER.info("=" * 60)
    LOGGER.info("PAIRWISE COSINE SIMILARITY (mean ± std)")
    LOGGER.info("=" * 60)
    for (s1, s2), stats in sorted(metrics["pairwise_cosine"].items()):
        LOGGER.info(f"  {s1:>10s} vs {s2:<10s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    mean_cos = float(metrics["per_jet_mean_cosine"].mean())
    mean_var = float(metrics["per_jet_var"].mean())
    LOGGER.info(f"\nPer-jet mean cosine to centroid: {mean_cos:.4f}")
    LOGGER.info(f"Per-jet mean variance across scales: {mean_var:.6f}")

    # Label-stratified analysis
    label_metrics = {}
    if "labels" in embeddings:
        label_metrics = compute_label_stratified_metrics(embeddings, embeddings["labels"])
        LOGGER.info("\nPer-class invariance:")
        for cls_name, cls_stats in label_metrics.items():
            LOGGER.info(
                f"  {cls_name}: n={cls_stats['n_jets']}, "
                f"cos={cls_stats['mean_cosine_to_centroid']:.4f}, "
                f"var={cls_stats['mean_variance']:.6f}"
            )

    # Heldout interpolation analysis
    heldout_metrics = {}
    if args.heldout:
        heldout_cfg = HELDOUT_CONFIGS[args.heldout]
        LOGGER.info(f"\n{'=' * 60}")
        LOGGER.info(f"HELDOUT INTERPOLATION ANALYSIS ({args.heldout})")
        LOGGER.info(f"  Seen scales (train): {heldout_cfg['seen']}")
        LOGGER.info(f"  Unseen scales (val): {heldout_cfg['unseen']}")
        LOGGER.info(f"{'=' * 60}")

        heldout_metrics = compute_heldout_metrics(
            embeddings, heldout_cfg["seen"], heldout_cfg["unseen"]
        )

        if "error" not in heldout_metrics:
            LOGGER.info(f"  Seen↔Seen cosine:   {heldout_metrics['seen_seen_cosine']:.4f}")
            LOGGER.info(f"  Unseen↔Unseen cosine: {heldout_metrics['unseen_unseen_cosine']:.4f}")
            LOGGER.info(f"  Seen↔Unseen cosine: {heldout_metrics['seen_unseen_cosine']:.4f}")
            LOGGER.info(f"  Interpolation gap:  {heldout_metrics['interpolation_gap']:.4f} "
                        f"(smaller = better)")
            LOGGER.info(f"\n  Per unseen scale:")
            for us, us_stats in heldout_metrics["per_unseen"].items():
                LOGGER.info(f"    {us}: mean cos to seen = {us_stats['mean_cosine_to_seen']:.4f} "
                            f"[{us_stats['min_cosine_to_seen']:.4f}, "
                            f"{us_stats['max_cosine_to_seen']:.4f}]")

            # Interpretation
            gap = heldout_metrics["interpolation_gap"]
            if gap < 0.01:
                LOGGER.info(f"\n  ✓ Excellent interpolation (gap={gap:.4f} < 0.01): "
                            f"CLS smoothly extends to unseen scales")
            elif gap < 0.05:
                LOGGER.info(f"\n  ~ Moderate interpolation (gap={gap:.4f}): "
                            f"some degradation at unseen scales")
            else:
                LOGGER.info(f"\n  ✗ Poor interpolation (gap={gap:.4f} > 0.05): "
                            f"CLS does not generalize to unseen scales")
        else:
            LOGGER.warning(f"  {heldout_metrics['error']}")
    else:
        # Auto-detect heldout config from job name
        for heldout_key in HELDOUT_CONFIGS:
            if heldout_key in job_name:
                LOGGER.info(f"\nAuto-detected heldout config: {heldout_key}")
                heldout_cfg = HELDOUT_CONFIGS[heldout_key]
                heldout_metrics = compute_heldout_metrics(
                    embeddings, heldout_cfg["seen"], heldout_cfg["unseen"]
                )
                if "error" not in heldout_metrics:
                    LOGGER.info(f"  Seen↔Seen: {heldout_metrics['seen_seen_cosine']:.4f}")
                    LOGGER.info(f"  Seen↔Unseen: {heldout_metrics['seen_unseen_cosine']:.4f}")
                    LOGGER.info(f"  Gap: {heldout_metrics['interpolation_gap']:.4f}")
                break

    # Save results
    # 1. Raw embeddings
    torch.save(embeddings, output_dir / "scale_embeddings.pt")
    LOGGER.info(f"Saved raw embeddings to {output_dir / 'scale_embeddings.pt'}")

    # 2. Statistics JSON
    stats_json = {
        "config": str(config_path),
        "load_epoch": str(load_epoch),
        "num_jets": n_jets,
        "d_model": d_model,
        "scales": available_scales,
        "cat_mean_pool": args.cat_mean_pool,
        "pooling": args.pooling or "cls",
        "summary": {
            "per_jet_mean_cosine_to_centroid": mean_cos,
            "per_jet_mean_variance": mean_var,
        },
        "pairwise_cosine": {
            f"{s1}_vs_{s2}": stats
            for (s1, s2), stats in metrics["pairwise_cosine"].items()
        },
    }
    if label_metrics:
        stats_json["per_class"] = label_metrics
    if heldout_metrics and "error" not in heldout_metrics:
        stats_json["heldout_interpolation"] = {
            k: v for k, v in heldout_metrics.items()
            if k not in ("per_unseen",)  # per_unseen saved separately for readability
        }
        stats_json["heldout_per_unseen"] = heldout_metrics.get("per_unseen", {})

    with open(output_dir / "cosine_similarity_stats.json", "w") as f:
        json.dump(stats_json, f, indent=2)
    LOGGER.info(f"Saved statistics to {output_dir / 'cosine_similarity_stats.json'}")

    # 3. Plots (no titles — added manually for paper if needed)
    # Detect unseen scales for heldout histogram coloring
    unseen = None
    if args.heldout:
        unseen = HELDOUT_CONFIGS[args.heldout]["unseen"]
    else:
        for heldout_key, heldout_cfg in HELDOUT_CONFIGS.items():
            if heldout_key.replace("-", "") in job_name.replace("-", ""):
                unseen = heldout_cfg["unseen"]
                break
    plot_cosine_matrix(metrics, output_dir / "cosine_similarity_matrix.pdf",
                       unseen_scales=unseen)
    plot_per_jet_histogram(metrics, output_dir / "per_jet_variance.pdf",
                           unseen_scales=unseen)
    plot_pca_from_embeddings(embeddings, output_dir, unseen_scales=unseen)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
