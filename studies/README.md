# Paper studies

This index maps each table and figure of the paper to the scripts that produce
it. The commands are documented in three subdirectories:

| Directory | Contents |
|---|---|
| [`evaluation/`](evaluation/README.md) | finetuning and probe results: top-tagging comparison, data efficiency, cross-topology and TopTagging→JetClass transfer, regularization sweep, MLP vs. linear head, ablations |
| [`representation/`](representation/README.md) | domain shift of finetuned and pretrained representations, cosine similarity across kT scales |
| [`physics/`](physics/README.md) | physical scales of the kT views, constituent multiplicity, positional-encoding collisions, scale-factor proxy |

All commands run from the repository root in the `parcel` environment and
write to `experiments/` by default. Most tables share one pipeline:

1. Pretrain RINO or a baseline, and convert the baseline checkpoints. See
   "Pretraining RINO" and "Baselines" in the [top-level README](../README.md).
2. Generate the finetuning configs with `configs/gen_le_configs.py`.
3. For each seed, finetune with `dino/classification_train.py` and evaluate
   with `dino/dino_inference.py`.
4. Harvest the outputs and tabulate them with the scripts in `evaluation/`.

## Main text

| Paper item | Scripts | Commands |
|---|---|---|
| Figure 1: parton shower and RINO pipeline | Schematic; no script | — |
| Table 1: OOD top tagging, JetNet → JetClass (linear and MLP heads, k-NN, linear probe) | Finetuning: steps 2–3 above. Probes: `dino/dino_inference.py` (`inference.acc_tests`) for RINO and JetCLR-scale, `baselines/scripts/backbone_probe.py` for the converted baselines. Tables: `evaluation/harvest_finetune.py`, `evaluation/harvest_probes.py`, `evaluation/make_table.py` | `evaluation/README.md`, "Table 1" |
| Figure 2: OOD accuracy vs. JetNet label fraction | `configs/gen_le_configs.py --fractions ...`, `evaluation/harvest_finetune.py`, `evaluation/plot_data_efficiency.py` | `evaluation/README.md`, "Data efficiency" |
| Table 2: cross-topology transfer (linear head) | `evaluation/harvest_finetune.py --subsets H4q_vs_QCD Hbb_vs_QCD`, `evaluation/make_table.py` | `evaluation/README.md`, "Cross-topology transfer" |
| Table 3: contribution of DINO self-distillation | Pretraining: `configs/dino/ibotonly-g6l2-pbin.yaml`, `configs/dino/ibot-g6l2-pbin.yaml`, `configs/dino/rino.yaml`; then the Table 1 pipeline and `evaluation/make_table.py --gain-ref previous` | `evaluation/README.md`, "Ablations" |
| Figure 3: factorization of the OOD gain | Diagram; its values come from Tables 1 and 3 | — |
| Table 4: domain shift in finetuned representations | `representation/domain_shift/compute_domain_shift.py`, `representation/domain_shift/summarize_domain_shift.py` | `representation/README.md`, "Finetuned representations" |

The C/A and anti-kT results quoted in the text reuse `configs/dino/ibot-g6l2-pbin.yaml`.
Recluster the views with `dino/preprocess/jetclass/cluster.py --algorithm
cambridge` or `--algorithm antikt` (see `evaluation/README.md`, "Ablations").

## Appendix

| Paper item | Scripts | Commands |
|---|---|---|
| Pretraining hyperparameters | `configs/dino/rino.yaml` | top-level README, "Pretraining RINO" |
| Per-particle input features and standardization | `_NORM` in `dino/dataloader/jetclass/processors.py` (used by `configs/dataloaders/*/kinematics*.yaml`) | — |
| Positional-encoding bin-collision rates, `r_scale` and `r_max` | `physics/pe_collision.py` | `physics/README.md` |
| Constituent-multiplicity CDF and dataset statistics | `physics/nparticles_count.py` → `physics/nparticles_cdf.py` | `physics/README.md` |
| Physical scales of the kT views | `physics/kt_scales.py` | `physics/README.md` |
| Finetuning hyperparameters | `configs/gen_le_configs.py` (generated finetuning configs) | top-level README, "Finetuning and OOD evaluation" |
| Robustness to finetuning regularization | `evaluation/finetune_reg_sweep.py generate`, `evaluation/harvest_finetune.py`, `evaluation/finetune_reg_sweep.py summarize` | `evaluation/README.md`, "Robustness to finetuning regularization" |
| Progressive construction of the recipe; independent design comparisons | Pretraining configs in `configs/dino/` (row-by-row mapping in `evaluation/README.md`); then the Table 1 pipeline and `evaluation/make_table.py --gain-ref` | `evaluation/README.md`, "Ablations" |
| Cosine similarity across kT scales (MPMv2, JetCLR-scale, RINO backbone, RINO backbone + head) | `representation/scale_structure/compute_scale_cosines.py`, `representation/scale_structure/plot_cosine_matrices.py` | `representation/README.md`, "Scale structure" |
| Swapped teacher–student roles | The same two scripts on `configs/dino/ibot-g6l2-pbin.yaml` and `configs/dino/ibot-g2l6-pbin.yaml` | `representation/README.md`, "Swapped teacher–student roles" |
| Cross-topology transfer with the MLP head | `evaluation/harvest_finetune.py`, `evaluation/make_table.py` | `evaluation/README.md`, "Cross-topology transfer" |
| TopTagging → JetClass accuracy and cross-topology AUC | `configs/gen_le_configs.py --task tt`, `evaluation/harvest_finetune.py`, `evaluation/make_table.py` | `evaluation/README.md`, "TopTagging → JetClass transfer" |
| Scale-factor proxy (TopTagging → JetClass) | `physics/scale_factor_proxy.py compute`, then `table` | `physics/README.md` |
| Path to experimental data (deployment) | Schematic; no script | — |
| Full data-efficiency curves | `evaluation/plot_data_efficiency.py` | `evaluation/README.md`, "Data efficiency" |
| MLP vs. linear head (Welch *t*-test) | `evaluation/mlp_vs_linear_welch.py` | `evaluation/README.md`, "MLP vs. linear head" |
| Domain shift in pretrained representations | `representation/domain_shift/extract_pretrained_reps.py`, `representation/domain_shift/compute_domain_shift.py`, `representation/domain_shift/summarize_domain_shift.py --pretrained` | `representation/README.md`, "Pretrained representations" |
| Baseline implementation summary | Baseline configs and conversion: `baselines/`, `baselines/scripts/convert_checkpoint.py` | top-level README, "Baselines" |
| Pretraining cost | Wall-clock time of the training runs; no script | — |
