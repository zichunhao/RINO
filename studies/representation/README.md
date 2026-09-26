# Representation studies

Scripts for the representation-space analyses of the paper: the JetNet→JetClass
domain shift of finetuned and pretrained representations, and the cosine
similarity of representations across kT clustering scales (backbone and
projection head, standard and swapped teacher–student roles).

All commands run from the repository root with the `rino` environment. The
scripts reuse the release code in `dino/` (configs, dataloaders, models,
`dino/domain_shift_metrics.py`); figures and tables are drawn only from the
JSON files the compute scripts write.

| Script | Paper item | Inputs | Output |
|---|---|---|---|
| `domain_shift/compute_domain_shift.py` | Table "Domain shift in finetuned representations" (Sec. "Why RINO Generalizes OOD") and App. table "Domain shift in pretrained representations" | per-run `output_test_jetnet_*.pt` / `output_test_jetclass_*.pt` with `rep` and `label` | JSON: W1 and MMD for QCD and top, per run and mean ± std over runs |
| `domain_shift/extract_pretrained_reps.py` | App. table "Domain shift in pretrained representations" | pretrained checkpoint (RINO, JetCLR-scale or a converted baseline `backbone.pt`) | `output_<split>_pretrained-0.pt`, same layout as `dino/dino_inference.py` |
| `domain_shift/summarize_domain_shift.py` | both domain-shift tables | JSONs of `compute_domain_shift.py` | markdown / LaTeX / CSV table; with `--pretrained`, signed change after finetuning |
| `scale_structure/compute_scale_cosines.py` | Fig. "Scale Structure" (App.) and Fig. "Swapped teacher–student roles" (App.) | pretrained checkpoint + clustered JetClass split with subjet views | `cosine_similarity_stats.json` (backbone and, with `--head-space`, head output and bottleneck) |
| `scale_structure/plot_cosine_matrices.py` | both scale-structure figures | `cosine_similarity_stats.json` files | PDF/PNG figure |
| `representation_utils.py` | shared helpers (configs, encoder loading, pooling, dataloaders) | — | — |
| `baseline-backbone.yaml` | architecture of the converted baseline backbones | — | — |

## Encoders

`extract_pretrained_reps.py` and `compute_scale_cosines.py` load a pretrained
encoder with `--model-type`:

| `--model-type` | Checkpoint | `--model-config` | Used for |
|---|---|---|---|
| `dino` | DINO/iBOT pretraining checkpoint; the EMA teacher backbone and teacher DINO head are used | its pretraining config, e.g. `configs/dino/rino.yaml` | RINO and its variants |
| `jetclr` | checkpoint of `dino/jetclr_train.py` | its pretraining config, e.g. `configs/jetclr/3tier-recon.yaml` | JetCLR-scale |
| `backbone` | `{"backbone": state_dict}` from `baselines/scripts/convert_checkpoint.py` | `studies/representation/baseline-backbone.yaml` | MPMv1, MPMv2, JetCLR, OmniJet-α |

For `dino` and `jetclr`, `--load-epoch best` (the default) resolves the
checkpoint from the config's `training.checkpoints_dir`; `--checkpoint <path>`
loads any file instead. Weights are loaded strictly. A converted OmniJet-α
backbone has no continuous input layer (see `convert_checkpoint.py`); pass
`--allow-missing-keys` to keep that layer at its seeded random initialisation,
as the finetuning model does.

`--pooling native` (default) uses each backbone's own pooled output: the CLS
token for RINO and JetCLR-scale, the masked mean over all tokens for the
converted baselines. `--pooling mean` instead averages the particle output
tokens only (CLS and register tokens excluded), as
`dino/diagnostics/scale_invariance_analysis.py --pooling mean`.

## Domain shift (W1 / MMD)

`compute_domain_shift.py` draws a seeded random subset of `--n-per-domain`
jets (default 5,000) per domain and class (QCD: JetNet label 0, JetClass label
0; top: JetNet label 1, JetClass label 8) and computes on the same subsets the
per-dimension 1-Wasserstein distance averaged over dimensions and the RBF-kernel
MMD with γ = 1/d (d = representation dimension). For a single run and seed it
gives the same values as `dino/domain_shift_metrics.py`. Given several run
directories it also reports the mean ± std over runs (the error-bar variant).

### Finetuned representations

Finetune and run inference as in `studies/evaluation/README.md`, with
`test_jetnet` and `test_jetclass` among the inference splits; every run then
has `<exp_dir>/run-<i>/inference/output_test_{jetnet,jetclass}_best-0.pt`.
For each method:

```bash
python studies/representation/domain_shift/compute_domain_shift.py \
    --method RINO --inference-dirs <exp_dir>/run-*/inference \
    --output experiments/studies/domain_shift/finetuned/RINO.json
```

(`--inference-dirs <exp_dir>/run-1/inference` gives a single-run table.) Then

```bash
python studies/representation/domain_shift/summarize_domain_shift.py \
    --finetuned experiments/studies/domain_shift/finetuned/*.json \
    --order Supervised JetCLR MPMv1 MPMv2 OmniJet-alpha JetCLR-scale RINO \
    --format latex
```

### Pretrained representations

Extract the representations of every pretrained model on the JetNet and
JetClass test splits. The split definitions (`inference.dataloader.test_jetnet`
and `test_jetclass`) are taken from `--data-config`, which defaults to
`--model-config`; any RINO pretraining config defines both.

```bash
# RINO
python studies/representation/domain_shift/extract_pretrained_reps.py \
    --model-type dino --model-config configs/dino/rino.yaml \
    --output-dir experiments/studies/domain_shift/reps/RINO

# JetCLR-scale
python studies/representation/domain_shift/extract_pretrained_reps.py \
    --model-type jetclr --model-config configs/jetclr/3tier-recon.yaml \
    --data-config configs/dino/rino.yaml \
    --output-dir experiments/studies/domain_shift/reps/JetCLR-scale

# converted baselines (MPMv1, MPMv2, JetCLR; add --allow-missing-keys for OmniJet-alpha)
python studies/representation/domain_shift/extract_pretrained_reps.py \
    --model-type backbone \
    --model-config studies/representation/baseline-backbone.yaml \
    --checkpoint <path/to/mpmv2/backbone.pt> --data-config configs/dino/rino.yaml \
    --output-dir experiments/studies/domain_shift/reps/MPMv2
```

Then compute the metrics on these files and tabulate them next to the
finetuned results:

```bash
python studies/representation/domain_shift/compute_domain_shift.py \
    --method RINO --inference-dirs experiments/studies/domain_shift/reps/RINO \
    --source-file output_test_jetnet_pretrained-0.pt \
    --target-file output_test_jetclass_pretrained-0.pt \
    --output experiments/studies/domain_shift/pretrained/RINO.json

python studies/representation/domain_shift/summarize_domain_shift.py \
    --pretrained experiments/studies/domain_shift/pretrained/*.json \
    --finetuned experiments/studies/domain_shift/finetuned/*.json \
    --order JetCLR OmniJet-alpha MPMv1 MPMv2 JetCLR-scale RINO --format latex
```

Each cell shows the pretrained value and the change after finetuning
(finetuned − pretrained); methods are matched by `--method` name.

## Scale structure

`compute_scale_cosines.py` encodes the same jets at every scale: the
unclustered constituents (`ALL`, "Uncl." in the figures) and the exclusive-kT
subjet views `subjet2` … `subjet16` stored by the clustered JetClass dataset
(see `configs/data-README.md`). The data come from `training.dataloader.<split>`
of `--data-config` (default: `--model-config`; `--split val`), e.g. the
clustered QCD validation split of `configs/dino/rino.yaml`. For every pair of
scales it summarises the per-jet cosine similarity over `--num-jets` jets
(default 10,000); the figures show the mean. With `--head-space` it repeats
this after the projection head, in the head output (2048-d for RINO) and in the
L2-normalised bottleneck before the head's last layer.

```bash
# RINO backbone and projection head
python studies/representation/scale_structure/compute_scale_cosines.py \
    --model-type dino --model-config configs/dino/rino.yaml --head-space \
    --output-dir experiments/studies/scale_structure/RINO

# MPMv2 backbone (same jets)
python studies/representation/scale_structure/compute_scale_cosines.py \
    --model-type backbone \
    --model-config studies/representation/baseline-backbone.yaml \
    --checkpoint <path/to/mpmv2/backbone.pt> --data-config configs/dino/rino.yaml \
    --output-dir experiments/studies/scale_structure/MPMv2

# JetCLR-scale backbone
python studies/representation/scale_structure/compute_scale_cosines.py \
    --model-type jetclr --model-config configs/jetclr/3tier-recon.yaml \
    --data-config configs/dino/rino.yaml \
    --output-dir experiments/studies/scale_structure/JetCLR-scale
```

The 2×2 figure (MPMv2, JetCLR-scale, RINO backbone, RINO backbone + projection
head). `teacher=` colours the tick labels by role (teacher red, student blue):

```bash
python studies/representation/scale_structure/plot_cosine_matrices.py \
    --panel stats=experiments/studies/scale_structure/MPMv2/cosine_similarity_stats.json \
            title="MPMv2 backbone" subtitle="Scale-ignorant" \
    --panel stats=experiments/studies/scale_structure/JetCLR-scale/cosine_similarity_stats.json \
            title="JetCLR-scale backbone" subtitle="Scale-invariant" \
    --panel stats=experiments/studies/scale_structure/RINO/cosine_similarity_stats.json \
            teacher=6,8,16 title="RINO backbone" subtitle="Scale-aware" \
    --panel stats=experiments/studies/scale_structure/RINO/cosine_similarity_stats.json \
            space=head_output teacher=6,8,16 \
            title="RINO backbone + projection head" subtitle="Scale-invariant" \
    --ncols 2 --colorbar bottom --output experiments/studies/figures/scale_structure.pdf
```

### Swapped teacher–student roles

Pretrain the swapped-role variant (teacher {2, 3, 4}; student additionally
{6, 8, 16, uncl.}) and the standard-assignment variant (teacher {6, 8, 16}) of
the pre-tuning recipe (`../evaluation/README.md`, "Ablations"), saved as
`configs/dino/standard.yaml` and `configs/dino/swapped.yaml` with distinct
`name`s. Compute both matrices and plot them side by side.

```bash
for cfg in standard swapped; do
  python studies/representation/scale_structure/compute_scale_cosines.py \
      --model-type dino --model-config configs/dino/$cfg.yaml \
      --data-config configs/dino/rino.yaml \
      --output-dir experiments/studies/scale_structure/$cfg
done

python studies/representation/scale_structure/plot_cosine_matrices.py \
    --panel stats=experiments/studies/scale_structure/standard/cosine_similarity_stats.json \
            teacher=6,8,16 title='Standard: teacher $\{6,8,16\}$' \
            subtitle='student $\{2,3,4,\mathrm{uncl.}\}$' \
    --panel stats=experiments/studies/scale_structure/swapped/cosine_similarity_stats.json \
            teacher=2,3,4 title='Swapped: teacher $\{2,3,4\}$' \
            subtitle='student $\{6,8,16,\mathrm{uncl.}\}$' \
    --ncols 2 --colorbar right --panel-size 4.5 --fontsize 7 \
    --output experiments/studies/figures/swapped_cosine.pdf
```

`--order` sets the scale order on both axes (default N = 2, 3, 4, 6, 8, 16,
Uncl.). The stats files use the schema of
`dino/diagnostics/scale_invariance_analysis.py`, so its outputs can be plotted
too; `--save-embeddings` additionally stores the per-scale representations.
