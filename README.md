# RINO: Renormalization Group Invariance with No Labels

Self-supervised representation learning for jets using multi-scale kT clustering views and DINO self-distillation.

## Setup

```bash
conda env create -f environment.yml
conda activate parcel
```

## Data Preparation

See [`configs/data-README.md`](configs/data-README.md) for dataset download and preprocessing instructions.

## Pretraining

```bash
# RINO pretraining (DINO + iBOT on kT-clustered QCD jets)
python dino/dino_train.py -c configs/dino/<config>.yaml

# Multi-GPU with Accelerate
accelerate launch dino/dino_train.py -c configs/dino/<config>.yaml
```

## Finetuning

```bash
# Classification finetuning (LP-FT protocol)
python dino/classification_train.py -c configs/dino/<finetune-config>.yaml
```

## Inference and Evaluation

```bash
# Run inference on test set
python dino/dino_inference.py -c configs/dino/<finetune-config>.yaml

# Per-class and ensemble evaluation
python dino/eval_per_class.py --exp-dir experiments/<exp-dir>/<model>
```

## Baselines

All SSL baselines (MPMv1, MPMv2, OmniJet-alpha, JetCLR, JetCLR-scale) are in `baselines/` with a shared backbone and finetuning protocol. See `baselines/` for implementation details.

## Repository Structure

```
dino/                   # RINO framework (training, inference, models, losses)
baselines/              # SSL baseline implementations
configs/                # All YAML configs (pretraining, finetuning, dataloaders)
environment.yml         # Conda environment specification
test_run.sh             # Quick sanity check
```

