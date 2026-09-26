# Evaluation studies

Scripts that turn finetuning and probing outputs into the evaluation tables and
figures of the paper: the top-tagging comparison (Table 1), data efficiency,
cross-topology transfer, TopTagging→JetClass transfer, robustness to finetuning
regularization, the MLP-vs-linear test and the design ablations. The
TopTagging→JetClass scale-factor proxy is computed by
`studies/physics/scale_factor_proxy.py`.

All commands run from the repository root with the `parcel` environment. The
scripts only read inference outputs; training and inference use the standard
entry points:

```bash
for i in $(seq 1 10); do   # 10 finetuning seeds (6 for TopTagging)
  python dino/classification_train.py -c <path/to/finetune-config.yaml> --run-index $i
  python dino/dino_inference.py      -c <path/to/finetune-config.yaml> --run-index $i
done
```

Each run writes `<exp_dir>/run-<i>/inference/output_<split>_best-0.pt`
(logits, labels, representations) and, for JetClass splits,
`metrics_<split>_best.json` with the metrics of the config's `eval_subsets`.

## Workflow

1. **Harvest** per-seed metrics into a CSV (`harvest_finetune.py`) and frozen-probe
   results into a summary CSV (`harvest_probes.py`).
2. **Tabulate** with `make_table.py`, or run the study-specific scripts
   (`plot_data_efficiency.py`, `mlp_vs_linear_welch.py`,
   `finetune_reg_sweep.py summarize`).

Experiments are listed in a YAML manifest (or with repeated `--exp key=value,...`).
Each entry locates its outputs with `config` (a finetuning config; its
`inference.output_dir` is resolved like in `dino/dino_inference.py`) or `dir` (an
experiment directory holding `run-<i>/inference/`). Every other key becomes a
column of the CSV and can be used to group, filter and pivot:

```yaml
# manifest.yaml
- {method: RINO,       head: mlp,    config: <path/to/rino-finetune-mlp.yaml>}
- {method: RINO,       head: linear, config: <path/to/rino-finetune-linear.yaml>}
- {method: Supervised, head: mlp,    dir: experiments/<finetune-dir>/<name>}
```

Metrics are the accuracy at a 0.5 threshold on `sigmoid(logit)` and the ROC AUC,
restricted to one JetClass signal class against QCD (label 0): `Tbqq_vs_QCD`
(label 8, the OOD top-tagging task), `H4q_vs_QCD` (4), `Hbb_vs_QCD` (1), or any
`NAME=POS:NEG`; `all` scores every non-zero label against 0 (the
in-distribution JetNet or TopTagging task). Subsets absent from the metrics JSON
are recomputed from the saved logits, which requires the evaluated JetClass
files to contain that class (e.g. `HToWW4Q_*.root`, `HToBB_*.root`). Standard
deviations over seeds use `ddof=1` by default (`--ddof` to change).

## Scripts

| Script | Paper item | Inputs | Output |
|---|---|---|---|
| `harvest_finetune.py` | Table 1 (linear/MLP heads), cross-topology tables (linear and MLP heads), TopTagging→JetClass accuracy and cross-topology tables, data-efficiency points, regularization cells, ablation OOD accuracy | manifest of finetuned experiments; `--split`, `--subsets` | per-seed CSV |
| `harvest_probes.py` | Table 1 k-NN and linear-probe columns; k-NN/LP columns of the ablation tables | probe JSON from `baselines/scripts/backbone_probe.py` or `dino/dino_inference.py` (`inference.acc_tests`) | summary CSV |
| `make_table.py` | all mean ± std tables above | per-seed and/or summary CSVs | markdown / LaTeX / CSV table |
| `plot_data_efficiency.py` | Data-efficiency figure (main text) and full data-efficiency curves (appendix) | per-seed CSV with `method` and `fraction` tags | PDF/PNG figure (+ optional CSV of plotted points) |
| `mlp_vs_linear_welch.py` | MLP vs. linear head table (Welch *t*) | per-seed CSV with `head: mlp/linear`, or a summary CSV with `mean,std,n` | table of *t*, Welch–Satterthwaite df, two-sided *p* |
| `finetune_reg_sweep.py` | Robustness-to-finetuning-regularization table | `generate`: default finetuning configs; `summarize`: per-seed CSV | sweep configs + manifest; Default/Worst/Best table |
| `common.py` | shared helpers (manifests, subsets, metrics, tables) | — | — |

## Table 1: top-tagging comparison

Finetune every method on JetNet with a linear and an MLP head (10 seeds each),
evaluating `test_jetclass` on `TTBar_*` and `ZJetsToNuNu_*` (plus `HToWW4Q_*`
and `HToBB_*` for the cross-topology tables). Then:

```bash
python studies/evaluation/harvest_finetune.py --manifest <path/to/table1.yaml> \
    --subsets Tbqq_vs_QCD H4q_vs_QCD Hbb_vs_QCD \
    --output experiments/studies/jetnet_finetune_per_seed.csv
```

The frozen-backbone k-NN (k=20) and linear-probe columns come from one of two
probes, depending on how the backbone was trained:

* backbones exported as `backbone.pt` (the SSL baselines):
  ```bash
  python baselines/scripts/backbone_probe.py --backbone-path <path/to/backbone.pt> \
      --dataloader-config configs/dataloaders/jetclass-raw/kinematics.yaml \
      --pooling mean --output experiments/studies/probes/<method>.json
  ```
  (standardised embeddings, cosine k-NN and logistic regression over 5 random
  80/20 splits of a 200k-jet subsample);
* checkpoints trained with this repository (RINO, its ablations, JetCLR-scale):
  `python dino/dino_inference.py -c <path/to/pretrain-config.yaml>` with the
  `inference.acc_tests` block of the pretraining config (k-NN on the pooled CLS
  token; linear probe on the CLS token concatenated with the mean over particles),
  which writes `metrics_test_jetclass_best.json` to the config's inference directory.

```bash
python studies/evaluation/harvest_probes.py \
    --exp method=MPMv2,path=experiments/studies/probes/mpmv2.json \
    --exp method=RINO,config=<path/to/rino-pretrain-config.yaml> \
    --output experiments/studies/probes.csv

python studies/evaluation/make_table.py \
    --csv experiments/studies/jetnet_finetune_per_seed.csv experiments/studies/probes.csv \
    --rows method --cols head --col-order knn linear_probe linear mlp \
    --row-order Supervised JetCLR OmniJet-alpha MPMv1 MPMv2 JetCLR-scale RINO \
    --subset Tbqq_vs_QCD --metric acc --highlight --format latex
```

The same per-seed CSV provides the MLP-head values used in the MLP-vs-linear
table, the Default column of the regularization table and the last row of the
ablation table; the numbers in the factorization diagram are those of Table 1
and the ablation table.

Per-class and seed-ensemble metrics of one experiment are printed by
`python dino/eval_per_class.py --exp-dir <exp_dir>`.

## MLP vs. linear head

```bash
python studies/evaluation/mlp_vs_linear_welch.py \
    --csv experiments/studies/jetnet_finetune_per_seed.csv --subset Tbqq_vs_QCD \
    --order RINO Supervised MPMv1 MPMv2 JetCLR JetCLR-scale OmniJet-alpha
```

`t = (μ_MLP − μ_linear) / sqrt(s²_MLP/n + s²_linear/n)` with sample standard
deviations; df from the Welch–Satterthwaite approximation. A summary CSV
(`method,head,mean,std,n`, e.g. from `make_table.py --summary-out`) can be
given instead of per-seed values.

## Cross-topology transfer

Evaluate the JetNet-finetuned models on `test_jetclass` files that also contain
`HToWW4Q_*` (label 4) and `HToBB_*` (label 1), ideally listing
`H4q_vs_QCD: {positive: [4], negative: [0]}` and
`Hbb_vs_QCD: {positive: [1], negative: [0]}` under `eval_subsets`. With the CSV
harvested above:

```bash
# linear head: ACC and AUC per unseen topology
python studies/evaluation/make_table.py --csv experiments/studies/jetnet_finetune_per_seed.csv \
    --where head=linear --rows method --cols subset metric \
    --subset H4q_vs_QCD Hbb_vs_QCD --metric acc auc --highlight --format latex

# MLP head: ACC including the finetuning topology
python studies/evaluation/make_table.py --csv experiments/studies/jetnet_finetune_per_seed.csv \
    --where head=mlp --rows method --cols subset \
    --col-order Tbqq_vs_QCD H4q_vs_QCD Hbb_vs_QCD --metric acc --highlight --format latex
```

## Data efficiency

Finetune with JetNet label fractions {0.1, 1, 10, 50}% (`train_fraction` in the
training dataloader; configs from `configs/gen_le_configs.py`) and 10 seeds; the
100% points are the Table 1 runs. Tag each manifest entry with `fraction`
(0.001 … 1.0):

```bash
python studies/evaluation/harvest_finetune.py --manifest <path/to/label_efficiency.yaml> \
    --output experiments/studies/label_eff_per_seed.csv

# main-text figure
python studies/evaluation/plot_data_efficiency.py --csv experiments/studies/label_eff_per_seed.csv \
    --where head=mlp --methods RINO Supervised MPMv1 MPMv2 JetCLR JetCLR-scale \
    --reference Supervised --annotate RINO@0.01 Supervised@1.0 \
    --output experiments/studies/data_efficiency.pdf

# appendix figure (all methods)
python studies/evaluation/plot_data_efficiency.py --csv experiments/studies/label_eff_per_seed.csv \
    --where head=mlp --reference Supervised --figsize 6.0 4.5 \
    --output experiments/studies/data_efficiency_full.pdf
```

Extra checks use the same tools: a batch-size variant is one more manifest entry
with its own tag (e.g. `batch_size: 256`) tabulated with `make_table.py`; the
held-out in-distribution accuracy comes from the JetNet test split:

```bash
python studies/evaluation/harvest_finetune.py --manifest <path/to/label_efficiency.yaml> \
    --split test_jetnet --subsets all --output experiments/studies/label_eff_jetnet.csv
python studies/evaluation/make_table.py --csv experiments/studies/label_eff_jetnet.csv \
    --rows method --cols fraction --subset all --metric acc
```

## TopTagging → JetClass transfer

Finetune on TopTagging (6 seeds, BCE positive weight 1.0) with inference on
`test_toptagging` and `test_jetclass` (including `HToWW4Q_*` for the
cross-topology column):

```bash
python studies/evaluation/harvest_finetune.py --manifest <path/to/toptagging.yaml> \
    --subsets Tbqq_vs_QCD H4q_vs_QCD --output experiments/studies/toptagging_per_seed.csv

# accuracy with MLP and linear heads
python studies/evaluation/make_table.py --csv experiments/studies/toptagging_per_seed.csv \
    --rows method --cols head --col-order mlp linear --subset Tbqq_vs_QCD --metric acc --highlight

# cross-topology AUC (linear head)
python studies/evaluation/make_table.py --csv experiments/studies/toptagging_per_seed.csv \
    --where head=linear --rows method --cols subset --subset Tbqq_vs_QCD H4q_vs_QCD --metric auc --highlight
```

The scale-factor proxy of the same MLP-head runs (source TopTagging, target
JetClass) is computed by `studies/physics/scale_factor_proxy.py compute
--method <NAME>=<exp_dir> ...`.

## Robustness to finetuning regularization

```bash
python studies/evaluation/finetune_reg_sweep.py generate \
    --template Supervised=<path/to/supervised-finetune.yaml> \
    --template MPMv1=<path/to/mpmv1-finetune.yaml> \
    --template MPMv2=<path/to/mpmv2-finetune.yaml> \
    --template RINO=<path/to/rino-finetune.yaml> \
    --out-dir configs/finetune-reg
# finetune + infer each generated config for 10 seeds, then
python studies/evaluation/harvest_finetune.py --manifest configs/finetune-reg/manifest.yaml \
    --output experiments/studies/finetune_reg_per_seed.csv
python studies/evaluation/finetune_reg_sweep.py summarize \
    --csv experiments/studies/finetune_reg_per_seed.csv \
    --order Supervised MPMv1 MPMv2 RINO --highlight --format latex
```

The five (weight decay, dropout) cells are (0.01, 0.1), (0.05, 0.1), (0.10, 0.1),
(0.05, 0.3) and (0.05, 0.5); only the AdamW weight decay, the head dropouts and
the job name change. The default cell (0.05, 0.1) is the template itself, so the
manifest reuses its runs; `--rerun-default` writes a separate config for it, and
`--jetclass-patterns` replaces the JetClass test files of the generated configs.
Worst and Best are the lowest and highest cell means; Best is selected on the
target (JetClass) accuracy.

## Ablations

Each ablation row is one pretraining run followed by the Table 1 pipeline (MLP
head, 10 seeds, plus the k-NN/LP probes of `dino/dino_inference.py`). Tag the
manifest entries with the row (e.g. `variant: ibot-only`) and tabulate with
`make_table.py --rows variant --cols head`; `--gain-ref previous` (progressive
table) or `--gain-ref first` (per-axis blocks, selected with `--where`) adds the
relative change of the OOD accuracy.

| Table row | Pretraining config |
|---|---|
| DINO only, mixed views, no PE | `configs/dino/dino-mixed-pbin.yaml` with the backbone `pos_encoding_kwargs` removed |
| + iBOT (mixed views, no PE) | `configs/dino/ibot-mixed-nope.yaml` |
| Mixed views, pT-rank PE | `configs/dino/ibot-mixed-ptrank.yaml` |
| Mixed views, polar-binned PE (mixed-scale view assignment) | `configs/dino/ibot-mixed-pbin.yaml` |
| kT views, pT-rank PE | `configs/dino/ibot-g6l2-ptrank.yaml` |
| kT views, polar-binned PE (RINO before teacher tuning) | `configs/dino/ibot-g6l2-pbin.yaml` |
| Swapped roles: teacher {2,3,4}, student {6,8,16,uncl.} | `configs/dino/ibot-g2l6-pbin.yaml` |
| C/A clustering | `configs/dino/ibot-g6l2-pbin.yaml` on views clustered with `--algorithm cambridge` |
| Anti-kT clustering | `configs/dino/ibot-g6l2-pbin.yaml` on views clustered with `--algorithm antikt` |
| iBOT only | `configs/dino/ibotonly-g6l2-pbin.yaml` |
| + teacher HP tuning (RINO, production) | `configs/dino/rino.yaml` |

The C/A and anti-kT variants differ from the kT run only in the clustered
training data: rerun `dino/preprocess/jetclass/cluster.py` with the given
`--algorithm` and point the config's `training.dataloader` at a dataloader
config for those files.
