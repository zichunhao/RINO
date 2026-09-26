# Physics and data studies

Scripts behind the physics and dataset analyses in the appendix: the physical
scales of the kT views, the constituent-multiplicity statistics, the
positional-encoding bin-collision rates (which also fix `r_scale` and `r_max`),
and the TopTagging-to-JetClass scale-factor proxy.

Run every command from the repository root. Paths may use the `PROJECT_ROOT`
placeholder, as in the configs. Inputs follow the data layout of
[`configs/data-README.md`](../../configs/data-README.md). JetClass QCD jets are
the `ZJetsToNuNu_*.root` files. By default, outputs go to
`experiments/studies/physics/`.

| Script | Paper item | Inputs | Output |
|---|---|---|---|
| `kt_scales.py` | Table `tab:kt-scales` (App. "Physical Scales of the kT Views") | JetClass QCD ROOT (`part_px/py/pz/energy`) | `kt_scales.json` |
| `nparticles_count.py` | Figure `fig:nparticles-cdf` and the multiplicity statistics in App. "Dataset Details" (harvester) | JetClass QCD ROOT, train/val/test | `nparticles_qcd.h5` |
| `nparticles_cdf.py` | Same (figure and statistics) | `nparticles_qcd.h5` | `nparticles_cdf.pdf`, optional stats JSON |
| `pe_collision.py` | Table `tab:pe-collision` and the `r_scale` / `r_max` values (App. "Architecture Details", Positional encoding) | JetClass QCD ROOT (`part_deta/dphi`), validation split | `pe_collision/pe_collision.{json,csv}` |
| `scale_factor_proxy.py` | Table `tab:sf-proxy` (App. "Scale-Factor Proxy") | TopTagging-finetuned runs: `run-*/inference/output_test_{toptagging,jetclass}_best-0.pt` | `sf_proxy/sf_proxy_<method>.json` |

`_common.py` holds the shared helpers: path placeholders, ROOT/HDF5 loading,
and access to the release modules under `dino/`. Every script documents all of
its options under `--help`.

Requirements: the `parcel` environment plus `fastjet`, which is needed by
`kt_scales.py` and also by `dino/preprocess/jetclass/cluster.py`
(`pip install fastjet`).

## kT view scales: `kt_scales.py`

The script reclusters each QCD jet that has more than 16 constituents with the
exclusive kT algorithm (R = 0.8). For each N in {2, 3, 4, 6, 8, 16} it
reports the median [IQR] of two quantities:

- the merge scale `q_N = sqrt(d_merge(N))`, where `d_merge(N)` is the kT
  distance of the N+1 → N step;
- `pT_soft / pT_jet`, the pT of the softest of the N subjets over the jet pT.

It also reports how often the q_N ladder is strictly decreasing. The paper
uses the first 50k QCD jets of the training split, which is the default:

```bash
python studies/physics/kt_scales.py \
    --inputs "PROJECT_ROOT/data/JetClass/raw/train_100M/ZJetsToNuNu_*.root" \
    --max-jets 50000
```

The script also accepts padded HDF5 files with `part_*` datasets. Real
particles are the entries with non-zero `part_energy`. Use `--h5-label 0` to
keep only QCD jets from a mixed-class file.

## Constituent multiplicity: `nparticles_count.py` → `nparticles_cdf.py`

This is a two-step pipeline:

1. `nparticles_count.py` counts the constituents of each jet (the non-zero
   `part_energy` entries) for every split. It writes one dataset per split,
   plus `combined`.
2. `nparticles_cdf.py` reads that file, prints the median, the mean ± std and
   P(n ≤ t), and plots the CDF with the kT view range (2 to 16) shaded.

```bash
python studies/physics/nparticles_count.py \
    --split train="PROJECT_ROOT/data/JetClass/raw/train_100M/ZJetsToNuNu_*.root" \
    --split val="PROJECT_ROOT/data/JetClass/raw/val_5M/ZJetsToNuNu_*.root" \
    --split test="PROJECT_ROOT/data/JetClass/raw/test_20M/ZJetsToNuNu_*.root"
python studies/physics/nparticles_cdf.py \
    --output PROJECT_ROOT/experiments/studies/physics/nparticles_cdf.pdf \
    --stats-json PROJECT_ROOT/experiments/studies/physics/nparticles_stats.json
```

## Positional-encoding collisions: `pe_collision.py`

The script standardizes (Δη, Δφ) with the training normalization and keeps
the first 90 constituents of each jet. JetClass constituents are pT-ordered,
so this matches the pretraining `max_seq_length`. It then computes:

- the radial percentiles: `r_scale` = median and `r_max` = 99th percentile,
  in standardized units;
- the per-particle, per-jet and mask–mask collision rates for the Cartesian,
  Log-Cartesian and Polar grids, from 20×20 to 120×120.

Patch indices come from the release encoders in
`dino/models/positional_encoding.py`, so they match training exactly.
Cartesian and Log-Cartesian share the bound max(p99 |Δη|, p99 |Δφ|).
Log-Cartesian uses the log scale `r_scale / 2`.

The mask–mask rate masks round(0.25·n) constituents per jet, an exact 25%
mask as in the paper table. `--mask-count floor` switches to the iBOT
training rule, int(0.25·n). The paper uses the 5×10^5 QCD jets of the
validation split:

```bash
python studies/physics/pe_collision.py \
    --inputs "PROJECT_ROOT/data/JetClass/raw/val_5M/ZJetsToNuNu_*.root"
```

## Scale-factor proxy: `scale_factor_proxy.py`

This uses the TopTagging-finetuned models, one experiment directory per
method, each holding `run-*` subdirectories with inference outputs from
`dino/dino_inference.py` or `baselines/scripts/classification_inference.py`.

For each run, the source (TopTagging, signal label 1) signal-score CDF is cut
at the quantiles `--rank-edges` (default 0, .3, .5, .7, .9, .95, 1). The same
score thresholds are then applied to JetClass Tbqq (label 8). The per-interval
SF is the ratio of the target to the source signal fraction.

Every run must pass three closure checks: the source fractions, the target
fractions and the source-weighted SFs each sum to one. An empty source
interval is an error, and every run must use the same test jets. The table
reports the mean ± std (ddof = 1) over runs, and the mean over intervals of
|mean SF − 1|.

```bash
python studies/physics/scale_factor_proxy.py compute \
    --method Supervised=PROJECT_ROOT/experiments/<tt-finetune-supervised> \
    --method RINO=PROJECT_ROOT/experiments/<tt-finetune-rino> \
    --expected-runs RINO=<n_runs>
# Re-print from the JSON files (rows follow the order of --inputs):
python studies/physics/scale_factor_proxy.py table \
    --inputs PROJECT_ROOT/experiments/studies/physics/sf_proxy/sf_proxy_Supervised.json \
             PROJECT_ROOT/experiments/studies/physics/sf_proxy/sf_proxy_RINO.json \
    --format latex
```
