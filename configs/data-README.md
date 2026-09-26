# Data Preparation

All commands are run from the repository root. Datasets live under `data/` in
the repository root. The configs refer to this directory as
`PROJECT_ROOT/data/...` (dataloader YAMLs, resolved by the dataloader code) or as
`${paths.root_dir}/../../data/...` (Hydra configs of the baselines). To keep the
data elsewhere, symlink it: `ln -s <path/to/storage> data`.

Layout expected by the configs:

```
data/
├── JetClass/
│   ├── raw/{train_100M,val_5M,test_20M}/*.root        # original JetClass files
│   ├── clustered/{train,val,test}/QCD.h5              # RINO pretraining (kT views)
│   └── mpm-rino/{train_100M,val_5M,test_20M}_combined_QCD.h5   # baselines
├── JetNet/JetNet30_gqt/gqt_{train,val,test}.h5        # finetuning
└── TopTagging/preprocessed/{train,validation,test}.h5 # finetuning
```

Intermediate files (the downloaded `.tar` archives, `JetClass/qcd-shuffled/`,
`JetClass/qcd-kt/`, the per-file directories under `JetClass/mpm-rino/` and
`TopTagging/raw/`) can be deleted once the final files are written.

## JetClass

[JetClass](https://zenodo.org/records/6619768) provides the pretraining data
(QCD jets only) and the out-of-distribution test set.

### 1. Download

```bash
mkdir -p data/JetClass/raw && cd data/JetClass/raw
pip install zenodo-get
zenodo_get 6619768                                    # JetClass_Pythia_*.tar
for f in JetClass_Pythia_*.tar; do tar -xf "$f"; done # -> train_100M/ val_5M/ test_20M/
cd -
```

The raw test split (`TTBar_*.root` and `ZJetsToNuNu_*.root` under
`raw/test_20M/`) is read directly by `configs/dataloaders/jetclass-raw/kinematics.yaml`
for the top-vs-QCD evaluation; no further processing is needed for evaluation.

### 2. RINO pretraining data (QCD, kT-clustered)

Three steps: shuffle the QCD (`ZJetsToNuNu`) files and keep the 90 highest-pT
particles, cluster every jet into its exclusive-kT subjet views
(N = 1, 2, 3, 4, 6, 8, 16), and concatenate the shards into one HDF5 file per
split.

```bash
# 2a. Shuffle QCD jets across files (seed 42, at most 90 particles per jet)
for split in train_100M:train val_5M:val test_20M:test; do
  src=${split%%:*}; dst=${split##*:}
  python dino/preprocess/jetclass/shuffle.py \
    --input data/JetClass/raw/${src}/ZJetsToNuNu_*.root \
    --out-dir data/JetClass/qcd-shuffled/${dst} \
    --chunk-size 1000 --output-digits 2 --format root \
    --compression 0 --seed 42 --max-particles 90 --threads 8
done

# 2b. Exclusive-kT subjet views (R = 0.8)
for split in train val test; do
  python dino/preprocess/jetclass/cluster.py \
    --in data/JetClass/qcd-shuffled/${split} \
    --out data/JetClass/qcd-kt/${split} \
    --algorithm kt --nums-prongs 1 2 3 4 6 8 16 \
    --subjet-kinematics-only --sort-by-pt --output-format hdf5 --threads 8
done

# 2c. One HDF5 file per split
for split in train val test; do
  python dino/preprocess/jetclass/combine.py \
    --in data/JetClass/qcd-kt/${split} \
    --out data/JetClass/clustered/${split}/QCD.h5 \
    --pattern "*.h5" --chunk-rows 1000
done
```

`configs/dataloaders/jetclass-clustered/kinematics.yaml` reads
`data/JetClass/clustered/{train,val,test}/`. `cluster.py` also supports
`--algorithm ca` and `--algorithm anti-kt`; write their output to a separate
directory and point a copy of the dataloader config at it.

### 3. Baseline data (MPMv1, MPMv2, OmniJet-alpha, VQ-VAE)

The baselines read the QCD jets as per-split HDF5 files with the same seven
RINO particle features:

```bash
# One HDF5 file per ROOT file: data/JetClass/mpm-rino/{train_100M,val_5M,test_20M}/
python baselines/mpmv2/scripts/make_jetclass.py \
  --source_path data/JetClass/raw \
  --dest_path data/JetClass/mpm-rino \
  --pattern "ZJetsToNuNu_*.root"

# Shuffle and concatenate: data/JetClass/mpm-rino/{split}_combined_QCD.h5
python baselines/mpmv2/scripts/combine_jetclass.py \
  --data_path data/JetClass/mpm-rino \
  --patterns "ZJetsToNuNu*.h5" --suffix QCD
```

MPMv1 additionally uses VQ-VAE token labels precomputed next to these files
(`*_tokens.h5`); they require the trained VQ-VAE, see the
[Baselines](../README.md#baselines) section of the main README.

## JetNet (finetuning)

[JetNet](https://github.com/jet-net/JetNet) gluon, light-quark and top jets
with 30 particles, split 70/15/15 and merged into one shuffled file per split
(the `jetnet` package downloads the data):

```bash
python dino/preprocess/jetnet/download.py \
  --download-dir data/JetNet/JetNet30_gqt \
  --jet-types g q t --num-particles 30 --together --seed 42
```

This writes `gqt_{train,val,test}.h5` (read by
`configs/dataloaders/jetnet/kinematics.yaml`), plus per-type files and the raw
download (remove the latter with `--clean-up`).

## Top Quark Tagging (finetuning)

The [Top Quark Tagging](https://zenodo.org/records/2603256) dataset is fetched
from its Hugging Face mirror (`dl4phys/top_tagging`, requires `datasets`) and
converted to HDF5 with the 128 highest-pT particles per jet:

```bash
python dino/preprocess/toptagging/download.py --download-dir data/TopTagging/raw

python dino/preprocess/toptagging/preprocess.py \
  --in data/TopTagging/raw \
  --out data/TopTagging/preprocessed \
  --dr 0.8 --algorithm kt --nums-prongs 2 3 4 \
  --num-particles 128 --chunk-size 50000 --output-format hdf5 \
  --shuffle --seed 42 --sort-by-pt --threads 8
```

This writes `{train,validation,test}.h5`, read by
`configs/dataloaders/toptagging/kinematics.yaml`.
