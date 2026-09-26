"""Harvest frozen-backbone k-NN and linear-probe accuracies into a summary CSV.

The k-NN (k=20) and linear-probe columns of Table 1 and of the ablation tables
are fit on frozen pretrained embeddings of JetClass t->bqq and QCD jets. Two
programs produce them, and this script reads the JSON written by either:

* ``baselines/scripts/backbone_probe.py --output <file>`` for backbones exported
  as ``backbone.pt``: standardised embeddings, cosine-metric k-NN and logistic
  regression, over repeated random 80/20 splits of a 200k-jet subsample.
* ``dino/dino_inference.py`` with ``inference.acc_tests`` in the pretraining
  config, for checkpoints trained with this repository (RINO, its ablations and
  JetCLR-scale): Euclidean k-NN on the CLS token over stratified folds, and a
  torch linear probe on the CLS token concatenated with the mean-pooled
  particle tokens. The file is ``<inference output_dir>/metrics_<split>_<epoch>.json``.

The stored mean and standard deviation over probe repeats are copied as they
are. Rows use the ``head`` column (``knn`` / ``linear_probe``) so they can be
tabulated next to the finetuned heads (``head: linear`` / ``head: mlp``).

Usage:
    python studies/evaluation/harvest_probes.py \
        --exp method=MPMv2,path=<path/to/probe_results.json> \
        --exp method=RINO,config=configs/dino/<pretrain-config>.yaml \
        --output experiments/studies/probes.csv

Manifest entries locate the probe JSON with ``path`` (a file), ``dir`` (an
experiment directory containing ``inference/``) or ``config`` (the pretraining
config used for inference); all other keys are tag columns.
"""

import argparse
import json
import sys

from common import (
    entry_label,
    entry_tags,
    load_manifest,
    locate_outputs,
    resolve_path,
    write_csv,
)


def probe_json_path(entry: dict, split: str, epoch: str, inference_subdir: str = "inference"):
    """Locate the probe JSON of a manifest entry (a file, or the inference directory of an experiment)."""
    if "path" in entry:
        return resolve_path(entry["path"])
    exp_dir, subdir, _ = locate_outputs(entry, inference_subdir)
    return exp_dir / subdir / f"metrics_{split}_{epoch}.json"


def parse_probe_json(data: dict, k: int) -> list[dict]:
    """Extract ``{head, metric, mean, std, n, values}`` records from either probe format."""
    records: list[dict] = []
    knn = data.get("knn")
    if knn:
        # backbone_probe.py stores one k at the top level; dino_inference.py keys by k.
        entry = knn if "acc_mean" in knn else knn.get(str(k))
        if entry is None:
            raise KeyError(f"k={k} not in k-NN results (available: {sorted(knn)})")
        records.append(_record("knn", "acc", entry["acc_mean"], entry["acc_std"], entry.get("acc_values")))
    lp = data.get("linear_probe")
    if lp:
        records.append(_record("linear_probe", "acc", lp["test_mean"], lp["test_std"], lp.get("test_values")))
        if "test_auc_mean" in lp:
            records.append(_record("linear_probe", "auc", lp["test_auc_mean"], lp["test_auc_std"], lp.get("test_auc_values")))
    return records


def _record(head: str, metric: str, mean: float, std: float, values) -> dict:
    return {
        "head": head,
        "metric": metric,
        "mean": float(mean),
        "std": float(std),
        "n": len(values) if values else "",
        "values": ";".join(f"{v:.6f}" for v in values) if values else "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=str, default=None, help="YAML manifest of probe results.")
    parser.add_argument("--exp", type=str, action="append", default=None, help="Entry spec key=value,...; repeatable.")
    parser.add_argument("--split", type=str, default="test_jetclass", help="Split the probes were run on.")
    parser.add_argument("--epoch", type=str, default="best", help="Checkpoint tag used at inference (load_epoch).")
    parser.add_argument("--k", type=int, default=20, help="Number of neighbours of the reported k-NN.")
    parser.add_argument("--subset", type=str, default="Tbqq_vs_QCD", help="Subset name recorded for the probe labels.")
    parser.add_argument("--inference-subdir", type=str, default="inference", help="Inference directory name for 'dir' entries.")
    parser.add_argument("--output", type=str, required=True, help="Summary CSV to write.")
    args = parser.parse_args()

    rows: list[dict] = []
    for entry in load_manifest(args.manifest, args.exp):
        path = probe_json_path(entry, args.split, args.epoch, args.inference_subdir)
        if not path.exists():
            print(f"[warn] {entry_label(entry)}: missing {path}", file=sys.stderr)
            continue
        with open(path) as f:
            records = parse_probe_json(json.load(f), args.k)
        for rec in records:
            rows.append({**entry_tags(entry), "split": args.split, "subset": args.subset, **rec})
            print(f"  {entry_label(entry):<40s} {rec['head']:<13s} {rec['metric']}: {rec['mean']:.4f} ± {rec['std']:.4f}")
    if not rows:
        raise SystemExit("Nothing harvested.")
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
