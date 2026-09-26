"""Harvest per-seed ACC/AUC of finetuned models into one CSV.

For every experiment in the manifest and every finetuning run (``run-<N>/``),
reads the inference outputs of ``dino/dino_inference.py`` and records the
accuracy (threshold 0.5) and ROC AUC on each requested label subset, e.g. the
OOD top-tagging task ``Tbqq_vs_QCD`` (JetClass labels 8 vs 0) or the unseen
topologies ``H4q_vs_QCD`` / ``Hbb_vs_QCD``. The output has one row per
(experiment, run, subset) and carries every manifest tag as a column, so the
same file feeds ``make_table.py``, ``plot_data_efficiency.py``,
``mlp_vs_linear_welch.py`` and ``finetune_reg_sweep.py summarize``.

Metrics are taken from the ``subsets`` block of ``metrics_<split>_<epoch>.json``
when the config listed the subset under ``eval_subsets``, and otherwise
recomputed from the saved logits (``--source logits`` always recomputes).
Recomputing needs the evaluated files to contain the subset's classes, e.g.
``HToWW4Q_*.root`` for ``H4q_vs_QCD``.

Usage:
    # Table 1 finetuned heads (JetNet -> JetClass, t->bqq vs QCD)
    python studies/evaluation/harvest_finetune.py \
        --manifest <path/to/manifest.yaml> \
        --subsets Tbqq_vs_QCD H4q_vs_QCD Hbb_vs_QCD \
        --output experiments/studies/finetune_per_seed.csv

    # One experiment from the command line
    python studies/evaluation/harvest_finetune.py \
        --exp method=RINO,head=mlp,config=<path/to/finetune-config.yaml> \
        --output experiments/studies/rino_per_seed.csv

    # In-distribution accuracy on the JetNet test split
    python studies/evaluation/harvest_finetune.py --manifest <path/to/manifest.yaml> \
        --split test_jetnet --subsets all --output experiments/studies/jetnet_per_seed.csv

Manifest format (YAML list; every key other than config/dir is a tag column):
    - {method: RINO, head: mlp, config: configs/<finetune-config>.yaml}
    - {method: MPMv2, head: linear, dir: experiments/<finetune-dir>/<name>}
"""

import argparse
import sys

from common import (
    ALL_SUBSET,
    PER_SEED_VALUE_COLUMNS,
    binary_metrics,
    entry_label,
    entry_tags,
    find_runs,
    load_json_metrics,
    load_manifest,
    load_scores,
    locate_outputs,
    mean_std,
    output_files,
    parse_subset,
    write_csv,
)


def harvest_run(
    inference_dir,
    split: str,
    epoch: str,
    subsets: list[tuple],
    template: str,
    source: str = "auto",
) -> dict[str, dict]:
    """Metrics of one finetuning run on each subset: ``{subset: {acc, auc, n_signal, n_background, source}}``.

    With ``source="auto"``, subsets already evaluated by ``dino_inference.py``
    are read from its metrics JSON and the rest are recomputed from logits.
    Subsets whose classes are absent from the evaluated files are left out.
    """
    results: dict[str, dict] = {}
    stored = load_json_metrics(inference_dir, split, epoch) if source in ("auto", "json") else {}
    missing = []
    for name, pos, neg in subsets:
        if name in stored:
            results[name] = {**stored[name], "source": "json"}
        else:
            missing.append((name, pos, neg))
    if missing and source == "json":
        raise FileNotFoundError(
            f"{inference_dir}: no stored metrics for {[m[0] for m in missing]} on {split}; use --source auto."
        )
    if missing:
        files = output_files(inference_dir, split, epoch, template)
        if not files:
            raise FileNotFoundError(f"{inference_dir}: no inference output for split '{split}', epoch '{epoch}'.")
        scores, labels = load_scores(files)
        for name, pos, neg in missing:
            try:
                results[name] = {**binary_metrics(scores, labels, pos, neg), "source": "logits"}
            except ValueError as err:  # the evaluated files lack this subset's classes
                print(f"[warn] {inference_dir}: subset {name} skipped: {err}", file=sys.stderr)
    return results


def harvest(entries: list[dict], split: str, epoch: str, subsets: list[tuple], source: str, inference_subdir: str) -> list[dict]:
    """Harvest every run of every manifest entry into per-seed rows."""
    rows: list[dict] = []
    for entry in entries:
        exp_dir, subdir, template = locate_outputs(entry, inference_subdir)
        runs = find_runs(exp_dir, subdir)
        if not runs:
            print(f"[warn] {entry_label(entry)}: no runs under {exp_dir}", file=sys.stderr)
            continue
        n_ok = 0
        for run, inference_dir in runs:
            try:
                metrics = harvest_run(inference_dir, split, epoch, subsets, template, source)
            except (FileNotFoundError, KeyError, ValueError) as err:
                print(f"[warn] {entry_label(entry)} run {run}: {err}", file=sys.stderr)
                continue
            n_ok += 1
            for name, _, _ in subsets:
                if name in metrics:
                    rows.append({**entry_tags(entry), "run": run, "split": split, "subset": name, **metrics[name]})
        print(f"{entry_label(entry)}: {n_ok}/{len(runs)} runs from {exp_dir}")
    return rows


def print_summary(rows: list[dict], ddof: int) -> None:
    """Print mean ± std of ACC and AUC per experiment and subset."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        tags = tuple((k, v) for k, v in r.items() if k not in PER_SEED_VALUE_COLUMNS)
        groups.setdefault((tags, r["subset"]), []).append(r)
    for (tags, subset), rs in groups.items():
        acc = mean_std([r["acc"] for r in rs], ddof)
        auc = mean_std([r["auc"] for r in rs], ddof)
        label = ", ".join(f"{k}={v}" for k, v in tags)
        print(f"  {label:<50s} {subset:<14s} ACC {acc[0]:.4f} ± {acc[1]:.4f}  AUC {auc[0]:.4f} ± {auc[1]:.4f}  (n={acc[2]})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=str, default=None, help="YAML manifest of experiments.")
    parser.add_argument("--exp", type=str, action="append", default=None, help="Experiment spec key=value,...; repeatable.")
    parser.add_argument("--split", type=str, default="test_jetclass", help="Inference split to read.")
    parser.add_argument("--epoch", type=str, default="best", help="Checkpoint tag used at inference (load_epoch).")
    parser.add_argument(
        "--subsets", type=str, nargs="+", default=["Tbqq_vs_QCD"],
        help=f"Label subsets: names such as Tbqq_vs_QCD, H4q_vs_QCD, Hbb_vs_QCD, '{ALL_SUBSET}', or NAME=POS:NEG.",
    )
    parser.add_argument("--source", choices=["auto", "json", "logits"], default="auto", help="Where metrics come from.")
    parser.add_argument("--inference-subdir", type=str, default="inference", help="Inference directory name for 'dir' entries.")
    parser.add_argument("--ddof", type=int, default=1, help="Delta degrees of freedom of the printed std.")
    parser.add_argument("--output", type=str, required=True, help="Per-seed CSV to write.")
    args = parser.parse_args()

    entries = load_manifest(args.manifest, args.exp)
    subsets = [parse_subset(s) for s in args.subsets]
    rows = harvest(entries, args.split, args.epoch, subsets, args.source, args.inference_subdir)
    if not rows:
        raise SystemExit("Nothing harvested.")
    print_summary(rows, args.ddof)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
