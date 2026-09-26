"""Robustness of OOD accuracy to the finetuning weight decay and dropout.

Two subcommands:

``generate``
    Writes one finetuning config per (method, weight decay, dropout) cell from
    each method's default finetuning config (the Table 1 recipe) by changing
    only ``training.optimizer.params.weight_decay``, the head
    ``models.head.params.dropouts`` and the job ``name``, and writes a manifest
    for ``harvest_finetune.py``. The default cell is the template itself, so its
    manifest entry points at the template and reuses the Table 1 runs (pass
    ``--rerun-default`` to write a separate config for it instead).

``summarize``
    Reads the per-seed CSV that ``harvest_finetune.py`` produced from that
    manifest and reports, per method, the default cell and the lowest (Worst)
    and highest (Best) mean over all cells, as mean ± std over seeds, plus the
    spread of the cell means. Best is selected on the evaluated target data.

Usage:
    python studies/evaluation/finetune_reg_sweep.py generate \
        --template Supervised=<path/to/supervised-finetune.yaml> \
        --template MPMv1=<path/to/mpmv1-finetune.yaml> \
        --template MPMv2=<path/to/mpmv2-finetune.yaml> \
        --template RINO=<path/to/rino-finetune.yaml> \
        --out-dir configs/finetune-reg

    # train + infer every generated config with --run-index 1..10, then
    python studies/evaluation/harvest_finetune.py --manifest configs/finetune-reg/manifest.yaml \
        --output experiments/studies/finetune_reg_per_seed.csv
    python studies/evaluation/finetune_reg_sweep.py summarize \
        --csv experiments/studies/finetune_reg_per_seed.csv --format latex
"""

import argparse
import copy
import os

import yaml

from common import (
    as_float,
    filter_rows,
    fmt_pm,
    load_yaml,
    mean_std,
    parse_where,
    read_csvs,
    render_table,
    resolve_path,
)

# (weight decay, dropout) cells: weight decay scanned at the default dropout and
# dropout scanned at the default weight decay.
DEFAULT_CELLS = ["0.01:0.1", "0.05:0.1", "0.10:0.1", "0.05:0.3", "0.05:0.5"]
DEFAULT_CELL = "0.05:0.1"


def parse_cell(cell: str) -> tuple[float, float]:
    """Parse ``WEIGHT_DECAY:DROPOUT``."""
    wd, do = cell.split(":")
    return float(wd), float(do)


def cell_tag(wd: float, do: float) -> str:
    """Name suffix of a cell, e.g. ``wd0.01-do0.1``."""
    return f"wd{wd:g}-do{do:g}"


def make_cell_config(template: dict, wd: float, do: float, jetclass_patterns: list[str] | None = None) -> dict:
    """Copy a finetuning config with a new weight decay, head dropout and job name."""
    # Each cell writes to its own experiment directory only if the paths depend on the job name.
    for key, path in (("training.checkpoints_dir", template["training"]["checkpoints_dir"]),
                      ("inference.output_dir", template["inference"]["output_dir"])):
        if "JOBNAME" not in path:
            raise SystemExit(f"{template['name']}: {key} must contain the JOBNAME placeholder.")
    cfg = copy.deepcopy(template)
    cfg["name"] = f"{template['name']}-{cell_tag(wd, do)}"
    cfg["training"]["optimizer"]["params"]["weight_decay"] = wd
    head = cfg["models"]["head"]["params"]
    head["dropouts"] = [do] * len(head.get("hidden_dims", head.get("dropouts", [])))
    if jetclass_patterns:
        cfg["inference"]["dataloader"]["test_jetclass"]["kwargs"]["patterns"] = list(jetclass_patterns)
    return cfg


def generate(args) -> None:
    """Write the sweep configs and their manifest."""
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_wd, default_do = parse_cell(args.default_cell)
    manifest = []
    for spec in args.template:
        method, path = spec.split("=", 1)
        template = load_yaml(path)
        for cell in args.cells:
            wd, do = parse_cell(cell)
            entry = {"method": method, "weight_decay": wd, "dropout": do}
            is_default = abs(wd - default_wd) < 1e-12 and abs(do - default_do) < 1e-12
            if is_default and not args.rerun_default:
                entry["config"] = str(path)
            else:
                cfg = make_cell_config(template, wd, do, args.jetclass_patterns)
                cfg_path = out_dir / f"{cfg['name']}.yaml"
                with open(cfg_path, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
                entry["config"] = os.path.relpath(cfg_path)
                print(f"  wrote {cfg_path}")
            manifest.append(entry)
    manifest_path = out_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"Manifest with {len(manifest)} entries: {manifest_path}")


def summarize(args) -> None:
    """Print Default / Worst / Best per method from a harvested per-seed CSV."""
    where = parse_where(args.where)
    where["subset"] = [args.subset]
    rows = filter_rows(read_csvs(args.csv), where)
    if not rows:
        raise SystemExit("No rows left after filtering.")
    default_wd, default_do = parse_cell(args.default_cell)

    cells: dict[str, dict[tuple, list[float]]] = {}
    for r in rows:
        key = (as_float(r["weight_decay"]), as_float(r["dropout"]))
        cells.setdefault(r[args.group], {}).setdefault(key, []).append(as_float(r[args.metric]))

    methods = list(cells)
    if args.order:
        methods = [m for m in args.order if m in cells] + [m for m in methods if m not in args.order]

    stats_rows = []
    for m in methods:
        per_cell = {k: mean_std(v, args.ddof) for k, v in cells[m].items()}
        default = next((s for (wd, do), s in per_cell.items() if abs(wd - default_wd) < 1e-12 and abs(do - default_do) < 1e-12), None)
        worst_key = min(per_cell, key=lambda k: per_cell[k][0])
        best_key = max(per_cell, key=lambda k: per_cell[k][0])
        stats_rows.append({
            "method": m,
            "default": default,
            "worst": per_cell[worst_key],
            "best": per_cell[best_key],
            "worst_cell": worst_key,
            "best_cell": best_key,
            "n_cells": len(per_cell),
            "spread": per_cell[best_key][0] - per_cell[worst_key][0],
        })

    columns = ["default", "worst", "best"]
    best_of_col = {c: max((r[c][0] for r in stats_rows if r[c] is not None), default=None) for c in columns}
    header = [args.group, "Default", "Worst", "Best (target-selected)", "spread", "cells"]
    body = []
    for r in stats_rows:
        line = [r["method"]]
        for c in columns:
            s = r[c]
            if s is None:
                line.append(fmt_pm(float("nan"), float("nan"), args.digits, args.format))
            else:
                line.append(fmt_pm(s[0], s[1], args.digits, args.format, bold=args.highlight and s[0] == best_of_col[c]))
        line.append(f"{r['spread']:.{args.digits}f}")
        line.append(str(r["n_cells"]))
        body.append(line)
    print(render_table(header, body, args.format))
    for r in stats_rows:
        print(f"  {r['method']}: worst cell (wd, dropout) = {r['worst_cell']}, best cell = {r['best_cell']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Write sweep configs and a manifest.")
    gen.add_argument("--template", type=str, action="append", required=True, help="METHOD=CONFIG; repeatable.")
    gen.add_argument("--out-dir", type=str, required=True, help="Directory for the generated configs and manifest.yaml.")
    gen.add_argument("--cells", type=str, nargs="+", default=DEFAULT_CELLS, help="Cells as WEIGHT_DECAY:DROPOUT.")
    gen.add_argument("--default-cell", type=str, default=DEFAULT_CELL, help="Cell equal to the template recipe.")
    gen.add_argument("--rerun-default", action="store_true", help="Also write a config for the default cell.")
    gen.add_argument(
        "--jetclass-patterns", type=str, nargs="+", default=None,
        help="Replace the test_jetclass file patterns of the generated configs (e.g. to evaluate on fewer files).",
    )
    gen.set_defaults(func=generate)

    summ = sub.add_parser("summarize", help="Default / Worst / Best table from a harvested CSV.")
    summ.add_argument("--csv", type=str, nargs="+", required=True, help="Per-seed CSV(s) from harvest_finetune.py.")
    summ.add_argument("--where", type=str, action="append", default=None, help="Filter key=value[,value]; repeatable.")
    summ.add_argument("--subset", type=str, default="Tbqq_vs_QCD")
    summ.add_argument("--metric", type=str, default="acc", choices=["acc", "auc"])
    summ.add_argument("--group", type=str, default="method")
    summ.add_argument("--default-cell", type=str, default=DEFAULT_CELL)
    summ.add_argument("--order", type=str, nargs="+", default=None, help="Row order of methods.")
    summ.add_argument("--ddof", type=int, default=1, help="Delta degrees of freedom of the std over seeds.")
    summ.add_argument("--digits", type=int, default=3)
    summ.add_argument("--format", choices=["markdown", "latex", "csv"], default="markdown")
    summ.add_argument("--highlight", action="store_true", help="Bold the best mean per column.")
    summ.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
