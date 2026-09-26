"""Shared helpers for the evaluation studies.

The evaluation scripts in this directory read the outputs that
``dino/dino_inference.py`` writes for every finetuning run::

    <exp_dir>/run-<N>/inference/output_<split>_<epoch>-<i>.pt   # logits, label, rep, ...
    <exp_dir>/run-<N>/inference/metrics_<split>_<epoch>.json    # JetClass splits only
    <exp_dir>/run-<N>/inference/metrics_<epoch>.json            # all labelled splits

and turn them into per-seed CSV files, summary tables and figures. This module
holds the pieces they share: experiment manifests, label subsets, metric
computation, mean/std aggregation and table rendering.

Scores are ``sigmoid(logits[:, 0])`` of a binary head with a fixed decision
threshold of 0.5, as in ``dino/dino_inference.py``.
"""

import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "dino"))

from utils.ckpt import process_placeholder  # noqa: E402

# JetClass label indices, as assigned by the jetclass_labeler processor
# (dino/dataloader/jetclass/processors.py).
JETCLASS_LABELS: dict[str, int] = {
    "QCD": 0,
    "Hbb": 1,
    "Hcc": 2,
    "Hgg": 3,
    "H4q": 4,
    "Hqql": 5,
    "Zqq": 6,
    "Wqq": 7,
    "Tbqq": 8,
    "Tbl": 9,
}

# Special subset name: every label other than 0 is signal and label 0 is
# background. On JetNet and TopTagging (labels 0/1) this is the in-distribution
# top-vs-QCD task.
ALL_SUBSET = "all"

# Columns written by harvest scripts that are not experiment tags.
PER_SEED_VALUE_COLUMNS = ("run", "split", "subset", "acc", "auc", "n_signal", "n_background", "source")

# Manifest keys that locate outputs rather than tag an experiment.
LOCATION_KEYS = ("dir", "config", "path")


# --------------------------------------------------------------------------- #
# Label subsets                                                               #
# --------------------------------------------------------------------------- #
def predefined_subsets() -> dict[str, tuple[list[int] | None, list[int]]]:
    """Return the named signal-vs-background subsets understood by ``parse_subset``.

    Each JetClass signal class ``X`` defines ``X_vs_QCD``; ``top_vs_QCD``
    combines both top decays; ``all`` treats every non-zero label as signal.
    """
    subsets: dict[str, tuple[list[int] | None, list[int]]] = {
        f"{name}_vs_QCD": ([idx], [JETCLASS_LABELS["QCD"]])
        for name, idx in JETCLASS_LABELS.items()
        if name != "QCD"
    }
    subsets["top_vs_QCD"] = ([JETCLASS_LABELS["Tbqq"], JETCLASS_LABELS["Tbl"]], [JETCLASS_LABELS["QCD"]])
    subsets[ALL_SUBSET] = (None, [0])
    return subsets


def parse_subset(spec: str) -> tuple[str, list[int] | None, list[int]]:
    """Parse a subset given by name or as ``NAME=POS[,POS...]:NEG[,NEG...]``.

    Examples: ``Tbqq_vs_QCD``, ``H4q_vs_QCD``, ``all``, ``top_vs_QCD=8,9:0``.
    A ``None`` positive list means "every label that is not a negative label".
    """
    if "=" in spec:
        name, labels = spec.split("=", 1)
        pos_str, neg_str = labels.split(":")
        return name, [int(x) for x in pos_str.split(",")], [int(x) for x in neg_str.split(",")]
    known = predefined_subsets()
    if spec not in known:
        raise ValueError(f"Unknown subset '{spec}'. Known: {sorted(known)}; or use NAME=POS:NEG.")
    pos, neg = known[spec]
    return spec, pos, neg


def subset_masks(labels: np.ndarray, positive: list[int] | None, negative: list[int]):
    """Return boolean (signal, background) masks for a label subset."""
    neg_mask = np.isin(labels, negative)
    pos_mask = ~neg_mask if positive is None else np.isin(labels, positive)
    return pos_mask, neg_mask


def binary_metrics(scores: np.ndarray, labels: np.ndarray, positive: list[int] | None, negative: list[int]) -> dict:
    """Accuracy (threshold 0.5) and ROC AUC of ``scores`` on one label subset."""
    pos_mask, neg_mask = subset_masks(labels, positive, negative)
    keep = pos_mask | neg_mask
    y = pos_mask[keep].astype(int)
    s = scores[keep]
    n_sig, n_bkg = int(y.sum()), int(len(y) - y.sum())
    if n_sig == 0 or n_bkg == 0:
        raise ValueError(f"Subset has {n_sig} signal and {n_bkg} background jets; check the evaluated files.")
    return {
        "acc": float(((s > 0.5).astype(int) == y).mean()),
        "auc": float(roc_auc_score(y, s)),
        "n_signal": n_sig,
        "n_background": n_bkg,
    }


# --------------------------------------------------------------------------- #
# Paths, manifests and run discovery                                          #
# --------------------------------------------------------------------------- #
def resolve_path(path: str | Path) -> Path:
    """Resolve ``PROJECT_ROOT`` and ``~``; relative paths are relative to the working directory."""
    p = Path(str(path).replace("PROJECT_ROOT", str(PROJECT_ROOT))).expanduser()
    return p if p.is_absolute() else Path.cwd() / p


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file."""
    with open(resolve_path(path)) as f:
        return yaml.safe_load(f)


def parse_exp_spec(spec: str) -> dict:
    """Parse a command-line experiment spec ``key=value[,key=value...]`` into a manifest entry."""
    entry = {}
    for item in spec.split(","):
        if "=" not in item:
            raise ValueError(f"Bad experiment spec '{spec}': expected key=value pairs separated by commas.")
        key, value = item.split("=", 1)
        entry[key.strip()] = value.strip()
    return entry


def load_manifest(manifest: str | None = None, exp_specs: list[str] | None = None) -> list[dict]:
    """Collect experiment entries from a YAML manifest and/or ``--exp`` specs.

    A manifest is a YAML list of mappings (or a mapping with an ``experiments``
    list). Each entry locates its outputs with one of

    * ``config``: a finetuning (or pretraining) config; outputs are found from its
      ``inference.output_dir`` and ``inference.output_filename``;
    * ``dir``: an experiment directory holding ``run-<N>/inference/`` (or a
      single ``inference/``);
    * ``path``: a single file (used by ``harvest_probes.py``);

    and every other key (``method``, ``head``, ``fraction``, ...) is copied to
    the output CSV as a tag column.
    """
    entries: list[dict] = []
    if manifest:
        data = load_yaml(manifest)
        if isinstance(data, dict):
            data = data.get("experiments", [])
        if not isinstance(data, list):
            raise ValueError(f"{manifest}: expected a list of experiments.")
        entries.extend(dict(e) for e in data)
    for spec in exp_specs or []:
        entries.append(parse_exp_spec(spec))
    if not entries:
        raise SystemExit("No experiments given: pass --manifest and/or --exp.")
    for e in entries:
        if not any(k in e for k in LOCATION_KEYS):
            raise ValueError(f"Experiment entry {e} has none of {LOCATION_KEYS}.")
    return entries


def entry_tags(entry: dict) -> dict:
    """Return the tag columns of a manifest entry (everything except location keys)."""
    return {k: v for k, v in entry.items() if k not in LOCATION_KEYS}


def entry_label(entry: dict) -> str:
    """Short human-readable label for log messages."""
    tags = entry_tags(entry)
    return ", ".join(f"{k}={v}" for k, v in tags.items()) or str(entry)


def locate_outputs(entry: dict, inference_subdir: str = "inference") -> tuple[Path, str, str]:
    """Return ``(exp_dir, inference_subdir, output_filename_template)`` for a manifest entry.

    For a ``config`` entry the inference directory and file-name template come
    from the config, with placeholders resolved as in ``dino/dino_inference.py``;
    ``--run-index N`` runs are written to ``<exp_dir>/run-N/<inference_subdir>``.
    """
    if "config" in entry:
        cfg = load_yaml(entry["config"])
        out_dir = Path(process_placeholder(cfg["inference"]["output_dir"], cfg, None))
        template = cfg["inference"].get("output_filename", "output_SPLIT_EPOCHNUM.pt")
        template = template.replace("JOBNAME", cfg.get("name", "") or "")
        return out_dir.parent, out_dir.name, template
    return resolve_path(entry["dir"]), inference_subdir, "output_SPLIT_EPOCHNUM.pt"


def find_runs(exp_dir: Path, inference_subdir: str = "inference") -> list[tuple[str, Path]]:
    """List ``(run_label, inference_dir)`` pairs of an experiment, sorted by run index.

    Runs launched with ``--run-index N`` live in ``run-N/``; an experiment run
    without it has a single ``inference/`` directory, labelled ``"0"``.
    """

    def _index(p: Path):
        suffix = p.name.split("-", 1)[1]
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    runs = [
        (rd.name.split("-", 1)[1], rd / inference_subdir)
        for rd in sorted(exp_dir.glob("run-*"), key=_index)
        if (rd / inference_subdir).is_dir()
    ]
    if not runs and (exp_dir / inference_subdir).is_dir():
        runs = [("0", exp_dir / inference_subdir)]
    return runs


# --------------------------------------------------------------------------- #
# Loading inference outputs                                                   #
# --------------------------------------------------------------------------- #
def output_files(inference_dir: Path, split: str, epoch: str, template: str = "output_SPLIT_EPOCHNUM.pt") -> list[Path]:
    """All ``output_<split>_<epoch>-<i>.pt`` shards of one split, in shard order."""
    stem = template.replace("SPLIT", split).replace("EPOCHNUM", str(epoch))
    name, ext = stem.rsplit(".", 1)
    files = list(inference_dir.glob(f"{name}-*.{ext}"))

    def _shard(p: Path) -> int:
        tail = p.name[len(name) + 1 : -(len(ext) + 1)]
        return int(tail) if tail.isdigit() else -1

    files = [p for p in files if _shard(p) >= 0]
    return sorted(files, key=_shard)


def _torch_load(path: Path) -> dict:
    """Load an inference output, memory-mapping it so the large ``rep`` tensor is not read."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:  # files written with the legacy (non-zip) serialization
        return torch.load(path, map_location="cpu", weights_only=False)


def load_scores(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ``sigmoid(logits[:, 0])`` and integer labels over inference shards."""
    scores, labels = [], []
    for path in files:
        data = _torch_load(path)
        if "logits" not in data or "label" not in data:
            raise KeyError(f"{path} has no 'logits'/'label' (keys: {sorted(data)}); was it a finetuned model?")
        logits = data["logits"].float()
        if logits.ndim == 2:
            if logits.shape[1] != 1:
                raise ValueError(f"{path}: expected a binary head (1 logit), got {logits.shape[1]}.")
            logits = logits[:, 0]
        # float32 sigmoid, as in dino_inference.py, so that saturated scores
        # tie identically and recomputed AUCs match the stored ones.
        scores.append(torch.sigmoid(logits).numpy())
        labels.append(data["label"].reshape(-1).long().numpy())
        del data
    return np.concatenate(scores), np.concatenate(labels)


def load_json_metrics(inference_dir: Path, split: str, epoch: str) -> dict[str, dict]:
    """Per-subset metrics stored by ``dino/dino_inference.py`` for one run.

    Returns ``{subset_name: {acc, auc, n_signal, n_background}}`` from the
    ``subsets`` block of ``metrics_<split>_<epoch>.json`` (the ``eval_subsets``
    of the config). For splits other than JetClass, the overall binary metrics
    in ``metrics_<epoch>.json`` are returned under the name ``all``.
    """
    found: dict[str, dict] = {}
    per_split = inference_dir / f"metrics_{split}_{epoch}.json"
    if per_split.exists():
        with open(per_split) as f:
            data = json.load(f)
        for name, m in data.get("subsets", {}).items():
            found[name] = {
                "acc": m["acc"],
                "auc": m["auc"],
                "n_signal": m.get("n_positive", ""),
                "n_background": m.get("n_negative", ""),
            }
    # On JetClass, "all" (every non-zero label vs QCD) is always recomputed from
    # the logits rather than taken from the overall metric of dino_inference.py.
    combined = inference_dir / f"metrics_{epoch}.json"
    if "jetclass" not in split and combined.exists():
        with open(combined) as f:
            data = json.load(f)
        if split in data and "acc" in data[split] and "auc" in data[split]:
            found[ALL_SUBSET] = {"acc": data[split]["acc"], "auc": data[split]["auc"], "n_signal": "", "n_background": ""}
    return found


# --------------------------------------------------------------------------- #
# CSV I/O and aggregation                                                     #
# --------------------------------------------------------------------------- #
def write_csv(path: str | Path, rows: list[dict]) -> None:
    """Write dict rows to CSV; columns are the union of keys in first-seen order."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def read_csvs(paths: list[str]) -> list[dict]:
    """Read and concatenate CSV files into a list of string-valued dict rows."""
    rows: list[dict] = []
    for p in paths:
        with open(resolve_path(p), newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def parse_where(clauses: list[str] | None) -> dict[str, list[str]]:
    """Parse ``key=value[,value...]`` filters into ``{key: allowed_values}``."""
    where: dict[str, list[str]] = {}
    for clause in clauses or []:
        key, value = clause.split("=", 1)
        where.setdefault(key, []).extend(value.split(","))
    return where


def filter_rows(rows: list[dict], where: dict[str, list[str]]) -> list[dict]:
    """Keep rows whose columns match every filter (numeric values compare numerically)."""

    def _match(value: str | None, allowed: list[str]) -> bool:
        if value is None:
            return False
        for a in allowed:
            if value == a:
                return True
            try:
                if math.isclose(float(value), float(a)):
                    return True
            except ValueError:
                pass
        return False

    return [r for r in rows if all(_match(r.get(k), v) for k, v in where.items())]


def mean_std(values, ddof: int = 1) -> tuple[float, float, int]:
    """Mean, standard deviation (``ddof`` as in numpy) and count; std is NaN if undefined."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), 0
    std = float(arr.std(ddof=ddof)) if n > ddof else float("nan")
    return float(arr.mean()), std, n


def to_summary(rows: list[dict], metrics=("acc", "auc"), ddof: int = 1) -> list[dict]:
    """Aggregate per-seed rows (``harvest_finetune.py``) or pass through summary rows.

    Per-seed rows are grouped by every column except the run bookkeeping and
    reduced to ``mean``/``std``/``n`` per metric. Rows that already carry
    ``mean``/``std`` (``harvest_probes.py``) are returned unchanged.
    """
    summary: list[dict] = []
    groups: dict[tuple, dict] = {}
    for row in rows:
        if "mean" in row and row.get("mean", "") != "":
            summary.append(dict(row))
            continue
        tags = {k: v for k, v in row.items() if k not in PER_SEED_VALUE_COLUMNS}
        for metric in metrics:
            value = row.get(metric, "")
            if value in ("", None):
                continue
            key_items = tuple(tags.items()) + (("split", row.get("split", "")), ("subset", row.get("subset", "")), ("metric", metric))
            groups.setdefault(key_items, {"values": []})["values"].append(float(value))
    for key_items, g in groups.items():
        m, s, n = mean_std(g["values"], ddof=ddof)
        summary.append({**dict(key_items), "mean": m, "std": s, "n": n})
    return summary


# --------------------------------------------------------------------------- #
# Formatting                                                                  #
# --------------------------------------------------------------------------- #
def fmt_pm(mean: float, std: float, digits: int = 3, style: str = "markdown", bold: bool = False, underline: bool = False) -> str:
    """Format ``mean ± std`` for a table cell; ``std`` is omitted when undefined."""
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "—" if style != "latex" else "--"
    has_std = std is not None and not (isinstance(std, float) and math.isnan(std))
    if style == "latex":
        body = f"{mean:.{digits}f} \\pm {std:.{digits}f}" if has_std else f"{mean:.{digits}f}"
        if bold:
            body = f"\\mathbf{{{body}}}"
        if underline:
            body = f"\\underline{{{body}}}"
        return f"${body}$"
    body = f"{mean:.{digits}f} ± {std:.{digits}f}" if has_std else f"{mean:.{digits}f}"
    if style == "markdown":
        if bold:
            body = f"**{body}**"
        if underline:
            body = f"<ins>{body}</ins>"
    return body


def render_table(header: list[str], rows: list[list[str]], style: str = "markdown") -> str:
    """Render a table as GitHub markdown, a LaTeX ``tabular`` body, or CSV."""
    if style == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        writer.writerows(rows)
        return buf.getvalue().rstrip("\n")
    if style == "latex":
        lines = [" & ".join(header) + r" \\", r"\midrule"]
        lines += [" & ".join(r) + r" \\" for r in rows]
        return "\n".join(lines)
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    fmt_row = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"  # noqa: E731
    lines = [fmt_row(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


def as_float(value, default=float("nan")) -> float:
    """Convert a CSV cell to float (blank cells become ``default``)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
