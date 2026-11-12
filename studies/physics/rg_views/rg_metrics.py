#!/usr/bin/env python3
"""Metrics for the RG study, computed from rg_views.py output.

paired    Compares two levels of the SAME jets (e.g. parton -> gen = hadronization,
          gen -> reco = detector), view by view: the relative shift of the kT merge
          scale |q_a - q_b| / q_b (the quantity for which power-correction arguments
          predict a ~ Lambda / q_N suppression), and the energy mover's distance between
          the two views normalized by the jet pT (absolute momentum transport).
classify  How well a small permutation-invariant network separates two samples
          (two Pythia settings, two levels of the same jets, or top vs QCD) using only
          one view. Reported as the test ROC AUC (0.5 = indistinguishable), mean and
          spread over seeds. This is the per-view "information" measure.
scaling   Mean number of exclusive kT subjets <N>(q) versus the resolution q, its
          local exponent -d ln<N>/d ln q, and the median kT merge scale q_N per view
          (the paper's Table 8 on these samples).

Examples:
  python rg_metrics.py paired --views views/qcd_nominal.h5 --pair parton gen --pair gen reco \\
      --algorithms kt ca antikt --out results/paired_qcd_nominal.json
  python rg_metrics.py classify --a views/qcd_nominal.h5 --b views/qcd_fsr_pTmin_2p0.h5 \\
      --level reco --algorithm kt --name "FSR pTmin 2.0 vs 0.5 GeV" --out results/auc_pTmin.json
  python rg_metrics.py scaling --views views/qcd_nominal.h5 --level reco --out results/scaling_reco.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np

import rg_common as rc


def _stats(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "sem": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "p16": float(np.percentile(x, 16)),
        "p84": float(np.percentile(x, 84)),
    }


def _q_table(qN, nums):
    """Median and IQR of the kT merge scale per view, over jets that have one
    (q_N = 0 when a jet has no more than N constituents)."""
    table = {}
    for i, n in enumerate(nums):
        q = qN[:, i][qN[:, i] > 0]
        table[f"N{n}"] = (
            {"median": float(np.median(q)), "p25": float(np.percentile(q, 25)),
             "p75": float(np.percentile(q, 75)), "n": int(len(q))}
            if len(q) else {"median": None, "p25": None, "p75": None, "n": 0}
        )
    return table


def _read_rows(ds, idx):
    return ds[np.sort(idx)] if len(idx) else ds[:0]


# ---------------------------------------------------------------------------
# paired
# ---------------------------------------------------------------------------


def _emd_rows(task):
    a_pt, a_dy, a_dphi, a_m, b_pt, b_dy, b_dphi, b_m, norm = task
    out = np.empty(len(norm))
    for i in range(len(norm)):
        ma, mb = a_m[i], b_m[i]
        out[i] = rc.emd(
            a_pt[i][ma], np.stack([a_dy[i][ma], a_dphi[i][ma]], -1),
            b_pt[i][mb], np.stack([b_dy[i][mb], b_dphi[i][mb]], -1),
        ) / norm[i]
    return out


def _paired_distances(arr_a, arr_b, axis, workers):
    a = rc.relative_coords(arr_a, axis)
    b = rc.relative_coords(arr_b, axis)
    norm = np.hypot(axis[:, 0], axis[:, 1])
    chunks = np.array_split(np.arange(len(norm)), max(1, workers * 4))
    tasks = [tuple(x[c] for x in (*a, *b, norm)) for c in chunks if len(c)]
    if workers > 1:
        with Pool(workers) as pool:
            parts = pool.map(_emd_rows, tasks)
    else:
        parts = [_emd_rows(t) for t in tasks]
    return np.concatenate(parts) if parts else np.array([])


def cmd_paired(args):
    f = h5py.File(args.views, "r")
    nums = [int(n) for n in f["meta"].attrs["nums"]]
    rng = np.random.default_rng(args.seed)
    result = {"views_file": args.views, "emd_backend": rc.emd_backend(), "nums": nums, "pairs": {}}
    print(f"EMD backend: {rc.emd_backend()}", file=sys.stderr)
    for la, lb in args.pair:
        for lev in (la, lb):
            if lev not in f:
                sys.exit(f"level '{lev}' is not in {args.views}")
        idx = np.flatnonzero(f[f"{la}/valid"][:] & f[f"{lb}/valid"][:])
        if args.max_jets and len(idx) > args.max_jets:
            idx = rng.choice(idx, args.max_jets, replace=False)
        idx = np.sort(idx)
        axis = f[f"{lb}/p4"][idx].astype(np.float64)  # reference axis and pT: the finer-truth level b
        key = f"{la}->{lb}"
        qa, qb = f[f"{la}/kt/qN"][idx], f[f"{lb}/kt/qN"][idx]
        na, nb = f[f"{la}/nconst"][idx], f[f"{lb}/nconst"][idx]
        entry = {
            "n_jets": int(len(idx)),
            "q_kt": {la: _q_table(qa, nums), lb: _q_table(qb, nums)},
            # relative shift of the kT merge scale of the same jet, |q_a - q_b| / q_b,
            # using only jets with more than N constituents at both levels
            "dq_over_q": {},
            "emd_over_pt": {},
        }
        for i, n in enumerate(nums):
            s = (na > n) & (nb > n) & (qb[:, i] > 0)
            entry["dq_over_q"][f"N{n}"] = _stats(np.abs(qa[s, i] - qb[s, i]) / qb[s, i])
            entry["dq_over_q"][f"N{n}"]["q_median"] = float(np.median(qb[s, i])) if s.any() else None
        t0 = time.time()
        d = _paired_distances(f[f"{la}/uncl"][idx], f[f"{lb}/uncl"][idx], axis, args.workers)
        entry["emd_over_pt"]["uncl"] = _stats(d)
        for algo in args.algorithms:
            if algo not in f[la] or algo not in f[lb]:
                print(f"  skipping {algo}: not in both levels", file=sys.stderr)
                continue
            entry["emd_over_pt"][algo] = {}
            for n in nums:
                d = _paired_distances(f[f"{la}/{algo}/N{n}"][idx], f[f"{lb}/{algo}/N{n}"][idx], axis, args.workers)
                entry["emd_over_pt"][algo][f"N{n}"] = _stats(d)
        print(f"  {key}: {len(idx)} jets in {time.time() - t0:.0f} s", file=sys.stderr)
        result["pairs"][key] = entry
    _dump(result, args.out)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


def _features(path, level, algo, view, idx):
    with h5py.File(path, "r") as f:
        arr = _read_rows(f[f"{level}/uncl"] if view == "uncl" else f[f"{level}/{algo}/{view}"], idx)
        axis = _read_rows(f[f"{level}/p4"], idx).astype(np.float64)
    pt, dy, dphi, mask = rc.relative_coords(arr.astype(np.float64), axis)
    ptj = np.hypot(axis[:, 0], axis[:, 1])[:, None]
    with np.errstate(divide="ignore"):
        lpt = np.where(mask, np.log(np.where(mask, pt, 1.0) / ptj), 0.0)
    x = np.stack([lpt, dy, dphi], -1).astype(np.float32)
    return x, mask


def _pad_width(x, m, width):
    extra = width - x.shape[1]
    if extra <= 0:
        return x, m
    return np.pad(x, ((0, 0), (0, extra), (0, 0))), np.pad(m, ((0, 0), (0, extra)))


def _train_eval(xa, ma, xb, mb, seed, args):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = np.concatenate([xa, xb])
    m = np.concatenate([ma, mb]).astype(np.float32)
    y = np.concatenate([np.zeros(len(xa)), np.ones(len(xb))]).astype(np.float32)
    perm = rng.permutation(len(y))
    x, m, y = x[perm], m[perm], y[perm]
    n_tr, n_va = int(0.6 * len(y)), int(0.2 * len(y))
    tr, va, te = slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, None)

    sel = m[tr].astype(bool)
    mu, sd = x[tr][sel].mean(0), x[tr][sel].std(0) + 1e-6
    x = np.where(m[..., None] > 0, (x - mu) / sd, 0.0).astype(np.float32)

    dev = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    h = args.hidden

    class DeepSets(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(3, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            self.rho = nn.Sequential(nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

        def forward(self, xx, mm):
            return self.rho((self.phi(xx) * mm[..., None]).sum(1)).squeeze(-1)

    model = DeepSets().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
    xtr, mtr, ytr = T(x[tr]), T(m[tr]), T(y[tr])
    xva, mva, yva = T(x[va]), T(m[va]), T(y[va])

    def predict(xx, mm):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(xx[i:i + 4096], mm[i:i + 4096]) for i in range(0, len(xx), 4096)])

    best, best_state, bad = np.inf, None, 0
    for _ in range(args.epochs):
        model.train()
        order = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            opt.zero_grad()
            loss = lossf(model(xtr[b], mtr[b]), ytr[b])
            loss.backward()
            opt.step()
        vloss = lossf(predict(xva, mva), yva).item()
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    scores = predict(T(x[te]), T(m[te])).cpu().numpy()
    return rc.roc_auc(y[te], scores)


def cmd_classify(args):
    rng = np.random.default_rng(args.seed)
    lev_a, lev_b = args.level_a or args.level, args.level_b or args.level
    with h5py.File(args.a, "r") as fa, h5py.File(args.b, "r") as fb:
        nums = [int(n) for n in fa["meta"].attrs["nums"]]
        for f, p, lev in ((fa, args.a, lev_a), (fb, args.b, lev_b)):
            if lev not in f or (args.algorithm not in f[lev]):
                sys.exit(f"{p} has no {lev}/{args.algorithm} views")
        va, vb = fa[f"{lev_a}/valid"][:], fb[f"{lev_b}/valid"][:]
        if args.min_constituents:
            # Views of jets with fewer than N constituents contain zero-padded subjets, so
            # the subjet count itself would separate e.g. parton from hadron level.
            va &= fa[f"{lev_a}/nconst"][:] >= args.min_constituents
            vb &= fb[f"{lev_b}/nconst"][:] >= args.min_constituents
    same_file = os.path.realpath(args.a) == os.path.realpath(args.b)
    if same_file:
        # Two levels of the same jets: give each class its own, disjoint half of the
        # jets, so the network never sees two versions of one jet.
        both = np.flatnonzero(va & vb)
        both = rng.permutation(both)
        half = len(both) // 2
        n = min(half, args.max_jets) if args.max_jets else half
        ia, ib = np.sort(both[:n]), np.sort(both[half:half + n])
    else:
        ia_all, ib_all = np.flatnonzero(va), np.flatnonzero(vb)
        n = min(len(ia_all), len(ib_all))
        n = min(n, args.max_jets) if args.max_jets else n
        ia = np.sort(rng.choice(ia_all, n, replace=False))
        ib = np.sort(rng.choice(ib_all, n, replace=False))
    with h5py.File(args.a, "r") as fa:
        q_table = _q_table(fa[f"{lev_a}/kt/qN"][ia], nums)
    views = args.views or rc.view_names(nums)
    default_name = f"{os.path.basename(args.b)}:{lev_b} vs {os.path.basename(args.a)}:{lev_a}"
    result = {
        "name": args.name or default_name,
        "a": args.a, "b": args.b, "level_a": lev_a, "level_b": lev_b, "algorithm": args.algorithm,
        "n_jets_per_class": int(n), "min_constituents": args.min_constituents, "seeds": args.seeds,
        "q_kt": q_table, "auc": {},
    }
    for view in views:
        xa, ma = _features(args.a, lev_a, args.algorithm, view, ia)
        xb, mb = _features(args.b, lev_b, args.algorithm, view, ib)
        width = max(xa.shape[1], xb.shape[1])  # levels store different numbers of constituents
        xa, ma = _pad_width(xa, ma, width)
        xb, mb = _pad_width(xb, mb, width)
        t0 = time.time()
        aucs = [_train_eval(xa, ma, xb, mb, s, args) for s in range(args.seeds)]
        result["auc"][view] = {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)), "values": aucs}
        print(f"  {view}: AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} ({time.time() - t0:.0f} s)", file=sys.stderr)
    _dump(result, args.out)


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------


def cmd_scaling(args):
    with h5py.File(args.views, "r") as f:
        nums = [int(n) for n in f["meta"].attrs["nums"]]
        qgrid = f["meta/qgrid"][:]
        sel = f[f"{args.level}/valid"][:] & (f[f"{args.level}/nconst"][:] >= args.min_constituents)
        nexcl = f[f"{args.level}/kt/nexcl"][:][sel].astype(float)
        qN = f[f"{args.level}/kt/qN"][:][sel]
    if len(nexcl) == 0:
        sys.exit("no jets selected")
    mean_n = nexcl.mean(0)
    sem_n = nexcl.std(0, ddof=1) / np.sqrt(len(nexcl))
    ok = mean_n > 0
    gamma = np.full(len(qgrid), np.nan)
    gamma[ok] = -np.gradient(np.log(mean_n[ok]), np.log(qgrid[ok]))
    table = _q_table(qN, nums)
    med = [table[f"N{n}"]["median"] or 0.0 for n in nums]
    slopes = {
        f"N{nums[i]}->N{nums[i + 1]}": float(np.log(nums[i + 1] / nums[i]) / np.log(med[i] / med[i + 1]))
        for i in range(len(nums) - 1)
        if med[i + 1] > 0 and med[i] > med[i + 1]
    }
    result = {
        "views_file": args.views, "level": args.level, "min_constituents": args.min_constituents,
        "n_jets": int(len(nexcl)), "qgrid": qgrid.tolist(), "mean_nexcl": mean_n.tolist(),
        "sem_nexcl": sem_n.tolist(), "local_exponent": [None if np.isnan(g) else float(g) for g in gamma],
        "q_kt": table, "exponent_from_medians": slopes,
    }
    _dump(result, args.out)
    for k, v in slopes.items():
        print(f"  {k}: d ln N / d ln(1/q_N) = {v:.3f}", file=sys.stderr)


def _dump(obj, path):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=1)
    print(f"wrote {path}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("paired", help="per-jet EMD between two levels, per view")
    p.add_argument("--views", required=True)
    p.add_argument("--pair", nargs=2, action="append", metavar=("FROM", "TO"), required=True,
                   help="levels to compare, e.g. --pair parton gen --pair gen reco")
    p.add_argument("--algorithms", nargs="+", default=["kt"], choices=rc.ALGORITHMS)
    p.add_argument("--max-jets", type=int, default=50000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)

    p = sub.add_parser("classify", help="per-view separability of two samples")
    p.add_argument("--a", required=True, help="views file of sample A (label 0)")
    p.add_argument("--b", required=True, help="views file of sample B (label 1)")
    p.add_argument("--level", default="reco", choices=rc.LEVELS, help="level used for both samples")
    p.add_argument("--level-a", default=None, choices=rc.LEVELS, help="override the level of sample A")
    p.add_argument("--level-b", default=None, choices=rc.LEVELS,
                   help="override the level of sample B; with --a = --b this compares two levels "
                        "of the same jets (e.g. parton vs gen), using disjoint halves of the jets")
    p.add_argument("--algorithm", default="kt", choices=rc.ALGORITHMS)
    p.add_argument("--views", nargs="+", default=None, help="subset of views, e.g. N2 N8 uncl")
    p.add_argument("--name", default=None)
    p.add_argument("--max-jets", type=int, default=100000, help="jets per sample")
    p.add_argument("--min-constituents", type=int, default=0,
                   help="keep jets with at least this many constituents at the compared level(s); "
                        "use 17 so that every N <= 16 view has N real subjets")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)

    p = sub.add_parser("scaling", help="<N>(q) and the merge-scale table")
    p.add_argument("--views", required=True)
    p.add_argument("--level", default="reco", choices=rc.LEVELS)
    p.add_argument("--min-constituents", type=int, default=17, help="the paper's Table 8 uses jets with > 16")
    p.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    {"paired": cmd_paired, "classify": cmd_classify, "scaling": cmd_scaling}[args.cmd](args)


if __name__ == "__main__":
    main()
