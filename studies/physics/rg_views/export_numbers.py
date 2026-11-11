#!/usr/bin/env python3
"""Export the classifier AUCs (mean, spread and per-seed values) and the kT scales as CSV,
for rebuilding the figures in the paper.

  python studies/physics/rg_views/export_numbers.py OUTDIR DATADIR

DATADIR holds the study outputs: results/ (classifier JSON files from rg_metrics.py) and
views/qcd_nominal.h5 (from rg_views.py). Writes OUTDIR/auc_numbers.csv, OUTDIR/kt_scales.csv
and copies the source JSON files to OUTDIR/json/.
"""
import csv
import json
import os
import shutil
import sys

import h5py
import numpy as np

D = sys.argv[2] if len(sys.argv) > 2 else "."
R = f"{D}/results"

# (key, figures that use the row, plain-language comparison, path)
COMPARISONS = [
    ("top_vs_qcd", "rg_classify_top_qcd_and_hadronization_kt; rg_classify_all_comparisons_detector", "top vs QCD", f"{R}/auc_top_qcd.json"),
    ("hadronization_kt", "rg_classify_top_qcd_and_hadronization_kt; rg_classify_hadronization_by_algorithm; rg_classify_all_comparisons_detector", "same jets before vs after hadronization", f"{R}/auc_parton_hadron.json"),
    ("hadronization_ca", "rg_classify_hadronization_by_algorithm", "same jets before vs after hadronization", f"{R}/algo/auc_parton_hadron_ca.json"),
    ("hadronization_antikt", "rg_classify_hadronization_by_algorithm", "same jets before vs after hadronization", f"{R}/algo/auc_parton_hadron_antikt.json"),
    ("detector_kt", "rg_classify_detector_by_algorithm; rg_classify_all_comparisons_detector", "same jets at hadron vs detector level", f"{R}/auc_gen_reco.json"),
    ("detector_ca", "rg_classify_detector_by_algorithm", "same jets at hadron vs detector level", f"{R}/algo/auc_gen_reco_ca.json"),
    ("detector_antikt", "rg_classify_detector_by_algorithm", "same jets at hadron vs detector level", f"{R}/algo/auc_gen_reco_antikt.json"),
    ("cutoff_detector", "rg_classify_all_comparisons_detector", "shower cutoff 2 vs 0.5 GeV", f"{R}/auc_pTmin.json"),
    ("coupling_detector", "rg_classify_all_comparisons_detector", "shower coupling, muR x2 vs nominal", f"{R}/vs_nominal/auc_muR_x2.json"),
    ("aLund_detector", "rg_classify_all_comparisons_detector", "fragmentation aLund 0.9 vs 0.5", f"{R}/auc_aLund.json"),
    ("sigmaPT_detector", "rg_classify_all_comparisons_detector", "fragmentation sigmaPT 0.45 vs 0.25 GeV", f"{R}/auc_sigmaPT.json"),
    ("cutoff_generator", "rg_classify_variations_generator", "shower cutoff 2 vs 0.5 GeV", f"{R}/checks/auc_pTmin_gen.json"),
    ("coupling_generator", "rg_classify_variations_generator", "shower coupling, muR x2 vs nominal", f"{R}/checks/auc_muR_x2_gen.json"),
    ("aLund_generator", "rg_classify_variations_generator", "fragmentation aLund 0.9 vs 0.5", f"{R}/checks/auc_aLund_gen.json"),
    ("sigmaPT_generator", "rg_classify_variations_generator", "fragmentation sigmaPT 0.45 vs 0.25 GeV", f"{R}/checks/auc_sigmaPT_gen.json"),
]
LEVEL = {"reco": "detector", "gen": "hadron", "parton": "parton"}


def write_auc(outdir):
    os.makedirs(f"{outdir}/json", exist_ok=True)
    rows = []
    for key, where, what, path in COMPARISONS:
        d = json.load(open(path))
        shutil.copy(path, f"{outdir}/json/{key}.json")
        for view, v in d["auc"].items():
            vals = list(v["values"])
            rows.append({
                "key": key, "figures": where, "comparison": what,
                "sample_a": os.path.basename(d["a"]), "level_a": LEVEL[d["level_a"]],
                "sample_b": os.path.basename(d["b"]), "level_b": LEVEL[d["level_b"]],
                "algorithm": d.get("algorithm", "kt"), "min_constituents": d["min_constituents"],
                "n_jets_per_class": d["n_jets_per_class"],
                "view": "all particles" if view == "uncl" else view,
                "auc_mean": f"{np.mean(vals):.4f}", "auc_std_pop": f"{np.std(vals):.4f}",
                "auc_std_sample": f"{np.std(vals, ddof=1):.4f}",
                **{f"auc_seed{i + 1}": f"{x:.4f}" for i, x in enumerate(vals)},
            })
    with open(f"{outdir}/auc_numbers.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {outdir}/auc_numbers.csv ({len(rows)} rows)")


def write_scales(outdir):
    f = h5py.File(f"{D}/views/qcd_nominal.h5", "r")
    nums = [int(n) for n in f["meta"].attrs["nums"]]
    lev = {l: dict(valid=f[f"{l}/valid"][:], n=f[f"{l}/nconst"][:], q=f[f"{l}/kt/qN"][:]) for l in ("reco", "gen", "parton")}
    ok = {l: lev[l]["valid"] & (lev[l]["n"] >= 17) for l in lev}
    selections = [
        ("QCD jets with >= 17 detector-level constituents (as Table 8; the top-vs-QCD curve)", ok["reco"], ["reco"]),
        ("QCD jets with >= 17 hadrons", ok["gen"], ["gen"]),
        ("hadronization test: QCD jets with >= 17 partons and >= 17 hadrons", ok["parton"] & ok["gen"], ["parton", "gen"]),
        ("detector test: QCD jets with >= 17 hadrons and >= 17 detector-level constituents", ok["gen"] & ok["reco"], ["gen", "reco"]),
    ]
    total = len(lev["reco"]["valid"])
    rows = []
    for name, mask, levels in selections:
        for l in levels:
            q = lev[l]["q"][mask]
            for i, n in enumerate(nums):
                rows.append({"selection": name, "n_jets": int(mask.sum()), "fraction_of_qcd_jets": f"{mask.mean():.3f}",
                             "level": LEVEL[l], "N": n, "q_median_GeV": f"{np.median(q[:, i]):.3f}",
                             "q_p25_GeV": f"{np.percentile(q[:, i], 25):.3f}", "q_p75_GeV": f"{np.percentile(q[:, i], 75):.3f}"})
    with open(f"{outdir}/kt_scales.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {outdir}/kt_scales.csv ({len(rows)} rows; {total} QCD jets in total)")


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    write_auc(out)
    write_scales(out)
