#!/usr/bin/env python3
"""Convert makeNtuplesRG.C (or makeNtuples.C) output to the public JetClass format.

Reproduces ../postprocessing/mergeTrees.C (500 < jet_pt < 1000 GeV) and rename.py
(label_*, part_is*, aux_* branches) with uproot, so that the RINO dataloader for raw
JetClass files (configs/dataloaders/jetclass-raw/kinematics.yaml) can read the
variation samples directly, e.g. for frozen-backbone probes or finetuned-tagger
evaluation on shower/hadronization variations.

Example:
  python to_jetclass.py --input 'out/Pythia/fsr_pTmin_2p0/TTBar/run_*/output_*.root' \\
      --output jetclass_rg/fsr_pTmin_2p0/TTBar_000.root
"""

from __future__ import annotations

import argparse
import os

import awkward as ak
import numpy as np
import uproot

import rg_common as rc

PART_KEEP = ("px", "py", "pz", "energy", "deta", "dphi", "d0val", "d0err", "dzval", "dzerr", "charge")
JET_KEEP = ("jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_nparticles", "jet_sdmass",
            "jet_tau1", "jet_tau2", "jet_tau3", "jet_tau4")
RG_KEEP_SCALAR = ("event_no", "jet_no", "genjet_pt", "genjet_eta", "genjet_phi", "genjet_energy", "genjet_dr")


def convert(arrays, keep_rg):
    gm = arrays["gen_match"]
    out = {k: arrays[k] for k in JET_KEEP}
    out["label_QCD"] = 1 - arrays["is_signal"]
    out["label_Tbqq"] = (gm == 11) | (gm == 12)
    out["label_Tbl"] = (gm == 15) | (gm == 16)
    out["label_Wqq"] = (gm == 21) | (gm == 22)
    out["label_Zqq"] = (gm == 31) | (gm == 32) | (gm == 33)
    out["label_Hbb"] = gm == 41
    out["label_Hcc"] = gm == 42
    out["label_Hgg"] = gm == 44
    out["label_H4q"] = gm == 45
    out["label_Hqql"] = gm == 46
    for k in list(out):
        if k.startswith("label_"):
            out[k] = ak.values_astype(out[k], np.float32)
    out["aux_truth_match"] = gm
    for k in ("pt", "eta", "phi", "pid"):
        out[f"aux_genpart_{k}"] = arrays[f"genpart_{k}"]

    pid = np.abs(arrays["part_pid"])
    part = {k: arrays[f"part_{k}"] for k in PART_KEEP}
    part["isChargedHadron"] = ak.values_astype((pid == 211) | (pid == 321) | (pid == 2212), np.float32)
    part["isNeutralHadron"] = ak.values_astype(pid == 0, np.float32)
    part["isPhoton"] = ak.values_astype(pid == 22, np.float32)
    part["isElectron"] = ak.values_astype(pid == 11, np.float32)
    part["isMuon"] = ak.values_astype(pid == 13, np.float32)
    out["part"] = ak.zip(part)

    if keep_rg:
        for k in RG_KEEP_SCALAR:
            if k in arrays.fields:
                out[k] = arrays[k]
        for level, name in (("gen", "gen_part"), ("parton", "parton")):
            pre = rc.LEVEL_PREFIX[level]
            if pre + "px" in arrays.fields:
                out[name] = ak.zip({c: arrays[pre + c] for c in ("px", "py", "pz", "energy", "pid")})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="ROOT files or glob patterns")
    ap.add_argument("--output", required=True, help="output ROOT file")
    ap.add_argument("--keep-rg", action="store_true", help="also keep the generator- and parton-level branches")
    ap.add_argument("--step-size", type=int, default=50000)
    args = ap.parse_args(argv)

    files = rc.expand_inputs(args.input)
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n = 0
    tree = None
    with uproot.recreate(args.output) as fout:
        for arrays in uproot.iterate([f"{f}:tree" for f in files], step_size=args.step_size, library="ak"):
            arrays = arrays[(arrays["jet_pt"] > 500) & (arrays["jet_pt"] < 1000)]
            if len(arrays) == 0:
                continue
            out = convert(arrays, args.keep_rg)
            if tree is None:
                # flat JetClass-style names: part_px, part_py, ... sharing the counter npart
                tree = fout.mktree(
                    "tree",
                    {k: v.type for k, v in out.items()},
                    field_name=lambda outer, inner: f"{outer}_{inner}",
                    counter_name=lambda counted: f"n{counted}",
                )
            tree.extend(out)
            n += len(arrays)
    print(f"wrote {n} jets to {args.output}")


if __name__ == "__main__":
    main()
