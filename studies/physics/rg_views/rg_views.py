#!/usr/bin/env python3
"""Build RINO-style multi-scale views of jets at detector, generator and parton level.

Reads makeNtuplesRG.C output (or public JetClass files, detector level only) and
writes one HDF5 file with, for every selected jet:

  /jet/{pt,eta,phi,energy,cls,gen_match,event_no}   detector-level jet, class 0=QCD 1=top
  /<level>/valid          (n,)        level available for this jet
  /<level>/p4             (n, 4)      level jet four-vector (sum of its constituents)
  /<level>/nconst         (n,)        number of constituents
  /<level>/uncl           (n, M, 4)   constituents [px, py, pz, E], pT-ordered, zero padded
  /<level>/<algo>/N<n>    (n, n, 4)   exclusive-<algo> subjets at N = n, pT-ordered
  /<level>/kt/qN          (n, len(N)) kT merge scale sqrt(d_merge) of the (N+1)->N step [GeV]
  /<level>/kt/nexcl       (n, len(q)) number of exclusive kT jets at d_cut = q^2, q on /meta/qgrid

Levels: reco = Delphes E-flow constituents (as in JetClass, 90 hardest kept as in
RINO), gen = stable particles of the matched GenFatJet, parton = partons entering
hadronization clustered with anti-kT R = 0.8 and matched to the jet.

Example:
  python rg_views.py --input 'out/Pythia/nominal/ZJetsToNuNu/run_*/output_*.root' \\
      --output views/qcd_nominal.h5 --label qcd --levels reco gen parton --algorithms kt ca antikt
"""

from __future__ import annotations

import argparse
import sys
import time

import awkward as ak
import h5py
import numpy as np
import uproot

import rg_common as rc

MAX_CONST = {"reco": 90, "gen": 160, "parton": 160}


class H5Appender:
    """Append rows to resizable datasets."""

    def __init__(self, path):
        self.f = h5py.File(path, "w")
        self.n = 0

    def append(self, name, arr):
        arr = np.asarray(arr)
        if name not in self.f:
            maxshape = (None,) + arr.shape[1:]
            chunks = (min(max(len(arr), 1), 1024),) + arr.shape[1:]
            self.f.create_dataset(name, data=arr, maxshape=maxshape, chunks=chunks, compression="gzip", compression_opts=4)
        else:
            ds = self.f[name]
            old = ds.shape[0]
            ds.resize(old + len(arr), axis=0)
            ds[old:] = arr

    def close(self):
        self.f.close()


def branches_for(levels, fields):
    want = ["jet_pt", "jet_eta", "jet_phi", "jet_energy"]
    for b in ("gen_match", "label_QCD", "label_Tbqq", "event_no", "genjet_dr", "genjet_eta", "genjet_phi"):
        if b in fields:
            want.append(b)
    for level in levels:
        pre = rc.LEVEL_PREFIX[level]
        for c in ("px", "py", "pz", "energy"):
            if pre + c not in fields:
                raise KeyError(
                    f"level '{level}' needs branch {pre + c}, which is not in the input "
                    "(gen and parton levels need makeNtuplesRG.C output)"
                )
            want.append(pre + c)
    return want


def process_chunk(arrays, args, out, qgrid):
    cls = rc.jet_class(arrays)
    pt = ak.to_numpy(arrays["jet_pt"])
    sel = (pt > args.pt_min) & (pt < args.pt_max) & (np.abs(ak.to_numpy(arrays["jet_eta"])) < 2.0)
    if args.label != "all":
        sel &= cls == rc.CLASS_CODES[args.label]
    if not sel.any():
        return 0
    arrays = arrays[sel]
    cls = cls[sel]
    n = len(arrays)

    out.append("jet/pt", ak.to_numpy(arrays["jet_pt"]).astype(np.float32))
    out.append("jet/eta", ak.to_numpy(arrays["jet_eta"]).astype(np.float32))
    out.append("jet/phi", ak.to_numpy(arrays["jet_phi"]).astype(np.float32))
    out.append("jet/energy", ak.to_numpy(arrays["jet_energy"]).astype(np.float32))
    out.append("jet/cls", cls.astype(np.int8))
    gm = ak.to_numpy(arrays["gen_match"]).astype(np.int16) if "gen_match" in arrays.fields else np.full(n, -1, np.int16)
    out.append("jet/gen_match", gm)
    ev = ak.to_numpy(arrays["event_no"]).astype(np.int64) if "event_no" in arrays.fields else np.full(n, -1, np.int64)
    out.append("jet/event_no", ev)

    has_gen = ak.to_numpy(arrays["genjet_dr"]) >= 0 if "genjet_dr" in arrays.fields else np.zeros(n, bool)

    for level in args.levels:
        parts = rc.level_p4(arrays, level)
        if level == "parton":
            use_gen = has_gen
            axis_eta = np.where(use_gen, ak.to_numpy(arrays["genjet_eta"]), ak.to_numpy(arrays["jet_eta"]))
            axis_phi = np.where(use_gen, ak.to_numpy(arrays["genjet_phi"]), ak.to_numpy(arrays["jet_phi"]))
            parts, valid = rc.parton_jet_constituents(parts, axis_eta, axis_phi)
        else:
            parts = rc.sort_by_pt(parts)
            valid = ak.to_numpy(ak.num(parts, axis=1)) > 0
            if level == "gen":
                valid &= has_gen
        nconst = ak.to_numpy(ak.num(parts, axis=1))
        if level == "reco":
            parts = parts[:, : MAX_CONST["reco"]]  # RINO keeps the 90 hardest particles
        out.append(f"{level}/valid", valid)
        out.append(f"{level}/nconst", nconst.astype(np.int16))
        out.append(f"{level}/p4", rc.sum_p4(parts).astype(np.float32))
        out.append(f"{level}/uncl", rc.to_numpy_p4(parts, args.max_const or MAX_CONST[level]))
        for algo in args.algorithms:
            views = rc.make_views(parts, algo, args.nums, qgrid if algo == "kt" else None)
            for key, arr in views.items():
                out.append(f"{level}/{algo}/{key}", arr)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="ROOT files or glob patterns")
    ap.add_argument("--output", required=True, help="output HDF5 file")
    ap.add_argument("--tree", default="tree")
    ap.add_argument("--label", choices=["qcd", "top", "all"], default="all", help="jet class to keep")
    ap.add_argument("--levels", nargs="+", choices=rc.LEVELS, default=["reco"])
    ap.add_argument("--algorithms", nargs="+", choices=rc.ALGORITHMS, default=["kt"])
    ap.add_argument("--nums", nargs="+", type=int, default=list(rc.DEFAULT_NUMS), help="subjet multiplicities N")
    ap.add_argument("--pt-min", type=float, default=500.0, help="JetClass keeps 500 < pT < 1000 GeV")
    ap.add_argument("--pt-max", type=float, default=1000.0)
    ap.add_argument("--max-const", type=int, default=None, help="override constituents stored in /<level>/uncl")
    ap.add_argument("--max-jets", type=int, default=None)
    ap.add_argument("--step-size", type=int, default=20000, help="jets read per chunk")
    ap.add_argument("--qgrid", nargs=3, type=float, default=[0.1, 200.0, 41], metavar=("QMIN", "QMAX", "NQ"),
                    help="log grid of kT resolutions q [GeV] for the exclusive-jet counts")
    args = ap.parse_args(argv)
    if "kt" not in args.algorithms:
        args.algorithms = ["kt"] + args.algorithms  # merge scales define the physical scale axis

    files = rc.expand_inputs(args.input)
    with uproot.open(f"{files[0]}:{args.tree}") as t:
        fields = set(t.keys())
    branches = branches_for(args.levels, fields)
    qgrid = np.geomspace(args.qgrid[0], args.qgrid[1], int(args.qgrid[2]))

    out = H5Appender(args.output)
    total, t0 = 0, time.time()
    for arrays in uproot.iterate([f"{f}:{args.tree}" for f in files], branches, step_size=args.step_size, library="ak"):
        if args.max_jets is not None:
            arrays = arrays[: max(0, args.max_jets - total)]
            if len(arrays) == 0:
                break
        total += process_chunk(arrays, args, out, qgrid)
        print(f"  {total} jets written ({time.time() - t0:.0f} s)", file=sys.stderr)
        if args.max_jets is not None and total >= args.max_jets:
            break

    meta = out.f.require_group("meta")
    meta.attrs["nums"] = np.array(args.nums)
    meta.attrs["levels"] = np.array(args.levels, dtype="S")
    meta.attrs["algorithms"] = np.array(args.algorithms, dtype="S")
    meta.attrs["label"] = args.label
    meta.attrs["R"] = rc.R_JET
    meta.attrs["inputs"] = np.array(files, dtype="S")
    meta.create_dataset("qgrid", data=qgrid)
    out.close()
    print(f"wrote {total} jets to {args.output}", file=sys.stderr)
    if total == 0:
        sys.exit("no jets passed the selection")


if __name__ == "__main__":
    main()
