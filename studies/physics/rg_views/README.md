# What the kT views keep and what they forget

Scripts and numbers behind the appendix on the renormalization-group link
(ICLR 2027 rebuttal, Reviewer ANRi). The study asks, view by view, which
information survives exclusive-kT coarse-graining. It tests the views
themselves; it does not use a RINO checkpoint.

Run every command from the repository root.

## Results in short

Each number is the test AUC of a small DeepSets network that separates two
samples using one view alone. 0.5 means the view carries no trace of the
difference. Full numbers, with the three seed values, are in `numbers/`.

| | N = 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| top vs QCD, detector level | 0.91 | 0.96 | 0.96 | 0.96 |
| same jets, before vs after hadronization, kT | 0.51 | 0.56 | 0.67 | 0.83 |
| same, Cambridge/Aachen | 0.71 | 0.84 | 0.96 | 1.00 |
| same, anti-kT | 0.90 | 0.98 | 1.00 | 1.00 |
| same jets, hadron vs detector level, kT | 0.56 | 0.61 | 0.68 | 0.81 |
| same, Cambridge/Aachen | 0.82 | 0.96 | 1.00 | 1.00 |
| same, anti-kT | 0.94 | 0.99 | 1.00 | 1.00 |

- **Coarse kT views keep what separates top from QCD** and carry little
  hadronization or detector detail.
- **This is specific to kT.** The order kT, C/A, anti-kT is also the order of
  the transfer accuracy when RINO is trained on each algorithm's views.

## Files

| File | Purpose |
|---|---|
| `rg_views.py` | Builds RINO-style views (exclusive kT, C/A or anti-kT; N = 2, 3, 4, 6, 8, 16, plus all particles) at parton, hadron and detector level; writes HDF5 |
| `rg_metrics.py` | `classify` (per-view AUC), `paired` (per-jet comparison of two levels), `scaling` (number of subjets against resolution) |
| `rg_plot.py` | Figures from the metric JSON files |
| `export_numbers.py` | Writes `numbers/auc_numbers.csv` and `numbers/kt_scales.csv` |
| `to_jetclass.py` | Converts the study ntuples to the public JetClass format read by the RINO dataloader |
| `rg_common.py` | Shared code; the view construction follows `dino/preprocess/jetclass/cluster.py` |
| `numbers/` | The AUCs (mean, spread and per-seed values), the kT scales for each jet selection, and the raw classifier outputs |

## Inputs: the samples

The samples are not in this repository. They are regenerated with the JetClass
recipe plus two additions: each jet is kept at parton, hadron and detector
level, and Pythia settings can be varied.

- **Generation code:** the submodule `external/jetclass_generation`, a fork of
  `jet-universe/jetclass_generation` on branch `rg-study`. Fetch it with
  `git submodule update --init`. Its `rg_study/README.md` has the commands,
  including a Docker runner for one machine.
- **Versions used:** MadGraph5_aMC@NLO 3.5.8, Pythia 8.31 and Delphes 3.5.0,
  with gridpacks rebuilt from the JetClass cards. JetClass itself used
  MadGraph 3.1.1.
- **Size used:** 195k QCD jets (Z(νν)+jets) and 96k top jets with
  500 < pT < 1000 GeV, plus about 100k QCD jets per Pythia variation.

## Requirements

`pip install -r studies/physics/rg_views/requirements.txt` on top of the `rino`
environment. `wasserstein` is optional; there is an exact `scipy` fallback,
which is slower for the all-particles view.

## Commands

```bash
OUT=/path/to/out/Pythia
# 1. views (kT always included; it defines the q_N scale axis)
python studies/physics/rg_views/rg_views.py --input "$OUT/nominal/ZJetsToNuNu/run_*/output_*.root" --output views/qcd_nominal.h5 \
    --label qcd --levels reco gen parton --algorithms kt ca antikt
python studies/physics/rg_views/rg_views.py --input "$OUT/nominal/TTBar/run_*/output_*.root" --output views/top_nominal.h5 \
    --label top --levels reco gen
for v in fsr_pTmin_2p0 fsr_muR_x0p5 fsr_muR_x2 frag_aLund_0p5 frag_aLund_0p9; do
  python studies/physics/rg_views/rg_views.py --input "$OUT/$v/ZJetsToNuNu/run_*/output_*.root" --output views/qcd_$v.h5 \
      --label qcd --levels reco gen --algorithms kt ca antikt
done

# 2. per-view information: AUC of a small DeepSets network using one view at a time
python studies/physics/rg_views/rg_metrics.py classify --a views/qcd_nominal.h5 --b views/top_nominal.h5 --level reco --min-constituents 17 \
    --name "top vs QCD (relevant)" --out results/auc_top_qcd.json
python studies/physics/rg_views/rg_metrics.py classify --a views/qcd_nominal.h5 --level-a parton --b views/qcd_nominal.h5 --level-b gen \
    --min-constituents 17 --name "parton vs hadron" --out results/auc_parton_hadron.json
python studies/physics/rg_views/rg_metrics.py classify --a views/qcd_nominal.h5 --level-a gen --b views/qcd_nominal.h5 --level-b reco \
    --min-constituents 17 --name "hadron vs detector" --out results/auc_gen_reco.json
python studies/physics/rg_views/rg_metrics.py classify --a views/qcd_nominal.h5 --b views/qcd_fsr_pTmin_2p0.h5 --level reco --min-constituents 17 \
    --name "FSR pTmin 2 vs 0.5 GeV" --out results/auc_pTmin.json
# mu_R: compare x2 with nominal (the x1/2 sample also has a raised shower cutoff, see below)
python studies/physics/rg_views/rg_metrics.py classify --a views/qcd_nominal.h5 --b views/qcd_fsr_muR_x2.h5 --level reco --min-constituents 17 \
    --name "FSR mu_R x2 vs x1" --out results/auc_muR_x2.json
python studies/physics/rg_views/rg_metrics.py classify --a views/qcd_frag_aLund_0p5.h5 --b views/qcd_frag_aLund_0p9.h5 --level reco --min-constituents 17 \
    --name "aLund 0.9 vs 0.5" --out results/auc_aLund.json
# clustering-algorithm comparison: repeat the two same-jet tests with --algorithm ca / antikt
for a in ca antikt; do
  python studies/physics/rg_views/rg_metrics.py classify --a views/qcd_nominal.h5 --level-a parton --b views/qcd_nominal.h5 --level-b gen \
      --min-constituents 17 --algorithm $a --name "parton vs hadron ($a)" --out results/algo/auc_parton_hadron_$a.json
  python studies/physics/rg_views/rg_metrics.py classify --a views/qcd_nominal.h5 --level-a gen --b views/qcd_nominal.h5 --level-b reco \
      --min-constituents 17 --algorithm $a --name "hadron vs detector ($a)" --out results/algo/auc_gen_reco_$a.json
done

# 3. same-jet comparisons and the scale axis
python studies/physics/rg_views/rg_metrics.py paired --views views/qcd_nominal.h5 --pair parton gen --pair gen reco \
    --algorithms kt ca antikt --out results/paired_qcd_nominal.json
python studies/physics/rg_views/rg_metrics.py scaling --views views/qcd_nominal.h5 --level reco --out results/scaling_reco.json

# 4. figures (--tag names the classify/scaling figure; --scaling overlays several results)
python studies/physics/rg_views/rg_plot.py --classify results/auc_*.json --tag kt --formats pdf png --outdir plots
python studies/physics/rg_views/rg_plot.py --classify results/auc_pTmin.json results/algo/auc_pTmin_{ca,antikt}.json \
    results/auc_parton_hadron.json results/algo/auc_parton_hadron_{ca,antikt}.json --tag algorithms --outdir plots
python studies/physics/rg_views/rg_plot.py --paired results/paired_qcd_nominal.json --outdir plots
python studies/physics/rg_views/rg_plot.py --scaling results/scaling_gen.json results/scaling_pTmin2_gen.json \
    --labels nominal "pTmin 2 GeV" --tag gen_pTmin --outdir plots
# --ratio adds a panel with each <N_excl>(q) divided by the first (reference) result
python studies/physics/rg_views/rg_plot.py --scaling results/scaling_gen.json results/scaling_fsr_muR_x{0p5,2}_gen.json \
    results/scaling_pTmin2_gen.json --labels nominal "muR x1/2" "muR x2" "pTmin 2 GeV" \
    --ratio --tag gen_variations --outdir plots
# slide versions:
#   --ratio-only   draws only the ratio panel, with ±1σ statistical bands
#   --labels       renames classify curves when no --scaling is given
#   --no-q-axis    drops the q_N axis, which would otherwise show parton-level scales
python studies/physics/rg_views/rg_plot.py --scaling results/scaling_gen.json results/scaling_fsr_muR_x2_gen.json \
    results/scaling_pTmin2_gen.json --labels nominal "muR x2" "pTmin 2 GeV" --ratio-only --tag slides --outdir plots
python studies/physics/rg_views/rg_plot.py --classify results/auc_parton_hadron.json results/algo/auc_parton_hadron_{ca,antikt}.json \
    --labels "kT" "C/A" "anti-kT" --no-q-axis --tag hadronization_algorithms --outdir plots
# --reach REF COUPLING CUTOFF HADRON...: the subjet count relative to REF on an inverted q axis
# (coarse on the left, like the classify figures), with the cutoff and hadronization scales marked,
# the region above the matching scale shaded, and the views' median scales on the top axis
python studies/physics/rg_views/rg_plot.py --reach results/scaling_gen.json results/scaling_fsr_muR_x2_gen.json results/scaling_pTmin2_gen.json \
    results/scaling_frag_{aLund_0p5,aLund_0p9,sigmaPT_0p25,sigmaPT_0p45}_gen.json --tag result3 --outdir plots
```

The top axis of a classify figure shows the median kT merge scale of sample A in
the first file, so list a detector-level comparison first.

What to look for:
- **`classify`.** If coarse-graining acts like an RG step, deformations confined
  to low scales (shower cutoff, fragmentation, hadronization, detector) give
  AUC near 0.5 in the coarse views. They become separable only in views whose
  merge scale q_N reaches their scale. Top vs QCD (relevant information) should
  stay separable down to N = 3. The mu_R variation changes the evolution itself
  and should stay visible at all N.
  - The N = 2 view needs no padding, so for it `--min-constituents 3 --views N2`
    avoids the selection effect of the 17-constituent cut.
  - That cut removes different fractions of jets from each sample. In stage 2 it
    lowered the mu_R x2-vs-nominal N = 2 AUC from 0.529 to 0.507 at detector
    level.
- **`scaling`.** Plot each variation's ⟨N_excl⟩(q) with `--ratio`, relative to
  nominal, at generator level.
  - A coupling change (mu_R x2) lowers the count at every q below the matching
    scale: by 3.9% at 20 GeV and 8% at 1 GeV.
  - Infrared changes (pTmin, aLund) leave it unchanged above roughly 10–15 GeV.
  - Detector-level curves carry the bias of the constituent cut.
  - The ratio approaches 1 at high q because nearly every jet is one subjet
    there; ⟨N_excl⟩ − 1 shows the coupling effect without that dilution.
- **Two limits of the mu_R variation.**
  - `renormMultFac` sets mu_R² = factor × pT² per emission; it is not a fixed
    scale.
  - `fsr_muR_x0p5` also raises the shower cutoff: Pythia prints "pTmin too
    low, raised to 0.715", because mu_R at the 0.5 GeV cutoff would fall below
    Lambda_QCD. Use `fsr_muR_x2` against nominal for a clean comparison.
  - The Z+jets gridpack is MLM-matched (xqcut 30 GeV, qCut 45 GeV). Above
    qCut the hardest splittings come from the matrix element, which the shower
    variation does not change.
- **`--min-constituents 17`.** Use it in every comparison. Views of jets with
  fewer than N constituents contain zero-padded subjets, so the subjet count
  alone separates samples whose multiplicities differ. That covers parton vs
  hadron level, but also shower variations (pTmin = 2 GeV lowers the median
  detector-level multiplicity from 32 to 29) and top vs QCD. The unclustered view
  always carries the multiplicity difference.
- **`paired`.** `dq_over_q`, the per-jet relative shift of the kT merge scale,
  is the quantity for which power-correction arguments predict ~ Lambda / q_N.
  `emd_over_pt` measures absolute momentum transport between the two versions of
  each view. It need not fall with coarsening.

## Representation-level test with RINO

`to_jetclass.py` writes the samples in the public JetClass format, so the RINO
code can run its frozen-backbone probes and finetuned taggers on them, through
`configs/dataloaders/jetclass-raw/kinematics.yaml` with the paths pointed at the
converted files:

```bash
python studies/physics/rg_views/to_jetclass.py --input "$OUT/fsr_pTmin_2p0/TTBar/run_*/output_*.root" --output jetclass_rg/fsr_pTmin_2p0/TTBar_000.root
python studies/physics/rg_views/to_jetclass.py --input "$OUT/fsr_pTmin_2p0/ZJetsToNuNu/run_*/output_*.root" --output jetclass_rg/fsr_pTmin_2p0/ZJetsToNuNu_000.root
```

## Scales of the tests

The kT scale q_N follows `studies/physics/kt_scales.py`. For detector-level QCD
jets with at least 17 constituents it reproduces the paper's table to within
1% (9.95, 5.74, 3.91, 2.25, 1.49, 0.46 GeV for N = 2 to 16).

The hadronization test needs jets with at least 17 partons, which only 40% of
QCD jets have. Those jets have larger scales (16.5, 8.65, 5.80, 3.45, 2.38,
0.91 GeV at hadron level), so plot that curve against its own scales.
`numbers/kt_scales.csv` has the scales for every selection and level.
