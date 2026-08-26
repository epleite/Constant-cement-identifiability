# Constant-cement identifiability and experimental design

Reproducibility repository for the manuscript:

> *Identifiable trajectory coordinates and experimental design in the
> constant-cement model*

The repository contains the exact compact Volve inputs, frozen publication
outputs, source code, independent verification scripts, and the additional
scientific audits supporting the manuscript. The central question is whether
static elastic observations identify cement volume and coordination number
separately, or primarily a compensated constant-cement trajectory coordinate.

## Scientific result in one paragraph

For two related Hugin trajectories, static `Vp`, `Vs`, and density identify a
stable finite coordinate combining `Vcem` and `Cn` more reliably than the two
nominal microstructural parameters separately. Sharing either that coordinate
or `Cn` across trajectories does not generate new sensitivity. A prospective
multi-pressure scenario produces strong ridge contraction only when nominal
and pressure-responsive fabric are linked. Once fabric freedom and
target-aligned model discrepancy are admitted, local information gains remain
small and the fully non-linear ridge stays open over the tested domain.

## Repository layout

- `data/`: compact public Volve trajectories and selected Hugin subsets;
- `experiments/E1_discover_explain/`: local sensitivity geometry, physical
  decomposition, and finite coordinate;
- `experiments/E2_stability_hierarchy/`: moving-block bootstrap,
  leave-one-trajectory-out transport, and shared-coordinate versus shared-`Cn`
  comparison;
- `experiments/E3_break_design/`: pressure no-go controls and prospective
  nuisance-adjusted multi-pressure design;
- `experiments/E4_nonlinear_ridge_audit/`: fully non-linear nuisance profiles,
  pointwise information, multistart, and contour refinement;
- `experiments/E5_prior_scale_audit/`: grouped and one-at-a-time nuisance-prior
  scale sweeps;
- `analysis_checks/qstar_finite_form/`: matched-tangent finite-coordinate and
  analytic--numerical slope self-consistency tests;
- `docs/DATA_PROVENANCE.md`: source, columns, units, intervals and selection;
- `docs/REPRODUCIBILITY.md`: environments, run modes and expected runtimes;
- `docs/RESULTS_INDEX.md`: manuscript claims mapped to machine-readable files;
- `MANIFEST.csv` and `SHA256SUMS.txt`: inventory and integrity records.

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/run_all.py --verify-only
```

The verification-only mode checks the frozen publication outputs without
rerunning the expensive bootstrap and non-linear optimizations.

For a reduced execution smoke test that does not overwrite publication
outputs:

```bash
python scripts/run_all.py --quick
```

For complete regeneration followed by verification:

```bash
python scripts/run_all.py
```

The full calculation can take several minutes to hours depending on hardware.
See `docs/REPRODUCIBILITY.md` for individual experiment commands.

## Reproducibility contract

1. Frozen outputs are versioned and hashed.
2. Verification scripts compare physical identities, reference values,
   bootstrap counts, pathwise ranges and contour classifications.
3. Quick mode executes in temporary copies and cannot replace publication
   outputs.
4. E1--E3 contain vendored upstream numerical dependencies so their results do
   not depend on an external RPIA installation.
5. E4, E5 and the finite-coordinate checks resolve dependencies only through
   repository-relative paths.

## Scope and limitations

The 15/9-19A and 15/9-19BT2 records are related borehole trajectories. Their
two-way use tests internal transport within the Hugin setting; it is not an
independent multiwell population validation. The pressure experiment is a
prospective design calculation based on a scenario model, not calibration to
observed pressure-dependent Hugin data. The bootstrap intervals are
conditional on the specified within-trajectory block resampling and the small
number of effective block-length units.

## Data and licences

Code authored for this study is released under the MIT License. The compact
Volve-derived tables are not relicensed under MIT and retain the source-data
conditions imposed by Equinor; see `DATA_LICENSE.md` and
`docs/DATA_PROVENANCE.md`. The source release must be acknowledged in derived
work.

## Citation and archival release

Version `1.0.0` is the submission release. Metadata are supplied in
`CITATION.cff` and `.zenodo.json`. After creating the GitHub release, archive
that exact release in Zenodo and replace the placeholder DOI fields before the
journal upload.

## Maintainer

Emilson Pereira Leite, Institute of Geosciences, University of Campinas
(UNICAMP), Brazil — `emilson@unicamp.br`.
