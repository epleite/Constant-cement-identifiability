# Reproducibility instructions

## Environment

Python 3.12 is the frozen publication environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The root requirements contain the exact runtime versions used for the frozen
publication outputs, including the PDF reader used by the E5 verifier. The
individual `environment.yml` and `requirements.txt` files inside E1--E3 record
the same frozen numerical stack.

## Three execution modes

### Verify frozen publication outputs

```bash
python scripts/run_all.py --verify-only
```

This runs the verification suites for E1--E5 and the finite-coordinate check,
then validates every file against the root SHA256 manifest. It does not change
results.

### Safe reduced smoke test

```bash
python scripts/run_all.py --quick
```

Quick mode copies E1--E3 into a temporary directory and runs their reduced
calculations there. It tests execution without replacing the 400-replicate and
full-grid publication outputs in the repository.

### Complete regeneration

```bash
python scripts/run_all.py
```

This runs and verifies E1, E2 and E3, regenerates the finite-coordinate check,
executes the complete non-linear multistart audit and prior-scale sweeps,
rebuilds the manifest, and verifies the frozen result tree. Runtime depends
strongly on hardware because E2 contains moving-block bootstrap fits and E4
contains hundreds of bounded non-linear optimizations.

## Individual experiment commands

E1--E3 can be run independently from their directories:

```bash
cd experiments/E1_discover_explain
python scripts/run_e1.py
python scripts/verify_e1.py
```

Replace `E1`/`e1` by `E2`/`e2` or `E3`/`e3` for the other primary packages.

The finite-coordinate check is:

```bash
python analysis_checks/qstar_finite_form/run_analysis.py
python analysis_checks/qstar_finite_form/verify.py
```

The full non-linear ridge audit is sequential:

```bash
python experiments/E4_nonlinear_ridge_audit/run_nonlinear_ridge.py
python experiments/E4_nonlinear_ridge_audit/multistart_audit.py
python experiments/E4_nonlinear_ridge_audit/refine_shared_supported.py
python experiments/E4_nonlinear_ridge_audit/recompute_multistart_pointwise.py
python experiments/E4_nonlinear_ridge_audit/synchronize_outputs.py
python experiments/E4_nonlinear_ridge_audit/verify.py
```

The nuisance-prior audit is:

```bash
python experiments/E5_prior_scale_audit/scripts/run_prior_scale_sweep.py
python experiments/E5_prior_scale_audit/scripts/verify_prior_scale_sweep.py
```

## Frozen numerical ingredients

- target scales: 0.015 for cement volume and 0.20 for `ln(Cn)`;
- nominal constant-cement packing reference: `phi_c = 0.40`;
- static observations: `Vp`, `Vs` and density;
- static nuisance priors: independent unit Gaussians in the nine standardized
  coordinates listed in `DATA_PROVENANCE.md`;
- primary E2 bootstrap: 400 non-circular, trajectory-stratified moving-block
  replicates with a five-sample (20 m) block;
- primary E3 design: expanded fabric nuisances and trajectory-state model-form
  discrepancy;
- E4 profile domain: `Vcem=0.001--0.060`, `Cn=3--18`, physical nuisance
  coordinates bounded at four declared standard deviations;
- E5 primary sensitivity interval: factor-of-two perturbations of declared
  nuisance standard deviations.

## Integrity

`SHA256SUMS.txt` hashes every payload file except itself and `MANIFEST.csv`.
After extracting the archive, verify it with either:

```bash
python scripts/run_all.py --verify-only
```

or, on systems with GNU coreutils:

```bash
sha256sum -c SHA256SUMS.txt
```

## Expected interpretation of verification

Passing verification shows that the packaged code and stored outputs satisfy
the declared numerical identities and reference values. It does not validate
the constant-cement model against independent core measurements, turn two
related trajectories into a population sample, or prove that every bounded
non-linear fit is the global optimum.
