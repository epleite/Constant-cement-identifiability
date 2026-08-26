# q-star finite-form and endpoint-slope audit

This self-contained review test addresses two manuscript criticisms without editing
the manuscript:

1. whether the finite factorized coordinate is uniquely supported relative to
   matched-tangent alternatives and a numerically integrated local slope; and
2. whether the analytic bulk/shear contact exponents close the physical link to
   the empirical slope near 20.3 and to the bootstrap distributions of `A` and
   `Gamma`.

The script reads the frozen E1 and E2 packages in this repository. It does not
alter those packages.

Run from the repository root:

```bash
python analysis_checks/qstar_finite_form/run_analysis.py
python analysis_checks/qstar_finite_form/verify.py
```

Outputs are written only below `analysis_checks/qstar_finite_form/results/`. The primary
scientific interpretation is in `REPORT.md`.
