# Constant-cement experiment E3: Break and experimental design

This package tests whether multi-pressure elastic observations can break the
local ambiguity between cement volume, `Vcem`, and coordination number, `Cn`,
in the frozen RPIA constant-cement Scheme 1 model.

## Main result

The frozen constant-cement implementation is exactly pressure blind. Repeated
states can improve precision, but pressure labels alone cannot create a new
constitutive sensitivity direction.

A prospective Avseth–Skjei-inspired bounding-average extension was therefore
used as an experimental-design scenario model. It is anchored to the frozen
constant-cement prediction at the RPIA effective pressure of 39 MPa. The key
result is an ablation of the fabric links inside that extension:

| Fabric treatment | Best added states (MPa) | Gain in adjusted lambda-min |
|---|---:|---:|
| Shared compliant-contact `Cn` | 5 + 7.5 | 633.8× |
| Locally fixed compliant-contact `Cn` | 5 + 7.5 | 402.2× |
| Independently parameterized compliant-contact `Cn` | 5 + 7.5 | 368.4× |
| Expanded fabric nuisances | 5 + 7.5 | 3.35× |

The expanded case also separates the stiff-bound `Cn` and the compliant-branch
critical porosity. It is the conservative primary analysis. Its post-design
spectral ratio is only `6.70e-4`, the local `Vcem`–`ln Cn` correlation remains
`-0.998`, and an additional 1% RMS target-aligned discrepancy reduces the gain
to 1.21×.
Thus, pressure does not robustly repair the nominal ambiguity in this scenario
model unless important fabric relations are independently constrained.

The 5 + 7.5 MPa pair is the best pair only among the tested candidates. Because
5 MPa is the lower search boundary, pressures below 5 MPa were not ruled out.

## Reproduce

From the package root:

```bash
python3 scripts/run_e3.py
python3 scripts/verify_e3.py
```

The full run performs 400 moving-block bootstrap replicates, pressure-reference
and trajectory controls, nine representative E2 operating-point tests, and
conditional re-optimization for 12, 20, and 40 m blocks. `--quick` reduces the
bootstrap and operating-point workload for development.

## Outputs

- `results/summary.json`: machine-readable headline results.
- `results/RESULTS.md`: concise scientific interpretation.
- `results/constant_cement_E3_results.xlsx`: curated, formula-auditable
  scientific workbook with eight result and provenance sheets.
- `results/tables/`: complete CSV outputs.
- `results/figures/`: four figures in PNG and PDF.
- `results/verification/`: 106-check verification report.
- `docs/`: methods, interpretation, sources, manuscript outline, and the E4
  decision gate.
- `MANIFEST.csv` and `SHA256SUMS.txt`: generated package inventory and payload
  hashes.

## Scope

This is a prospective design calculation, not validation with observed
pressure-dependent Hugin measurements. The pressure extension is a scenario
generator, not a replacement for the frozen RPIA law. Target responses are
evaluated nonlinearly on the finite grid, while nuisance effects in that grid
are adjusted with a tangent space frozen at the operating point. Boundary-
censored profile widths are reported as lower bounds.

Version: 1.0.0
