# Constant-cement experiment E1: Discover and explain

This self-contained package executes the first formal **Discover–Explain** experiment for the proposed paper:

> *From effective contact state to microstructural separability: Identifiability and experimental design for the constant-cement model*

The experiment asks whether the empirical pooled coordinate

\[
\ln C_n + 20.3\,V_{\rm cem}
\]

is reproduced by the exact constant-cement Scheme 1 physics, how it varies over \((V_{\rm cem},C_n,\phi)\), and whether any residual separation survives nuisance adjustment.

## Frozen ingredients

- exact RPIA v1 `constant_cement` forward model;
- 29 selected Hugin samples from 19A and 30 from BT2;
- target scales \(D=\operatorname{diag}(0.015,0.20)\);
- transfer-aware observable covariance;
- unit-Gaussian priors for the standardized nuisance parameters;
- Scheme 1 dependence

\[
a_c=2\left[\frac{V_{\rm cem}}{3C_n(1-\phi_c)}\right]^{1/4},
\qquad \phi_b=\phi_c-V_{\rm cem},
\qquad T=\phi/\phi_b.
\]

The vendored `vendor/rpia_v1/rpia_core.py` is byte-identical to the submitted RPIA reproducibility package. E1-specific calculations are isolated in `src/e1_analysis.py`.

## Run

```bash
python scripts/run_e1.py
python scripts/verify_e1.py
```

For a fast smoke test:

```bash
python scripts/run_e1.py --quick
```

The full run creates four publication-oriented figures, eight numerical tables, `results/summary.json`, and `results/RESULTS.md`.

## Interpretation boundary

The package deliberately distinguishes:

1. observable-specific physical sensitivity ratios;
2. the weak eigen-direction of the raw target Gram matrix;
3. the projection coefficient between target columns;
4. the weak eigen-direction after nuisance adjustment.

These coincide only near rank one. The original 20.3 value is an eigen-direction slope, not a globally invariant material constant.

E1 also derives and tests a pooled, metric-dependent, physics-factored approximate trajectory invariant,

\[
q_{\rm phys}
=\ln\frac{C_n}{C_{n,0}}
+A\ln\frac{V_{\rm cem}}{V_0}
+\Gamma\ln\frac{\phi_c-V_{\rm cem}}{\phi_c-V_0},
\]

where \(A\) and \(\Gamma\) are obtained by separating the contact-radius and HS-path contributions in whitened observable space. The generated results quantify both its local slope and its finite-ridge invariance.

The model is static and contains no pressure response for constant cement. This package therefore does **not** claim that repeated pressure states break the ambiguity; that requires a validated pressure-aware constitutive extension.
