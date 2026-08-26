# E4 decision gate: independent fabric or explicit cemented-contact stress physics?

## Why E4 is needed

E3 rules out a simple interpretation of the pressure experiment. The
prospective bounding-average extension yields a `3.35x` adjusted minimum-
eigenvalue gain under expanded fabric nuisances, but an additional 1%-RMS discrepancy
aligned with the weak target direction reduces that gain to `1.21x` and drives
the spectral ratio below the static baseline. The finite-grid target profiles
remain boundary-censored.

The next experiment must therefore distinguish two remediation routes:

1. **Independent fabric constraint.** Add a measurement of contact number,
   contact radius, or another calibrated fabric proxy and determine the
   precision required to close the ridge.
2. **Explicit cemented-contact stress physics.** Replace the bounding-average
   scenario with a stress-dependent cemented-contact law and test whether its
   differential response remains independent after its own constitutive
   parameters are treated as nuisances.

Route 1 is a short fail-fast benchmark. Route 2 is the preferred scientific
route because it tests whether the missing direction comes from load transfer
at cemented contacts rather than from an imposed fabric relation.

## Provisional success gate

Call a remediation successful only if all of the following hold after expanded
fabric adjustment, the generic discrepancy basis, an additional 1%-RMS
target-aligned discrepancy, and a cross-model
generation/inversion test:

- `lambda_min(G_adj)` gain is at least `10x` in the reference analysis;
- the adjusted spectral ratio is at least `0.01`;
- the lower 95% moving-block-bootstrap limits remain above `5x` and `0.005`,
  respectively;
- the two-parameter `Delta Phi = 2.30` support is interior to the physical grid;
- the resolved profile widths of both `Vcem` and `Cn` contract by at least 50%.

These are operational design targets, not universal identifiability theorems.
They deliberately require much more than the fragile `3.35x` local gain found
in E3.

## Route 1: fabric-precision benchmark

Sweep the uncertainty of an independent contact-density/contact-radius
measurement while keeping exactly the same fabric priors in the static and
augmented analyses. This avoids counting the prior itself as pressure-derived
information. Report the minimum measurement precision at which the full gate
is crossed, and repeat the sweep separately by trajectory.

`Vcem` petrography should be retained as a holdout validation target rather
than used simultaneously as the constraint that makes `Vcem` identifiable.

## Route 2: cross-model stress-law test

Implement two physically distinct stress-response generators:

1. the Elata–Dvorkin (1996) micromechanical model, in which the area and normal
   and tangential stiffnesses of a cemented contact evolve with load; and
2. the Avseth–Skjei patchy-cement family (2011; operational equations in
   Avseth, Skjei & Skålnes, 2013), in which the pressure response comes from the
   compliant fraction between pressure-insensitive stiff cemented bounds.

For each generator, include stress-response amplitudes, exponents, pressure
calibration, Biot response, shear slip, hysteresis, residual stiff-frame
sensitivity, and fabric variables in the nuisance block. Generate with one law
and invert/design with the other, then reverse the pairing. Success requires
the new target direction to survive this cross-model discrepancy, not merely
to be present when the data-generating and inverse laws are identical.

The two generators must not be presented as interchangeable corrections to
Scheme 1. Elata–Dvorkin uses a coating/contact geometry closer to a Scheme-2
interpretation and lets contact area grow with load; its loaded radius cannot
simply be added to the Scheme-1 radius fixed by `Vcem` and `Cn`. The patchy-
cement weights are macroscopic positions between bounds, not unique cement
volumes or contact fractions, and all of their pressure sensitivity vanishes
as the compliant fraction goes to zero.

## Minimum laboratory inputs

- dry `Vp` and `Vs` on the same plugs at a reference and at least three
  additional effective pressures spanning both sides of the reference;
- at least one loading-unloading cycle and a repeated reference state;
- porosity and mineral composition for every plug;
- an independent fabric/contact-count or contact-radius measurement, ideally
  from micro-CT or microscopy;
- cement-volume petrography reserved for validation.

Saturated measurements can be added after the dry-frame gate is passed; they
should not be the first remedy because fluid substitution can add observable
count without adding an independent frame-sensitivity direction.

## Execution order

1. Run the analytic fabric-precision sweep to set a measurement requirement.
2. Implement and finite-difference-audit the first explicit stress law.
3. Add its constitutive parameters to the nuisance block before optimizing
   pressures.
4. Perform the cross-model test and only then select candidate pressure states.
5. Use the laboratory data to test the frozen gate without retuning its
   thresholds after seeing the result.
