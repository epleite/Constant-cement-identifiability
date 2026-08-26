# Scientific interpretation

## What E3 discovered

The exact no-go result survives every audit: the frozen constant-cement model
cannot acquire pressure sensitivity by relabelling repeated static states.

The more consequential finding is that the apparent success of the prospective
pressure extension is almost entirely conditional on fabric links. Sharing the
compliant-contact coordination number produces a 634× adjusted minimum-
eigenvalue gain. Making only that coordination number independent still gives
368×. Once the stiff-bound coordination number and compliant critical porosity
are also adjusted, the gain falls to 3.35×.

That residual gain is not practical separability:

- spectral ratio: `6.70e-4`;
- local `Vcem`–`ln Cn` correlation: `-0.9984`;
- unconstrained local-Gaussian marginal `Vcem` SD: `7.65` percentage points;
- unconstrained local-Gaussian marginal `ln Cn` SD: `1.54`;
- additional 1% target-aligned discrepancy gain: `1.21×`;
- finite-grid `Vcem` support remains boundary-censored.

The right statement is therefore not “pressure breaks the ridge.” It is:

> A stress-sensitive extension can appear to rotate the weak direction, but
> the rotation is absorbed when the fabric variables that generate it are
> treated as uncertain.

## Implication for the paper

This negative result is publishable because it identifies the hidden condition
behind a common experimental-design intuition. Observable diversity must
survive adjustment for constitutive fabric, not only ordinary mineral/fluid
nuisances.

A sharper paper question is:

> Which independently constrained fabric quantity, or which alternative
> stress-response mechanism, is required before multi-pressure elasticity can
> separate cement volume from coordination number?

The current bounding-average experiment answers the first half: pressure-only
`Vp,Vs` is insufficient when compliant and stiff fabrics are allowed to move
independently.

## Claims supported by E3

- The standard Scheme 1 constant-cement implementation is pressure blind.
- Replication improves precision but adds no raw constitutive direction.
- Fabric-sharing assumptions dominate the predicted remediation.
- The conservative pressure design leaves the ridge essentially intact.
- Smooth generic discrepancy is not enough; target-aligned model error can
  absorb almost all residual gain.
- The selected 5 and 7.5 MPa additions are conditional on a 39 MPa reference,
  the finite candidate set, assumed errors, priors, and scenario physics.

## Claims not supported

- That 5, 7.5, and 39 MPa are a universally optimal laboratory schedule.
- That Hugin `Vcem` and `Cn` are recoverable from pressure data.
- That the heuristic bounding average is validated for these rocks.
- That the finite-grid curves are fully nonlinear nuisance profiles.
- That a shared target pair is valid for arbitrary independent plugs.

## Candidate validation experiment

A laboratory validation could pair multi-pressure `Vp,Vs` with at least one
independent fabric constraint, for example contact-density/contact-radius
imaging, cement-volume petrography, or a calibrated compliant-contact proxy.
Loading and unloading would expose hysteresis; a repeated reference state could
estimate drift; dry and saturated responses would help distinguish frame and
fluid effects. E3 does not yet prescribe a definitive acquisition schedule.

The alternative theoretical route is to replace the bounding-average scenario
with a pressure law in which cemented-contact load transfer is explicit, then
repeat the same nuisance-expanded E-optimal test. The success criterion remains
an increase in `lambda_min(Gadj)` that persists under fabric and model-form
adjustment.
