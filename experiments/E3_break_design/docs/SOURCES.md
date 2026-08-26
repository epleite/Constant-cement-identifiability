# Sources and provenance

## Rock physics

- Avseth, P., Dvorkin, J., Mavko, G. & Rykkje, J. (2000), *Rock physics
  diagnostic of North Sea sands: Link between microstructure and seismic
  properties*. DOI: https://doi.org/10.1029/1999GL008468 — introduces the
  constant-cement diagnostic; the publisher page is marked Free Access.

- Avseth, P. (2000), *Combining Rock Physics and Sedimentology for Seismic
  Reservoir Characterization of North Sea Turbidite Systems*, Stanford PhD
  thesis. Official PDF:
  https://srb2.sites.stanford.edu/sites/g/files/sbiybj22311/files/media/file/Combining%20Rock%20Physics%20and%20Sedminentology%20for%20Seismic%20Reservoir%20Characterization%20of%20North%20Sea%20Turbidite%20Systems%20-%20Per%20Avseth.pdf
  — authorial open source for the constant-cement endpoints and modified
  lower-Hashin–Shtrikman path. Its appendix uses Scheme 2 and is not the sole
  source for the Scheme 1 contact-radius geometry.

- Avseth, P. & Skjei, N. (2011), *Rock physics modeling of static and dynamic
  reservoir properties—A heuristic approach for cemented sandstone
  reservoirs*. DOI: https://doi.org/10.1190/1.3535437 — introduces the
  patchy-cement/bounding-average pressure-response interpretation used only as
  inspiration for the E3 prospective scenario. The official SEG page is not
  open access; an author-shared copy is publicly visible on ResearchGate, but
  no open-content licence is assumed.

- Avseth, P., Skjei, N. & Skålnes, Å. (2013), *Rock physics modelling of 4D
  time-shifts and time-shift derivatives using well log data—a North Sea
  demonstration*. DOI:
  https://doi.org/10.1111/j.1365-2478.2012.01134.x — writes the operational
  bulk- and shear-modulus bounding equations explicitly. The official Wiley
  page is not marked open access; an author-shared full text is publicly
  visible on ResearchGate.

- Dvorkin, J. & Nur, A. (1996), *Elasticity of high-porosity sandstones:
  Theory for two North Sea data sets*. DOI: https://doi.org/10.1190/1.1444059
  — original contact-cement and friable/soft-sand branches.

- Lin, H., Suleiman, M.T. & Brown, D.G. (2020), *Investigation of pore-scale
  CaCO3 distributions and their effects on stiffness and permeability of
  sands treated by microbially induced carbonate precipitation*. DOI:
  https://doi.org/10.1016/j.sandf.2020.07.003 — open article writing Scheme 1
  explicitly as `a = 2[Vcem/{3 Cn (1-phic)}]^(1/4)`.

- Elata, D. & Dvorkin, J. (1996), *Pressure sensitivity of cemented granular
  materials*. DOI: https://doi.org/10.1016/0167-6636(96)00005-1 — physical
  example in which cemented-contact area and normal/tangential stiffness evolve
  with load. The official full text is paywalled; institutional metadata and
  abstract are public.

- Bachrach, R. & Avseth, P. (2008), *Rock physics modeling of unconsolidated
  sands: Accounting for nonuniform contacts and heterogeneous stress fields in
  the effective media approximation with applications to hydrocarbon
  exploration*. DOI: https://doi.org/10.1190/1.2985821 — motivates more
  realistic compliant-contact stress physics than uniform Hertz–Mindlin.

## Inverse theory and design

- Cox, D.R. & Reid, N. (1987), *Parameter Orthogonality and Approximate
  Conditional Inference*. DOI:
  https://doi.org/10.1111/j.2517-6161.1987.tb01422.x — efficient information
  after nuisance adjustment; the implemented Gaussian-prior form is the
  corresponding block Schur complement.

- Kiefer, J. (1974), *General Equivalence Theory for Optimum Designs
  (Approximate Theory)*. DOI: https://doi.org/10.1214/aos/1176342810 — classic
  source for E-optimal design based on the minimum information eigenvalue.

## Internal provenance

- `vendor/e1_v1/`: exact E1 frozen-model implementation and expected anchors.
- `reference/E2_summary.json`: E2 headline stability/hierarchy results.
- `reference/E2_bootstrap_replicates.csv`: E2 operating-point replicates used
  by the E3 sensitivity and bounding-weight controls.
- SHA-256 provenance hashes are recorded in `results/summary.json` and checked
  by `scripts/verify_e3.py`.
