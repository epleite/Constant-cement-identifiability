# Data provenance and processing

## Source

The trajectory tables are derived from the Volve subsurface dataset released
by Equinor and the Volve licence partners in 2018. Equinor currently lists the
Volve subsurface data in its public data-sharing programme and states that data
hosted on that platform can be used by anyone:

- https://www.equinor.com/energy/data-sharing
- https://www.equinor.com/news/archive/14jun2018-disclosing-volve-data

The compact tables were inherited from the frozen RPIA Volve reproducibility
snapshot. No confidential or proprietary field data are used.

## Trajectories and sampling

| trajectory | source well identifier | Hugin interval used (m) | compact rows in interval | selected rows | selected sampled depths (m) |
|---|---|---:|---:|---:|---:|
| 19A | 15/9-19A | 3033.9554193–3152.3797046 | 30 | 29 | 3036–3152 |
| BT2 | 15/9-19BT2 | 3153.9039509–3295.8296572 | 36 | 30 | 3156–3272 |

The compact trajectories are sampled at 4 m. The two records are related
borehole trajectories and are not treated as spatially independent wells.

## Columns and units

| column | meaning | unit / convention |
|---|---|---|
| `depth_m` | sampled trajectory depth coordinate | m |
| `phi` | total/effective porosity input used by the frozen workflow | fraction |
| `vsh` | shale-volume log proxy | fraction |
| `sw` | water saturation | fraction |
| `vp_mps` | compressional-wave velocity | m s^-1 |
| `vs_mps` | shear-wave velocity | m s^-1 |
| `rho_gcc` | bulk density | g cm^-3 |

The seven required columns contain no missing values in either compact input or
selected subset.

## Selection

The Hugin subsets are regenerated from `data/compact/*_training_window.csv` by
applying the interval endpoints above and the inclusive clean-sand criteria

```text
0.08 <= phi <= 0.30
vsh <= 0.30
```

This leaves 29 samples for 19A and 30 for BT2. Rows are retained jointly, so
depth, porosity, shale proxy, saturation, velocities and density remain paired.
`vsh` is a first-order mineralogical proxy, not a direct measurement of a clay
mineral fraction. Uncertainty in that mapping is represented in the nuisance
space.

## File roles

- `data/compact/19A_training_window.csv`: 84-row compact 19A input window;
- `data/compact/BT2_training_window.csv`: 70-row compact BT2 input window;
- `data/compact/rpia_metadata.json`: exact interval and filter metadata;
- `data/derived/*_Hugin_sand_subset.csv`: selected 29- and 30-row analysis
  tables, included for inspection and checked against regenerated outputs.

## Nuisance coordinates used by the static constant-cement analysis

The nine nuisance coordinates have independent standard-normal priors after
division by the following one-standard-deviation scales:

| nuisance coordinate | baseline | scale |
|---|---:|---:|
| porosity bias | 0 | 0.01 absolute fraction |
| `vsh` bias | 0 | 0.03 absolute fraction |
| water-saturation bias | 0 | 0.05 absolute fraction |
| framework K-feldspar fraction shift | 0.15 baseline | 0.10 absolute fraction |
| log clay-modulus scale | 0 | 0.25 |
| brine-salinity shift | 0.070 baseline | 0.030 mass fraction |
| gas-oil-ratio shift | 114 baseline | 20 Sm^3/Sm^3 |
| log cement-modulus scale | 0 | 0.20 |
| packing-reference porosity shift | 0.40 baseline | 0.02 absolute fraction |

Pore pressure (32.77 MPa) and temperature (106.2 degrees C) define the frozen
fluid-property state. The 39 MPa effective-stress reference enters the
prospective pressure-design experiment, but the frozen constant-cement branch
is pressure blind.

