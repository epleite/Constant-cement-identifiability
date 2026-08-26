# E2 results: stability, LOTO transport, and hierarchy

## Main result

The factorized coordinate is stable under paired within-trajectory resampling and transports much better between the two Hugin trajectories than a local exponential tangent. This does **not** make the nominal parameters separately identifiable: the hierarchical comparison shows that static elastic data can absorb both a shared-`q_star` and a shared-`C_n` restriction along the same weak ridge.

## Bootstrap stability

The primary analysis used 400 non-circular moving-block replicates, stratified by trajectory, with a 5-sample (20 m) block. Successful replicates: 400.

- `A_raw`: median 0.3039, 95% interval [0.2910, 0.3109].
- `Gamma_raw`: median 1.4927, 95% interval [1.4425, 1.5254].
- `A_adjusted`: median 0.3041, 95% interval [0.2901, 0.3114].
- `Gamma_adjusted`: median 1.6207, 95% interval [1.5528, 1.6680].

These intervals are conditional on the two available Hugin trajectories. With only two trajectory clusters, they are not estimates of between-trajectory population uncertainty.

## Leave-one-trajectory-out level transport

- 19A to BT2, raw: factored ratio 0.9936; train-local exponential 1.4731.
- 19A to BT2, adjusted: factored ratio 1.0003; train-local exponential 1.4823.
- BT2 to 19A, raw: factored ratio 1.0329; train-local exponential 6.4270.
- BT2 to 19A, adjusted: factored ratio 1.0297; train-local exponential 6.4193.

This is an internal transport test under the frozen E1 metric, not external validation: the transfer-aware scale was estimated from both trajectories.

## Hierarchical comparison

For equal-dimensional shared-state models, positive `delta_Cn_minus_qstar` favors shared `q_star`.

- Fixed-nuisance observed-data difference: 0.1629.
- Joint nuisance-MAP difference: -0.0294.
- Local Schur-adjusted difference: -0.000289.
- In the 20 m block bootstrap, shared `q_star` had the lower raw objective in 29.0% of replicates; the 95% interval for the objective difference was [-9.235, 0.6361].

The scientifically conservative interpretation is that the data support the **transportability of the factorized ridge coordinate**, but do not yet discriminate decisively between geological sharing assumptions. Sharing `C_n` can still slide along the local ridge at negligible cost. A hierarchy is therefore not, by itself, a source of new sensitivity diversity.
