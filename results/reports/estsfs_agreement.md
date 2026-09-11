# B3 — Ancestree (FixedTreeInference) vs fastDFE est-sfs

## Simulation matrix
- ploidy: `1` (haploid throughout)
- ingroup: `40` haps (Ne=10000)
- outgroups: 3 nested splits at `[50000.0, 100000.0, 150000.0]` gens
  (each outgroup contributes 1 haploid sample, Ne_out=100000)
- sequence length: `2e+06`
- mutation rate: `1e-08` per-site per-gen
- recombination rate: `1e-08`
- sim sweeps: ``sim`` ∈ ``['jc', 'hky2', 'gtr']``. Per-sim Ts/Tv (effective κ): {'jc': 1.0, 'hky2': 2.0, 'gtr': 4.0}.
  The ``gtr`` sim uses skewed equilibrium π=(0.3, 0.2, 0.2, 0.3) on top of
  transition-biased exchangeability rates.
- inference subsets to ``n_out`` ∈ ``[1, 2, 3]`` closest outgroups
- seed: `42`

## Ground truth (`I → outgroup_k` divergence, substitutions per site)
Span-weighted average of the `I → O_k` branch-length sum over local
ARG segments, taken directly from the simulated tskit trees (so it
includes the actual realised coalescent depths in this replicate). The
span-weighted path lengths are `1.042e+05, 2.012e+05, 2.996e+05` generations for
(O_1, O_2, O_3), shared by every sim since all sims use the same
demography and seed.

Generations are converted to substitutions per site by `mu * rho`,
where `mu = 1e-08` is the per-site per-generation rate of
mutation events and `rho` is the expected number of substitutions per
mutation event under that sim's own mutation model. ``msprime``
normalises its transition matrix by the largest row sum, so an event
leaves the state unchanged with probability `P_ii` and
`rho = 1 - sum_i pi_i P_ii`, with `pi_i` the equilibrium frequency of
state `i`. The fitted branch rates carry the same units.

| sim | rho (subs per mutation event) | O_1 (closest) | O_2 | O_3 (farthest) |
|---|---|---|---|---|
| jc | 1.0000 | 1.042e-03 | 2.012e-03 | 2.996e-03 |
| hky2 | 1.0000 | 1.042e-03 | 2.012e-03 | 2.996e-03 |
| gtr | 0.8588 | 8.950e-04 | 1.728e-03 | 2.573e-03 |


## MAP-allele agreement (Ancestree vs fastDFE)

| sim | n_out | model | prior | shared | non-hom / hom | agreement (non-hom) | agreement (hom) |
|---|---|---|---|---|---|---|---|
| jc | 1 | JC | kingman | 11151 | 11134 / 17 | 0.9233 (92.33%) | 0.1765 (17.65%) |
| jc | 1 | JC | adaptive | 11151 | 11134 / 17 | 0.9233 (92.33%) | 0.1765 (17.65%) |
| jc | 1 | K2 | kingman | 11151 | 11134 / 17 | 0.9233 (92.33%) | 0.1765 (17.65%) |
| jc | 1 | K2 | adaptive | 11151 | 11134 / 17 | 0.9233 (92.33%) | 0.1765 (17.65%) |
| jc | 1 | R6 | kingman | — | — | — | — |
| jc | 1 | R6 | adaptive | — | — | — | — |
| jc | 2 | JC | kingman | 11151 | 11134 / 17 | 0.9235 (92.35%) | 0.3529 (35.29%) |
| jc | 2 | JC | adaptive | 11151 | 11134 / 17 | 0.9235 (92.35%) | 0.3529 (35.29%) |
| jc | 2 | K2 | kingman | 11151 | 11134 / 17 | 0.9235 (92.35%) | 0.3529 (35.29%) |
| jc | 2 | K2 | adaptive | 11151 | 11134 / 17 | 0.9235 (92.35%) | 0.3529 (35.29%) |
| jc | 2 | R6 | kingman | — | — | — | — |
| jc | 2 | R6 | adaptive | — | — | — | — |
| jc | 3 | JC | kingman | 11151 | 11134 / 17 | 1.0000 (100.00%) | 0.3529 (35.29%) |
| jc | 3 | JC | adaptive | 11151 | 11134 / 17 | 1.0000 (100.00%) | 0.3529 (35.29%) |
| jc | 3 | K2 | kingman | 11151 | 11134 / 17 | 1.0000 (100.00%) | 0.3529 (35.29%) |
| jc | 3 | K2 | adaptive | 11151 | 11134 / 17 | 1.0000 (100.00%) | 0.3529 (35.29%) |
| jc | 3 | R6 | kingman | — | — | — | — |
| jc | 3 | R6 | adaptive | — | — | — | — |
| hky2 | 1 | JC | kingman | 11151 | 11134 / 17 | 0.9232 (92.32%) | 0.1176 (11.76%) |
| hky2 | 1 | JC | adaptive | 11151 | 11134 / 17 | 0.9232 (92.32%) | 0.1176 (11.76%) |
| hky2 | 1 | K2 | kingman | 11151 | 11134 / 17 | 0.9232 (92.32%) | 0.1176 (11.76%) |
| hky2 | 1 | K2 | adaptive | 11151 | 11134 / 17 | 0.9232 (92.32%) | 0.1176 (11.76%) |
| hky2 | 1 | R6 | kingman | — | — | — | — |
| hky2 | 1 | R6 | adaptive | — | — | — | — |
| hky2 | 2 | JC | kingman | 11151 | 11134 / 17 | 0.9234 (92.34%) | 0.1176 (11.76%) |
| hky2 | 2 | JC | adaptive | 11151 | 11134 / 17 | 0.9234 (92.34%) | 0.1176 (11.76%) |
| hky2 | 2 | K2 | kingman | 11151 | 11134 / 17 | 0.9234 (92.34%) | 0.1176 (11.76%) |
| hky2 | 2 | K2 | adaptive | 11151 | 11134 / 17 | 0.9234 (92.34%) | 0.1176 (11.76%) |
| hky2 | 2 | R6 | kingman | — | — | — | — |
| hky2 | 2 | R6 | adaptive | — | — | — | — |
| hky2 | 3 | JC | kingman | 11151 | 11134 / 17 | 0.9999 (99.99%) | 0.1176 (11.76%) |
| hky2 | 3 | JC | adaptive | 11151 | 11134 / 17 | 0.9999 (99.99%) | 0.1176 (11.76%) |
| hky2 | 3 | K2 | kingman | 11151 | 11134 / 17 | 0.9999 (99.99%) | 0.1176 (11.76%) |
| hky2 | 3 | K2 | adaptive | 11151 | 11134 / 17 | 0.9999 (99.99%) | 0.1176 (11.76%) |
| hky2 | 3 | R6 | kingman | — | — | — | — |
| hky2 | 3 | R6 | adaptive | — | — | — | — |
| gtr | 1 | JC | kingman | 11151 | 11134 / 17 | 0.9344 (93.44%) | 0.4118 (41.18%) |
| gtr | 1 | JC | adaptive | 11151 | 11134 / 17 | 0.9344 (93.44%) | 0.4118 (41.18%) |
| gtr | 1 | K2 | kingman | 11151 | 11134 / 17 | 0.9344 (93.44%) | 0.4118 (41.18%) |
| gtr | 1 | K2 | adaptive | 11151 | 11134 / 17 | 0.9344 (93.44%) | 0.4118 (41.18%) |
| gtr | 1 | R6 | kingman | — | — | — | — |
| gtr | 1 | R6 | adaptive | — | — | — | — |
| gtr | 2 | JC | kingman | 11151 | 11134 / 17 | 0.9345 (93.45%) | 0.4706 (47.06%) |
| gtr | 2 | JC | adaptive | 11151 | 11134 / 17 | 0.9345 (93.45%) | 0.4706 (47.06%) |
| gtr | 2 | K2 | kingman | 11151 | 11134 / 17 | 0.9345 (93.45%) | 0.4706 (47.06%) |
| gtr | 2 | K2 | adaptive | 11151 | 11134 / 17 | 0.9345 (93.45%) | 0.4706 (47.06%) |
| gtr | 2 | R6 | kingman | — | — | — | — |
| gtr | 2 | R6 | adaptive | — | — | — | — |
| gtr | 3 | JC | kingman | 11151 | 11134 / 17 | 0.9999 (99.99%) | 0.5294 (52.94%) |
| gtr | 3 | JC | adaptive | 11151 | 11134 / 17 | 0.9999 (99.99%) | 0.5294 (52.94%) |
| gtr | 3 | K2 | kingman | 11151 | 11134 / 17 | 0.9999 (99.99%) | 0.5294 (52.94%) |
| gtr | 3 | K2 | adaptive | 11151 | 11134 / 17 | 0.9999 (99.99%) | 0.5294 (52.94%) |
| gtr | 3 | R6 | kingman | — | — | — | — |
| gtr | 3 | R6 | adaptive | — | — | — | — |

## Posterior numerical agreement

| sim | n_out | model | prior | mean diff (agree only) | max diff (agree only) | mean diff (all) | max diff (all) |
|---|---|---|---|---|---|---|---|
| jc | 1 | JC | kingman | 3.644e-05 | 1.111e-02 | 7.656e-02 | 9.950e-01 |
| jc | 1 | JC | adaptive | 5.262e-05 | 1.401e-02 | 7.674e-02 | 9.973e-01 |
| jc | 1 | K2 | kingman | 3.625e-05 | 1.141e-02 | 7.656e-02 | 9.953e-01 |
| jc | 1 | K2 | adaptive | 5.255e-05 | 1.441e-02 | 7.674e-02 | 9.974e-01 |
| jc | 1 | R6 | kingman | — | — | — | — |
| jc | 1 | R6 | adaptive | — | — | — | — |
| jc | 2 | JC | kingman | 1.388e-03 | 4.330e-02 | 7.589e-02 | 9.755e-01 |
| jc | 2 | JC | adaptive | 8.467e-03 | 2.309e-01 | 8.243e-02 | 9.756e-01 |
| jc | 2 | K2 | kingman | 1.387e-03 | 4.332e-02 | 7.589e-02 | 9.755e-01 |
| jc | 2 | K2 | adaptive | 8.463e-03 | 2.310e-01 | 8.243e-02 | 9.756e-01 |
| jc | 2 | R6 | kingman | — | — | — | — |
| jc | 2 | R6 | adaptive | — | — | — | — |
| jc | 3 | JC | kingman | 2.344e-06 | 6.854e-04 | 2.344e-06 | 6.854e-04 |
| jc | 3 | JC | adaptive | 3.028e-06 | 7.364e-04 | 3.028e-06 | 7.364e-04 |
| jc | 3 | K2 | kingman | 2.351e-06 | 6.991e-04 | 2.351e-06 | 6.991e-04 |
| jc | 3 | K2 | adaptive | 3.028e-06 | 7.537e-04 | 3.028e-06 | 7.537e-04 |
| jc | 3 | R6 | kingman | — | — | — | — |
| jc | 3 | R6 | adaptive | — | — | — | — |
| hky2 | 1 | JC | kingman | 3.643e-05 | 1.111e-02 | 7.634e-02 | 9.950e-01 |
| hky2 | 1 | JC | adaptive | 5.258e-05 | 1.400e-02 | 7.652e-02 | 9.972e-01 |
| hky2 | 1 | K2 | kingman | 4.100e-05 | 1.606e-02 | 7.631e-02 | 9.958e-01 |
| hky2 | 1 | K2 | adaptive | 5.966e-05 | 2.023e-02 | 7.651e-02 | 9.975e-01 |
| hky2 | 1 | R6 | kingman | — | — | — | — |
| hky2 | 1 | R6 | adaptive | — | — | — | — |
| hky2 | 2 | JC | kingman | 1.382e-03 | 4.328e-02 | 7.592e-02 | 9.755e-01 |
| hky2 | 2 | JC | adaptive | 8.430e-03 | 2.308e-01 | 8.243e-02 | 9.756e-01 |
| hky2 | 2 | K2 | kingman | 1.385e-03 | 4.329e-02 | 7.592e-02 | 9.755e-01 |
| hky2 | 2 | K2 | adaptive | 8.434e-03 | 2.308e-01 | 8.244e-02 | 9.756e-01 |
| hky2 | 2 | R6 | kingman | — | — | — | — |
| hky2 | 2 | R6 | adaptive | — | — | — | — |
| hky2 | 3 | JC | kingman | 2.377e-06 | 6.961e-04 | 2.377e-06 | 6.961e-04 |
| hky2 | 3 | JC | adaptive | 3.094e-06 | 7.493e-04 | 3.094e-06 | 7.493e-04 |
| hky2 | 3 | K2 | kingman | 2.698e-06 | 1.046e-03 | 2.698e-06 | 1.046e-03 |
| hky2 | 3 | K2 | adaptive | 3.498e-06 | 1.128e-03 | 3.498e-06 | 1.128e-03 |
| hky2 | 3 | R6 | kingman | — | — | — | — |
| hky2 | 3 | R6 | adaptive | — | — | — | — |
| gtr | 1 | JC | kingman | 3.093e-05 | 9.485e-03 | 7.568e-02 | 9.955e-01 |
| gtr | 1 | JC | adaptive | 4.439e-05 | 1.106e-02 | 7.584e-02 | 9.974e-01 |
| gtr | 1 | K2 | kingman | 4.564e-05 | 1.839e-02 | 7.558e-02 | 9.971e-01 |
| gtr | 1 | K2 | adaptive | 6.607e-05 | 2.155e-02 | 7.581e-02 | 9.981e-01 |
| gtr | 1 | R6 | kingman | — | — | — | — |
| gtr | 1 | R6 | adaptive | — | — | — | — |
| gtr | 2 | JC | kingman | 1.390e-03 | 9.783e-02 | 7.522e-02 | 9.755e-01 |
| gtr | 2 | JC | adaptive | 8.244e-03 | 2.298e-01 | 8.156e-02 | 9.756e-01 |
| gtr | 2 | K2 | kingman | 1.398e-03 | 9.772e-02 | 7.522e-02 | 9.755e-01 |
| gtr | 2 | K2 | adaptive | 8.254e-03 | 2.297e-01 | 8.157e-02 | 9.756e-01 |
| gtr | 2 | R6 | kingman | — | — | — | — |
| gtr | 2 | R6 | adaptive | — | — | — | — |
| gtr | 3 | JC | kingman | 2.617e-06 | 2.651e-03 | 2.617e-06 | 2.651e-03 |
| gtr | 3 | JC | adaptive | 3.512e-06 | 4.099e-03 | 3.512e-06 | 4.099e-03 |
| gtr | 3 | K2 | kingman | 3.571e-06 | 2.623e-03 | 3.571e-06 | 2.623e-03 |
| gtr | 3 | K2 | adaptive | 4.803e-06 | 4.137e-03 | 4.803e-06 | 4.137e-03 |
| gtr | 3 | R6 | kingman | — | — | — | — |
| gtr | 3 | R6 | adaptive | — | — | — | — |

## Truth recovery (msprime `Site.ancestral_state`)

| sim | n_out | model | prior | Ancestree | fastDFE |
|---|---|---|---|---|---|
| jc | 1 | JC | kingman | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 1 | JC | adaptive | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 1 | K2 | kingman | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 1 | K2 | adaptive | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 1 | R6 | kingman | 0.9994 (99.94%) | — |
| jc | 1 | R6 | adaptive | 0.9994 (99.94%) | — |
| jc | 2 | JC | kingman | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 2 | JC | adaptive | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 2 | K2 | kingman | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 2 | K2 | adaptive | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 2 | R6 | kingman | 0.9994 (99.94%) | — |
| jc | 2 | R6 | adaptive | 0.9994 (99.94%) | — |
| jc | 3 | JC | kingman | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 3 | JC | adaptive | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 3 | K2 | kingman | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 3 | K2 | adaptive | 0.9994 (99.94%) | 0.9964 (99.64%) |
| jc | 3 | R6 | kingman | 0.9994 (99.94%) | — |
| jc | 3 | R6 | adaptive | 0.9994 (99.94%) | — |
| hky2 | 1 | JC | kingman | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 1 | JC | adaptive | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 1 | K2 | kingman | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 1 | K2 | adaptive | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 1 | R6 | kingman | 0.9994 (99.94%) | — |
| hky2 | 1 | R6 | adaptive | 0.9994 (99.94%) | — |
| hky2 | 2 | JC | kingman | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 2 | JC | adaptive | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 2 | K2 | kingman | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 2 | K2 | adaptive | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 2 | R6 | kingman | 0.9994 (99.94%) | — |
| hky2 | 2 | R6 | adaptive | 0.9994 (99.94%) | — |
| hky2 | 3 | JC | kingman | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 3 | JC | adaptive | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 3 | K2 | kingman | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 3 | K2 | adaptive | 0.9994 (99.94%) | 0.9958 (99.58%) |
| hky2 | 3 | R6 | kingman | 0.9994 (99.94%) | — |
| hky2 | 3 | R6 | adaptive | 0.9994 (99.94%) | — |
| gtr | 1 | JC | kingman | 0.9070 (90.70%) | 0.9046 (90.46%) |
| gtr | 1 | JC | adaptive | 0.9070 (90.70%) | 0.9046 (90.46%) |
| gtr | 1 | K2 | kingman | 0.9070 (90.70%) | 0.9046 (90.46%) |
| gtr | 1 | K2 | adaptive | 0.9070 (90.70%) | 0.9046 (90.46%) |
| gtr | 1 | R6 | kingman | 0.9070 (90.70%) | — |
| gtr | 1 | R6 | adaptive | 0.9070 (90.70%) | — |
| gtr | 2 | JC | kingman | 0.9070 (90.70%) | 0.9046 (90.46%) |
| gtr | 2 | JC | adaptive | 0.9070 (90.70%) | 0.9046 (90.46%) |
| gtr | 2 | K2 | kingman | 0.9070 (90.70%) | 0.9046 (90.46%) |
| gtr | 2 | K2 | adaptive | 0.9070 (90.70%) | 0.9046 (90.46%) |
| gtr | 2 | R6 | kingman | 0.9070 (90.70%) | — |
| gtr | 2 | R6 | adaptive | 0.9070 (90.70%) | — |
| gtr | 3 | JC | kingman | 0.9082 (90.82%) | 0.9058 (90.58%) |
| gtr | 3 | JC | adaptive | 0.9082 (90.82%) | 0.9058 (90.58%) |
| gtr | 3 | K2 | kingman | 0.9082 (90.82%) | 0.9058 (90.58%) |
| gtr | 3 | K2 | adaptive | 0.9082 (90.82%) | 0.9058 (90.58%) |
| gtr | 3 | R6 | kingman | 0.9082 (90.82%) | — |
| gtr | 3 | R6 | adaptive | 0.9082 (90.82%) | — |

## Branch rates vs ground truth (ingroup MRCA → each outgroup)

| sim | n_out | model | prior | empirical (tskit, subs/site) | Ancestree fitted | fastDFE fitted |
|---|---|---|---|---|---|---|
| jc | 1 | JC | kingman | 1.042e-03 | 1.057e-03 | 5.546e-04 |
| jc | 1 | JC | adaptive | 1.042e-03 | 1.047e-03 | 5.546e-04 |
| jc | 1 | K2 | kingman | 1.042e-03 | 1.057e-03 | 5.546e-04 |
| jc | 1 | K2 | adaptive | 1.042e-03 | 1.047e-03 | 5.546e-04 |
| jc | 1 | R6 | kingman | 1.042e-03 | 1.081e-03 | — |
| jc | 1 | R6 | adaptive | 1.042e-03 | 1.065e-03 | — |
| jc | 2 | JC | kingman | 1.042e-03, 2.012e-03 | 1.057e-03, 2.054e-03 | 7.260e-04, 1.307e-03 |
| jc | 2 | JC | adaptive | 1.042e-03, 2.012e-03 | 1.047e-03, 2.044e-03 | 7.260e-04, 1.307e-03 |
| jc | 2 | K2 | kingman | 1.042e-03, 2.012e-03 | 1.057e-03, 2.054e-03 | 7.259e-04, 1.307e-03 |
| jc | 2 | K2 | adaptive | 1.042e-03, 2.012e-03 | 1.047e-03, 2.044e-03 | 7.259e-04, 1.307e-03 |
| jc | 2 | R6 | kingman | 1.042e-03, 2.012e-03 | 1.102e-03, 2.140e-03 | — |
| jc | 2 | R6 | adaptive | 1.042e-03, 2.012e-03 | 1.055e-03, 2.061e-03 | — |
| jc | 3 | JC | kingman | 1.042e-03, 2.012e-03, 2.996e-03 | 1.057e-03, 2.054e-03, 3.050e-03 | 9.401e-04, 1.843e-03, 2.725e-03 |
| jc | 3 | JC | adaptive | 1.042e-03, 2.012e-03, 2.996e-03 | 1.047e-03, 2.044e-03, 3.040e-03 | 9.401e-04, 1.843e-03, 2.725e-03 |
| jc | 3 | K2 | kingman | 1.042e-03, 2.012e-03, 2.996e-03 | 1.057e-03, 2.054e-03, 3.050e-03 | 9.399e-04, 1.843e-03, 2.725e-03 |
| jc | 3 | K2 | adaptive | 1.042e-03, 2.012e-03, 2.996e-03 | 1.047e-03, 2.044e-03, 3.040e-03 | 9.399e-04, 1.843e-03, 2.725e-03 |
| jc | 3 | R6 | kingman | 1.042e-03, 2.012e-03, 2.996e-03 | 1.062e-03, 2.063e-03, 3.064e-03 | — |
| jc | 3 | R6 | adaptive | 1.042e-03, 2.012e-03, 2.996e-03 | 1.051e-03, 2.052e-03, 3.050e-03 | — |
| hky2 | 1 | JC | kingman | 1.042e-03 | 1.057e-03 | 5.547e-04 |
| hky2 | 1 | JC | adaptive | 1.042e-03 | 1.047e-03 | 5.547e-04 |
| hky2 | 1 | K2 | kingman | 1.042e-03 | 1.057e-03 | 5.548e-04 |
| hky2 | 1 | K2 | adaptive | 1.042e-03 | 1.047e-03 | 5.548e-04 |
| hky2 | 1 | R6 | kingman | 1.042e-03 | 1.109e-03 | — |
| hky2 | 1 | R6 | adaptive | 1.042e-03 | 1.095e-03 | — |
| hky2 | 2 | JC | kingman | 1.042e-03, 2.012e-03 | 1.057e-03, 2.055e-03 | 7.260e-04, 1.306e-03 |
| hky2 | 2 | JC | adaptive | 1.042e-03, 2.012e-03 | 1.047e-03, 2.044e-03 | 7.260e-04, 1.306e-03 |
| hky2 | 2 | K2 | kingman | 1.042e-03, 2.012e-03 | 1.057e-03, 2.055e-03 | 7.261e-04, 1.306e-03 |
| hky2 | 2 | K2 | adaptive | 1.042e-03, 2.012e-03 | 1.047e-03, 2.045e-03 | 7.261e-04, 1.306e-03 |
| hky2 | 2 | R6 | kingman | 1.042e-03, 2.012e-03 | 1.070e-03, 2.079e-03 | — |
| hky2 | 2 | R6 | adaptive | 1.042e-03, 2.012e-03 | 1.204e-03, 2.352e-03 | — |
| hky2 | 3 | JC | kingman | 1.042e-03, 2.012e-03, 2.996e-03 | 1.057e-03, 2.055e-03, 3.051e-03 | 9.389e-04, 1.843e-03, 2.724e-03 |
| hky2 | 3 | JC | adaptive | 1.042e-03, 2.012e-03, 2.996e-03 | 1.047e-03, 2.044e-03, 3.040e-03 | 9.389e-04, 1.843e-03, 2.724e-03 |
| hky2 | 3 | K2 | kingman | 1.042e-03, 2.012e-03, 2.996e-03 | 1.058e-03, 2.055e-03, 3.051e-03 | 9.389e-04, 1.843e-03, 2.724e-03 |
| hky2 | 3 | K2 | adaptive | 1.042e-03, 2.012e-03, 2.996e-03 | 1.047e-03, 2.045e-03, 3.041e-03 | 9.389e-04, 1.843e-03, 2.724e-03 |
| hky2 | 3 | R6 | kingman | 1.042e-03, 2.012e-03, 2.996e-03 | 1.069e-03, 2.076e-03, 3.084e-03 | — |
| hky2 | 3 | R6 | adaptive | 1.042e-03, 2.012e-03, 2.996e-03 | 1.059e-03, 2.067e-03, 3.074e-03 | — |
| gtr | 1 | JC | kingman | 8.950e-04 | 9.022e-04 | 4.760e-04 |
| gtr | 1 | JC | adaptive | 8.950e-04 | 8.932e-04 | 4.760e-04 |
| gtr | 1 | K2 | kingman | 8.950e-04 | 9.023e-04 | 4.763e-04 |
| gtr | 1 | K2 | adaptive | 8.950e-04 | 8.933e-04 | 4.763e-04 |
| gtr | 1 | R6 | kingman | 8.950e-04 | 9.917e-04 | — |
| gtr | 1 | R6 | adaptive | 8.950e-04 | 9.765e-04 | — |
| gtr | 2 | JC | kingman | 8.950e-04, 1.728e-03 | 9.024e-04, 1.777e-03 | 6.255e-04, 1.137e-03 |
| gtr | 2 | JC | adaptive | 8.950e-04, 1.728e-03 | 8.934e-04, 1.768e-03 | 6.255e-04, 1.137e-03 |
| gtr | 2 | K2 | kingman | 8.950e-04, 1.728e-03 | 9.025e-04, 1.777e-03 | 6.256e-04, 1.138e-03 |
| gtr | 2 | K2 | adaptive | 8.950e-04, 1.728e-03 | 8.935e-04, 1.768e-03 | 6.256e-04, 1.138e-03 |
| gtr | 2 | R6 | kingman | 8.950e-04, 1.728e-03 | 1.056e-03, 2.079e-03 | — |
| gtr | 2 | R6 | adaptive | 8.950e-04, 1.728e-03 | 9.508e-04, 1.882e-03 | — |
| gtr | 3 | JC | kingman | 8.950e-04, 1.728e-03, 2.573e-03 | 9.027e-04, 1.776e-03, 2.642e-03 | 8.030e-04, 1.594e-03, 2.356e-03 |
| gtr | 3 | JC | adaptive | 8.950e-04, 1.728e-03, 2.573e-03 | 8.938e-04, 1.767e-03, 2.634e-03 | 8.030e-04, 1.594e-03, 2.356e-03 |
| gtr | 3 | K2 | kingman | 8.950e-04, 1.728e-03, 2.573e-03 | 9.030e-04, 1.777e-03, 2.643e-03 | 8.031e-04, 1.594e-03, 2.356e-03 |
| gtr | 3 | K2 | adaptive | 8.950e-04, 1.728e-03, 2.573e-03 | 8.939e-04, 1.767e-03, 2.634e-03 | 8.031e-04, 1.594e-03, 2.356e-03 |
| gtr | 3 | R6 | kingman | 8.950e-04, 1.728e-03, 2.573e-03 | 9.770e-04, 1.923e-03, 2.860e-03 | — |
| gtr | 3 | R6 | adaptive | 8.950e-04, 1.728e-03, 2.573e-03 | 9.781e-04, 1.934e-03, 2.883e-03 | — |

## Per-K branch rates (n_out=3 cells, both tools)

The optimiser variable vector `[K_0, K_1, K_2, K_3, K_4]` parameterises
the ladder branches in the est-sfs topology:
``I →(K_0) n_1 →(K_1) O_1`` and ``n_1 →(K_2) n_2 →(K_3,K_4) (O_2, O_3)``.
fastDFE uses the same naming.

| sim | model | prior | K_0 (I→n_1) | K_1 (n_1→O_1) | K_2 (n_1→n_2) | K_3 (n_2→O_2) | K_4 (n_2→O_3) | tool |
|---|---|---|---|---|---|---|---|---|
| jc | JC | kingman | 4.383e-04 | 6.191e-04 | 5.131e-04 | 1.103e-03 | 2.099e-03 | Ancestree |
| jc | JC | kingman | 3.851e-04 | 5.549e-04 | 4.655e-04 | 9.923e-04 | 1.875e-03 | fastDFE |
| jc | JC | adaptive | 4.279e-04 | 6.191e-04 | 5.131e-04 | 1.103e-03 | 2.099e-03 | Ancestree |
| jc | JC | adaptive | 3.851e-04 | 5.549e-04 | 4.655e-04 | 9.923e-04 | 1.875e-03 | fastDFE |
| jc | K2 | kingman | 4.383e-04 | 6.191e-04 | 5.131e-04 | 1.103e-03 | 2.099e-03 | Ancestree |
| jc | K2 | kingman | 3.851e-04 | 5.548e-04 | 4.654e-04 | 9.923e-04 | 1.874e-03 | fastDFE |
| jc | K2 | adaptive | 4.279e-04 | 6.191e-04 | 5.131e-04 | 1.103e-03 | 2.099e-03 | Ancestree |
| jc | K2 | adaptive | 3.851e-04 | 5.548e-04 | 4.654e-04 | 9.923e-04 | 1.874e-03 | fastDFE |
| jc | R6 | kingman | 4.405e-04 | 6.210e-04 | 5.141e-04 | 1.109e-03 | 2.109e-03 | Ancestree |
| jc | R6 | kingman | nan | nan | nan | nan | nan | fastDFE |
| jc | R6 | adaptive | 4.296e-04 | 6.212e-04 | 5.156e-04 | 1.106e-03 | 2.105e-03 | Ancestree |
| jc | R6 | adaptive | nan | nan | nan | nan | nan | fastDFE |
| hky2 | JC | kingman | 4.388e-04 | 6.187e-04 | 5.133e-04 | 1.103e-03 | 2.099e-03 | Ancestree |
| hky2 | JC | kingman | 3.847e-04 | 5.542e-04 | 4.657e-04 | 9.922e-04 | 1.873e-03 | fastDFE |
| hky2 | JC | adaptive | 4.283e-04 | 6.187e-04 | 5.133e-04 | 1.103e-03 | 2.099e-03 | Ancestree |
| hky2 | JC | adaptive | 3.847e-04 | 5.542e-04 | 4.657e-04 | 9.922e-04 | 1.873e-03 | fastDFE |
| hky2 | K2 | kingman | 4.389e-04 | 6.187e-04 | 5.133e-04 | 1.103e-03 | 2.099e-03 | Ancestree |
| hky2 | K2 | kingman | 3.847e-04 | 5.542e-04 | 4.657e-04 | 9.923e-04 | 1.873e-03 | fastDFE |
| hky2 | K2 | adaptive | 4.284e-04 | 6.187e-04 | 5.133e-04 | 1.103e-03 | 2.099e-03 | Ancestree |
| hky2 | K2 | adaptive | 3.847e-04 | 5.542e-04 | 4.657e-04 | 9.923e-04 | 1.873e-03 | fastDFE |
| hky2 | R6 | kingman | 4.435e-04 | 6.252e-04 | 5.181e-04 | 1.115e-03 | 2.123e-03 | Ancestree |
| hky2 | R6 | kingman | nan | nan | nan | nan | nan | fastDFE |
| hky2 | R6 | adaptive | 4.332e-04 | 6.259e-04 | 5.189e-04 | 1.115e-03 | 2.122e-03 | Ancestree |
| hky2 | R6 | adaptive | nan | nan | nan | nan | nan | fastDFE |
| gtr | JC | kingman | 3.744e-04 | 5.283e-04 | 4.452e-04 | 9.567e-04 | 1.823e-03 | Ancestree |
| gtr | JC | kingman | 3.268e-04 | 4.762e-04 | 4.059e-04 | 8.609e-04 | 1.624e-03 | fastDFE |
| gtr | JC | adaptive | 3.654e-04 | 5.283e-04 | 4.452e-04 | 9.566e-04 | 1.823e-03 | Ancestree |
| gtr | JC | adaptive | 3.268e-04 | 4.762e-04 | 4.059e-04 | 8.609e-04 | 1.624e-03 | fastDFE |
| gtr | K2 | kingman | 3.745e-04 | 5.285e-04 | 4.452e-04 | 9.569e-04 | 1.823e-03 | Ancestree |
| gtr | K2 | kingman | 3.268e-04 | 4.764e-04 | 4.058e-04 | 8.610e-04 | 1.624e-03 | fastDFE |
| gtr | K2 | adaptive | 3.654e-04 | 5.285e-04 | 4.452e-04 | 9.569e-04 | 1.824e-03 | Ancestree |
| gtr | K2 | adaptive | 3.268e-04 | 4.764e-04 | 4.058e-04 | 8.610e-04 | 1.624e-03 | fastDFE |
| gtr | R6 | kingman | 4.052e-04 | 5.717e-04 | 4.819e-04 | 1.036e-03 | 1.973e-03 | Ancestree |
| gtr | R6 | kingman | nan | nan | nan | nan | nan | fastDFE |
| gtr | R6 | adaptive | 3.995e-04 | 5.786e-04 | 4.874e-04 | 1.047e-03 | 1.996e-03 | Ancestree |
| gtr | R6 | adaptive | nan | nan | nan | nan | nan | fastDFE |

## Fitted κ vs simulation truth (K2 model cells)

Effective transition/transversion ratio. HKY sims display the literal κ.
The ``gtr`` sim displays the on-paper Ts/Tv ratio implied by its
exchangeability rates (transversion-symmetric construction).

| sim | n_out | prior | truth κ | Ancestree fitted κ | fastDFE fitted κ |
|---|---|---|---|---|---|
| jc | 1 | kingman | 1.0000 | 0.9097 | 0.8700 |
| jc | 1 | adaptive | 1.0000 | 0.9086 | 0.8700 |
| jc | 2 | kingman | 1.0000 | 1.0143 | 1.0185 |
| jc | 2 | adaptive | 1.0000 | 1.0145 | 1.0185 |
| jc | 3 | kingman | 1.0000 | 0.9938 | 0.9888 |
| jc | 3 | adaptive | 1.0000 | 0.9937 | 0.9888 |
| hky2 | 1 | kingman | 2.0000 | 1.8604 | 1.7419 |
| hky2 | 1 | adaptive | 2.0000 | 1.8561 | 1.7419 |
| hky2 | 2 | kingman | 2.0000 | 2.0480 | 2.0496 |
| hky2 | 2 | adaptive | 2.0000 | 2.0469 | 2.0496 |
| hky2 | 3 | kingman | 2.0000 | 2.0326 | 2.0335 |
| hky2 | 3 | adaptive | 2.0000 | 2.0320 | 2.0335 |
| gtr | 1 | kingman | 4.0000 | 3.7918 | 3.7627 |
| gtr | 1 | adaptive | 4.0000 | 3.7895 | 3.7627 |
| gtr | 2 | kingman | 4.0000 | 4.0277 | 4.0560 |
| gtr | 2 | adaptive | 4.0000 | 4.0278 | 4.0560 |
| gtr | 3 | kingman | 4.0000 | 3.8351 | 3.8733 |
| gtr | 3 | adaptive | 4.0000 | 3.8351 | 3.8733 |

## Fitted GTR exchangeability rates vs simulation truth (R6 cells, Ancestree)

Truth rates use the GTR parameterisation. For HKY sims the truth column
shows the collapsed-equivalent (transitions=κ, transversions=1). For the
``gtr`` sim it is the literal simulator rate vector. All rates are scaled
so ``rate_AG = 1`` (Ancestree's reference scale during the joint MLE).

| sim | n_out | prior | r_AC (t) | r_AG (t) | r_AT (t) | r_CG (t) | r_CT (t) | r_GT (t) | r_AC (f) | r_AG (f) | r_AT (f) | r_CG (f) | r_CT (f) | r_GT (f) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jc | 1 | kingman | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.029 | 1.000 | 1.028 | 1.134 | 1.073 | 1.132 |
| jc | 2 | kingman | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.985 | 1.000 | 0.952 | 1.099 | 0.966 | 1.058 |
| jc | 3 | kingman | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 | 1.008 | 1.019 | 1.003 | 1.010 |
| jc | 1 | adaptive | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.082 | 1.000 | 1.074 | 1.170 | 1.000 | 1.097 |
| jc | 2 | adaptive | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.969 | 1.000 | 0.991 | 1.040 | 1.016 | 0.985 |
| jc | 3 | adaptive | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.003 | 1.000 | 1.010 | 1.003 | 1.007 | 1.010 |
| hky2 | 1 | kingman | 0.500 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.496 | 1.000 | 0.486 | 0.572 | 0.956 | 0.565 |
| hky2 | 2 | kingman | 0.500 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.473 | 1.000 | 0.480 | 0.519 | 0.991 | 0.471 |
| hky2 | 3 | kingman | 0.500 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.484 | 1.000 | 0.491 | 0.521 | 1.009 | 0.481 |
| hky2 | 1 | adaptive | 0.500 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.499 | 1.000 | 0.491 | 0.575 | 0.963 | 0.565 |
| hky2 | 2 | adaptive | 0.500 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.609 | 1.000 | 0.578 | 0.727 | 0.594 | 0.685 |
| hky2 | 3 | adaptive | 0.500 | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.490 | 1.000 | 0.490 | 0.515 | 1.010 | 0.481 |
| gtr | 1 | kingman | 0.250 | 1.000 | 0.250 | 0.250 | 1.000 | 0.250 | 0.243 | 1.000 | 0.373 | 0.147 | 0.927 | 0.252 |
| gtr | 2 | kingman | 0.250 | 1.000 | 0.250 | 0.250 | 1.000 | 0.250 | 0.233 | 1.000 | 0.502 | 0.123 | 0.776 | 0.333 |
| gtr | 3 | kingman | 0.250 | 1.000 | 0.250 | 0.250 | 1.000 | 0.250 | 0.236 | 1.000 | 0.374 | 0.144 | 0.981 | 0.274 |
| gtr | 1 | adaptive | 0.250 | 1.000 | 0.250 | 0.250 | 1.000 | 0.250 | 0.244 | 1.000 | 0.379 | 0.148 | 0.944 | 0.254 |
| gtr | 2 | adaptive | 0.250 | 1.000 | 0.250 | 0.250 | 1.000 | 0.250 | 0.212 | 1.000 | 0.327 | 0.150 | 0.951 | 0.252 |
| gtr | 3 | adaptive | 0.250 | 1.000 | 0.250 | 0.250 | 1.000 | 0.250 | 0.255 | 1.000 | 0.373 | 0.182 | 0.935 | 0.230 |

## Inference wall-clock runtime (seconds, inference step only, excluding I/O)

Runtime is reported per (sim, n_out, model). The est-sfs baseline has no
prior wildcard, so we list one row per Ancestree prior at the first entry
in ``priors`` (other priors run at near-identical cost).

| sim | n_out | model | Ancestree (s) | est-sfs (s) | fastDFE (s) |
|---|---|---|---|---|---|
| jc | 1 | JC | 0.651 | 2.962 | — |
| jc | 1 | K2 | 0.783 | 3.239 | — |
| jc | 1 | R6 | 1.517 | 3.064 | — |
| jc | 2 | JC | 1.132 | 5.185 | — |
| jc | 2 | K2 | 1.596 | 5.527 | — |
| jc | 2 | R6 | 3.169 | 6.417 | — |
| jc | 3 | JC | 2.645 | 13.169 | — |
| jc | 3 | K2 | 2.878 | 17.520 | — |
| jc | 3 | R6 | 9.716 | 19.668 | — |
| hky2 | 1 | JC | 0.647 | 2.792 | — |
| hky2 | 1 | K2 | 0.772 | 2.853 | — |
| hky2 | 1 | R6 | 1.799 | 3.065 | — |
| hky2 | 2 | JC | 1.162 | 5.005 | — |
| hky2 | 2 | K2 | 1.554 | 5.240 | — |
| hky2 | 2 | R6 | 3.089 | 6.263 | — |
| hky2 | 3 | JC | 2.605 | 12.813 | — |
| hky2 | 3 | K2 | 3.839 | 17.082 | — |
| hky2 | 3 | R6 | 10.274 | 15.967 | — |
| gtr | 1 | JC | 0.647 | 2.855 | — |
| gtr | 1 | K2 | 0.914 | 2.460 | — |
| gtr | 1 | R6 | 1.851 | 2.691 | — |
| gtr | 2 | JC | 1.097 | 4.436 | — |
| gtr | 2 | K2 | 1.601 | 4.784 | — |
| gtr | 2 | R6 | 3.142 | 7.032 | — |
| gtr | 3 | JC | 2.340 | 13.026 | — |
| gtr | 3 | K2 | 3.734 | 16.578 | — |
| gtr | 3 | R6 | 12.915 | 18.428 | — |
