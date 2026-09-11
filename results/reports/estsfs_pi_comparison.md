# B3-π — AdaptivePolarizationPrior π_i sanity comparison

Larger dataset designed so the per-bin Adaptive MLE has enough data
to actually converge (≥ 100 sites per fittable bin under Kingman SFS).
Should recover Kingman within sampling noise.

## Simulation
- ingroup: `10` haps (Ne=10000)
- outgroups: 3 nested splits at `[50000.0, 100000.0, 150000.0]` gens
- sequence length: `2e+06`
- mutation rate: `1e-08` per-site per-gen
- sim model: `JC` (κ=1.0)
- seed: `42`

## Per-bin polarization probabilities π_i = P(major ancestral | minor count = i)

n_sites = sites that contributed to the Ancestree per-bin MLE (only
biallelic + fully-observed ingroup sites count). Bins ``i=0`` and
``i=n_ingroup`` are fixed by convention (no fit). Folded SFS:
``π_{n-i} = 1 - π_i``.

| i | n_sites | Kingman | Ancestree | fastDFE | Δ(anc-king) | Δ(fd-king) | Δ(anc-fd) |
|---|---|---|---|---|---|---|---|
|  0 |     0 | 1.0000 | 1.0000 | 1.0000 |   0.0000 |   0.0000 |   0.0000 |
|  1 |   466 | 0.9000 | 0.8864 | 0.8862 |  -0.0136 |  -0.0138 |   0.0002 |
|  2 |   285 | 0.8000 | 0.8597 | 0.8588 |   0.0597 |   0.0588 |   0.0009 |
|  3 |   223 | 0.7000 | 0.6532 | 0.6601 |  -0.0468 |  -0.0399 |  -0.0069 |
|  4 |   164 | 0.6000 | 0.5976 | 0.6000 |  -0.0024 |   0.0000 |  -0.0024 |
|  5 |     0 | 0.5000 | 0.5000 | 0.5000 |   0.0000 |   0.0000 |   0.0000 |
|  6 |     0 | 0.4000 | 0.4024 | 0.4000 |   0.0024 |  -0.0000 |   0.0024 |
|  7 |     0 | 0.3000 | 0.3468 | 0.3399 |   0.0468 |   0.0399 |   0.0069 |
|  8 |     0 | 0.2000 | 0.1403 | 0.1412 |  -0.0597 |  -0.0588 |  -0.0009 |
|  9 |     0 | 0.1000 | 0.1136 | 0.1138 |   0.0136 |   0.0138 |  -0.0002 |
| 10 |     0 | 0.0000 | 0.0000 | 0.0000 |   0.0000 |   0.0000 |   0.0000 |

## Summary (excluding boundary bins i=0 and i=10)

| comparison | n_bins | mean \|Δ\| | median \|Δ\| | max \|Δ\| |
|---|---|---|---|---|
| Ancestree vs Kingman | 9 | 0.0272 | 0.0136 | 0.0597 |
| fastDFE   vs Kingman | 9 | 0.0250 | 0.0138 | 0.0588 |
| Ancestree vs fastDFE | 9 | 0.0023 | 0.0009 | 0.0069 |
