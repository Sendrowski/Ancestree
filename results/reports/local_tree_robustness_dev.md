# Local-tree DEV robustness benchmark (not in manuscript)

Defaults: window=30snp, block=~4snp

| axis | cell | bp/SNP | rho | RF | KC | logbias | slope | Brier |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| density | sparse | 1559 | 0.491 | 0.607 | 3.97e+05 | +0.403 | 0.148 | 0.3043 |
| density | mid | 373 | 0.766 | 0.424 | 3.23e+05 | +0.153 | 0.322 | 0.2555 |
| density | dense | 94 | 0.903 | 0.264 | 2.18e+05 | +0.063 | 0.592 | 0.2152 |
| hetero | rec_hotspots:overall | 377 | 0.721 | 0.430 | 3.06e+05 | +0.162 | 0.344 | 0.2562 |
| hetero | rec_hotspots:hi_rate_seg | 377 | 0.542 | 0.570 | 3.65e+05 | +0.307 | 0.185 | 0.2934 |
| hetero | rec_hotspots:lo_rate_seg | 377 | 0.882 | 0.291 | 2.48e+05 | +0.018 | 0.491 | 0.2193 |
| hetero | mu_hetero:overall | 337 | 0.468 | 0.471 | 3.61e+05 | -0.238 | 0.348 | 0.2552 |
| hetero | mu_hetero:hi_rate_seg | 337 | 0.789 | 0.386 | 2.96e+05 | +0.704 | 0.598 | 0.2530 |
| hetero | mu_hetero:lo_rate_seg | 337 | 0.610 | 0.556 | 4.27e+05 | -1.180 | 0.084 | 0.2789 |
| contigs | separate (K=5) | - | 0.752 | 0.427 | 3.31e+05 | +0.168 | 0.311 | 0.2534 |
| contigs | together (K=5) | - | 0.750 | 0.425 | 3.16e+05 | +0.155 | 0.316 | 0.2514 |
