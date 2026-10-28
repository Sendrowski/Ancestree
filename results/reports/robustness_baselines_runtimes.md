# Robustness heatmap — per-cell wall time (seconds)

Times are in seconds for the in-process Ancestree / parsimony / majority methods (per-cell ``time.perf_counter`` around the inference call).

| Method | n_out | baseline | cpg_hypermut | strong_ils | outgroup_clade | slim |
| --- | --- | --- | --- | --- | --- | --- |
| true ARG (oracle) | 0 | 34.6s | 30.5s | 40.7s | 51.7s | 4.9m |
| true ARG (oracle) | 1 | 31.6s | 25.0s | 38.8s | 44.8s | 1.3m |
| true ARG (oracle) | 3 | 37.1s | 21.5s | 53.7s | 47.6s | 4.6m |
| true ARG (oracle) | 10 | 49.2s | 35.9s | 54.1s | 1.0m | 5.0m |
| max-parsimony ($\mu\!\to\!0$) | 0 | 22.9s | 19.2s | 30.1s | 37.9s | 4.0m |
| max-parsimony ($\mu\!\to\!0$) | 1 | 48.1s | 22.6s | 50.9s | 47.7s | 1.4m |
| max-parsimony ($\mu\!\to\!0$) | 3 | 1.1m | 53.0s | 1.3m | 43.5s | 4.2m |
| max-parsimony ($\mu\!\to\!0$) | 10 | 1.2m | 37.7s | 1.7m | 58.2s | 4.5m |
| local-tree mode | 0 | 1.4m | 59.9s | 1.1m | 1.2m | 7.9m |
| local-tree mode | 1 | 1.7m | 1.2m | 1.2m | 1.3m | 2.3m |
| local-tree mode | 3 | 1.8m | 1.1m | 1.4m | 1.5m | 8.8m |
| local-tree mode | 10 | 2.4m | 1.5m | 1.7m | 1.8m | 13.4m |
| SINGER | 0 | 8.41h | 3.47h | 22.72h | 19.26h | 10.06h |
| SINGER | 1 | 11.21h | 4.32h | 25.62h | 29.47h | 4.08h |
| SINGER | 3 | 23.39h | 7.81h | 40.98h | 44.81h | 31.70h |
| SINGER | 10 | 2.7d | 15.37h | 3.5d | 3.5d | 5.7d |
| SINGER (fixed-tree) | 0 | 8.17h | 3.56h | 23.22h | 19.11h | 10.16h |
| SINGER (fixed-tree) | 1 | 10.54h | 4.04h | 30.24h | 26.55h | 3.47h |
| SINGER (fixed-tree) | 3 | 25.59h | 6.98h | 41.40h | 45.31h | 29.69h |
| SINGER (fixed-tree) | 10 | 2.7d | 15.38h | 3.7d | 3.6d | 5.6d |
| tsinfer + tsdate | 0 | 7.1m | 7.3m | 6.0m | 6.3m | 55.0m |
| tsinfer + tsdate | 1 | 7.1m | 7.4m | 6.2m | 6.2m | 16.7m |
| tsinfer + tsdate | 3 | 7.5m | 8.2m | 6.3m | 6.3m | 1.03h |
| tsinfer + tsdate | 10 | 8.8m | 9.2m | 6.7m | 6.9m | 1.04h |
| Relate | 0 | 1.7m | 1.9m | 44.5s | 55.8s | 6.0m |
| Relate | 1 | 2.1m | 2.1m | 54.8s | 51.8s | 1.6m |
| Relate | 3 | 2.1m | 2.5m | 1.1m | 1.0m | 4.5m |
| Relate | 10 | 3.1m | 3.8m | 1.7m | 1.6m | 7.5m |
| fixed-tree mode | 0 | 4.6s | 3.3s | 2.7s | 2.9s | 19.5s |
| fixed-tree mode | 1 | 56.5s | 46.3s | 37.3s | 1.9m | 3.3m |
| fixed-tree mode | 3 | 55.2s | 1.3m | 53.9s | 2.7m | 15.1m |
| fixed-tree mode | 10 | 10.3m | 28.5m | 25.0m | 10.9m | 55.0m |
| fixed-tree (adaptive) | 0 | 8.7s | 6.4s | 9.1s | 8.3s | 33.7s |
| fixed-tree (adaptive) | 1 | 1.5m | 58.0s | 57.9s | 1.3m | 3.0m |
| fixed-tree (adaptive) | 3 | 1.1m | 1.6m | 1.2m | 1.0m | 13.2m |
| fixed-tree (adaptive) | 10 | 10.2m | 28.7m | 22.9m | 11.5m | 52.1m |
| majority outgroup | 0 | 40.2s | 55.4s | 2.7s | 3.1s | 13.8s |
| majority outgroup | 1 | 32.1s | 42.3s | 2.1s | 2.3s | 3.7s |
| majority outgroup | 3 | 36.6s | 49.0s | 2.3s | 2.7s | 13.5s |
| majority outgroup | 10 | 44.9s | 1.0m | 2.7s | 3.1s | 15.4s |
