# Robustness heatmap — per-cell wall time (seconds)

Times are in seconds for the in-process Ancestree / parsimony / majority methods (per-cell ``time.perf_counter`` around the inference call).

| Method | n_out | baseline | cpg_hypermut | strong_ils | outgroup_clade | slim |
| --- | --- | --- | --- | --- | --- | --- |
| true ARG (oracle) | 0 | 19.0s | 30.4s | 34.0s | 51.9s | 4.9m |
| true ARG (oracle) | 1 | 23.5s | 16.1s | 41.6s | 44.4s | 1.6m |
| true ARG (oracle) | 3 | 26.9s | 30.1s | 1.0m | 45.4s | 5.8m |
| true ARG (oracle) | 10 | 36.0s | 35.2s | 44.7s | 57.0s | 5.4m |
| max-parsimony ($\mu\!\to\!0$) | 0 | 16.6s | 22.5s | 41.0s | 45.6s | 4.2m |
| max-parsimony ($\mu\!\to\!0$) | 1 | 20.5s | 16.7s | 30.6s | 1.1m | 1.1m |
| max-parsimony ($\mu\!\to\!0$) | 3 | 24.3s | 27.6s | 47.5s | 53.5s | 5.0m |
| max-parsimony ($\mu\!\to\!0$) | 10 | 33.0s | 37.5s | 55.2s | 54.0s | 5.3m |
| local-tree mode | 0 | 1.3m | 51.2s | 1.3m | 1.1m | 12.8m |
| local-tree mode | 1 | 1.4m | 55.4s | 1.0m | 2.7m | 3.0m |
| local-tree mode | 3 | 1.5m | 56.1s | 1.3m | 1.2m | 14.7m |
| local-tree mode | 10 | 2.3m | 1.3m | 1.7m | 1.8m | 15.6m |
| SINGER | 0 | 8.18h | 3.38h | 22.35h | 18.98h | 9.39h |
| SINGER | 1 | 10.88h | 4.30h | 25.42h | 29.20h | 3.84h |
| SINGER | 3 | 23.34h | 7.77h | 40.99h | 44.50h | 30.79h |
| SINGER | 10 | 2.7d | 15.35h | 3.5d | 3.5d | 5.6d |
| SINGER (fixed-tree) | 0 | 8.02h | 3.34h | 23.04h | 19.01h | 9.37h |
| SINGER (fixed-tree) | 1 | 10.42h | 3.96h | 29.88h | 26.24h | 3.22h |
| SINGER (fixed-tree) | 3 | 25.49h | 6.87h | 41.17h | 44.92h | 28.67h |
| SINGER (fixed-tree) | 10 | 2.7d | 15.28h | 3.7d | 3.5d | 5.5d |
| tsinfer + tsdate | 0 | 6.9m | 7.4m | 6.1m | 6.2m | 55.5m |
| tsinfer + tsdate | 1 | 6.9m | 7.4m | 6.1m | 6.2m | 17.2m |
| tsinfer + tsdate | 3 | 7.3m | 8.4m | 6.5m | 6.4m | 1.04h |
| tsinfer + tsdate | 10 | 8.3m | 9.3m | 6.7m | 7.0m | 1.04h |
| Relate | 0 | 1.6m | 2.0m | 39.3s | 51.7s | 4.6m |
| Relate | 1 | 1.7m | 1.9m | 1.0m | 47.7s | 1.5m |
| Relate | 3 | 1.9m | 2.8m | 1.2m | 53.6s | 6.4m |
| Relate | 10 | 2.9m | 4.0m | 1.7m | 1.4m | 8.0m |
| fixed-tree mode | 0 | 51.4s | 1.2m | 4.5s | 4.9s | 19.9s |
| fixed-tree mode | 1 | 1.3m | 2.4m | 1.2m | 2.0m | 4.3m |
| fixed-tree mode | 3 | 1.4m | 2.2m | 2.0m | 1.6m | 11.7m |
| fixed-tree mode | 10 | 7.8m | 22.6m | 18.4m | 9.2m | 44.9m |
| fixed-tree (adaptive) | 0 | 51.3s | 1.3m | 8.2s | 8.3s | 22.1s |
| fixed-tree (adaptive) | 1 | 1.4m | 2.2m | 1.5m | 1.3m | 4.6m |
| fixed-tree (adaptive) | 3 | 1.5m | 2.6m | 1.9m | 2.2m | 11.5m |
| fixed-tree (adaptive) | 10 | 7.8m | 22.2m | 19.5m | 8.6m | 44.8m |
| majority outgroup | 0 | 32.6s | 42.1s | 2.0s | 2.2s | 8.6s |
| majority outgroup | 1 | 32.7s | 42.7s | 1.9s | 2.2s | 2.6s |
| majority outgroup | 3 | 33.7s | 44.9s | 2.2s | 2.4s | 9.5s |
| majority outgroup | 10 | 39.5s | 51.9s | 2.3s | 2.8s | 10.6s |
