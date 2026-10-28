# Robustness heatmap — per-cell wall time (seconds)

Times are in seconds for the in-process Ancestree / parsimony / majority methods (per-cell ``time.perf_counter`` around the inference call).

| Method | n_out | baseline | cpg_hypermut | strong_ils | outgroup_clade | slim |
| --- | --- | --- | --- | --- | --- | --- |
| true ARG (oracle) | 0 | 34.6s | 30.5s | 40.7s | 51.7s | 4.9m |
| true ARG (oracle) | 1 | 31.6s | 25.0s | 38.8s | 44.8s | 1.3m |
| true ARG (oracle) | 3 | 37.1s | 21.5s | 53.7s | 47.6s | 4.6m |
| true ARG (oracle) | 10 | 49.2s | 35.9s | 54.1s | 1.0m | 5.0m |
| local-tree mode | 0 | 1.4m | 59.9s | 1.1m | 1.2m | 7.9m |
| local-tree mode | 1 | 1.7m | 1.2m | 1.2m | 1.3m | 2.3m |
| local-tree mode | 3 | 1.8m | 1.1m | 1.4m | 1.5m | 8.8m |
| local-tree mode | 10 | 2.4m | 1.5m | 1.7m | 1.8m | 13.4m |
| fixed-tree mode | 0 | 4.6s | 3.3s | 2.7s | 2.9s | 19.5s |
| fixed-tree mode | 1 | 56.5s | 46.3s | 37.3s | 1.9m | 3.3m |
| fixed-tree mode | 3 | 55.2s | 1.3m | 53.9s | 2.7m | 15.1m |
| fixed-tree mode | 10 | 10.3m | 28.5m | 25.0m | 10.9m | 55.0m |
| SINGER | 0 | 8.41h | 3.47h | 22.72h | 19.26h | 10.06h |
| SINGER | 1 | 11.21h | 4.32h | 25.62h | 29.47h | 4.08h |
| SINGER | 3 | 23.39h | 7.81h | 40.98h | 44.81h | 31.70h |
| SINGER | 10 | 2.7d | 15.37h | 3.5d | 3.5d | 5.7d |
| tsinfer + tsdate | 0 | 7.1m | 7.3m | 6.0m | 6.3m | 55.0m |
| tsinfer + tsdate | 1 | 7.1m | 7.4m | 6.2m | 6.2m | 16.7m |
| tsinfer + tsdate | 3 | 7.5m | 8.2m | 6.3m | 6.3m | 1.03h |
| tsinfer + tsdate | 10 | 8.8m | 9.2m | 6.7m | 6.9m | 1.04h |
| Relate | 0 | 1.7m | 1.9m | 44.5s | 55.8s | 6.0m |
| Relate | 1 | 2.1m | 2.1m | 54.8s | 51.8s | 1.6m |
| Relate | 3 | 2.1m | 2.5m | 1.1m | 1.0m | 4.5m |
| Relate | 10 | 3.1m | 3.8m | 1.7m | 1.6m | 7.5m |
