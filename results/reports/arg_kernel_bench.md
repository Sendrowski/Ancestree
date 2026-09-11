# Kernel benchmark — all three modes on one shared dataset

Simulation (msprime, seed=42): 17 ingroup + 3 outgroups (splits [50000, 100000, 150000] gens), length=100,000,000 bp, mu=1.25e-08, Ne=10,000, JC69 — 778,783 sites, 250,657 trees. Every config runs in its own subprocess (one config per process), so peak RSS is that config's alone; the parent sim wall is excluded.

| configuration | runtime (s) | peak mem (GB) |
|---|---:|---:|
| ARG mode / 1w | 34.8 | 1.4 |
| ARG mode / 2w | 18.2 | 2.2 |
| ARG mode / 4w | 10.7 | 2.7 |
| ARG mode / 8w | 8.3 | 3.9 |
| fixed-tree mode / 1w | 12.5 | 1.4 |
| fixed-tree mode / 2w | 13.2 | 2.7 |
| local-tree mode / 1 thread | 116.5 | 2.3 |
| local-tree mode / 2 threads | 64.4 | 2.2 |
| local-tree mode / 4 threads | 39.3 | 2.2 |
| local-tree mode / 8 threads | 32.0 | 2.2 |
