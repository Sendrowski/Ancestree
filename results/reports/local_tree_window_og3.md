# Local-tree window-size + recombination-misspecification benchmark

One neutral msprime ARG (ingroup-only, 20 haplotypes, L=20,000,000 bp, mu=1.25e-08, true rec_rate=1e-08, Ne=30,000, JC69, seed=42): 1,153,635 sites, 164,621 true local trees.

`LocalTreeInference` is run per cell and scored against the msprime ancestral truth. Each cell shows **Brier / MAP / P(true)** (Brier lower = better). The three sweeps are independent 1-D slices through the same origin (window=30snp, r=r_true, mu=mu_true); the true-ARG ceiling scores the genuine simulated genealogy through `ARGBasedInference`.

## Window size x assumed recombination rate (mu well-specified)

| window (SNPs) | 0.1*r_true | r_true | 10*r_true | runtime |
| --- | --- | --- | --- | --- |
| 1 | 0.042 / 0.977 / 0.976 | 0.033 / 0.982 / 0.981 | 0.031 / 0.983 / 0.981 | 19.6m |
| 2 | 0.042 / 0.977 / 0.976 | 0.033 / 0.982 / 0.981 | 0.031 / 0.983 / 0.982 | 12.6m |
| 5 | 0.043 / 0.976 / 0.976 | 0.034 / 0.982 / 0.981 | 0.033 / 0.982 / 0.981 | 7.1m |
| 10 | 0.043 / 0.976 / 0.976 | 0.035 / 0.981 / 0.980 | 0.034 / 0.982 / 0.981 | 7.1m |
| 25 | 0.045 / 0.975 / 0.974 | 0.037 / 0.979 / 0.979 | 0.036 / 0.980 / 0.979 | 6.2m |
| 50 | 0.047 / 0.973 / 0.973 | 0.041 / 0.977 / 0.976 | 0.040 / 0.978 / 0.977 | 6.0m |
| 100 | 0.053 / 0.969 / 0.969 | 0.050 / 0.972 / 0.971 | 0.048 / 0.974 / 0.972 | 6.0m |
| 200 | 0.064 / 0.963 / 0.962 | 0.064 / 0.963 / 0.962 | 0.063 / 0.965 / 0.963 | 6.5m |
| 400 | 0.086 / 0.950 / 0.950 | 0.087 / 0.951 / 0.949 | 0.089 / 0.950 / 0.948 | 6.4m |
| 1000 | 0.121 / 0.932 / 0.932 | 0.126 / 0.931 / 0.929 | 0.132 / 0.927 / 0.925 | 5.9m |
| 2000 | 0.150 / 0.917 / 0.916 | 0.157 / 0.914 / 0.913 | 0.163 / 0.911 / 0.909 | 5.8m |
| **true ARG (ceiling)** | 0.033 / 0.982 / 0.982 | 0.033 / 0.982 / 0.982 | 0.033 / 0.982 / 0.982 | 1.2m |

## Assumed mutation rate (window=30 SNPs, r=r_true)

The inference's assumed `mu` is swept as a multiple of the true simulation mu; the true mu (and the simulated data) are unchanged. A misspecified mu rescales the inferred TMRCAs through both the emission rate and the time-grid calibration.

| assumed mu | 0.01*mu_true | 0.1*mu_true | mu_true | 10*mu_true | 100*mu_true | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Brier / MAP / P(true) | 0.037 / 0.980 / 0.979 | 0.037 / 0.980 / 0.979 | 0.038 / 0.979 / 0.978 | 0.045 / 0.975 / 0.974 | 0.059 / 0.967 / 0.966 | 10.4m |
| **true ARG (ceiling)** | 0.033 / 0.982 / 0.982 | 0.033 / 0.982 / 0.982 | 0.033 / 0.982 / 0.982 | 0.033 / 0.982 / 0.982 | 0.033 / 0.982 / 0.982 | 1.2m |

## Phasing switch-error rate (window=30 SNPs, r=r_true, mu=mu_true)

A switch-error process is injected into the ingroup genotypes fed to inference: consecutive haplotypes are paired into pseudo-diploids and a per-pair phase orientation flips with the given probability at each heterozygous site. The msprime ancestral truth (and ingroup allele frequencies) are unchanged. `0%` = perfectly phased; `1%` is the rate used in the main-text robustness section.

| switch rate | 0% | 5% | 10% | 20% | 50% | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Brier / MAP / P(true) | 0.038 / 0.979 / 0.978 | 0.039 / 0.979 / 0.978 | 0.040 / 0.978 / 0.977 | 0.044 / 0.975 / 0.974 | 0.052 / 0.969 / 0.968 | 10.6m |
| **true ARG (ceiling)** | 0.033 / 0.982 / 0.982 | 0.033 / 0.982 / 0.982 | 0.033 / 0.982 / 0.982 | 0.033 / 0.982 / 0.982 | 0.033 / 0.982 / 0.982 | 1.2m |
