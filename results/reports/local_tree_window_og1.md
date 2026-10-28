# Local-tree window-size + recombination-misspecification benchmark

One neutral msprime ARG (ingroup-only, 20 haplotypes, L=20,000,000 bp, mu=1.25e-08, true rec_rate=1e-08, Ne=30,000, JC69, seed=42): 1,153,635 sites, 164,621 true local trees.

`LocalTreeInference` is run per cell and scored against the msprime ancestral truth. Each cell shows **Brier / MAP / P(true)** (Brier lower = better). The three sweeps are independent 1-D slices through the same origin (window=8snp, r=r_true, mu=mu_true); the true-ARG ceiling scores the genuine simulated genealogy through `ARGBasedInference`.

## Window size x assumed recombination rate (mu well-specified)

| window (SNPs) | 0.1*r_true | r_true | 10*r_true | runtime |
| --- | --- | --- | --- | --- |
| 1 | 0.018 / 0.989 / 0.987 | 0.012 / 0.994 / 0.986 | 0.013 / 0.994 / 0.982 | 6.0m |
| 2 | 0.019 / 0.989 / 0.986 | 0.012 / 0.994 / 0.988 | 0.013 / 0.993 / 0.986 | 4.5m |
| 5 | 0.022 / 0.987 / 0.984 | 0.015 / 0.991 / 0.988 | 0.015 / 0.991 / 0.987 | 2.8m |
| 10 | 0.023 / 0.986 / 0.984 | 0.017 / 0.990 / 0.987 | 0.015 / 0.991 / 0.988 | 2.7m |
| 25 | 0.034 / 0.979 / 0.977 | 0.028 / 0.983 / 0.980 | 0.026 / 0.984 / 0.982 | 2.3m |
| 50 | 0.048 / 0.971 / 0.969 | 0.046 / 0.972 / 0.970 | 0.044 / 0.974 / 0.972 | 2.6m |
| 100 | 0.070 / 0.959 / 0.957 | 0.071 / 0.959 / 0.956 | 0.069 / 0.960 / 0.958 | 2.4m |
| 200 | 0.096 / 0.944 / 0.943 | 0.103 / 0.942 / 0.940 | 0.100 / 0.943 / 0.941 | 2.5m |
| 400 | 0.129 / 0.927 / 0.926 | 0.136 / 0.924 / 0.922 | 0.137 / 0.923 / 0.921 | 2.4m |
| 1000 | 0.166 / 0.908 / 0.907 | 0.173 / 0.905 / 0.904 | 0.177 / 0.903 / 0.901 | 2.6m |
| 2000 | 0.183 / 0.900 / 0.898 | 0.189 / 0.898 / 0.895 | 0.194 / 0.894 / 0.892 | 2.6m |
| **true ARG (ceiling)** | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 1.0m |

## Assumed mutation rate (window=8 SNPs, r=r_true)

The inference's assumed `mu` is swept as a multiple of the true simulation mu; the true mu (and the simulated data) are unchanged. A misspecified mu rescales the inferred TMRCAs through both the emission rate and the time-grid calibration.

| assumed mu | 0.01*mu_true | 0.1*mu_true | mu_true | 10*mu_true | 100*mu_true | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Brier / MAP / P(true) | 0.014 / 0.991 / 0.989 | 0.014 / 0.992 / 0.989 | 0.015 / 0.991 / 0.988 | 0.022 / 0.986 / 0.984 | 0.038 / 0.977 / 0.975 | 5.2m |
| **true ARG (ceiling)** | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 1.0m |

## Phasing switch-error rate (window=8 SNPs, r=r_true, mu=mu_true)

A switch-error process is injected into the ingroup genotypes fed to inference: consecutive haplotypes are paired into pseudo-diploids and a per-pair phase orientation flips with the given probability at each heterozygous site. The msprime ancestral truth (and ingroup allele frequencies) are unchanged. `0%` = perfectly phased; `1%` is the rate used in the main-text robustness section.

| switch rate | 0% | 5% | 10% | 20% | 50% | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Brier / MAP / P(true) | 0.015 / 0.991 / 0.988 | 0.017 / 0.990 / 0.987 | 0.020 / 0.988 / 0.985 | 0.026 / 0.984 / 0.981 | 0.040 / 0.973 / 0.972 | 4.7m |
| **true ARG (ceiling)** | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 1.0m |
