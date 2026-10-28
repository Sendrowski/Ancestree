# Local-tree window-size + recombination-misspecification benchmark

One neutral msprime ARG (ingroup-only, 20 haplotypes, L=20,000,000 bp, mu=1.25e-08, true rec_rate=1e-08, Ne=30,000, JC69, seed=42): 1,153,635 sites, 164,621 true local trees.

`LocalTreeInference` is run per cell and scored against the msprime ancestral truth. Each cell shows **Brier / MAP / P(true)** (Brier lower = better). The three sweeps are independent 1-D slices through the same origin (window=8snp, r=r_true, mu=mu_true); the true-ARG ceiling scores the genuine simulated genealogy through `ARGBasedInference`.

## Window size x assumed recombination rate (mu well-specified)

| window (SNPs) | 0.1*r_true | r_true | 10*r_true | runtime |
| --- | --- | --- | --- | --- |
| 1 | 0.011 / 0.993 / 0.992 | 0.005 / 0.997 / 0.995 | 0.006 / 0.996 / 0.994 | 28.9m |
| 2 | 0.012 / 0.992 / 0.991 | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 18.4m |
| 5 | 0.012 / 0.992 / 0.991 | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 9.8m |
| 10 | 0.012 / 0.992 / 0.991 | 0.005 / 0.996 / 0.995 | 0.005 / 0.997 / 0.995 | 9.8m |
| 25 | 0.013 / 0.991 / 0.990 | 0.007 / 0.995 / 0.994 | 0.006 / 0.996 / 0.995 | 7.9m |
| 50 | 0.016 / 0.989 / 0.988 | 0.010 / 0.993 / 0.992 | 0.009 / 0.994 / 0.992 | 8.3m |
| 100 | 0.022 / 0.985 / 0.984 | 0.019 / 0.987 / 0.986 | 0.017 / 0.989 / 0.987 | 7.7m |
| 200 | 0.034 / 0.978 / 0.977 | 0.033 / 0.979 / 0.978 | 0.032 / 0.980 / 0.978 | 8.2m |
| 400 | 0.054 / 0.967 / 0.966 | 0.056 / 0.966 / 0.965 | 0.058 / 0.966 / 0.964 | 8.7m |
| 1000 | 0.089 / 0.949 / 0.948 | 0.096 / 0.946 / 0.944 | 0.101 / 0.942 / 0.940 | 9.2m |
| 2000 | 0.117 / 0.933 / 0.932 | 0.127 / 0.929 / 0.928 | 0.134 / 0.926 / 0.924 | 9.6m |
| **true ARG (ceiling)** | 0.002 / 0.999 / 0.998 | 0.002 / 0.999 / 0.998 | 0.002 / 0.999 / 0.998 | 1.3m |

## Assumed mutation rate (window=8 SNPs, r=r_true)

The inference's assumed `mu` is swept as a multiple of the true simulation mu; the true mu (and the simulated data) are unchanged. A misspecified mu rescales the inferred TMRCAs through both the emission rate and the time-grid calibration.

| assumed mu | 0.01*mu_true | 0.1*mu_true | mu_true | 10*mu_true | 100*mu_true | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Brier / MAP / P(true) | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 0.005 / 0.997 / 0.995 | 0.012 / 0.992 / 0.991 | 0.025 / 0.984 / 0.983 | 16.1m |
| **true ARG (ceiling)** | 0.002 / 0.999 / 0.998 | 0.002 / 0.999 / 0.998 | 0.002 / 0.999 / 0.998 | 0.002 / 0.999 / 0.998 | 0.002 / 0.999 / 0.998 | 1.3m |

## Phasing switch-error rate (window=8 SNPs, r=r_true, mu=mu_true)

A switch-error process is injected into the ingroup genotypes fed to inference: consecutive haplotypes are paired into pseudo-diploids and a per-pair phase orientation flips with the given probability at each heterozygous site. The msprime ancestral truth (and ingroup allele frequencies) are unchanged. `0%` = perfectly phased; `1%` is the rate used in the main-text robustness section.

| switch rate | 0% | 5% | 10% | 20% | 50% | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Brier / MAP / P(true) | 0.005 / 0.997 / 0.995 | 0.006 / 0.996 / 0.995 | 0.006 / 0.996 / 0.994 | 0.008 / 0.995 / 0.993 | 0.013 / 0.991 / 0.989 | 16.0m |
| **true ARG (ceiling)** | 0.002 / 0.999 / 0.998 | 0.002 / 0.999 / 0.998 | 0.002 / 0.999 / 0.998 | 0.002 / 0.999 / 0.998 | 0.002 / 0.999 / 0.998 | 1.3m |
