# Local-tree window-size + recombination-misspecification benchmark

One neutral msprime ARG (ingroup-only, 20 haplotypes, L=20,000,000 bp, mu=1.25e-08, true rec_rate=1e-08, Ne=30,000, JC69, seed=42): 1,153,635 sites, 164,621 true local trees.

`LocalTreeInference` is run per cell and scored against the msprime ancestral truth. Each cell shows **Brier / MAP / P(true)** (Brier lower = better). The three sweeps are independent 1-D slices through the same origin (window=30snp, r=r_true, mu=mu_true); the true-ARG ceiling scores the genuine simulated genealogy through `ARGBasedInference`.

## Window size x assumed recombination rate (mu well-specified)

| window (SNPs) | 0.1*r_true | r_true | 10*r_true | runtime |
| --- | --- | --- | --- | --- |
| 1 | 0.049 / 0.973 / 0.971 | 0.043 / 0.978 / 0.970 | 0.044 / 0.978 / 0.967 | 5.1m |
| 2 | 0.050 / 0.973 / 0.970 | 0.043 / 0.978 / 0.972 | 0.043 / 0.978 / 0.970 | 3.6m |
| 5 | 0.054 / 0.970 / 0.968 | 0.046 / 0.975 / 0.972 | 0.046 / 0.976 / 0.973 | 2.3m |
| 10 | 0.057 / 0.968 / 0.967 | 0.050 / 0.973 / 0.971 | 0.048 / 0.974 / 0.972 | 2.3m |
| 25 | 0.067 / 0.962 / 0.961 | 0.063 / 0.965 / 0.963 | 0.060 / 0.967 / 0.965 | 2.5m |
| 50 | 0.082 / 0.954 / 0.952 | 0.080 / 0.955 / 0.953 | 0.076 / 0.958 / 0.955 | 2.4m |
| 100 | 0.102 / 0.942 / 0.941 | 0.103 / 0.943 / 0.941 | 0.101 / 0.944 / 0.942 | 2.3m |
| 200 | 0.129 / 0.928 / 0.927 | 0.133 / 0.926 / 0.924 | 0.132 / 0.927 / 0.925 | 2.3m |
| 400 | 0.160 / 0.911 / 0.910 | 0.165 / 0.910 / 0.908 | 0.168 / 0.908 / 0.906 | 2.3m |
| 1000 | 0.196 / 0.894 / 0.892 | 0.202 / 0.891 / 0.889 | 0.206 / 0.889 / 0.887 | 2.3m |
| 2000 | 0.213 / 0.885 / 0.883 | 0.220 / 0.882 / 0.880 | 0.224 / 0.880 / 0.878 | 2.3m |
| **true ARG (ceiling)** | 0.038 / 0.980 / 0.978 | 0.038 / 0.980 / 0.978 | 0.038 / 0.980 / 0.978 | 1.0m |

## Assumed mutation rate (window=30 SNPs, r=r_true)

The inference's assumed `mu` is swept as a multiple of the true simulation mu; the true mu (and the simulated data) are unchanged. A misspecified mu rescales the inferred TMRCAs through both the emission rate and the time-grid calibration.

| assumed mu | 0.01*mu_true | 0.1*mu_true | mu_true | 10*mu_true | 100*mu_true | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Brier / MAP / P(true) | 0.063 / 0.965 / 0.963 | 0.063 / 0.965 / 0.963 | 0.065 / 0.964 / 0.962 | 0.070 / 0.960 / 0.959 | 0.080 / 0.955 / 0.953 | 3.5m |
| **true ARG (ceiling)** | 0.038 / 0.980 / 0.978 | 0.038 / 0.980 / 0.978 | 0.038 / 0.980 / 0.978 | 0.038 / 0.980 / 0.978 | 0.038 / 0.980 / 0.978 | 1.0m |

## Phasing switch-error rate (window=30 SNPs, r=r_true, mu=mu_true)

A switch-error process is injected into the ingroup genotypes fed to inference: consecutive haplotypes are paired into pseudo-diploids and a per-pair phase orientation flips with the given probability at each heterozygous site. The msprime ancestral truth (and ingroup allele frequencies) are unchanged. `0%` = perfectly phased; `1%` is the rate used in the main-text robustness section.

| switch rate | 0% | 5% | 10% | 20% | 50% | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Brier / MAP / P(true) | 0.065 / 0.964 / 0.962 | 0.069 / 0.962 / 0.959 | 0.074 / 0.958 / 0.956 | 0.084 / 0.952 / 0.950 | 0.101 / 0.940 / 0.940 | 3.5m |
| **true ARG (ceiling)** | 0.038 / 0.980 / 0.978 | 0.038 / 0.980 / 0.978 | 0.038 / 0.980 / 0.978 | 0.038 / 0.980 / 0.978 | 0.038 / 0.980 / 0.978 | 1.0m |
