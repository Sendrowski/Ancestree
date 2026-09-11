# Local-tree window-size + recombination-misspecification benchmark

One neutral msprime ARG (ingroup-only, 20 haplotypes, L=100,000,000 bp, mu=1.25e-08, true rec_rate=1e-08, Ne=30,000, JC69, seed=42): 265,079 sites, 174,446 true local trees.

`LocalTreeInference` is run per cell and scored against the msprime ancestral truth. Each cell shows **Brier / MAP / P(true)** (Brier lower = better). The three sweeps are independent 1-D slices through the same origin (window=8snp, r=r_true, mu=mu_true); the true-ARG ceiling scores the genuine simulated genealogy through `ARGBasedInference`.

## Window size x assumed recombination rate (mu well-specified)

| window (SNPs) | 0.1*r_true | r_true | 10*r_true | runtime |
| --- | --- | --- | --- | --- |
| 1 | 0.210 / 0.844 / 0.810 | 0.211 / 0.843 / 0.786 | 0.233 / 0.832 / 0.757 | 28.2m |
| 2 | 0.213 / 0.843 / 0.811 | 0.214 / 0.842 / 0.796 | 0.227 / 0.834 / 0.781 | 16.7m |
| 5 | 0.224 / 0.841 / 0.815 | 0.223 / 0.841 / 0.812 | 0.225 / 0.839 / 0.809 | 8.3m |
| 10 | 0.231 / 0.840 / 0.816 | 0.229 / 0.841 / 0.816 | 0.229 / 0.840 / 0.814 | 7.9m |
| 25 | 0.249 / 0.836 / 0.818 | 0.248 / 0.837 / 0.820 | 0.247 / 0.837 / 0.819 | 6.8m |
| 50 | 0.267 / 0.832 / 0.818 | 0.267 / 0.833 / 0.820 | 0.266 / 0.833 / 0.820 | 6.7m |
| 100 | 0.285 / 0.828 / 0.817 | 0.287 / 0.829 / 0.819 | 0.287 / 0.828 / 0.818 | 6.5m |
| 200 | 0.307 / 0.822 / 0.814 | 0.308 / 0.823 / 0.816 | 0.309 / 0.823 / 0.815 | 6.2m |
| 400 | 0.326 / 0.816 / 0.810 | 0.327 / 0.817 / 0.811 | 0.328 / 0.816 / 0.811 | 6.3m |
| 1000 | 0.345 / 0.811 / 0.806 | 0.345 / 0.812 / 0.807 | 0.346 / 0.812 / 0.808 | 6.3m |
| 2000 | 0.356 / 0.807 / 0.804 | 0.355 / 0.808 / 0.805 | 0.354 / 0.809 / 0.805 | 6.3m |
| **true ARG (ceiling)** | 0.165 / 0.859 / 0.835 | 0.165 / 0.859 / 0.835 | 0.165 / 0.859 / 0.835 | 37.7s |

## Assumed mutation rate (window=8 SNPs, r=r_true)

The inference's assumed `mu` is swept as a multiple of the true simulation mu; the true mu (and the simulated data) are unchanged. A misspecified mu rescales the inferred TMRCAs through both the emission rate and the time-grid calibration.

| assumed mu | 0.01*mu_true | 0.1*mu_true | mu_true | 10*mu_true | 100*mu_true | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Brier / MAP / P(true) | 0.226 / 0.838 / 0.806 | 0.225 / 0.839 / 0.809 | 0.223 / 0.841 / 0.812 | 0.224 / 0.841 / 0.815 | 0.238 / 0.837 / 0.816 | 16.5m |
| **true ARG (ceiling)** | 0.165 / 0.859 / 0.835 | 0.165 / 0.859 / 0.835 | 0.165 / 0.859 / 0.835 | 0.165 / 0.859 / 0.835 | 0.165 / 0.859 / 0.835 | 37.7s |

## Phasing switch-error rate (window=8 SNPs, r=r_true, mu=mu_true)

A switch-error process is injected into the ingroup genotypes fed to inference: consecutive haplotypes are paired into pseudo-diploids and a per-pair phase orientation flips with the given probability at each heterozygous site. The msprime ancestral truth (and ingroup allele frequencies) are unchanged. `0%` = perfectly phased; `1%` is the rate used in the main-text robustness section.

| switch rate | 0% | 5% | 10% | 20% | 50% | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| Brier / MAP / P(true) | 0.223 / 0.841 / 0.812 | 0.232 / 0.835 / 0.805 | 0.241 / 0.831 / 0.801 | 0.258 / 0.824 / 0.793 | 0.286 / 0.812 / 0.784 | 16.4m |
| **true ARG (ceiling)** | 0.165 / 0.859 / 0.835 | 0.165 / 0.859 / 0.835 | 0.165 / 0.859 / 0.835 | 0.165 / 0.859 / 0.835 | 0.165 / 0.859 / 0.835 | 37.7s |
