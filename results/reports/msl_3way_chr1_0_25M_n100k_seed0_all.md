# B13 MSL real-data 3-way comparison

MSL chr1 SNPs in the population VCF: **1,140,085**

All pairwise comparisons below are restricted to positions called by Ancestree-VCF, so each method is evaluated on the same site set and the three pairwise numbers are directly comparable.

## Sites called per method

| Method | Full call set | Restricted to Ancestree-VCF positions | Full coverage (% of MSL SNPs) |
|---|---:|---:|---:|
| ancestree_vcf | 133,172 | 133,172 | 11.68% |
| polarbear | 687,338 | 75,744 | 60.29% |
| estsfs | 985,038 | 112,930 | 86.40% |

**Ancestree-VCF calls 9,184 sites that NEITHER PolarBEAR nor est-sfs call** (6.9% of Ancestree's set). These are ingroup-polymorphic MSL SNPs that PolarBEAR's parsimony filter and est-sfs's biallelic / data-format filters exclude from their headline call sets.

## Pairwise agreement (on the restricted overlap)

| A | B | Overlap | Hard agree | Hard rate | Soft (mean P(B's MAP) under A's posterior) |
|---|---|---:|---:|---:|---:|
| ancestree_vcf | polarbear | 75,744 | 72,238 | 0.9537 | 0.9433 |
| ancestree_vcf | estsfs | 112,930 | 112,235 | 0.9938 | 0.9870 |
| polarbear | estsfs | 64,686 | 61,631 | 0.9528 | — |

## 3-way overlap: 64,686 sites

All three methods agree on **61,461** / 64,686 = **95.01%**

## Per-region agreement

### exon (49,367 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 5,153 | 4,949 | 0.9604 |
| ancestree_vcf | estsfs | 7,764 | 7,727 | 0.9952 |
| polarbear | estsfs | 4,611 | 4,428 | 0.9603 |

### UTR_3 (14,389 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 1,311 | 1,241 | 0.9466 |
| ancestree_vcf | estsfs | 2,019 | 2,013 | 0.9970 |
| polarbear | estsfs | 1,215 | 1,153 | 0.9490 |

### UTR_5 (2,231 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 256 | 248 | 0.9688 |
| ancestree_vcf | estsfs | 361 | 356 | 0.9861 |
| polarbear | estsfs | 235 | 227 | 0.9660 |

### intron (641,329 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 43,582 | 41,667 | 0.9561 |
| ancestree_vcf | estsfs | 66,524 | 66,127 | 0.9940 |
| polarbear | estsfs | 38,181 | 36,488 | 0.9557 |

### intergenic (432,769 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 25,442 | 24,133 | 0.9485 |
| ancestree_vcf | estsfs | 36,262 | 36,012 | 0.9931 |
| polarbear | estsfs | 20,444 | 19,335 | 0.9458 |

### synonymous (3,002 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 381 | 371 | 0.9738 |
| ancestree_vcf | estsfs | 615 | 608 | 0.9886 |
| polarbear | estsfs | 365 | 355 | 0.9726 |

### non_synonymous (6,833 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 841 | 827 | 0.9834 |
| ancestree_vcf | estsfs | 1,243 | 1,243 | 1.0000 |
| polarbear | estsfs | 785 | 773 | 0.9847 |

