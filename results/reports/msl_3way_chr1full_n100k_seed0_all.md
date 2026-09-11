# B13 MSL real-data 3-way comparison

MSL chr1 SNPs in the population VCF: **1,140,085**

All pairwise comparisons below are restricted to positions called by Ancestree-VCF, so each method is evaluated on the same site set and the three pairwise numbers are directly comparable.

## Sites called per method

| Method | Full call set | Restricted to Ancestree-VCF positions | Full coverage (% of MSL SNPs) |
|---|---:|---:|---:|
| ancestree_vcf | 1,140,085 | 1,140,085 | 100.00% |
| polarbear | 687,338 | 687,338 | 60.29% |
| estsfs | 985,038 | 985,038 | 86.40% |

**Ancestree-VCF calls 63,111 sites that NEITHER PolarBEAR nor est-sfs call** (5.5% of Ancestree's set). These are ingroup-polymorphic MSL SNPs that PolarBEAR's parsimony filter and est-sfs's biallelic / data-format filters exclude from their headline call sets.

## Pairwise agreement (on the restricted overlap)

| A | B | Overlap | Hard agree | Hard rate | Soft (mean P(B's MAP) under A's posterior) |
|---|---|---:|---:|---:|---:|
| ancestree_vcf | polarbear | 687,338 | 660,467 | 0.9609 | 0.9514 |
| ancestree_vcf | estsfs | 985,038 | 980,713 | 0.9956 | 0.9905 |
| polarbear | estsfs | 595,402 | 571,551 | 0.9599 | — |

## 3-way overlap: 595,402 sites

All three methods agree on **570,663** / 595,402 = **95.84%**

## Per-region agreement

### exon (49,367 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 30,103 | 29,045 | 0.9649 |
| ancestree_vcf | estsfs | 44,523 | 44,330 | 0.9957 |
| polarbear | estsfs | 27,253 | 26,293 | 0.9648 |

### UTR_3 (14,389 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 8,659 | 8,345 | 0.9637 |
| ancestree_vcf | estsfs | 13,233 | 13,178 | 0.9958 |
| polarbear | estsfs | 8,006 | 7,725 | 0.9649 |

### UTR_5 (2,231 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 1,309 | 1,277 | 0.9756 |
| ancestree_vcf | estsfs | 2,048 | 2,037 | 0.9946 |
| polarbear | estsfs | 1,213 | 1,182 | 0.9744 |

### intron (641,329 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 388,777 | 373,854 | 0.9616 |
| ancestree_vcf | estsfs | 566,583 | 564,205 | 0.9958 |
| polarbear | estsfs | 344,022 | 330,515 | 0.9607 |

### intergenic (432,769 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 258,490 | 247,946 | 0.9592 |
| ancestree_vcf | estsfs | 358,651 | 356,963 | 0.9953 |
| polarbear | estsfs | 214,908 | 205,836 | 0.9578 |

### synonymous (3,002 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 1,787 | 1,709 | 0.9564 |
| ancestree_vcf | estsfs | 2,870 | 2,849 | 0.9927 |
| polarbear | estsfs | 1,714 | 1,640 | 0.9568 |

### non_synonymous (6,833 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 4,547 | 4,462 | 0.9813 |
| ancestree_vcf | estsfs | 6,430 | 6,420 | 0.9984 |
| polarbear | estsfs | 4,288 | 4,210 | 0.9818 |

