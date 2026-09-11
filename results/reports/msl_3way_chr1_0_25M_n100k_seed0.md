# B13 MSL real-data 3-way comparison

MSL chr1 SNPs in the population VCF: **1,140,085**

All pairwise comparisons below are restricted to positions called by Ancestree-VCF, so each method is evaluated on the same site set and the three pairwise numbers are directly comparable.

## Sites called per method

| Method | Full call set | Restricted to Ancestree-VCF positions | Full coverage (% of MSL SNPs) |
|---|---:|---:|---:|
| ancestree_vcf | 100,000 | 100,000 | 8.77% |
| polarbear | 687,338 | 6,786 | 60.29% |
| estsfs | 985,038 | 10,169 | 86.40% |

**Ancestree-VCF calls 88,835 sites that NEITHER PolarBEAR nor est-sfs call** (88.8% of Ancestree's set). These are sites where the ingroup is monomorphic but the outgroup carries a divergent allele — Ancestree's polymorphic-anywhere filter retains them; PolarBEAR and est-sfs effectively require ingroup polymorphism.

## Pairwise agreement (on the restricted overlap)

| A | B | Overlap | Hard agree | Hard rate | Soft (mean P(B's MAP) under A's posterior) |
|---|---|---:|---:|---:|---:|
| ancestree_vcf | polarbear | 6,786 | 6,474 | 0.9540 | 0.9443 |
| ancestree_vcf | estsfs | 10,169 | 10,121 | 0.9953 | 0.9881 |
| polarbear | estsfs | 5,790 | 5,526 | 0.9544 | — |

## 3-way overlap: 5,790 sites

All three methods agree on **5,511** / 5,790 = **95.18%**

## Per-region agreement

### exon (49,367 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 438 | 420 | 0.9589 |
| ancestree_vcf | estsfs | 660 | 658 | 0.9970 |
| polarbear | estsfs | 382 | 368 | 0.9634 |

### UTR_3 (14,389 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 119 | 112 | 0.9412 |
| ancestree_vcf | estsfs | 188 | 187 | 0.9947 |
| polarbear | estsfs | 113 | 107 | 0.9469 |

### UTR_5 (2,231 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 25 | 24 | 0.9600 |
| ancestree_vcf | estsfs | 32 | 32 | 1.0000 |
| polarbear | estsfs | 21 | 20 | 0.9524 |

### intron (641,329 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 3,941 | 3,772 | 0.9571 |
| ancestree_vcf | estsfs | 6,012 | 5,980 | 0.9947 |
| polarbear | estsfs | 3,440 | 3,299 | 0.9590 |

### intergenic (432,769 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 2,263 | 2,146 | 0.9483 |
| ancestree_vcf | estsfs | 3,277 | 3,264 | 0.9960 |
| polarbear | estsfs | 1,834 | 1,732 | 0.9444 |

### synonymous (3,002 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 41 | 40 | 0.9756 |
| ancestree_vcf | estsfs | 62 | 62 | 1.0000 |
| polarbear | estsfs | 38 | 37 | 0.9737 |

### non_synonymous (6,833 SNPs)

| A | B | Overlap | Agree | Rate |
|---|---|---:|---:|---:|
| ancestree_vcf | polarbear | 69 | 69 | 1.0000 |
| ancestree_vcf | estsfs | 98 | 98 | 1.0000 |
| polarbear | estsfs | 59 | 59 | 1.0000 |

