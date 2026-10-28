# Orientation damage to inferred genealogies

orientation damage on an ingroup-only 20-haplotype ARG panel: identical genotypes, identical seeds, one simulation, and one declared ancestral allele per orientation scheme. Every mis-orientation arm is paired per site against the true-ancestral-allele arm of the same ARG tool

Shared sites: 102,913 over 20,000,000 bp, 20 ingroup haplotypes, rf_max = 36.

Every excess is (mis-oriented arm) minus (true-ancestral-allele arm), so a positive value means the mis-orientation made the inferred genealogy worse.

## Mis-orientation rate per scheme

| scheme | mis-oriented sites | % of shared sites |
| --- | ---: | ---: |
| major_allele | 18,127 | 17.614 |
| random5 | 5,146 | 5.000 |
| freq_biased5 | 5,320 | 5.169 |
| fixed_tree_n1 | 393 | 0.382 |
| fixed_tree_n3 | 265 | 0.257 |

## tsinfer / major_allele, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 18,127 | 0.35466 | 0.39319 | +0.03853 | 0.00052 | +3.087e+05 | +0.23096 |
| (0, 100] | 14,483 | 0.35842 | 0.38779 | +0.02936 | 0.00057 | +2.89e+05 | +0.19081 |
| (100, 300] | 19,307 | 0.36729 | 0.38945 | +0.02216 | 0.00049 | +2.329e+05 | +0.16152 |
| (300, 700] | 21,142 | 0.38062 | 0.39231 | +0.01168 | 0.00046 | +1.326e+05 | +0.11034 |
| (700, 1500] | 17,989 | 0.39393 | 0.39609 | +0.00216 | 0.00047 | +4.225e+04 | +0.04881 |
| > 1500 | 11,865 | 0.40806 | 0.40130 | -0.00675 | 0.00052 | -4254 | +0.00468 |
| overall (all shared sites) | 102,913 | 0.37591 | 0.39299 | +0.01708 | 0.00021 | +1.729e+05 | +0.14104 |

## tsinfer / major_allele, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 49,972 | 0.38935 | 0.39016 | +0.00081 | 0.00028 | +4.928e+04 | +0.05598 |
| exactly 1 | 26,030 | 0.36981 | 0.39600 | +0.02620 | 0.00044 | +1.669e+05 | +0.17349 |
| exactly 2 | 13,158 | 0.36011 | 0.39507 | +0.03495 | 0.00058 | +2.79e+05 | +0.21238 |
| exactly 3 | 5,691 | 0.34999 | 0.39045 | +0.04046 | 0.00089 | +4.067e+05 | +0.22637 |
| 4 to 6 | 6,455 | 0.35382 | 0.39602 | +0.04220 | 0.00079 | +5.739e+05 | +0.23852 |
| more than 6 | 1,607 | 0.36690 | 0.41191 | +0.04501 | 0.00171 | +8.047e+05 | +0.24444 |
| overall (all shared sites) | 102,913 | 0.37591 | 0.39299 | +0.01708 | 0.00021 | +1.729e+05 | +0.14104 |

## tsinfer / random5, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 5,146 | 0.37713 | 0.39320 | +0.01608 | 0.00096 | +3.798e+05 | +0.04244 |
| (0, 100] | 5,379 | 0.36277 | 0.37260 | +0.00982 | 0.00082 | +2.429e+05 | +0.02044 |
| (100, 300] | 9,737 | 0.36700 | 0.37505 | +0.00805 | 0.00058 | +1.088e+05 | +0.01491 |
| (300, 700] | 16,181 | 0.36912 | 0.37558 | +0.00646 | 0.00043 | +4.507e+04 | +0.01276 |
| (700, 1500] | 23,003 | 0.37646 | 0.38177 | +0.00530 | 0.00033 | +2.51e+04 | +0.00672 |
| > 1500 | 43,467 | 0.38163 | 0.38501 | +0.00339 | 0.00020 | +5408 | +0.00034 |
| overall (all shared sites) | 102,913 | 0.37591 | 0.38162 | +0.00571 | 0.00016 | +5.697e+04 | +0.00831 |

## tsinfer / random5, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 76,235 | 0.37765 | 0.38197 | +0.00433 | 0.00017 | +1.797e+04 | +0.00322 |
| exactly 1 | 21,344 | 0.37458 | 0.38355 | +0.00897 | 0.00039 | +1.549e+05 | +0.02165 |
| exactly 2 | 4,318 | 0.36343 | 0.37520 | +0.01177 | 0.00094 | +2.179e+05 | +0.02480 |
| exactly 3 | 851 | 0.32648 | 0.34192 | +0.01544 | 0.00208 | +2.288e+05 | +0.02796 |
| 4 to 6 | 165 | 0.32710 | 0.34259 | +0.01549 | 0.00460 | +3.109e+05 | +0.05711 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.37591 | 0.38162 | +0.00571 | 0.00016 | +5.697e+04 | +0.00831 |

## tsinfer / freq_biased5, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 5,320 | 0.35700 | 0.37661 | +0.01961 | 0.00096 | +2.351e+05 | +0.05914 |
| (0, 100] | 5,752 | 0.35370 | 0.36540 | +0.01171 | 0.00083 | +1.86e+05 | +0.02821 |
| (100, 300] | 10,068 | 0.35587 | 0.36562 | +0.00975 | 0.00059 | +8.108e+04 | +0.02392 |
| (300, 700] | 16,273 | 0.36525 | 0.37176 | +0.00651 | 0.00045 | +4.477e+04 | +0.01318 |
| (700, 1500] | 22,503 | 0.37660 | 0.38105 | +0.00445 | 0.00035 | +1.285e+04 | +0.00577 |
| > 1500 | 42,997 | 0.38959 | 0.39112 | +0.00153 | 0.00023 | -1148 | -0.00080 |
| overall (all shared sites) | 102,913 | 0.37591 | 0.38117 | +0.00526 | 0.00017 | +3.989e+04 | +0.01085 |

## tsinfer / freq_biased5, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 77,154 | 0.38146 | 0.38467 | +0.00320 | 0.00018 | +1.191e+04 | +0.00326 |
| exactly 1 | 20,390 | 0.36337 | 0.37464 | +0.01127 | 0.00043 | +1.132e+05 | +0.03110 |
| exactly 2 | 4,294 | 0.34808 | 0.36032 | +0.01225 | 0.00099 | +1.257e+05 | +0.03082 |
| exactly 3 | 845 | 0.32051 | 0.33406 | +0.01354 | 0.00210 | +1.253e+05 | +0.01934 |
| 4 to 6 | 230 | 0.34843 | 0.35072 | +0.00229 | 0.00392 | +1.014e+06 | +0.03762 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.37591 | 0.38117 | +0.00526 | 0.00017 | +3.989e+04 | +0.01085 |

## tsinfer / fixed_tree_n1, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 393 | 0.36436 | 0.37051 | +0.00615 | 0.00276 | +1.509e+05 | +0.02687 |
| (0, 100] | 599 | 0.34136 | 0.34122 | -0.00014 | 0.00187 | +6.28e+04 | +0.01255 |
| (100, 300] | 1,037 | 0.35999 | 0.35431 | -0.00568 | 0.00129 | +4.947e+04 | +0.01051 |
| (300, 700] | 1,752 | 0.36330 | 0.36555 | +0.00225 | 0.00097 | +1.637e+04 | +0.00741 |
| (700, 1500] | 2,984 | 0.37093 | 0.37425 | +0.00331 | 0.00077 | +1.946e+04 | +0.00649 |
| > 1500 | 96,148 | 0.37673 | 0.37534 | -0.00139 | 0.00010 | +2531 | -0.00008 |
| overall (all shared sites) | 102,913 | 0.37591 | 0.37471 | -0.00120 | 0.00010 | +4648 | +0.00069 |

## tsinfer / fixed_tree_n1, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 100,204 | 0.37632 | 0.37501 | -0.00131 | 0.00010 | +2624 | +0.00029 |
| exactly 1 | 2,361 | 0.36226 | 0.36597 | +0.00371 | 0.00090 | +9.508e+04 | +0.01368 |
| exactly 2 | 247 | 0.38012 | 0.37416 | -0.00596 | 0.00358 | -5.057e+04 | -0.01414 |
| exactly 3 | 83 | 0.22390 | 0.23628 | +0.01238 | 0.00152 | +1.854e+04 | +0.02590 |
| 4 to 6 | 18 | 0.52160 | 0.52160 | +0.00000 | 0.00000 | +1.008e+05 | +0.00018 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.37591 | 0.37471 | -0.00120 | 0.00010 | +4648 | +0.00069 |

## tsinfer / fixed_tree_n3, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 265 | 0.36111 | 0.36751 | +0.00639 | 0.00301 | +1.657e+05 | +0.02609 |
| (0, 100] | 414 | 0.34561 | 0.34601 | +0.00040 | 0.00172 | +3.934e+04 | -0.00082 |
| (100, 300] | 733 | 0.35349 | 0.35293 | -0.00057 | 0.00122 | -3.741e+04 | +0.01008 |
| (300, 700] | 1,202 | 0.36377 | 0.36712 | +0.00335 | 0.00090 | -1.57e+04 | +0.00816 |
| (700, 1500] | 2,071 | 0.37171 | 0.37489 | +0.00318 | 0.00073 | +1.682e+04 | +0.00382 |
| > 1500 | 98,228 | 0.37648 | 0.37707 | +0.00059 | 0.00005 | -2544 | -0.00113 |
| overall (all shared sites) | 102,913 | 0.37591 | 0.37659 | +0.00068 | 0.00005 | -1954 | -0.00068 |

## tsinfer / fixed_tree_n3, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 101,095 | 0.37628 | 0.37690 | +0.00062 | 0.00005 | -2741 | -0.00096 |
| exactly 1 | 1,645 | 0.35196 | 0.35659 | +0.00463 | 0.00090 | +4.429e+04 | +0.01214 |
| exactly 2 | 145 | 0.38276 | 0.38257 | -0.00019 | 0.00179 | -9.582e+04 | +0.00082 |
| exactly 3 | 10 | 0.21111 | 0.21111 | +0.00000 | 0.00000 | +1.777e+06 | -0.00114 |
| 4 to 6 | 18 | 0.52160 | 0.52160 | +0.00000 | 0.00000 | -4.089e+04 | +0.00000 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.37591 | 0.37659 | +0.00068 | 0.00005 | -1954 | -0.00068 |

## relate / major_allele, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 18,127 | 0.34018 | 0.40329 | +0.06311 | 0.00067 | +4.542e+05 | +0.22897 |
| (0, 100] | 14,483 | 0.34437 | 0.39679 | +0.05241 | 0.00073 | +4.339e+05 | +0.20004 |
| (100, 300] | 19,307 | 0.35298 | 0.39917 | +0.04619 | 0.00063 | +2.966e+05 | +0.17765 |
| (300, 700] | 21,142 | 0.36842 | 0.40268 | +0.03426 | 0.00058 | +1.563e+05 | +0.13587 |
| (700, 1500] | 17,989 | 0.38013 | 0.39911 | +0.01898 | 0.00059 | +5.067e+04 | +0.07815 |
| > 1500 | 11,865 | 0.39799 | 0.40545 | +0.00746 | 0.00065 | +1758 | +0.01961 |
| overall (all shared sites) | 102,913 | 0.36262 | 0.40099 | +0.03837 | 0.00027 | +2.379e+05 | +0.15815 |

## relate / major_allele, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 29,744 | 0.37408 | 0.38855 | +0.01447 | 0.00043 | +4.721e+04 | +0.05423 |
| exactly 1 | 27,805 | 0.36489 | 0.40428 | +0.03939 | 0.00051 | +1.745e+05 | +0.15252 |
| exactly 2 | 17,711 | 0.35981 | 0.40851 | +0.04870 | 0.00068 | +2.865e+05 | +0.18962 |
| exactly 3 | 10,615 | 0.35754 | 0.41001 | +0.05246 | 0.00086 | +3.707e+05 | +0.20877 |
| 4 to 6 | 13,267 | 0.34469 | 0.40063 | +0.05594 | 0.00077 | +5.01e+05 | +0.19809 |
| more than 6 | 3,771 | 0.34605 | 0.41555 | +0.06951 | 0.00146 | +6.806e+05 | +0.20755 |
| overall (all shared sites) | 102,913 | 0.36262 | 0.40099 | +0.03837 | 0.00027 | +2.379e+05 | +0.15815 |

## relate / random5, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 5,146 | 0.36217 | 0.38522 | +0.02305 | 0.00117 | +1.058e+05 | +0.07390 |
| (0, 100] | 5,379 | 0.34717 | 0.36241 | +0.01523 | 0.00102 | +5.627e+04 | +0.03790 |
| (100, 300] | 9,737 | 0.35024 | 0.36240 | +0.01216 | 0.00074 | +3.973e+04 | +0.02621 |
| (300, 700] | 16,181 | 0.35543 | 0.36300 | +0.00757 | 0.00055 | +1.478e+04 | +0.01269 |
| (700, 1500] | 23,003 | 0.36224 | 0.36862 | +0.00637 | 0.00043 | +3455 | +0.00562 |
| > 1500 | 43,467 | 0.37023 | 0.37187 | +0.00163 | 0.00025 | +909 | -0.00042 |
| overall (all shared sites) | 102,913 | 0.36262 | 0.36903 | +0.00641 | 0.00020 | +1.547e+04 | +0.01235 |

## relate / random5, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 62,056 | 0.36593 | 0.36882 | +0.00289 | 0.00023 | +2284 | +0.00135 |
| exactly 1 | 29,393 | 0.36060 | 0.37070 | +0.01009 | 0.00041 | +3.216e+04 | +0.02532 |
| exactly 2 | 8,474 | 0.35446 | 0.36864 | +0.01418 | 0.00081 | +5.009e+04 | +0.03403 |
| exactly 3 | 2,396 | 0.33461 | 0.35107 | +0.01646 | 0.00167 | +2.488e+04 | +0.02766 |
| 4 to 6 | 566 | 0.34806 | 0.38378 | +0.03573 | 0.00331 | +4.296e+04 | +0.04829 |
| more than 6 | 28 | 0.30952 | 0.42857 | +0.11905 | 0.00740 | -1.044e+05 | +0.01037 |
| overall (all shared sites) | 102,913 | 0.36262 | 0.36903 | +0.00641 | 0.00020 | +1.547e+04 | +0.01235 |

## relate / freq_biased5, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 5,320 | 0.34293 | 0.36151 | +0.01858 | 0.00108 | +9.609e+04 | +0.06183 |
| (0, 100] | 5,752 | 0.34134 | 0.35288 | +0.01154 | 0.00091 | +5.894e+04 | +0.03005 |
| (100, 300] | 10,068 | 0.34396 | 0.35467 | +0.01071 | 0.00066 | +5.035e+04 | +0.02441 |
| (300, 700] | 16,273 | 0.35139 | 0.35957 | +0.00818 | 0.00050 | +2.163e+04 | +0.01503 |
| (700, 1500] | 22,503 | 0.36327 | 0.36840 | +0.00514 | 0.00039 | +5734 | +0.00688 |
| > 1500 | 42,997 | 0.37619 | 0.37840 | +0.00221 | 0.00023 | +283.6 | +0.00244 |
| overall (all shared sites) | 102,913 | 0.36262 | 0.36861 | +0.00599 | 0.00018 | +1.798e+04 | +0.01341 |

## relate / freq_biased5, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 62,825 | 0.36877 | 0.37214 | +0.00337 | 0.00020 | +3831 | +0.00376 |
| exactly 1 | 28,391 | 0.35920 | 0.36870 | +0.00950 | 0.00040 | +3.264e+04 | +0.02474 |
| exactly 2 | 8,793 | 0.34059 | 0.35069 | +0.01010 | 0.00072 | +6.2e+04 | +0.03023 |
| exactly 3 | 2,230 | 0.33104 | 0.34589 | +0.01485 | 0.00152 | +4.744e+04 | +0.02526 |
| 4 to 6 | 674 | 0.32484 | 0.34496 | +0.02011 | 0.00211 | +4.777e+04 | +0.01304 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.36262 | 0.36861 | +0.00599 | 0.00018 | +1.798e+04 | +0.01341 |

## relate / fixed_tree_n1, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 393 | 0.35963 | 0.36924 | +0.00961 | 0.00320 | +3.694e+04 | +0.02223 |
| (0, 100] | 599 | 0.34270 | 0.34929 | +0.00659 | 0.00185 | +1.26e+04 | +0.00863 |
| (100, 300] | 1,037 | 0.34362 | 0.34774 | +0.00413 | 0.00151 | +6747 | +0.00893 |
| (300, 700] | 1,752 | 0.34878 | 0.34805 | -0.00073 | 0.00117 | +8231 | +0.00749 |
| (700, 1500] | 2,984 | 0.36011 | 0.36314 | +0.00303 | 0.00072 | -1510 | -0.00038 |
| > 1500 | 96,148 | 0.36329 | 0.36341 | +0.00011 | 0.00006 | +376 | -0.00021 |
| overall (all shared sites) | 102,913 | 0.36262 | 0.36292 | +0.00030 | 0.00007 | +730 | +0.00032 |

## relate / fixed_tree_n1, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 98,635 | 0.36319 | 0.36335 | +0.00016 | 0.00007 | +277.3 | -0.00007 |
| exactly 1 | 3,773 | 0.34675 | 0.34975 | +0.00300 | 0.00075 | +1.671e+04 | +0.00734 |
| exactly 2 | 434 | 0.38786 | 0.39619 | +0.00832 | 0.00232 | +1.595e+04 | +0.00490 |
| exactly 3 | 51 | 0.20370 | 0.20370 | +0.00000 | 0.00000 | -4.451e+05 | +0.00102 |
| 4 to 6 | 20 | 0.42778 | 0.42778 | +0.00000 | 0.00000 | +2.564e+04 | +0.00265 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.36262 | 0.36292 | +0.00030 | 0.00007 | +730 | +0.00032 |

## relate / fixed_tree_n3, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 265 | 0.35220 | 0.36143 | +0.00922 | 0.00344 | +9.93e+04 | +0.02208 |
| (0, 100] | 414 | 0.34152 | 0.34662 | +0.00510 | 0.00188 | +7.968e+04 | +0.00879 |
| (100, 300] | 733 | 0.33917 | 0.34319 | +0.00402 | 0.00153 | +4.336e+04 | +0.00909 |
| (300, 700] | 1,202 | 0.35071 | 0.35025 | -0.00046 | 0.00114 | +9896 | +0.00245 |
| (700, 1500] | 2,071 | 0.36383 | 0.36810 | +0.00427 | 0.00077 | +7146 | +0.00142 |
| > 1500 | 98,228 | 0.36303 | 0.36293 | -0.00010 | 0.00005 | -235.2 | +0.00017 |
| overall (all shared sites) | 102,913 | 0.36262 | 0.36268 | +0.00006 | 0.00006 | +920 | +0.00045 |

## relate / fixed_tree_n3, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 99,955 | 0.36297 | 0.36293 | -0.00004 | 0.00006 | -200.4 | +0.00029 |
| exactly 1 | 2,661 | 0.34720 | 0.35085 | +0.00365 | 0.00076 | +4.474e+04 | +0.00358 |
| exactly 2 | 254 | 0.38583 | 0.38736 | +0.00153 | 0.00150 | -1.581e+04 | +0.00345 |
| exactly 3 | 23 | 0.31401 | 0.31401 | +0.00000 | 0.00000 | -3.323e+04 | -0.00438 |
| 4 to 6 | 20 | 0.42778 | 0.42778 | +0.00000 | 0.00000 | +2.15e+04 | +0.00037 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.36262 | 0.36268 | +0.00006 | 0.00006 | +920 | +0.00045 |

