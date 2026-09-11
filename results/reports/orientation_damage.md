# Orientation damage to inferred genealogies

orientation damage on an ingroup-only 20-haplotype ARG panel: identical genotypes, identical seeds, one simulation, and one declared ancestral allele per orientation scheme. Every mis-orientation arm is paired per site against the true-ancestral-allele arm of the same ARG tool

Shared sites: 102,913 over 20,000,000 bp, 20 ingroup haplotypes, rf_max = 36.

Every excess is (mis-oriented arm) minus (true-ancestral-allele arm), so a positive value means the mis-orientation made the inferred genealogy worse.

## Mis-orientation rate per scheme

| scheme | mis-oriented sites | % of shared sites |
| --- | ---: | ---: |
| major_allele | 18,495 | 17.971 |
| random5 | 5,146 | 5.000 |
| freq_biased5 | 5,336 | 5.185 |
| fixed_tree_n1 | 768 | 0.746 |
| fixed_tree_n3 | 592 | 0.575 |

## tsinfer / major_allele, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 18,495 | 0.35515 | 0.39336 | +0.03821 | 0.00054 | +2.889e+05 | +0.22695 |
| (0, 100] | 14,705 | 0.35773 | 0.38762 | +0.02989 | 0.00059 | +2.759e+05 | +0.18782 |
| (100, 300] | 19,527 | 0.36768 | 0.38979 | +0.02211 | 0.00052 | +2.26e+05 | +0.15901 |
| (300, 700] | 21,247 | 0.38132 | 0.39243 | +0.01111 | 0.00048 | +1.235e+05 | +0.10716 |
| (700, 1500] | 17,732 | 0.39327 | 0.39595 | +0.00268 | 0.00050 | +3.622e+04 | +0.04436 |
| > 1500 | 11,207 | 0.40690 | 0.40129 | -0.00561 | 0.00057 | -4749 | +0.00520 |
| overall (all shared sites) | 102,913 | 0.37550 | 0.39298 | +0.01748 | 0.00022 | +1.654e+05 | +0.13903 |

## tsinfer / major_allele, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 49,879 | 0.38957 | 0.39102 | +0.00145 | 0.00030 | +4.671e+04 | +0.05433 |
| exactly 1 | 26,457 | 0.37017 | 0.39592 | +0.02576 | 0.00045 | +1.591e+05 | +0.17108 |
| exactly 2 | 12,928 | 0.35843 | 0.39462 | +0.03619 | 0.00062 | +2.71e+05 | +0.20963 |
| exactly 3 | 5,624 | 0.34955 | 0.39100 | +0.04145 | 0.00093 | +3.801e+05 | +0.22721 |
| 4 to 6 | 6,475 | 0.35062 | 0.39218 | +0.04155 | 0.00083 | +5.736e+05 | +0.23990 |
| more than 6 | 1,550 | 0.35444 | 0.40258 | +0.04814 | 0.00164 | +7.299e+05 | +0.23893 |
| overall (all shared sites) | 102,913 | 0.37550 | 0.39298 | +0.01748 | 0.00022 | +1.654e+05 | +0.13903 |

## tsinfer / random5, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 5,146 | 0.37729 | 0.39375 | +0.01646 | 0.00098 | +3.871e+05 | +0.04702 |
| (0, 100] | 5,379 | 0.36170 | 0.37329 | +0.01158 | 0.00084 | +2.708e+05 | +0.02404 |
| (100, 300] | 9,737 | 0.36680 | 0.37594 | +0.00914 | 0.00060 | +1.116e+05 | +0.01854 |
| (300, 700] | 16,181 | 0.36898 | 0.37642 | +0.00743 | 0.00044 | +3.974e+04 | +0.01529 |
| (700, 1500] | 23,003 | 0.37623 | 0.38222 | +0.00599 | 0.00034 | +1.174e+04 | +0.00814 |
| > 1500 | 43,467 | 0.38099 | 0.38525 | +0.00426 | 0.00022 | +9683 | +0.00286 |
| overall (all shared sites) | 102,913 | 0.37550 | 0.38211 | +0.00660 | 0.00016 | +5.703e+04 | +0.01070 |

## tsinfer / random5, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 76,674 | 0.37729 | 0.38248 | +0.00519 | 0.00018 | +1.773e+04 | +0.00628 |
| exactly 1 | 21,209 | 0.37349 | 0.38415 | +0.01065 | 0.00041 | +1.577e+05 | +0.02240 |
| exactly 2 | 3,992 | 0.36366 | 0.37344 | +0.00978 | 0.00104 | +2.268e+05 | +0.02764 |
| exactly 3 | 860 | 0.32762 | 0.34667 | +0.01906 | 0.00211 | +2.63e+05 | +0.04049 |
| 4 to 6 | 178 | 0.34145 | 0.34410 | +0.00265 | 0.00476 | +1.846e+05 | +0.02295 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.37550 | 0.38211 | +0.00660 | 0.00016 | +5.703e+04 | +0.01070 |

## tsinfer / freq_biased5, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 5,336 | 0.35713 | 0.37656 | +0.01943 | 0.00096 | +2.134e+05 | +0.05699 |
| (0, 100] | 5,766 | 0.35273 | 0.36498 | +0.01225 | 0.00083 | +1.561e+05 | +0.02675 |
| (100, 300] | 10,133 | 0.35553 | 0.36571 | +0.01018 | 0.00059 | +6.637e+04 | +0.02269 |
| (300, 700] | 16,325 | 0.36481 | 0.37174 | +0.00692 | 0.00045 | +2.921e+04 | +0.01301 |
| (700, 1500] | 22,512 | 0.37669 | 0.38072 | +0.00403 | 0.00035 | -1079 | +0.00456 |
| > 1500 | 42,841 | 0.38903 | 0.39191 | +0.00288 | 0.00022 | -722.7 | +0.00135 |
| overall (all shared sites) | 102,913 | 0.37550 | 0.38138 | +0.00587 | 0.00016 | +3.045e+04 | +0.01110 |

## tsinfer / freq_biased5, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 77,449 | 0.38153 | 0.38521 | +0.00367 | 0.00018 | +4675 | +0.00365 |
| exactly 1 | 20,101 | 0.36196 | 0.37464 | +0.01268 | 0.00042 | +9.595e+04 | +0.03277 |
| exactly 2 | 4,371 | 0.34158 | 0.35469 | +0.01312 | 0.00096 | +1.35e+05 | +0.02263 |
| exactly 3 | 787 | 0.32430 | 0.33047 | +0.00618 | 0.00236 | +1.235e+05 | +0.03908 |
| 4 to 6 | 205 | 0.34512 | 0.35799 | +0.01287 | 0.00441 | +7.563e+05 | +0.01503 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.37550 | 0.38138 | +0.00587 | 0.00016 | +3.045e+04 | +0.01110 |

## tsinfer / fixed_tree_n1, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 768 | 0.37822 | 0.37514 | -0.00307 | 0.00204 | -1.431e+05 | -0.01556 |
| (0, 100] | 1,081 | 0.35718 | 0.35510 | -0.00208 | 0.00132 | -2.343e+04 | -0.00439 |
| (100, 300] | 1,795 | 0.35899 | 0.35699 | -0.00200 | 0.00092 | -4470 | +0.00328 |
| (300, 700] | 3,367 | 0.36876 | 0.36855 | -0.00021 | 0.00064 | -8097 | -0.00153 |
| (700, 1500] | 5,790 | 0.37506 | 0.37403 | -0.00103 | 0.00047 | -8537 | -0.00160 |
| > 1500 | 90,112 | 0.37631 | 0.37557 | -0.00074 | 0.00008 | -707.6 | -0.00121 |
| overall (all shared sites) | 102,913 | 0.37550 | 0.37471 | -0.00079 | 0.00008 | -2756 | -0.00128 |

## tsinfer / fixed_tree_n1, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 98,647 | 0.37585 | 0.37514 | -0.00071 | 0.00008 | -2382 | -0.00116 |
| exactly 1 | 3,782 | 0.36741 | 0.36400 | -0.00340 | 0.00071 | -1.969e+04 | -0.00580 |
| exactly 2 | 382 | 0.38954 | 0.39332 | +0.00378 | 0.00206 | +5.273e+04 | +0.00304 |
| exactly 3 | 72 | 0.23110 | 0.23457 | +0.00347 | 0.00109 | +1.38e+04 | +0.00357 |
| 4 to 6 | 30 | 0.41111 | 0.41111 | +0.00000 | 0.00000 | +1.528e+05 | +0.00941 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.37550 | 0.37471 | -0.00079 | 0.00008 | -2756 | -0.00128 |

## tsinfer / fixed_tree_n3, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 592 | 0.38016 | 0.37453 | -0.00563 | 0.00235 | -2.256e+05 | -0.02742 |
| (0, 100] | 826 | 0.36084 | 0.35671 | -0.00414 | 0.00164 | -6.964e+04 | -0.01453 |
| (100, 300] | 1,406 | 0.35663 | 0.35655 | -0.00008 | 0.00110 | -4.755e+04 | -0.00239 |
| (300, 700] | 2,676 | 0.36822 | 0.36908 | +0.00086 | 0.00076 | -3.774e+04 | -0.00789 |
| (700, 1500] | 4,658 | 0.37443 | 0.37417 | -0.00026 | 0.00056 | -9016 | -0.00611 |
| > 1500 | 92,755 | 0.37615 | 0.37743 | +0.00127 | 0.00009 | -6061 | -0.00199 |
| overall (all shared sites) | 102,913 | 0.37550 | 0.37659 | +0.00109 | 0.00009 | -9358 | -0.00264 |

## tsinfer / fixed_tree_n3, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 99,830 | 0.37584 | 0.37702 | +0.00118 | 0.00009 | -7683 | -0.00234 |
| exactly 1 | 2,870 | 0.36232 | 0.36004 | -0.00228 | 0.00083 | -6.845e+04 | -0.01253 |
| exactly 2 | 185 | 0.39369 | 0.39730 | +0.00360 | 0.00239 | -7.376e+04 | -0.00217 |
| exactly 3 | 10 | 0.21111 | 0.21111 | +0.00000 | 0.00000 | +1.542e+06 | -0.00120 |
| 4 to 6 | 18 | 0.52160 | 0.52160 | +0.00000 | 0.00000 | -8.227e+04 | +0.00923 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.37550 | 0.37659 | +0.00109 | 0.00009 | -9358 | -0.00264 |

## relate / major_allele, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 18,495 | 0.34109 | 0.40325 | +0.06216 | 0.00067 | +4.426e+05 | +0.22497 |
| (0, 100] | 14,705 | 0.34464 | 0.39669 | +0.05205 | 0.00073 | +4.254e+05 | +0.19671 |
| (100, 300] | 19,527 | 0.35366 | 0.39934 | +0.04568 | 0.00063 | +2.879e+05 | +0.17274 |
| (300, 700] | 21,247 | 0.36864 | 0.40204 | +0.03341 | 0.00059 | +1.516e+05 | +0.13189 |
| (700, 1500] | 17,732 | 0.38063 | 0.39894 | +0.01831 | 0.00061 | +4.892e+04 | +0.07585 |
| > 1500 | 11,207 | 0.39833 | 0.40698 | +0.00866 | 0.00067 | +4219 | +0.02617 |
| overall (all shared sites) | 102,913 | 0.36272 | 0.40099 | +0.03827 | 0.00027 | +2.351e+05 | +0.15585 |

## relate / major_allele, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 29,058 | 0.37356 | 0.38855 | +0.01499 | 0.00045 | +4.622e+04 | +0.05437 |
| exactly 1 | 28,423 | 0.36570 | 0.40335 | +0.03766 | 0.00052 | +1.683e+05 | +0.14484 |
| exactly 2 | 17,939 | 0.35959 | 0.40797 | +0.04838 | 0.00068 | +2.822e+05 | +0.19018 |
| exactly 3 | 10,660 | 0.36062 | 0.41300 | +0.05238 | 0.00087 | +3.625e+05 | +0.20170 |
| 4 to 6 | 13,120 | 0.34352 | 0.40082 | +0.05730 | 0.00077 | +5.028e+05 | +0.20194 |
| more than 6 | 3,713 | 0.34399 | 0.41256 | +0.06857 | 0.00151 | +6.859e+05 | +0.20114 |
| overall (all shared sites) | 102,913 | 0.36272 | 0.40099 | +0.03827 | 0.00027 | +2.351e+05 | +0.15585 |

## relate / random5, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 5,146 | 0.36260 | 0.38453 | +0.02193 | 0.00119 | +1.044e+05 | +0.07141 |
| (0, 100] | 5,379 | 0.34790 | 0.36217 | +0.01427 | 0.00103 | +5.68e+04 | +0.03722 |
| (100, 300] | 9,737 | 0.35058 | 0.36255 | +0.01198 | 0.00073 | +3.277e+04 | +0.02399 |
| (300, 700] | 16,181 | 0.35516 | 0.36284 | +0.00768 | 0.00055 | +1.194e+04 | +0.01270 |
| (700, 1500] | 23,003 | 0.36279 | 0.36894 | +0.00615 | 0.00043 | +1.007e+04 | +0.00813 |
| > 1500 | 43,467 | 0.37005 | 0.37216 | +0.00211 | 0.00026 | -1481 | -0.00017 |
| overall (all shared sites) | 102,913 | 0.36272 | 0.36916 | +0.00645 | 0.00020 | +1.479e+04 | +0.01252 |

## relate / random5, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 62,354 | 0.36592 | 0.36905 | +0.00312 | 0.00023 | +2673 | +0.00241 |
| exactly 1 | 29,321 | 0.36081 | 0.37088 | +0.01007 | 0.00041 | +2.62e+04 | +0.02388 |
| exactly 2 | 8,380 | 0.35465 | 0.36793 | +0.01328 | 0.00081 | +5.503e+04 | +0.03447 |
| exactly 3 | 2,249 | 0.33249 | 0.34929 | +0.01680 | 0.00173 | +5.159e+04 | +0.02876 |
| 4 to 6 | 581 | 0.35083 | 0.38698 | +0.03614 | 0.00331 | +2.049e+04 | +0.04157 |
| more than 6 | 28 | 0.30952 | 0.42857 | +0.11905 | 0.00740 | -6.21e+04 | +0.03969 |
| overall (all shared sites) | 102,913 | 0.36272 | 0.36916 | +0.00645 | 0.00020 | +1.479e+04 | +0.01252 |

## relate / freq_biased5, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 5,336 | 0.34305 | 0.36167 | +0.01863 | 0.00107 | +9.291e+04 | +0.06102 |
| (0, 100] | 5,766 | 0.34111 | 0.35282 | +0.01172 | 0.00090 | +5.683e+04 | +0.03061 |
| (100, 300] | 10,133 | 0.34350 | 0.35441 | +0.01090 | 0.00066 | +3.691e+04 | +0.02325 |
| (300, 700] | 16,325 | 0.35060 | 0.35885 | +0.00825 | 0.00050 | +2.311e+04 | +0.01599 |
| (700, 1500] | 22,512 | 0.36354 | 0.36876 | +0.00522 | 0.00039 | +6930 | +0.00626 |
| > 1500 | 42,841 | 0.37680 | 0.37847 | +0.00167 | 0.00023 | -1423 | +0.00037 |
| overall (all shared sites) | 102,913 | 0.36272 | 0.36856 | +0.00584 | 0.00018 | +1.622e+04 | +0.01237 |

## relate / freq_biased5, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 63,146 | 0.36872 | 0.37201 | +0.00329 | 0.00020 | +1232 | +0.00223 |
| exactly 1 | 28,251 | 0.35971 | 0.36900 | +0.00929 | 0.00040 | +3.374e+04 | +0.02476 |
| exactly 2 | 8,801 | 0.34062 | 0.34986 | +0.00924 | 0.00072 | +5.652e+04 | +0.02858 |
| exactly 3 | 2,062 | 0.32619 | 0.34271 | +0.01652 | 0.00152 | +5.2e+04 | +0.02722 |
| 4 to 6 | 653 | 0.32508 | 0.34873 | +0.02365 | 0.00215 | +5.209e+04 | +0.02079 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.36272 | 0.36856 | +0.00584 | 0.00018 | +1.622e+04 | +0.01237 |

## relate / fixed_tree_n1, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 768 | 0.37941 | 0.37297 | -0.00644 | 0.00251 | -7.29e+04 | -0.02237 |
| (0, 100] | 1,081 | 0.35348 | 0.35379 | +0.00031 | 0.00161 | -3.174e+04 | -0.00728 |
| (100, 300] | 1,795 | 0.34977 | 0.34915 | -0.00062 | 0.00125 | -4.184e+04 | -0.01215 |
| (300, 700] | 3,367 | 0.35637 | 0.35549 | -0.00087 | 0.00087 | -1.01e+04 | -0.00451 |
| (700, 1500] | 5,790 | 0.36843 | 0.36853 | +0.00010 | 0.00059 | +1767 | -0.00306 |
| > 1500 | 90,112 | 0.36281 | 0.36313 | +0.00032 | 0.00009 | +229.6 | -0.00001 |
| overall (all shared sites) | 102,913 | 0.36272 | 0.36292 | +0.00020 | 0.00010 | -1637 | -0.00097 |

## relate / fixed_tree_n1, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 96,332 | 0.36248 | 0.36289 | +0.00041 | 0.00009 | +129.4 | -0.00014 |
| exactly 1 | 5,937 | 0.36691 | 0.36256 | -0.00435 | 0.00068 | -2.883e+04 | -0.01274 |
| exactly 2 | 518 | 0.36401 | 0.37752 | +0.01351 | 0.00203 | -2.847e+04 | +0.00283 |
| exactly 3 | 106 | 0.32128 | 0.32652 | +0.00524 | 0.00369 | +5.194e+04 | +0.03213 |
| 4 to 6 | 20 | 0.42778 | 0.42778 | +0.00000 | 0.00000 | -2.789e+04 | +0.00265 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.36272 | 0.36292 | +0.00020 | 0.00010 | -1637 | -0.00097 |

## relate / fixed_tree_n3, binned by bp to the nearest mis-oriented site

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (at a mis-oriented site) | 592 | 0.38044 | 0.36984 | -0.01060 | 0.00280 | -7.42e+04 | -0.03260 |
| (0, 100] | 826 | 0.35398 | 0.35311 | -0.00087 | 0.00194 | -1.749e+04 | -0.01035 |
| (100, 300] | 1,406 | 0.34618 | 0.34574 | -0.00043 | 0.00145 | -2.777e+04 | -0.01360 |
| (300, 700] | 2,676 | 0.35694 | 0.35733 | +0.00039 | 0.00098 | -6188 | -0.00417 |
| (700, 1500] | 4,658 | 0.36851 | 0.36827 | -0.00024 | 0.00066 | +3123 | -0.00025 |
| > 1500 | 92,755 | 0.36281 | 0.36284 | +0.00004 | 0.00008 | -533.6 | -0.00003 |
| overall (all shared sites) | 102,913 | 0.36272 | 0.36268 | -0.00004 | 0.00009 | -1447 | -0.00084 |

## relate / fixed_tree_n3, binned by mis-oriented sites in the local tree interval

| bin | sites | mean normalised RF, true-AA arm | mean normalised RF, mis-oriented arm | excess normalised RF | SEM of excess | excess Kendall-Colijn (generations) | excess TMRCA rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 mis-oriented sites in the interval | 98,057 | 0.36245 | 0.36260 | +0.00015 | 0.00008 | -467.9 | +0.00001 |
| exactly 1 | 4,511 | 0.36890 | 0.36321 | -0.00569 | 0.00078 | -2.19e+04 | -0.01611 |
| exactly 2 | 302 | 0.35578 | 0.37712 | +0.02134 | 0.00259 | -1.142e+04 | -0.00396 |
| exactly 3 | 23 | 0.31401 | 0.31401 | +0.00000 | 0.00000 | -7913 | -0.00130 |
| 4 to 6 | 20 | 0.42778 | 0.42778 | +0.00000 | 0.00000 | -3.203e+04 | +0.00037 |
| more than 6 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| overall (all shared sites) | 102,913 | 0.36272 | 0.36268 | -0.00004 | 0.00009 | -1447 | -0.00084 |

