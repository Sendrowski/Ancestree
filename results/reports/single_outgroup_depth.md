# Single-outgroup depth sweep

Ancestral-allele scoring for an ingroup plus exactly one outgroup, as a
function of that outgroup's split time. Depth is reported both in
generations and in coalescent units ``tau = T / (2 Ne)``, where ``T`` is
the split time in generations and ``Ne`` the diploid effective
population size of the ingroup. Two inference modes per depth:
ARG-based (full tree sequence) and fixed-tree (VCF,
``FixedTreeInference`` with a Kingman ingroup weight). Scoring is over
ingroup-polymorphic sites only. The mean Brier score is over the four
nucleotide states, so lower is better; MAP accuracy is a fraction of
the scored sites, so higher is better.

**Simulator settings (shared)**

- ``n_ingroup`` = 20
- ``length`` = 1000000.0
- ``mu`` = 1.25e-08
- ``rec_rate`` = 1e-08
- ``pop_size`` = 30000.0
- ``ingroup_pop_size`` = 30000.0
- ``seed`` = 42
- ``n_outgroup_pops`` = 1
- ``depths`` = ['1e4', '1.6e4', '2.5e4', '4e4', '6.3e4', '1e5', '1.6e5', '2.5e5', '4e5', '6.3e5', '1e6', '1.6e6', '2.5e6', '4e6', '6.3e6', '1e7', '1.6e7', '2.5e7']

## ARG mode — score by outgroup depth

| depth (generations) | tau = T / (2 Ne) | n_poly | mean Brier (lower better) | MAP accuracy (higher better) | mean max_prob |
|--------------------:|-----------------:|-------:|------------------------:|-----------------------------:|--------------:|
| 1e4 | 0.1667 | 5187 | 0.1566 | 0.8625 | 0.8657 |
| 1.6e4 | 0.2667 | 5190 | 0.1523 | 0.8676 | 0.8722 |
| 2.5e4 | 0.4167 | 5083 | 0.1531 | 0.8648 | 0.8712 |
| 4e4 | 0.6667 | 5014 | 0.1285 | 0.8875 | 0.8942 |
| 6.3e4 | 1.05 | 5400 | 0.1101 | 0.9109 | 0.9112 |
| 1e5 | 1.667 | 5183 | 0.0584 | 0.9564 | 0.9569 |
| 1.6e5 | 2.667 | 5168 | 0.0252 | 0.9818 | 0.9819 |
| 2.5e5 | 4.167 | 5152 | 0.0078 | 0.9953 | 0.9947 |
| 4e5 | 6.667 | 5202 | 0.0059 | 0.9963 | 0.9972 |
| 6.3e5 | 10.5 | 5207 | 0.0054 | 0.9971 | 0.9968 |
| 1e6 | 16.67 | 5208 | 0.0076 | 0.9956 | 0.9948 |
| 1.6e6 | 26.67 | 5208 | 0.0158 | 0.9904 | 0.9894 |
| 2.5e6 | 41.67 | 5208 | 0.0206 | 0.9868 | 0.9874 |
| 4e6 | 66.67 | 5208 | 0.0330 | 0.9785 | 0.9791 |
| 6.3e6 | 105 | 5208 | 0.0514 | 0.9664 | 0.9687 |
| 1e7 | 166.7 | 5206 | 0.0640 | 0.9587 | 0.9534 |
| 1.6e7 | 266.7 | 5208 | 0.1052 | 0.9257 | 0.9316 |
| 2.5e7 | 416.7 | 5208 | 0.1241 | 0.9124 | 0.9092 |

## VCF mode — score by outgroup depth

| depth (generations) | tau = T / (2 Ne) | n_poly | mean Brier (lower better) | MAP accuracy (higher better) | mean max_prob |
|--------------------:|-----------------:|-------:|------------------------:|-----------------------------:|--------------:|
| 1e4 | 0.1667 | 5187 | 0.4250 | 0.7874 | 0.9999 |
| 1.6e4 | 0.2667 | 5190 | 0.3723 | 0.8137 | 0.9998 |
| 2.5e4 | 0.4167 | 5083 | 0.3240 | 0.8377 | 0.9996 |
| 4e4 | 0.6667 | 5014 | 0.2623 | 0.8686 | 0.9995 |
| 6.3e4 | 1.05 | 5400 | 0.1970 | 0.9009 | 0.9990 |
| 1e5 | 1.667 | 5183 | 0.0908 | 0.9543 | 0.9989 |
| 1.6e5 | 2.667 | 5168 | 0.0362 | 0.9816 | 0.9980 |
| 2.5e5 | 4.167 | 5152 | 0.0106 | 0.9942 | 0.9972 |
| 4e5 | 6.667 | 5202 | 0.0094 | 0.9948 | 0.9955 |
| 6.3e5 | 10.5 | 5207 | 0.0109 | 0.9939 | 0.9925 |
| 1e6 | 16.67 | 5208 | 0.0170 | 0.9902 | 0.9887 |
| 1.6e6 | 26.67 | 5208 | 0.0349 | 0.9791 | 0.9805 |
| 2.5e6 | 41.67 | 5208 | 0.0476 | 0.9704 | 0.9711 |
| 4e6 | 66.67 | 5208 | 0.0625 | 0.9616 | 0.9567 |
| 6.3e6 | 105 | 5208 | 0.1002 | 0.9353 | 0.9338 |
| 1e7 | 166.7 | 5206 | 0.1308 | 0.9130 | 0.9089 |
| 1.6e7 | 266.7 | 5208 | 0.1893 | 0.8719 | 0.8769 |
| 2.5e7 | 416.7 | 5208 | 0.2263 | 0.8445 | 0.8512 |

## Take-aways

**ARG**: lowest mean Brier at depth 6.3e5 generations (tau = 10.5), 0.0054 over 5207 ingroup-polymorphic sites.

**VCF**: lowest mean Brier at depth 4e5 generations (tau = 6.667), 0.0094 over 5202 ingroup-polymorphic sites.

