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
- ``depths`` = ['1e4', '1.6e4', '2.5e4', '4e4', '6.3e4', '1e5', '1.6e5', '2.5e5', '4e5', '6.3e5', '1e6', '1.6e6', '2.5e6', '4e6', '6.3e6', '1e7']

## ARG mode — score by outgroup depth

| depth (generations) | tau = T / (2 Ne) | n_poly | mean Brier (lower better) | MAP accuracy (higher better) | mean max_prob |
|--------------------:|-----------------:|-------:|------------------------:|-----------------------------:|--------------:|
| 1e4 | 0.1667 | 5187 | 0.1566 | 0.8625 | 0.8657 |
| 1.6e4 | 0.2667 | 5190 | 0.1523 | 0.8676 | 0.8722 |
| 2.5e4 | 0.4167 | 5083 | 0.1531 | 0.8648 | 0.8711 |
| 4e4 | 0.6667 | 5014 | 0.1287 | 0.8873 | 0.8940 |
| 6.3e4 | 1.05 | 5400 | 0.1105 | 0.9106 | 0.9109 |
| 1e5 | 1.667 | 5183 | 0.0583 | 0.9564 | 0.9567 |
| 1.6e5 | 2.667 | 5168 | 0.0251 | 0.9818 | 0.9812 |
| 2.5e5 | 4.167 | 5152 | 0.0086 | 0.9936 | 0.9936 |
| 4e5 | 6.667 | 5202 | 0.0064 | 0.9954 | 0.9956 |
| 6.3e5 | 10.5 | 5207 | 0.0084 | 0.9944 | 0.9931 |
| 1e6 | 16.67 | 5208 | 0.0140 | 0.9871 | 0.9882 |
| 1.6e6 | 26.67 | 5208 | 0.0278 | 0.9754 | 0.9771 |
| 2.5e6 | 41.67 | 5208 | 0.0375 | 0.9702 | 0.9688 |
| 4e6 | 66.67 | 5208 | 0.0528 | 0.9562 | 0.9551 |
| 6.3e6 | 105 | 5208 | 0.0854 | 0.9303 | 0.9272 |
| 1e7 | 166.7 | 5206 | 0.1322 | 0.8886 | 0.8862 |

## VCF mode — score by outgroup depth

| depth (generations) | tau = T / (2 Ne) | n_poly | mean Brier (lower better) | MAP accuracy (higher better) | mean max_prob |
|--------------------:|-----------------:|-------:|------------------------:|-----------------------------:|--------------:|
| 1e4 | 0.1667 | 5187 | 0.4250 | 0.7874 | 0.9999 |
| 1.6e4 | 0.2667 | 5190 | 0.3723 | 0.8137 | 0.9998 |
| 2.5e4 | 0.4167 | 5083 | 0.3240 | 0.8377 | 0.9996 |
| 4e4 | 0.6667 | 5014 | 0.2619 | 0.8688 | 0.9995 |
| 6.3e4 | 1.05 | 5400 | 0.1977 | 0.9007 | 0.9990 |
| 1e5 | 1.667 | 5183 | 0.0907 | 0.9543 | 0.9989 |
| 1.6e5 | 2.667 | 5168 | 0.0361 | 0.9816 | 0.9980 |
| 2.5e5 | 4.167 | 5152 | 0.0114 | 0.9938 | 0.9972 |
| 4e5 | 6.667 | 5202 | 0.0080 | 0.9956 | 0.9955 |
| 6.3e5 | 10.5 | 5207 | 0.0133 | 0.9927 | 0.9925 |
| 1e6 | 16.67 | 5208 | 0.0238 | 0.9868 | 0.9887 |
| 1.6e6 | 26.67 | 5208 | 0.0472 | 0.9727 | 0.9805 |
| 2.5e6 | 41.67 | 5208 | 0.0588 | 0.9670 | 0.9711 |
| 4e6 | 66.67 | 5208 | 0.0912 | 0.9489 | 0.9567 |
| 6.3e6 | 105 | 5208 | 0.1462 | 0.9073 | 0.9338 |
| 1e7 | 166.7 | 5206 | 0.2324 | 0.8586 | 0.9089 |

## Take-aways

**ARG**: lowest mean Brier at depth 4e5 generations (tau = 6.667), 0.0064 over 5202 ingroup-polymorphic sites.

**VCF**: lowest mean Brier at depth 4e5 generations (tau = 6.667), 0.0080 over 5202 ingroup-polymorphic sites.

