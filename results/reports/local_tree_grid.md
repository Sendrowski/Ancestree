# Local-tree window x block grid

One ingroup-only msprime ARG (n_ingroup=20, mu=1.25e-08, rec_rate=1e-08), true rates given to the inference, perfectly phased genotypes.

Cells scored: 28 of 28 (block <= window only).
True-ARG ceiling mean Brier: 0.1646.

Requested widths label the rows and columns. An explicit block is not capped at the window and the window is rounded up to a whole number of blocks, so the effective-width tables below record what was run.

## Mean Brier score (lower is better)

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 0.2108 |  |  |  |  |  |  |
| 2snp | 0.2109 | 0.2137 |  |  |  |  |  |
| 4snp | 0.2131 | 0.2153 | 0.2197 |  |  |  |  |
| 8snp | 0.2200 | 0.2207 | 0.2229 | 0.2277 |  |  |  |
| 16snp | 0.2333 | 0.2330 | 0.2337 | 0.2362 | 0.2451 |  |  |
| 32snp | 0.2519 | 0.2516 | 0.2516 | 0.2522 | 0.2553 | 0.2653 |  |
| 64snp | 0.2738 | 0.2735 | 0.2725 | 0.2726 | 0.2736 | 0.2793 | 0.2901 |

## Mean MAP accuracy (higher is better)

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 0.8431 |  |  |  |  |  |  |
| 2snp | 0.8428 | 0.8421 |  |  |  |  |  |
| 4snp | 0.8428 | 0.8422 | 0.8410 |  |  |  |  |
| 8snp | 0.8420 | 0.8417 | 0.8412 | 0.8407 |  |  |  |
| 16snp | 0.8403 | 0.8401 | 0.8400 | 0.8394 | 0.8373 |  |  |
| 32snp | 0.8375 | 0.8375 | 0.8368 | 0.8359 | 0.8352 | 0.8329 |  |
| 64snp | 0.8320 | 0.8322 | 0.8325 | 0.8318 | 0.8310 | 0.8296 | 0.8261 |

## Mean posterior probability of the true allele

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 0.7863 |  |  |  |  |  |  |
| 2snp | 0.7934 | 0.7961 |  |  |  |  |  |
| 4snp | 0.8024 | 0.8038 | 0.8063 |  |  |  |  |
| 8snp | 0.8113 | 0.8118 | 0.8123 | 0.8137 |  |  |  |
| 16snp | 0.8180 | 0.8181 | 0.8179 | 0.8177 | 0.8179 |  |  |
| 32snp | 0.8211 | 0.8209 | 0.8203 | 0.8195 | 0.8190 | 0.8182 |  |
| 64snp | 0.8203 | 0.8202 | 0.8201 | 0.8193 | 0.8183 | 0.8173 | 0.8160 |

## Effective window (SNP equivalents at the panel's mean density)

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 1.00 |  |  |  |  |  |  |
| 2snp | 2.00 | 2.00 |  |  |  |  |  |
| 4snp | 4.00 | 4.00 | 4.00 |  |  |  |  |
| 8snp | 7.99 | 7.99 | 8.00 | 8.00 |  |  |  |
| 16snp | 15.99 | 15.99 | 16.00 | 16.00 | 16.00 |  |  |
| 32snp | 31.98 | 31.98 | 32.00 | 32.00 | 32.00 | 32.00 |  |
| 64snp | 63.96 | 63.96 | 64.00 | 64.00 | 64.00 | 64.00 | 64.00 |

## Effective block (SNP equivalents at the panel's mean density)

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 1.00 |  |  |  |  |  |  |
| 2snp | 1.00 | 2.00 |  |  |  |  |  |
| 4snp | 1.00 | 2.00 | 4.00 |  |  |  |  |
| 8snp | 1.00 | 2.00 | 4.00 | 8.00 |  |  |  |
| 16snp | 1.00 | 2.00 | 4.00 | 8.00 | 16.00 |  |  |
| 32snp | 1.00 | 2.00 | 4.00 | 8.00 | 16.00 | 32.00 |  |
| 64snp | 1.00 | 2.00 | 4.00 | 8.00 | 16.00 | 32.00 | 64.00 |

## Effective window width (bp)

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 377 |  |  |  |  |  |  |
| 2snp | 754 | 754 |  |  |  |  |  |
| 4snp | 1508 | 1508 | 1509 |  |  |  |  |
| 8snp | 3016 | 3016 | 3018 | 3018 |  |  |  |
| 16snp | 6032 | 6032 | 6036 | 6036 | 6036 |  |  |
| 32snp | 12064 | 12064 | 12072 | 12072 | 12072 | 12072 |  |
| 64snp | 24128 | 24128 | 24144 | 24144 | 24144 | 24144 | 24144 |

## Effective block width (bp)

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 377 |  |  |  |  |  |  |
| 2snp | 377 | 754 |  |  |  |  |  |
| 4snp | 377 | 754 | 1509 |  |  |  |  |
| 8snp | 377 | 754 | 1509 | 3018 |  |  |  |
| 16snp | 377 | 754 | 1509 | 3018 | 6036 |  |  |
| 32snp | 377 | 754 | 1509 | 3018 | 6036 | 12072 |  |
| 64snp | 377 | 754 | 1509 | 3018 | 6036 | 12072 | 24144 |

## Blocks per window

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 1.00 |  |  |  |  |  |  |
| 2snp | 2.00 | 1.00 |  |  |  |  |  |
| 4snp | 4.00 | 2.00 | 1.00 |  |  |  |  |
| 8snp | 8.00 | 4.00 | 2.00 | 1.00 |  |  |  |
| 16snp | 16.00 | 8.00 | 4.00 | 2.00 | 1.00 |  |  |
| 32snp | 32.00 | 16.00 | 8.00 | 4.00 | 2.00 | 1.00 |  |
| 64snp | 64.00 | 32.00 | 16.00 | 8.00 | 4.00 | 2.00 | 1.00 |

## Wall-clock runtime (s)

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 101.2 |  |  |  |  |  |  |
| 2snp | 91.9 | 68.9 |  |  |  |  |  |
| 4snp | 77.9 | 55.5 | 56.4 |  |  |  |  |
| 8snp | 80.8 | 70.5 | 38.3 | 25.8 |  |  |  |
| 16snp | 70.0 | 45.5 | 29.3 | 24.9 | 19.9 |  |  |
| 32snp | 73.0 | 41.7 | 29.9 | 20.8 | 17.1 | 16.3 |  |
| 64snp | 72.8 | 42.8 | 26.7 | 21.8 | 19.2 | 18.4 | 13.6 |

Lowest mean Brier: window=1snp, block=1snp (0.2108).

