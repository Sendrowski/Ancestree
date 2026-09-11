# Local-tree window x block grid

One ingroup-only msprime ARG (n_ingroup=20, mu=1.25e-08, rec_rate=1e-08), true rates given to the inference, perfectly phased genotypes.

Cells scored: 28 of 28 (block <= window only).
True-ARG ceiling mean Brier: 0.1646.

Requested widths label the rows and columns. An explicit block is not capped at the window and the window is rounded up to a whole number of blocks, so the effective-width tables below record what was run.

## Mean Brier score (lower is better)

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 0.2108 |  |  |  |  |  |  |
| 2snp | 0.2110 | 0.2137 |  |  |  |  |  |
| 4snp | 0.2141 | 0.2173 | 0.2196 |  |  |  |  |
| 8snp | 0.2221 | 0.2232 | 0.2228 | 0.2280 |  |  |  |
| 16snp | 0.2345 | 0.2361 | 0.2338 | 0.2364 | 0.2452 |  |  |
| 32snp | 0.2537 | 0.2527 | 0.2518 | 0.2525 | 0.2552 | 0.2652 |  |
| 64snp | 0.2743 | 0.2742 | 0.2730 | 0.2720 | 0.2742 | 0.2795 | 0.2895 |

## Mean MAP accuracy (higher is better)

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 0.8431 |  |  |  |  |  |  |
| 2snp | 0.8425 | 0.8422 |  |  |  |  |  |
| 4snp | 0.8430 | 0.8425 | 0.8413 |  |  |  |  |
| 8snp | 0.8417 | 0.8419 | 0.8411 | 0.8406 |  |  |  |
| 16snp | 0.8407 | 0.8397 | 0.8399 | 0.8392 | 0.8374 |  |  |
| 32snp | 0.8368 | 0.8371 | 0.8365 | 0.8358 | 0.8351 | 0.8330 |  |
| 64snp | 0.8321 | 0.8321 | 0.8322 | 0.8323 | 0.8307 | 0.8293 | 0.8261 |

## Mean posterior probability of the true allele

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 0.7863 |  |  |  |  |  |  |
| 2snp | 0.7933 | 0.7961 |  |  |  |  |  |
| 4snp | 0.8055 | 0.8085 | 0.8064 |  |  |  |  |
| 8snp | 0.8126 | 0.8144 | 0.8124 | 0.8136 |  |  |  |
| 16snp | 0.8185 | 0.8188 | 0.8180 | 0.8176 | 0.8179 |  |  |
| 32snp | 0.8206 | 0.8210 | 0.8202 | 0.8194 | 0.8191 | 0.8184 |  |
| 64snp | 0.8203 | 0.8204 | 0.8199 | 0.8194 | 0.8181 | 0.8174 | 0.8162 |

## Effective window (SNP equivalents at the panel's mean density)

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 1.00 |  |  |  |  |  |  |
| 2snp | 2.00 | 2.00 |  |  |  |  |  |
| 4snp | 5.00 | 6.00 | 4.00 |  |  |  |  |
| 8snp | 8.99 | 9.99 | 8.00 | 8.00 |  |  |  |
| 16snp | 16.99 | 17.99 | 16.00 | 16.00 | 16.00 |  |  |
| 32snp | 32.98 | 33.98 | 32.00 | 32.00 | 32.00 | 32.00 |  |
| 64snp | 64.96 | 65.96 | 64.00 | 64.00 | 64.00 | 64.00 | 64.00 |

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
| 4snp | 1885 | 2262 | 1509 |  |  |  |  |
| 8snp | 3393 | 3770 | 3018 | 3018 |  |  |  |
| 16snp | 6409 | 6786 | 6036 | 6036 | 6036 |  |  |
| 32snp | 12441 | 12818 | 12072 | 12072 | 12072 | 12072 |  |
| 64snp | 24505 | 24882 | 24144 | 24144 | 24144 | 24144 | 24144 |

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
| 4snp | 5.00 | 3.00 | 1.00 |  |  |  |  |
| 8snp | 9.00 | 5.00 | 2.00 | 1.00 |  |  |  |
| 16snp | 17.00 | 9.00 | 4.00 | 2.00 | 1.00 |  |  |
| 32snp | 33.00 | 17.00 | 8.00 | 4.00 | 2.00 | 1.00 |  |
| 64snp | 65.00 | 33.00 | 16.00 | 8.00 | 4.00 | 2.00 | 1.00 |

## Wall-clock runtime (s)

| requested window | block 1snp | block 2snp | block 4snp | block 8snp | block 16snp | block 32snp | block 64snp |
|---|---|---|---|---|---|---|---|
| 1snp | 99.8 |  |  |  |  |  |  |
| 2snp | 87.5 | 62.7 |  |  |  |  |  |
| 4snp | 76.6 | 50.6 | 40.7 |  |  |  |  |
| 8snp | 73.3 | 50.4 | 32.9 | 24.9 |  |  |  |
| 16snp | 70.7 | 47.6 | 29.4 | 21.8 | 19.6 |  |  |
| 32snp | 70.8 | 42.3 | 27.7 | 20.2 | 17.5 | 16.4 |  |
| 64snp | 73.9 | 44.8 | 26.9 | 19.7 | 16.8 | 15.3 | 14.6 |

Lowest mean Brier: window=1snp, block=1snp (0.2108).

