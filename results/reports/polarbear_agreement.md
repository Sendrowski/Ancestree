# B2 — Ancestree vs PolarBEAR agreement

## Simulation
- samples (haploid): `40` (msprime `samples=20`, default ploidy 2)
- sequence length: `2e+06`
- mutation rate: `1e-08`
- recombination rate: `1e-08`
- population size: `10000`
- seed: `42`

## Counts
- PolarBEAR sites (polymorphic only): 3348
- Ancestree sites: 3348
- shared positions compared: 3348
- non-homoplastic (parsimony score ≤ 1): 3344
- homoplastic (parsimony score > 1): 4

## MAP-allele agreement
| Subset | Agreement rate |
|---|---|
| non-homoplastic | 1.0000 (100.00%) |
| homoplastic     | 1.0000 (100.00%) |

## Posterior numerical agreement
### `max_prob` scalar (max over states under each tool)
| Subset | n | mean abs diff | max abs diff |
|---|---|---|---|
| non-homoplastic | 3344 | 3.526e-06 | 1.417e-04 |
| homoplastic     | 4 | 1.411e-05 | 4.589e-05 |

### Per-allele posterior (worst-allele `|Δp|` per site, requires patched PolarBEAR output)
| Subset | n | mean abs diff | max abs diff |
|---|---|---|---|
| non-homoplastic | 3344 | 3.530e-06 | 1.417e-04 |
| homoplastic     | 4 | 1.411e-05 | 4.589e-05 |

## Truth recovery (vs msprime's `Site.ancestral_state`)
| Tool | accuracy |
|---|---|
| Ancestree | 0.8832 (88.32%) |
| PolarBEAR | 0.8832 (88.32%) |

## Mean Brier score against the simulated truth
| Tool | mean Brier |
|---|---|
| Ancestree | 1.361e-01 |
| PolarBEAR | 1.361e-01 |

## Inference wall-clock runtime (seconds, inference step only, excluding I/O)
| Tool | Runtime (s) |
|---|---|
| Ancestree | 0.786 |
| PolarBEAR | 1.018 |

## PolarBEAR uninformativeness flags (from `filter_tree.filter_tree`)
| Flag | Count | Fraction |
|---|---|---|
| non-informative (multiple alleles tie for min parsimony) | 991 | 0.296 |
| multibranch (mutation on polytomous node) | 0 | 0.000 |
| homoplasy (parsimony score > 1) | 4 | 0.001 |

### Ancestree's own confidence on the PolarBEAR-flagged non-informative subset
| Subset | mean `max_prob` (Ancestree) |
|---|---|
| PolarBEAR labels informative | 1.0000 |
| PolarBEAR labels non-informative | 0.6085 |
