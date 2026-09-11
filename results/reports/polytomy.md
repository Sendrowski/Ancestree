# B5 — Polytomy correctness / information-loss sweep

## Simulation (held fixed across thresholds)
- samples (haploid): `60` (msprime `samples=30`, ploidy 2)
- sequence length: `200000`
- recombination rate: `1e-08`
- population size: `10000`
- mutation rate: `1e-07`
- thresholds (generations): `[0, 100, 500, 1000, 5000, 10000, 50000]`
- seed: `42`
- sites with truth + parsimony info: 3442

## Collapse rule
Every internal node whose **parent branch** is shorter than the threshold is
removed; its children attach directly to its grandparent, **keeping their
original child-side branch lengths**. This is the realistic tsinfer-style
polytomy: the topology forgets the internal split, branch lengths don't
compensate.

## Sweep

| threshold | n | coll/site | poly/site | max arity | truth | agree-baseline | mean max_prob |
|---|---|---|---|---|---|---|---|
| 0 | 3442 | 0.00 | 0.00 | 2 | 0.9064 | 1.0000 | 0.9121 |
| 100 | 3442 | 7.31 | 4.29 | 6 | 0.9064 | 1.0000 | 0.9121 |
| 500 | 3442 | 24.58 | 10.55 | 12 | 0.9064 | 1.0000 | 0.9121 |
| 1000 | 3442 | 34.45 | 10.09 | 18 | 0.9064 | 1.0000 | 0.9122 |
| 5000 | 3442 | 51.34 | 4.57 | 50 | 0.8992 | 0.9927 | 0.9078 |
| 10000 | 3442 | 54.51 | 3.28 | 55 | 0.8675 | 0.9593 | 0.9085 |
| 50000 | 3442 | 57.78 | 1.12 | 60 | 0.8777 | 0.9195 | 0.9754 |
