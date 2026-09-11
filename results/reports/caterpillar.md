# Monoallelic-ingroup caterpillar polarisation benchmark (B10)

Compares the two ways :class:`~ancestree.inference.FixedTreeInference`
can polarise a **monoallelic-ingroup** site: the bare
:class:`~ancestree.trees.OutgroupLadderTree` rooted at the ingroup
MRCA (which roots on the *derived* side of the only informative
substitution and so reports the ingroup's own fixed allele) versus the
:class:`~ancestree.trees.SpeciesLadderTree` caterpillar rooted at the
deepest outgroup ancestor (full Felsenstein over all outgroups recovers
the ancestral state). The caterpillar is what ``FixedTreeInference``
automatically uses today for monoallelic-ingroup sites; both modes are
run here explicitly via the kernel-level API.

Both modes use ``JC69 + BaseComposition.uniform() + uniform root prior``;
branch lengths come from the simulator's known split times scaled by
``mu`` (no MLE fit — the comparison isolates the *tree-choice* axis).

## Overall accuracy on monoallelic-ingroup sites

| tree_type | n_sites | accuracy | mean max_prob | mean p_true |
|:----------|--------:|---------:|--------------:|------------:|
| ladder      | 52536 | 0.6339 | 0.9246 | 0.7044 |
| caterpillar | 52536 | 0.8521 | 0.8502 | 0.7861 |

## Accuracy stratified by truth ancestral state

| tree_type | truth | n_sites | accuracy | mean max_prob |
|:----------|:-----:|--------:|---------:|--------------:|
| ladder | A | 13159 | 0.6321 | 0.9251 |
| ladder | C | 13224 | 0.6235 | 0.9230 |
| ladder | G | 13218 | 0.6412 | 0.9254 |
| ladder | T | 12935 | 0.6390 | 0.9251 |
| caterpillar | A | 13159 | 0.8492 | 0.8493 |
| caterpillar | C | 13224 | 0.8461 | 0.8498 |
| caterpillar | G | 13218 | 0.8580 | 0.8514 |
| caterpillar | T | 12935 | 0.8552 | 0.8503 |

## Accuracy stratified by ingroup-fixed allele state

``ingroup_fixed_ancestral`` rows: the single allele segregating in
the ingroup happens to be the simulator's ancestral allele. The
ladder trivially gets these right (its MAP = ingroup allele = truth);
the caterpillar should agree.

``ingroup_fixed_derived`` rows: the ingroup is fixed for the
*derived* allele (a fixed substitution swept on the ingroup lineage).
The ladder reports the ingroup's allele as MAP — *wrong*. The
caterpillar, which roots at the deep outgroup ancestor, can recover
the ancestral state via the outgroup tips.

| tree_type | ingroup_fixed | n_sites | accuracy | mean max_prob |
|:----------|:--------------|--------:|---------:|--------------:|
| ladder | ancestral | 34797 | 0.8714 | 0.9553 |
| ladder | derived | 17739 | 0.1681 | 0.8644 |
| caterpillar | ancestral | 34797 | 0.9934 | 0.8417 |
| caterpillar | derived | 17739 | 0.5749 | 0.8669 |

## Cross-cell MAP agreement

- shared sites: 52536
- MAP agreement rate: 0.7702

Agreement is high on ``ingroup_fixed_ancestral`` sites (both trees
land on the ingroup-allele MAP, which happens to be ancestral) and
low on ``ingroup_fixed_derived`` sites (the ladder picks the
ingroup's derived allele; the caterpillar picks the outgroup-implied
ancestor). The cross-cell mismatch is the exact size of the
``ingroup_fixed_derived`` failure rate of the ladder.

## Failure mode the caterpillar fixes

A monoallelic-ingroup site carries no within-ingroup polarisation
signal — every ingroup tip reads the same allele, so the SFS-prior
is uninformative and the Felsenstein root at the **ingroup MRCA**
sees a single observation. Under ``OutgroupLadderTree`` alone the
MAP at the ingroup MRCA is just the ingroup's own allele: the tree
is rooted on the *derived* side of the single substitution that
separates the ingroup from the outgroups, so its MAP is
self-confirming and wrong whenever the fixed allele is derived.

``SpeciesLadderTree`` reroots the inference question at the deepest
outgroup ancestor and carries the collapsed ingroup as a real tip on
a branch of length ``K_in``. Full Felsenstein over all outgroup
tips + the collapsed ingroup tip then recovers the deep-root state
from the outgroup-majority signal. The accuracy delta in the
``ingroup_fixed_derived`` row above is the exact size of the bug
this fixes.

