# Outgroup-effect benchmark (ARGBasedInference)

Posterior certainty and accuracy as a function of the number of
outgroups available to the Felsenstein kernel. Same simulated ARG;
each row simplifies the tree sequence to ingroup + first ``n_out``
outgroups (closest-first) before inference. ``JC69 + uniform prior +
full`` throughout.

**Simulator settings**

- ``n_ingroup`` = 20
- ``n_outgroup_pops`` = 3
- ``length`` = 5000000.0
- ``mu`` = 1.25e-08
- ``rec_rate`` = 1e-08
- ``pop_size`` = 30000.0
- ``ingroup_pop_size`` = 30000.0
- ``outgroup_split_times`` = [300000.0, 900000.0, 1500000.0]
- ``seed`` = 42
- ``n_outs`` = [0, 1, 2, 3]

## Summary

| n_out | n_sites | mean max_prob | median max_prob | mean entropy (bits) | accuracy | mean Brier | baseline agreement |
|------:|--------:|--------------:|----------------:|--------------------:|---------:|-----------:|-------------------:|
| 0 | 288304 | 0.9866 | 1.0000 | 0.0322 | 0.6767 | 0.6353 | n/a |
| 1 | 288304 | 0.9477 | 1.0000 | 0.1257 | 0.6897 | 0.5773 | 0.8717 |
| 2 | 288304 | 0.8696 | 1.0000 | 0.3374 | 0.7399 | 0.4178 | 0.9223 |
| 3 | 288304 | 0.8666 | 0.9894 | 0.4305 | 0.8655 | 0.1936 | 0.9920 |

## Reading the trend

- **Adding outgroups *lowers* nominal confidence (mean max_prob)**
  because the kernel sees more conflicting tip patterns: with zero
  outgroups the local tree's root is the ingroup MRCA and the
  ingroup-only topology often resolves a single MAP allele with
  near-1.0 mass; outgroup tips inject deep-time variation that
  the kernel correctly reflects as posterior uncertainty.
- **Adding outgroups *raises* accuracy** because the inference
  question — what allele was ancestral at the simulator's deep
  ancestor — requires information beyond the ingroup MRCA. Without
  outgroups the MAP allele is the *ingroup-MRCA* state, which is
  only weakly correlated with the deep ancestor under finite Ne.
- The combination — high confidence and low accuracy at n_out=0
  shrinking toward calibrated confidence at larger n_out — is the
  expected signature of identifiability improving as the kernel
  gains a longer evolutionary baseline.

## Baseline-MAP agreement (vs `MajorityOutgroupInference`)

``baseline agreement`` is the fraction of sites at which Ancestree's
MAP allele matches the simple outgroup-majority rule (the
:class:`~ancestree.inference.MajorityOutgroupInference` baseline,
deliberately ignorant of substitution model, branch lengths, and
ingroup SFS class). High agreement = the polariser signal
dominates and the substitution-model + prior layers are
minimally biasing the call. Low agreement = the model is moving
posteriors away from the bare outgroup vote, worth investigating
whether the bias is correctly identifying recurrent / homoplastic
sites or whether it reflects model misspecification.

The ``n_out=0`` cell omits this column — the baseline rule needs
at least one outgroup observation per site to vote.

## Re-simulation variability (20 seeds, same demographic)

Each row aggregates over independent ARG draws from the *coalescent
prior* (different seed, same demographic). Per-site averaging
isn't possible because re-simulated mutations land at different
positions, so we report per-seed aggregate stats (mean ± std).
This is an **upper bound** on what proper ARG-posterior sampling
(ARGweaver / SINGER / Relate-MCMC) could shift these numbers,
since the coalescent prior is *not* conditioned on observed
mutations.

| n_out | mean max_prob (mean ± std) | accuracy (mean ± std) | mean #sites |
|------:|---------------------------:|----------------------:|------------:|
| 0 | 0.9871 ± 0.0003 | 0.6770 ± 0.0011 | 288216 |
| 1 | 0.9467 ± 0.0006 | 0.6895 ± 0.0012 | 288216 |
| 2 | 0.8702 ± 0.0006 | 0.7412 ± 0.0007 | 288216 |
| 3 | 0.8666 ± 0.0005 | 0.8655 ± 0.0007 | 288216 |

## Per-segregating-alleles breakdown

Stratified by the site's number of segregating alleles (after the source's SNP filter; almost all sites are biallelic).

| n_out | n_segregating | n_sites | accuracy | mean Brier | mean max_prob |
|------:|--------------:|--------:|---------:|-----------:|--------------:|
| 0 | 1 | 172322 | 1.0000 | 0.0000 | 1.0000 |
| 0 | 2 | 115073 | 0.1974 | 1.5773 | 0.9669 |
| 0 | 3 | 908 | 0.0507 | 1.8159 | 0.9232 |
| 0 | 4 | 1 | 0.0000 | 1.5725 | 0.6910 |
| 1 | 1 | 150597 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 2 | 136419 | 0.3524 | 1.2054 | 0.8913 |
| 1 | 3 | 1285 | 0.1307 | 1.5490 | 0.8129 |
| 1 | 4 | 3 | 0.3333 | 1.1732 | 0.5340 |
| 2 | 1 | 92727 | 1.0000 | 0.0000 | 1.0000 |
| 2 | 2 | 192918 | 0.6204 | 0.6103 | 0.8096 |
| 2 | 3 | 2649 | 0.3450 | 1.0155 | 0.6718 |
| 2 | 4 | 10 | 0.1000 | 1.3006 | 0.6360 |
| 3 | 2 | 282547 | 0.8705 | 0.1876 | 0.8706 |
| 3 | 3 | 5718 | 0.6207 | 0.4902 | 0.6706 |
| 3 | 4 | 39 | 0.2564 | 0.7828 | 0.4891 |

