# Substitution-model comparison (B9)

Per-cell ancestral-allele recovery as a function of the Ancestree
substitution model. The simulator mutates the ARG under ``msprime.HKY``
with both a transition/transversion bias and a non-uniform AT-rich
equilibrium distribution — every misspecified model is wrong along at
least one axis the simulator cares about.

All accuracy numbers below are on **ingroup-polymorphic sites only**
(matches B8's convention): ingroup-monomorphic sites carry no
polarisation signal and would dilute the headline.

**Simulator mutation model:** ``HKY(kappa=4.0, pi=(0.3, 0.2, 0.2, 0.3))`` over ``(A, C, G, T)``.

**Simulator settings**

- ``n_ingroup`` = 20
- ``n_outgroup_pops`` = 3
- ``length`` = 10000000.0
- ``mu`` = 1.25e-08
- ``rec_rate`` = 1e-08
- ``pop_size`` = 30000.0
- ``ingroup_pop_size`` = 30000.0
- ``outgroup_split_times`` = [300000.0, 900000.0, 1500000.0]
- ``kappa`` = 4.0
- ``equilibrium_frequencies`` = [0.3, 0.2, 0.2, 0.3]
- ``seed`` = 42

## Accuracy on ingroup-polymorphic sites

| model | acc (arg) | acc (vcf) | max_prob (arg) | max_prob (vcf) | baseline agree (arg) | baseline agree (vcf) | n_sites |
|:---|---:|---:|---:|---:|---:|---:|---:|
| jc69 | 0.9931 | 0.9843 | 0.9928 | 0.9975 | 0.9997 | 0.9909 | 45201 |
| k2 | 0.9931 | 0.9843 | 0.9928 | 0.9970 | 0.9998 | 0.9909 | 45201 |
| f81 | 0.9932 | 0.9844 | 0.9931 | 0.9975 | 0.9997 | 0.9910 | 45201 |
| hky | 0.9931 | 0.9844 | 0.9931 | 0.9970 | 0.9998 | 0.9910 | 45201 |
| gtr | 0.9931 | 0.9844 | 0.9931 | 0.9970 | 0.9998 | 0.9910 | 45201 |

**Baseline-MAP agreement** is the fraction of sites where Ancestree's
MAP allele matches the simple outgroup-majority rule
(:class:`~ancestree.inference.MajorityOutgroupInference`). On a well-
specified model these should be high — the polariser signal dominates
and the model + prior only nudge marginal calls. A model that drives
agreement noticeably *lower* than the truth-matching ``hky`` row is
shifting calls away from the bare outgroup vote; the value of that
shift depends on whether it's correctly flagging recurrence /
homoplasy or reflecting model misspecification.

## Model parameters per cell

| model | mode | empirical params | fitted κ (post-MLE) |
|:---|:---|:---|---:|
| jc69 | arg | kappa=1.0 | — |
| jc69 | vcf | kappa=1.0 | — |
| k2 | arg | kappa=3.8328438858216844 | — |
| k2 | vcf | kappa=3.8328438858216844 | 3.8358 |
| f81 | arg | pi=[0.305, 0.198, 0.198, 0.298] | — |
| f81 | vcf | pi=[0.305, 0.198, 0.198, 0.298] | — |
| hky | arg | pi=[0.305, 0.198, 0.198, 0.298], kappa=3.8206801786327724 | — |
| hky | vcf | pi=[0.305, 0.198, 0.198, 0.298], kappa=3.8206801786327724 | 4.0107 |
| gtr | arg | pi=[0.305, 0.198, 0.198, 0.298], rates=[0.244, 1.0, 0.253, 0.254, 1.026, 0.259] | — |
| gtr | vcf | pi=[0.305, 0.198, 0.198, 0.298], rates=[0.244, 1.0, 0.253, 0.254, 1.026, 0.259] | — |

## Fitted vs truth outgroup divergences (subs/site)

Truth is the ingroup-to-outgroup genetic divergence, the sum of the
two branches from their split: ``2 * mu * T_k * rho`` subs/site,
where ``rho`` is the expected number of substitutions per mutation
event under the simulating model. ``msprime`` scales its transition
matrix by the largest row sum, so a mutation event leaves the state
unchanged with probability ``P_ii`` and ``rho <= 1``. For our sim
``(T_1, T_2, T_3) = (3e+05, 9e+05, 1.5e+06)`` gen,
``μ = 1.25e-08`` per gen and ``rho = 0.8588`` → divergences
``[0.0064, 0.0193, 0.0322]`` subs/site.

| model | mode | div₁ (truth=0.0064) | div₂ (truth=0.0193) | div₃ (truth=0.0322) |
|:---|:---|---:|---:|---:|
| jc69 | vcf | 0.0067 (1.03×) | 0.0195 (1.01×) | 0.0323 (1.00×) |
| k2 | vcf | 0.0067 (1.04×) | 0.0196 (1.01×) | 0.0324 (1.01×) |
| f81 | vcf | 0.0067 (1.03×) | 0.0195 (1.01×) | 0.0323 (1.00×) |
| hky | vcf | 0.0067 (1.04×) | 0.0196 (1.01×) | 0.0324 (1.01×) |
| gtr | vcf | 0.0067 (1.04×) | 0.0196 (1.01×) | 0.0324 (1.01×) |

## Per-minor-allele-count breakdown (ingroup-polymorphic)

| model | mode | minor_count | n_sites | accuracy | mean max_prob |
|:---|:---|---:|---:|---:|---:|
| jc69 | arg | 1 | 13211 | 0.9924 | 0.9928 |
| jc69 | arg | 2 | 7327 | 0.9930 | 0.9923 |
| jc69 | arg | 3 | 5229 | 0.9925 | 0.9918 |
| jc69 | arg | 4 | 4012 | 0.9925 | 0.9925 |
| jc69 | arg | 5 | 3445 | 0.9945 | 0.9943 |
| jc69 | arg | 6 | 3046 | 0.9934 | 0.9920 |
| jc69 | arg | 7 | 2787 | 0.9932 | 0.9922 |
| jc69 | arg | 8 | 2535 | 0.9953 | 0.9938 |
| jc69 | arg | 9 | 2377 | 0.9941 | 0.9940 |
| jc69 | arg | 10 | 1232 | 0.9943 | 0.9959 |
| jc69 | vcf | 1 | 13211 | 0.9846 | 0.9988 |
| jc69 | vcf | 2 | 7327 | 0.9843 | 0.9982 |
| jc69 | vcf | 3 | 5229 | 0.9845 | 0.9979 |
| jc69 | vcf | 4 | 4012 | 0.9840 | 0.9975 |
| jc69 | vcf | 5 | 3445 | 0.9878 | 0.9970 |
| jc69 | vcf | 6 | 3046 | 0.9842 | 0.9957 |
| jc69 | vcf | 7 | 2787 | 0.9889 | 0.9968 |
| jc69 | vcf | 8 | 2535 | 0.9803 | 0.9945 |
| jc69 | vcf | 9 | 2377 | 0.9798 | 0.9962 |
| jc69 | vcf | 10 | 1232 | 0.9773 | 0.9932 |
| k2 | arg | 1 | 13211 | 0.9924 | 0.9928 |
| k2 | arg | 2 | 7327 | 0.9930 | 0.9922 |
| k2 | arg | 3 | 5229 | 0.9925 | 0.9917 |
| k2 | arg | 4 | 4012 | 0.9923 | 0.9924 |
| k2 | arg | 5 | 3445 | 0.9945 | 0.9943 |
| k2 | arg | 6 | 3046 | 0.9934 | 0.9919 |
| k2 | arg | 7 | 2787 | 0.9932 | 0.9921 |
| k2 | arg | 8 | 2535 | 0.9953 | 0.9937 |
| k2 | arg | 9 | 2377 | 0.9941 | 0.9939 |
| k2 | arg | 10 | 1232 | 0.9943 | 0.9958 |
| k2 | vcf | 1 | 13211 | 0.9846 | 0.9983 |
| k2 | vcf | 2 | 7327 | 0.9843 | 0.9977 |
| k2 | vcf | 3 | 5229 | 0.9845 | 0.9973 |
| k2 | vcf | 4 | 4012 | 0.9840 | 0.9973 |
| k2 | vcf | 5 | 3445 | 0.9878 | 0.9966 |
| k2 | vcf | 6 | 3046 | 0.9842 | 0.9953 |
| k2 | vcf | 7 | 2787 | 0.9889 | 0.9965 |
| k2 | vcf | 8 | 2535 | 0.9803 | 0.9942 |
| k2 | vcf | 9 | 2377 | 0.9798 | 0.9957 |
| k2 | vcf | 10 | 1232 | 0.9773 | 0.9928 |
| f81 | arg | 1 | 13211 | 0.9924 | 0.9931 |
| f81 | arg | 2 | 7327 | 0.9933 | 0.9926 |
| f81 | arg | 3 | 5229 | 0.9925 | 0.9922 |
| f81 | arg | 4 | 4012 | 0.9928 | 0.9928 |
| f81 | arg | 5 | 3445 | 0.9948 | 0.9946 |
| f81 | arg | 6 | 3046 | 0.9934 | 0.9926 |
| f81 | arg | 7 | 2787 | 0.9932 | 0.9927 |
| f81 | arg | 8 | 2535 | 0.9953 | 0.9939 |
| f81 | arg | 9 | 2377 | 0.9937 | 0.9940 |
| f81 | arg | 10 | 1232 | 0.9943 | 0.9960 |
| f81 | vcf | 1 | 13211 | 0.9846 | 0.9987 |
| f81 | vcf | 2 | 7327 | 0.9843 | 0.9981 |
| f81 | vcf | 3 | 5229 | 0.9845 | 0.9979 |
| f81 | vcf | 4 | 4012 | 0.9840 | 0.9975 |
| f81 | vcf | 5 | 3445 | 0.9878 | 0.9970 |
| f81 | vcf | 6 | 3046 | 0.9823 | 0.9957 |
| f81 | vcf | 7 | 2787 | 0.9878 | 0.9971 |
| f81 | vcf | 8 | 2535 | 0.9842 | 0.9948 |
| f81 | vcf | 9 | 2377 | 0.9815 | 0.9963 |
| f81 | vcf | 10 | 1232 | 0.9773 | 0.9932 |
| hky | arg | 1 | 13211 | 0.9924 | 0.9931 |
| hky | arg | 2 | 7327 | 0.9933 | 0.9925 |
| hky | arg | 3 | 5229 | 0.9927 | 0.9922 |
| hky | arg | 4 | 4012 | 0.9923 | 0.9927 |
| hky | arg | 5 | 3445 | 0.9948 | 0.9945 |
| hky | arg | 6 | 3046 | 0.9934 | 0.9925 |
| hky | arg | 7 | 2787 | 0.9932 | 0.9926 |
| hky | arg | 8 | 2535 | 0.9953 | 0.9939 |
| hky | arg | 9 | 2377 | 0.9937 | 0.9940 |
| hky | arg | 10 | 1232 | 0.9943 | 0.9959 |
| hky | vcf | 1 | 13211 | 0.9846 | 0.9983 |
| hky | vcf | 2 | 7327 | 0.9843 | 0.9976 |
| hky | vcf | 3 | 5229 | 0.9845 | 0.9972 |
| hky | vcf | 4 | 4012 | 0.9840 | 0.9972 |
| hky | vcf | 5 | 3445 | 0.9878 | 0.9965 |
| hky | vcf | 6 | 3046 | 0.9823 | 0.9953 |
| hky | vcf | 7 | 2787 | 0.9878 | 0.9966 |
| hky | vcf | 8 | 2535 | 0.9842 | 0.9944 |
| hky | vcf | 9 | 2377 | 0.9815 | 0.9958 |
| hky | vcf | 10 | 1232 | 0.9773 | 0.9927 |
| gtr | arg | 1 | 13211 | 0.9924 | 0.9931 |
| gtr | arg | 2 | 7327 | 0.9933 | 0.9925 |
| gtr | arg | 3 | 5229 | 0.9927 | 0.9922 |
| gtr | arg | 4 | 4012 | 0.9923 | 0.9927 |
| gtr | arg | 5 | 3445 | 0.9948 | 0.9945 |
| gtr | arg | 6 | 3046 | 0.9934 | 0.9925 |
| gtr | arg | 7 | 2787 | 0.9932 | 0.9926 |
| gtr | arg | 8 | 2535 | 0.9953 | 0.9939 |
| gtr | arg | 9 | 2377 | 0.9937 | 0.9940 |
| gtr | arg | 10 | 1232 | 0.9943 | 0.9959 |
| gtr | vcf | 1 | 13211 | 0.9846 | 0.9983 |
| gtr | vcf | 2 | 7327 | 0.9843 | 0.9976 |
| gtr | vcf | 3 | 5229 | 0.9845 | 0.9972 |
| gtr | vcf | 4 | 4012 | 0.9840 | 0.9972 |
| gtr | vcf | 5 | 3445 | 0.9878 | 0.9965 |
| gtr | vcf | 6 | 3046 | 0.9823 | 0.9953 |
| gtr | vcf | 7 | 2787 | 0.9878 | 0.9966 |
| gtr | vcf | 8 | 2535 | 0.9842 | 0.9944 |
| gtr | vcf | 9 | 2377 | 0.9815 | 0.9958 |
| gtr | vcf | 10 | 1232 | 0.9773 | 0.9927 |

## Reading the result

- JC69 assumes uniform equilibrium frequencies and one exchangeability
  for every substitution type.
- K2 assumes uniform equilibrium frequencies and separate transition
  and transversion exchangeabilities.
- F81 assumes the empirical equilibrium frequencies π̂ and one
  exchangeability for every substitution type, so the rate into a
  state is proportional to that state's frequency.
- HKY assumes π̂ and separate transition and transversion
  exchangeabilities, the parameterisation the simulator mutates under.
- GTR assumes π̂ and one free exchangeability for each of the six
  reversible substitution types.
- ARG mode vs fixed-tree (VCF) mode: ARG inference reads the local
  trees directly and uses the substitution model only to translate
  branch lengths into per-branch transition probabilities. Fixed-tree
  mode fits per-branch rates along the outgroup ladder under the same
  model, so the model enters both the Stage 1 branch-rate MLE and the
  per-site posterior.
- Neither per-bin accuracy nor branch-rate recovery differentiates
  the models on this setup. The three-outgroup signal dominates the
  per-site posterior, so accuracy is flat across the grid. Every model
  scales ``Q`` to one expected substitution per unit branch length, so
  the fitted outgroup divergences are in the same units whatever ``π``
  and ``κ`` the model assumes, and they agree with each other and with
  truth. See the fitted-vs-truth table above.

