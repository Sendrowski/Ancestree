# Missing-data robustness sweep (B12)

Per-bin SFS accuracy as a function of outgroup-tip missingness.
Single simulated ARG (B6-style demography: ingroup of 20 +
3 outgroups at 3e5/9e5/1.5e6 gens); each cell randomly nulls out
``missing_frac`` of the outgroup tip slots per site (independent
Bernoulli per (site, outgroup)) with a per-cell RNG seed, then runs
``FixedTreeInference`` (HKY + empirical π, Kingman prior,
full). 3 seeds per ``missing_frac`` for error bars.

**Folded SFS bin** = ingroup minor-allele count. Each row reports
the per-bin accuracy mean ± std across the 3 seeds.

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
- ``missing_fractions`` = [0.0, 0.05, 0.1, 0.25, 0.5]
- ``seeds`` = [100, 101, 102]

## Overall summary (averaged over 3 seeds)

| missing_frac | n_seeds | n_sites | empirical mask rate | accuracy (mean ± std) | mean max_prob (mean ± std) | mean entropy bits |
|-------------:|--------:|--------:|--------------------:|----------------------:|---------------------------:|------------------:|
| 0.00 | 3 | 288304 | 0.000 | 0.6898 ± 0.0000 | 0.9998 ± 0.0000 | 0.0014 |
| 0.05 | 3 | 288304 | 0.050 | 0.6898 ± 0.0000 | 0.9997 ± 0.0000 | 0.0015 |
| 0.10 | 3 | 288304 | 0.100 | 0.6898 ± 0.0000 | 0.9997 ± 0.0000 | 0.0017 |
| 0.25 | 3 | 288304 | 0.250 | 0.6895 ± 0.0000 | 0.9993 ± 0.0000 | 0.0031 |
| 0.50 | 3 | 288304 | 0.500 | 0.6875 ± 0.0000 | 0.9974 ± 0.0000 | 0.0097 |

## Per-bin SFS accuracy (mean ± std over 3 seeds)

Rows: missing_frac. Columns: folded-SFS bin (minor-allele count).

| missing_frac | mc=1 | mc=2 | mc=3 | mc=4 | mc=5 | mc=6 | mc=7 | mc=8 | mc=9 | mc=10 |
|-------------:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 0.00 | 0.983 ± 0.000 | 0.978 ± 0.000 | 0.980 ± 0.000 | 0.976 ± 0.000 | 0.979 ± 0.000 | 0.980 ± 0.000 | 0.986 ± 0.000 | 0.983 ± 0.000 | 0.980 ± 0.000 | 0.983 ± 0.000 |
| 0.05 | 0.983 ± 0.000 | 0.978 ± 0.000 | 0.980 ± 0.001 | 0.976 ± 0.000 | 0.979 ± 0.000 | 0.980 ± 0.000 | 0.985 ± 0.001 | 0.983 ± 0.001 | 0.980 ± 0.000 | 0.983 ± 0.000 |
| 0.10 | 0.983 ± 0.000 | 0.978 ± 0.000 | 0.980 ± 0.001 | 0.976 ± 0.001 | 0.978 ± 0.001 | 0.981 ± 0.002 | 0.983 ± 0.000 | 0.982 ± 0.001 | 0.980 ± 0.001 | 0.983 ± 0.001 |
| 0.25 | 0.981 ± 0.001 | 0.977 ± 0.000 | 0.976 ± 0.001 | 0.972 ± 0.002 | 0.975 ± 0.001 | 0.976 ± 0.001 | 0.975 ± 0.003 | 0.978 ± 0.002 | 0.974 ± 0.000 | 0.976 ± 0.003 |
| 0.50 | 0.976 ± 0.001 | 0.966 ± 0.002 | 0.958 ± 0.002 | 0.949 ± 0.003 | 0.944 ± 0.001 | 0.942 ± 0.006 | 0.937 ± 0.008 | 0.933 ± 0.008 | 0.924 ± 0.003 | 0.918 ± 0.004 |

## Per-bin site counts (mean over seeds)

Same layout as the accuracy table; helps spot bins with few sites where the accuracy estimate is noisy.

| missing_frac | mc=1 | mc=2 | mc=3 | mc=4 | mc=5 | mc=6 | mc=7 | mc=8 | mc=9 | mc=10 |
|-------------:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 0.00 | 7944 | 4228 | 3013 | 2310 | 1959 | 1780 | 1753 | 1579 | 1502 | 727 |
| 0.05 | 7944 | 4228 | 3013 | 2310 | 1959 | 1780 | 1753 | 1579 | 1502 | 727 |
| 0.10 | 7944 | 4228 | 3013 | 2310 | 1959 | 1780 | 1753 | 1579 | 1502 | 727 |
| 0.25 | 7944 | 4228 | 3013 | 2310 | 1959 | 1780 | 1753 | 1579 | 1502 | 727 |
| 0.50 | 7944 | 4228 | 3013 | 2310 | 1959 | 1780 | 1753 | 1579 | 1502 | 727 |

## Reading the trend

accuracy is essentially flat across the full sweep (within ±1% of the no-masking baseline up to missing_frac=50%). HKY + Kingman prior + 3 outgroups is robust to heavy outgroup-tip dropout on this demography — the prior and the surviving outgroups carry enough signal to keep MAP recovery stable.

- Outgroup-tip masking erodes the polariser signal by removing
  evidence about the deep-ancestor state, which is exactly the
  information ``OutgroupLadderTree`` is built to exploit.
- The Kingman prior partially absorbs the loss at high-minor-count
  bins where the ingroup-SFS class is informative; the worst-hit
  bins should be the singletons / doubletons where the prior is
  near-symmetric and the outgroup vote was the deciding signal.
- The fully-masked endpoint (``missing_frac=1.0``) would degenerate
  to the no-outgroup mode (posterior = Kingman prior over the
  ingroup) — not in this sweep, but it's the asymptote each cell
  is moving toward as more outgroups get nulled out.

