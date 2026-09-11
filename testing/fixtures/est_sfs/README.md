# est-sfs reference fixture

Reference output from the original est-sfs (Keightley & Jackson 2018), version
2.04, for `testing/test_est_sfs_agreement.py`. One directory per outgroup
count (`n_out_1/`, `n_out_2/`, `n_out_3/`), since est-sfs supports one, two,
or three outgroups and the ladder's topology differs between them: with a
single outgroup there is no join node, so `O_1` hangs straight off the ingroup
MRCA and the one branch carries the whole divergence.

`generate.py` writes all three and runs est-sfs on each. It is the definition
of the fixture. The prose below only describes what it does.

```
python generate.py                 # regenerate everything
python generate.py --counts 1 2    # just these
python generate.py --no-run        # inputs only, no est-sfs
```

## Design

Sites are evolved down the outgroup ladder rather than enumerated: an ancestral
state uniform over A/C/G/T at the ingroup MRCA, an ingroup derived count from a
neutral `1/i` spectrum over 20 haplotypes, and the outgroup states evolved from
that MRCA under JC69 (40000 sites). Both tools then have identifiable rates and
class weights. An earlier version enumerated every ingroup split against every
outgroup configuration. That design is symmetric under exchanging ancestral and
derived, so est-sfs's ancestral-probability fit is unidentifiable and lands at
0.5.

The realised ingroup-to-outgroup divergences are the same at every count
(`0.0035`, `0.0054`, `0.0068`, each count using the prefix it needs), so `O_1`
sits at the same distance whether or not `O_2` and `O_3` are present. A count
therefore changes only how many outgroups carry the signal. est-sfs recovers
the rates from the terminal branches closely and the internal ones loosely,
which is the identifiability one expects of a ladder:

| count | simulated `K`                              | est-sfs fitted `K`                                    |
| ----- | ------------------------------------------ | ----------------------------------------------------- |
| 1     | 0.0035                                     | 0.003609                                              |
| 2     | 0.0020, 0.0015, 0.0034                     | 0.001882, 0.001520, 0.003420                          |
| 3     | 0.0020, 0.0015, 0.0018, 0.0016, 0.0030     | 0.001956, 0.001685, 0.001397, 0.002079, 0.003009      |

Per directory:

- `config.txt`, `sites.txt`, `seed.txt`: inputs, as est-sfs reads them.
- `truth.txt`: per site: ancestral allele, derived count, derived allele, then
  the simulated state at each internal ladder node.
- `p_anc.txt`: per-site output. Header lines start with `0 `, and fitted
  branch rates are on the `0 Rates:` line. Data lines are
  `<site> <config> <p_anc_major> <ptree_1> ...`.
- `sfs.txt`: the fitted spectrum.
- `hidden_states.npz`: the simulated state at the ingroup MRCA and at each
  internal ladder node (`ingroup_mrca`, `n1`, `n2`), so a reading at any of them
  can be graded against what the generator drew rather than against another
  estimator.

## Reading the output

`p_anc_major` is the posterior that the major ingroup allele is ancestral at
the ingroup MRCA, carrying est-sfs's per-frequency-class weight `sfs_p`, which
it fits by maximum likelihood (`est-sfs.c:353-360`, reached with
`use_sfs_p_as_prior = 1`). It normalises over two hypotheses, major or minor
ancestral. Sites with a monomorphic ingroup are reported as `1.0` regardless of
the outgroups. est-sfs does not infer the ancestral allele for them.

The `4^(n_outgroup-1)` `ptree` entries are the joint over the ladder's internal
node states, indexed `n_1 = (i - 1) mod 4` and `n_2 = (i - 1) div 4`
(`set_up_state_tab_3_outgroup`). For segregating sites they are
`(ptree1[j] + ptree2[j]) / 2`, where `ptree1` conditions on the ingroup MRCA
carrying the major allele and `ptree2` on the minor: a marginal under a uniform
prior over the two segregating alleles, carrying no frequency information.
The `0.5` factor is harmless on the log-likelihood line above it, where a
constant drops out of the fit, but on `ptree` it replaces `sfs_p` and changes
the reported internal-node posterior.

Marginalising that joint onto `n_1` and `n_2` reproduces this kernel's readout
at the same nodes to within 1.2e-04 when the ingroup enters with the same equal
weighting. Under the Kingman weight the two differ by up to 0.49 at `n_1`,
which is the frequency information est-sfs does not propagate above `b1`. By
`n_2` the ladder has washed it out and the two agree again to 0.002.

## Expected agreement

MAP estimates match on every site with an unambiguous major ingroup allele, at all
three outgroup counts. Sites with an exact tie are excluded: "major allele" is
undefined and the two implementations break the tie differently.

Posteriors differ by two measured components. One scales with branch length,
from est-sfs approximating branch transition probabilities by a Poisson
mutation count truncated at three (`compute_poisson_vec`) where the kernel here
uses the exact `P(t)`. The other is independent of branch length, from est-sfs
fitting one ancestral probability per frequency class where
`KingmanIngroupWeight` fixes it at `(n - i) / n`.

| count | comparable sites | mean abs. diff. | max abs. diff. |
| ----- | ---------------- | --------------- | -------------- |
| 1     | 38897            | 4.5e-05         | 1.1e-02        |
| 2     | 38864            | 5.4e-05         | 2.0e-02        |
| 3     | 38838            | 1.9e-05         | 2.0e-02        |

Which component dominates depends on how much outgroup evidence there is. With
three outgroups most sites are decided confidently by the outgroups and both
tools saturate, so what is left is the class-weight difference: the per-class
mean residual falls with the major count (4.6e-05 at a near tie, 9.4e-06 at
17-19 copies) and the worst site is a near tie. With one or two outgroups the
evidence is weaker, no class saturates, and the branch-length component is
comparable throughout. The per-class mean then rises with the major count
instead.
