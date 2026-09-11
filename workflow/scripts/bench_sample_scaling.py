"""Runtime of the three inference modes against ingroup size.

One simulation is subsampled to a ladder of ingroup sizes and every mode is
timed on each, so the modes are compared over a range rather than at a single
panel size. Local-tree mode carries an ``O(n^2)`` pairwise stage and is the
only one whose cost turns on the sample count, which is what the curve shows.
"""
import json
import time
from pathlib import Path

import msprime  # noqa: E402

import ancestree as anc  # noqa: E402
from ancestree import JC69, FixedTreeInference, LocalTreeInference  # noqa: E402
from ancestree.priors import KingmanIngroupWeight  # noqa: E402

try:
    out_json = str(snakemake.output.json)  # type: ignore[name-defined]
    SIZES = list(snakemake.params.sizes)  # type: ignore[name-defined]
except NameError:
    out_json = "results/reports/sample_scaling.json"
    SIZES = [5, 10, 20, 40, 80]

LENGTH, REC, NE, MU, SEED = 100_000_000, 1e-8, 3e4, 1.25e-8, 42
SPLITS = [50_000, 100_000, 150_000]
N_ENSEMBLE = 32


def _simulate(n_ingroup):
    """Ingroup plus three outgroups on a nested-split ladder."""
    d = msprime.Demography()
    d.add_population(name="ingroup", initial_size=NE)
    for k in range(1, len(SPLITS) + 1):
        d.add_population(name=f"outgroup_{k}", initial_size=NE)
    d.add_population(name="anc_root", initial_size=NE)
    chain = ["ingroup"]
    for k in range(1, len(SPLITS) + 1):
        anc = "anc_root" if k == len(SPLITS) else f"anc_{k}"
        if anc != "anc_root":
            d.add_population(name=anc, initial_size=NE)
        d.add_population_split(time=SPLITS[k - 1], ancestral=anc,
                               derived=[chain[-1], f"outgroup_{k}"])
        chain.append(anc)
    sets = [msprime.SampleSet(n_ingroup, population="ingroup", ploidy=1)]
    sets += [msprime.SampleSet(1, population=f"outgroup_{k}", ploidy=1)
             for k in range(1, len(SPLITS) + 1)]
    ts = msprime.sim_ancestry(samples=sets, demography=d,
                              sequence_length=LENGTH, recombination_rate=REC,
                              random_seed=SEED)
    return msprime.sim_mutations(ts, rate=MU, model=msprime.JC69(),
                                 random_seed=SEED)


def _panel(ts):
    """``(sites, names, ingroup, outgroup)`` with samples named by node id."""
    samples = list(ts.samples())
    names = [str(int(n)) for n in samples]
    pop = {p.id: (p.metadata or {}).get("name") for p in ts.populations()}
    ingroup, outgroup = [], []
    for n, nm in zip(samples, names):
        (ingroup if pop[ts.node(int(n)).population] == "ingroup"
         else outgroup).append(nm)
    sites = [
        anc.Site(chrom="1", pos=int(v.site.position),
                alleles=tuple(a for a in v.alleles if a),
                tip_alleles={nm: v.alleles[g] for nm, g in zip(names, v.genotypes)})
        for v in ts.variants()
    ]
    return sites, names, ingroup, outgroup


def _timed(fn):
    t0 = time.perf_counter()
    n = fn()
    return time.perf_counter() - t0, n


rows = []
for n_ing in SIZES:
    ts = _simulate(n_ing)
    sites, names, ingroup, outgroup = _panel(ts)
    sample_map = {nm: int(node) for nm, node in zip(names, ts.samples())}

    t_arg, n_arg = _timed(lambda: sum(1 for _ in anc.ARGBasedInference(
        ts, JC69(), mu=MU, sample_map=sample_map, progress=False,
        ingroup_samples=ingroup, outgroup_samples=outgroup).infer()))

    def _fixed_tree():
        # Timed as fit+infer, matching the fixed-tree rows of the kernel table.
        inf = FixedTreeInference(
            sites, JC69(), n_target_sites=int(ts.sequence_length),
            ingroup_samples=ingroup, outgroup_samples=outgroup,
            ingroup_weight=KingmanIngroupWeight(ingroup_samples=ingroup),
            progress=False, baseline_check=False)
        inf.fit()
        return sum(1 for _ in inf.infer())

    t_ft, _ = _timed(_fixed_tree)

    t_lt, _ = _timed(lambda: sum(1 for _ in LocalTreeInference(
        sites, JC69(), mu=MU, rec_rate=REC, sample_names=names,
        ingroup_samples=ingroup, outgroup_samples=outgroup,
        sequence_length=float(ts.sequence_length), window="30snp",
        n_ensemble=N_ENSEMBLE, chunk_size="250kb", progress=False).infer()))

    rows.append({"n_ingroup": n_ing, "n_outgroup": len(outgroup),
                 "n_sites": len(sites), "arg_s": t_arg,
                 "fixed_tree_s": t_ft, "local_tree_s": t_lt})
    print(f"n={n_ing:3d}  sites={len(sites):6,}  arg={t_arg:6.1f}s  "
          f"fixed={t_ft:6.1f}s  local={t_lt:7.1f}s", flush=True)

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
Path(out_json).write_text(json.dumps(rows, indent=2))
print(f"wrote {out_json}", flush=True)
