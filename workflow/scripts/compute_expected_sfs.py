"""Expected (fractional) vs MAP unfolded SFS on the robustness baseline
ARG (10-outgroup ladder), n_out=0.

For each biallelic ingroup site the unfolded derived count depends on which
allele is ancestral. A MAP estimate commits each site to one bin; the
posterior instead spreads it fractionally across the two candidate bins
(i and n-i). Three spectra are accumulated over the true ingroup derived count
i in {1, ..., n-1} --- true, MAP, and expected --- plus the per-bin
confidence histogram (``p_max`` binned, matching
``_robustness_common.summarize_persite``). The spectra are pooled over the
baseline scenario's independent chunks, the same simulation the robustness
heatmap uses, so there is a single baseline sim.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tskit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc  # noqa: E402
from ancestree.inference import ARGBasedInference  # noqa: E402
from ancestree.models import JC69  # noqa: E402

try:
    CHUNKS = [Path(p) for p in snakemake.input.trees]  # type: ignore[name-defined]
    METAS = [Path(p) for p in snakemake.input.metas]  # type: ignore[name-defined]
    OUT = Path(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    DATA = Path("results/data")
    CHUNKS = [DATA / f"baseline10_chunk{i}.trees"
              for i in range(rc.ROBUSTNESS_MSP_N_CHUNKS)]
    METAS = [DATA / f"baseline10_chunk{i}_meta.json"
             for i in range(rc.ROBUSTNESS_MSP_N_CHUNKS)]
    OUT = DATA / "expected_sfs_baseline_n0.json"

# The chunks are i.i.d. replicates of one scenario, so the panel description
# is read from the first of them.
meta = json.loads(METAS[0].read_text())
ingroup_names = list(meta["ingroup_names"])
mu = float(meta["mu"])
n = int(meta["n_ingroup"])

T = np.zeros(n + 1)  # true
M = np.zeros(n + 1)  # MAP
E = np.zeros(n + 1)  # expected / fractional
conf_hist: dict[str, list] = {}
n_used = 0

for chunk_path in CHUNKS:
    ts = tskit.load(str(chunk_path))
    sample_id_to_node = {f"tsk_{ind.id}": int(ind.nodes[0])
                         for ind in ts.individuals()}
    ingroup_counts, truth = {}, {}
    for v in ts.variants():
        pos = int(v.site.position)
        cnt = Counter(v.alleles[v.genotypes[j]] for j in range(n)
                      if v.alleles[v.genotypes[j]] not in ("", None))
        ingroup_counts[pos] = cnt
        truth[pos] = v.site.ancestral_state

    nodes = [sample_id_to_node[s] for s in ingroup_names]
    ts_sub = ts.simplify(samples=nodes, filter_sites=False)
    inf = ARGBasedInference(ts_sub, JC69(), mu=mu,
                            sample_map={s: i for i, s in enumerate(ingroup_names)},
                            recurrence="full", progress=False)
    for site, post in inf.infer():
        pos = int(site.pos)
        cnt = ingroup_counts.get(pos)
        anc_true = truth.get(pos)
        if cnt is None or anc_true is None:
            continue
        alleles = [a for a, c in cnt.items() if c > 0]
        if len(alleles) != 2:  # biallelic ingroup only
            continue
        nobs = sum(cnt.values())
        if nobs != n:  # fully-called sites only
            continue
        der_true = n - cnt[anc_true]
        if der_true == 0 or der_true == n:  # segregating only
            continue
        T[der_true] += 1
        amap = post.map_allele
        if amap in cnt:
            M[n - cnt[amap]] += 1
        w = {a: max(post[a], 0.0) for a in alleles}
        z = sum(w.values()) or 1.0
        for a in alleles:
            E[n - cnt[a]] += w[a] / z
        p_true = w[anc_true] / z
        conf = p_true if p_true >= 0.5 else 1.0 - p_true
        conf_hist.setdefault(str(der_true),
                             [0] * rc.N_CONF_BINS)[rc._conf_bin(conf)] += 1
        n_used += 1
    print(f"{chunk_path.name}: pooled, running total {n_used} sites", flush=True)

print(f"biallelic segregating sites used: {n_used}", flush=True)
out = {"n": n, "n_used": n_used,
       "true": T.tolist(), "map": M.tolist(), "expected": E.tolist(),
       "conf_hist": conf_hist}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out))
print("wrote", OUT, flush=True)
