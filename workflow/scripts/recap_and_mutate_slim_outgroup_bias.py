"""SLiM-outgroup-bias benchmark: recapitate the SLiM ts, overlay neutral
mutations, write a final tree sequence + VCF + meta.

The upstream SLiM script simulates the ingroup p0 + N outgroups p1..pN
(nested closest-first ladder) in a single forward-time run. N is read from
the length of ``--outgroup-split-times-target`` (3 for the B14 / est-sfs
benchmark, 10 for the robustness heatmap), so this step is purely a
clean-up pass:

1. Load the SLiM ts (already has the 4-population structure + correct
   nested splits + per-population sample sets).
2. :func:`pyslim.recapitate` extends the SLiM trees back through the
   deep coalescent above the deepest split (T3) so every lineage has an
   ancestor at the root — required before mutation overlay.
3. :func:`msprime.sim_mutations` adds neutral JC69 mutations across the
   whole ARG with ``keep=False`` (drops SLiM's integer-coded selection
   mutations) so every site lives in a clean A/C/G/T alphabet.
4. Identify ingroup vs outgroup samples by their tskit population
   metadata. Pick one haploid per outgroup pop (closest first by split
   time → p1, p2, p3) and subsample p0 to ``--n-ingroup-haps``.
5. Write VCF + meta. ``outgroup_names`` is the closest-first ladder
   (``out_T1, out_T2, out_T3``) that :class:`OutgroupLadderTree` expects.
6. Truth ancestral state at each polymorphic site = ``site.ancestral_state``
   on the mutated ts (i.e. the state at the deepest local-tree root).

Run via snakemake or directly as ``python recap_and_mutate_slim_outgroup_bias.py``.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import msprime
import numpy as np
import pyslim
import tskit


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """CLI fallback when not driven by snakemake. Mirrors the params block."""
    p = argparse.ArgumentParser()
    p.add_argument("--slim-trees", required=True)
    p.add_argument("--out-trees", required=True)
    p.add_argument("--out-vcf", required=True)
    p.add_argument("--out-meta", required=True)
    p.add_argument("--f-del", type=float, required=True)
    p.add_argument("--chunk-idx", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mu-slim", type=float, required=True)
    p.add_argument("--rec-rate-slim", type=float, required=True)
    p.add_argument("--n-e-slim", type=int, required=True)
    p.add_argument("--mu-target", type=float, required=True)
    p.add_argument("--time-scaling", type=float, required=True,
                   help="s = Ne_target / Ne_slim; back-converts SLiM times to target scale.")
    p.add_argument("--outgroup-split-times-target", type=float, nargs="+", required=True,
                   help="Closest-first (T1, T2, T3) at the target Ne scale.")
    p.add_argument("--n-ingroup-haps", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    return p.parse_args(argv)


try:
    in_slim_trees = snakemake.input.slim_trees  # type: ignore[name-defined]
    out_trees = snakemake.output.trees  # type: ignore[name-defined]
    out_vcf = snakemake.output.vcf  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    f_del = float(snakemake.wildcards.f_del)  # type: ignore[name-defined]
    chunk_idx = int(snakemake.wildcards.chunk_idx)  # type: ignore[name-defined]
    base_seed = int(snakemake.params.seed)  # type: ignore[name-defined]
    mu_slim = float(snakemake.params.mu_slim)  # type: ignore[name-defined]
    rec_rate_slim = float(snakemake.params.rec_rate_slim)  # type: ignore[name-defined]
    N_e_slim = int(snakemake.params.N_e_slim)  # type: ignore[name-defined]
    mu_target = float(snakemake.params.mu_target)  # type: ignore[name-defined]
    time_scaling = float(snakemake.params.time_scaling)  # type: ignore[name-defined]
    # Closest-first split times at the target Ne scale. A list of length
    # n_out (3 for the est-sfs cell, 10 for the robustness cell), passed
    # straight through as a snakemake param.
    outgroup_split_times_target = [
        float(t)
        for t in snakemake.params.outgroup_split_times_target  # type: ignore[name-defined]
    ]
    n_ingroup_haps = int(snakemake.params.n_ingroup_haps)  # type: ignore[name-defined]
    seq_len = int(snakemake.params.seq_len)  # type: ignore[name-defined]
except NameError:
    _args = _parse_args(sys.argv[1:])
    in_slim_trees = _args.slim_trees
    out_trees = _args.out_trees
    out_vcf = _args.out_vcf
    out_meta = _args.out_meta
    f_del = _args.f_del
    chunk_idx = _args.chunk_idx
    base_seed = _args.seed
    mu_slim = _args.mu_slim
    rec_rate_slim = _args.rec_rate_slim
    N_e_slim = _args.n_e_slim
    mu_target = _args.mu_target
    time_scaling = _args.time_scaling
    outgroup_split_times_target = list(_args.outgroup_split_times_target)
    n_ingroup_haps = _args.n_ingroup_haps
    seq_len = _args.seq_len

# Each chunk uses an independent per-chunk seed so the per-chunk SLiM /
# recap / mutate runs are reproducible AND statistically independent.
seed = base_seed + chunk_idx

STATES = ("A", "C", "G", "T")
_STATE_INDEX = {s: i for i, s in enumerate(STATES)}

# Number of outgroups is the length of the split-times list (closest-first),
# so this script handles 3 (B14) or 10 (robustness) identically. SLiM names
# the outgroup subpops p1..pN. The manuscript display names are out_T1..out_TN.
n_outgroups = len(outgroup_split_times_target)
outgroup_pops = [f"p{k}" for k in range(1, n_outgroups + 1)]  # closest-first
outgroup_names = [f"out_T{k}" for k in range(1, n_outgroups + 1)]  # closest-first
outgroup_pop_to_display = dict(zip(outgroup_pops, outgroup_names))


# --------------------------------------------------------------- Load SLiM ts
print(f"Loading SLiM ts: {in_slim_trees}", flush=True)
slim_ts = tskit.load(in_slim_trees)
print(
    f"  SLiM ts: {slim_ts.num_samples} samples, "
    f"{slim_ts.num_sites} sites, {slim_ts.num_trees} trees, "
    f"{slim_ts.num_populations} populations",
    flush=True,
)
pop_name_by_id = {pop.id: pop.metadata.get("name", "") for pop in slim_ts.populations()}
print(f"  populations: {pop_name_by_id}", flush=True)

# --------------------------------------------------------------- Recapitate
t0 = time.perf_counter()
# pyslim.recapitate extends every uncoalesced lineage in a single
# ancestral population of size ``ancestral_Ne``. At the deepest SLiM
# tick (gen 1), the surviving subpops are p0 (size N_E_SLIM) and p3
# (size N_E_OUT), so we coalesce them in a single ancestral pop of size
# N_e_slim (matching the ingroup's effective size at the split).
recap_ts = pyslim.recapitate(
    slim_ts,
    ancestral_Ne=N_e_slim,
    recombination_rate=rec_rate_slim,
    random_seed=seed,
)
print(
    f"  Recapitated: {recap_ts.num_samples} samples, "
    f"{recap_ts.num_sites} sites, {recap_ts.num_trees} trees "
    f"({time.perf_counter() - t0:.2f}s)",
    flush=True,
)

# --------------------------------------------------------------- Mutate
t0 = time.perf_counter()
# msprime.sim_mutations with keep=False strips SLiM's selection mutations
# (integer-coded states, not A/C/G/T) and overlays clean JC69 mutations
# across the whole ARG (ingroup + outgroup branches). The mutation rate
# at the SLiM scale is ``mu_slim`` = mu_target × time_scaling so
# theta = 4·Ne·μ at every Ne is invariant.
mut_ts = msprime.sim_mutations(
    recap_ts, rate=mu_slim, model=msprime.JC69(),
    random_seed=seed, keep=False,
    discrete_genome=True,
)
print(
    f"  Mutated (JC69 overlay, keep=False, rate={mu_slim:g}): "
    f"{mut_ts.num_sites} sites ({time.perf_counter() - t0:.2f}s)",
    flush=True,
)

# --------------------------------------------------------------- Identify samples
ingroup_sample_nodes: list[int] = []
outgroup_sample_nodes_by_name: dict[str, list[int]] = {p: [] for p in outgroup_pops}
for node_id in mut_ts.samples():
    node = mut_ts.node(int(node_id))
    pop_name = pop_name_by_id.get(int(node.population), "")
    if pop_name == "p0":
        ingroup_sample_nodes.append(int(node_id))
    elif pop_name in outgroup_sample_nodes_by_name:
        outgroup_sample_nodes_by_name[pop_name].append(int(node_id))
    # else: ancestral / recap-only populations — not sampled.
print(
    f"  ingroup p0: {len(ingroup_sample_nodes)} haps; "
    + ", ".join(f"{p}: {len(outgroup_sample_nodes_by_name[p])}"
                for p in outgroup_pops),
    flush=True,
)

# --------------------------------------------------------------- Subsample
rng_sub = np.random.default_rng(seed)
ingroup_sample_nodes = np.array(ingroup_sample_nodes, dtype=np.int64)
if ingroup_sample_nodes.size > n_ingroup_haps:
    chosen = rng_sub.choice(
        ingroup_sample_nodes.size, size=n_ingroup_haps, replace=False,
    )
    chosen.sort()
    ingroup_sample_nodes = ingroup_sample_nodes[chosen]
elif ingroup_sample_nodes.size < n_ingroup_haps:
    print(
        f"  WARNING: only {ingroup_sample_nodes.size} ingroup haps available "
        f"(requested {n_ingroup_haps}); using all of them.",
        flush=True,
    )

# Pick exactly 1 haploid per outgroup pop. The OutgroupLadderTree ladder
# is closest-first → p1 (T1) → … → pN (TN).
outgroup_picked: dict[str, int] = {}
for pop_name in outgroup_pops:
    avail = outgroup_sample_nodes_by_name[pop_name]
    if not avail:
        raise RuntimeError(
            f"No samples found in outgroup population {pop_name}; "
            f"check the SLiM script kept the pop alive at sim end."
        )
    picked = int(rng_sub.choice(avail))
    outgroup_picked[pop_name] = picked

keep_nodes = list(ingroup_sample_nodes) + [outgroup_picked[p] for p in outgroup_pops]
mut_ts = mut_ts.simplify(samples=keep_nodes, filter_sites=True)

# After simplify, the new sample order is exactly ``keep_nodes`` in
# their NEW node-id space. tskit's simplify assigns sample ids 0..k-1
# in the order of the ``samples`` argument, so:
n_ingroup_kept = int(ingroup_sample_nodes.size)
new_samples = list(mut_ts.samples())
ingroup_indices = list(range(n_ingroup_kept))
outgroup_indices_by_pop = {
    p: n_ingroup_kept + i for i, p in enumerate(outgroup_pops)
}
print(
    f"  Sub-sampled to {n_ingroup_kept} ingroup haps + {n_outgroups} outgroups; "
    f"{mut_ts.num_sites} polymorphic sites remain",
    flush=True,
)

n_haps = n_ingroup_kept
ingroup_names = [f"ingroup_{i}" for i in range(n_haps)]
# outgroup_names / outgroup_pop_to_display were derived up front (closest-first).

# --------------------------------------------------------------- Per-site states
positions: list[int] = []
ancestral_states: list[str] = []
ingroup_alleles_per_site: list[list[str]] = []
outgroup_alleles_per_site: dict[str, list[str]] = {n: [] for n in outgroup_names}
allele_lists: list[list[str]] = []

for variant in mut_ts.variants():
    alleles = list(variant.alleles)
    if not all((a in STATES) for a in alleles if a is not None):
        continue
    ancestral = mut_ts.site(variant.site.id).ancestral_state
    if ancestral not in STATES:
        continue
    gens = variant.genotypes
    haplotype_alleles = [alleles[int(g)] if int(g) >= 0 else None for g in gens]
    positions.append(int(variant.site.position))
    ancestral_states.append(ancestral)
    ingroup_alleles_per_site.append([haplotype_alleles[i] for i in ingroup_indices])
    for pop_name in outgroup_pops:
        display = outgroup_pop_to_display[pop_name]
        outgroup_alleles_per_site[display].append(
            haplotype_alleles[outgroup_indices_by_pop[pop_name]]
        )
    allele_lists.append(alleles)

n_sites = len(positions)
print(
    f"  Polymorphic sites for inference: {n_sites} (after A/C/G/T filter)",
    flush=True,
)

# Diagnostic: per-outgroup divergence from truth ON POLYMORPHIC SITES.
# Note: this conditional rate is NOT mu·T — polymorphic sites are biased
# toward mutations falling on long branches, and the outgroup branches
# (~T_k gens each) dominate the total tree branch length (the ingroup
# MRCA is at most a few Ne_slim above the present, ~3·N_e_slim = 3000
# gens vs T_k ~ 30000-90000 gens). The OutgroupLadderTree fit recovers
# the true unconditional K_k = mu·T_k from the per-site likelihood
# evaluated against the full sequence length (via n_target_sites in
# FixedTreeInference) — see the report for the fitted K_k.
truth_idx = np.array([_STATE_INDEX[s] for s in ancestral_states], dtype=np.int8)
for k, display in enumerate(outgroup_names):
    og_idx = np.array(
        [_STATE_INDEX[a] if a is not None else -1
         for a in outgroup_alleles_per_site[display]], dtype=np.int8,
    )
    n_diff = int(((og_idx != truth_idx) & (og_idx >= 0)).sum())
    n_obs = int((og_idx >= 0).sum())
    expected_K_unconditional = mu_target * outgroup_split_times_target[k]
    print(
        f"  Outgroup {display} (T_target={outgroup_split_times_target[k]:g}): "
        f"{n_diff}/{n_obs} diverged from truth on poly sites "
        f"({100 * n_diff / max(1, n_obs):.2f}%); "
        f"unconditional K = μ·T ≈ {100 * expected_K_unconditional:.4f}% "
        f"(over all sites in the {seq_len}-bp chunk)",
        flush=True,
    )

# --------------------------------------------------------------- VCF
all_sample_names = ingroup_names + outgroup_names

vcf_lines: list[str] = []
vcf_lines.append("##fileformat=VCFv4.2")
vcf_lines.append("##source=ancestree_slim_outgroup_bias")
vcf_lines.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
vcf_lines.append(f"##contig=<ID=1,length={seq_len}>")
vcf_lines.append("#" + "\t".join(
    ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    + all_sample_names
))

for s in range(n_sites):
    pos = positions[s]
    ing_haps = ingroup_alleles_per_site[s]
    og_haps = [outgroup_alleles_per_site[name][s] for name in outgroup_names]
    site_alleles = list(allele_lists[s])
    for a in og_haps:
        if a not in site_alleles:
            site_alleles.append(a)
    ref = site_alleles[0]
    alt = site_alleles[1:] if len(site_alleles) > 1 else ["."]
    alt_str = ",".join(alt)
    a_to_idx = {a: i for i, a in enumerate(site_alleles)}
    gts = (
        [str(a_to_idx[a]) if a is not None else "." for a in ing_haps]
        + [str(a_to_idx[a]) for a in og_haps]
    )
    vcf_lines.append("\t".join([
        "1", str(pos + 1),  # 1-based VCF
        ".", ref, alt_str, ".", "PASS", ".", "GT",
        *gts,
    ]))

Path(out_vcf).parent.mkdir(parents=True, exist_ok=True)
with open(out_vcf, "w") as f:
    f.write("\n".join(vcf_lines) + "\n")
print(
    f"  Wrote VCF: {out_vcf} ({n_sites} sites, {len(all_sample_names)} haplotypes)",
    flush=True,
)

# --------------------------------------------------------------- ts + meta
mut_ts.dump(out_trees)
print(f"  Wrote ts: {out_trees}", flush=True)

meta = {
    "f_del": f_del,
    "chunk_idx": chunk_idx,
    "seed": seed,
    "base_seed": base_seed,
    "mu_slim": mu_slim,
    "mu_target": mu_target,
    "time_scaling": time_scaling,
    "rec_rate_slim": rec_rate_slim,
    "N_e_slim": N_e_slim,
    "outgroup_split_times_target": outgroup_split_times_target,
    "n_ingroup_haps": n_haps,
    "ingroup_names": ingroup_names,
    "outgroup_names": outgroup_names,  # closest-first ladder
    "n_sites": n_sites,
    "seq_len": seq_len,
    "positions": positions,
    "truth_ancestral_states": ancestral_states,  # parallel to positions
    "vcf_path": str(Path(out_vcf).resolve()),
    "trees_path": str(Path(out_trees).resolve()),
}
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Wrote meta: {out_meta}", flush=True)
