"""Run SINGER on a sample panel and average the Felsenstein-kernel ancestral
posteriors over its posterior ARG samples.

SINGER (Deng, Nielsen & Song) is a Bayesian ARG sampler that, unlike tsinfer,
does not require the ancestral allele: ``-polar p`` sets the per-site
probability that the input orientation is correct, and ``-polar 0.5`` infers the
ARG from the genotypes alone. The panel genotypes (the ingroup, plus any
outgroups added as extra lineages) are exported to VCF, the thinned posterior ARG
samples of one MCMC chain are drawn and converted to tskit, their binary 0/1
alleles relabelled back to the observed nucleotides (SINGER works on a binary
alphabet, the kernel needs A/C/G/T and re-infers the orientation regardless of
the input), :class:`~ancestree.inference.ARGBasedInference` runs on each, and the
per-site posteriors are averaged --- the Bayesian marginalisation over the ARG posterior. Independent
``seed`` values give independent chains, averaged together downstream.

``orient_by_pos`` (pos -> ancestral allele) orients the VCF so REF is that
allele. Paired with a high ``-polar`` this is the fixed-tree-oriented variant.
``None`` leaves the orientation arbitrary for the ``-polar 0.5`` variant.

SINGER wires into the robustness baseline (Appendix figure) via the
``score_robustness_singer_*`` and ``infer_robustness_singer_arg_shard`` rules. It provides only Linux x86-64 release binaries.
The arm64 source build (``install_singer.sh``) compiles and validates on tiny
inputs but segfaults deterministically on standard panels, so the benchmark's
SINGER cells are produced on Linux rather than on Apple Silicon.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import tskit

from ancestree import ARGBasedInference, STATES

_STATE_IX = {s: i for i, s in enumerate(STATES)}


def singer_dir() -> str:
    """Directory holding the built ``singer`` + ``convert_to_tskit`` (SINGER_DIR)."""
    d = os.environ.get("SINGER_DIR")
    if not d or not os.path.isfile(os.path.join(d, "singer")):
        raise RuntimeError(
            "SINGER_DIR is not set to a built SINGER install; run "
            "workflow/scripts/install_singer.sh first.")
    return d


def _write_vcf(ts, path, orient_by_pos):
    """Write the biallelic-SNP ingroup genotypes to ``path`` (one window,
    contig '1'). Returns {int(pos): (REF, ALT)} for the relabelling step.
    With ``orient_by_pos`` the ancestral allele is placed as REF."""
    names = [f"tsk_{i}" for i in range(ts.num_samples)]
    refalt: dict[int, tuple[str, str]] = {}
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n##contig=<ID=1>\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(names) + "\n")
        for v in ts.variants():
            al = v.alleles
            if len(al) != 2 or any(a not in STATES for a in al if a is not None):
                continue
            g = np.asarray(v.genotypes)
            if len(set(g.tolist())) < 2:
                continue
            pos = int(round(v.site.position))
            ref, alt = al[0], al[1]
            geno = g.copy()
            anc = None if orient_by_pos is None else orient_by_pos.get(pos)
            if anc is not None and anc in al and anc != ref:
                ref, alt = alt, ref  # orient so REF = ancestral
                geno = 1 - geno
            refalt[pos] = (ref, alt)
            f.write(f"1\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t"
                    + "\t".join(str(int(x)) for x in geno) + "\n")
    return refalt


def score_arg_topology(arg, ts_panel, *, model, mu, sample_map,
                       focal=None, ingroup_samples=None,
                       outgroup_samples=None, bake_only=False,
                       mu_matches_time_units=False):
    """Score every site in ``ts_panel`` on the local trees of an inferred ARG.

    An inferred ARG (SINGER, Relate, ...) carries a genealogy at every genomic
    position, but the biallelic VCF the tool is fed drops the multiallelic /
    hypermutable sites, so scoring only the ARG's own site table would grade the
    tool on an easy subset (and let it spuriously outperform the true-ARG
    reference). Here the panel's full observed site set is laid onto the ARG's
    local trees: each site's observed genotypes are parsimony-placed onto
    ``arg.at(pos)`` with :meth:`tskit.Tree.map_mutations`, reproducing the
    observed tip alleles exactly. :meth:`ARGBasedInference.infer` then
    re-polarises from topology and tip alleles --- the identical computation the
    true-ARG reference performs, differing only in that the topology is
    inferred. The placed mutations merely encode the genotypes. Their labelling
    does not affect the posterior.

    :param arg: The inferred ARG (tskit tree sequence). Only its topology is used.
    :param ts_panel: The panel restricted to the scored window, with all sites
        retained (``simplify(..., filter_sites=False)``) so monomorphic and
        multiallelic sites are scored alongside the segregating ones.
    :param focal: Node to report the posterior at, as
        :class:`~ancestree.inference.ARGBasedInference` takes it: ``None``
        (default) or ``"panel_root"`` reads at each local tree's root,
        ``"ingroup_mrca"`` at the MRCA of ``ingroup_samples``.
    :param ingroup_samples: Ingroup ids in ``sample_map``'s naming, required to
        resolve ``focal="ingroup_mrca"``.
    :param outgroup_samples: Outgroup ids in ``sample_map``'s naming.
    :param mu_matches_time_units: Whether ``mu`` is expressed per unit of the
        ARG's own node times, as an undated ARG scored at a rate fitted to
        itself requires. Forwarded to
        :class:`~ancestree.inference.ARGBasedInference`.
    :return: ``(post_by_pos, obs_by_pos)`` --- per-position posterior array over
        ``STATES`` and observed-allele set, keyed by integer position.
    """
    L = float(arg.sequence_length)
    t = arg.dump_tables()
    t.sites.clear()
    t.mutations.clear()
    obs_by_pos: dict[int, set] = {}
    for v in ts_panel.variants():
        pos = int(round(v.site.position))
        if not (0.0 <= pos < L):
            continue
        alleles = list(v.alleles)
        anc, muts = arg.at(pos).map_mutations(np.asarray(v.genotypes), alleles)
        sid = t.sites.add_row(position=float(pos), ancestral_state=anc)
        for m in muts:
            t.mutations.add_row(site=sid, node=m.node,
                                derived_state=m.derived_state,
                                parent=-1, time=tskit.UNKNOWN_TIME)
        obs_by_pos[pos] = {a for a in alleles if a}
    t.sort()
    t.build_index()
    t.compute_mutation_parents()
    if bake_only:
        return t.tree_sequence(), obs_by_pos
    inf = ARGBasedInference(t.tree_sequence(), model, mu=mu,
                            sample_map=sample_map,
                            focal=focal, ingroup_samples=ingroup_samples,
                            outgroup_samples=outgroup_samples,
                            progress=False,
                            mu_matches_time_units=mu_matches_time_units)
    post_by_pos = {int(round(s.pos)): np.asarray(p.values, dtype=float)
                   for s, p in inf.infer()}
    return post_by_pos, obs_by_pos


def dump_args(draws, dirpath):
    """Persist a shard's ARG draws as tszip-compressed ``draw_{k}.tsz`` under
    ``dirpath`` --- the inference step's output, so scoring
    (:func:`score_arg_draws`) can re-run off disk without re-inference. tszip is
    lossless and ~3--4x smaller than a raw ``.trees`` dump, which matters for
    SINGER's tens of thousands of draws. A single-genealogy method writes one
    file.

    :return: the number of draws written."""
    import glob
    import tszip
    os.makedirs(dirpath, exist_ok=True)
    # Drop an earlier attempt's draws first. The rule's output is the
    # meta.json marker rather than the directory, so snakemake leaves the
    # directory in place on failure. A run writing fewer draws than the last
    # would otherwise load the leftover higher-numbered ones as its own.
    for stale in glob.glob(os.path.join(dirpath, "draw_*.tsz")):
        os.remove(stale)
    for k, arg in enumerate(draws):
        tszip.compress(arg, os.path.join(dirpath, f"draw_{k}.tsz"))
    return len(draws)


def load_args(dirpath):
    """Load persisted ARG draws (``draw_{k}.tsz``) from ``dirpath`` in MCMC
    order, the inverse of :func:`dump_args`."""
    import glob
    import re
    import tszip
    paths = glob.glob(os.path.join(dirpath, "draw_*.tsz"))
    paths.sort(key=lambda p: int(re.search(r"draw_(\d+)\.tsz$", p).group(1)))
    return [tszip.decompress(p) for p in paths]


def singer_infer(ts_ingroup, *, mu, rec_rate=1e-8, polar="0.5",
                 orient_by_pos=None, n_iters=300, thin=20, seed=0,
                 start=None, end=None, ne_diploid=None):
    """Run SINGER and return its thinned posterior ARG draws --- inference only.

    This is the expensive, arm64-blocked step, split from scoring so the draws
    can be persisted once (on Linux) and re-scored cheaply anywhere (see
    :func:`score_arg_draws`). ``seed`` selects the MCMC chain; ``orient_by_pos``
    orients the input VCF (the ``_ft`` variant) and affects only the inferred
    genealogy, not the scoring. Draws are returned in MCMC order (draw 0 first),
    so a downstream ``burn_in`` can discard the earliest, unconverged ones.

    :return: ``(draws, runtime)`` --- a list of tskit tree sequences (the
        thinned posterior ARG samples) and the wall time of the SINGER work.
    """
    import time
    sdir = singer_dir()
    Ne = (max(float(ts_ingroup.diversity(mode="site")) / (4.0 * mu), 1.0)
          if ne_diploid is None else float(ne_diploid))
    start = 0 if start is None else int(start)
    end = int(ts_ingroup.sequence_length) if end is None else int(end)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as wd:
        vcf = os.path.join(wd, "in.vcf")
        _write_vcf(ts_ingroup, vcf, orient_by_pos)
        out_pre = os.path.join(wd, "out")
        subprocess.run(
            [os.path.join(sdir, "singer"), "-Ne", str(int(round(Ne))), "-m", f"{mu:.3e}",
             "-r", f"{rec_rate:.3e}", "-input", os.path.join(wd, "in"),
             "-output", out_pre, "-start", str(start), "-end", str(end),
             "-polar", str(polar), "-n", str(n_iters), "-thin", str(thin),
             "-seed", str(int(seed)), "-ploidy", "1"],
            check=True, capture_output=True, text=True)
        n_samples = max(n_iters // thin, 1)
        subprocess.run(
            [sys.executable, os.path.join(sdir, "convert_to_tskit"), "-input", out_pre,
             "-output", os.path.join(wd, "arg"), "-start", "0",
             "-end", str(n_samples)],
            check=True, capture_output=True, text=True)
        draws = [tskit.load(os.path.join(wd, f"arg_{k}.trees"))
                 for k in range(n_samples)
                 if os.path.exists(os.path.join(wd, f"arg_{k}.trees"))]
    return draws, time.perf_counter() - t0


def score_arg_draws(draws, ts_panel, *, model, mu, sample_map,
                    burn_in=0, return_per_sample=False, focal=None,
                    ingroup_samples=None, outgroup_samples=None,
                    mu_matches_time_units=False):
    """Average :func:`score_arg_topology` posteriors over a list of ARG draws.

    The Bayesian marginalisation over an ARG posterior: score every in-window
    site on each draw's topology, then average across draws (dropping the first
    ``burn_in`` MCMC warm-up draws). A single-draw method (Relate / tsinfer)
    passes a one-element list with ``burn_in=0`` and gets a plain per-site score.

    :return: ``(post_by_pos, obs_by_pos)``, or, with ``return_per_sample``, a
        third element: the list of per-draw ``{pos: array}`` posteriors.
    """
    acc: dict[int, np.ndarray] = {}
    nseen: dict[int, int] = {}
    obs_by_pos: dict[int, set] = {}
    per_sample: list[dict] = []
    # Every draw is baked onto the panel's genotypes and handed to ONE
    # ARGBasedInference, which combines them in likelihood space with the
    # prior applied once --- the estimator the ensemble path uses and the one
    # the manuscript states. Accumulating normalised per-draw posteriors here
    # instead gave every draw equal weight, which is only correct for exact
    # posterior samples; SINGER's are MCMC draws.
    baked = []
    for k, arg in enumerate(draws):
        ts_baked, s_obs = score_arg_topology(
            arg, ts_panel, model=model, mu=mu,
            sample_map=sample_map, focal=focal,
            ingroup_samples=ingroup_samples,
            outgroup_samples=outgroup_samples, bake_only=True)
        obs_by_pos.update(s_obs)
        if k >= burn_in:
            baked.append(ts_baked)
        if return_per_sample:
            single, _ = score_arg_topology(
                arg, ts_panel, model=model, mu=mu,
                sample_map=sample_map, focal=focal,
                ingroup_samples=ingroup_samples,
                outgroup_samples=outgroup_samples,
                mu_matches_time_units=mu_matches_time_units)
            per_sample.append(single)
    if not baked:
        return ({}, obs_by_pos, per_sample) if return_per_sample \
            else ({}, obs_by_pos)
    inf = ARGBasedInference(
        baked[0] if len(baked) == 1 else baked, model, mu=mu,
        sample_map=sample_map, focal=focal,
        ingroup_samples=ingroup_samples, outgroup_samples=outgroup_samples,
        progress=False, mu_matches_time_units=mu_matches_time_units)
    post_by_pos = {int(round(st.pos)): np.asarray(pp.values, dtype=float)
                   for st, pp in inf.infer()}
    if return_per_sample:
        return post_by_pos, obs_by_pos, per_sample
    return post_by_pos, obs_by_pos
