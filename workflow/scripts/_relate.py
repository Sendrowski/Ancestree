"""Run Relate on a sample panel and score the Felsenstein-kernel ancestral
posteriors on its inferred genealogy.

Relate (Speidel et al. 2019) infers genome-wide genealogies from phased
haplotypes under a Li--Stephens copying model. Unlike a Bayesian sampler it
returns a single genealogy, so there is no posterior-ARG average. The panel
genotypes (the ingroup, plus any outgroups added as extra lineages) are exported
to a haploid VCF, converted to Relate's ``.haps``/``.sample`` with a flat genetic
map at the sequence's recombination rate, ``Relate --mode All`` builds and dates
the trees, the result is converted to tskit, its binary 0/1 alleles are
relabelled back to the observed nucleotides, and
:class:`~ancestree.inference.ARGBasedInference` runs once. Relate needs an
ancestral allele only to orient mutations, and it is left arbitrary (REF treated
as ancestral): the JC69 kernel re-infers the orientation from the topology and
the outgroup tips.

``orient_by_pos`` (pos -> ancestral allele) orients the VCF so REF is that
allele, the fixed-tree-oriented variant; ``None`` leaves orientation arbitrary.

Relate wires into the robustness baseline via the ``score_robustness_relate_*`` and ``infer_robustness_relate_arg_shard``
rules. It provides native binaries for every platform, including Apple Silicon
(``relate_v1.2.4_MacOSX_M``), so its cells run on arm64 unlike SINGER.
"""
import os
import subprocess
import tempfile
import time

import tskit

from _singer import _write_vcf  # shared panel VCF + kernel scoring


def relate_dir() -> str:
    """Directory holding the built ``bin/Relate`` (RELATE_DIR)."""
    d = os.environ.get("RELATE_DIR")
    if not d or not os.path.isfile(os.path.join(d, "bin", "Relate")):
        raise RuntimeError(
            "RELATE_DIR is not set to a Relate install with bin/Relate; run "
            "workflow/scripts/install_relate.sh first.")
    return d


def _write_genetic_map(path, start, end, rec_rate):
    """Flat Relate genetic map over ``[start, end]`` at constant ``rec_rate``
    (per bp per generation). Columns: bp position, rate in cM/Mb, cumulative cM
    (``1`` cM/Mb == ``1e-8`` per bp)."""
    rate_cm_mb = rec_rate * 1e8
    lo, hi = int(start), max(int(end), int(start) + 1)
    with open(path, "w") as f:
        f.write("Position(bp) Rate(cM/Mb) Map(cM)\n")
        f.write(f"{lo} {rate_cm_mb:.8f} 0.0\n")
        f.write(f"{hi} {rate_cm_mb:.8f} {rate_cm_mb * (hi - lo) / 1e6:.8f}\n")


def relate_infer(ts_ingroup, *, mu, rec_rate=1e-8, orient_by_pos=None,
                 seed=1, start=None, end=None, ne_diploid=None):
    """Infer Relate's genealogy for one panel and return it --- inference only.

    This is the arm64-native, expensive step, split from scoring so the ARG can
    be persisted once and re-scored cheaply (see :func:`score_arg_topology`).
    ``ts_ingroup`` is the panel (ingroup plus any outgroup lineages) restricted
    to the target window; ``start`` / ``end`` bound the genome region.
    ``orient_by_pos`` orients the input VCF (the ``_ft`` variant). It affects
    only the inferred genealogy, not the downstream scoring.

    :return: ``(arg, runtime)`` --- the inferred tskit tree sequence (its
        topology is what scoring consumes) and the wall time of the Relate work.
    """
    rdir = relate_dir()
    relate = os.path.join(rdir, "bin", "Relate")
    rff = os.path.join(rdir, "bin", "RelateFileFormats")
    # Relate's -N is the haploid effective size (2 x diploid Ne). Estimate the
    # diploid Ne by Watterson from the panel, as SINGER does.
    ne_dip = (max(float(ts_ingroup.diversity(mode="site")) / (4.0 * mu), 1.0)
              if ne_diploid is None else float(ne_diploid))
    n_hap = 2.0 * ne_dip
    start = 0 if start is None else int(start)
    end = int(ts_ingroup.sequence_length) if end is None else int(end)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as wd:
        vcf = os.path.join(wd, "in.vcf")
        _write_vcf(ts_ingroup, vcf, orient_by_pos)
        gmap = os.path.join(wd, "genetic_map.txt")
        _write_genetic_map(gmap, start, end, rec_rate)
        # VCF -> Relate haps/sample (REF treated ancestral, kernel re-polarises).
        subprocess.run(
            [rff, "--mode", "ConvertFromVcf", "--haps", os.path.join(wd, "in.haps"),
             "--sample", os.path.join(wd, "in.sample"), "-i", os.path.join(wd, "in")],
            check=True, capture_output=True, text=True)
        # Build + date the trees. Relate writes <prefix>.anc/.mut into its CWD,
        # so run it inside the temp dir with a bare output prefix.
        subprocess.run(
            [relate, "--mode", "All", "--haps", os.path.join(wd, "in.haps"),
             "--sample", os.path.join(wd, "in.sample"), "--map", gmap,
             "-N", f"{n_hap:.0f}", "-m", f"{mu:.3e}", "-o", "out", "--seed", str(int(seed))],
            check=True, capture_output=True, text=True, cwd=wd)
        # Relate -> tskit tree sequence (only its topology is used downstream).
        subprocess.run(
            [rff, "--mode", "ConvertToTreeSequence", "-i", os.path.join(wd, "out"),
             "-o", os.path.join(wd, "out")],
            check=True, capture_output=True, text=True)
        arg = tskit.load(os.path.join(wd, "out.trees"))
    return arg, time.perf_counter() - t0
