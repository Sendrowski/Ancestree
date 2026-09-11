"""Regenerate the est-sfs reference fixtures, one per outgroup count.

Sites are evolved down the outgroup ladder rather than enumerated: an
ancestral state uniform over A/C/G/T at the ingroup MRCA, an ingroup derived
count from a neutral ``1/i`` spectrum over 20 haplotypes, and the outgroup
states evolved from that MRCA under JC69 along the ladder's branches. Both
tools then see data their models can fit, so est-sfs's rates and class weights
are identifiable. (An earlier design enumerated every ingroup split against
every outgroup configuration; that is symmetric under exchanging ancestral and
derived, so est-sfs's ancestral-probability fit is unidentifiable and lands at
0.5.)

The branch rates are chosen so the realised ingroup-to-outgroup divergences are
the SAME at every outgroup count -- dropping O_3 does not move O_1 or O_2. The
counts therefore differ only in how many outgroups carry the signal, which is
what the agreement test is comparing across.

Run from this directory::

    python generate.py            # writes n_out_{1,2,3}/, then runs est-sfs
    python generate.py --no-run   # write inputs only

est-sfs 2.04 is resolved from the workflow's bench conda env; pass
``--est-sfs PATH`` to point at another build.
"""
import argparse
import pathlib
import subprocess
import sys

import numpy as np

STATES = "ACGT"
N_SITES = 40_000
N_INGROUP = 20
SEED = 7

#: Ingroup-to-outgroup divergences, closest outgroup first. Every fixture uses
#: the prefix of this list matching its outgroup count, so a given outgroup sits
#: at the same distance whatever the count.
DIVERGENCES = [0.0035, 0.0054, 0.0068]

#: The internal I -> n_1 branch, shared by every count with >= 2 outgroups. The
#: remaining rates are solved from :data:`DIVERGENCES` in :func:`ladder_rates`.
K_ROOT = 0.0020

#: The n_1 -> n_2 branch, for the three-outgroup ladder.
K_INTERNAL = 0.0018


def ladder_rates(n_out):
    """Branch rates for an ``n_out`` ladder realising :data:`DIVERGENCES`.

    Ordered as :class:`~ancestree.trees.OutgroupLadderTree` takes them:
    ``K_0`` is ``I -> n_1``, then each ladder rung contributes the branch to its
    outgroup and the branch onward.

    :return: ``(rates, edges)``, where ``edges`` lists ``(parent, child, rate)``
        with nodes named ``"I"``, ``"n1"``, ``"n2"``, ``"o1"``, ...
    """
    d = DIVERGENCES[:n_out]
    if n_out == 1:
        # No join node: O_1 hangs straight off I, so the single branch IS the
        # whole divergence. This is the case OutgroupLadderTree.from_divergences
        # used to halve.
        return [d[0]], [("I", "o1", d[0])]
    if n_out == 2:
        rates = [K_ROOT, d[0] - K_ROOT, d[1] - K_ROOT]
        edges = [("I", "n1", rates[0]), ("n1", "o1", rates[1]),
                 ("n1", "o2", rates[2])]
        return rates, edges
    rates = [K_ROOT, d[0] - K_ROOT, K_INTERNAL,
             d[1] - K_ROOT - K_INTERNAL, d[2] - K_ROOT - K_INTERNAL]
    edges = [("I", "n1", rates[0]), ("n1", "o1", rates[1]),
             ("n1", "n2", rates[2]), ("n2", "o2", rates[3]),
             ("n2", "o3", rates[4])]
    return rates, edges


def _evolve(state, rate, rng):
    """One JC69 branch: change with probability ``3/4 (1 - e^{-4k/3})``."""
    p_change = 0.75 * (1.0 - np.exp(-4.0 * rate / 3.0))
    changed = rng.random(state.size) < p_change
    offset = rng.integers(1, 4, state.size)  # uniform over the other three
    return np.where(changed, (state + offset) % 4, state)


def simulate(n_out, rng):
    """Draw the fixture.

    :return: ``(counts, outgroups, ancestral, derived_count, derived, hidden)``
        with ``counts`` the per-site allele counts over the ingroup,
        ``outgroups`` the observed outgroup states, and ``hidden`` the internal
        node states keyed by node name.
    """
    ancestral = rng.integers(0, 4, N_SITES)
    freq = 1.0 / np.arange(1, N_INGROUP)
    derived_count = rng.choice(np.arange(1, N_INGROUP), size=N_SITES,
                               p=freq / freq.sum())
    derived = (ancestral + rng.integers(1, 4, N_SITES)) % 4

    counts = np.zeros((N_SITES, 4), dtype=int)
    rows = np.arange(N_SITES)
    counts[rows, ancestral] = N_INGROUP - derived_count
    counts[rows, derived] = derived_count

    _, edges = ladder_rates(n_out)
    node = {"I": ancestral}
    for parent, child, rate in edges:
        node[child] = _evolve(node[parent], rate, rng)

    outgroups = np.stack([node[f"o{k + 1}"] for k in range(n_out)], axis=1)
    hidden = {"ingroup_mrca": ancestral}
    hidden.update({k: v for k, v in node.items() if k.startswith("n")})
    return counts, outgroups, ancestral, derived_count, derived, hidden


def write_inputs(out_dir, n_out, counts, outgroups, ancestral, derived_count,
                 derived, hidden):
    """Write est-sfs's inputs plus the truth this fixture is graded against."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(N_SITES):
        blocks = [",".join(str(c) for c in counts[i])]
        for k in range(n_out):
            one_hot = [0, 0, 0, 0]
            one_hot[outgroups[i, k]] = 1
            blocks.append(",".join(str(c) for c in one_hot))
        lines.append(" ".join(blocks))
    (out_dir / "sites.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "config.txt").write_text(
        f"n_outgroup {n_out}\nmodel 0\nnrandom 10\n")
    (out_dir / "seed.txt").write_text("60754485\n")

    internal = [k for k in hidden if k != "ingroup_mrca"]
    truth = []
    for i in range(N_SITES):
        row = [STATES[ancestral[i]], str(derived_count[i]), STATES[derived[i]]]
        row += [STATES[hidden[k][i]] for k in internal]
        truth.append("\t".join(row))
    (out_dir / "truth.txt").write_text("\n".join(truth) + "\n")
    np.savez(out_dir / "hidden_states.npz", **hidden)


def find_est_sfs():
    """The est-sfs 2.04 binary from the workflow's bench env, if present."""
    root = pathlib.Path(__file__).resolve().parents[3]
    found = sorted(root.glob(".snakemake/conda/*/bin/est-sfs"))
    if not found:
        sys.exit("est-sfs not found; pass --est-sfs PATH")
    return found[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-run", action="store_true",
                    help="write inputs but do not run est-sfs")
    ap.add_argument("--est-sfs", type=pathlib.Path, default=None)
    ap.add_argument("--counts", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    here = pathlib.Path(__file__).parent
    binary = None if args.no_run else (args.est_sfs or find_est_sfs())

    for n_out in args.counts:
        # One stream per count, so adding a count never perturbs another's draw.
        rng = np.random.default_rng(SEED + 100 * n_out)
        out_dir = here / f"n_out_{n_out}"
        write_inputs(out_dir, n_out, *simulate(n_out, rng))
        rates, _ = ladder_rates(n_out)
        print(f"n_out={n_out}: rates={rates} "
              f"divergences={DIVERGENCES[:n_out]} -> {out_dir}")
        if binary is not None:
            subprocess.run([str(binary), "config.txt", "sites.txt", "seed.txt",
                            "sfs.txt", "p_anc.txt"],
                           cwd=out_dir, check=True,
                           stdout=subprocess.DEVNULL)
            print(f"  est-sfs wrote {out_dir/'p_anc.txt'}")


if __name__ == "__main__":
    main()
