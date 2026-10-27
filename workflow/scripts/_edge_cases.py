"""The four hand-built site classes of the appendix edge-case table.

The ingroup is a balanced quartet ``((i0,i1),(i2,i3))`` except in the polytomy
class, which is a 4-way star. Outgroups ``o1``..``o3`` attach as a nested
ladder, ``o1`` closest. Branch lengths are in substitutions per site and short,
so the tree is unsaturated and the deeper outgroups stay informative.
"""
import numpy as np

#: Ingroup tip branch.
IT = 0.015
#: Ingroup cherry to ingroup MRCA.
IC = 0.015
#: One ladder step between successive outgroup joins.
RUNG = 0.01
#: Depth of the ingroup MRCA, also the star's tip branch.
S = IT + IC

#: The four classes. ``ingroup`` and ``outgroup`` map tip names to alleles.
CLASSES = [
    {"name": "Homoplastic biallelic", "star": False, "polymorphic": True,
     "ingroup": {"i0": "A", "i1": "G", "i2": "A", "i3": "G"},
     "outgroup": {"o1": "A", "o2": "A", "o3": "A"}},
    {"name": "Tri-allelic", "star": False, "polymorphic": True,
     "ingroup": {"i0": "A", "i1": "A", "i2": "C", "i3": "G"},
     "outgroup": {"o1": "T", "o2": "T", "o3": "T"}},
    {"name": "Ingroup-monomorphic", "star": False, "polymorphic": False,
     "ingroup": {"i0": "A", "i1": "A", "i2": "A", "i3": "A"},
     "outgroup": {"o1": "G", "o2": "G", "o3": "G"}},
    {"name": "4-way polytomy", "star": True, "polymorphic": True,
     "ingroup": {"i0": "A", "i1": "A", "i2": "A", "i3": "G"},
     "outgroup": {"o1": "A", "o2": "A", "o3": "A"}},
]


def ladder(ingroup_block: str) -> str:
    """Attach ``o1``..``o3`` as a nested ladder above ``ingroup_block``, whose
    MRCA sits at depth :data:`S`. Ultrametric, every leaf at ``S + 3 RUNG``."""
    return (f"((({ingroup_block}:{RUNG},o1:{S+RUNG}):{RUNG},"
            f"o2:{S+2*RUNG}):{RUNG},o3:{S+3*RUNG});")


def newick(star: bool, outgroups: bool) -> str:
    """Newick of one class's tree.

    :param star: Whether the ingroup is the 4-way star rather than the quartet.
    :param outgroups: Whether to attach the outgroup ladder.
    :return: The Newick string.
    """
    block = (f"(i0:{S},i1:{S},i2:{S},i3:{S})" if star else
             f"((i0:{IT},i1:{IT}):{IC},(i2:{IT},i3:{IT}):{IC})")
    return ladder(block) if outgroups else block + ";"


def simulate_quartet_ladder(n_sites: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Site patterns drawn under JC69 on the quartet-with-ladder tree.

    :param n_sites: Number of independent sites.
    :param seed: Seed of ``numpy.random.default_rng``.
    :return: ``(ingroup, outgroup)`` state indices into ``ACGT``, of shapes
        ``(n_sites, 4)`` for ``i0..i3`` and ``(n_sites, 3)`` for ``o1..o3``.
    """
    rng = np.random.default_rng(seed)

    def evolve(parent: np.ndarray, t: float) -> np.ndarray:
        # JC69: with probability 1 - exp(-4t/3) the state is redrawn uniformly.
        redraw = rng.random(parent.size) < 1.0 - np.exp(-4.0 * t / 3.0)
        return np.where(redraw, rng.integers(0, 4, parent.size), parent)

    root = rng.integers(0, 4, n_sites)
    o3 = evolve(root, S + 3 * RUNG)
    join2 = evolve(root, RUNG)
    o2 = evolve(join2, S + 2 * RUNG)
    join1 = evolve(join2, RUNG)
    o1 = evolve(join1, S + RUNG)
    mrca = evolve(join1, RUNG)
    cherries = [evolve(mrca, IC) for _ in range(2)]
    ingroup = [evolve(cherries[k // 2], IT) for k in range(4)]
    return np.stack(ingroup, axis=1), np.stack([o1, o2, o3], axis=1)
