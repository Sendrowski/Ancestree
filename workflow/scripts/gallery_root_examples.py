"""Appendix figure (augments the local-tree schematic): one fixed genealogy,
four example tip-allele configurations x = (x_u), and the resulting posterior
over the root state. Holding the tree fixed isolates how the observed allele
pattern alone drives the inferred ancestral state, from a near-certain call for
a low-frequency derived allele to a genuinely ambiguous one at high frequency
or under homoplasy.

Reuses the kernel call and drawing primitives of gallery.py. Outputs
results/reports/gallery_root_examples.pdf.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gallery as g
from ancestree import JC69, Site, TskitLocalTree

# One six-tip genealogy with a balanced 3|3 deepest split, shared across all
# panels (so a clade-aligned split can leave the root genuinely ambiguous).
NWK = ("(((i1:0.02,i2:0.02):0.02,i3:0.04):0.04,"
       "((i4:0.02,i5:0.02):0.02,i6:0.04):0.04):0.0;")


def case(title, tips):
    tree = TskitLocalTree.from_newick(NWK)
    alleles = tuple(sorted(set(tips.values())))
    site = Site(chrom="1", pos=1, alleles=alleles, tip_alleles=dict(tips))
    return dict(title=title, note="", tree=tree, site=site, model=JC69())


CASES = [
    case(r"(a) Singleton derived ($i=1$)",
         {"i1": "C", "i2": "A", "i3": "A", "i4": "A", "i5": "A", "i6": "A"}),
    case(r"(b) Near-fixed derived ($i=5$)",
         {"i1": "A", "i2": "C", "i3": "C", "i4": "C", "i5": "C", "i6": "C"}),
    case(r"(c) Clade-aligned split ($i=3$)",
         {"i1": "C", "i2": "C", "i3": "C", "i4": "A", "i5": "A", "i6": "A"}),
    case(r"(d) Homoplastic ($C$ on two clades)",
         {"i1": "C", "i2": "A", "i3": "A", "i4": "C", "i5": "A", "i6": "A"}),
]


def render(cases, out_pdf: Path):
    fig = plt.figure(figsize=(8.4, 6.4))
    outer = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.18)
    for i, c in enumerate(cases):
        post = g._posterior(c)
        sub = outer[i // 2, i % 2].subgridspec(2, 1, hspace=0.28,
                                               height_ratios=[3, 1])
        ax_tree = fig.add_subplot(sub[0])
        ax_post = fig.add_subplot(sub[1])
        g._draw_tree(ax_tree, c["tree"], c["site"])
        g._draw_posterior(ax_post, post)
        ax_tree.set_title(c["title"], fontsize=10, loc="left", pad=4)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.25)
    print("wrote", out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    render(CASES, Path("results/reports/gallery_root_examples.pdf"))
