"""The IQ-TREE marginal reconstruction at the (W,X) node, for the docs.

IQ-TREE labels its internal nodes ``Node1``, ``Node2``, ... in the tree file
written beside the ``.state`` table. This script finds the label of the node
whose leaves are ``W`` and ``X``, keeps that node's rows of the table with its
comment header, and writes them gzip-compressed together with a copy of the
tree file. Rule ``docs_alignment_iqtree_state`` runs it, which also runs
directly::

    python workflow/scripts/extract_docs_iqtree_state.py
"""
import gzip
import shutil

import newick

try:
    in_state = snakemake.input.state  # type: ignore[name-defined]
    in_tree = snakemake.input.treefile  # type: ignore[name-defined]
    out_state = snakemake.output.state  # type: ignore[name-defined]
    out_tree = snakemake.output.treefile  # type: ignore[name-defined]
    clade = set(snakemake.params.clade)  # type: ignore[name-defined]
except NameError:
    in_state = "results/docs/iqtree/alignment.state"
    in_tree = "results/docs/iqtree/alignment.treefile"
    out_state = "docs/_static/alignment.state.gz"
    out_tree = "docs/_static/alignment.treefile"
    clade = {"W", "X"}

with open(in_tree, encoding="utf-8") as f:
    root = newick.loads(f.read().strip())[0]
label = next(node.name for node in root.walk()
             if {leaf.name for leaf in node.get_leaves()} == clade)

with open(in_state, encoding="utf-8") as src, \
        gzip.open(out_state, "wt", encoding="utf-8") as out:
    for line in src:
        if line.startswith(("#", "Node\t")) or line.split("\t", 1)[0] == label:
            out.write(line)
shutil.copyfile(in_tree, out_tree)
print(f"Kept the rows of {label}, the MRCA of {sorted(clade)}")
