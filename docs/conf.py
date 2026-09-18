"""Sphinx configuration for Ancestree's docs."""
from __future__ import annotations

import sys
from pathlib import Path

# Make ancestree importable for autodoc without needing an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# -- Project information ------------------------------------------------------

project = "Ancestree"
author = "Janek Sendrowski"

# The version is read from the package itself, so the two cannot diverge.
from ancestree import __version__ as _ancestree_version  # noqa: E402
release = _ancestree_version
html_show_copyright = False

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "autodocsumm",  # per-class method-summary table at the top of each class
    "myst_nb",  # render committed .ipynb notebooks
    "sphinx_paramlinks",  # anchors on :param: entries, so kwargs are linkable
    "sphinxarg.ext",  # CLI flag tables generated from cli.build_parser
]

# Page-level ``.. autosummary::`` blocks render an inline class table linking
# to the autoclass docs on the same page; no stub pages need generating.
rst_prolog = """
.. |tsinfer| raw:: html

   <a class="reference external" href="https://tskit.dev/tsinfer/docs/stable/"><code class="docutils literal notranslate"><span class="pre">tsinfer</span></code></a>

.. |Relate| raw:: html

   <a class="reference external" href="https://myersgroup.github.io/relate/"><code class="docutils literal notranslate"><span class="pre">Relate</span></code></a>

.. |cyvcf2| raw:: html

   <a class="reference external" href="https://brentp.github.io/cyvcf2/"><code class="docutils literal notranslate"><span class="pre">cyvcf2</span></code></a>
"""

autosummary_generate = False

# MyST-NB renders the committed notebook outputs. Notebooks are executed
# through the Snakemake refresh rule, not by the docs build.
nb_execution_mode = "off"
# Merge consecutive stream outputs (stdout/stderr) into one render block so
# multi-line log emission (e.g. FixedTreeInference fit summary) shows as one
# compact block rather than four stacked frames.
nb_merge_streams = True
myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "colon_fence"]

# Render type hints from the function signature (not duplicated in docstrings).
typehints_use_signature = True
typehints_use_signature_return = True
typehints_fully_qualified = False
always_document_param_types = True

# Objects are rendered without their module prefix.
add_module_names = False

# Autodoc defaults: include everything by default so module pages need minimal
# boilerplate. Individual .rst pages can override per-class.
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "special-members": "__iter__,__len__,__getitem__,__contains__",
    "private-members": False,
    "undoc-members": False,
    # autodocsumm: prepend a compact method summary table (names + one-line
    # descriptions) to each class, before the full member docs. Limit it to
    # the Methods section, since the Attributes summary duplicates the
    # per-attribute docs below it.
    "autosummary": True,
    "autosummary-sections": "Methods",
}

# Intersphinx mappings so cross-references resolve (numpy, scipy, tskit).
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "tskit": ("https://tskit.dev/tskit/docs/stable", None),
    "msprime": ("https://tskit.dev/msprime/docs/stable", None),
    "tsinfer": ("https://tskit.dev/tsinfer/docs/stable", None),
}

exclude_patterns = ["_build", "outputs", "Thumbs.db", ".DS_Store"]

# -- HTML output --------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = f"Ancestree {release}"
html_logo = "logo.svg"
html_favicon = "favicon.ico"
html_css_files = ["custom.css"]

html_theme_options = {
    # sphinx-book-theme puts the search in the primary sidebar and clears this in its
    # theme.conf, but pydata only honours that when the key is set here, so it re-adds a
    # second search field to the header. Clear it explicitly.
    'navbar_persistent': [],
    "search_bar_text": "Search…",
    "repository_url": "https://github.com/Sendrowski/Ancestree",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": False,
    "use_download_button": False,
    "show_navbar_depth": 2,
}


def _drop_argparse_usage(app, doctree):
    """Remove the usage lines the argparse directive places above each flag table."""
    from docutils import nodes

    for node in list(doctree.findall(nodes.literal_block)):
        if node.astext().startswith("usage: ancestree"):
            node.parent.remove(node)


def setup(app):
    app.connect("doctree-read", _drop_argparse_usage)
