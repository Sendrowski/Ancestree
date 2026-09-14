# Ancestree  <img align="right" width="100" src="https://raw.githubusercontent.com/Sendrowski/Ancestree/main/docs/logo_thumbnail.png">
[![codecov](https://codecov.io/gh/Sendrowski/Ancestree/branch/main/graph/badge.svg)](https://codecov.io/gh/Sendrowski/Ancestree)
[![Documentation Status](https://readthedocs.org/projects/ancestree/badge/?version=latest)](https://ancestree.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/ancestree-popgen.svg)](https://badge.fury.io/py/ancestree-popgen)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ancestree.svg)](https://anaconda.org/conda-forge/ancestree)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Downloads](https://static.pepy.tech/badge/ancestree-popgen)](https://pepy.tech/project/ancestree-popgen)
[![DOI](https://img.shields.io/badge/DOI-10.64898/2026.09.11.750934-blue)](https://doi.org/10.64898/2026.09.11.750934)

``ancestree`` is a package for likelihood-based, per-site ancestral-allele annotation, unifying outgroup-based (EST-SFS-style) and ARG-based (PolarBEAR-style) inference behind a single likelihood kernel. It runs in three modes: fixed-tree, ARG, and local-tree, the last inferring its own local genealogies from the variant data rather than taking a supplied ARG. Every site receives a full posterior over the four nucleotide states, with multi-allelic sites, missing data and recurrent mutation handled natively.

Please see the [documentation](https://ancestree.readthedocs.io/en/latest/) for all the details.
