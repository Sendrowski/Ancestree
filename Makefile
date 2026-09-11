# Top-level project Makefile. Standard targets:
#   make figures        regenerate all benchmark reports + manuscript figures (snakemake)
#   make manuscript     build the manuscript PDF (tectonic)
#   make benchmark      regenerate reports via snakemake, then assert them
#   make test           fast unit tests only
#   make test-all       fast unit tests + slow benchmark assertions
#   make test-full      every test (slow + not), incl. zarr / SLiM / numba paths
#   make docs           build HTML docs (delegates to docs/Makefile)
#
# The data->figures pipeline lives in workflow/Snakefile (file-dependency DAG,
# conda envs, -jN parallelism). These targets are thin front-doors over it,
# pytest, sphinx, and tectonic. Never hand-render a benchmark figure: regenerate
# it through `make figures` so the cells->figures edge stays tracked.

CORES ?= 1

# Test targets run in the project env, which carries msprime / zarr / SLiM /
# numba / cyvcf2 / newick. A bare `pytest` resolves against whichever env is
# active, and from `base` the suite dies in conftest on the msprime import.
# Override to use the active interpreter: `make test PYTEST="python -m pytest"`.
CONDA_ENV ?= dev-ancestree
DOCS_ENV ?= docs-ancestree

# On a SLURM cluster the suite is dispatched to a compute node: the GenomeDK
# login node is shared, and `-n auto` there claims all 16 of its cores. Detected
# by srun being on PATH while no allocation is active. Inside an existing job
# (SLURM_JOB_ID set) and on a workstation the suite runs in place. Account and
# partition mirror workflow/profiles/slurm/config.yaml.
SLURM_ACCOUNT ?= primates_fast_dfe
SLURM_PARTITION ?= short
TEST_CPUS ?= 16
TEST_MEM ?= 48G
TEST_TIME ?= 2:00:00
ifeq ($(origin SLURM_JOB_ID),undefined)
SRUN := $(shell command -v srun 2>/dev/null)
endif

# One worker per core. Override with `make test XDIST=` for a serial debug run,
# or XDIST="-n 4" to leave headroom on a shared machine. Under srun the count is
# pinned to the allocation: `auto` counts the compute node's full core list and
# would oversubscribe what SLURM actually granted.
ifdef SRUN
SRUN_PREFIX := $(SRUN) --account=$(SLURM_ACCOUNT) --partition=$(SLURM_PARTITION) \
  --cpus-per-task=$(TEST_CPUS) --mem=$(TEST_MEM) --time=$(TEST_TIME)
XDIST ?= -n $(TEST_CPUS)
else
XDIST ?= -n auto
endif

PYTEST ?= $(SRUN_PREFIX) conda run --no-capture-output -n $(CONDA_ENV) python -m pytest $(XDIST)
# Reads its settings from [tool.mypy] in pyproject.toml, including which
# modules are quarantined, so the target takes no flags.
MYPY ?= conda run --no-capture-output -n $(CONDA_ENV) mypy

.PHONY: help figures manuscript manuscript-diff benchmark test test-all test-full typecheck docs

help:
	@echo "Targets:"
	@echo "  make figures          # snakemake -j$(CORES): regenerate reports + manuscript figures"
	@echo "  make manuscript       # build the manuscript PDF (tectonic)"
	@echo "  make manuscript-diff  # latexdiff vs the ensemble baseline"
	@echo "  make benchmark        # figures, then pytest -m slow"
	@echo "  make test             # fast unit tests (via srun where SLURM is present)"
	@echo "  make test-all         # unit + slow benchmark assertions"
	@echo "  make test-full        # entire suite, zero skips (needs dev-ancestree env)"
	@echo "  make typecheck        # mypy over the published annotations"
	@echo "  make docs             # build HTML docs"

# Every figure and table the manuscript includes, through the snakemake DAG.
# Named target: a bare `snakemake` runs the first rule, `manuscript`, which
# only re-runs tectonic. --use-conda / --rerun-triggers come from profiles/default.
figures:
	snakemake -j $(CORES) manuscript_figures

# The manuscript PDF via the `manuscript` rule (tectonic). --allowed-rules
# confines the run to that rule, so a checkout lacking the cluster artifacts
# does not plan the whole pipeline. --scheduler greedy: pulp's CBC is x86-only.
manuscript:
	snakemake -j $(CORES) --scheduler greedy --allowed-rules manuscript -- manuscript

# The revision diff against reports/manuscripts/latex_baseline/, the manuscript
# before the focal node and ensemble sampling. Same --allowed-rules confinement
# as above, widened to the two diff rules.
manuscript-diff:
	snakemake -j $(CORES) --scheduler greedy \
	  --allowed-rules manuscript manuscript_diff_tex manuscript_diff \
	  -- reports/manuscripts/latex_diff/main.pdf

# Regenerate all benchmark reports, then run the pytest assertions over them.
# The baseline rules build their own envs from workflow/envs/polarbear.yml and
# workflow/envs/fastdfe.yml under snakemake --use-conda.
benchmark: figures
	$(PYTEST) -m slow testing/test_benchmarks.py

test:
	$(PYTEST)

typecheck:
	$(MYPY)

test-all:
	$(PYTEST)
	$(PYTEST) -m slow testing/test_benchmarks.py

# Whole suite with no dependency skips. The report-gated benchmark tests still
# need `make benchmark` to have run first.
test-full:
	$(PYTEST) -m "slow or not slow"

# Sphinx and the theme live in their own env. `clean html` rather than `html`:
# an incremental build reuses stale autodoc entries.
docs:
	conda run --no-capture-output -n $(DOCS_ENV) $(MAKE) -C docs clean html
