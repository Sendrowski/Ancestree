# Cluster run: robustness panel-ARG benchmark

SINGER is arm64-blocked, so the panel-ARG rows (SINGER, Relate, tsinfer) are
produced on a Linux cluster. Inference and scoring both run there. Only the ~KB
per-cell summaries come back and feed the local figure scripts.

The workflow separates inference from scoring: each method persists its inferred
ARG (tszip `draw_*.tsz`), then a method-agnostic scoring step lays the panel's
full site set onto those trees. So a later scoring change re-scores off the
saved ARGs without re-running the (expensive) inference.

## What moves where

| step | where | size |
|---|---|---|
| ground-truth sims → cluster | `push-data` | ~1 GiB (`-z` in transit) |
| code → cluster | `push-code` | small |
| inference + scoring | cluster | ARGs persist there (~7 GiB, mostly SINGER) |
| per-cell summaries → local | `pull-results` | ~100 × few KB |

## Steps

Edit `CLUSTER` / `REMOTE` at the top of `handoff.sh` (or pass them as env vars).

```bash
cd workflow/cluster
CLUSTER=me@login.cluster REMOTE='~/Ancestree' ./handoff.sh push-code
CLUSTER=me@login.cluster REMOTE='~/Ancestree' ./handoff.sh push-data
```

On the cluster, from `$REMOTE`, build the tools once and run:

```bash
# One-time tool installs (Linux x86 for SINGER; Relate provides native binaries):
bash workflow/scripts/install_singer.sh ~/opt/singer
bash workflow/scripts/install_relate.sh ~/opt/relate
export SINGER_DIR=~/opt/singer RELATE_DIR=~/opt/relate

# Run every panel-ARG cell (infer → persist ARG → score → chunk → cell).
# --use-conda builds envs/bench.yml + envs/tsinfer.yml on first run.
snakemake --use-conda --scheduler greedy -j "$SLURM_CPUS" --rerun-triggers=mtime \
    $(cat workflow/cluster/cell_targets.txt)
```

`SINGER_DIR` / `RELATE_DIR` are also passed as rule params, so a batch-scheduler
submit environment that does not inherit the shell env still finds them.

Back on the laptop:

```bash
cd workflow/cluster
CLUSTER=me@login.cluster REMOTE='~/Ancestree' ./handoff.sh pull-results
```

then regenerate the figures locally.

## Re-scoring later (no re-inference)

The ARGs persist on the cluster under `results/data/robustness_arg_*`. After a
scoring change, delete only the `robustness_*_shard_*` / `robustness_chunk_*` /
`robustness_cell_*` outputs (not the `robustness_arg_*` dirs) and re-run the same
targets, and snakemake replays the cheap scoring step off the saved trees.
