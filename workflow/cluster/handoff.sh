#!/usr/bin/env bash
# Cluster handoff for the robustness panel-ARG benchmark (SINGER + Relate +
# tsinfer). Inference AND scoring run on the cluster (Linux); only the ~KB
# per-cell summaries come back. The inferred ARGs are persisted (tszip) on the
# cluster, so any later scoring change re-scores there without re-inference.
#
# Usage:
#   ./handoff.sh push-code      # rsync the repo (code only) up
#   ./handoff.sh push-data      # rsync the ~1 GB ground-truth sims up
#   ./handoff.sh clean          # remove stale panel intermediates on the cluster
#   ./handoff.sh pull-results   # rsync the per-cell summaries back
#
# Then, ON THE CLUSTER (see README.md), build SINGER/Relate once and run:
#   snakemake --use-conda --scheduler greedy -j <N> --rerun-triggers=mtime \
#       $(cat workflow/cluster/cell_targets.txt)
set -euo pipefail

# ----------------------------- EDIT THESE -----------------------------
CLUSTER="${CLUSTER:-user@login.cluster.edu}"   # ssh destination
REMOTE="${REMOTE:-~/Ancestree}"                # repo path on the cluster
# ----------------------------------------------------------------------

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

case "${1:-}" in
  push-code)
    # Code + workflow only; never the local results/ or conda cache.
    rsync -avz --relative \
      --exclude '__pycache__' \
      ./ancestree ./workflow ./pyproject.toml ./README* \
      "$CLUSTER:$REMOTE/" 2>/dev/null || \
    rsync -avz \
      --exclude '.git' --exclude 'results' --exclude '.snakemake' \
      --exclude '__pycache__' \
      ./ "$CLUSTER:$REMOTE/"
    ;;
  push-data)
    # Ground-truth sims only (manifest); rsync -z compresses in transit.
    rsync -avz --files-from=workflow/cluster/ground_truth_manifest.txt \
      ./ "$CLUSTER:$REMOTE/"
    ;;
  clean)
    # Drop any stale panel intermediates so the run rebuilds with current code
    # (mtime triggers do not detect code changes).
    ssh "$CLUSTER" "cd $REMOTE && find results/data -maxdepth 1 \\( \
        -name 'robustness_arg_*' -o -name 'robustness_*_shard_*' \
        -o -name 'robustness_chunk_*singer*' -o -name 'robustness_chunk_*relate*' \
        -o -name 'robustness_chunk_*tsinfer*' \
        -o -name 'robustness_cell_*singer*_summary.json' \
        -o -name 'robustness_cell_*relate*_summary.json' \
        -o -name 'robustness_cell_*tsinfer*_summary.json' \
        -o -name 'robustness_cell_intersect_*' \
        -o -name 'intersectmask_*' -o -name 'ftmap_*' -o -name 'singerseg_*' \\) \
        -exec rm -rf {} + 2>/dev/null; echo cleaned"
    ;;
  pull-results)
    mkdir -p results/data
    for tool in singer relate tsinfer; do
      rsync -avz \
        "$CLUSTER:$REMOTE/results/data/robustness_cell_*_${tool}_*_summary.json" \
        results/data/ 2>/dev/null || true
    done
    ;;
  *)
    echo "usage: $0 {push-code|push-data|clean|pull-results}" >&2
    exit 1
    ;;
esac
