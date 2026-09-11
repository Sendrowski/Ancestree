"""Pool the per-chunk TMRCA-distribution shards onto their shared axes."""
import json

import numpy as np

try:  # snakemake execution
    IN_SHARDS = list(snakemake.input.shards)
    OUT_NPZ = snakemake.output.npz
    OUT_META = snakemake.output.meta
except NameError:  # direct execution
    import glob
    IN_SHARDS = sorted(glob.glob("results/data/tmrca_distributions_baseline_chunk*.npz"))
    OUT_NPZ = "results/data/tmrca_distributions_baseline.npz"
    OUT_META = "results/reports/tmrca_distributions_baseline_meta.json"

#: Summed across shards. The axes themselves are fixed in the shard script, so
#: they are carried through unchanged and checked for agreement instead.
SUMMED = ("hist_truth", "hist_point", "hist_draws", "joint_point",
          "joint_draw_mean", "moments_truth", "moments_point",
          "moments_draws", "n_windows", "n_pairs")
SHARED = ("bin_edges", "true_edges", "est_edges")


def main() -> None:
    """Sum every shard's accumulators and write the pooled archive."""
    if not IN_SHARDS:
        raise ValueError("no shards to aggregate")
    total, axes, t_rep, steps, metas = {}, {}, [], [], []
    for path in IN_SHARDS:
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        metas.append(meta)
        for key in SHARED:
            if key in axes:
                if not np.array_equal(axes[key], d[key]):
                    raise ValueError(
                        f"{path} disagrees with the earlier shards on {key}; "
                        "the shards were written by different axis settings "
                        "and cannot be pooled")
            else:
                axes[key] = d[key]
        for key in SUMMED:
            total[key] = total.get(key, 0) + d[key]
        # The shard concatenates its segments' grids, each ascending, so a
        # drop marks a segment boundary. The within-segment spacing is what
        # the figure needs to bin above; the pooled union of near-identical
        # grids has a spacing near zero and says nothing about either.
        raw = np.asarray(d["t_rep"], float)
        for grid in np.split(raw, np.flatnonzero(np.diff(raw) < 0) + 1):
            if grid.size > 1:
                steps.append(float(np.median(np.diff(np.log10(grid)))))
        t_rep.append(np.unique(raw))

    # n_pairs is a property of the panel, not a count to add up.
    n_pairs = {int(m["n_hap"]) for m in metas}
    if len(n_pairs) != 1:
        raise ValueError(f"shards disagree on the panel size: {sorted(n_pairs)}")
    total["n_pairs"] = total["n_pairs"] // len(IN_SHARDS)

    for field in ("mu", "rec_rate", "window", "n_members", "seed"):
        values = {json.dumps(m[field]) for m in metas}
        if len(values) != 1:
            raise ValueError(
                f"shards disagree on {field}: {sorted(values)}; pooling them "
                "would mix runs of different configurations")

    np.savez_compressed(OUT_NPZ, t_rep=np.unique(np.concatenate(t_rep)),
                        grid_step=float(np.median(steps)), **axes, **total)
    summary = {
        **{k: metas[0][k] for k in ("mu", "rec_rate", "window", "n_members",
                                    "seed", "n_hap")},
        "n_shards": len(IN_SHARDS),
        "chunks": [m["trees"] for m in metas],
        "sequence_length_total": sum(m["sequence_length"] for m in metas),
        "n_windows": int(total["n_windows"]),
        "n_pairs": int(total["n_pairs"]),
        "n_window_pair_values": int(total["n_windows"]) * int(total["n_pairs"]),
        "n_time_grids": len(steps),
        "grid_step_log10": float(np.median(steps)),
    }
    with open(OUT_META, "w") as handle:
        json.dump(summary, handle, indent=1)
    print(f"wrote {OUT_NPZ} and {OUT_META}: {summary['n_shards']} chunks, "
          f"{summary['sequence_length_total'] / 1e6:.0f} Mb, "
          f"{summary['n_window_pair_values']:,} window-pair values")


main()
