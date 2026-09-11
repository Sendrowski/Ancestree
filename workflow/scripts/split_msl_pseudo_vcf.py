"""B13b stage 1a: split the 208 MB est-sfs MSL pseudo-VCF into N
positional chunks (one .vcf.gz per chunk). No filtering at this stage —
the chunk files contain every line in their position range, including
ingroup-monomorphic ones. Filtering happens in the next rule
(``extract_msl_polymorphic_chunk``).

Trades a one-time ~25 min streaming pass for cached per-chunk VCFs that
downstream steps can read independently and in parallel.

Output schema: one ``.vcf.gz`` per chunk, format matches the input
(no CHROM column; ``POS REF ALT [ingroup GTs] [outgroup GTs]``).
"""
import bisect
import gzip
import json
import time
from pathlib import Path


try:
    in_simplify_vcf = snakemake.input.simplify_vcf  # type: ignore[name-defined]
    out_chunks = list(snakemake.output.chunks)  # type: ignore[name-defined]
    out_manifest = snakemake.output.manifest  # type: ignore[name-defined]
    chunks: list = list(snakemake.params.chunks)  # type: ignore[name-defined]
except NameError:
    in_simplify_vcf = (
        "external/polarbear/data/real_data/ancestral_state/est_sfs/"
        "chr1.snp_only.collapsed.bed_filtered.MSL.cleaned.ponAbe2_macFas5.simplify.vcf.gz"
    )
    chr1_len = 248_956_422
    step = 6_250_000
    chunks = []
    for cs in range(0, chr1_len, step):
        ce = min(cs + step, chr1_len)
        def _fmt(bp: int) -> str:
            if bp == 0:
                return "0"
            return f"{bp // 1_000_000}M" if bp % 1_000_000 == 0 else str(bp)
        region = f"chr1_{_fmt(cs)}_{_fmt(ce)}"
        chunks.append((region, cs, ce))
    out_chunks = [f"results/data/msl_pseudo_vcf_{r}.vcf.gz" for r, _, _ in chunks]
    out_manifest = "results/data/msl_pseudo_vcf_chunks_manifest.json"


# Sort chunks by start position and prepare bisect indices.
chunks = sorted(chunks, key=lambda c: c[1])
chunk_starts = [c[1] for c in chunks]
chunk_ends = [c[2] for c in chunks]
chunk_regions = [c[0] for c in chunks]
global_end = chunk_ends[-1]

# Map region → output path (snakemake gives them in the same order we declared).
region_to_out = {}
for path in out_chunks:
    stem = Path(path).name  # msl_pseudo_vcf_chr1_0_6250000.vcf.gz
    # Strip prefix + suffix
    region = stem.replace("msl_pseudo_vcf_", "").replace(".vcf.gz", "")
    region_to_out[region] = path

print(
    f"split_msl_pseudo_vcf: {len(chunks)} chunks, "
    f"span {chunk_starts[0]}..{global_end:,}",
    flush=True,
)


# Open all chunk output files for writing. Keep them open through the stream.
Path(out_chunks[0]).parent.mkdir(parents=True, exist_ok=True)
chunk_files = {
    region: gzip.open(region_to_out[region], "wt")
    for region in chunk_regions
}
chunk_line_counts = {region: 0 for region in chunk_regions}


n_lines_seen = 0
t0 = time.perf_counter()
report_every = 10_000_000

try:
    with gzip.open(in_simplify_vcf, "rt") as fin:
        for line in fin:
            n_lines_seen += 1
            if n_lines_seen % report_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  [stream] {n_lines_seen:>11,} lines  ({elapsed:>5.1f}s)",
                    flush=True,
                )
            # Quick parse: only need POS (first whitespace-separated field).
            sp = line.find(" ")
            if sp == -1:
                sp = line.find("\t")
                if sp == -1:
                    continue
            try:
                pos = int(line[:sp])
            except ValueError:
                continue
            if pos >= global_end:
                # File is position-sorted, can stop here.
                break
            idx = bisect.bisect_right(chunk_starts, pos) - 1
            if idx < 0 or pos >= chunk_ends[idx]:
                continue
            region = chunk_regions[idx]
            chunk_files[region].write(line)
            chunk_line_counts[region] += 1
finally:
    for f in chunk_files.values():
        f.close()


t_pass = time.perf_counter() - t0
print(
    f"split_msl_pseudo_vcf: streamed {n_lines_seen:,} lines in {t_pass:.1f}s",
    flush=True,
)
for region in chunk_regions:
    sz = Path(region_to_out[region]).stat().st_size / 1024 / 1024
    print(
        f"  {region}: {chunk_line_counts[region]:>9,} lines  ({sz:>5.1f} MB gzipped)",
        flush=True,
    )

manifest = {
    "n_chunks": len(chunks),
    "n_lines_seen_total": n_lines_seen,
    "elapsed_seconds": t_pass,
    "chunks": [
        {
            "region": region,
            "start": int(start),
            "end": int(end),
            "n_lines": int(chunk_line_counts[region]),
            "path": region_to_out[region],
        }
        for (region, start, end) in chunks
    ],
}
Path(out_manifest).parent.mkdir(parents=True, exist_ok=True)
with open(out_manifest, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"split_msl_pseudo_vcf: wrote {out_manifest}", flush=True)
