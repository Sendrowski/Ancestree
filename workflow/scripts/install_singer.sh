#!/usr/bin/env bash
# Build SINGER from source and install the binary + helper scripts.
#
# NOTE: the resulting arm64 build runs on tiny inputs but segfaults on standard
# panels (SINGER source bugs in Node::move_iterator / write_state; see
# _singer.py). Intended for a Linux build or a future retry, not the active
# benchmark.
#
# SINGER publishes only Linux x86-64 release binaries, so on Apple Silicon (and
# for reproducibility everywhere) it is compiled from source with the C++17
# compiler under workflow/envs/singer.yml. The official beta_mac_M1_compile.sh
# uses a single `clang++ -std=c++17 -O3 *.cpp`. The same is done here with
# ${CXX} so it works on macOS (clang) and Linux (gcc) alike.
#
# Usage: install_singer.sh <install_dir> [version]
#   <install_dir>  where `singer`, `singer_master`, `convert_to_tskit` land
#   [version]      git tag/branch of popgenmethods/SINGER (default: main)
set -euo pipefail

DEST="${1:?usage: install_singer.sh <install_dir> [version]}"
VERSION="${2:-v0.1.8-beta}"
REPO="https://github.com/popgenmethods/SINGER.git"
CXX_BIN="${CXX:-clang++}"; command -v "${CXX_BIN}" >/dev/null 2>&1 || CXX_BIN="g++"

DEST="$(mkdir -p "${DEST}" && cd "${DEST}" && pwd)"
if [ -x "${DEST}/singer" ]; then
    echo "SINGER already installed at ${DEST}/singer"; exit 0
fi

WORK="$(mktemp -d)"; trap 'rm -rf "${WORK}"' EXIT
git clone --depth 1 --branch "${VERSION}" "${REPO}" "${WORK}/SINGER"
SRC="${WORK}/SINGER/SINGER/SINGER"

echo "Compiling SINGER with ${CXX_BIN} (C++17, -O3) ..."
( cd "${SRC}" && "${CXX_BIN}" -std=c++17 -O3 ./*.cpp -o "${DEST}/singer" )
cp "${SRC}/singer_master" "${SRC}/convert_to_tskit" "${DEST}/"
chmod +x "${DEST}/singer" "${DEST}/singer_master" "${DEST}/convert_to_tskit"

echo "Installed to ${DEST}:"
"${DEST}/singer" 2>&1 | head -1 || true   # prints its arg banner
echo "  $(file "${DEST}/singer")"
