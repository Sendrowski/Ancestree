#!/usr/bin/env bash
# Install Relate (Speidel et al. 2019) by fetching the official prebuilt release
# for the current platform. Relate ships native binaries for every platform ---
# including Apple Silicon (relate_v1.2.4_MacOSX_M) --- so no compilation is
# needed and the robustness Relate cells run on arm64 (unlike SINGER).
#
# Lands bin/{Relate,RelateFileFormats,...} + scripts/ under <install_dir>, which
# is what _relate.relate_dir() expects RELATE_DIR to point at.
#
# Usage: install_relate.sh <install_dir> [version]
#   <install_dir>  where bin/ + scripts/ land (RELATE_DIR)
#   [version]      Relate release version (default: v1.2.4)
set -euo pipefail

DEST="${1:?usage: install_relate.sh <install_dir> [version]}"
VERSION="${2:-v1.2.4}"

# Map (kernel, machine) -> the release-tarball platform suffix.
KERNEL="$(uname -s)"; MACH="$(uname -m)"
case "${KERNEL}:${MACH}" in
    Darwin:arm64)          PLAT="MacOSX_M" ;;
    Darwin:x86_64)         PLAT="MacOSX_Intel" ;;
    Linux:x86_64|Linux:amd64) PLAT="x86_64_static" ;;   # static = cluster-portable
    Linux:aarch64|Linux:arm64) echo "No prebuilt Relate for Linux arm64; build from source." >&2; exit 1 ;;
    *) echo "Unsupported platform ${KERNEL}:${MACH}" >&2; exit 1 ;;
esac

REL="relate_${VERSION}_${PLAT}"
URL="https://myersgroup.github.io/relate/download/${REL}.tgz"

DEST="$(mkdir -p "${DEST}" && cd "${DEST}" && pwd)"
if [ -x "${DEST}/bin/Relate" ]; then
    echo "Relate already installed at ${DEST}/bin/Relate"; exit 0
fi

WORK="$(mktemp -d)"; trap 'rm -rf "${WORK}"' EXIT
echo "Downloading ${URL} ..."
curl -fsSL --max-time 300 "${URL}" -o "${WORK}/relate.tgz"
tar -xzf "${WORK}/relate.tgz" -C "${WORK}"
cp -R "${WORK}/${REL}/bin" "${WORK}/${REL}/scripts" "${DEST}/"
chmod +x "${DEST}/bin/"* 2>/dev/null || true

echo "Installed Relate ${VERSION} (${PLAT}) to ${DEST}:"
"${DEST}/bin/Relate" 2>&1 | head -1 || true
echo "  $(file "${DEST}/bin/Relate")"
