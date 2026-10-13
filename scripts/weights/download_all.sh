#!/usr/bin/env bash
#
# download_all.sh -- stage every model weight the GRPO pipeline needs, into data/weights/.
#
# By default it fetches the OPEN weights the DEFAULT pipeline needs to run:
#   - proteinmpnn : verify the in-repo vendored weights (no download)
#   - boltz2      : download Boltz-2 (default structure backend, ~6 GB)
#   - rfd3        : install RFD3 checkpoints via foundry (backbone sampler)
# and always prints the Protenix v2 (proprietary, licensed-only) placement instructions.
#
# Optional extras (off by default): --with-protenix-v1 (alternate open backend), --with-esm2.
#
# Everything lands under ${AAPG_ROOT:-<repo>}/data/weights/<model>/ (git-ignored). PyRosetta (the
# interface scorer) is a separately-licensed PACKAGE, not weights -- a non-PyPI wheel from the
# RosettaCommons channel, installed out-of-band (NOT pulled by --extra tools on python>=3.11).
#
# Usage:
#   scripts/weights/download_all.sh [--dry-run] [--with-protenix-v1] [--with-esm2]
#
# shellcheck source=scripts/weights/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_PROTENIX_V1=0
WITH_ESM2=0
PASS=()
for arg in "$@"; do
  case "${arg}" in
    --with-protenix-v1) WITH_PROTENIX_V1=1 ;;
    --with-esm2)        WITH_ESM2=1 ;;
    --dry-run|-n)       PASS+=("--dry-run") ;;
    -h|--help)          grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument '${arg}' (see --help)" ;;
  esac
done

log "weights root: ${WEIGHTS_DIR}"
log "=== ProteinMPNN (vendored, verify) ==="; bash "${HERE}/proteinmpnn.sh" "${PASS[@]}"
log "=== Boltz-2 (default structure backend) ==="; bash "${HERE}/boltz2.sh" "${PASS[@]}"
log "=== RFdiffusion3 (sampler) ==="; bash "${HERE}/rfd3.sh" "${PASS[@]}"

if [[ "${WITH_PROTENIX_V1}" -eq 1 ]]; then
  log "=== Protenix v1 (optional alt backend) ==="; bash "${HERE}/protenix_v1.sh" "${PASS[@]}"
fi
if [[ "${WITH_ESM2}" -eq 1 ]]; then
  log "=== ESM-2 (optional) ==="; bash "${HERE}/esm2.sh" "${PASS[@]}"
fi

log "=== Protenix v2 (proprietary; instructions only) ==="; bash "${HERE}/protenix_v2.sh" "${PASS[@]}"

cat <<EOF

[weights] Done. For a real GRPO run you also need (not weights, installed separately):
  - PyRosetta (interface scorer)  : licensed; a non-PyPI wheel from the RosettaCommons channel.
                                     On python>=3.11 it is NOT pulled by 'uv sync --extra tools'
                                     (its marker is never satisfied); install it out-of-band.
  - the heavy model stacks (torch, boltz, foundry/rfd3) : installed by their scripts above or the
                                     per-model containers under containers/.
The mock pipeline needs none of these: 'uv run aapg-grpo steering.mock=true ...' runs on CPU.
EOF
