#!/usr/bin/env bash
#
# protenix_v1.sh -- install Protenix v1 (Apache-2.0, open) and stage its data under
# data/weights/protenix_v1/.
#
# Protenix v1 is an OPTIONAL alternate structure backend (the default is Boltz-2; licensed users
# get Protenix v2 -- see protenix_v2.sh). The open v1 model ships via the `protenix` pip package
# (github.com/bytedance/Protenix):
#   pip install protenix
#   protenix pred -i <input.json> -o <out> -n protenix_base_default_v1.0.0
# The v1 checkpoint auto-downloads on first `protenix pred`. The CCD / auxiliary data is fetched by
# Protenix's own script into $PROTENIX_ROOT_DIR, which we point at data/weights/protenix_v1.
#
# Usage: scripts/weights/protenix_v1.sh [--dry-run]
#
# shellcheck source=scripts/weights/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_common_flags "$@"

DEST="$(dest_for protenix_v1)"
log "Protenix v1 (Apache-2.0, open) -> ${DEST}"

# 1. Install the protenix package (brings the CLI + its weight auto-downloader).
if ! have protenix; then
  warn "'protenix' CLI not found; installing the protenix package."
  if have uv; then
    run uv pip install protenix
  else
    run pip install protenix
  fi
fi

# 2. Fetch the inference auxiliary data (CCD etc.) into our pinned root.
#    Protenix's own script reads $PROTENIX_ROOT_DIR; --inference_only skips the big training data.
export PROTENIX_ROOT_DIR="${DEST}"
cat <<EOF
[weights] Protenix v1 set up:
  - package installed (provides 'protenix pred -n protenix_base_default_v1.0.0')
  - PROTENIX_ROOT_DIR=${DEST}
  To stage the CCD/auxiliary data, from a Protenix checkout run:
      PROTENIX_ROOT_DIR=${DEST} bash scripts/database/download_protenix_data.sh --inference_only
  The v1 model checkpoint auto-downloads on the first 'protenix pred'. To use this backend in the
  pipeline, set the cascade to protenix_v1 (configs/scorer/protenix.yaml) instead of the default
  Boltz-2. Nothing is committed (data/weights/ is git-ignored).
EOF
