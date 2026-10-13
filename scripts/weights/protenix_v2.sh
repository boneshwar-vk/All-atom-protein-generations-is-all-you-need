#!/usr/bin/env bash
#
# protenix_v2.sh -- place the PROPRIETARY Protenix v2 weights (LICENSED USERS ONLY).
#
# Protenix v2 weights are NOT open source and are NOT downloaded or committed by this project. If
# you are licensed to use them, download them yourself and drop them into data/weights/protenix_v2/.
# The default `structure_auto` scorer detects them there and automatically switches from Boltz-2 to
# Protenix v2; otherwise the pipeline stays on open Boltz-2. (PX2 is "added later" -- this script
# only prepares the directory and prints instructions; it never fetches anything.)
#
# Usage: scripts/weights/protenix_v2.sh [--dry-run]
#
# shellcheck source=scripts/weights/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_common_flags "$@"

DEST="$(dest_for protenix_v2)"

cat <<EOF
[weights] Protenix v2 (PROPRIETARY -- licensed users only).
  This project does NOT download or redistribute these weights (it would violate the license).
  If you are licensed:
    1. Obtain the Protenix v2 weights from your licensed source.
    2. Place them under: ${DEST}
    3. Re-run the pipeline -- configs/scorer/protenix.yaml and the structure_auto scorer point
       'protenix_weights_dir' at ${DEST} and will auto-select Protenix v2 once files are present.
  Until then the pipeline runs on open Boltz-2. These weights must never be committed to git.
EOF
