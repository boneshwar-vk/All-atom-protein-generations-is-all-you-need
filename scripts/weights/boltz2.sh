#!/usr/bin/env bash
#
# boltz2.sh -- download the Boltz-2 weights (MIT, open) into data/weights/boltz2/.
#
# Boltz-2 is the DEFAULT structure-prediction backend for the GRPO pipeline (configs/scorer/
# boltz2.yaml; the structure_auto scorer uses it unless licensed Protenix v2 weights are present).
# Its weights are MIT-licensed and openly downloadable from the `boltz-community/boltz-2` Hugging
# Face repo (~6 GB: boltz2_conf.ckpt, boltz2_aff.ckpt, mols/CCD). The repo's BoltzScorer passes
# this directory to the `boltz` CLI via `--cache`, so staging the snapshot here is all that is
# needed; Boltz also auto-downloads to this cache on first `boltz predict` if you skip this.
#
# Usage: scripts/weights/boltz2.sh [--dry-run]
#
# shellcheck source=scripts/weights/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_common_flags "$@"

REPO="boltz-community/boltz-2"
DEST="$(dest_for boltz2)"

log "Boltz-2 (MIT, open) -> ${DEST}"

# Skip if the main checkpoint is already staged (idempotent).
if [[ -f "${DEST}/boltz2_conf.ckpt" ]]; then
  log "boltz2_conf.ckpt already present; nothing to do."
  exit 0
fi

log "fetching Hugging Face snapshot ${REPO} (~6 GB) ..."
hf_snapshot "${REPO}" "${DEST}"

cat <<EOF
[weights] Boltz-2 staged under ${DEST}.
  The pipeline points configs/scorer/boltz2.yaml 'weights_dir' here and the BoltzScorer runs
  'boltz predict --cache ${DEST}'. Boltz unpacks the CCD/mols on first use. Nothing to commit
  (data/weights/ is git-ignored).
EOF
