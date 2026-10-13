#!/usr/bin/env bash
#
# proteinmpnn.sh -- verify the ProteinMPNN weights (MIT, open) are in place.
#
# ProteinMPNN is the sequence designer for the GRPO pipeline. Unlike the other models, its weights
# are VENDORED INTO THE REPO (MIT-licensed, ~6.7 MB each) under
#   packages/proteinmpnn/src/proteinmpnn/vendor/{vanilla,soluble}_model_weights/v_48_*.pt
# and the MPNNDesigner adapter loads them from there directly -- so there is nothing to download.
# This script just verifies they are present (a clone should already have them) and drops a
# convenience symlink at data/weights/proteinmpnn -> the vendored dir so every model is
# discoverable under one tree.
#
# Usage: scripts/weights/proteinmpnn.sh [--dry-run]
#
# shellcheck source=scripts/weights/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_common_flags "$@"

VENDOR="${REPO_ROOT}/packages/proteinmpnn/src/proteinmpnn/vendor"
log "ProteinMPNN (MIT, open) -- vendored in-repo at ${VENDOR}"

missing=0
for set in vanilla soluble; do
  for ckpt in v_48_002 v_48_010 v_48_020 v_48_030; do
    f="${VENDOR}/${set}_model_weights/${ckpt}.pt"
    if [[ -f "${f}" ]]; then
      log "  ok: ${set}/${ckpt}.pt"
    else
      warn "  MISSING: ${f}"
      missing=1
    fi
  done
done

if [[ "${missing}" -eq 1 ]]; then
  die "vendored ProteinMPNN weights are missing; re-run scripts/vendor_proteinmpnn or re-clone."
fi

# Convenience: expose the vendored dir under data/weights/proteinmpnn too (not required by the
# adapter, which reads the package path directly).
if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "${WEIGHTS_DIR}"
  link="${WEIGHTS_DIR}/proteinmpnn"
  if [[ ! -e "${link}" ]]; then
    ln -s "${VENDOR}" "${link}" && log "symlinked ${link} -> ${VENDOR}"
  fi
fi

log "ProteinMPNN weights verified (8 checkpoints present)."
