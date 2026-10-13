#!/usr/bin/env bash
#
# esm2.sh -- download ESM-2 (MIT, open) into data/weights/esm2/.  OPTIONAL.
#
# ESM-2 is not required by the default GRPO pipeline (Boltz-2 + Rosetta). It is provided for
# optional single-sequence embeddings / experiments. Default model is the 650M checkpoint; override
# with MODEL=<hf repo> (e.g. MODEL=facebook/esm2_t36_3B_UR50D).
#
# Usage: MODEL=facebook/esm2_t33_650M_UR50D scripts/weights/esm2.sh [--dry-run]
#
# shellcheck source=scripts/weights/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_common_flags "$@"

MODEL="${MODEL:-facebook/esm2_t33_650M_UR50D}"
DEST="$(dest_for esm2)"
SUBDIR="${DEST}/${MODEL##*/}"

log "ESM-2 (MIT, open) ${MODEL} -> ${SUBDIR}"
if [[ -f "${SUBDIR}/config.json" ]]; then
  log "already present; nothing to do."
  exit 0
fi
hf_snapshot "${MODEL}" "${SUBDIR}"
log "ESM-2 staged under ${SUBDIR} (optional; not used by the default cascade)."
