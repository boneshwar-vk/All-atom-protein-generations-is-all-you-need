#!/usr/bin/env bash
#
# rfd3.sh -- install RFdiffusion3 (RFD3) and download its checkpoints (BSD-3, open) into
# data/weights/rfd3/.
#
# RFD3 is the backbone sampler for the GRPO pipeline. It ships via RosettaCommons/foundry (public,
# BSD-3-Clause). Foundry is the official installer/weights manager:
#   pip install "rc-foundry[all]"
#   foundry install rfd3 --checkpoint-dir <dir>
# Without --checkpoint-dir, foundry puts checkpoints in ~/.foundry/checkpoints; we pin them under
# data/weights/rfd3 so they sit beside the other models. Point the sampler at them via
# configs/sampler/default.yaml (`checkpoint:`) or `aapg-grpo ... model.checkpoint=<path>`.
#
# This needs a working `foundry` toolchain + a machine that can run RFD3 (GPU for real sampling).
# Requires network + the GPU stack; no-ops cleanly under --dry-run.
#
# Usage: scripts/weights/rfd3.sh [--dry-run]
#
# shellcheck source=scripts/weights/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_common_flags "$@"

DEST="$(dest_for rfd3)"
log "RFdiffusion3 (BSD-3, open) -> ${DEST}"

# 1. Ensure the Foundry CLI is available (install into the active environment if missing).
if ! have foundry; then
  warn "'foundry' CLI not found; installing rc-foundry into the active Python environment."
  if have uv; then
    run uv pip install "rc-foundry[all]"
  else
    run pip install "rc-foundry[all]"
  fi
fi

# 2. Install the RFD3 checkpoints into our pinned directory.
if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '    [dry-run] foundry install rfd3 --checkpoint-dir %s\n' "${DEST}"
  exit 0
fi
if ! have foundry; then
  die "'foundry' still not on PATH after install; open a new shell or check your environment."
fi

log "installing RFD3 checkpoints via foundry ..."
foundry install rfd3 --checkpoint-dir "${DEST}"

cat <<EOF
[weights] RFD3 checkpoints installed under ${DEST}.
  Inspect with: foundry list-installed
  Point the sampler at them, e.g.:
    aapg-grpo +experiment=grpo_binder model.checkpoint=${DEST}/<rfd3_ckpt> ...
  (RFD3's exact Hydra flag names are version-dependent -- see packages/rfdiffusion3/README.md
  and configs/sampler/default.yaml 'spec'.)
EOF
