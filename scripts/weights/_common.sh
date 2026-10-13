#!/usr/bin/env bash
#
# _common.sh -- shared helpers for the data/weights download scripts. SOURCE this; do not run it.
#
# Every model script resolves the weights root the same way the Hydra `paths` group does
# (configs/paths/default.yaml): ${AAPG_ROOT:-<repo root>}/data/weights/<model>. So the files land
# in the exact locations the scorers/sampler read (e.g. configs/scorer/boltz2.yaml points
# `weights_dir` at ${paths.weights_dir}/boltz2). data/weights/ is git-ignored: weights are fetched
# on setup, never committed.

set -euo pipefail

# --- paths ----------------------------------------------------------------------------------
# scripts/weights/_common.sh -> repo root is two levels up.
_WEIGHTS_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_WEIGHTS_COMMON_DIR}/../.." && pwd)"
# Mirror configs/paths/default.yaml: root = ${AAPG_ROOT:-repo root}, weights = $root/data/weights.
AAPG_ROOT="${AAPG_ROOT:-${REPO_ROOT}}"
WEIGHTS_DIR="${AAPG_ROOT}/data/weights"

# --- logging --------------------------------------------------------------------------------
log()  { printf '\033[1m[weights]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[weights] WARNING:\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[weights] ERROR:\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

# --- flags shared by every model script -----------------------------------------------------
DRY_RUN=0
parse_common_flags() {
  for arg in "$@"; do
    case "${arg}" in
      --dry-run|-n) DRY_RUN=1 ;;
      -h|--help)
        # Print the script's leading comment block as help (minus the shebang), then exit.
        grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'
        exit 0 ;;
    esac
  done
}

# Run a command, or just print it under --dry-run.
run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '    [dry-run] %s\n' "$*"
  else
    log "running: $*"
    "$@"
  fi
}

have() { command -v "$1" >/dev/null 2>&1; }

# Destination dir for a model (created unless dry-run); echoes the path.
dest_for() {
  local model="$1" target="${WEIGHTS_DIR}/$1"
  if [[ "${DRY_RUN}" -eq 0 ]]; then mkdir -p "${target}"; fi
  printf '%s' "${target}"
}

# Download a Hugging Face repo snapshot into a local dir, preferring the CLI and falling back to
# the Python API. $1 = repo id, $2 = local dir, rest = extra args (e.g. --include patterns).
hf_snapshot() {
  local repo="$1" local_dir="$2"; shift 2
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '    [dry-run] huggingface snapshot %s -> %s %s\n' "${repo}" "${local_dir}" "$*"
    return 0
  fi
  if have huggingface-cli; then
    huggingface-cli download "${repo}" --local-dir "${local_dir}" "$@"
  elif have hf; then
    hf download "${repo}" --local-dir "${local_dir}" "$@"
  else
    # Fall back to the Python API (runs inside the project env if `uv` is available).
    local runner=(python3)
    if have uv; then runner=(uv run python); fi
    "${runner[@]}" - "${repo}" "${local_dir}" <<'PY'
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit(
        "huggingface_hub is required: `pip install huggingface_hub` "
        "(or `pip install \"huggingface_hub[cli]\"` for the huggingface-cli)."
    )
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
  fi
}
