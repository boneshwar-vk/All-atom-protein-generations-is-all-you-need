#!/usr/bin/env bash
#
# run_all.sh -- stage-wise, memory-safe GRPO binder run for a 24 GB MacBook.
#
# Runs the five pipeline stages IN ORDER, each as its OWN `uv run aapg-grpo-stage` process, so
# the previous stage's model memory (RFD3, ProteinMPNN, Boltz-2, PyRosetta) is fully released
# before the next stage loads. Every stage checkpoints to manifests under $RUN and is resumable:
# re-run this script (or a single stage) and completed ids are skipped.
#
# Usage:
#   scripts/macbook/run_all.sh                 # real run (needs weights + heavy deps on PATH)
#   MOCK=1 scripts/macbook/run_all.sh          # CPU mock run (no GPU/weights) -- CI-safe
#
# Environment knobs (all optional; defaults shown):
#   RUN=experiments/outputs/egfr_mac   # work dir for manifests + structures
#   MOCK=0                             # 1 => --mock (CPU mocks, no heavy deps)
#   N=60                               # backbones to sample (stage 1)
#   SEQS_PER=2                         # sequences designed per backbone (stage 2)
#   SEED=0                             # sampling/design seed
#   DESIGNER=proteinmpnn               # mpnn designer backend
#   WEIGHTS_SET=soluble                # MPNN weight set
#   BACKEND=boltz2                     # fold backend: boltz2 | structure_auto | protenix_v2
#   RECYCLING_STEPS=1                  # fold recycling steps (keep low on Apple Silicon)
#   DIFFUSION_SAMPLES=1                # fold diffusion samples (1 => one complex at a time)
#   MIN_FREE_GB=4                      # fold bails (exit 75) below this free-memory floor
#   OUTPUT=$RUN/ranked.parquet         # ranked table (extension picks the format)
#   Target spec (pick ONE source):
#     TARGET_JSON=path/to/target.json  # {structure_path,chains,binder_chains,hotspots,binder_length}
#   or individual flags:
#     STRUCTURE_PATH, CHAINS, BINDER_CHAINS, HOTSPOTS, BINDER_LENGTH
#
# Exit 75 from the fold stage means "low memory, progress checkpointed" -- just re-run to resume.

set -euo pipefail

usage() {
  grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'
}
case "${1:-}" in
  -h | --help)
    usage
    exit 0
    ;;
  "") ;;
  *)
    echo "unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN="${RUN:-experiments/outputs/egfr_mac}"
MOCK="${MOCK:-0}"
N="${N:-60}"
SEQS_PER="${SEQS_PER:-2}"
SEED="${SEED:-0}"
DESIGNER="${DESIGNER:-proteinmpnn}"
WEIGHTS_SET="${WEIGHTS_SET:-soluble}"
BACKEND="${BACKEND:-boltz2}"
RECYCLING_STEPS="${RECYCLING_STEPS:-1}"
DIFFUSION_SAMPLES="${DIFFUSION_SAMPLES:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-4}"
OUTPUT="${OUTPUT:-${RUN}/ranked.parquet}"

mkdir -p "${RUN}"

# Shared args every stage receives: the work dir, the mock switch, and the target spec.
COMMON=(--run "${RUN}")
if [ "${MOCK}" = "1" ]; then
  COMMON+=(--mock)
fi
if [ -n "${TARGET_JSON:-}" ]; then
  COMMON+=(--target-json "${TARGET_JSON}")
fi
[ -n "${STRUCTURE_PATH:-}" ] && COMMON+=(--structure-path "${STRUCTURE_PATH}")
[ -n "${CHAINS:-}" ] && COMMON+=(--chains "${CHAINS}")
[ -n "${BINDER_CHAINS:-}" ] && COMMON+=(--binder-chains "${BINDER_CHAINS}")
[ -n "${HOTSPOTS:-}" ] && COMMON+=(--hotspots "${HOTSPOTS}")
[ -n "${BINDER_LENGTH:-}" ] && COMMON+=(--binder-length "${BINDER_LENGTH}")

stage() { # stage <name> <extra args...>
  local name="$1"
  shift
  printf '\n\033[1m>>> stage: %s\033[0m\n' "${name}"
  uv run aapg-grpo-stage "${name}" "${COMMON[@]}" "$@"
}

# One stage = one process. They run sequentially so RAM never stacks across models.
stage rfd3 --n "${N}" --seed "${SEED}"
stage mpnn --seqs-per "${SEQS_PER}" --designer "${DESIGNER}" --weights-set "${WEIGHTS_SET}" --seed "${SEED}"
stage fold --backend "${BACKEND}" --recycling-steps "${RECYCLING_STEPS}" \
  --diffusion-samples "${DIFFUSION_SAMPLES}" --min-free-gb "${MIN_FREE_GB}"
stage rosetta
stage reward --output "${OUTPUT}"

printf '\n\033[1;32mdone\033[0m: ranked table at %s\n' "${OUTPUT}"
