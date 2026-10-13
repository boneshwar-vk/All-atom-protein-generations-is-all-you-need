#!/usr/bin/env bash
#
# smoke_test.sh -- exhaustive CPU/mock smoke tests for the GRPO pipeline.
#
# These prove the repo WIRING is correct without any GPU, heavy deps (torch/boltz/pyrosetta/
# protenix) or downloaded weights: everything runs through the deterministic mock adapters. Run
# this on any machine (laptop included) to confirm a clone is healthy. A REAL structure-folding
# run needs GPU + weights + PyRosetta and is out of scope here (see docs/EGFR_smoke_test.md).
#
# Each check prints PASS/FAIL and the suite continues on failure, then exits non-zero if any
# check failed, so you get the full picture in one run.
#
# Usage: scripts/smoke_test.sh [--egfr-only] [--quick]
#   --quick      skip lint/type checks (faster)
#   --egfr-only  run only the EGFR mock pipeline check

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

QUICK=0
EGFR_ONLY=0
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;;
    --egfr-only) EGFR_ONLY=1 ;;
    -h|--help) grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

PASS=0; FAIL=0
RESULTS=()
check() {  # check "<name>" <cmd...>
  local name="$1"; shift
  printf '\n\033[1m>>> %s\033[0m\n' "${name}"
  if "$@"; then
    printf '\033[1;32mPASS\033[0m: %s\n' "${name}"; PASS=$((PASS+1)); RESULTS+=("PASS  ${name}")
  else
    printf '\033[1;31mFAIL\033[0m: %s\n' "${name}"; FAIL=$((FAIL+1)); RESULTS+=("FAIL  ${name}")
  fi
}

# --- EGFR mock pipeline: a reusable helper used by the full suite and --egfr-only -----------
egfr_mock() {
  # Exercises the whole selection pipeline on EGFR-shaped inputs via the mock stack (no folding).
  # Target chains/hotspots mimic an EGFR domain-III epitope binder task; the mock ignores the
  # (absent) structure, so this validates config + plumbing, not biology.
  uv run python - <<'PY'
import sys
from aapg.grpo.cli import main
rc = main([
    "mock=true",
    "n_backbones=8",
    "seqs_per_backbone=4",
    "top_k=4",
    "group_size=8",
    "target.binder_length=80",
    "target.chains=A",
    "target.binder_chains=B",
    "target.hotspots=A465,A467,A468",
    "output_path=experiments/outputs/egfr_smoke/ranked.jsonl",
])
sys.exit(rc)
PY
}

if [[ "${EGFR_ONLY}" -eq 1 ]]; then
  check "EGFR mock pipeline" egfr_mock
  printf '\n%s\n' "${RESULTS[@]}"
  exit $(( FAIL > 0 ))
fi

# --- 0. environment -------------------------------------------------------------------------
check "env: uv available" command -v uv
check "env: python importable" uv run python -c "import sys; print(sys.version)"

# --- 1. install / sync ----------------------------------------------------------------------
check "uv sync --all-packages (CPU-only, no heavy deps)" uv sync --all-packages

# --- 2. imports + registries ----------------------------------------------------------------
check "import core + registries list the expected scorers/rewards" uv run python -c "
import aapg.toolkit, aapg.score_head, aapg.grpo.reward  # register side-effects
from aapg.core.scoring import SCORERS, REWARDS
s = set(SCORERS.available()); r = set(REWARDS.available())
need_s = {'boltz2','boltz2_mock','protenix_v2','protenix_v2_mock','rosetta_interface',
          'rosetta_interface_mock','structure_auto','structure_auto_mock','structure_head',
          'structure_head_mock'}
need_r = {'ipsae_gated'}
miss = (need_s - s) | (need_r - r)
assert not miss, f'missing registrations: {miss}'
print('scorers:', sorted(s)); print('rewards:', sorted(r))
"
check "import adapters (mock samplers/designers)" uv run python -c "
from rfdiffusion3.adapter import MockRFD3Sampler, Target
from proteinmpnn.adapter import MockMPNNDesigner, get_designer
print('rfd3 + proteinmpnn adapters import OK')
"

# --- 3. unit tests --------------------------------------------------------------------------
check "pytest (full suite, CPU-only)" uv run pytest -q

# --- 4. lint / format / type ----------------------------------------------------------------
if [[ "${QUICK}" -eq 0 ]]; then
  check "ruff check" uv run ruff check .
  check "ruff format --check" uv run ruff format --check .
  check "mypy (first-party src)" uv run mypy \
    packages/core/src packages/toolkit/src packages/rfd3_ellipsoid/src packages/grpo_steer/src \
    packages/mpnn_iter/src packages/validation/src packages/pipelines/src \
    packages/rfdiffusion3/src packages/proteinmpnn/src packages/score_head/src
fi

# --- 5. vendored ProteinMPNN weights present ------------------------------------------------
check "ProteinMPNN vendored weights present" bash scripts/weights/proteinmpnn.sh

# --- 6. weight download scripts are sound (dry-run; no network) ------------------------------
check "weight scripts dry-run (no download)" bash scripts/weights/download_all.sh --dry-run

# --- 7. mock pipelines ----------------------------------------------------------------------
check "mock pipeline (default target)" uv run python -c "
from aapg.grpo.cli import main; raise SystemExit(main(['mock=true','n_backbones=6','seqs_per_backbone=3']))
"
check "EGFR mock pipeline (ranked table written)" egfr_mock
check "EGFR ranked table has all 11 metric columns + rows" uv run python -c "
import json
from aapg.grpo.selection import METRIC_COLUMNS
rows = [json.loads(l) for l in open('experiments/outputs/egfr_smoke/ranked.jsonl')]
assert len(rows) == 8*4, f'expected 32 rows, got {len(rows)}'
for m in METRIC_COLUMNS:
    assert m in rows[0], f'missing metric column {m}'
assert any(r['selected'] for r in rows), 'no candidate selected'
print(f'{len(rows)} EGFR candidate rows, columns OK, top reward={max(r[\"reward\"] for r in rows):.4f}')
"

# --- summary --------------------------------------------------------------------------------
printf '\n\033[1m==== SMOKE TEST SUMMARY ====\033[0m\n'
printf '%s\n' "${RESULTS[@]}"
printf '\n%d passed, %d failed\n' "${PASS}" "${FAIL}"
exit $(( FAIL > 0 ))
