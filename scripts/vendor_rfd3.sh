#!/usr/bin/env bash
#
# vendor_rfd3.sh -- (re-)vendor the RFdiffusion3 (RFD3) library into
# packages/rfdiffusion3/src/rfdiffusion3/, keeping the upstream import name.
#
# RFD3 is a *modified* upstream (ellipsoid conditioning, module 2), so it is vendored as a
# first-party workspace package rather than consumed as an external dependency. See the repo
# root VENDORING.md for policy and the provenance table you must update after running this.
#
# The script is idempotent (safe to re-run) and guards against a missing git / network.
# It does NOT commit, does NOT download model weights, and does NOT edit VENDORING.md for you
# (it prints a reminder with the exact commit SHA to record).
#
# Usage:
#   scripts/vendor_rfd3.sh [--repo <git-url>] [--branch <branch>] [--lib-subdir <path>]
#
# Defaults target RosettaCommons/foundry @ production, copying its models/rfd3 subtree.

set -euo pipefail

REPO_URL="https://github.com/RosettaCommons/foundry.git"
BRANCH="production"
LIB_SUBDIR="models/rfd3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)       REPO_URL="$2"; shift 2 ;;
    --branch)     BRANCH="$2"; shift 2 ;;
    --lib-subdir) LIB_SUBDIR="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "error: unknown argument '$1'" >&2
      exit 2 ;;
  esac
done

# --- locate the repo root and the destination package ---------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${REPO_ROOT}/packages/rfdiffusion3/src/rfdiffusion3"
LICENSE_DEST="${REPO_ROOT}/packages/rfdiffusion3/LICENSE"

# --- guard required tools -------------------------------------------------------------------
for tool in git rsync; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "error: '${tool}' is required but not found on PATH." >&2
    exit 1
  fi
done

# --- guard network reachability (best effort) -----------------------------------------------
if ! git ls-remote --exit-code "${REPO_URL}" >/dev/null 2>&1; then
  echo "error: cannot reach '${REPO_URL}' (no network, or wrong URL/branch/permissions)." >&2
  echo "       Clone it manually and re-run with --repo pointing at a local mirror." >&2
  exit 1
fi

# --- shallow clone into a temp dir that is always cleaned up ---------------------------------
TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

echo "[vendor_rfd3] cloning ${REPO_URL} (branch ${BRANCH}) ..."
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${TMP_DIR}/src"

SRC_LIB="${TMP_DIR}/src/${LIB_SUBDIR}"
if [[ ! -d "${SRC_LIB}" ]]; then
  echo "error: library subdir '${LIB_SUBDIR}' not found in the upstream checkout." >&2
  echo "       Inspect the repo layout and pass the correct --lib-subdir." >&2
  exit 1
fi

COMMIT_SHA="$(git -C "${TMP_DIR}/src" rev-parse HEAD)"

# --- sync the library source into the package (idempotent) ----------------------------------
# --delete makes the copy a clean mirror of upstream; local edits must live in clearly marked,
# separately importable modules so they can be re-applied after this (see VENDORING.md).
echo "[vendor_rfd3] syncing ${LIB_SUBDIR} -> ${DEST} ..."
mkdir -p "${DEST}"
rsync -a --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  "${SRC_LIB}/" "${DEST}/"

# --- copy the upstream license (RFD3 keeps its own terms) -----------------------------------
LICENSE_SRC=""
for cand in LICENSE LICENSE.txt LICENSE.md COPYING; do
  if [[ -f "${TMP_DIR}/src/${cand}" ]]; then
    LICENSE_SRC="${TMP_DIR}/src/${cand}"
    break
  fi
done
if [[ -n "${LICENSE_SRC}" ]]; then
  cp "${LICENSE_SRC}" "${LICENSE_DEST}"
  echo "[vendor_rfd3] copied upstream license -> ${LICENSE_DEST}"
else
  echo "[vendor_rfd3] WARNING: no upstream LICENSE found; confirm terms before distributing." >&2
fi

# --- reminder -------------------------------------------------------------------------------
cat <<EOF

[vendor_rfd3] done.
  Upstream : ${REPO_URL} (branch ${BRANCH}, subdir ${LIB_SUBDIR})
  Commit   : ${COMMIT_SHA}
  Vendored : ${DEST}

NEXT STEPS (not automated):
  1. Update VENDORING.md: add the upstream URL, the commit SHA above, today's date, and a
     one-line summary of local modifications to the provenance table.
  2. Confirm the upstream license permits modification and redistribution.
  3. Re-apply local edits (ellipsoid conditioning, module 2) and flip rfdiffusion3.__vendored__
     to True in src/rfdiffusion3/__init__.py.
  4. Provide RFD3 model weights separately (not vendored).
EOF
