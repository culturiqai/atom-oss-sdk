#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PY_BIN="venv/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

SCIENTIFIC_PROFILE="${ATOM_SCIENTIFIC_INTEGRITY_PROFILE:-fast_jax}"
SCIENTIFIC_FORCE_SPARSE="${ATOM_SCIENTIFIC_INTEGRITY_FORCE_SPARSE:-1}"
SCIENTIFIC_SEEDS_STR="${ATOM_SCIENTIFIC_INTEGRITY_SEEDS:-}"
SCIENTIFIC_NOISE_STR="${ATOM_SCIENTIFIC_INTEGRITY_NOISE_LEVELS:-}"
SCIENTIFIC_SAMPLES="${ATOM_SCIENTIFIC_INTEGRITY_SAMPLES:-}"
SCIENTIFIC_NULL_TRIALS_PER_SEED="${ATOM_SCIENTIFIC_INTEGRITY_NULL_TRIALS_PER_SEED:-}"
SCIENTIFIC_OUTPUT="validation_outputs/scientific_integrity.json"

SCIENTIFIC_CMD=(
  "${PY_BIN}" scripts/run_scientific_integrity_bench.py
  --profile "${SCIENTIFIC_PROFILE}"
  --check
  --output "${SCIENTIFIC_OUTPUT}"
)
if [[ "${SCIENTIFIC_FORCE_SPARSE}" == "1" ]]; then
  SCIENTIFIC_CMD+=(--force-sparse)
fi
if [[ -n "${SCIENTIFIC_SEEDS_STR}" ]]; then
  IFS=' ' read -r -a SCIENTIFIC_SEEDS <<< "${SCIENTIFIC_SEEDS_STR}"
  SCIENTIFIC_CMD+=(--seeds "${SCIENTIFIC_SEEDS[@]}")
fi
if [[ -n "${SCIENTIFIC_NOISE_STR}" ]]; then
  IFS=' ' read -r -a SCIENTIFIC_NOISE_LEVELS <<< "${SCIENTIFIC_NOISE_STR}"
  SCIENTIFIC_CMD+=(--noise-levels "${SCIENTIFIC_NOISE_LEVELS[@]}")
fi
if [[ -n "${SCIENTIFIC_SAMPLES}" ]]; then
  SCIENTIFIC_CMD+=(--samples "${SCIENTIFIC_SAMPLES}")
fi
if [[ -n "${SCIENTIFIC_NULL_TRIALS_PER_SEED}" ]]; then
  SCIENTIFIC_CMD+=(--null-trials-per-seed "${SCIENTIFIC_NULL_TRIALS_PER_SEED}")
fi
PYTHONDONTWRITEBYTECODE=1 "${SCIENTIFIC_CMD[@]}"

RELEASE_EVIDENCE_PROFILE="${ATOM_RELEASE_EVIDENCE_PROFILE:-ci}"
PYTHONDONTWRITEBYTECODE=1 "${PY_BIN}" scripts/generate_release_evidence.py \
  --profile "${RELEASE_EVIDENCE_PROFILE}" \
  --scientific "${SCIENTIFIC_OUTPUT}" \
  --allow-missing \
  --output-json "validation_outputs/release_evidence.json" \
  --output-md "validation_outputs/release_evidence.md"

echo "Scientific evidence gate passed."
