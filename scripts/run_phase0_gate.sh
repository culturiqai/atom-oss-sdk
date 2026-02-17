#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTEST_BIN="venv/bin/pytest"
PY_BIN="venv/bin/python"

if [[ ! -x "${PYTEST_BIN}" ]]; then
  echo "missing ${PYTEST_BIN}. Create/install project venv first." >&2
  exit 1
fi
if [[ ! -x "${PY_BIN}" ]]; then
  echo "missing ${PY_BIN}. Create/install project venv first." >&2
  exit 1
fi

PYTHONDONTWRITEBYTECODE=1 "${PYTEST_BIN}" -p no:cacheprovider \
  tests/unit/test_governance.py \
  tests/unit/test_memory.py \
  tests/unit/test_scientist.py \
  tests/unit/test_platform_contracts.py \
  tests/unit/test_platform_ablation_harness.py \
  tests/unit/test_platform_ablation_plots.py \
  tests/unit/test_platform_webapp.py \
  tests/unit/test_runtime_safety.py \
  tests/unit/test_inverse_design.py \
  tests/unit/test_experiment_runner_inverse.py \
  tests/unit/test_release_evidence.py \
  tests/unit/test_build_release_packet.py \
  tests/unit/test_scientific_integrity_bench.py \
  tests/unit/test_supersonic_challenge.py \
  tests/unit/test_supersonic_world.py \
  tests/unit/test_supersonic_validation_pack.py \
  tests/unit/test_scientist_v2_contracts.py \
  tests/integration/test_training_loop.py \
  tests/test_distributions.py -q

"${PY_BIN}" scripts/reliability_kernel_smoke.py \
  --steps "${ATOM_PHASE0_STEPS:-20}" \
  --seed "${ATOM_PHASE0_SEED:-123}" \
  --out "validation_outputs/reliability_kernel_smoke.json"

"${PY_BIN}" scripts/supersonic_validation_pack.py \
  --steps "${ATOM_SUPERSONIC_VALIDATION_STEPS:-12}" \
  --nx "${ATOM_SUPERSONIC_VALIDATION_NX:-48}" \
  --ny "${ATOM_SUPERSONIC_VALIDATION_NY:-24}" \
  --seed "${ATOM_SUPERSONIC_VALIDATION_SEED:-123}" \
  --allow-skip \
  --output "validation_outputs/supersonic_validation.json"

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
"${PY_BIN}" scripts/generate_release_evidence.py \
  --profile "${RELEASE_EVIDENCE_PROFILE}" \
  --allow-missing \
  --output-json "validation_outputs/release_evidence.json" \
  --output-md "validation_outputs/release_evidence.md"

if [[ "${ATOM_ENABLE_FULL_ABLATION_GATE:-0}" == "1" ]]; then
  FULL_OUT="${ATOM_FULL_ABLATION_OUT:-benchmark_results/platform_ablations_gate}"
  FULL_PROFILE="${ATOM_FULL_ABLATION_PROFILE:-fast_jax}"
  FULL_WORLD="${ATOM_FULL_ABLATION_WORLD:-lbm2d:cylinder}"
  FULL_STEPS="${ATOM_FULL_ABLATION_STEPS:-72}"
  FULL_DEVICE="${ATOM_FULL_ABLATION_DEVICE:-cpu}"
  FULL_VARIANT_SET="${ATOM_FULL_ABLATION_VARIANT_SET:-production}"
  FULL_GRID_STR="${ATOM_FULL_ABLATION_GRID:-64 64 1}"
  FULL_SEEDS_STR="${ATOM_FULL_ABLATION_SEEDS:-0 1}"
  FULL_SAFETY_SHIFT_WARMUP="${ATOM_FULL_ABLATION_SAFETY_SHIFT_WARMUP:-8}"
  FULL_MIN_DELTA="${ATOM_FULL_ABLATION_MIN_DELTA:--0.25}"
  FULL_PACKET_DIR="${ATOM_FULL_ABLATION_PACKET_DIR:-release_packets/full_ablation_latest}"

  IFS=' ' read -r -a FULL_GRID <<< "${FULL_GRID_STR}"
  IFS=' ' read -r -a FULL_SEEDS <<< "${FULL_SEEDS_STR}"
  if [[ "${#FULL_GRID[@]}" -ne 3 ]]; then
    echo "ATOM_FULL_ABLATION_GRID must contain exactly 3 integers, got: ${FULL_GRID_STR}" >&2
    exit 2
  fi
  if [[ "${#FULL_SEEDS[@]}" -lt 1 ]]; then
    echo "ATOM_FULL_ABLATION_SEEDS must contain at least one seed, got: ${FULL_SEEDS_STR}" >&2
    exit 2
  fi

  INCLUDE_SB3_FLAG=()
  if [[ "${ATOM_FULL_ABLATION_INCLUDE_SB3:-0}" == "1" ]]; then
    INCLUDE_SB3_FLAG=(--include-sb3)
  fi

  "${PY_BIN}" scripts/run_platform_ablations.py \
    --profile "${FULL_PROFILE}" \
    --world "${FULL_WORLD}" \
    --grid "${FULL_GRID[0]}" "${FULL_GRID[1]}" "${FULL_GRID[2]}" \
    --steps "${FULL_STEPS}" \
    --seeds "${FULL_SEEDS[@]}" \
    --safety-shift-warmup "${FULL_SAFETY_SHIFT_WARMUP}" \
    --atom-variant-set "${FULL_VARIANT_SET}" \
    --force-sparse-scientist \
    --device "${FULL_DEVICE}" \
    "${INCLUDE_SB3_FLAG[@]}" \
    --out "${FULL_OUT}"

  "${PY_BIN}" scripts/plot_platform_ablations.py \
    --summary "${FULL_OUT}/summary.json" \
    --runs "${FULL_OUT}/runs.json" \
    --output-dir "${FULL_OUT}" \
    --title "ATOM Platform Ablation Gate Dashboard"

  "${PY_BIN}" scripts/generate_release_evidence.py \
    --profile "${ATOM_FULL_ABLATION_RELEASE_PROFILE:-release}" \
    --ablation "${FULL_OUT}/summary.json" \
    --min-atom-vs-baseline-reward-delta "${FULL_MIN_DELTA}" \
    --output-json "validation_outputs/release_evidence_full_ablation.json" \
    --output-md "validation_outputs/release_evidence_full_ablation.md" \
    --check

  "${PY_BIN}" scripts/build_release_packet.py \
    --ablation-dir "${FULL_OUT}" \
    --release-evidence-json "validation_outputs/release_evidence_full_ablation.json" \
    --release-evidence-md "validation_outputs/release_evidence_full_ablation.md" \
    --output-dir "${FULL_PACKET_DIR}" \
    --profile "${ATOM_FULL_ABLATION_RELEASE_PROFILE:-release}"
fi

echo "Phase-0 gate passed."
