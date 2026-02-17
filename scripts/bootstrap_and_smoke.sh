#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${ROOT_DIR}/venv"
PY_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

SKIP_INSTALL=0
if [[ "${1:-}" == "--skip-install" ]]; then
  SKIP_INSTALL=1
fi

if [[ ! -x "${PY_BIN}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

if [[ "${SKIP_INSTALL}" -eq 0 ]]; then
  echo "Installing/refreshing dependencies..."
  "${PIP_BIN}" install --upgrade pip setuptools wheel
  "${PIP_BIN}" install -e ".[dev,web]"
else
  echo "Skipping dependency installation (--skip-install)"
fi

echo "Running bootstrap smoke checks..."
PYTHONDONTWRITEBYTECODE=1 "${PY_BIN}" - <<'PY'
import ast
from pathlib import Path

targets = [
    "atom_experiment_runner.py",
    "atom_worlds.py",
    "src/atom/platform/webapp.py",
    "scripts/run_platform_ablations.py",
    "scripts/plot_platform_ablations.py",
    "scripts/supersonic_validation_pack.py",
    "scripts/run_scientific_integrity_bench.py",
    "scripts/generate_release_evidence.py",
    "scripts/build_release_packet.py",
    "scripts/check_docs_gate.py",
    "scripts/export_openapi.py",
]
for rel in targets:
    path = Path(rel)
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
print("AST syntax checks passed.")
PY

PYTHONDONTWRITEBYTECODE=1 "${PY_BIN}" scripts/export_openapi.py --output docs/openapi.json
PYTHONDONTWRITEBYTECODE=1 "${PY_BIN}" scripts/check_docs_gate.py
PYTHONDONTWRITEBYTECODE=1 "${PY_BIN}" scripts/reliability_kernel_smoke.py \
  --steps 8 \
  --seed 123 \
  --out validation_outputs/bootstrap_smoke.json
PYTHONDONTWRITEBYTECODE=1 "${PY_BIN}" scripts/supersonic_validation_pack.py \
  --steps 8 \
  --nx 32 \
  --ny 16 \
  --seed 123 \
  --allow-skip \
  --output validation_outputs/bootstrap_supersonic_validation.json
PYTHONDONTWRITEBYTECODE=1 "${PY_BIN}" scripts/run_scientific_integrity_bench.py \
  --force-sparse \
  --seeds 0 1 \
  --noise-levels 0.0 0.03 \
  --samples 96 \
  --check \
  --output validation_outputs/bootstrap_scientific_integrity.json
PYTHONDONTWRITEBYTECODE=1 "${PY_BIN}" scripts/generate_release_evidence.py \
  --allow-missing \
  --output-json validation_outputs/bootstrap_release_evidence.json \
  --output-md validation_outputs/bootstrap_release_evidence.md

ABLATION_SMOKE_SUMMARY="benchmark_results/platform_ablations_smoke/summary.json"
if [[ -f "${ABLATION_SMOKE_SUMMARY}" ]]; then
  PYTHONDONTWRITEBYTECODE=1 "${PY_BIN}" scripts/plot_platform_ablations.py \
    --summary "${ABLATION_SMOKE_SUMMARY}" \
    --output-dir benchmark_results/platform_ablations_smoke \
    --title "ATOM Platform Ablation Smoke Dashboard"
else
  echo "Skipping ablation plotting: missing ${ABLATION_SMOKE_SUMMARY}"
fi

echo "Bootstrap and smoke complete."
