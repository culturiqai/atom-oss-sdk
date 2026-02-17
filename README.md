# ATOM Research SDK (Production Path)

ATOM is a simulation-first neuro-symbolic platform for:
- Closed-loop scientific discovery
- Inverse design over real world-adapter rollouts
- Runtime-safe control experiments (with intervention/fallback envelope)

This repository contains the canonical runner, challenge worlds, API/UI surface, and validation gates.

## 1. Environment

From repo root:

```bash
cd /path/to/Atom-OSS-SDK-RP1
```

Use the project virtual environment when available:

```bash
venv/bin/python --version
venv/bin/pytest --version
```

Install modes (editable):



```bash
# Core SDK only
./setup_venv.sh OR venv/bin/python -m pip install -e . 

# Platform API/UI extras
./setup_venv.sh web OR venv/bin/python -m pip install -e ".[web]"
```

Release and support docs:
- `docs/SDK_RELEASE.md`
- `docs/RELEASE_RUNBOOK.md`
- `docs/SUPPORT_MATRIX.md`
- `CHANGELOG.md`

## 2. Core Commands

### 2.1 Reliability Gate (recommended first)

```bash
PYTHONDONTWRITEBYTECODE=1 scripts/run_phase0_gate.sh
```

This runs:
- Unit/integration tests
- Reliability smoke run
- Supersonic validation pack
- Scientific integrity benchmark gate
- Release evidence synthesis artifact

### 2.2 Canonical Training Loop


**Canonical entrypoint:** `atom_experiment_runner.py` (experiment pipeline + ablations). `training_loop.py` is a lower-level orchestrator used for deeper integration and internal experiments.

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python atom_experiment_runner.py \
  --name atom_train \
  --world analytical:taylor_green \
  --grid 32 32 16 \
  --steps 300 \
  --device cpu
```

Ablation toggles:

```bash
--ablate-scientist
--ablate-symplectic
--ablate-trust-gate
```

Safety controls:

```bash
--disable-safety
--safety-min-theory-trust 0.08
--safety-max-stress 8.0
--safety-shift-z-threshold 8.0
--safety-shift-warmup 24
--safety-density-min 0.05
--safety-density-max 8.0
--safety-speed-max 3.0
```

### 2.3 Inverse Design

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python atom_experiment_runner.py \
  --inverse-design \
  --name inverse_run \
  --world analytical:taylor_green \
  --grid 32 32 16 \
  --inverse-backend evolutionary \
  --inverse-iterations 8 \
  --inverse-population 12 \
  --inverse-top-k 5 \
  --inverse-rollout-steps 64 \
  --device cpu
```

### 2.4 Supersonic Challenge

Via ATOM CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m atom.cli run supersonic --steps 300 --headless
```

Direct challenge runner:

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python src/atom/challenges/supersonic_wedge_challenge.py \
  --steps 300 \
  --headless \
  --output-dir challenge_results
```

### 2.5 Platform API + UI

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m uvicorn atom.platform.webapp:create_app \
  --factory --host 0.0.0.0 --port 8000
```

Open:
- UI: `http://localhost:8000/`
- Swagger: `http://localhost:8000/docs`
- OpenAPI: `http://localhost:8000/openapi.json`

### 2.6 Platform Ablation Harness (ATOM + PPO baseline)

Canonical full-suite run:

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/run_platform_ablations.py \
  --world analytical:taylor_green \
  --grid 32 32 16 \
  --steps 200 \
  --seeds 0 1 2 \
  --atom-variant-set production \
  --force-sparse-scientist \
  --device cpu \
  --out benchmark_results/platform_ablations
```

Re=100 stable whole-suite run (all ATOM production variants + PPO baseline):

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/run_platform_ablations.py \
  --world lbm2d:cylinder \
  --grid 64 64 1 \
  --steps 120 \
  --seeds 0 1 2 \
  --atom-variant-set production \
  --force-sparse-scientist \
  --device cpu \
  --world-u-inlet 0.05 \
  --world-tau 0.53 \
  --safety-shift-warmup 8 \
  --disable-safety \
  --scientist-warmup-steps 0 \
  --out benchmark_results/platform_ablations_lbm_re100_nosafety
```

Artifacts:
- `benchmark_results/platform_ablations/manifest.json`
- `benchmark_results/platform_ablations/runs.json`
- `benchmark_results/platform_ablations/summary.json`
- `benchmark_results/platform_ablations/summary.csv`
- `benchmark_results/platform_ablations/platform_ablation_dashboard.png`
- `benchmark_results/platform_ablations/platform_ablation_dashboard.pdf`
- `benchmark_results/platform_ablations/platform_ablation_claims.md`
- `benchmark_results/platform_ablations/platform_ablation_significance.json`
- `benchmark_results/platform_ablations/platform_ablation_plots.json`

Production ATOM matrix variants:
- `v2_hybrid`
- `v2_residual`
- `v2_raw`
- `v1_raw`
- `no_scientist`

Generate dashboard plots and claim report from summary:

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/plot_platform_ablations.py \
  --summary benchmark_results/platform_ablations/summary.json \
  --runs benchmark_results/platform_ablations/runs.json \
  --output-dir benchmark_results/platform_ablations \
  --bootstrap-samples 4000 \
  --bootstrap-seed 123 \
  --title "ATOM Platform Ablation Dashboard"
```

### 2.7 Supersonic Validation Pack

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/supersonic_validation_pack.py \
  --steps 16 \
  --nx 64 \
  --ny 32 \
  --seed 123 \
  --output validation_outputs/supersonic_validation.json
```

### 2.8 Export Versioned OpenAPI Snapshot

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/export_openapi.py \
  --output docs/openapi.json
```

### 2.9 Bootstrap + Smoke (new machine)

```bash
scripts/bootstrap_and_smoke.sh
```

Skip install if dependencies are already present:

```bash
scripts/bootstrap_and_smoke.sh --skip-install
```

### 2.10 Docs Gate

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/check_docs_gate.py
```

### 2.11 Scientific Integrity Benchmark Gate

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/run_scientific_integrity_bench.py \
  --force-sparse \
  --seeds 0 1 2 \
  --noise-levels 0.0 0.03 0.06 \
  --samples 192 \
  --check \
  --output validation_outputs/scientific_integrity.json
```

### 2.12 Release Evidence Artifact

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/generate_release_evidence.py \
  --output-json validation_outputs/release_evidence.json \
  --output-md validation_outputs/release_evidence.md
```

Convenience gate script:

```bash
scripts/run_scientific_evidence_gate.sh
```

### 2.13 Optional Full Ablation Gate Profile

```bash
ATOM_ENABLE_FULL_ABLATION_GATE=1 \
ATOM_FULL_ABLATION_STEPS=48 \
ATOM_FULL_ABLATION_SEEDS="0 1 2 3 4" \
PYTHONDONTWRITEBYTECODE=1 scripts/run_phase0_gate.sh
```

Outputs:
- `benchmark_results/platform_ablations_gate/`
- `validation_outputs/release_evidence_full_ablation.json`
- `validation_outputs/release_evidence_full_ablation.md`
- `release_packets/full_ablation_latest/`

### 2.14 Build Release Packet

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/build_release_packet.py \
  --ablation-dir benchmark_results/platform_ablations \
  --release-evidence-json validation_outputs/release_evidence.json \
  --release-evidence-md validation_outputs/release_evidence.md \
  --output-dir release_packets/manual_packet
```

## 3. Key Runtime Artifacts

- Runner results: `experiment_results/<name>/results.json`
- Inverse-design reports: `experiment_results/<name>/inverse_design_report.json`
- API async inverse jobs: `runs/inverse_design_jobs/<job_name>/`
- API async supersonic jobs: `runs/supersonic_jobs/<job_name>/`
- Scientific IDE demo templates: `examples/studio/*.json`
- Challenge audit: `challenge_results/challenge_audit.json`
- Validation outputs: `validation_outputs/`
- Telemetry (orchestrator): `logs/telemetry.json`
- Release evidence: `validation_outputs/release_evidence.json`
- Release evidence markdown: `validation_outputs/release_evidence.md`

## 4. API Surfaces

See full details in:
- `docs/PLATFORM_API_UI.md`
- `docs/openapi.json`
- `docs/PRODUCTION_ROADMAP.md`
- `docs/SDK_RELEASE.md`

Notable API endpoints:
- `GET /api/v1/studio/catalog`
- `GET /api/v1/challenges/supersonic/jobs/{job_id}/telemetry`
- `GET /api/v1/runtime/telemetry`
- `POST /api/v1/assistant/query`

## 5. Notes

- The repository intentionally keeps `scientist.py` as core symbolic backend and `scientist_v2.py` as structural coupling wrapper.
- Canonical theory signal for production path is 10D.
- Platform direction is reliability-first: contracts, replay, safety, and validation before scaling autonomy.

## 6. Open-Source Readiness

- License: Apache-2.0 (`LICENSE`) + optional `NOTICE` attribution file
- Patent grant: included via Apache-2.0
- Contributions: inbound=outbound (Apache-2.0) + DCO sign-off recommended (`git commit -s`)
- Contribution guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- Issue templates: `.github/ISSUE_TEMPLATE/`
- PR template: `.github/pull_request_template.md`
- CI quality gates: `.github/workflows/quality-gates.yml`


## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.
