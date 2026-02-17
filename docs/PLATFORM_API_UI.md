# ATOM Platform API + UI

This document covers the production API/UI service for:
- Inverse-design orchestration
- Supersonic challenge job orchestration
- Live job telemetry and results retrieval
- Grounded assistant analysis over active jobs

## Goal

Provide a production-facing service surface for:
- Submitting inverse-design jobs
- Submitting supersonic challenge jobs
- Tracking job status and results
- Reviewing top candidates in a UI
- Publishing OpenAPI documentation for integration

## Launch

Install web dependencies:

```bash
pip install fastapi uvicorn jinja2
```

Run from repository root:

```bash
venv/bin/python -m uvicorn atom.platform.webapp:create_app --factory --host 0.0.0.0 --port 8000
```

Director-pack endpoint security defaults:
- `ATOM_ENABLE_DIRECTOR_PACK=0` by default (endpoint returns `403`).
- Set `ATOM_ENABLE_DIRECTOR_PACK=1` to enable `/api/v1/studio/director-pack`.
- `ATOM_RELEASE_PROFILE` controls strictness for packet/evidence scripts: `dev`, `ci`, `release`.
- In `release` profile, permissive `allow_missing` mode is rejected.

Open:
- UI: `http://localhost:8000/`
- Swagger: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

Versioned OpenAPI artifact in-repo:

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/export_openapi.py --output docs/openapi.json
```

## API Endpoints

- `GET /healthz`
  - liveness probe
- `GET /api/v1/worlds`
  - world registry from `atom_worlds.py`
- `GET /api/v1/studio/catalog`
  - simulator inventory + curated demo experiments from `examples/studio/*.json`
- `POST /api/v1/studio/director-pack`
  - one-click deterministic replay + evidence-bundle build for launch-grade demo packets (disabled by default unless `ATOM_ENABLE_DIRECTOR_PACK=1`)
- `GET /api/v1/runtime/telemetry`
  - latest orchestrator telemetry snapshot with latent-saliency proxy and recent hypothesis tail
- `GET /api/v1/inverse-design/spec-template`
  - objective/parameter template JSON plus machine-readable `world_kwargs` contracts by simulator
- `POST /api/v1/inverse-design/jobs`
  - submit async inverse-design job
- `GET /api/v1/inverse-design/jobs`
  - list submitted jobs
- `GET /api/v1/inverse-design/jobs/{job_id}`
  - retrieve one job status/result
- `POST /api/v1/challenges/supersonic/jobs`
  - submit async supersonic wedge challenge job
- `GET /api/v1/challenges/supersonic/jobs`
  - list supersonic challenge jobs
- `GET /api/v1/challenges/supersonic/jobs/{job_id}`
  - get one supersonic challenge job including latest telemetry snapshot
- `GET /api/v1/challenges/supersonic/jobs/{job_id}/telemetry`
  - get bounded telemetry timeseries + derived trend metrics + recent law events
- `POST /api/v1/assistant/query`
  - deterministic grounded assistant answer with explicit citations and recommended actions

## Request Example

```json
{
  "name": "wing_sweep_v1",
  "world_spec": "analytical:taylor_green",
  "grid_shape": [32, 32, 16],
  "backend": "evolutionary",
  "iterations": 8,
  "population": 12,
  "top_k": 5,
  "rollout_steps": 64,
  "device": "cpu"
}
```

Supersonic challenge submit:

```json
{
  "name": "supersonic_wedge_run_01",
  "steps": 256,
  "headless": true
}
```

Minimal `curl` usage:

```bash
curl -s http://localhost:8000/api/v1/worlds | jq .
```

```bash
curl -s -X POST http://localhost:8000/api/v1/inverse-design/jobs -H "Content-Type: application/json" -d '{"name":"inverse_ui_job","world_spec":"analytical:taylor_green","grid_shape":[32,32,16],"backend":"evolutionary","iterations":8,"population":12,"top_k":5,"rollout_steps":64,"device":"cpu"}' | jq .
```

```bash
curl -s -X POST http://localhost:8000/api/v1/challenges/supersonic/jobs \
  -H "Content-Type: application/json" \
  -d '{"name":"supersonic_ui_job","steps":192,"headless":true}' | jq .
```

```bash
curl -s "http://localhost:8000/api/v1/challenges/supersonic/jobs/<job_id>/telemetry?limit=256" | jq .
```

```bash
curl -s -X POST http://localhost:8000/api/v1/assistant/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Assess current posture","mode":"risk","telemetry_window":96}' | jq .
```

```bash
ATOM_ENABLE_DIRECTOR_PACK=1 \
curl -s -X POST http://localhost:8000/api/v1/studio/director-pack \
  -H "Content-Type: application/json" \
  -d '{"tag":"ui_inverse_design","run_reliability":true,"run_supersonic_validation":true,"run_release_evidence":true,"allow_missing":true}' | jq .
```

Fetch schema contracts for objective, parameter bounds, and simulator-specific `world_kwargs`:

```bash
curl -s http://localhost:8000/api/v1/inverse-design/spec-template | jq .
```

The response includes:
- `objective_spec`
- `parameter_space`
- `world_kwargs_contracts.profiles` (field-level type/range requirements)
- `world_kwargs_contracts.world_spec_profile_map` (which profile applies to each supported world)
- `supported_world_specs`

UI contract behavior:
- The inverse-design form renders `world_kwargs` inputs from `world_kwargs_contracts` instead of raw JSON text entry.
- Field type/range checks are enforced client-side before submit and revalidated server-side on job creation.
- For `lbm:custom`, selecting a geometry asset suppresses manual `stl_path` entry in the UI and relies on geometry binding at submit.

## Runtime Model

- Jobs are tracked in-memory by `InverseDesignJobService`.
- Job outputs are written under `runs/inverse_design_jobs/<job_name>/`.
- Candidate evaluation uses real `WorldAdapter` rollouts through `ATOMExperiment.run_inverse_design`.
- Supersonic jobs are tracked by `SupersonicChallengeJobService`.
- Supersonic job outputs are written under `runs/supersonic_jobs/<job_name>/`.
- Live supersonic step telemetry is attached to each job as `latest_telemetry`.
- Supersonic telemetry trace and recent law events are persisted in-memory for UI charts and API consumers.
- Assistant responses are deterministic and grounded only on live job/result artifacts in this service.

## Documentation Gate

Run docs freshness checks:

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/check_docs_gate.py
```

This gate validates:
- command snippets are shell-parseable
- referenced local script/doc paths exist
- endpoint markers are present in docs
- `docs/openapi.json` matches live `create_app().openapi()`

## UI Surfaces

The single-page UI exposes:
- Scientific IDE Experiment Studio (simulator library + launchable demo cards)
- One-click Director Pack builder for deterministic evidence artifacts
- Inverse-design launcher + queue
- Contract-driven world parameter editor (typed fields from `/api/v1/inverse-design/spec-template`)
- Supersonic challenge launcher + queue
- Live supersonic telemetry panel
- Observability KPI strip with reward/shock/reduction trends
- Dual telemetry charts (reward/reduction and shock/jet power)
- Runtime saliency chart from latest `theory_packet.jacobian` telemetry
- Attention overlay view that projects Eyes2 and Brain saliency peaks on live flow fields
- Grounded assistant panel (summary/status/risk/next actions modes)
- Selected job full JSON report

## UI Theme

The frontend is intentionally styled as **DaVinci meets terminal**:
- parchment gradients and engraved panel lines
- mono console report panel
- NVIDIA-green execution accents for control actions and status chips
- live status matrix (queued/running/succeeded/failed)
- supersonic telemetry panel with streaming shock metrics
- observability dashboards and deterministic assistant console
- responsive layout for desktop and mobile

## Examples Pack

Curated demo templates used by the IDE are versioned in:
- `examples/studio/inverse_taylor_green_evolutionary.json`
- `examples/studio/inverse_lbm2d_cylinder_bayesian.json`
- `examples/studio/inverse_lbm3d_cylinder_gradient.json`
- `examples/studio/supersonic_wedge_control.json`

## Planning Artifact

Program-level roadmap and phase gates are documented in:
- `docs/PRODUCTION_ROADMAP.md`
