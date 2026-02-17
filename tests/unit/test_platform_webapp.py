"""Unit tests for platform webapp job orchestration."""

import json
import time
from importlib.util import find_spec

from atom.platform import webapp


class _DummyRunner:
    class ExperimentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ATOMExperiment:
        def __init__(self, config):
            self.config = config

        def run_inverse_design(self, **kwargs):
            return {
                "backend": kwargs["backend"],
                "top_candidates": [{"candidate_id": "cand_000001"}],
                "summary": {"ok": True},
            }


class _FailingRunner:
    class ExperimentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ATOMExperiment:
        def __init__(self, config):
            self.config = config

        def run_inverse_design(self, **kwargs):
            raise RuntimeError("simulated runner failure")


class _DummyChallenge:
    @staticmethod
    def run_challenge(max_steps=100, headless=True, output_dir=".", on_step_callback=None, on_law_callback=None):
        if on_step_callback is not None:
            on_step_callback(
                {
                    "step": 0,
                    "reward": 0.1,
                    "shock_strength": 0.2,
                    "shock_reduction": 0.05,
                    "jet_power": 0.001,
                }
            )
            on_step_callback(
                {
                    "step": 1,
                    "reward": 0.12,
                    "shock_strength": 0.18,
                    "shock_reduction": 0.07,
                    "jet_power": 0.0012,
                }
            )
        if on_law_callback is not None:
            on_law_callback("reward ~ x", 0)
            on_law_callback("shock ~ y", 1)
        return {
            "challenge": "supersonic_wedge_control",
            "summary": {
                "steps_executed": int(max_steps),
                "final_reward_50": 0.11,
                "final_shock_strength_50": 0.19,
                "final_shock_reduction_50": 0.06,
            },
            "history": {
                "reward": [0.1, 0.12],
                "shock_strength": [0.2, 0.18],
                "shock_reduction": [0.05, 0.07],
                "jet_power": [0.001, 0.0012],
            },
            "output_dir": output_dir,
            "headless": bool(headless),
        }


class _FailingChallenge:
    @staticmethod
    def run_challenge(*args, **kwargs):
        raise RuntimeError("simulated challenge failure")


class _SlowChallenge:
    @staticmethod
    def run_challenge(max_steps=100, headless=True, output_dir=".", on_step_callback=None, on_law_callback=None):
        rewards = []
        shock = []
        reduction = []
        jet = []
        for step in range(min(int(max_steps), 64)):
            telemetry = {
                "step": step,
                "reward": 0.2 + 0.001 * step,
                "shock_strength": 0.2 + 0.0015 * step,
                "shock_reduction": 0.05 + 0.0008 * step,
                "jet_power": 0.001 + 0.0001 * step,
            }
            rewards.append(telemetry["reward"])
            shock.append(telemetry["shock_strength"])
            reduction.append(telemetry["shock_reduction"])
            jet.append(telemetry["jet_power"])
            if on_step_callback is not None:
                on_step_callback(dict(telemetry))
            if on_law_callback is not None and step in {0, 8, 16}:
                on_law_callback(f"law_step_{step}", step)
            time.sleep(0.003)
        return {
            "challenge": "supersonic_wedge_control",
            "summary": {
                "steps_executed": len(rewards),
                "final_reward_50": rewards[-1] if rewards else 0.0,
                "final_shock_strength_50": shock[-1] if shock else 0.0,
                "final_shock_reduction_50": reduction[-1] if reduction else 0.0,
            },
            "history": {
                "reward": rewards,
                "shock_strength": shock,
                "shock_reduction": reduction,
                "jet_power": jet,
            },
            "output_dir": output_dir,
            "headless": bool(headless),
        }


def test_job_service_run_sync_success(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    service = webapp.InverseDesignJobService(workspace_root=tmp_path)

    req = webapp.InverseDesignJobRequest(
        name="webapp_test_success",
        world_spec="analytical:taylor_green",
        grid_shape=[16, 16, 8],
        backend="evolutionary",
        iterations=2,
        population=3,
        rollout_steps=8,
    )
    job = service.run_sync(req)

    assert job.status == "succeeded"
    assert job.result is not None
    assert job.result["backend"] == "evolutionary"
    fetched = service.get_job(job.job_id)
    assert fetched is not None
    assert fetched["status"] == "succeeded"


def test_job_service_run_sync_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _FailingRunner)
    service = webapp.InverseDesignJobService(workspace_root=tmp_path)

    req = webapp.InverseDesignJobRequest(
        name="webapp_test_failure",
        world_spec="analytical:taylor_green",
        grid_shape=[16, 16, 8],
        backend="evolutionary",
    )
    job = service.run_sync(req)

    assert job.status == "failed"
    assert job.error is not None
    fetched = service.get_job(job.job_id)
    assert fetched is not None
    assert fetched["status"] == "failed"


def test_supersonic_service_run_sync_success(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "_resolve_challenge_module", lambda: _DummyChallenge)
    service = webapp.SupersonicChallengeJobService(workspace_root=tmp_path)

    req = webapp.SupersonicChallengeJobRequest(
        name="supersonic_success",
        steps=12,
        headless=True,
    )
    job = service.run_sync(req)

    assert job.status == "succeeded"
    assert job.result is not None
    assert job.result["challenge"] == "supersonic_wedge_control"
    assert job.latest_telemetry is not None
    fetched = service.get_job(job.job_id)
    assert fetched is not None
    assert fetched["status"] == "succeeded"
    assert fetched["latest_telemetry"]["step"] == 1
    assert fetched["telemetry_points"] >= 2
    assert fetched["law_events"] >= 2

    telemetry = service.get_telemetry(job.job_id, limit=8)
    assert telemetry is not None
    assert telemetry["job_id"] == job.job_id
    assert telemetry["timeseries"]["reward"] == [0.1, 0.12]
    assert telemetry["derived"]["reward_mean_last_32"] > 0.0
    assert len(telemetry["law_trace"]) >= 2


def test_supersonic_service_run_sync_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "_resolve_challenge_module", lambda: _FailingChallenge)
    service = webapp.SupersonicChallengeJobService(workspace_root=tmp_path)

    req = webapp.SupersonicChallengeJobRequest(
        name="supersonic_failure",
        steps=12,
        headless=True,
    )
    job = service.run_sync(req)

    assert job.status == "failed"
    assert job.error is not None
    fetched = service.get_job(job.job_id)
    assert fetched is not None
    assert fetched["status"] == "failed"


def test_supersonic_service_control_plane(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "_resolve_challenge_module", lambda: _SlowChallenge)
    service = webapp.SupersonicChallengeJobService(workspace_root=tmp_path)
    req = webapp.SupersonicChallengeJobRequest(name="supersonic_control", steps=40, headless=True)
    job = service.submit(req)

    # Wait for active execution window.
    for _ in range(200):
        snap = service.get_job(job.job_id)
        if snap and snap["status"] == "running":
            break
        time.sleep(0.005)

    pause = service.control(job.job_id, webapp.SupersonicControlRequest(action="pause", note="hold"))
    assert pause is not None
    cancel = service.control(job.job_id, webapp.SupersonicControlRequest(action="cancel", note="stop"))
    assert cancel is not None
    bookmark = service.control(
        job.job_id, webapp.SupersonicControlRequest(action="bookmark", note="operator_mark")
    )
    assert bookmark is not None

    for _ in range(300):
        snap = service.get_job(job.job_id)
        if snap and snap["status"] in {"cancelled", "failed", "succeeded"}:
            break
        time.sleep(0.01)

    telemetry = service.get_telemetry(job.job_id, limit=64)
    assert telemetry is not None
    assert "timeline_tail" in telemetry
    assert "incidents_tail" in telemetry
    assert "control_state" in telemetry
    timeline = service.get_timeline(job.job_id, limit=64)
    assert timeline is not None
    assert len(timeline["timeline"]) >= 1


def test_geometry_asset_store_roundtrip(tmp_path):
    store = webapp.GeometryAssetStore(tmp_path)
    asset = store.save_upload("wing.stl", b"solid wing\nendsolid wing\n")
    assert asset["geometry_id"].startswith("geo_")
    assert asset["is_stl"] is True
    listed = store.list_assets()
    assert listed
    resolved = store.resolve(asset["geometry_id"])
    assert resolved is not None
    assert resolved["filename"] == "wing.stl"


def test_create_app_dependency_guard():
    if find_spec("fastapi") is not None:
        # Environment has FastAPI; this guard test is only for missing-dependency mode.
        return
    try:
        webapp.create_app()
    except RuntimeError as exc:
        assert "FastAPI stack not installed" in str(exc)
    else:  # pragma: no cover - should not happen in this environment
        raise AssertionError("Expected create_app to fail when FastAPI is unavailable")


def test_assistant_service_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    monkeypatch.setattr(webapp, "_resolve_challenge_module", lambda: _DummyChallenge)
    inverse_service = webapp.InverseDesignJobService(workspace_root=tmp_path)
    challenge_service = webapp.SupersonicChallengeJobService(workspace_root=tmp_path)

    inverse_job = inverse_service.run_sync(
        webapp.InverseDesignJobRequest(
            name="assistant_inverse",
            world_spec="analytical:taylor_green",
            grid_shape=[16, 16, 8],
            backend="evolutionary",
            iterations=2,
            population=3,
            rollout_steps=8,
        )
    )
    challenge_job = challenge_service.run_sync(
        webapp.SupersonicChallengeJobRequest(
            name="assistant_supersonic",
            steps=12,
            headless=True,
        )
    )
    assistant = webapp.GovernedAssistantService()
    response = assistant.respond(
        webapp.AssistantQueryRequest(
            question="Summarize current state",
            mode="summary",
            inverse_job_id=inverse_job.job_id,
            supersonic_job_id=challenge_job.job_id,
            telemetry_window=32,
        ),
        inverse_service,
        challenge_service,
    )

    assert response.grounded is True
    assert response.mode == "summary"
    assert response.intent == "scientific_discovery"
    assert response.diagnostic_mode == "overview"
    assert response.engine == "deterministic"
    assert response.citations
    assert "Grounding policy" in response.answer
    assert response.recommended_actions


def test_assistant_service_llm_engine_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    monkeypatch.setattr(webapp, "_resolve_challenge_module", lambda: _DummyChallenge)
    monkeypatch.setenv("ATOM_ASSISTANT_ENABLE_LLM", "0")
    inverse_service = webapp.InverseDesignJobService(workspace_root=tmp_path)
    challenge_service = webapp.SupersonicChallengeJobService(workspace_root=tmp_path)

    inverse_job = inverse_service.run_sync(
        webapp.InverseDesignJobRequest(
            name="assistant_inverse_fallback",
            world_spec="analytical:taylor_green",
            grid_shape=[16, 16, 8],
            backend="evolutionary",
            iterations=2,
            population=3,
            rollout_steps=8,
        )
    )
    challenge_job = challenge_service.run_sync(
        webapp.SupersonicChallengeJobRequest(
            name="assistant_supersonic_fallback",
            steps=12,
            headless=True,
        )
    )
    assistant = webapp.GovernedAssistantService(workspace_root=tmp_path)
    response = assistant.respond(
        webapp.AssistantQueryRequest(
            question="Summarize current state with llm grounding",
            mode="summary",
            intent="control",
            diagnostic_mode="brain_saliency",
            engine="llm_grounded",
            inverse_job_id=inverse_job.job_id,
            supersonic_job_id=challenge_job.job_id,
            telemetry_window=32,
        ),
        inverse_service,
        challenge_service,
    )

    assert response.grounded is True
    assert response.intent == "control"
    assert response.diagnostic_mode == "brain_saliency"
    assert response.engine == "deterministic_fallback"
    assert response.recommended_actions


def test_runtime_telemetry_snapshot_includes_diagnostics_maps(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": 7,
        "reward": 0.21,
        "stress": 0.03,
        "theory": 0.12,
        "theory_packet": {
            "prediction": 0.12,
            "jacobian": [0.1, -0.2, 0.3],
            "trust_raw": 0.6,
            "trust_verified": 0.55,
            "trust_structural_floor": 0.25,
        },
        "governance_decision": {
            "approved": False,
            "novelty_score": 0.04,
            "stability_score": 0.52,
            "intervention_consistency": 0.61,
            "same_support_refinement": False,
            "reasons": ["novelty_below_threshold"],
        },
        "verifier_metrics": {
            "rmse": 0.22,
            "correlation": 0.41,
            "interventional": 0.63,
        },
        "baseline_context": {
            "ready": True,
            "slope": -0.15,
            "r2": 0.81,
            "discovery_target_mode": "hybrid",
            "discovery_hybrid_alpha": 0.35,
            "discovery_target_value": 0.04,
        },
        "diagnostics": {
            "enabled": True,
            "updated_step": 6,
            "stale_steps": 1,
            "interval": 8,
            "live_view": {
                "projection": "xy_mean",
                "speed_xy": {
                    "map_xy": [[0.1, 0.2], [0.3, 0.4]],
                    "shape": [2, 2],
                    "stride": 1,
                    "normalized": True,
                    "raw_min": 0.01,
                    "raw_max": 0.99,
                },
                "density_xy": {
                    "map_xy": [[0.4, 0.3], [0.2, 0.1]],
                    "shape": [2, 2],
                    "stride": 1,
                    "normalized": True,
                    "raw_min": 0.1,
                    "raw_max": 1.1,
                },
                "divergence_xy": {
                    "map_xy": [[0.5, 0.5], [0.5, 0.5]],
                    "shape": [2, 2],
                    "stride": 1,
                    "normalized": True,
                    "raw_min": -0.2,
                    "raw_max": 0.2,
                },
                "obstacle_xy": {
                    "map_xy": [[0.0, 1.0], [1.0, 1.0]],
                    "shape": [2, 2],
                    "stride": 1,
                    "normalized": False,
                    "raw_min": 0.0,
                    "raw_max": 1.0,
                },
            },
            "eyes2_saliency": {
                "method": "grad_embedding_energy",
                "objective": 0.7,
                "projection": "xy_mean",
                "target_label": "latent_2",
                "target_index": 2,
                "map": {
                    "map_xy": [[0.5, 0.6], [0.7, 0.8]],
                    "shape": [2, 2],
                    "stride": 1,
                    "normalized": True,
                    "raw_min": 0.0,
                    "raw_max": 1.0,
                },
            },
            "brain_saliency": {
                "method": "grad_policy_energy",
                "objective": 0.4,
                "projection": "xy_mean",
                "map": {
                    "map_xy": [[0.2, 0.1], [0.3, 0.2]],
                    "shape": [2, 2],
                    "stride": 1,
                    "normalized": True,
                    "raw_min": 0.0,
                    "raw_max": 1.0,
                },
                "theory_feature_importance": [1.0, 0.4, 0.2],
                "theory_feature_labels": ["prediction", "latent_0", "trust"],
            },
        },
    }
    (logs / "telemetry.json").write_text(json.dumps(payload), encoding="utf-8")
    (logs / "hypotheses.jsonl").write_text("", encoding="utf-8")

    snapshot = webapp._load_runtime_telemetry_snapshot(tmp_path)
    assert snapshot["available"] is True
    assert snapshot["diagnostics"]["enabled"] is True
    assert snapshot["diagnostics"]["live_view"]["speed_xy"]["shape"] == [2, 2]
    assert snapshot["diagnostics"]["live_view"]["obstacle_xy"]["shape"] == [2, 2]
    assert snapshot["diagnostics"]["eyes2_saliency"]["map"]["shape"] == [2, 2]
    assert snapshot["diagnostics"]["eyes2_saliency"]["target_label"] == "latent_2"
    assert snapshot["diagnostics"]["eyes2_saliency"]["target_index"] == 2
    assert snapshot["diagnostics"]["brain_saliency"]["map"]["shape"] == [2, 2]
    assert snapshot["saliency"]["theory_feature_importance"] == [1.0, 0.4, 0.2]
    assert snapshot["governance_decision"]["approved"] is False
    assert snapshot["verifier_metrics"]["rmse"] == 0.22
    assert snapshot["baseline_context"]["ready"] is True


def test_assistant_runtime_metrics_include_obstacle_shape(tmp_path):
    assistant = webapp.GovernedAssistantService(workspace_root=tmp_path)
    runtime = assistant._runtime_metrics(
        {
            "available": True,
            "status": "ok",
            "step": 3,
            "reward": 0.2,
            "stress": 0.04,
            "theory": 0.1,
            "trust_raw": 0.6,
            "trust_verified": 0.5,
            "trust_structural_floor": 0.2,
            "safety_summary": {
                "interventions": 3,
                "fallback_uses": 2,
                "mode": "cautious",
                "hard_event_rate": 0.28,
                "degrade_window": 64,
                "mode_transitions": {"normal_to_cautious": 1},
                "reason_histogram": {"low_theory_trust": 2},
            },
            "governance_decision": {
                "approved": False,
                "novelty_score": 0.07,
                "stability_score": 0.52,
                "intervention_consistency": 0.64,
                "reasons": ["novelty_below_threshold"],
            },
            "verifier_metrics": {
                "rmse": 0.19,
                "correlation": 0.41,
                "interventional": 0.64,
            },
            "baseline_context": {
                "ready": True,
                "slope": -0.22,
                "r2": 0.78,
                "discovery_target_mode": "hybrid",
                "discovery_target_value": 0.03,
            },
            "diagnostics": {
                "enabled": True,
                "updated_step": 3,
                "stale_steps": 0,
                "interval": 8,
                "live_view": {
                    "speed_xy": {"shape": [2, 2], "raw_min": 0.0, "raw_max": 1.0},
                    "density_xy": {"shape": [2, 2], "raw_min": 0.0, "raw_max": 1.0},
                    "divergence_xy": {"shape": [2, 2], "raw_min": -0.2, "raw_max": 0.2},
                    "obstacle_xy": {"shape": [2, 2], "raw_min": 0.0, "raw_max": 1.0},
                },
                "eyes2_saliency": {"map": {"shape": [2, 2]}, "objective": 0.7, "target_label": "latent_0"},
                "brain_saliency": {"map": {"shape": [2, 2]}, "objective": 0.4},
            },
            "saliency": {"top_latents": []},
        }
    )
    assert runtime["available"] is True
    assert runtime["live_obstacle_shape"] == [2, 2]
    assert runtime["live_obstacle_range"]["max"] == 1.0
    assert runtime["safety_mode"] == "cautious"
    assert runtime["safety_interventions"] == 3
    assert runtime["safety_fallback_uses"] == 2
    assert runtime["safety_hard_event_rate"] == 0.28
    assert runtime["governance_approved"] is False
    assert runtime["governance_reasons"] == ["novelty_below_threshold"]
    assert runtime["verifier_rmse"] == 0.19
    assert runtime["baseline_ready"] is True


def test_assistant_extract_llm_text_chat_completion(tmp_path):
    assistant = webapp.GovernedAssistantService(workspace_root=tmp_path)
    payload = {"choices": [{"message": {"content": "chat-completion-response"}}]}
    assert assistant._extract_llm_text(payload) == "chat-completion-response"


def test_create_app_routes_and_assistant_contract():
    if find_spec("fastapi") is None:
        return
    from fastapi.testclient import TestClient

    app = webapp.create_app()
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "ATOM Platform Console" in root.text

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    studio = client.get("/api/v1/studio/catalog")
    assert studio.status_code == 200
    studio_payload = studio.json()
    assert isinstance(studio_payload.get("simulators"), list)
    assert isinstance(studio_payload.get("demos"), list)

    geometries = client.get("/api/v1/geometries")
    assert geometries.status_code == 200
    assert isinstance(geometries.json().get("assets"), list)
    empty_geometry = client.post("/api/v1/geometries/upload", content=b"", headers={"x-atom-filename": "wing.stl"})
    assert empty_geometry.status_code == 400

    runtime = client.get("/api/v1/runtime/telemetry")
    assert runtime.status_code == 200
    runtime_payload = runtime.json()
    assert runtime_payload["available"] in {True, False}

    spec_template = client.get("/api/v1/inverse-design/spec-template")
    assert spec_template.status_code == 200
    template_payload = spec_template.json()
    assert "objective_spec" in template_payload
    assert "parameter_space" in template_payload
    assert "world_kwargs_contracts" in template_payload
    assert "supported_world_specs" in template_payload
    contracts = template_payload["world_kwargs_contracts"]
    assert "profiles" in contracts
    assert "world_spec_profile_map" in contracts
    assert "analytical" in contracts["profiles"]
    assert "supersonic" in contracts["profiles"]
    assert "analytical:taylor_green" in contracts["world_spec_profile_map"]
    assert "supersonic:wedge_d2q25" in template_payload["supported_world_specs"]

    missing_telemetry = client.get("/api/v1/challenges/supersonic/jobs/missing/telemetry")
    assert missing_telemetry.status_code == 404
    missing_timeline = client.get("/api/v1/challenges/supersonic/jobs/missing/timeline")
    assert missing_timeline.status_code == 404

    assistant = client.post(
        "/api/v1/assistant/query",
        json={"question": "what is the current state", "mode": "summary"},
    )
    assert assistant.status_code == 200
    payload = assistant.json()
    assert payload["grounded"] is True
    assert payload["intent"] == "scientific_discovery"
    assert payload["diagnostic_mode"] == "overview"
    assert payload["engine"] in {"deterministic", "deterministic_fallback"}
    assert "Grounding policy" in payload["answer"]


def test_inverse_design_submit_rejects_invalid_parameter_space(monkeypatch, tmp_path):
    if find_spec("fastapi") is None:
        return
    from fastapi.testclient import TestClient

    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    bad_param_path = tmp_path / "bad_parameter_space.json"
    bad_param_path.write_text(
        json.dumps({"action_bias": [1.0, 1.0]}),
        encoding="utf-8",
    )

    app = webapp.create_app(workspace_root=str(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/api/v1/inverse-design/jobs",
        json={
            "name": "bad_param_contract",
            "parameter_space_path": "bad_parameter_space.json",
        },
    )

    assert resp.status_code == 422
    detail = str(resp.json().get("detail", ""))
    assert "parameter_space_path" in detail


def test_inverse_design_submit_rejects_invalid_objective_constraints(monkeypatch, tmp_path):
    if find_spec("fastapi") is None:
        return
    from fastapi.testclient import TestClient

    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    bad_obj_path = tmp_path / "bad_objective.json"
    bad_obj_path.write_text(
        json.dumps(
            {
                "targets": {"reward_mean": 0.0},
                "constraints": {"reward_mean": {"weight": 1.0}},
                "penalties": {"reward_std": 0.1},
                "hard_bounds": {"action_bias": [-1.0, 1.0]},
                "solver_budget": {"iterations": 2, "population": 2, "seed": 7},
            }
        ),
        encoding="utf-8",
    )

    app = webapp.create_app(workspace_root=str(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/api/v1/inverse-design/jobs",
        json={
            "name": "bad_obj_contract",
            "objective_spec_path": "bad_objective.json",
        },
    )

    assert resp.status_code == 422
    detail = str(resp.json().get("detail", ""))
    assert "objective_spec_path" in detail
    assert "constraints" in detail


def test_inverse_design_submit_rejects_hard_bounds_not_in_parameter_space(monkeypatch, tmp_path):
    if find_spec("fastapi") is None:
        return
    from fastapi.testclient import TestClient

    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    param_path = tmp_path / "parameter_space.json"
    param_path.write_text(
        json.dumps({"action_bias": [-1.0, 1.0]}),
        encoding="utf-8",
    )
    obj_path = tmp_path / "objective_spec.json"
    obj_path.write_text(
        json.dumps(
            {
                "targets": {"reward_mean": 0.0},
                "constraints": {"reward_mean": {"min": -1.0}},
                "penalties": {"reward_std": 0.1},
                "hard_bounds": {"unknown_param": [0.0, 1.0]},
                "solver_budget": {"iterations": 2, "population": 2, "seed": 7},
            }
        ),
        encoding="utf-8",
    )

    app = webapp.create_app(workspace_root=str(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/api/v1/inverse-design/jobs",
        json={
            "name": "mismatched_contract",
            "objective_spec_path": "objective_spec.json",
            "parameter_space_path": "parameter_space.json",
        },
    )

    assert resp.status_code == 422
    detail = str(resp.json().get("detail", ""))
    assert "hard_bounds" in detail
    assert "parameter_space_path" in detail


def test_inverse_design_submit_accepts_valid_contract_files(monkeypatch, tmp_path):
    if find_spec("fastapi") is None:
        return
    from fastapi.testclient import TestClient

    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    param_path = tmp_path / "parameter_space.json"
    param_path.write_text(
        json.dumps(
            {
                "action_bias": [-1.0, 1.0],
                "action_gain": [0.0, 1.0],
                "action_frequency": [0.0, 0.3],
                "action_phase": [-3.14, 3.14],
            }
        ),
        encoding="utf-8",
    )
    obj_path = tmp_path / "objective_spec.json"
    obj_path.write_text(
        json.dumps(
            {
                "targets": {"reward_mean": 0.0, "turbulence": 0.0},
                "constraints": {
                    "reward_mean": {"min": -1.0},
                    "action_energy": {"max": 1.0},
                },
                "penalties": {"reward_std": 0.1},
                "hard_bounds": {
                    "action_bias": [-1.0, 1.0],
                    "action_gain": [0.0, 1.0],
                },
                "solver_budget": {"iterations": 2, "population": 2, "seed": 11},
            }
        ),
        encoding="utf-8",
    )

    app = webapp.create_app(workspace_root=str(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/api/v1/inverse-design/jobs",
        json={
            "name": "valid_contract",
            "world_spec": "analytical:taylor_green",
            "grid_shape": [16, 16, 8],
            "backend": "evolutionary",
            "rollout_steps": 8,
            "objective_spec_path": "objective_spec.json",
            "parameter_space_path": "parameter_space.json",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["job_id"]
    assert payload["status"] in {"queued", "running", "succeeded"}

    jobs_payload = client.get("/api/v1/inverse-design/jobs").json()["jobs"]
    matching = [job for job in jobs_payload if job.get("job_id") == payload["job_id"]]
    assert matching
    submitted = matching[0]["request"]
    assert submitted["objective_spec_path"] == str(obj_path.resolve())
    assert submitted["parameter_space_path"] == str(param_path.resolve())


def test_job_service_rejects_invalid_world_kwargs_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    service = webapp.InverseDesignJobService(workspace_root=tmp_path)
    req = webapp.InverseDesignJobRequest(
        name="invalid_world_kwargs",
        world_spec="analytical:taylor_green",
        world_kwargs={"u_inlet": 0.05},
    )
    try:
        service.run_sync(req)
    except ValueError as exc:
        assert "world_kwargs" in str(exc)
    else:  # pragma: no cover - contract enforcement should reject request
        raise AssertionError("Expected invalid world_kwargs to raise ValueError")


def test_inverse_design_submit_rejects_invalid_world_kwargs_key(monkeypatch, tmp_path):
    if find_spec("fastapi") is None:
        return
    from fastapi.testclient import TestClient

    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    app = webapp.create_app(workspace_root=str(tmp_path))
    client = TestClient(app)

    resp = client.post(
        "/api/v1/inverse-design/jobs",
        json={
            "name": "invalid_world_kwargs_key",
            "world_spec": "analytical:taylor_green",
            "world_kwargs": {"u_inlet": 0.05},
        },
    )
    assert resp.status_code == 422
    detail = str(resp.json().get("detail", ""))
    assert "world_kwargs" in detail
    assert "unsupported fields" in detail


def test_inverse_design_submit_rejects_invalid_supersonic_world_kwargs(monkeypatch, tmp_path):
    if find_spec("fastapi") is None:
        return
    from fastapi.testclient import TestClient

    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    app = webapp.create_app(workspace_root=str(tmp_path))
    client = TestClient(app)

    resp = client.post(
        "/api/v1/inverse-design/jobs",
        json={
            "name": "invalid_supersonic_kwargs",
            "world_spec": "supersonic:wedge_d2q25",
            "world_kwargs": {"tau": 0.45},
        },
    )
    assert resp.status_code == 422
    detail = str(resp.json().get("detail", ""))
    assert "world_kwargs.tau" in detail


def test_inverse_design_submit_accepts_valid_supersonic_world_kwargs(monkeypatch, tmp_path):
    if find_spec("fastapi") is None:
        return
    from fastapi.testclient import TestClient

    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    app = webapp.create_app(workspace_root=str(tmp_path))
    client = TestClient(app)

    resp = client.post(
        "/api/v1/inverse-design/jobs",
        json={
            "name": "valid_supersonic_kwargs",
            "world_spec": "supersonic:wedge_d2q25",
            "world_kwargs": {
                "tau": 0.8,
                "inflow_velocity": 0.16,
                "warmup_steps": 10,
                "wedge_half_angle_deg": 14.0,
                "jet_gain": 0.01,
                "seed": 5,
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    jobs_payload = client.get("/api/v1/inverse-design/jobs").json()["jobs"]
    matching = [job for job in jobs_payload if job.get("job_id") == payload["job_id"]]
    assert matching
    submitted = matching[0]["request"]
    assert submitted["world_spec"] == "supersonic:wedge_d2q25"
    assert submitted["world_kwargs"]["tau"] == 0.8
    assert submitted["world_kwargs"]["warmup_steps"] == 10
    assert submitted["world_kwargs"]["seed"] == 5


def test_inverse_design_submit_custom_world_binds_geometry_asset(monkeypatch, tmp_path):
    if find_spec("fastapi") is None:
        return
    from fastapi.testclient import TestClient

    monkeypatch.setattr(webapp, "_resolve_runner_module", lambda: _DummyRunner)
    app = webapp.create_app(workspace_root=str(tmp_path))
    client = TestClient(app)

    spec = client.get("/api/v1/inverse-design/spec-template")
    assert spec.status_code == 200
    contracts = spec.json()["world_kwargs_contracts"]
    profile_name = contracts["world_spec_profile_map"]["lbm:custom"]
    stl_spec = contracts["profiles"][profile_name]["fields"]["stl_path"]
    assert stl_spec["required"] is True

    upload = client.post(
        "/api/v1/geometries/upload",
        content=b"solid wing\nendsolid wing\n",
        headers={"x-atom-filename": "wing.stl"},
    )
    assert upload.status_code == 200
    asset = upload.json()["asset"]
    geometry_id = asset["geometry_id"]

    submit = client.post(
        "/api/v1/inverse-design/jobs",
        json={
            "name": "custom_geometry_binding",
            "world_spec": "lbm:custom",
            "grid_shape": [16, 16, 8],
            "backend": "evolutionary",
            "rollout_steps": 8,
            "top_k": 1,
            "geometry_id": geometry_id,
            "world_kwargs": {},
        },
    )
    assert submit.status_code == 200
    payload = submit.json()
    jobs_payload = client.get("/api/v1/inverse-design/jobs").json()["jobs"]
    matching = [job for job in jobs_payload if job.get("job_id") == payload["job_id"]]
    assert matching
    submitted = matching[0]["request"]
    assert submitted["geometry_id"] == geometry_id
    assert submitted["world_kwargs"]["geometry_asset_id"] == geometry_id
    assert submitted["world_kwargs"]["stl_path"].endswith(".stl")


def test_studio_director_pack_endpoint(monkeypatch, tmp_path):
    if find_spec("fastapi") is None:
        return
    from fastapi.testclient import TestClient

    def _fake_pack(workspace_root, request):
        return {
            "ok": True,
            "generated_at": "2026-02-15T00:00:00Z",
            "pack_id": "20260215T000000Z_ui_inverse_design",
            "output_dir": str(tmp_path / "release_packets" / "fake"),
            "validation_dir": str(tmp_path / "validation_outputs" / "director_pack" / "fake"),
            "release_packet_manifest": {"ok": True, "copied_artifacts": []},
            "release_packet_manifest_path": str(tmp_path / "release_packets" / "fake" / "release_packet_manifest.json"),
            "release_evidence": {"overall_pass": True},
            "operations": [{"command": ["python3", "scripts/reliability_kernel_smoke.py"], "return_code": 0}],
        }

    monkeypatch.setattr(webapp, "_build_director_pack", _fake_pack)
    monkeypatch.setenv("ATOM_ENABLE_DIRECTOR_PACK", "1")
    app = webapp.create_app(workspace_root=str(tmp_path))
    client = TestClient(app)
    resp = client.post(
        "/api/v1/studio/director-pack",
        json={
            "tag": "ui_inverse_design",
            "run_reliability": True,
            "run_supersonic_validation": True,
            "run_release_evidence": True,
            "allow_missing": True,
            "reliability_steps": 20,
            "reliability_seed": 123,
            "supersonic_steps": 16,
            "supersonic_nx": 64,
            "supersonic_ny": 32,
            "supersonic_seed": 123,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["pack_id"].startswith("20260215")
    assert payload["release_packet_manifest"]["ok"] is True


def test_studio_director_pack_endpoint_disabled_by_default(monkeypatch, tmp_path):
    if find_spec("fastapi") is None:
        return
    from fastapi.testclient import TestClient

    called = {"count": 0}

    def _fake_pack(workspace_root, request):
        called["count"] += 1
        return {"ok": True}

    monkeypatch.delenv("ATOM_ENABLE_DIRECTOR_PACK", raising=False)
    monkeypatch.setattr(webapp, "_build_director_pack", _fake_pack)

    app = webapp.create_app(workspace_root=str(tmp_path))
    client = TestClient(app)
    resp = client.post("/api/v1/studio/director-pack", json={})

    assert resp.status_code == 403
    detail = str(resp.json().get("detail", ""))
    assert "disabled by default" in detail
    assert called["count"] == 0


def test_build_director_pack_rejects_allow_missing_in_release_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOM_RELEASE_PROFILE", "release")
    request = webapp.DirectorPackRequest(allow_missing=True)
    try:
        webapp._build_director_pack(tmp_path, request)
    except ValueError as exc:
        assert "allow_missing is not permitted" in str(exc)
    else:
        raise AssertionError("Expected release profile to reject allow_missing for director pack")


def test_run_local_command_rejects_disallowed_script(tmp_path):
    out = webapp._run_local_command(["python3", "scripts/not_allowed.py"], cwd=tmp_path)
    assert out["return_code"] != 0
    assert "not allowed" in str(out.get("stderr_tail", ""))


def test_build_director_pack_passes_profile_to_release_scripts(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOM_RELEASE_PROFILE", "ci")
    commands = []

    def _fake_run_local_command(args, cwd):
        commands.append([str(a) for a in args])
        return {"command": [str(a) for a in args], "return_code": 0, "duration_sec": 0.0}

    monkeypatch.setattr(webapp, "_run_local_command", _fake_run_local_command)
    request = webapp.DirectorPackRequest(
        run_reliability=False,
        run_supersonic_validation=False,
        run_release_evidence=True,
        allow_missing=False,
    )
    payload = webapp._build_director_pack(tmp_path, request)

    assert payload["profile"] == "ci"
    assert payload["ok"] is True
    evidence_cmds = [cmd for cmd in commands if "scripts/generate_release_evidence.py" in cmd]
    packet_cmds = [cmd for cmd in commands if "scripts/build_release_packet.py" in cmd]
    assert len(evidence_cmds) == 1
    assert len(packet_cmds) == 1
    assert "--profile" in evidence_cmds[0]
    assert evidence_cmds[0][evidence_cmds[0].index("--profile") + 1] == "ci"
    assert "--profile" in packet_cmds[0]
    assert packet_cmds[0][packet_cmds[0].index("--profile") + 1] == "ci"
