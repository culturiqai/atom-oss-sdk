"""ATOM platform API + UI for inverse-design orchestration."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from urllib import error as url_error
from urllib import request as url_request
from urllib.parse import urlparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("atom.platform.webapp")
_MAX_TELEMETRY_POINTS = 5000
_MAX_LAW_EVENTS = 512
_MAX_TIMELINE_EVENTS = 1024
_MAX_INCIDENT_EVENTS = 256
_MAX_BOOKMARK_EVENTS = 256
_MAX_GEOMETRY_UPLOAD_BYTES = 64 * 1024 * 1024
_SUPPORTED_GEOMETRY_EXTENSIONS = {
    ".stl",
    ".obj",
    ".ply",
    ".step",
    ".stp",
    ".iges",
    ".igs",
}
_ALLOWED_DIRECTOR_PACK_SCRIPTS = {
    "scripts/reliability_kernel_smoke.py",
    "scripts/supersonic_validation_pack.py",
    "scripts/generate_release_evidence.py",
    "scripts/build_release_packet.py",
}


class InverseDesignJobRequest(BaseModel):
    """Payload for launching an inverse-design job."""

    name: str = "inverse_design_job"
    world_spec: str = "analytical:taylor_green"
    grid_shape: List[int] = Field(default_factory=lambda: [32, 32, 16], min_length=3, max_length=3)
    backend: Literal["evolutionary", "gradient", "bayesian"] = "evolutionary"
    iterations: Optional[int] = Field(default=None, ge=1)
    population: Optional[int] = Field(default=None, ge=1)
    top_k: int = Field(default=5, ge=1, le=100)
    rollout_steps: int = Field(default=64, ge=4, le=5000)
    objective_spec_path: Optional[str] = None
    parameter_space_path: Optional[str] = None
    geometry_id: Optional[str] = None
    world_kwargs: Dict[str, Any] = Field(default_factory=dict)
    device: str = "cpu"
    report_name: str = "inverse_design_report.json"


class SupersonicChallengeJobRequest(BaseModel):
    """Payload for launching a supersonic wedge challenge run."""

    name: str = "supersonic_wedge_job"
    steps: int = Field(default=256, ge=8, le=20000)
    headless: bool = True
    report_name: str = "challenge_audit.json"


class SupersonicControlRequest(BaseModel):
    """Payload for controlling a running supersonic job."""

    action: Literal["pause", "resume", "cancel", "bookmark"]
    note: Optional[str] = Field(default=None, max_length=400)


class AssistantCitation(BaseModel):
    """Grounding citation for assistant responses."""

    source: str
    field_path: str
    value: str


class AssistantQueryRequest(BaseModel):
    """Payload for grounded assistant queries over platform job data."""

    question: str = Field(min_length=3, max_length=1200)
    mode: Literal["summary", "status", "risk", "next_actions"] = "summary"
    intent: Literal[
        "scientific_discovery",
        "inverse_design",
        "engineering",
        "control",
    ] = "scientific_discovery"
    diagnostic_mode: Literal[
        "overview",
        "live_flow",
        "eyes_saliency",
        "brain_saliency",
        "trust_surface",
    ] = "overview"
    engine: Literal["deterministic", "llm_grounded"] = "deterministic"
    inverse_job_id: Optional[str] = None
    supersonic_job_id: Optional[str] = None
    telemetry_window: int = Field(default=64, ge=8, le=1024)


class AssistantQueryResponse(BaseModel):
    """Deterministic assistant response with grounding evidence."""

    answer: str
    mode: str
    intent: str = "scientific_discovery"
    diagnostic_mode: str = "overview"
    engine: str = "deterministic"
    generated_at: str
    grounded: bool = True
    citations: List[AssistantCitation] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class JobSubmitResponse(BaseModel):
    """Submission response for async jobs."""

    job_id: str
    status: str
    created_at: str


class DirectorPackRequest(BaseModel):
    """Payload for building deterministic demo evidence packets."""

    tag: str = "director_demo"
    run_reliability: bool = True
    run_supersonic_validation: bool = True
    run_release_evidence: bool = True
    allow_missing: bool = True
    reliability_steps: int = Field(default=20, ge=8, le=256)
    reliability_seed: int = Field(default=123, ge=0, le=2**31 - 1)
    supersonic_steps: int = Field(default=16, ge=8, le=512)
    supersonic_nx: int = Field(default=64, ge=16, le=512)
    supersonic_ny: int = Field(default=32, ge=16, le=512)
    supersonic_seed: int = Field(default=123, ge=0, le=2**31 - 1)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(ts: Optional[datetime] = None) -> str:
    return (ts or _utc_now()).isoformat()


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _release_profile() -> str:
    raw = str(os.getenv("ATOM_RELEASE_PROFILE", "dev")).strip().lower()
    return raw if raw in {"dev", "ci", "release"} else "dev"


def _director_pack_enabled() -> bool:
    return _env_flag("ATOM_ENABLE_DIRECTOR_PACK", default=False)


def _require_finite_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _require_int(value: Any, name: str) -> int:
    raw = _require_finite_float(value, name)
    rounded = int(round(raw))
    if not math.isclose(raw, float(rounded), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return rounded


def _resolve_request_json_path(path_str: str, workspace_root: Path, field_name: str) -> Path:
    raw = str(path_str).strip()
    if not raw:
        raise ValueError(f"{field_name} cannot be empty when provided")
    path = Path(raw)
    if not path.is_absolute():
        path = (workspace_root / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise ValueError(f"{field_name} not found at '{path}'")
    if not path.is_file():
        raise ValueError(f"{field_name} must reference a JSON file, got '{path}'")
    return path


def _load_json_object(path: Path, field_name: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{field_name} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must contain a JSON object")
    return payload


def _resolve_existing_file_path(path_str: str, workspace_root: Path, field_name: str) -> Path:
    raw = str(path_str).strip()
    if not raw:
        raise ValueError(f"{field_name} cannot be empty")
    path = Path(raw)
    if not path.is_absolute():
        path = (workspace_root / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise ValueError(f"{field_name} not found at '{path}'")
    if not path.is_file():
        raise ValueError(f"{field_name} must reference a file, got '{path}'")
    return path


def _validate_bounds_mapping(
    raw_bounds: Dict[str, Any],
    *,
    field_name: str,
    allow_empty: bool,
) -> Dict[str, List[float]]:
    if not isinstance(raw_bounds, dict):
        raise ValueError(f"{field_name} must be an object mapping parameter -> [low, high]")
    if not raw_bounds and not allow_empty:
        raise ValueError(f"{field_name} must define at least one parameter bound")

    normalized: Dict[str, List[float]] = {}
    for raw_name, raw_range in raw_bounds.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{field_name} contains an empty parameter name")
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            raise ValueError(f"{field_name}.{name} must be [low, high]")
        low = _require_finite_float(raw_range[0], f"{field_name}.{name}[0]")
        high = _require_finite_float(raw_range[1], f"{field_name}.{name}[1]")
        if low >= high:
            raise ValueError(f"{field_name}.{name} requires low < high, got [{low}, {high}]")
        normalized[name] = [float(low), float(high)]
    return normalized


def _validate_objective_constraints(raw_constraints: Dict[str, Any], field_name: str) -> None:
    if not raw_constraints:
        return
    from atom.platform.contracts import ObjectiveSpec
    from atom.platform.inverse_design import ObjectiveDSL

    try:
        ObjectiveDSL.compile_constraints(
            ObjectiveSpec(
                targets={},
                constraints=dict(raw_constraints),
                penalties={},
                hard_bounds={},
                solver_budget={},
            )
        )
    except Exception as exc:
        raise ValueError(f"{field_name} contains invalid constraint clauses: {exc}") from exc


def _validate_objective_spec_payload(raw_payload: Dict[str, Any], field_name: str) -> Dict[str, Any]:
    if not isinstance(raw_payload, dict):
        raise ValueError(f"{field_name} must contain a JSON object")

    allowed_fields = {"targets", "constraints", "penalties", "hard_bounds", "solver_budget"}
    unknown_fields = sorted(set(raw_payload.keys()) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"{field_name} has unsupported fields: {unknown_fields}")

    raw_targets = raw_payload.get("targets", {})
    if raw_targets is None:
        raw_targets = {}
    if not isinstance(raw_targets, dict):
        raise ValueError(f"{field_name}.targets must be an object")
    targets = {
        str(k): _require_finite_float(v, f"{field_name}.targets.{k}")
        for k, v in raw_targets.items()
    }

    raw_penalties = raw_payload.get("penalties", {})
    if raw_penalties is None:
        raw_penalties = {}
    if not isinstance(raw_penalties, dict):
        raise ValueError(f"{field_name}.penalties must be an object")
    penalties = {
        str(k): _require_finite_float(v, f"{field_name}.penalties.{k}")
        for k, v in raw_penalties.items()
    }

    raw_constraints = raw_payload.get("constraints", {})
    if raw_constraints is None:
        raw_constraints = {}
    if not isinstance(raw_constraints, dict):
        raise ValueError(f"{field_name}.constraints must be an object")
    constraints = dict(raw_constraints)
    _validate_objective_constraints(constraints, f"{field_name}.constraints")

    raw_hard_bounds = raw_payload.get("hard_bounds", {})
    if raw_hard_bounds is None:
        raw_hard_bounds = {}
    hard_bounds = _validate_bounds_mapping(
        dict(raw_hard_bounds),
        field_name=f"{field_name}.hard_bounds",
        allow_empty=True,
    )

    raw_solver_budget = raw_payload.get("solver_budget", {})
    if raw_solver_budget is None:
        raw_solver_budget = {}
    if not isinstance(raw_solver_budget, dict):
        raise ValueError(f"{field_name}.solver_budget must be an object")
    allowed_budget = {
        "iterations",
        "population",
        "seed",
        "mutation_scale",
        "learning_rate",
        "bayes_beta",
        "bayes_length_scale",
    }
    unknown_budget = sorted(set(raw_solver_budget.keys()) - allowed_budget)
    if unknown_budget:
        raise ValueError(f"{field_name}.solver_budget has unsupported fields: {unknown_budget}")

    solver_budget: Dict[str, Any] = {}
    if "iterations" in raw_solver_budget:
        iterations = int(raw_solver_budget["iterations"])
        if iterations < 1:
            raise ValueError(f"{field_name}.solver_budget.iterations must be >= 1")
        solver_budget["iterations"] = iterations
    if "population" in raw_solver_budget:
        population = int(raw_solver_budget["population"])
        if population < 1:
            raise ValueError(f"{field_name}.solver_budget.population must be >= 1")
        solver_budget["population"] = population
    if "seed" in raw_solver_budget:
        solver_budget["seed"] = int(raw_solver_budget["seed"])
    for key in ("mutation_scale", "learning_rate", "bayes_beta", "bayes_length_scale"):
        if key in raw_solver_budget:
            value = _require_finite_float(raw_solver_budget[key], f"{field_name}.solver_budget.{key}")
            if value <= 0.0:
                raise ValueError(f"{field_name}.solver_budget.{key} must be > 0")
            solver_budget[key] = value

    if not targets and not penalties and not constraints:
        raise ValueError(
            f"{field_name} must define at least one of targets, penalties, or constraints"
        )

    return {
        "targets": targets,
        "constraints": constraints,
        "penalties": penalties,
        "hard_bounds": hard_bounds,
        "solver_budget": solver_budget,
    }


def _validate_inverse_contract_paths(
    request: InverseDesignJobRequest,
    workspace_root: Path,
) -> Dict[str, Optional[str]]:
    resolved_parameter_space: Optional[str] = None
    resolved_objective: Optional[str] = None
    parameter_bounds: Dict[str, List[float]] = {}
    objective_payload: Dict[str, Any] = {}

    if request.parameter_space_path:
        param_path = _resolve_request_json_path(
            request.parameter_space_path,
            workspace_root,
            "parameter_space_path",
        )
        parameter_bounds = _validate_bounds_mapping(
            _load_json_object(param_path, "parameter_space_path"),
            field_name="parameter_space_path",
            allow_empty=False,
        )
        resolved_parameter_space = str(param_path)

    if request.objective_spec_path:
        objective_path = _resolve_request_json_path(
            request.objective_spec_path,
            workspace_root,
            "objective_spec_path",
        )
        objective_payload = _validate_objective_spec_payload(
            _load_json_object(objective_path, "objective_spec_path"),
            "objective_spec_path",
        )
        resolved_objective = str(objective_path)

    if parameter_bounds and objective_payload.get("hard_bounds"):
        hard_bound_keys = set(objective_payload["hard_bounds"].keys())
        parameter_keys = set(parameter_bounds.keys())
        unknown = sorted(hard_bound_keys - parameter_keys)
        if unknown:
            raise ValueError(
                "objective_spec_path.hard_bounds includes keys not present in "
                f"parameter_space_path: {unknown}"
            )

    return {
        "objective_spec_path": resolved_objective,
        "parameter_space_path": resolved_parameter_space,
    }


def _validate_world_spec(world_spec: str) -> str:
    key = str(world_spec).strip().lower()
    if not key:
        raise ValueError("world_spec cannot be empty")
    if ":" not in key:
        raise ValueError(
            "world_spec must use '<backend>:<scenario>' format "
            "(e.g. analytical:taylor_green)"
        )
    backend, scenario = key.split(":", 1)
    if backend not in {"analytical", "lbm", "lbm2d", "supersonic"}:
        raise ValueError(
            "Unsupported world_spec backend. Supported: analytical, lbm, lbm2d, supersonic."
        )
    if backend == "analytical" and scenario not in {
        "taylor_green",
        "burgers_shock",
        "kelvin_helmholtz",
    }:
        raise ValueError(
            "Unsupported analytical scenario. Supported: "
            "taylor_green, burgers_shock, kelvin_helmholtz."
        )
    if backend == "lbm" and scenario not in {"fluid", "cylinder", "custom"}:
        raise ValueError("Unsupported lbm scenario. Supported: fluid, cylinder, custom.")
    if backend == "lbm2d" and scenario not in {"cylinder"}:
        raise ValueError("Unsupported lbm2d scenario. Supported: cylinder.")
    if backend == "supersonic" and scenario not in {"wedge_d2q25", "wedge"}:
        raise ValueError("Unsupported supersonic scenario. Supported: wedge_d2q25, wedge.")
    return key


def _validate_world_kwargs(
    *,
    world_spec: str,
    world_kwargs: Dict[str, Any],
    workspace_root: Path,
) -> Dict[str, Any]:
    if not isinstance(world_kwargs, dict):
        raise ValueError("world_kwargs must be an object")
    kwargs = {str(k): v for k, v in world_kwargs.items()}
    key = _validate_world_spec(world_spec)
    backend, scenario = key.split(":", 1)

    metadata_keys = {"geometry_asset_id", "geometry_asset_path"}
    normalized: Dict[str, Any] = {}
    for meta_key in metadata_keys:
        if meta_key in kwargs:
            normalized[meta_key] = str(kwargs[meta_key])

    allowed: set[str] = set(metadata_keys)

    def _unknown_keys_error() -> None:
        unknown = sorted(set(kwargs.keys()) - allowed)
        if unknown:
            raise ValueError(
                f"world_kwargs has unsupported fields for '{key}': {unknown}"
            )

    if backend == "analytical":
        allowed.update({"dt", "batch_size"})
        if "dt" in kwargs:
            dt = _require_finite_float(kwargs["dt"], "world_kwargs.dt")
            if dt <= 0.0:
                raise ValueError("world_kwargs.dt must be > 0")
            normalized["dt"] = dt
        if "batch_size" in kwargs:
            batch_size = _require_int(kwargs["batch_size"], "world_kwargs.batch_size")
            if batch_size < 1:
                raise ValueError("world_kwargs.batch_size must be >= 1")
            normalized["batch_size"] = batch_size
        _unknown_keys_error()
        return normalized

    if backend == "lbm2d":
        allowed.update({"u_inlet", "tau", "batch_size"})
        if "u_inlet" in kwargs:
            u_inlet = _require_finite_float(kwargs["u_inlet"], "world_kwargs.u_inlet")
            if u_inlet <= 0.0:
                raise ValueError("world_kwargs.u_inlet must be > 0")
            normalized["u_inlet"] = u_inlet
        if "tau" in kwargs:
            tau = _require_finite_float(kwargs["tau"], "world_kwargs.tau")
            if tau <= 0.5:
                raise ValueError("world_kwargs.tau must be > 0.5")
            normalized["tau"] = tau
        if "batch_size" in kwargs:
            batch_size = _require_int(kwargs["batch_size"], "world_kwargs.batch_size")
            if batch_size < 1:
                raise ValueError("world_kwargs.batch_size must be >= 1")
            normalized["batch_size"] = batch_size
        _unknown_keys_error()
        return normalized

    if backend == "lbm":
        allowed.update({"batch_size"})
        if scenario == "custom":
            allowed.add("stl_path")
            stl_path_raw = kwargs.get("stl_path")
            if stl_path_raw is None:
                raise ValueError("lbm:custom requires world_kwargs.stl_path")
            stl_path = _resolve_existing_file_path(
                str(stl_path_raw), workspace_root, "world_kwargs.stl_path"
            )
            if stl_path.suffix.lower() != ".stl":
                raise ValueError(
                    f"world_kwargs.stl_path must reference an STL file, got '{stl_path.suffix}'."
                )
            normalized["stl_path"] = str(stl_path)
        if "batch_size" in kwargs:
            batch_size = _require_int(kwargs["batch_size"], "world_kwargs.batch_size")
            if batch_size < 1:
                raise ValueError("world_kwargs.batch_size must be >= 1")
            normalized["batch_size"] = batch_size
        _unknown_keys_error()
        return normalized

    if backend == "supersonic":
        allowed.update(
            {
                "tau",
                "inflow_velocity",
                "noise_amp",
                "warmup_steps",
                "max_steps",
                "episode_steps",
                "wedge_start_x",
                "wedge_length",
                "wedge_half_angle_deg",
                "jet_gain",
                "jet_radius",
                "reward_control_penalty",
                "instability_density_cap",
                "seed",
            }
        )
        if "tau" in kwargs:
            tau = _require_finite_float(kwargs["tau"], "world_kwargs.tau")
            if tau <= 0.5:
                raise ValueError("world_kwargs.tau must be > 0.5")
            normalized["tau"] = tau
        if "inflow_velocity" in kwargs:
            inflow_velocity = _require_finite_float(
                kwargs["inflow_velocity"], "world_kwargs.inflow_velocity"
            )
            if inflow_velocity <= 0.0:
                raise ValueError("world_kwargs.inflow_velocity must be > 0")
            normalized["inflow_velocity"] = inflow_velocity
        if "noise_amp" in kwargs:
            noise_amp = _require_finite_float(kwargs["noise_amp"], "world_kwargs.noise_amp")
            if noise_amp < 0.0:
                raise ValueError("world_kwargs.noise_amp must be >= 0")
            normalized["noise_amp"] = noise_amp
        if "warmup_steps" in kwargs:
            warmup_steps = _require_int(kwargs["warmup_steps"], "world_kwargs.warmup_steps")
            if warmup_steps < 0:
                raise ValueError("world_kwargs.warmup_steps must be >= 0")
            normalized["warmup_steps"] = warmup_steps
        if "max_steps" in kwargs:
            max_steps = _require_int(kwargs["max_steps"], "world_kwargs.max_steps")
            if max_steps < 1:
                raise ValueError("world_kwargs.max_steps must be >= 1")
            normalized["max_steps"] = max_steps
        if "episode_steps" in kwargs:
            episode_steps = _require_int(kwargs["episode_steps"], "world_kwargs.episode_steps")
            if episode_steps < 1:
                raise ValueError("world_kwargs.episode_steps must be >= 1")
            normalized["episode_steps"] = episode_steps
        if "wedge_start_x" in kwargs:
            wedge_start_x = _require_int(kwargs["wedge_start_x"], "world_kwargs.wedge_start_x")
            if wedge_start_x < 1:
                raise ValueError("world_kwargs.wedge_start_x must be >= 1")
            normalized["wedge_start_x"] = wedge_start_x
        if "wedge_length" in kwargs:
            wedge_length = _require_int(kwargs["wedge_length"], "world_kwargs.wedge_length")
            if wedge_length < 1:
                raise ValueError("world_kwargs.wedge_length must be >= 1")
            normalized["wedge_length"] = wedge_length
        if "wedge_half_angle_deg" in kwargs:
            wedge_half_angle_deg = _require_finite_float(
                kwargs["wedge_half_angle_deg"], "world_kwargs.wedge_half_angle_deg"
            )
            if wedge_half_angle_deg <= 0.0 or wedge_half_angle_deg >= 80.0:
                raise ValueError("world_kwargs.wedge_half_angle_deg must be in (0, 80)")
            normalized["wedge_half_angle_deg"] = wedge_half_angle_deg
        if "jet_gain" in kwargs:
            jet_gain = _require_finite_float(kwargs["jet_gain"], "world_kwargs.jet_gain")
            if jet_gain < 0.0:
                raise ValueError("world_kwargs.jet_gain must be >= 0")
            normalized["jet_gain"] = jet_gain
        if "jet_radius" in kwargs:
            jet_radius = _require_finite_float(kwargs["jet_radius"], "world_kwargs.jet_radius")
            if jet_radius <= 0.0:
                raise ValueError("world_kwargs.jet_radius must be > 0")
            normalized["jet_radius"] = jet_radius
        if "reward_control_penalty" in kwargs:
            reward_control_penalty = _require_finite_float(
                kwargs["reward_control_penalty"], "world_kwargs.reward_control_penalty"
            )
            if reward_control_penalty < 0.0:
                raise ValueError("world_kwargs.reward_control_penalty must be >= 0")
            normalized["reward_control_penalty"] = reward_control_penalty
        if "instability_density_cap" in kwargs:
            instability_density_cap = _require_finite_float(
                kwargs["instability_density_cap"], "world_kwargs.instability_density_cap"
            )
            if instability_density_cap <= 0.0:
                raise ValueError("world_kwargs.instability_density_cap must be > 0")
            normalized["instability_density_cap"] = instability_density_cap
        if "seed" in kwargs:
            normalized["seed"] = _require_int(kwargs["seed"], "world_kwargs.seed")
        _unknown_keys_error()
        return normalized

    _unknown_keys_error()
    return normalized


def _world_kwargs_contract_template() -> Dict[str, Any]:
    return {
        "validation_policy": {
            "strict_unknown_keys": True,
            "common_optional_fields": ["geometry_asset_id", "geometry_asset_path"],
        },
        "profiles": {
            "analytical": {
                "applies_to": [
                    "analytical:taylor_green",
                    "analytical:burgers_shock",
                    "analytical:kelvin_helmholtz",
                ],
                "fields": {
                    "dt": {"type": "float", "required": False, "exclusive_min": 0.0},
                    "batch_size": {"type": "int", "required": False, "min": 1},
                },
            },
            "lbm_3d": {
                "applies_to": ["lbm:fluid", "lbm:cylinder"],
                "fields": {
                    "batch_size": {"type": "int", "required": False, "min": 1},
                },
            },
            "lbm_3d_custom": {
                "applies_to": ["lbm:custom"],
                "fields": {
                    "stl_path": {
                        "type": "path",
                        "required": True,
                        "must_exist": True,
                        "suffix": ".stl",
                    },
                    "batch_size": {"type": "int", "required": False, "min": 1},
                },
            },
            "lbm_2d": {
                "applies_to": ["lbm2d:cylinder"],
                "fields": {
                    "u_inlet": {"type": "float", "required": False, "exclusive_min": 0.0},
                    "tau": {"type": "float", "required": False, "exclusive_min": 0.5},
                    "batch_size": {"type": "int", "required": False, "min": 1},
                },
            },
            "supersonic": {
                "applies_to": ["supersonic:wedge_d2q25", "supersonic:wedge"],
                "fields": {
                    "tau": {"type": "float", "required": False, "exclusive_min": 0.5},
                    "inflow_velocity": {"type": "float", "required": False, "exclusive_min": 0.0},
                    "noise_amp": {"type": "float", "required": False, "min": 0.0},
                    "warmup_steps": {"type": "int", "required": False, "min": 0},
                    "max_steps": {"type": "int", "required": False, "min": 1},
                    "episode_steps": {"type": "int", "required": False, "min": 1},
                    "wedge_start_x": {"type": "int", "required": False, "min": 1},
                    "wedge_length": {"type": "int", "required": False, "min": 1},
                    "wedge_half_angle_deg": {
                        "type": "float",
                        "required": False,
                        "exclusive_min": 0.0,
                        "exclusive_max": 80.0,
                    },
                    "jet_gain": {"type": "float", "required": False, "min": 0.0},
                    "jet_radius": {"type": "float", "required": False, "exclusive_min": 0.0},
                    "reward_control_penalty": {"type": "float", "required": False, "min": 0.0},
                    "instability_density_cap": {
                        "type": "float",
                        "required": False,
                        "exclusive_min": 0.0,
                    },
                    "seed": {"type": "int", "required": False},
                },
            },
        },
        "world_spec_profile_map": {
            "analytical:taylor_green": "analytical",
            "analytical:burgers_shock": "analytical",
            "analytical:kelvin_helmholtz": "analytical",
            "lbm:fluid": "lbm_3d",
            "lbm:cylinder": "lbm_3d",
            "lbm:custom": "lbm_3d_custom",
            "lbm2d:cylinder": "lbm_2d",
            "supersonic:wedge_d2q25": "supersonic",
            "supersonic:wedge": "supersonic",
        },
        "examples": {
            "analytical:taylor_green": {"dt": 0.1, "batch_size": 1},
            "lbm2d:cylinder": {"u_inlet": 0.05, "tau": 0.56},
            "lbm:custom": {"stl_path": "assets/geometries/wing.stl", "batch_size": 1},
            "supersonic:wedge_d2q25": {
                "tau": 0.8,
                "inflow_velocity": 0.16,
                "warmup_steps": 20,
                "wedge_half_angle_deg": 14.0,
                "jet_gain": 0.012,
                "reward_control_penalty": 0.05,
                "seed": 42,
            },
        },
    }


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _slope(values: List[float], window: int = 32) -> float:
    if len(values) < 2:
        return 0.0
    n = max(2, min(window, len(values)))
    tail = values[-n:]
    return float((tail[-1] - tail[0]) / float(max(1, n - 1)))


def _align_series(values: List[float], target_len: int) -> List[float]:
    if target_len <= 0:
        return []
    if len(values) >= target_len:
        return values[-target_len:]
    return [0.0] * (target_len - len(values)) + values


def _resolve_runner_module() -> Any:
    """Load runner module even when launched as installed package."""
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        import atom_experiment_runner as runner
    except Exception as exc:  # pragma: no cover - runtime environment issue
        raise RuntimeError(
            "Unable to import atom_experiment_runner.py. "
            "Run from repository root or ensure project root is on PYTHONPATH."
        ) from exc
    return runner


def _resolve_challenge_module() -> Any:
    """Load supersonic challenge module for API-exposed challenge jobs."""
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        import atom.challenges.supersonic_wedge_challenge as challenge
    except Exception as exc:  # pragma: no cover - runtime environment issue
        raise RuntimeError(
            "Unable to import supersonic challenge module. "
            "Ensure project root is on PYTHONPATH."
        ) from exc
    return challenge


def _load_world_registry() -> Dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from atom_worlds import list_available_worlds

        return dict(list_available_worlds())
    except Exception:
        # Safe fallback for API availability when world registry import fails.
        return {"analytical:taylor_green": "Analytical fallback world"}


def _simulator_family(world_key: str) -> str:
    key = str(world_key).lower()
    if key.startswith("analytical:"):
        return "analytical"
    if key.startswith("lbm2d:"):
        return "lbm_2d"
    if key.startswith("lbm:"):
        return "lbm_3d"
    if key.startswith("supersonic:"):
        return "supersonic_control"
    return "custom"


def _simulator_modalities(world_key: str) -> Dict[str, str]:
    key = str(world_key).lower()
    if key.startswith("lbm2d:"):
        return {
            "dimensions": "2D",
            "control_surface": "jet_actuation",
            "research_fit": "vortex_shedding_and_flow_control",
        }
    if key.startswith("lbm:"):
        return {
            "dimensions": "3D",
            "control_surface": "jet_or_body_force",
            "research_fit": "full_flowfield_control_and_inverse_design",
        }
    if key.startswith("supersonic:"):
        return {
            "dimensions": "2D_lifted",
            "control_surface": "wedge_shock_jet",
            "research_fit": "high_speed_flow_control_and_runtime_assurance",
        }
    if key.startswith("analytical:"):
        return {
            "dimensions": "3D",
            "control_surface": "synthetic_action_field",
            "research_fit": "algorithm_validation_and_deterministic_replay",
        }
    return {
        "dimensions": "unknown",
        "control_surface": "custom",
        "research_fit": "adapter_defined",
    }


def _build_simulator_catalog(world_registry: Dict[str, str]) -> List[Dict[str, Any]]:
    simulators: List[Dict[str, Any]] = []
    for key, description in sorted(world_registry.items(), key=lambda item: item[0]):
        modalities = _simulator_modalities(key)
        simulators.append(
            {
                "key": str(key),
                "description": str(description),
                "family": _simulator_family(key),
                "dimensions": modalities["dimensions"],
                "control_surface": modalities["control_surface"],
                "research_fit": modalities["research_fit"],
            }
        )
    return simulators


def _normalize_token(raw: str, *, fallback: str = "item", max_len: int = 64) -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(raw or "").strip())
    token = token.strip("._-")
    if not token:
        token = fallback
    return token[:max_len]


def _json_safe_value(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe_value(v, depth + 1) for v in value[:64]]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= 64:
                break
            out[str(k)] = _json_safe_value(v, depth + 1)
        return out
    return str(value)


def _is_local_base_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0"}


class GeometryAssetStore:
    """Persist and resolve uploaded geometry assets for custom-world runs."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root.resolve()
        self.root = (self.workspace_root / "assets" / "geometries").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, geometry_id: str) -> Path:
        token = _normalize_token(geometry_id, fallback="geometry")
        return self.root / f"{token}.json"

    def _read_meta(self, geometry_id: str) -> Optional[Dict[str, Any]]:
        path = self._meta_path(geometry_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        asset_path = Path(str(payload.get("path", "")))
        if not asset_path.exists():
            return None
        return payload

    def list_assets(self) -> List[Dict[str, Any]]:
        assets: List[Dict[str, Any]] = []
        for meta_path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            asset_path = Path(str(payload.get("path", "")))
            if not asset_path.exists():
                continue
            payload["bytes"] = int(payload.get("bytes", asset_path.stat().st_size))
            payload["is_stl"] = str(asset_path.suffix).lower() == ".stl"
            assets.append(payload)
        return assets

    def resolve(self, geometry_id: str) -> Optional[Dict[str, Any]]:
        payload = self._read_meta(geometry_id)
        if payload is None:
            return None
        asset_path = Path(str(payload.get("path", "")))
        payload["bytes"] = int(payload.get("bytes", asset_path.stat().st_size))
        payload["is_stl"] = str(asset_path.suffix).lower() == ".stl"
        return payload

    def save_upload(self, filename: str, payload: bytes) -> Dict[str, Any]:
        raw_name = str(filename or "").strip()
        if not raw_name:
            raise ValueError("Missing filename.")
        suffix = str(Path(raw_name).suffix).lower()
        if suffix not in _SUPPORTED_GEOMETRY_EXTENSIONS:
            supported = ", ".join(sorted(_SUPPORTED_GEOMETRY_EXTENSIONS))
            raise ValueError(f"Unsupported geometry extension '{suffix}'. Supported: {supported}")

        stem = _normalize_token(Path(raw_name).stem, fallback="geometry")
        geometry_id = f"geo_{_utc_now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        stored_name = f"{geometry_id}_{stem}{suffix}"
        out_path = (self.root / stored_name).resolve()
        out_path.write_bytes(payload)

        record = {
            "geometry_id": geometry_id,
            "filename": raw_name,
            "stored_name": stored_name,
            "path": str(out_path),
            "bytes": int(len(payload)),
            "extension": suffix,
            "is_stl": suffix == ".stl",
            "created_at": _utc_iso(),
        }
        self._meta_path(geometry_id).write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record


def _fallback_studio_examples(world_registry: Dict[str, str]) -> List[Dict[str, Any]]:
    demos: List[Dict[str, Any]] = []
    demos.append(
        {
            "id": "inverse_taylor_green_evolutionary",
            "title": "Taylor-Green Inverse Design Baseline",
            "kind": "inverse_design",
            "description": (
                "Deterministic analytical benchmark for validating objective and "
                "constraint plumbing before expensive simulators."
            ),
            "simulator": "analytical:taylor_green",
            "available": "analytical:taylor_green" in world_registry,
            "availability_reason": "ok",
            "tags": ["deterministic", "baseline", "inverse_design"],
            "payload": {
                "name": "demo_taylor_green_inverse",
                "world_spec": "analytical:taylor_green",
                "grid_shape": [32, 32, 16],
                "backend": "evolutionary",
                "iterations": 8,
                "population": 12,
                "top_k": 5,
                "rollout_steps": 96,
                "device": "cpu",
            },
            "launch_endpoint": "/api/v1/inverse-design/jobs",
        }
    )
    return demos


def _load_studio_examples(
    workspace_root: Path,
    world_registry: Dict[str, str],
) -> Dict[str, Any]:
    examples_dir = (workspace_root / "examples" / "studio").resolve()
    demos: List[Dict[str, Any]] = []

    challenge_available = True
    try:
        _resolve_challenge_module()
    except Exception:
        challenge_available = False

    if examples_dir.exists():
        for path in sorted(examples_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            kind = str(payload.get("kind", "")).strip().lower()
            if kind not in {"inverse_design", "supersonic_challenge"}:
                continue

            launch_payload = payload.get("payload")
            if not isinstance(launch_payload, dict):
                continue

            demo_id = str(payload.get("id", "")).strip() or path.stem
            title = str(payload.get("title", "")).strip() or demo_id
            description = str(payload.get("description", "")).strip()
            simulator = str(
                payload.get("simulator", launch_payload.get("world_spec", ""))
            ).strip()
            tags_raw = payload.get("tags", [])
            tags: List[str] = []
            if isinstance(tags_raw, list):
                for item in tags_raw:
                    token = str(item).strip()
                    if token:
                        tags.append(token)

            available = True
            availability_reason = "ok"
            if kind == "inverse_design":
                world_spec = str(launch_payload.get("world_spec", simulator)).strip()
                simulator = world_spec
                if world_spec not in world_registry:
                    available = False
                    availability_reason = f"world_unavailable:{world_spec or 'missing'}"
            else:
                if not challenge_available:
                    available = False
                    availability_reason = "challenge_module_unavailable"

            demos.append(
                {
                    "id": demo_id,
                    "title": title,
                    "kind": kind,
                    "description": description,
                    "simulator": simulator,
                    "available": bool(available),
                    "availability_reason": availability_reason,
                    "tags": tags,
                    "payload": launch_payload,
                    "launch_endpoint": (
                        "/api/v1/inverse-design/jobs"
                        if kind == "inverse_design"
                        else "/api/v1/challenges/supersonic/jobs"
                    ),
                }
            )

    if not demos:
        demos = _fallback_studio_examples(world_registry)

    demos.sort(key=lambda item: (0 if item.get("available") else 1, str(item.get("id", ""))))
    return {
        "examples_dir": str(examples_dir),
        "challenge_available": challenge_available,
        "demos": demos,
    }


def _load_runtime_telemetry_snapshot(workspace_root: Path) -> Dict[str, Any]:
    logs_dir = workspace_root / "logs"
    telemetry_path = logs_dir / "telemetry.json"
    hypotheses_path = logs_dir / "hypotheses.jsonl"

    payload: Optional[Dict[str, Any]] = None
    if telemetry_path.exists():
        try:
            raw = json.loads(telemetry_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload = raw
        except Exception:
            payload = None

    if payload is None:
        return {
            "available": False,
            "status": "no_runtime_telemetry",
            "telemetry_path": str(telemetry_path),
        }

    theory_packet_any = payload.get("theory_packet", {}) if isinstance(payload, dict) else {}
    theory_packet = theory_packet_any if isinstance(theory_packet_any, dict) else {}
    diagnostics_any = payload.get("diagnostics", {}) if isinstance(payload, dict) else {}
    diagnostics = diagnostics_any if isinstance(diagnostics_any, dict) else {}

    def _sanitize_map_block(raw: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        values = raw.get("map_xy", [])
        if not isinstance(values, list):
            return None
        rows: List[List[float]] = []
        max_rows = 192
        for row in values[:max_rows]:
            if not isinstance(row, list):
                continue
            row_vals = [_coerce_float(v) for v in row[:max_rows]]
            rows.append(row_vals)
        if not rows:
            return None
        return {
            "map_xy": rows,
            "shape": [int(len(rows)), int(len(rows[0]))],
            "stride": _coerce_int(raw.get("stride", 1), default=1),
            "normalized": bool(raw.get("normalized", True)),
            "raw_min": _coerce_float(raw.get("raw_min", 0.0)),
            "raw_max": _coerce_float(raw.get("raw_max", 0.0)),
        }

    live_view_raw = diagnostics.get("live_view", {})
    live_view = live_view_raw if isinstance(live_view_raw, dict) else {}
    live_speed = _sanitize_map_block(live_view.get("speed_xy"))
    live_density = _sanitize_map_block(live_view.get("density_xy"))
    live_divergence = _sanitize_map_block(live_view.get("divergence_xy"))
    live_obstacle = _sanitize_map_block(live_view.get("obstacle_xy"))

    eyes_raw = diagnostics.get("eyes2_saliency", {})
    eyes = eyes_raw if isinstance(eyes_raw, dict) else {}
    eyes_map = _sanitize_map_block(eyes.get("map"))

    brain_raw = diagnostics.get("brain_saliency", {})
    brain = brain_raw if isinstance(brain_raw, dict) else {}
    brain_map = _sanitize_map_block(brain.get("map"))
    brain_importance_raw = brain.get("theory_feature_importance", [])
    if not isinstance(brain_importance_raw, list):
        brain_importance_raw = []
    brain_importance = [_coerce_float(v) for v in brain_importance_raw]
    brain_labels_raw = brain.get("theory_feature_labels", [])
    if not isinstance(brain_labels_raw, list):
        brain_labels_raw = []
    brain_labels = [str(v) for v in brain_labels_raw]

    jacobian_raw = theory_packet.get("jacobian", [])
    if not isinstance(jacobian_raw, list):
        jacobian_raw = []
    jacobian = [_coerce_float(v) for v in jacobian_raw]
    saliency = [abs(v) for v in jacobian]
    saliency_norm = max(saliency) if saliency else 0.0
    if saliency_norm > 1e-12:
        saliency = [float(v / saliency_norm) for v in saliency]

    top_saliency = sorted(
        [
            {
                "latent_index": int(i),
                "importance": float(v),
                "signed_jacobian": float(jacobian[i]),
            }
            for i, v in enumerate(saliency)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[:3]

    hypotheses_tail: List[Dict[str, Any]] = []
    if hypotheses_path.exists():
        try:
            lines = hypotheses_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-5:]:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    hypotheses_tail.append(parsed)
        except Exception:
            hypotheses_tail = []

    governance_raw = payload.get("governance_decision", {})
    governance = governance_raw if isinstance(governance_raw, dict) else {}
    reasons_raw = governance.get("reasons", [])
    reasons = [str(v) for v in reasons_raw] if isinstance(reasons_raw, list) else []
    approved_raw = governance.get("approved")
    if isinstance(approved_raw, bool):
        governance_approved: Optional[bool] = approved_raw
    elif isinstance(approved_raw, (int, float)):
        governance_approved = bool(int(approved_raw))
    else:
        governance_approved = None
    governance_decision = {
        "approved": governance_approved,
        "novelty_score": _coerce_float(governance.get("novelty_score", 0.0)),
        "stability_score": _coerce_float(governance.get("stability_score", 0.0)),
        "intervention_consistency": _coerce_float(
            governance.get("intervention_consistency", 0.0)
        ),
        "same_support_refinement": bool(governance.get("same_support_refinement", False)),
        "reasons": reasons,
        "null_false_discovery_rate": (
            _coerce_float(governance.get("null_false_discovery_rate", 0.0))
            if governance.get("null_false_discovery_rate") is not None
            else None
        ),
        "seed_perturbation_stability": (
            _coerce_float(governance.get("seed_perturbation_stability", 0.0))
            if governance.get("seed_perturbation_stability") is not None
            else None
        ),
    }

    verifier_raw = payload.get("verifier_metrics", {})
    verifier = verifier_raw if isinstance(verifier_raw, dict) else {}
    verifier_metrics = {
        "rmse": _coerce_float(verifier.get("rmse", 0.0)),
        "correlation": _coerce_float(verifier.get("correlation", 0.0)),
        "interventional": _coerce_float(verifier.get("interventional", 0.0)),
    }

    baseline_raw = payload.get("baseline_context", {})
    baseline = baseline_raw if isinstance(baseline_raw, dict) else {}
    baseline_context = {
        "ready": bool(baseline.get("ready", False)),
        "slope": _coerce_float(baseline.get("slope", 0.0)),
        "r2": _coerce_float(baseline.get("r2", 0.0)),
        "discovery_target_mode": str(baseline.get("discovery_target_mode", "hybrid")),
        "discovery_hybrid_alpha": _coerce_float(
            baseline.get("discovery_hybrid_alpha", 0.0)
        ),
        "discovery_target_value": _coerce_float(
            baseline.get("discovery_target_value", 0.0)
        ),
    }

    return {
        "available": True,
        "status": "ok",
        "step": _coerce_int(payload.get("step")),
        "reward": _coerce_float(payload.get("reward")),
        "stress": _coerce_float(payload.get("stress")),
        "theory": _coerce_float(payload.get("theory")),
        "theory_packet": theory_packet if theory_packet else None,
        "trust_raw": _coerce_float(theory_packet.get("trust_raw")),
        "trust_verified": _coerce_float(theory_packet.get("trust_verified")),
        "trust_structural_floor": _coerce_float(theory_packet.get("trust_structural_floor")),
        "hypothesis_record": payload.get("hypothesis_record"),
        "governance_decision": governance_decision,
        "verifier_metrics": verifier_metrics,
        "baseline_context": baseline_context,
        "experiment_plan": payload.get("experiment_plan"),
        "safety_summary": payload.get("safety_summary"),
        "diagnostics": {
            "enabled": bool(diagnostics.get("enabled", False)),
            "updated_step": _coerce_int(diagnostics.get("updated_step", 0)),
            "stale_steps": _coerce_int(diagnostics.get("stale_steps", 0)),
            "interval": _coerce_int(diagnostics.get("interval", 0)),
            "live_view": {
                "projection": str(live_view.get("projection", "xy_mean")),
                "speed_xy": live_speed,
                "density_xy": live_density,
                "divergence_xy": live_divergence,
                "obstacle_xy": live_obstacle,
            },
            "eyes2_saliency": {
                "method": str(eyes.get("method", "")),
                "objective": _coerce_float(eyes.get("objective", 0.0)),
                "projection": str(eyes.get("projection", "xy_mean")),
                "target_label": str(eyes.get("target_label", "")),
                "target_index": (
                    _coerce_int(eyes.get("target_index"), default=-1)
                    if eyes.get("target_index") is not None
                    else None
                ),
                "map": eyes_map,
            },
            "brain_saliency": {
                "method": str(brain.get("method", "")),
                "objective": _coerce_float(brain.get("objective", 0.0)),
                "projection": str(brain.get("projection", "xy_mean")),
                "map": brain_map,
                "theory_feature_importance": brain_importance,
                "theory_feature_labels": brain_labels,
            },
        },
        "saliency": {
            "jacobian_abs_normalized": saliency,
            "top_latents": top_saliency,
            "theory_feature_importance": brain_importance,
            "theory_feature_labels": brain_labels,
        },
        "hypotheses_tail": hypotheses_tail,
        "telemetry_path": str(telemetry_path),
        "hypotheses_path": str(hypotheses_path),
    }


def _truncate_text(raw: str, limit: int = 4000) -> str:
    text = str(raw or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated]"


def _run_local_command(args: List[str], cwd: Path) -> Dict[str, Any]:
    if not args:
        return {
            "command": [],
            "return_code": 1,
            "duration_sec": 0.0,
            "stdout_tail": "",
            "stderr_tail": "command args cannot be empty",
        }

    if len(args) >= 2:
        script_token = Path(str(args[1])).as_posix()
        if script_token not in _ALLOWED_DIRECTOR_PACK_SCRIPTS:
            return {
                "command": [str(a) for a in args],
                "return_code": 1,
                "duration_sec": 0.0,
                "stdout_tail": "",
                "stderr_tail": f"script '{script_token}' is not allowed",
            }

    timeout_s = max(5, _coerce_int(os.getenv("ATOM_DIRECTOR_PACK_CMD_TIMEOUT_S", 180), 180))
    started = time.time()
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        stderr_tail = _truncate_text(proc.stderr or "")
        return_code = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        proc_stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr_tail = _truncate_text(
            (proc_stderr + f"\nCommand timed out after {timeout_s} seconds.").strip()
        )
        return_code = 124
        timed_out = True
        proc = None
    except Exception as exc:
        duration = time.time() - started
        return {
            "command": [str(a) for a in args],
            "return_code": 1,
            "duration_sec": float(duration),
            "stdout_tail": "",
            "stderr_tail": _truncate_text(f"{type(exc).__name__}: {exc}"),
            "timed_out": False,
        }

    duration = time.time() - started
    return {
        "command": [str(a) for a in args],
        "return_code": return_code,
        "duration_sec": float(duration),
        "stdout_tail": _truncate_text("" if proc is None else (proc.stdout or "")),
        "stderr_tail": stderr_tail,
        "timed_out": timed_out,
    }


def _build_director_pack(
    workspace_root: Path,
    request: DirectorPackRequest,
) -> Dict[str, Any]:
    profile = _release_profile()
    if profile == "release" and bool(request.allow_missing):
        raise ValueError("allow_missing is not permitted when ATOM_RELEASE_PROFILE=release")

    root = workspace_root.resolve()
    safe_tag = _normalize_token(request.tag, fallback="director_demo", max_len=48)
    pack_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{safe_tag}"
    validation_root = (root / "validation_outputs" / "director_pack" / pack_id).resolve()
    packet_root = (root / "release_packets" / pack_id).resolve()
    validation_root.mkdir(parents=True, exist_ok=True)
    packet_root.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable or "python3"
    reliability_path = (validation_root / "reliability_kernel_smoke.json").resolve()
    safety_path = (validation_root / "supersonic_validation.json").resolve()
    evidence_json = (validation_root / "release_evidence.json").resolve()
    evidence_md = (validation_root / "release_evidence.md").resolve()
    scientific_path = (root / "validation_outputs" / "scientific_integrity.json").resolve()
    ablation_summary = (root / "benchmark_results" / "platform_ablations_rc1" / "summary.json").resolve()
    ablation_dir = ablation_summary.parent

    operations: List[Dict[str, Any]] = []
    effective_allow_missing = bool(request.allow_missing) and profile != "release"

    if bool(request.run_reliability):
        operations.append(
            _run_local_command(
                [
                    str(python_bin),
                    "scripts/reliability_kernel_smoke.py",
                    "--steps",
                    str(int(request.reliability_steps)),
                    "--seed",
                    str(int(request.reliability_seed)),
                    "--nx",
                    "16",
                    "--ny",
                    "16",
                    "--nz",
                    "8",
                    "--out",
                    str(reliability_path),
                ],
                cwd=root,
            )
        )
    else:
        reliability_path = (root / "validation_outputs" / "reliability_kernel_smoke.json").resolve()

    if bool(request.run_supersonic_validation):
        operations.append(
            _run_local_command(
                [
                    str(python_bin),
                    "scripts/supersonic_validation_pack.py",
                    "--steps",
                    str(int(request.supersonic_steps)),
                    "--nx",
                    str(int(request.supersonic_nx)),
                    "--ny",
                    str(int(request.supersonic_ny)),
                    "--seed",
                    str(int(request.supersonic_seed)),
                    "--allow-skip",
                    "--output",
                    str(safety_path),
                ],
                cwd=root,
            )
        )
    else:
        safety_path = (root / "validation_outputs" / "supersonic_validation.json").resolve()

    if bool(request.run_release_evidence):
        cmd = [
            str(python_bin),
            "scripts/generate_release_evidence.py",
            "--reliability",
            str(reliability_path),
            "--scientific",
            str(scientific_path),
            "--safety",
            str(safety_path),
            "--ablation",
            str(ablation_summary),
            "--output-json",
            str(evidence_json),
            "--output-md",
            str(evidence_md),
            "--profile",
            profile,
        ]
        if effective_allow_missing:
            cmd.append("--allow-missing")
        operations.append(_run_local_command(cmd, cwd=root))
    else:
        evidence_json = (root / "validation_outputs" / "release_evidence.json").resolve()
        evidence_md = (root / "validation_outputs" / "release_evidence.md").resolve()

    packet_cmd = [
        str(python_bin),
        "scripts/build_release_packet.py",
        "--ablation-dir",
        str(ablation_dir),
        "--release-evidence-json",
        str(evidence_json),
        "--release-evidence-md",
        str(evidence_md),
        "--output-dir",
        str(packet_root),
        "--profile",
        profile,
    ]
    if effective_allow_missing:
        packet_cmd.append("--allow-missing")
    operations.append(_run_local_command(packet_cmd, cwd=root))

    release_manifest_path = (packet_root / "release_packet_manifest.json").resolve()
    release_manifest: Optional[Dict[str, Any]] = None
    if release_manifest_path.exists():
        try:
            parsed = json.loads(release_manifest_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                release_manifest = parsed
        except Exception:
            release_manifest = None

    release_evidence_payload: Optional[Dict[str, Any]] = None
    if evidence_json.exists():
        try:
            parsed = json.loads(evidence_json.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                release_evidence_payload = parsed
        except Exception:
            release_evidence_payload = None

    ok = all(int(op.get("return_code", 1)) == 0 for op in operations)
    if release_manifest is not None:
        ok = ok and bool(release_manifest.get("ok", False) or effective_allow_missing)

    return {
        "ok": bool(ok),
        "generated_at": _utc_iso(),
        "profile": profile,
        "allow_missing_effective": bool(effective_allow_missing),
        "pack_id": pack_id,
        "output_dir": str(packet_root),
        "validation_dir": str(validation_root),
        "release_packet_manifest": release_manifest,
        "release_packet_manifest_path": str(release_manifest_path),
        "release_evidence": release_evidence_payload,
        "operations": operations,
    }


def _build_supersonic_timeseries(
    result: Optional[Dict[str, Any]],
    trace: List[Dict[str, Any]],
    limit: int,
) -> Dict[str, List[float]]:
    limit = max(8, int(limit))
    history = result.get("history", {}) if isinstance(result, dict) else {}

    reward_hist = history.get("reward", []) if isinstance(history, dict) else []
    shock_hist = history.get("shock_strength", []) if isinstance(history, dict) else []
    reduction_hist = history.get("shock_reduction", []) if isinstance(history, dict) else []
    jet_hist = history.get("jet_power", []) if isinstance(history, dict) else []

    reward = [_coerce_float(v) for v in reward_hist] if isinstance(reward_hist, list) else []
    shock_strength = [_coerce_float(v) for v in shock_hist] if isinstance(shock_hist, list) else []
    shock_reduction = [_coerce_float(v) for v in reduction_hist] if isinstance(reduction_hist, list) else []
    jet_power = [_coerce_float(v) for v in jet_hist] if isinstance(jet_hist, list) else []

    if reward:
        n = len(reward)
        step = list(range(n))
        shock_strength = _align_series(shock_strength, n)
        shock_reduction = _align_series(shock_reduction, n)
        jet_power = _align_series(jet_power, n)
        if n > limit:
            step = step[-limit:]
            reward = reward[-limit:]
            shock_strength = shock_strength[-limit:]
            shock_reduction = shock_reduction[-limit:]
            jet_power = jet_power[-limit:]
        return {
            "step": [float(v) for v in step],
            "reward": reward,
            "shock_strength": shock_strength,
            "shock_reduction": shock_reduction,
            "jet_power": jet_power,
        }

    trace_tail = trace[-limit:] if len(trace) > limit else list(trace)
    if not trace_tail:
        return {
            "step": [],
            "reward": [],
            "shock_strength": [],
            "shock_reduction": [],
            "jet_power": [],
        }
    step = [_coerce_int(item.get("step"), idx) for idx, item in enumerate(trace_tail)]
    return {
        "step": [float(v) for v in step],
        "reward": [_coerce_float(item.get("reward")) for item in trace_tail],
        "shock_strength": [_coerce_float(item.get("shock_strength")) for item in trace_tail],
        "shock_reduction": [_coerce_float(item.get("shock_reduction")) for item in trace_tail],
        "jet_power": [_coerce_float(item.get("jet_power")) for item in trace_tail],
    }


def _series_derived_metrics(timeseries: Dict[str, List[float]]) -> Dict[str, float]:
    reward = timeseries.get("reward", [])
    shock_strength = timeseries.get("shock_strength", [])
    shock_reduction = timeseries.get("shock_reduction", [])
    jet_power = timeseries.get("jet_power", [])
    window = 32
    return {
        "points": float(len(timeseries.get("step", []))),
        "reward_last": reward[-1] if reward else 0.0,
        "reward_mean_last_32": _mean(reward[-window:]) if reward else 0.0,
        "reward_slope_last_32": _slope(reward, window=window),
        "shock_strength_mean_last_32": _mean(shock_strength[-window:]) if shock_strength else 0.0,
        "shock_strength_slope_last_32": _slope(shock_strength, window=window),
        "shock_reduction_mean_last_32": _mean(shock_reduction[-window:]) if shock_reduction else 0.0,
        "shock_reduction_slope_last_32": _slope(shock_reduction, window=window),
        "jet_power_mean_last_32": _mean(jet_power[-window:]) if jet_power else 0.0,
    }


@dataclass
class InverseDesignJob:
    """In-memory job record for API/UI monitoring."""

    job_id: str
    request: Dict[str, Any]
    status: str = "queued"
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    output_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "request": dict(self.request),
            "created_at": _utc_iso(self.created_at),
            "updated_at": _utc_iso(self.updated_at),
            "result": self.result,
            "error": self.error,
            "output_dir": self.output_dir,
        }


class InverseDesignJobService:
    """Thread-safe in-memory async job service for inverse design runs."""

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parents[3]).resolve()
        self.jobs_root = self.workspace_root / "runs" / "inverse_design_jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, InverseDesignJob] = {}
        self._lock = threading.Lock()

    def _normalize_request(self, request: InverseDesignJobRequest) -> InverseDesignJobRequest:
        normalized = request.model_copy(deep=True)
        world_spec = _validate_world_spec(str(normalized.world_spec))
        world_kwargs = _validate_world_kwargs(
            world_spec=world_spec,
            world_kwargs=dict(normalized.world_kwargs)
            if isinstance(normalized.world_kwargs, dict)
            else {},
            workspace_root=self.workspace_root,
        )
        resolved = _validate_inverse_contract_paths(normalized, self.workspace_root)
        return normalized.model_copy(
            update={
                "world_spec": world_spec,
                "world_kwargs": world_kwargs,
                **resolved,
            }
        )

    def submit(self, request: InverseDesignJobRequest) -> InverseDesignJob:
        request = self._normalize_request(request)
        job_id = uuid.uuid4().hex[:12]
        job = InverseDesignJob(job_id=job_id, request=request.model_dump())
        with self._lock:
            self._jobs[job_id] = job

        t = threading.Thread(target=self._execute, args=(job_id,), daemon=True)
        t.start()
        return job

    def run_sync(self, request: InverseDesignJobRequest) -> InverseDesignJob:
        request = self._normalize_request(request)
        job_id = uuid.uuid4().hex[:12]
        job = InverseDesignJob(job_id=job_id, request=request.model_dump())
        with self._lock:
            self._jobs[job_id] = job
        self._execute(job_id)
        return job

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
        return None if job is None else job.to_dict()

    def _resolve_optional_path(self, path_str: Optional[str]) -> Optional[str]:
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_absolute():
            p = (self.workspace_root / p).resolve()
        return str(p)

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.updated_at = _utc_now()

        payload = job.request
        run_name = payload.get("name") or f"inverse_{job_id}"
        output_root = self.jobs_root / run_name
        output_root.mkdir(parents=True, exist_ok=True)

        try:
            runner = _resolve_runner_module()
            raw_world_kwargs = payload.get("world_kwargs", {})
            world_kwargs = (
                _json_safe_value(raw_world_kwargs)
                if isinstance(raw_world_kwargs, dict)
                else {}
            )
            cfg = runner.ExperimentConfig(
                name=run_name,
                world_spec=str(payload["world_spec"]),
                grid_shape=tuple(int(v) for v in payload["grid_shape"]),
                max_steps=8,  # Unused in inverse mode but explicit for config integrity.
                output_dir=str(self.jobs_root),
                seed=42,
                device=str(payload.get("device", "cpu")),
                world_kwargs=world_kwargs,
            )
            experiment = runner.ATOMExperiment(cfg)

            result = experiment.run_inverse_design(
                backend=str(payload["backend"]),
                iterations=payload.get("iterations"),
                population=payload.get("population"),
                top_k=int(payload["top_k"]),
                rollout_steps=int(payload["rollout_steps"]),
                objective_spec_path=self._resolve_optional_path(payload.get("objective_spec_path")),
                parameter_space_path=self._resolve_optional_path(payload.get("parameter_space_path")),
                report_name=str(payload.get("report_name", "inverse_design_report.json")),
            )

            with self._lock:
                job.status = "succeeded"
                job.updated_at = _utc_now()
                job.result = result
                job.output_dir = str(output_root)
        except Exception as exc:  # pragma: no cover - requires runtime failures
            logger.exception("Inverse design job failed: %s", job_id)
            with self._lock:
                job.status = "failed"
                job.updated_at = _utc_now()
                job.error = f"{type(exc).__name__}: {exc}"


@dataclass
class SupersonicChallengeJob:
    """In-memory challenge job record with live telemetry snapshots."""

    job_id: str
    request: Dict[str, Any]
    status: str = "queued"
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    output_dir: Optional[str] = None
    latest_telemetry: Optional[Dict[str, Any]] = None
    telemetry_trace: List[Dict[str, Any]] = field(default_factory=list)
    law_trace: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    incidents: List[Dict[str, Any]] = field(default_factory=list)
    bookmarks: List[Dict[str, Any]] = field(default_factory=list)
    paused: bool = False
    cancel_requested: bool = False
    incident_cooldowns: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        latest_law = self.law_trace[-1] if self.law_trace else None
        return {
            "job_id": self.job_id,
            "status": self.status,
            "request": dict(self.request),
            "created_at": _utc_iso(self.created_at),
            "updated_at": _utc_iso(self.updated_at),
            "result": self.result,
            "error": self.error,
            "output_dir": self.output_dir,
            "latest_telemetry": self.latest_telemetry,
            "telemetry_points": len(self.telemetry_trace),
            "law_events": len(self.law_trace),
            "latest_law": latest_law,
            "paused": bool(self.paused),
            "cancel_requested": bool(self.cancel_requested),
            "timeline_events": len(self.timeline),
            "incident_events": len(self.incidents),
            "bookmark_events": len(self.bookmarks),
        }


class _SupersonicJobCancelled(RuntimeError):
    """Raised by callbacks when a running job is cancelled by control plane."""


class SupersonicChallengeJobService:
    """Thread-safe async service for supersonic challenge orchestration."""

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parents[3]).resolve()
        self.jobs_root = self.workspace_root / "runs" / "supersonic_jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, SupersonicChallengeJob] = {}
        self._lock = threading.Lock()

    def submit(self, request: SupersonicChallengeJobRequest) -> SupersonicChallengeJob:
        job_id = uuid.uuid4().hex[:12]
        job = SupersonicChallengeJob(job_id=job_id, request=request.model_dump())
        with self._lock:
            self._jobs[job_id] = job

        t = threading.Thread(target=self._execute, args=(job_id,), daemon=True)
        t.start()
        return job

    def run_sync(self, request: SupersonicChallengeJobRequest) -> SupersonicChallengeJob:
        job_id = uuid.uuid4().hex[:12]
        job = SupersonicChallengeJob(job_id=job_id, request=request.model_dump())
        with self._lock:
            self._jobs[job_id] = job
        self._execute(job_id)
        return job

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
        return None if job is None else job.to_dict()

    def _append_timeline_locked(
        self,
        job: SupersonicChallengeJob,
        event_type: str,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = {
            "ts": _utc_iso(),
            "type": str(event_type),
            "message": str(message),
        }
        if isinstance(payload, dict) and payload:
            event["payload"] = _json_safe_value(payload)
        job.timeline.append(event)
        if len(job.timeline) > _MAX_TIMELINE_EVENTS:
            job.timeline = job.timeline[-_MAX_TIMELINE_EVENTS:]

    def _append_bookmark_locked(
        self,
        job: SupersonicChallengeJob,
        note: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        bookmark = {
            "ts": _utc_iso(),
            "note": str(note or "bookmark"),
        }
        if isinstance(payload, dict) and payload:
            bookmark["payload"] = _json_safe_value(payload)
        job.bookmarks.append(bookmark)
        if len(job.bookmarks) > _MAX_BOOKMARK_EVENTS:
            job.bookmarks = job.bookmarks[-_MAX_BOOKMARK_EVENTS:]

    def _append_incident_locked(
        self,
        job: SupersonicChallengeJob,
        *,
        code: str,
        severity: str,
        message: str,
        step: int,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = {
            "ts": _utc_iso(),
            "code": str(code),
            "severity": str(severity),
            "message": str(message),
            "step": int(step),
        }
        if isinstance(payload, dict) and payload:
            event["payload"] = _json_safe_value(payload)
        job.incidents.append(event)
        if len(job.incidents) > _MAX_INCIDENT_EVENTS:
            job.incidents = job.incidents[-_MAX_INCIDENT_EVENTS:]
        self._append_timeline_locked(
            job,
            event_type=f"incident:{code}",
            message=message,
            payload=event,
        )

    def _incident_emit_allowed_locked(
        self,
        job: SupersonicChallengeJob,
        *,
        code: str,
        step: int,
        cooldown_steps: int,
    ) -> bool:
        last_step = int(job.incident_cooldowns.get(code, -10**9))
        if int(step) - last_step < int(cooldown_steps):
            return False
        job.incident_cooldowns[code] = int(step)
        return True

    def get_telemetry(self, job_id: str, limit: int = 400) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            trace = [dict(item) for item in job.telemetry_trace]
            law_trace = [dict(item) for item in job.law_trace]
            result = dict(job.result) if isinstance(job.result, dict) else None
            latest_telemetry = (
                dict(job.latest_telemetry) if isinstance(job.latest_telemetry, dict) else job.latest_telemetry
            )
            status = str(job.status)
            updated_at = _utc_iso(job.updated_at)
            paused = bool(job.paused)
            cancel_requested = bool(job.cancel_requested)
            timeline = [dict(item) for item in job.timeline]
            incidents = [dict(item) for item in job.incidents]
            bookmarks = [dict(item) for item in job.bookmarks]

        timeseries = _build_supersonic_timeseries(result=result, trace=trace, limit=limit)
        derived = _series_derived_metrics(timeseries)
        summary = result.get("summary") if isinstance(result, dict) else None
        return {
            "job_id": job_id,
            "status": status,
            "updated_at": updated_at,
            "latest_telemetry": latest_telemetry,
            "summary": summary,
            "timeseries": timeseries,
            "derived": derived,
            "law_trace": law_trace[-64:] if len(law_trace) > 64 else law_trace,
            "timeline_tail": timeline[-128:] if len(timeline) > 128 else timeline,
            "incidents_tail": incidents[-64:] if len(incidents) > 64 else incidents,
            "bookmarks": bookmarks[-64:] if len(bookmarks) > 64 else bookmarks,
            "control_state": {
                "paused": paused,
                "cancel_requested": cancel_requested,
            },
        }

    def get_timeline(self, job_id: str, limit: int = 256) -> Optional[Dict[str, Any]]:
        lim = max(1, int(limit))
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            timeline = [dict(item) for item in job.timeline]
            incidents = [dict(item) for item in job.incidents]
            bookmarks = [dict(item) for item in job.bookmarks]
            status = str(job.status)
            updated_at = _utc_iso(job.updated_at)
            paused = bool(job.paused)
            cancel_requested = bool(job.cancel_requested)
        return {
            "job_id": job_id,
            "status": status,
            "updated_at": updated_at,
            "timeline": timeline[-lim:] if len(timeline) > lim else timeline,
            "incidents": incidents[-lim:] if len(incidents) > lim else incidents,
            "bookmarks": bookmarks[-lim:] if len(bookmarks) > lim else bookmarks,
            "control_state": {
                "paused": paused,
                "cancel_requested": cancel_requested,
            },
        }

    def control(self, job_id: str, request: SupersonicControlRequest) -> Optional[Dict[str, Any]]:
        action = str(request.action)
        note = str(request.note or "").strip()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None

            if action == "pause":
                if job.status == "running":
                    job.paused = True
                self._append_timeline_locked(
                    job,
                    event_type="control:pause",
                    message="Pause requested by operator.",
                    payload={"note": note or None},
                )
            elif action == "resume":
                job.paused = False
                self._append_timeline_locked(
                    job,
                    event_type="control:resume",
                    message="Resume requested by operator.",
                    payload={"note": note or None},
                )
            elif action == "cancel":
                if job.status == "running":
                    job.cancel_requested = True
                    job.paused = False
                self._append_timeline_locked(
                    job,
                    event_type="control:cancel",
                    message="Cancel requested by operator.",
                    payload={"note": note or None},
                )
                self._append_incident_locked(
                    job,
                    code="operator_cancel",
                    severity="info",
                    message="Operator requested controlled stop.",
                    step=_coerce_int(job.latest_telemetry.get("step") if isinstance(job.latest_telemetry, dict) else 0),
                    payload={"note": note or None},
                )
            elif action == "bookmark":
                step = _coerce_int(job.latest_telemetry.get("step") if isinstance(job.latest_telemetry, dict) else 0)
                payload = {"step": int(step)}
                if note:
                    payload["note"] = note
                self._append_bookmark_locked(job, note=note or f"bookmark_step_{step}", payload=payload)
                self._append_timeline_locked(
                    job,
                    event_type="control:bookmark",
                    message="Bookmark captured.",
                    payload=payload,
                )
            else:  # pragma: no cover - pydantic blocks invalid actions
                raise ValueError(f"Unsupported control action: {action}")
            job.updated_at = _utc_now()

            return {
                "job_id": job.job_id,
                "status": job.status,
                "paused": bool(job.paused),
                "cancel_requested": bool(job.cancel_requested),
                "timeline_events": len(job.timeline),
                "incident_events": len(job.incidents),
                "bookmark_events": len(job.bookmarks),
                "updated_at": _utc_iso(job.updated_at),
            }

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.updated_at = _utc_now()
            self._append_timeline_locked(
                job,
                event_type="job:running",
                message="Supersonic challenge job entered running state.",
                payload={"job_id": job.job_id},
            )

        payload = job.request
        run_name = payload.get("name") or f"supersonic_{job_id}"
        output_root = self.jobs_root / run_name
        output_root.mkdir(parents=True, exist_ok=True)

        def _on_step(telemetry: Dict[str, Any]) -> None:
            while True:
                with self._lock:
                    target = self._jobs.get(job_id)
                    if target is None:
                        return
                    if target.cancel_requested:
                        self._append_timeline_locked(
                            target,
                            event_type="job:cancelled",
                            message="Run cancelled by operator control plane.",
                            payload={"job_id": job_id},
                        )
                        raise _SupersonicJobCancelled("cancel requested")
                    paused = bool(target.paused)
                if not paused:
                    break
                time.sleep(0.05)

            with self._lock:
                target = self._jobs.get(job_id)
                if target is None:
                    return
                telemetry_payload = dict(telemetry)
                telemetry_payload["updated_at"] = _utc_iso()
                target.latest_telemetry = telemetry_payload
                target.telemetry_trace.append(telemetry_payload)
                if len(target.telemetry_trace) > _MAX_TELEMETRY_POINTS:
                    target.telemetry_trace = target.telemetry_trace[-_MAX_TELEMETRY_POINTS:]

                step = _coerce_int(telemetry_payload.get("step"), default=len(target.telemetry_trace))
                shock_strength = _coerce_float(telemetry_payload.get("shock_strength", 0.0))
                reward = _coerce_float(telemetry_payload.get("reward", 0.0))
                if (
                    shock_strength > 0.24
                    and self._incident_emit_allowed_locked(
                        target,
                        code="high_shock_strength",
                        step=step,
                        cooldown_steps=24,
                    )
                ):
                    self._append_incident_locked(
                        target,
                        code="high_shock_strength",
                        severity="warning",
                        message="Shock strength exceeded control envelope threshold.",
                        step=step,
                        payload={
                            "shock_strength": shock_strength,
                            "reward": reward,
                        },
                    )
                if len(target.telemetry_trace) >= 12:
                    recent = [
                        _coerce_float(item.get("shock_strength", 0.0))
                        for item in target.telemetry_trace[-12:]
                    ]
                    trend = _slope(recent, window=len(recent))
                    if (
                        trend > 0.0025
                        and self._incident_emit_allowed_locked(
                            target,
                            code="shock_trend_up",
                            step=step,
                            cooldown_steps=32,
                        )
                    ):
                        self._append_incident_locked(
                            target,
                            code="shock_trend_up",
                            severity="warning",
                            message="Shock trend rising over recent telemetry window.",
                            step=step,
                            payload={"slope_last_12": trend},
                        )
                target.updated_at = _utc_now()

        def _on_law(law_str: str, step: int) -> None:
            with self._lock:
                target = self._jobs.get(job_id)
                if target is None:
                    return
                payload = {
                    "step": int(step),
                    "law": str(law_str),
                    "updated_at": _utc_iso(),
                }
                target.law_trace.append(payload)
                if len(target.law_trace) > _MAX_LAW_EVENTS:
                    target.law_trace = target.law_trace[-_MAX_LAW_EVENTS:]
                self._append_timeline_locked(
                    target,
                    event_type="law:update",
                    message="Scientist law update captured.",
                    payload=payload,
                )
                target.updated_at = _utc_now()

        try:
            challenge = _resolve_challenge_module()
            report = challenge.run_challenge(
                max_steps=int(payload.get("steps", 256)),
                headless=bool(payload.get("headless", True)),
                output_dir=str(output_root),
                on_step_callback=_on_step,
                on_law_callback=_on_law,
            )
            with self._lock:
                job.status = "succeeded"
                job.updated_at = _utc_now()
                job.result = report
                job.output_dir = str(output_root)
                self._append_timeline_locked(
                    job,
                    event_type="job:succeeded",
                    message="Supersonic challenge run completed successfully.",
                    payload={"output_dir": str(output_root)},
                )
        except _SupersonicJobCancelled:
            with self._lock:
                job.status = "cancelled"
                job.updated_at = _utc_now()
                job.output_dir = str(output_root)
                self._append_timeline_locked(
                    job,
                    event_type="job:cancelled",
                    message="Supersonic challenge run exited due to cancellation.",
                    payload={"output_dir": str(output_root)},
                )
        except Exception as exc:  # pragma: no cover - requires runtime failures
            logger.exception("Supersonic challenge job failed: %s", job_id)
            with self._lock:
                job.status = "failed"
                job.updated_at = _utc_now()
                job.error = f"{type(exc).__name__}: {exc}"
                self._append_incident_locked(
                    job,
                    code="runtime_failure",
                    severity="critical",
                    message=f"Supersonic run failed: {type(exc).__name__}",
                    step=_coerce_int(job.latest_telemetry.get("step") if isinstance(job.latest_telemetry, dict) else 0),
                    payload={"error": str(exc)},
                )


class GovernedAssistantService:
    """Grounded assistant with deterministic core and optional LLM synthesis."""

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = (workspace_root or Path(__file__).resolve().parents[3]).resolve()
        self.llm_enabled = str(os.getenv("ATOM_ASSISTANT_ENABLE_LLM", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.llm_model = str(os.getenv("ATOM_ASSISTANT_MODEL", "gpt-4.1-mini"))
        self.llm_base_url = str(
            os.getenv("ATOM_ASSISTANT_OPENAI_URL", "https://api.openai.com/v1/responses")
        )
        self.llm_timeout_s = max(3.0, _coerce_float(os.getenv("ATOM_ASSISTANT_TIMEOUT_S", 15.0), 15.0))

    def _resolve_inverse_job(
        self,
        request: AssistantQueryRequest,
        inverse_service: InverseDesignJobService,
    ) -> Optional[Dict[str, Any]]:
        if request.inverse_job_id:
            return inverse_service.get_job(request.inverse_job_id)
        jobs = inverse_service.list_jobs()
        return jobs[0] if jobs else None

    def _resolve_supersonic_job(
        self,
        request: AssistantQueryRequest,
        challenge_service: SupersonicChallengeJobService,
    ) -> Optional[Dict[str, Any]]:
        if request.supersonic_job_id:
            return challenge_service.get_telemetry(
                request.supersonic_job_id,
                limit=request.telemetry_window,
            )
        jobs = challenge_service.list_jobs()
        if not jobs:
            return None
        first_id = str(jobs[0]["job_id"])
        return challenge_service.get_telemetry(first_id, limit=request.telemetry_window)

    def _resolve_runtime_snapshot(self) -> Dict[str, Any]:
        try:
            return _load_runtime_telemetry_snapshot(self.workspace_root)
        except Exception as exc:  # pragma: no cover - runtime environment issue
            logger.debug("Runtime telemetry snapshot unavailable for assistant: %s", exc)
            return {
                "available": False,
                "status": f"runtime_snapshot_error:{type(exc).__name__}",
            }

    def _inverse_metrics(self, job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(job, dict):
            return {"available": False}
        result = job.get("result", {})
        candidates = result.get("candidates", []) if isinstance(result, dict) else []
        if not isinstance(candidates, list):
            candidates = []
        if not candidates:
            return {
                "available": True,
                "job_id": job.get("job_id"),
                "status": job.get("status", "unknown"),
                "candidate_count": 0,
                "world_spec": job.get("request", {}).get("world_spec", "unknown"),
            }

        def _candidate_key(candidate: Dict[str, Any]) -> Any:
            rank = _coerce_int(candidate.get("rank"), default=10**9)
            score = _coerce_float(
                candidate.get(
                    "objective_score",
                    candidate.get("raw_objective_score", 10**9),
                )
            )
            return (rank, score)

        best = sorted(candidates, key=_candidate_key)[0]
        feasible_count = sum(1 for c in candidates if bool(c.get("feasible", False)))
        return {
            "available": True,
            "job_id": job.get("job_id"),
            "status": job.get("status", "unknown"),
            "candidate_count": len(candidates),
            "best_candidate_id": best.get("candidate_id", "unknown"),
            "best_objective_score": _coerce_float(
                best.get("objective_score", best.get("raw_objective_score"))
            ),
            "feasible_rate": float(feasible_count / float(len(candidates))),
            "world_spec": job.get("request", {}).get("world_spec", "unknown"),
        }

    def _supersonic_metrics(self, packet: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(packet, dict):
            return {"available": False}
        summary = packet.get("summary", {})
        derived = packet.get("derived", {})
        control_state = packet.get("control_state", {})
        incidents = packet.get("incidents_tail", [])
        bookmarks = packet.get("bookmarks", [])
        return {
            "available": True,
            "job_id": packet.get("job_id"),
            "status": packet.get("status", "unknown"),
            "final_shock_strength_50": _coerce_float(
                summary.get("final_shock_strength_50", 0.0)
            ),
            "final_shock_reduction_50": _coerce_float(
                summary.get("final_shock_reduction_50", 0.0)
            ),
            "reward_mean_last_32": _coerce_float(derived.get("reward_mean_last_32", 0.0)),
            "shock_strength_slope_last_32": _coerce_float(
                derived.get("shock_strength_slope_last_32", 0.0)
            ),
            "shock_reduction_slope_last_32": _coerce_float(
                derived.get("shock_reduction_slope_last_32", 0.0)
            ),
            "telemetry_points": _coerce_int(derived.get("points", 0)),
            "paused": bool(control_state.get("paused", False)),
            "cancel_requested": bool(control_state.get("cancel_requested", False)),
            "incident_count": len(incidents) if isinstance(incidents, list) else 0,
            "bookmark_count": len(bookmarks) if isinstance(bookmarks, list) else 0,
        }

    def _runtime_metrics(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(snapshot, dict) or not bool(snapshot.get("available", False)):
            return {"available": False, "status": str(snapshot.get("status", "no_runtime_telemetry"))}

        safety_summary_raw = snapshot.get("safety_summary", {})
        safety_summary = safety_summary_raw if isinstance(safety_summary_raw, dict) else {}
        diagnostics = snapshot.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        live_view = diagnostics.get("live_view", {})
        if not isinstance(live_view, dict):
            live_view = {}
        eyes = diagnostics.get("eyes2_saliency", {})
        if not isinstance(eyes, dict):
            eyes = {}
        brain = diagnostics.get("brain_saliency", {})
        if not isinstance(brain, dict):
            brain = {}

        def _shape(block: Any) -> List[int]:
            if not isinstance(block, dict):
                return [0, 0]
            shape = block.get("shape", [])
            if not isinstance(shape, list) or len(shape) != 2:
                return [0, 0]
            return [_coerce_int(shape[0], 0), _coerce_int(shape[1], 0)]

        def _range(block: Any) -> Dict[str, float]:
            if not isinstance(block, dict):
                return {"min": 0.0, "max": 0.0}
            return {
                "min": _coerce_float(block.get("raw_min", 0.0)),
                "max": _coerce_float(block.get("raw_max", 0.0)),
            }

        speed_block = live_view.get("speed_xy")
        density_block = live_view.get("density_xy")
        divergence_block = live_view.get("divergence_xy")
        obstacle_block = live_view.get("obstacle_xy")
        eyes_block = eyes.get("map")
        brain_block = brain.get("map")

        labels_raw = brain.get("theory_feature_labels", [])
        importance_raw = brain.get("theory_feature_importance", [])
        labels = [str(v) for v in labels_raw] if isinstance(labels_raw, list) else []
        importance = (
            [_coerce_float(v) for v in importance_raw]
            if isinstance(importance_raw, list)
            else []
        )
        if len(labels) != len(importance):
            labels = [f"theory_{i}" for i in range(len(importance))]
        top_features = sorted(
            [
                {"label": labels[i], "importance": float(importance[i])}
                for i in range(len(importance))
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )[:3]

        governance_raw = snapshot.get("governance_decision", {})
        governance = governance_raw if isinstance(governance_raw, dict) else {}
        governance_reasons = governance.get("reasons", [])
        if not isinstance(governance_reasons, list):
            governance_reasons = []
        verifier_raw = snapshot.get("verifier_metrics", {})
        verifier = verifier_raw if isinstance(verifier_raw, dict) else {}
        baseline_raw = snapshot.get("baseline_context", {})
        baseline = baseline_raw if isinstance(baseline_raw, dict) else {}

        return {
            "available": True,
            "status": str(snapshot.get("status", "ok")),
            "step": _coerce_int(snapshot.get("step", 0), 0),
            "reward": _coerce_float(snapshot.get("reward", 0.0)),
            "stress": _coerce_float(snapshot.get("stress", 0.0)),
            "theory": _coerce_float(snapshot.get("theory", 0.0)),
            "trust_raw": _coerce_float(snapshot.get("trust_raw", 0.0)),
            "trust_verified": _coerce_float(snapshot.get("trust_verified", 0.0)),
            "trust_structural_floor": _coerce_float(
                snapshot.get("trust_structural_floor", 0.0)
            ),
            "safety_mode": str(safety_summary.get("mode", "normal")),
            "safety_interventions": _coerce_int(safety_summary.get("interventions", 0)),
            "safety_fallback_uses": _coerce_int(safety_summary.get("fallback_uses", 0)),
            "safety_hard_event_rate": _coerce_float(safety_summary.get("hard_event_rate", 0.0)),
            "safety_degrade_window": _coerce_int(safety_summary.get("degrade_window", 0)),
            "safety_mode_transitions": (
                dict(safety_summary.get("mode_transitions", {}))
                if isinstance(safety_summary.get("mode_transitions", {}), dict)
                else {}
            ),
            "safety_reason_histogram": (
                dict(safety_summary.get("reason_histogram", {}))
                if isinstance(safety_summary.get("reason_histogram", {}), dict)
                else {}
            ),
            "governance_approved": (
                bool(governance.get("approved"))
                if governance.get("approved") is not None
                else None
            ),
            "governance_reasons": [str(v) for v in governance_reasons],
            "governance_novelty": _coerce_float(governance.get("novelty_score", 0.0)),
            "governance_stability": _coerce_float(governance.get("stability_score", 0.0)),
            "governance_intervention_consistency": _coerce_float(
                governance.get("intervention_consistency", 0.0)
            ),
            "verifier_rmse": _coerce_float(verifier.get("rmse", 0.0)),
            "verifier_correlation": _coerce_float(verifier.get("correlation", 0.0)),
            "verifier_interventional": _coerce_float(verifier.get("interventional", 0.0)),
            "baseline_ready": bool(baseline.get("ready", False)),
            "baseline_slope": _coerce_float(baseline.get("slope", 0.0)),
            "baseline_r2": _coerce_float(baseline.get("r2", 0.0)),
            "discovery_target_mode": str(baseline.get("discovery_target_mode", "hybrid")),
            "discovery_target_value": _coerce_float(
                baseline.get("discovery_target_value", 0.0)
            ),
            "diagnostics_enabled": bool(diagnostics.get("enabled", False)),
            "diagnostics_step": _coerce_int(diagnostics.get("updated_step", 0)),
            "diagnostics_stale_steps": _coerce_int(diagnostics.get("stale_steps", 0)),
            "diagnostics_interval": _coerce_int(diagnostics.get("interval", 0)),
            "live_speed_shape": _shape(speed_block),
            "live_density_shape": _shape(density_block),
            "live_divergence_shape": _shape(divergence_block),
            "live_obstacle_shape": _shape(obstacle_block),
            "live_speed_range": _range(speed_block),
            "live_density_range": _range(density_block),
            "live_divergence_range": _range(divergence_block),
            "live_obstacle_range": _range(obstacle_block),
            "eyes_shape": _shape(eyes_block),
            "eyes_objective": _coerce_float(eyes.get("objective", 0.0)),
            "eyes_target_label": str(eyes.get("target_label", "")),
            "eyes_target_index": (
                _coerce_int(eyes.get("target_index"), default=-1)
                if eyes.get("target_index") is not None
                else None
            ),
            "brain_shape": _shape(brain_block),
            "brain_objective": _coerce_float(brain.get("objective", 0.0)),
            "top_theory_features": top_features,
            "top_latents": snapshot.get("saliency", {}).get("top_latents", []),
        }

    def _diagnostic_line(self, mode: str, runtime: Dict[str, Any]) -> str:
        if not runtime.get("available"):
            return (
                "Diagnostic surface: runtime telemetry unavailable, live maps and saliency "
                "must be refreshed from an active training run."
            )
        if mode == "live_flow":
            s_shape = runtime.get("live_speed_shape", [0, 0])
            d_shape = runtime.get("live_density_shape", [0, 0])
            div_shape = runtime.get("live_divergence_shape", [0, 0])
            obs_shape = runtime.get("live_obstacle_shape", [0, 0])
            return (
                "Live-flow diagnostics: speed_shape="
                f"{s_shape[0]}x{s_shape[1]}, density_shape={d_shape[0]}x{d_shape[1]}, "
                f"divergence_shape={div_shape[0]}x{div_shape[1]}, obstacle_shape={obs_shape[0]}x{obs_shape[1]}."
            )
        if mode == "eyes_saliency":
            shape = runtime.get("eyes_shape", [0, 0])
            target_label = str(runtime.get("eyes_target_label", "")).strip()
            return (
                "Eyes2 saliency diagnostics: objective="
                f"{runtime.get('eyes_objective', 0.0):.6f}, map_shape={shape[0]}x{shape[1]}, "
                f"target={target_label or 'embedding_energy'}."
            )
        if mode == "brain_saliency":
            shape = runtime.get("brain_shape", [0, 0])
            top = runtime.get("top_theory_features", [])
            if top:
                lead = top[0]
                return (
                    "Brain saliency diagnostics: objective="
                    f"{runtime.get('brain_objective', 0.0):.6f}, map_shape={shape[0]}x{shape[1]}, "
                    f"top_theory_feature={lead.get('label')} ({lead.get('importance', 0.0):.3f})."
                )
            return (
                "Brain saliency diagnostics: objective="
                f"{runtime.get('brain_objective', 0.0):.6f}, map_shape={shape[0]}x{shape[1]}."
            )
        if mode == "trust_surface":
            return (
                "Trust surface diagnostics: trust_raw="
                f"{runtime.get('trust_raw', 0.0):.6f}, trust_verified={runtime.get('trust_verified', 0.0):.6f}, "
                f"structural_floor={runtime.get('trust_structural_floor', 0.0):.6f}."
            )
        return (
            "Overview diagnostics: runtime_step="
            f"{runtime.get('step', 0)}, diagnostics_step={runtime.get('diagnostics_step', 0)}, "
            f"stale_steps={runtime.get('diagnostics_stale_steps', 0)}, "
            f"safety_mode={runtime.get('safety_mode', 'normal')}, "
            f"hard_event_rate={runtime.get('safety_hard_event_rate', 0.0):.3f}, "
            f"governance_approved={runtime.get('governance_approved')}."
        )

    def _intent_actions(
        self,
        intent: str,
        inverse: Dict[str, Any],
        supersonic: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> List[str]:
        actions: List[str] = []
        if intent == "scientific_discovery":
            actions.extend(
                [
                    "Capture hypothesis provenance with explicit holdout-window calibration.",
                    "Re-run deterministic replay on the latest runtime step before accepting any law update.",
                    "Schedule perturbation experiments that maximize uncertainty reduction.",
                ]
            )
        elif intent == "inverse_design":
            actions.extend(
                [
                    "Promote only candidates that satisfy hard constraints and holdout validation.",
                    "Increase solver budget only after feasible-rate remains stable across seeds.",
                    "Export top candidates with objective traces for reproducible audit.",
                ]
            )
        elif intent == "engineering":
            actions.extend(
                [
                    "Lock schema hashes for observation/action features before deployment handoff.",
                    "Run contract tests for simulator adapters and API payloads in CI gates.",
                    "Track SLOs for queue latency, run completion, and telemetry freshness.",
                ]
            )
        else:
            actions.extend(
                [
                    "Keep runtime safety supervisor enabled with explicit intervention thresholds.",
                    "Define fallback control policy and verify bounded action envelope under disturbances.",
                    "Escalate when shock trend and trust metrics diverge for more than one control window.",
                ]
            )

        if supersonic.get("available") and supersonic.get("shock_strength_slope_last_32", 0.0) > 0.002:
            actions.append("Trigger fallback policy review: shock strength trend is increasing.")
        if supersonic.get("available") and supersonic.get("incident_count", 0) > 0:
            actions.append("Inspect supersonic incident timeline and annotate mitigation bookmarks.")
        if supersonic.get("available") and supersonic.get("paused", False):
            actions.append("Supersonic run is paused; resume or cancel before scheduling new control actions.")
        if inverse.get("available") and inverse.get("candidate_count", 0) > 0 and inverse.get("feasible_rate", 1.0) < 0.70:
            actions.append("Tighten constraint checks: feasible-rate is below 70%.")
        if runtime.get("available") and runtime.get("trust_verified", 0.0) < runtime.get("trust_structural_floor", 0.0):
            actions.append("Block promotion: verified trust is below structural trust floor.")
        if runtime.get("available") and runtime.get("governance_approved") is False:
            reasons = runtime.get("governance_reasons", [])
            if isinstance(reasons, list) and reasons:
                actions.append(
                    f"Governance rejected current law candidate: {', '.join(str(r) for r in reasons[:3])}."
                )
            else:
                actions.append("Governance rejected current law candidate; require additional evidence before adoption.")
        if runtime.get("available") and not bool(runtime.get("baseline_ready", False)):
            actions.append("Reward baseline is not calibrated yet; defer structural-law promotion until baseline stabilizes.")
        if runtime.get("available") and str(runtime.get("safety_mode", "normal")) in {"cautious", "safe_hold"}:
            actions.append(
                "Runtime assurance is in degrade mode; reduce control aggressiveness and inspect safety reasons."
            )
        if runtime.get("available") and float(runtime.get("safety_hard_event_rate", 0.0)) > 0.20:
            actions.append(
                "Hard-event rate is elevated; schedule disturbance-focused replay before promoting this controller."
            )
        return actions

    def _extract_llm_text(self, payload: Dict[str, Any]) -> str:
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

        choices = payload.get("choices")
        if isinstance(choices, list):
            for item in choices:
                if not isinstance(item, dict):
                    continue
                message = item.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    chunks: List[str] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            chunks.append(text.strip())
                    if chunks:
                        return "\n".join(chunks).strip()

        output = payload.get("output")
        if not isinstance(output, list):
            return ""
        chunks: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        return "\n".join(chunks).strip()

    def _llm_synthesize(
        self,
        request: AssistantQueryRequest,
        deterministic_answer: str,
        inverse: Dict[str, Any],
        supersonic: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> Optional[str]:
        if not self.llm_enabled:
            return None

        api_key = str(
            os.getenv("ATOM_ASSISTANT_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        ).strip()
        local_url = _is_local_base_url(self.llm_base_url)
        if not api_key and not local_url:
            return None

        context_payload = {
            "question": request.question,
            "mode": request.mode,
            "intent": request.intent,
            "diagnostic_mode": request.diagnostic_mode,
            "inverse": inverse,
            "supersonic": supersonic,
            "runtime": runtime,
            "deterministic_baseline": deterministic_answer,
        }
        system_prompt = (
            "You are an engineering copilot for a scientific control platform. "
            "Use only provided JSON context, do not invent data, keep conclusions bounded by evidence, "
            "and return concise operational guidance."
        )
        user_prompt = (
            "Synthesize a grounded answer for the operator. "
            "Include current state, risk posture, and next actions.\n\n"
            f"Context:\n{json.dumps(context_payload, sort_keys=True)}"
        )

        use_chat_completions = "/chat/completions" in self.llm_base_url.rstrip("/")
        if use_chat_completions:
            request_payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 420,
            }
        else:
            request_payload = {
                "model": self.llm_model,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    },
                ],
                "temperature": 0.1,
                "max_output_tokens": 420,
            }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = url_request.Request(
            self.llm_base_url,
            method="POST",
            data=json.dumps(request_payload).encode("utf-8"),
            headers=headers,
        )
        try:
            with url_request.urlopen(req, timeout=self.llm_timeout_s) as resp:
                body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            text = self._extract_llm_text(parsed)
            return text if text else None
        except (url_error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("LLM synthesis failed; falling back to deterministic answer: %s", exc)
            return None

    def respond(
        self,
        request: AssistantQueryRequest,
        inverse_service: InverseDesignJobService,
        challenge_service: SupersonicChallengeJobService,
    ) -> AssistantQueryResponse:
        inverse_job = self._resolve_inverse_job(request, inverse_service)
        supersonic_packet = self._resolve_supersonic_job(request, challenge_service)
        runtime_snapshot = self._resolve_runtime_snapshot()
        inverse = self._inverse_metrics(inverse_job)
        supersonic = self._supersonic_metrics(supersonic_packet)
        runtime = self._runtime_metrics(runtime_snapshot)

        citations: List[AssistantCitation] = []
        lines: List[str] = [
            f"Intent={request.intent}. Diagnostic mode={request.diagnostic_mode}. Query mode={request.mode}."
        ]
        actions = self._intent_actions(request.intent, inverse, supersonic, runtime)

        if inverse.get("available"):
            citations.extend(
                [
                    AssistantCitation(
                        source="inverse_design",
                        field_path="job.status",
                        value=str(inverse.get("status", "unknown")),
                    ),
                    AssistantCitation(
                        source="inverse_design",
                        field_path="candidate_count",
                        value=str(inverse.get("candidate_count", 0)),
                    ),
                    AssistantCitation(
                        source="inverse_design",
                        field_path="request.world_spec",
                        value=str(inverse.get("world_spec", "unknown")),
                    ),
                ]
            )
            if inverse.get("candidate_count", 0) > 0:
                citations.append(
                    AssistantCitation(
                        source="inverse_design",
                        field_path="best_objective_score",
                        value=f"{inverse.get('best_objective_score', 0.0):.6f}",
                    )
                )
                lines.append(
                    "Inverse status="
                    f"{inverse.get('status')} candidates={inverse.get('candidate_count', 0)} "
                    f"best={inverse.get('best_candidate_id')} "
                    f"objective={inverse.get('best_objective_score', 0.0):.6f} "
                    f"feasible_rate={inverse.get('feasible_rate', 0.0):.2%}."
                )
            else:
                lines.append(
                    f"Inverse status={inverse.get('status')} with no ranked candidates yet."
                )

        if supersonic.get("available"):
            citations.extend(
                [
                    AssistantCitation(
                        source="supersonic",
                        field_path="job.status",
                        value=str(supersonic.get("status", "unknown")),
                    ),
                    AssistantCitation(
                        source="supersonic",
                        field_path="summary.final_shock_strength_50",
                        value=f"{supersonic.get('final_shock_strength_50', 0.0):.6f}",
                    ),
                    AssistantCitation(
                        source="supersonic",
                        field_path="derived.shock_strength_slope_last_32",
                        value=f"{supersonic.get('shock_strength_slope_last_32', 0.0):.6f}",
                    ),
                ]
            )
            lines.append(
                "Supersonic status="
                f"{supersonic.get('status')} points={supersonic.get('telemetry_points', 0)} "
                f"shock_strength_50={supersonic.get('final_shock_strength_50', 0.0):.6f} "
                f"shock_reduction_50={supersonic.get('final_shock_reduction_50', 0.0):.6f} "
                f"shock_slope_32={supersonic.get('shock_strength_slope_last_32', 0.0):.6f}."
            )

        if runtime.get("available"):
            citations.extend(
                [
                    AssistantCitation(
                        source="runtime",
                        field_path="step",
                        value=str(runtime.get("step", 0)),
                    ),
                    AssistantCitation(
                        source="runtime",
                        field_path="trust_verified",
                        value=f"{runtime.get('trust_verified', 0.0):.6f}",
                    ),
                    AssistantCitation(
                        source="runtime",
                        field_path="diagnostics.updated_step",
                        value=str(runtime.get("diagnostics_step", 0)),
                    ),
                ]
            )
            lines.append(
                "Runtime summary: step="
                f"{runtime.get('step', 0)}, reward={runtime.get('reward', 0.0):.6f}, "
                f"stress={runtime.get('stress', 0.0):.6f}, trust_verified={runtime.get('trust_verified', 0.0):.6f}."
            )
        else:
            lines.append(
                "Runtime summary: no telemetry snapshot is currently available."
            )

        if request.mode == "risk":
            risk_signals: List[str] = []
            if supersonic.get("available") and supersonic.get("status") == "failed":
                risk_signals.append("supersonic job failed")
            if supersonic.get("available") and supersonic.get("status") == "cancelled":
                risk_signals.append("supersonic job cancelled")
            if supersonic.get("available") and supersonic.get("final_shock_strength_50", 0.0) > 0.20:
                risk_signals.append("high residual shock strength")
            if supersonic.get("available") and supersonic.get("shock_strength_slope_last_32", 0.0) > 0.002:
                risk_signals.append("shock strength trending upward")
            if inverse.get("available") and inverse.get("candidate_count", 0) > 0 and inverse.get("feasible_rate", 0.0) < 0.70:
                risk_signals.append("inverse feasible-rate below target")
            if runtime.get("available") and runtime.get("trust_verified", 0.0) < runtime.get("trust_structural_floor", 0.0):
                risk_signals.append("verified trust below structural floor")
            if risk_signals:
                lines.append("Risk posture: elevated (" + "; ".join(risk_signals) + ").")
            else:
                lines.append("Risk posture: contained in current telemetry window.")
        elif request.mode == "status":
            lines.append("Status mode: reporting live queue, telemetry, and trust surfaces only.")
        elif request.mode == "next_actions":
            lines.append("Next-actions mode: recommendations prioritized for immediate execution.")

        lines.append(self._diagnostic_line(request.diagnostic_mode, runtime))
        lines.append(
            "Grounding policy: response computed only from current ATOM job artifacts and telemetry."
        )
        deterministic_answer = " ".join(lines)

        answer = deterministic_answer
        engine_used = "deterministic"
        if request.engine == "llm_grounded":
            llm_answer = self._llm_synthesize(
                request=request,
                deterministic_answer=deterministic_answer,
                inverse=inverse,
                supersonic=supersonic,
                runtime=runtime,
            )
            if llm_answer:
                answer = llm_answer
                if "Grounding policy" not in answer:
                    answer = (
                        f"{answer} Grounding policy: response computed only from current ATOM job "
                        "artifacts and telemetry."
                    )
                engine_used = "llm_grounded"
            else:
                engine_used = "deterministic_fallback"
                actions.append(
                    "Set ATOM_ASSISTANT_ENABLE_LLM=1 and configure either OPENAI_API_KEY "
                    "or a local OpenAI-compatible endpoint in ATOM_ASSISTANT_OPENAI_URL."
                )

        if not inverse.get("available") and not supersonic.get("available") and not runtime.get("available"):
            actions = [
                "Submit an inverse-design job to populate design metrics.",
                "Submit a supersonic challenge job to populate control telemetry.",
                "Run a short training loop to generate runtime saliency diagnostics.",
            ]

        return AssistantQueryResponse(
            answer=answer,
            mode=request.mode,
            intent=request.intent,
            diagnostic_mode=request.diagnostic_mode,
            engine=engine_used,
            generated_at=_utc_iso(),
            grounded=True,
            citations=citations,
            recommended_actions=actions,
        )


def create_app(workspace_root: Optional[str] = None) -> Any:
    """Create FastAPI app if dependencies are available."""
    try:
        from fastapi import FastAPI, HTTPException, Query, Request
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "FastAPI stack not installed. Install with: pip install fastapi uvicorn"
        ) from exc
    # Ensure Request is available in module globals for annotation resolution
    # during OpenAPI schema generation.
    globals()["Request"] = Request

    base_dir = Path(__file__).resolve().parent
    web_dir = base_dir / "web"
    template_dir = web_dir / "templates"
    static_dir = web_dir / "static"

    if not template_dir.exists() or not static_dir.exists():
        raise RuntimeError(
            "Web assets missing. Expected directories: "
            f"{template_dir} and {static_dir}"
        )

    app = FastAPI(
        title="ATOM Platform API",
        description=(
            "Inverse-design orchestration API with NVIDIA-grade runtime contracts "
            "and platform telemetry."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = Jinja2Templates(directory=str(template_dir))
    resolved_workspace = (
        Path(workspace_root).expanduser().resolve() if workspace_root is not None else None
    )
    service = InverseDesignJobService(workspace_root=resolved_workspace)
    challenge_service = SupersonicChallengeJobService(workspace_root=service.workspace_root)
    assistant_service = GovernedAssistantService(workspace_root=service.workspace_root)
    geometry_store = GeometryAssetStore(workspace_root=service.workspace_root)
    director_pack_enabled = _director_pack_enabled()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Any:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "title": "ATOM Platform Console",
            },
        )

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/worlds")
    async def list_worlds() -> Dict[str, Any]:
        return {"worlds": _load_world_registry()}

    @app.get("/api/v1/geometries")
    async def list_geometries() -> Dict[str, Any]:
        return {
            "generated_at": _utc_iso(),
            "assets": geometry_store.list_assets(),
        }

    @app.post("/api/v1/geometries/upload")
    async def upload_geometry(request: Request) -> Dict[str, Any]:
        filename = str(request.headers.get("x-atom-filename", "geometry.stl")).strip()
        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty geometry upload payload.")
        if len(raw) > _MAX_GEOMETRY_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Geometry exceeds max upload size ({_MAX_GEOMETRY_UPLOAD_BYTES} bytes).",
            )
        try:
            asset = geometry_store.save_upload(filename, raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"asset": asset}

    @app.get("/api/v1/studio/catalog")
    async def studio_catalog() -> Dict[str, Any]:
        worlds = _load_world_registry()
        examples = _load_studio_examples(service.workspace_root, worlds)
        return {
            "generated_at": _utc_iso(),
            "simulators": _build_simulator_catalog(worlds),
            "demos": examples["demos"],
            "examples_dir": examples["examples_dir"],
            "challenge_available": bool(examples["challenge_available"]),
        }

    @app.post("/api/v1/studio/director-pack")
    async def studio_director_pack(request: DirectorPackRequest) -> Dict[str, Any]:
        if not director_pack_enabled:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Director pack endpoint is disabled by default in OSS mode. "
                    "Set ATOM_ENABLE_DIRECTOR_PACK=1 to enable."
                ),
            )
        try:
            return _build_director_pack(service.workspace_root, request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Director pack build failed: {exc}") from exc

    @app.get("/api/v1/runtime/telemetry")
    async def runtime_telemetry() -> Dict[str, Any]:
        return _load_runtime_telemetry_snapshot(service.workspace_root)

    @app.get("/api/v1/inverse-design/jobs")
    async def list_jobs() -> Dict[str, Any]:
        return {"jobs": service.list_jobs()}

    @app.post("/api/v1/inverse-design/jobs", response_model=JobSubmitResponse)
    async def submit_job(request: InverseDesignJobRequest) -> JobSubmitResponse:
        request_payload = request.model_copy(deep=True)
        world_spec = str(request_payload.world_spec).strip().lower()
        world_kwargs = (
            dict(request_payload.world_kwargs)
            if isinstance(request_payload.world_kwargs, dict)
            else {}
        )

        if request_payload.geometry_id:
            asset = geometry_store.resolve(request_payload.geometry_id)
            if asset is None:
                raise HTTPException(status_code=404, detail="Geometry asset not found")
            world_kwargs.setdefault("geometry_asset_path", str(asset.get("path", "")))
            world_kwargs.setdefault("geometry_asset_id", str(asset.get("geometry_id", "")))
            if world_spec.startswith("lbm:custom"):
                if not bool(asset.get("is_stl", False)):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "lbm:custom requires STL geometry. "
                            f"Uploaded extension was '{asset.get('extension', 'unknown')}'."
                        ),
                    )
                world_kwargs["stl_path"] = str(asset.get("path", ""))

        if world_spec.startswith("lbm:custom"):
            stl_path_raw = str(world_kwargs.get("stl_path", "")).strip()
            if not stl_path_raw:
                raise HTTPException(
                    status_code=422,
                    detail="lbm:custom requires world_kwargs.stl_path or geometry_id.",
                )
            stl_path = Path(stl_path_raw)
            if not stl_path.is_absolute():
                stl_path = (service.workspace_root / stl_path).resolve()
            if not stl_path.exists():
                raise HTTPException(
                    status_code=422,
                    detail=f"STL geometry not found at '{stl_path}'.",
                )
            world_kwargs["stl_path"] = str(stl_path)

        normalized_request = request_payload.model_copy(
            update={"world_kwargs": _json_safe_value(world_kwargs)}
        )
        try:
            job = service.submit(normalized_request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JobSubmitResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=_utc_iso(job.created_at),
        )

    @app.get("/api/v1/inverse-design/jobs/{job_id}")
    async def get_job(job_id: str) -> Dict[str, Any]:
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/api/v1/challenges/supersonic/jobs")
    async def list_supersonic_jobs() -> Dict[str, Any]:
        return {"jobs": challenge_service.list_jobs()}

    @app.post("/api/v1/challenges/supersonic/jobs", response_model=JobSubmitResponse)
    async def submit_supersonic_job(request: SupersonicChallengeJobRequest) -> JobSubmitResponse:
        job = challenge_service.submit(request)
        return JobSubmitResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=_utc_iso(job.created_at),
        )

    @app.get("/api/v1/challenges/supersonic/jobs/{job_id}")
    async def get_supersonic_job(job_id: str) -> Dict[str, Any]:
        job = challenge_service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Challenge job not found")
        return job

    @app.get("/api/v1/challenges/supersonic/jobs/{job_id}/telemetry")
    async def get_supersonic_telemetry(
        job_id: str,
        limit: int = Query(default=400, ge=8, le=4096),
    ) -> Dict[str, Any]:
        payload = challenge_service.get_telemetry(job_id, limit=limit)
        if payload is None:
            raise HTTPException(status_code=404, detail="Challenge job not found")
        return payload

    @app.get("/api/v1/challenges/supersonic/jobs/{job_id}/timeline")
    async def get_supersonic_timeline(
        job_id: str,
        limit: int = Query(default=256, ge=1, le=2048),
    ) -> Dict[str, Any]:
        payload = challenge_service.get_timeline(job_id, limit=limit)
        if payload is None:
            raise HTTPException(status_code=404, detail="Challenge job not found")
        return payload

    @app.post("/api/v1/challenges/supersonic/jobs/{job_id}/control")
    async def control_supersonic_job(
        job_id: str,
        request: SupersonicControlRequest,
    ) -> Dict[str, Any]:
        payload = challenge_service.control(job_id, request)
        if payload is None:
            raise HTTPException(status_code=404, detail="Challenge job not found")
        return payload

    @app.post("/api/v1/assistant/query", response_model=AssistantQueryResponse)
    async def assistant_query(request: AssistantQueryRequest) -> AssistantQueryResponse:
        if request.inverse_job_id and service.get_job(request.inverse_job_id) is None:
            raise HTTPException(status_code=404, detail="Inverse-design job not found")
        if request.supersonic_job_id and challenge_service.get_job(request.supersonic_job_id) is None:
            raise HTTPException(status_code=404, detail="Challenge job not found")
        return assistant_service.respond(request, service, challenge_service)

    @app.get("/api/v1/inverse-design/spec-template")
    async def spec_template() -> Dict[str, Any]:
        world_contracts = _world_kwargs_contract_template()
        return {
            "objective_spec": {
                "targets": {"reward_mean": 0.0, "turbulence": 0.0},
                "constraints": {
                    "reward_mean": {"min": -1.0},
                    "action_energy": {"max": 1.0},
                    "termination_ratio": {"max": 0.5},
                },
                "penalties": {
                    "action_energy": 0.2,
                    "reward_std": 0.1,
                    "density_var": 0.05,
                },
                "hard_bounds": {
                    "action_bias": [-1.0, 1.0],
                    "action_gain": [0.0, 1.0],
                    "action_frequency": [0.0, 0.35],
                    "action_phase": [-3.14159, 3.14159],
                },
                "solver_budget": {"iterations": 8, "population": 12, "seed": 42},
            },
            "parameter_space": {
                "action_bias": [-1.0, 1.0],
                "action_gain": [0.0, 1.0],
                "action_frequency": [0.0, 0.35],
                "action_phase": [-3.14159, 3.14159],
                "world_dt": [0.02, 0.30],
            },
            "world_kwargs_contracts": world_contracts,
            "supported_world_specs": sorted(world_contracts.get("world_spec_profile_map", {}).keys()),
        }

    return app


def main() -> None:
    """Dev entrypoint for local API server."""
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("uvicorn is not installed. Install with: pip install uvicorn") from exc
    uvicorn.run("atom.platform.webapp:create_app", factory=True, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
