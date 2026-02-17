"""Typed platform contracts for ATOM reliability and product surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def schema_hash_from_names(feature_names: Sequence[str]) -> str:
    """Stable schema hash used in TheoryPacket for replay-safe contracts."""
    payload = "|".join(str(name) for name in feature_names)
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TheoryPacket:
    """Versioned theory/trust packet emitted by Scientist V2."""

    prediction: float
    jacobian: Tuple[float, ...]
    trust_raw: float
    trust_verified: float
    trust_structural_floor: float
    version: str = "theory_packet.v1"
    timestamp: str = field(default_factory=_utc_now_iso)
    feature_schema_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["jacobian"] = list(self.jacobian)
        return data


@dataclass(frozen=True)
class HypothesisRecord:
    """Auditable hypothesis metadata for discovery governance."""

    equation: str
    fit_dataset_id: str
    calibration_dataset_id: str
    validity_window: Dict[str, Any]
    failure_modes: List[str]
    confidence_components: Dict[str, float]
    version: str = "hypothesis_record.v1"
    timestamp: str = field(default_factory=_utc_now_iso)
    scientist_version: str = "scientist_v2"
    feature_schema_hash: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveSpec:
    """Objective/constraint contract for inverse-design workflows."""

    targets: Dict[str, float]
    constraints: Dict[str, Any]
    penalties: Dict[str, float]
    hard_bounds: Dict[str, Tuple[float, float]]
    solver_budget: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlEnvelope:
    """Runtime control safety envelope contract."""

    action_bounds: Mapping[str, Tuple[float, float]]
    state_bounds: Mapping[str, Tuple[float, float]]
    intervention_policy: str
    fallback_policy_id: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["action_bounds"] = dict(self.action_bounds)
        data["state_bounds"] = dict(self.state_bounds)
        return data
