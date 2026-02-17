"""Discovery governance utilities for Scientist V2.

Phase-1 scope:
- Hypothesis registry with persistent JSONL provenance
- Novelty scoring against historical equations
- Stability and intervention consistency checks for law adoption
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple



def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()



def _canonical_equation(equation: str) -> str:
    return " ".join(str(equation).strip().split())



def _equation_hash(equation: str) -> str:
    return sha1(_canonical_equation(equation).encode("utf-8")).hexdigest()[:16]



def _to_support_set(support: Iterable[int]) -> Set[int]:
    return {int(i) for i in support}



def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / float(len(union))


logger = logging.getLogger("atom.governance")


@dataclass
class GovernanceDecision:
    approved: bool
    novelty_score: float
    stability_score: float
    intervention_consistency: float
    same_support_refinement: bool
    reasons: List[str]
    null_false_discovery_rate: Optional[float] = None
    seed_perturbation_stability: Optional[float] = None
    calibration_error_mean: Optional[float] = None
    intervention_consistency_mean: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": bool(self.approved),
            "novelty_score": float(self.novelty_score),
            "stability_score": float(self.stability_score),
            "intervention_consistency": float(self.intervention_consistency),
            "same_support_refinement": bool(self.same_support_refinement),
            "reasons": list(self.reasons),
            "null_false_discovery_rate": (
                None
                if self.null_false_discovery_rate is None
                else float(self.null_false_discovery_rate)
            ),
            "seed_perturbation_stability": (
                None
                if self.seed_perturbation_stability is None
                else float(self.seed_perturbation_stability)
            ),
            "calibration_error_mean": (
                None
                if self.calibration_error_mean is None
                else float(self.calibration_error_mean)
            ),
            "intervention_consistency_mean": (
                None
                if self.intervention_consistency_mean is None
                else float(self.intervention_consistency_mean)
            ),
        }


@dataclass
class ExperimentPlan:
    """Action-level proposal for uncertainty-reducing data collection."""

    target_action: float
    candidate_actions: List[float]
    priority: float
    expected_information_gain: float
    uncertainty_breakdown: Dict[str, float]
    rationale: List[str]
    support: List[int]
    timestamp: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_action": float(self.target_action),
            "candidate_actions": [float(v) for v in self.candidate_actions],
            "priority": float(self.priority),
            "expected_information_gain": float(self.expected_information_gain),
            "uncertainty_breakdown": {
                str(k): float(v) for k, v in self.uncertainty_breakdown.items()
            },
            "rationale": list(self.rationale),
            "support": [int(v) for v in self.support],
            "timestamp": str(self.timestamp),
        }


def _clip01(value: float, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not (v == v):
        return float(default)
    return float(max(0.0, min(1.0, v)))


def _maybe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if not (v == v):  # NaN
        return None
    return float(v)


class HypothesisRegistry:
    """Persistent registry for candidate/accepted/rejected hypotheses."""

    def __init__(self, path: Optional[str] = None):
        default_path = os.getenv("ATOM_HYPOTHESIS_LOG", "logs/hypotheses.jsonl")
        self.path = Path(path or default_path)
        self._persistence_enabled = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._persistence_enabled = False
            logger.warning(
                "HypothesisRegistry persistence disabled (mkdir failed for %s): %s",
                self.path,
                exc,
            )

        self._records: List[Dict[str, Any]] = []
        if self._persistence_enabled:
            self._load_existing()

    def _load_existing(self) -> None:
        if not self._persistence_enabled:
            return
        if not self.path.exists():
            return

        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    self._records.append(rec)
            except Exception:
                continue

    def _append(self, record: Dict[str, Any]) -> None:
        if self._persistence_enabled:
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as exc:
                self._persistence_enabled = False
                logger.warning(
                    "HypothesisRegistry append failed for %s (falling back to in-memory only): %s",
                    self.path,
                    exc,
                )

        # Always retain in-memory history for runtime governance behavior.
        self._records.append(record)

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def last_accepted(self) -> Optional[Dict[str, Any]]:
        for rec in reversed(self._records):
            if rec.get("status") == "accepted":
                return rec
        return None

    def novelty_score(self, equation: str) -> float:
        if not self._records:
            return 1.0

        canon = _canonical_equation(equation)
        best_similarity = 0.0
        for rec in self._records:
            prev = _canonical_equation(str(rec.get("equation", "")))
            sim = SequenceMatcher(None, canon, prev).ratio()
            if sim > best_similarity:
                best_similarity = sim

        return float(max(0.0, 1.0 - best_similarity))

    def support_stability(self, support: Iterable[int]) -> float:
        accepted = [r for r in self._records if r.get("status") == "accepted"]
        if not accepted:
            return 0.5

        cur = _to_support_set(support)
        best = 0.0
        for rec in accepted:
            prev_support = _to_support_set(rec.get("support", []))
            best = max(best, _jaccard(cur, prev_support))
        return float(best)

    def same_support_as_last_accepted(self, support: Iterable[int]) -> bool:
        last = self.last_accepted()
        if last is None:
            return False
        return _to_support_set(last.get("support", [])) == _to_support_set(support)

    def record_decision(
        self,
        equation: str,
        support: Iterable[int],
        status: str,
        decision: GovernanceDecision,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec = {
            "timestamp": _utc_now_iso(),
            "equation": str(equation),
            "equation_hash": _equation_hash(equation),
            "support": sorted(int(i) for i in support),
            "status": str(status),
            "decision": decision.to_dict(),
        }
        if extra:
            rec.update(extra)
        self._append(rec)


class DiscoveryGovernance:
    """Evaluates candidate laws before adoption in StructuralScientist."""

    def __init__(self, registry: HypothesisRegistry):
        self.registry = registry

        self.min_novelty = float(os.getenv("ATOM_GOV_MIN_NOVELTY", "0.05"))
        self.min_stability = float(os.getenv("ATOM_GOV_MIN_STABILITY", "0.25"))
        self.min_intervention = float(os.getenv("ATOM_GOV_MIN_INTERVENTION", "0.30"))
        self.exploration_trust = float(os.getenv("ATOM_GOV_EXPLORATION_TRUST", "0.20"))
        self.max_null_false_discovery_rate = float(
            os.getenv("ATOM_GOV_MAX_NULL_FALSE_DISCOVERY_RATE", "0.35")
        )
        self.min_seed_perturbation_stability = float(
            os.getenv("ATOM_GOV_MIN_SEED_PERTURBATION_STABILITY", "0.35")
        )
        self.max_calibration_error_mean = float(
            os.getenv("ATOM_GOV_MAX_CALIBRATION_ERROR_MEAN", "0.60")
        )
        self.min_intervention_consistency_mean = float(
            os.getenv("ATOM_GOV_MIN_INTERVENTION_CONSISTENCY_MEAN", "0.35")
        )
        self.require_scientific_evidence = (
            str(os.getenv("ATOM_GOV_REQUIRE_SCIENTIFIC_EVIDENCE", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
        )

    def _evaluate_scientific_evidence(
        self,
        scientific_evidence: Optional[Mapping[str, Any]],
    ) -> Tuple[
        bool,
        List[str],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        reasons: List[str] = []
        null_fdr: Optional[float] = None
        seed_stability: Optional[float] = None
        calibration_error_mean: Optional[float] = None
        intervention_consistency_mean: Optional[float] = None

        if scientific_evidence is not None:
            null_fdr = _maybe_float(
                scientific_evidence.get(
                    "null_false_discovery_rate",
                    scientific_evidence.get("false_discovery_rate"),
                )
            )
            seed_stability = _maybe_float(
                scientific_evidence.get(
                    "seed_perturbation_stability",
                    scientific_evidence.get("seed_stability"),
                )
            )
            calibration_error_mean = _maybe_float(
                scientific_evidence.get("calibration_error_mean")
            )
            intervention_consistency_mean = _maybe_float(
                scientific_evidence.get("intervention_consistency_mean")
            )

        if self.require_scientific_evidence and (
            null_fdr is None or seed_stability is None
        ):
            reasons.append("scientific_evidence_missing")
            return (
                False,
                reasons,
                null_fdr,
                seed_stability,
                calibration_error_mean,
                intervention_consistency_mean,
            )

        ok = True
        if null_fdr is not None and null_fdr > self.max_null_false_discovery_rate:
            ok = False
            reasons.append("null_false_discovery_rate_above_threshold")

        if (
            seed_stability is not None
            and seed_stability < self.min_seed_perturbation_stability
        ):
            ok = False
            reasons.append("seed_perturbation_stability_below_threshold")

        if (
            calibration_error_mean is not None
            and calibration_error_mean > self.max_calibration_error_mean
        ):
            ok = False
            reasons.append("calibration_error_mean_above_threshold")

        if (
            intervention_consistency_mean is not None
            and intervention_consistency_mean < self.min_intervention_consistency_mean
        ):
            ok = False
            reasons.append("intervention_consistency_mean_below_threshold")

        return (
            ok,
            reasons,
            null_fdr,
            seed_stability,
            calibration_error_mean,
            intervention_consistency_mean,
        )

    def evaluate(
        self,
        equation: str,
        support: Sequence[int],
        verified_trust: float,
        interventional_metric: float,
        scientific_evidence: Optional[Mapping[str, Any]] = None,
    ) -> GovernanceDecision:
        novelty = self.registry.novelty_score(equation)
        stability = self.registry.support_stability(support)
        same_support = self.registry.same_support_as_last_accepted(support)

        # Interventional metric can be absent/neutral in early runs.
        if interventional_metric < 0.0 or interventional_metric > 1.0:
            interventional_metric = 0.5

        novelty_pass = novelty >= self.min_novelty or same_support
        stability_pass = stability >= self.min_stability or verified_trust < self.exploration_trust
        intervention_pass = interventional_metric >= self.min_intervention
        (
            evidence_pass,
            evidence_reasons,
            null_fdr,
            seed_stability,
            calibration_error_mean,
            intervention_consistency_mean,
        ) = (
            self._evaluate_scientific_evidence(scientific_evidence)
        )

        reasons: List[str] = []
        if not novelty_pass:
            reasons.append("novelty_below_threshold")
        if not stability_pass:
            reasons.append("stability_below_threshold")
        if not intervention_pass:
            reasons.append("intervention_inconsistency")
        reasons.extend(evidence_reasons)

        return GovernanceDecision(
            approved=bool(
                novelty_pass and stability_pass and intervention_pass and evidence_pass
            ),
            novelty_score=float(novelty),
            stability_score=float(stability),
            intervention_consistency=float(interventional_metric),
            same_support_refinement=bool(same_support),
            reasons=reasons,
            null_false_discovery_rate=null_fdr,
            seed_perturbation_stability=seed_stability,
            calibration_error_mean=calibration_error_mean,
            intervention_consistency_mean=intervention_consistency_mean,
        )


class ActiveExperimentScheduler:
    """Uncertainty-driven action probe scheduler for discovery loops."""

    def __init__(
        self,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        exploration_stride: float = 0.5,
    ):
        low, high = float(action_bounds[0]), float(action_bounds[1])
        if low >= high:
            raise ValueError("action_bounds must satisfy low < high")
        self.low = low
        self.high = high
        self.exploration_stride = max(0.05, float(exploration_stride))

    def propose(
        self,
        *,
        verified_trust: float,
        novelty_score: float,
        stability_score: float,
        intervention_consistency: float,
        same_support_refinement: bool,
        approved: Optional[bool],
        last_action: float,
        support: Iterable[int],
    ) -> ExperimentPlan:
        trust_unc = 1.0 - _clip01(verified_trust, default=0.0)
        intervention_unc = 1.0 - _clip01(intervention_consistency, default=0.5)
        stability_unc = 1.0 - _clip01(stability_score, default=0.5)
        novelty_deficit = 1.0 - _clip01(novelty_score, default=0.5)

        uncertainty = (
            0.40 * trust_unc
            + 0.35 * intervention_unc
            + 0.20 * stability_unc
            + 0.05 * novelty_deficit
        )

        if approved is False:
            uncertainty = max(uncertainty, 0.60)
        if same_support_refinement:
            uncertainty *= 0.90
        uncertainty = _clip01(uncertainty, default=0.5)

        span = self.high - self.low
        step = max(self.exploration_stride * span, 0.10)
        anchor = max(self.low, min(self.high, float(last_action)))

        raw_candidates = [self.low, self.high, 0.0, anchor, anchor + step, anchor - step]
        candidates: List[float] = []
        for cand in raw_candidates:
            c = max(self.low, min(self.high, float(cand)))
            if all(abs(c - prev) > 1e-6 for prev in candidates):
                candidates.append(c)

        def candidate_score(candidate: float) -> float:
            distance = abs(candidate - anchor) / (span + 1e-8)
            edge_bonus = 0.10 if abs(candidate) >= 0.80 * max(abs(self.low), abs(self.high)) else 0.0
            return float(uncertainty * (0.35 + 0.65 * distance) + edge_bonus * intervention_unc)

        scored = [(candidate_score(c), c) for c in candidates]
        expected_information_gain, target_action = max(scored, key=lambda item: item[0])
        priority = _clip01(0.5 * uncertainty + 0.5 * expected_information_gain, default=uncertainty)

        rationale: List[str] = []
        if intervention_unc > 0.5:
            rationale.append("probe_action_extremes_for_intervention_consistency")
        if trust_unc > 0.5:
            rationale.append("low_verified_trust_requires_disambiguating_data")
        if stability_unc > 0.5:
            rationale.append("support_stability_is_low_collect_more_regime_data")
        if novelty_deficit > 0.7:
            rationale.append("low_novelty_candidate_requires_counterexample_sampling")
        if not rationale:
            rationale.append("maintain_monitoring_with_low_uncertainty_probe")

        return ExperimentPlan(
            target_action=float(target_action),
            candidate_actions=[float(v) for v in candidates],
            priority=float(priority),
            expected_information_gain=float(_clip01(expected_information_gain, default=uncertainty)),
            uncertainty_breakdown={
                "trust": float(trust_unc),
                "intervention": float(intervention_unc),
                "stability": float(stability_unc),
                "novelty": float(novelty_deficit),
            },
            rationale=rationale,
            support=sorted(int(v) for v in support),
        )
