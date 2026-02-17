"""Runtime safety supervision for closed-loop control execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contracts import ControlEnvelope


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


@dataclass
class SafetyDecision:
    """Decision returned by runtime supervisor for one control step."""

    action: np.ndarray
    intervened: bool
    fallback_used: bool
    mode: str = "normal"
    hard_intervention: bool = False
    shift_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    violations: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.astype(np.float32).tolist(),
            "intervened": bool(self.intervened),
            "fallback_used": bool(self.fallback_used),
            "mode": str(self.mode),
            "hard_intervention": bool(self.hard_intervention),
            "shift_score": float(self.shift_score),
            "reasons": list(self.reasons),
            "violations": {str(k): float(v) for k, v in self.violations.items()},
        }


class RuntimeSafetySupervisor:
    """Action/state safety checker with fallback arbitration and shift detection."""

    def __init__(
        self,
        envelope: ControlEnvelope,
        action_dim: int,
        *,
        min_theory_trust: float = 0.08,
        max_stress: float = 8.0,
        shift_z_threshold: float = 8.0,
        shift_warmup: int = 24,
        degrade_window: int = 64,
        cautious_hard_rate: float = 0.30,
        safe_hold_hard_rate: float = 0.60,
        cautious_action_scale: float = 0.5,
        recovery_hysteresis: float = 0.8,
        adaptive_trust: bool = True,
        adaptive_trust_window: int = 128,
        adaptive_trust_quantile: float = 0.25,
        adaptive_trust_floor: float = 0.01,
        low_trust_hard_ratio: float = 0.20,
        low_trust_hard_warmup: int = 24,
        safe_hold_recovery_patience: int = 12,
        safe_hold_max_steps: int = 48,
        safe_hold_recovery_trust_ratio: float = 0.8,
    ):
        self.envelope = envelope
        self.action_dim = int(action_dim)
        self.min_theory_trust = float(min_theory_trust)
        self.max_stress = float(max_stress)
        self.shift_z_threshold = float(shift_z_threshold)
        self.shift_warmup = int(max(4, shift_warmup))
        self.degrade_window = int(max(8, degrade_window))
        self.cautious_hard_rate = float(max(0.01, min(0.95, cautious_hard_rate)))
        self.safe_hold_hard_rate = float(max(self.cautious_hard_rate, min(0.99, safe_hold_hard_rate)))
        self.cautious_action_scale = float(max(0.05, min(1.0, cautious_action_scale)))
        self.recovery_hysteresis = float(max(0.10, min(0.99, recovery_hysteresis)))
        self.adaptive_trust = bool(adaptive_trust)
        self.adaptive_trust_window = int(max(16, adaptive_trust_window))
        self.adaptive_trust_quantile = float(max(0.01, min(0.99, adaptive_trust_quantile)))
        self.adaptive_trust_floor = float(max(0.0, min(self.min_theory_trust, adaptive_trust_floor)))
        self.low_trust_hard_ratio = float(max(0.05, min(1.0, low_trust_hard_ratio)))
        self.low_trust_hard_warmup = int(max(0, low_trust_hard_warmup))
        self.safe_hold_recovery_patience = int(max(1, safe_hold_recovery_patience))
        self.safe_hold_max_steps = int(max(4, safe_hold_max_steps))
        self.safe_hold_recovery_trust_ratio = float(max(0.1, min(1.25, safe_hold_recovery_trust_ratio)))
        self._trust_history: deque[float] = deque(maxlen=self.adaptive_trust_window)
        self._last_effective_min_trust = float(self.min_theory_trust)
        self._safe_hold_steps = 0
        self._safe_hold_recovery_streak = 0

        self._feature_count = 0
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_m2: Optional[np.ndarray] = None
        self._recent_hard_events: deque[int] = deque(maxlen=self.degrade_window)
        self.mode = "normal"
        self.mode_transitions: Dict[str, int] = {
            "normal_to_cautious": 0,
            "cautious_to_safe_hold": 0,
            "safe_hold_to_cautious": 0,
            "cautious_to_normal": 0,
            "normal_to_safe_hold": 0,
            "safe_hold_to_normal": 0,
        }

        self.total_steps = 0
        self.total_interventions = 0
        self.total_fallbacks = 0
        self.reason_histogram: Dict[str, int] = {}
        self.mode_step_counts: Dict[str, int] = {"normal": 0, "cautious": 0, "safe_hold": 0}

    def _action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.full((self.action_dim,), -1.0, dtype=np.float32)
        hi = np.full((self.action_dim,), 1.0, dtype=np.float32)
        for i in range(self.action_dim):
            key = f"action_{i}"
            if key in self.envelope.action_bounds:
                low, high = self.envelope.action_bounds[key]
                lo[i] = np.float32(low)
                hi[i] = np.float32(high)
            elif "default" in self.envelope.action_bounds:
                low, high = self.envelope.action_bounds["default"]
                lo[i] = np.float32(low)
                hi[i] = np.float32(high)
        return lo, hi

    def _clip_action(self, action: np.ndarray) -> Tuple[np.ndarray, float]:
        lo, hi = self._action_bounds()
        clipped = np.clip(action, lo, hi).astype(np.float32)
        delta = float(np.max(np.abs(clipped - action))) if action.size else 0.0
        return clipped, delta

    def _state_violations(self, diagnostics: Mapping[str, Any]) -> Dict[str, float]:
        violations: Dict[str, float] = {}
        for key, bounds in self.envelope.state_bounds.items():
            if key not in diagnostics:
                continue
            value = _safe_float(diagnostics.get(key), default=np.nan)
            if not np.isfinite(value):
                violations[f"{key}_nonfinite"] = 1.0
                continue
            low, high = float(bounds[0]), float(bounds[1])
            if value < low:
                violations[f"{key}_low"] = low - value
            if value > high:
                violations[f"{key}_high"] = value - high
        return violations

    def _check_distribution_shift(self, feature_vector: Optional[Sequence[float]]) -> float:
        if feature_vector is None:
            return 0.0
        vec = np.asarray(feature_vector, dtype=np.float64).reshape(-1)
        if vec.size == 0 or not np.isfinite(vec).all():
            return 0.0

        if self._feature_mean is None:
            self._feature_mean = vec.copy()
            self._feature_m2 = np.zeros_like(vec)
            self._feature_count = 1
            return 0.0

        count = self._feature_count
        mean = self._feature_mean
        m2 = self._feature_m2
        assert mean is not None
        assert m2 is not None

        score = 0.0
        if count >= self.shift_warmup:
            var = m2 / max(1, count - 1)
            std = np.sqrt(np.maximum(var, 1e-8))
            z = np.abs((vec - mean) / std)
            score = float(np.max(z))

        count_new = count + 1
        delta = vec - mean
        mean_new = mean + delta / float(count_new)
        delta2 = vec - mean_new
        m2_new = m2 + delta * delta2

        self._feature_count = count_new
        self._feature_mean = mean_new
        self._feature_m2 = m2_new
        return score

    def _record_reasons(self, reasons: Sequence[str]) -> None:
        for reason in reasons:
            self.reason_histogram[reason] = self.reason_histogram.get(reason, 0) + 1

    def _fallback_action(self, action_like: np.ndarray) -> np.ndarray:
        fallback = np.zeros_like(action_like, dtype=np.float32)
        fallback_clipped, _ = self._clip_action(fallback)
        return fallback_clipped

    def _hard_event_rate(self) -> float:
        if not self._recent_hard_events:
            return 0.0
        denom = max(1, self.degrade_window)
        return float(sum(self._recent_hard_events) / float(denom))

    def _record_mode_transition(self, previous: str, new_mode: str) -> None:
        if previous == new_mode:
            return
        key = f"{previous}_to_{new_mode}"
        if key not in self.mode_transitions:
            self.mode_transitions[key] = 0
        self.mode_transitions[key] += 1

    def _resolve_mode(self, hard_rate: float) -> str:
        target = "normal"
        if hard_rate >= self.safe_hold_hard_rate:
            target = "safe_hold"
        elif hard_rate >= self.cautious_hard_rate:
            target = "cautious"

        # Hysteresis: require lower rate to recover to a less conservative mode.
        if self.mode == "safe_hold" and target != "safe_hold":
            if hard_rate > self.safe_hold_hard_rate * self.recovery_hysteresis:
                return "safe_hold"
        if self.mode in {"safe_hold", "cautious"} and target == "normal":
            if hard_rate > self.cautious_hard_rate * self.recovery_hysteresis:
                return "cautious"
        return target

    def _effective_min_trust(self, trust: float) -> float:
        trust_f = float(max(0.0, min(1.0, _safe_float(trust, default=self.min_theory_trust))))
        self._trust_history.append(trust_f)
        if not self.adaptive_trust:
            self._last_effective_min_trust = float(self.min_theory_trust)
            return self._last_effective_min_trust
        min_samples = max(8, self.adaptive_trust_window // 8)
        if len(self._trust_history) < min_samples:
            self._last_effective_min_trust = float(self.min_theory_trust)
            return self._last_effective_min_trust
        q = float(np.quantile(np.asarray(self._trust_history, dtype=np.float64), self.adaptive_trust_quantile))
        eff = float(np.clip(q, self.adaptive_trust_floor, self.min_theory_trust))
        self._last_effective_min_trust = eff
        return eff

    def review(self, action: np.ndarray, diagnostics: Mapping[str, Any]) -> SafetyDecision:
        self.total_steps += 1
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape[0] != self.action_dim:
            raise ValueError(
                f"Action dim mismatch: expected {self.action_dim}, got {action_arr.shape[0]}"
            )

        reasons: List[str] = []
        violations: Dict[str, float] = {}

        if not np.isfinite(action_arr).all():
            reasons.append("action_nonfinite")
            violations["action_nonfinite"] = 1.0
            action_arr = np.nan_to_num(action_arr, nan=0.0, posinf=1.0, neginf=-1.0)

        clipped_action, clip_delta = self._clip_action(action_arr)
        if clip_delta > 0.0:
            reasons.append("action_clipped")
            violations["action_clip_delta"] = clip_delta

        trust = _safe_float(diagnostics.get("theory_trust", 1.0), default=1.0)
        scientist_warmup_active = bool(diagnostics.get("scientist_warmup_active", False))
        effective_min_trust = self._effective_min_trust(trust)
        if (not scientist_warmup_active) and trust < effective_min_trust:
            reasons.append("low_theory_trust")
            violations["theory_trust_gap"] = effective_min_trust - trust
        low_trust_hard_cutoff = max(self.adaptive_trust_floor, effective_min_trust * self.low_trust_hard_ratio)
        low_trust_hard_enabled = bool(self.total_steps >= self.low_trust_hard_warmup)
        low_trust_hard = bool(
            (not scientist_warmup_active)
            and low_trust_hard_enabled
            and trust < low_trust_hard_cutoff
        )

        stress = _safe_float(diagnostics.get("stress", 0.0), default=0.0)
        if stress > self.max_stress:
            reasons.append("high_stress")
            violations["stress_overflow"] = stress - self.max_stress

        state_viol = self._state_violations(diagnostics)
        if state_viol:
            reasons.append("state_bound_violation")
            violations.update(state_viol)

        shift_score = self._check_distribution_shift(diagnostics.get("feature_vector"))
        if shift_score > self.shift_z_threshold:
            reasons.append("distribution_shift")
            violations["feature_shift_z"] = shift_score

        hard_intervention = any(
            reason in {
                "action_nonfinite",
                "high_stress",
                "state_bound_violation",
                "distribution_shift",
            }
            for reason in reasons
        )
        hard_intervention = bool(hard_intervention or low_trust_hard)
        self._recent_hard_events.append(1 if hard_intervention else 0)
        hard_rate = self._hard_event_rate()
        previous_mode = str(self.mode)
        self.mode = self._resolve_mode(hard_rate)
        if self.mode == "safe_hold":
            self._safe_hold_steps += 1
        else:
            self._safe_hold_steps = 0
            self._safe_hold_recovery_streak = 0

        low_trust_only = bool(
            len(reasons) > 0
            and set(reasons).issubset({"low_theory_trust"})
        )
        if self.mode == "safe_hold":
            if (not hard_intervention) and trust >= (effective_min_trust * self.safe_hold_recovery_trust_ratio):
                self._safe_hold_recovery_streak += 1
            elif low_trust_only and (not low_trust_hard):
                self._safe_hold_recovery_streak += 1
            else:
                self._safe_hold_recovery_streak = 0
            if (
                self._safe_hold_recovery_streak >= self.safe_hold_recovery_patience
                or self._safe_hold_steps >= self.safe_hold_max_steps
            ):
                self.mode = "cautious"
        self._record_mode_transition(previous_mode, self.mode)
        if self.mode not in self.mode_step_counts:
            self.mode_step_counts[self.mode] = 0
        self.mode_step_counts[self.mode] += 1

        final_action = clipped_action
        fallback_used = False
        if self.mode == "safe_hold":
            if "mode_safe_hold" not in reasons:
                reasons.append("mode_safe_hold")
            final_action = self._fallback_action(clipped_action)
            fallback_used = True
        elif hard_intervention:
            final_action = self._fallback_action(clipped_action)
            fallback_used = True
        elif self.mode == "cautious":
            if self.cautious_action_scale < 0.999:
                scaled = clipped_action * np.float32(self.cautious_action_scale)
                final_action, scale_delta = self._clip_action(scaled)
                reasons.append("mode_cautious_scale")
                violations["cautious_scale_delta"] = float(scale_delta)

        intervened = bool(fallback_used or ("action_clipped" in reasons))
        if intervened:
            self.total_interventions += 1
        if fallback_used:
            self.total_fallbacks += 1
        if reasons:
            self._record_reasons(reasons)

        return SafetyDecision(
            action=final_action,
            intervened=intervened,
            fallback_used=fallback_used,
            mode=str(self.mode),
            hard_intervention=bool(hard_intervention),
            shift_score=float(shift_score),
            reasons=reasons,
            violations=violations,
        )

    def snapshot(self) -> Dict[str, Any]:
        hard_rate = self._hard_event_rate()
        total_mode_steps = max(1, int(sum(self.mode_step_counts.values())))
        mode_step_fraction = {
            str(k): float(v) / float(total_mode_steps)
            for k, v in sorted(self.mode_step_counts.items())
        }
        return {
            "steps": int(self.total_steps),
            "interventions": int(self.total_interventions),
            "fallback_uses": int(self.total_fallbacks),
            "mode": str(self.mode),
            "hard_event_rate": float(hard_rate),
            "degrade_window": int(self.degrade_window),
            "mode_transitions": dict(self.mode_transitions),
            "mode_step_counts": dict(self.mode_step_counts),
            "mode_step_fraction": mode_step_fraction,
            "reason_histogram": dict(self.reason_histogram),
            "envelope": self.envelope.to_dict(),
            "policy": {
                "min_theory_trust": float(self.min_theory_trust),
                "effective_min_theory_trust": float(self._last_effective_min_trust),
                "max_stress": float(self.max_stress),
                "shift_z_threshold": float(self.shift_z_threshold),
                "shift_warmup": int(self.shift_warmup),
                "cautious_hard_rate": float(self.cautious_hard_rate),
                "safe_hold_hard_rate": float(self.safe_hold_hard_rate),
                "cautious_action_scale": float(self.cautious_action_scale),
                "recovery_hysteresis": float(self.recovery_hysteresis),
                "adaptive_trust": bool(self.adaptive_trust),
                "adaptive_trust_window": int(self.adaptive_trust_window),
                "adaptive_trust_quantile": float(self.adaptive_trust_quantile),
                "adaptive_trust_floor": float(self.adaptive_trust_floor),
                "low_trust_hard_ratio": float(self.low_trust_hard_ratio),
                "low_trust_hard_warmup": int(self.low_trust_hard_warmup),
                "safe_hold_recovery_patience": int(self.safe_hold_recovery_patience),
                "safe_hold_max_steps": int(self.safe_hold_max_steps),
                "safe_hold_recovery_trust_ratio": float(self.safe_hold_recovery_trust_ratio),
            },
        }
