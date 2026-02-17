"""Unit tests for runtime safety supervisor."""

import numpy as np

from atom.platform.contracts import ControlEnvelope
from atom.platform.safety import RuntimeSafetySupervisor


def _make_supervisor(**overrides: object) -> RuntimeSafetySupervisor:
    envelope = ControlEnvelope(
        action_bounds={"action_0": (-1.0, 1.0)},
        state_bounds={"rho_min": (0.05, 100.0), "rho_max": (0.0, 8.0)},
        intervention_policy="runtime_safety_supervisor.v1",
        fallback_policy_id="fallback:zero_action",
    )
    kwargs = dict(
        envelope=envelope,
        action_dim=1,
        min_theory_trust=0.1,
        max_stress=5.0,
        shift_z_threshold=10.0,
        shift_warmup=4,
    )
    kwargs.update(overrides)
    return RuntimeSafetySupervisor(**kwargs)


def test_action_clip_without_fallback():
    sup = _make_supervisor()
    decision = sup.review(
        np.asarray([1.5], dtype=np.float32),
        diagnostics={
            "theory_trust": 0.8,
            "stress": 0.1,
            "rho_min": 1.0,
            "rho_max": 1.2,
            "feature_vector": [0.1, 0.2, 0.3],
        },
    )
    assert decision.intervened is True
    assert decision.fallback_used is False
    assert np.isclose(decision.action[0], 1.0)


def test_low_trust_triggers_fallback():
    sup = _make_supervisor(low_trust_hard_warmup=0)
    decision = sup.review(
        np.asarray([0.3], dtype=np.float32),
        diagnostics={
            "theory_trust": 0.01,
            "stress": 0.1,
            "rho_min": 1.0,
            "rho_max": 1.2,
            "feature_vector": [0.1, 0.2, 0.3],
        },
    )
    assert decision.intervened is True
    assert decision.fallback_used is True
    assert np.isclose(decision.action[0], 0.0)
    snap = sup.snapshot()
    assert snap["interventions"] >= 1
    assert snap["fallback_uses"] >= 1
    assert "mode_step_fraction" in snap
    assert "effective_min_theory_trust" in snap["policy"]


def test_low_trust_is_not_hard_during_warmup_window():
    sup = _make_supervisor(low_trust_hard_warmup=24)
    decision = sup.review(
        np.asarray([0.3], dtype=np.float32),
        diagnostics={
            "theory_trust": 0.01,
            "stress": 0.1,
            "rho_min": 1.0,
            "rho_max": 1.2,
            "feature_vector": [0.1, 0.2, 0.3],
        },
    )
    assert "low_theory_trust" in decision.reasons
    assert decision.fallback_used is False


def test_scientist_warmup_bypasses_low_trust_intervention():
    sup = _make_supervisor(low_trust_hard_warmup=0)
    decision = sup.review(
        np.asarray([0.3], dtype=np.float32),
        diagnostics={
            "theory_trust": 0.01,
            "stress": 0.1,
            "rho_min": 1.0,
            "rho_max": 1.2,
            "scientist_warmup_active": True,
            "feature_vector": [0.1, 0.2, 0.3],
        },
    )
    assert "low_theory_trust" not in decision.reasons
    assert decision.fallback_used is False


def test_marginal_low_trust_enters_cautious_without_forced_fallback():
    sup = _make_supervisor()
    decision = sup.review(
        np.asarray([0.3], dtype=np.float32),
        diagnostics={
            "theory_trust": 0.06,
            "stress": 0.1,
            "rho_min": 1.0,
            "rho_max": 1.2,
            "feature_vector": [0.1, 0.2, 0.3],
        },
    )
    # Trust is below nominal threshold, but not severely low.
    # Supervisor should mark the issue without always forcing fallback.
    assert "low_theory_trust" in decision.reasons
    assert decision.fallback_used is False


def test_cautious_mode_scales_action_after_hard_event_burst():
    envelope = ControlEnvelope(
        action_bounds={"action_0": (-1.0, 1.0)},
        state_bounds={"rho_min": (0.05, 100.0), "rho_max": (0.0, 8.0)},
        intervention_policy="runtime_safety_supervisor.v1",
        fallback_policy_id="fallback:zero_action",
    )
    sup = RuntimeSafetySupervisor(
        envelope=envelope,
        action_dim=1,
        min_theory_trust=0.1,
        max_stress=5.0,
        shift_z_threshold=10.0,
        shift_warmup=4,
        degrade_window=10,
        cautious_hard_rate=0.2,
        safe_hold_hard_rate=0.8,
        cautious_action_scale=0.5,
        low_trust_hard_warmup=0,
    )

    # Trigger several hard interventions to move into cautious mode.
    for _ in range(3):
        sup.review(
            np.asarray([0.4], dtype=np.float32),
            diagnostics={
                "theory_trust": 0.01,
                "stress": 0.1,
                "rho_min": 1.0,
                "rho_max": 1.1,
                "feature_vector": [0.1, 0.2, 0.3],
            },
        )

    decision = sup.review(
        np.asarray([0.8], dtype=np.float32),
        diagnostics={
            "theory_trust": 0.9,
            "stress": 0.1,
            "rho_min": 1.0,
            "rho_max": 1.1,
            "feature_vector": [0.1, 0.2, 0.3],
        },
    )
    assert decision.mode == "cautious"
    assert decision.fallback_used is False
    assert np.isclose(decision.action[0], 0.4, atol=1e-5)
    assert "mode_cautious_scale" in decision.reasons


def test_safe_hold_mode_forces_zero_action_and_recovers():
    envelope = ControlEnvelope(
        action_bounds={"action_0": (-1.0, 1.0)},
        state_bounds={"rho_min": (0.05, 100.0), "rho_max": (0.0, 8.0)},
        intervention_policy="runtime_safety_supervisor.v1",
        fallback_policy_id="fallback:zero_action",
    )
    sup = RuntimeSafetySupervisor(
        envelope=envelope,
        action_dim=1,
        min_theory_trust=0.1,
        max_stress=5.0,
        shift_z_threshold=10.0,
        shift_warmup=4,
        degrade_window=8,
        cautious_hard_rate=0.2,
        safe_hold_hard_rate=0.5,
        recovery_hysteresis=0.8,
        low_trust_hard_warmup=0,
    )

    # Persistent hard events should push mode to safe_hold.
    for _ in range(5):
        sup.review(
            np.asarray([0.6], dtype=np.float32),
            diagnostics={
                "theory_trust": 0.01,
                "stress": 0.1,
                "rho_min": 1.0,
                "rho_max": 1.1,
                "feature_vector": [0.1, 0.2, 0.3],
            },
        )

    held = sup.review(
        np.asarray([0.6], dtype=np.float32),
        diagnostics={
            "theory_trust": 0.9,
            "stress": 0.1,
            "rho_min": 1.0,
            "rho_max": 1.1,
            "feature_vector": [0.1, 0.2, 0.3],
        },
    )
    assert held.mode == "safe_hold"
    assert held.fallback_used is True
    assert np.isclose(held.action[0], 0.0)
    assert "mode_safe_hold" in held.reasons

    # Sustained healthy behavior should eventually recover from safe_hold.
    recovered = None
    for _ in range(20):
        recovered = sup.review(
            np.asarray([0.6], dtype=np.float32),
            diagnostics={
                "theory_trust": 0.9,
                "stress": 0.1,
                "rho_min": 1.0,
                "rho_max": 1.1,
                "feature_vector": [0.1, 0.2, 0.3],
            },
        )
    assert recovered is not None
    assert recovered.mode in {"normal", "cautious"}
    snap = sup.snapshot()
    assert snap["mode"] in {"normal", "cautious"}
    assert isinstance(snap["mode_transitions"], dict)
