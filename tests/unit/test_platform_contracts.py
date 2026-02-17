"""Unit tests for platform-level contract dataclasses."""

from atom.platform.contracts import (
    ControlEnvelope,
    HypothesisRecord,
    ObjectiveSpec,
    TheoryPacket,
    schema_hash_from_names,
)


def test_schema_hash_is_stable():
    names = ["action", "mean_speed", "latent_0"]
    h1 = schema_hash_from_names(names)
    h2 = schema_hash_from_names(list(names))
    assert h1 == h2
    assert len(h1) == 16


def test_theory_packet_to_dict_shape():
    pkt = TheoryPacket(
        prediction=0.25,
        jacobian=(0.1, -0.2, 0.0),
        trust_raw=0.7,
        trust_verified=0.6,
        trust_structural_floor=0.2,
        feature_schema_hash="abc123def4567890",
    )

    d = pkt.to_dict()
    assert d["version"] == "theory_packet.v1"
    assert isinstance(d["jacobian"], list)
    assert d["jacobian"] == [0.1, -0.2, 0.0]


def test_hypothesis_record_to_dict_roundtrip():
    rec = HypothesisRecord(
        equation="latent_3 * (-0.2)",
        fit_dataset_id="wake_buffer:120",
        calibration_dataset_id="conformal_holdout",
        validity_window={"verification_window": 20, "warmup_steps": 50},
        failure_modes=["low_verified_trust", "structural_shift_detected"],
        confidence_components={"verified_trust": 0.55, "baseline_r2": 0.91},
    )

    d = rec.to_dict()
    assert d["equation"] == "latent_3 * (-0.2)"
    assert d["validity_window"]["verification_window"] == 20
    assert d["version"] == "hypothesis_record.v1"
    assert d["scientist_version"] == "scientist_v2"


def test_objective_spec_and_control_envelope_to_dict():
    obj = ObjectiveSpec(
        targets={"drag": 0.1},
        constraints={"lift": {"min": 0.5}},
        penalties={"mass": 0.2},
        hard_bounds={"temperature": (250.0, 400.0)},
        solver_budget={"max_evals": 1000},
    )
    env = ControlEnvelope(
        action_bounds={"throttle": (0.0, 1.0)},
        state_bounds={"pressure": (0.8, 1.2)},
        intervention_policy="manual_override",
        fallback_policy_id="pid_v1",
    )

    obj_d = obj.to_dict()
    env_d = env.to_dict()

    assert obj_d["hard_bounds"]["temperature"] == (250.0, 400.0)
    assert env_d["action_bounds"]["throttle"] == (0.0, 1.0)
