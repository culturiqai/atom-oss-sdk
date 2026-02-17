"""Contract-focused tests for StructuralScientist outputs."""

import math
import numpy as np
import pytest

from atom.mind.scientist_v2 import StructuralScientist
from atom.platform.contracts import HypothesisRecord, TheoryPacket, schema_hash_from_names


class _DummyV1Scientist:
    def __init__(self, law: str, score: float = 0.2):
        self.best_law = law
        self.best_law_score = score
        self.short_term_X = [np.zeros(11, dtype=np.float32) for _ in range(64)]
        self.conformal = None
        self.observed_targets = []

    def predict_theory(self, features):
        _ = features
        return 0.25, 0.8

    def observe(self, features, target_metric):
        _ = features
        self.observed_targets.append(float(target_metric))

    def ponder(self):
        return None

    def fit_offline(self, X, y):
        _ = (X, y)

    def shutdown(self):
        return None


def _variable_names():
    return ["action", "mean_speed", "turbulence"] + [f"latent_{i}" for i in range(8)]


def test_structural_scientist_emits_theory_packet_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOM_HYPOTHESIS_LOG", str(tmp_path / "hypotheses.jsonl"))
    names = _variable_names()
    base = _DummyV1Scientist(law="latent_3*(-0.2)")
    s2 = StructuralScientist(
        scientist=base,
        variable_names=names,
        warmup_steps=0,
    )

    features = np.zeros(len(names), dtype=np.float32)
    out = s2.get_signal(features=features, reward=0.1, action=0.0, device="cpu")

    assert out["packet_version"] == "theory_packet.v1"
    assert out["feature_schema_hash"] == schema_hash_from_names(names)
    assert isinstance(out["packet"], TheoryPacket)
    assert isinstance(s2.get_theory_packet(), TheoryPacket)
    assert len(out["packet"].jacobian) == 8
    assert "trust_structural_floor" in out
    assert isinstance(out["governance"], dict)
    assert isinstance(out["experiment_plan"], dict)
    assert "target_action" in out["experiment_plan"]
    assert "expected_information_gain" in out["experiment_plan"]
    assert isinstance(s2.get_experiment_plan(), dict)


def test_structural_scientist_compiles_hypothesis_record_on_new_law(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOM_HYPOTHESIS_LOG", str(tmp_path / "hypotheses.jsonl"))
    names = _variable_names()
    base = _DummyV1Scientist(law="-0.045/latent_6")
    s2 = StructuralScientist(
        scientist=base,
        variable_names=names,
        warmup_steps=0,
    )

    out = s2.get_signal(features=np.ones(len(names), dtype=np.float32), reward=0.2, action=0.0, device="cpu")

    rec = out["hypothesis_record"]
    assert isinstance(rec, HypothesisRecord)
    assert rec.equation == base.best_law
    assert "verification_window" in rec.validity_window
    assert "verified_trust" in rec.confidence_components
    assert rec.feature_schema_hash == schema_hash_from_names(names)
    assert rec.provenance.get("packet_version") == "theory_packet.v1"


def test_structural_scientist_raw_target_mode(monkeypatch):
    monkeypatch.setenv("ATOM_SCIENTIST_DISCOVERY_TARGET_MODE", "raw")
    monkeypatch.setenv("ATOM_HYPOTHESIS_LOG", "/tmp/atom_hypotheses_raw_target_test.jsonl")
    names = _variable_names()
    base = _DummyV1Scientist(law="mean_speed*2.0 - turbulence")
    s2 = StructuralScientist(scientist=base, variable_names=names, warmup_steps=0)

    features = np.ones(len(names), dtype=np.float32)
    reward = 0.42
    s2.observe_and_verify(features=features, reward=reward, prediction=0.1, action=0.0)
    out = s2.get_signal(features=features, reward=reward, action=0.0, device="cpu")

    assert base.observed_targets
    assert base.observed_targets[-1] == pytest.approx(reward, abs=1e-6)
    assert out["discovery_target_mode"] == "raw"


def test_structural_scientist_advantage_target_mode(monkeypatch):
    monkeypatch.setenv("ATOM_SCIENTIST_DISCOVERY_TARGET_MODE", "advantage")
    monkeypatch.setenv("ATOM_SCIENTIST_REWARD_EMA_BETA", "0.5")
    monkeypatch.setenv("ATOM_HYPOTHESIS_LOG", "/tmp/atom_hypotheses_advantage_target_test.jsonl")
    names = _variable_names()
    base = _DummyV1Scientist(law="mean_speed + latent_0")
    s2 = StructuralScientist(scientist=base, variable_names=names, warmup_steps=0)

    features = np.ones(len(names), dtype=np.float32)
    s2.observe_and_verify(features=features, reward=1.0, prediction=0.1, action=0.0)
    s2.observe_and_verify(features=features, reward=1.2, prediction=0.1, action=0.0)
    out = s2.get_signal(features=features, reward=1.2, action=0.0, device="cpu")

    assert len(base.observed_targets) >= 2
    # First reward initializes EMA, second should be a positive improvement.
    assert base.observed_targets[0] == pytest.approx(0.0, abs=1e-6)
    assert base.observed_targets[1] == pytest.approx(0.2, abs=1e-6)
    assert out["discovery_target_mode"] == "advantage"


def test_structural_scientist_loads_scientific_evidence_hint_from_file(monkeypatch, tmp_path):
    evidence_path = tmp_path / "scientific_integrity.json"
    evidence_path.write_text(
        (
            '{"summary":{"false_discovery_rate":0.12,'
            '"seed_perturbation_stability":0.74,'
            '"calibration_error_mean":0.21,'
            '"intervention_consistency_mean":0.66}}'
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("ATOM_GOV_NULL_FDR_HINT", raising=False)
    monkeypatch.delenv("ATOM_GOV_SEED_STABILITY_HINT", raising=False)
    monkeypatch.setenv("ATOM_SCI_EVIDENCE_PATH", str(evidence_path))
    monkeypatch.setenv("ATOM_HYPOTHESIS_LOG", str(tmp_path / "hypotheses.jsonl"))

    names = _variable_names()
    base = _DummyV1Scientist(law="latent_0 + latent_1")
    s2 = StructuralScientist(scientist=base, variable_names=names, warmup_steps=0)
    assert s2._scientific_evidence_hint["null_false_discovery_rate"] == pytest.approx(0.12, abs=1e-9)
    assert s2._scientific_evidence_hint["seed_perturbation_stability"] == pytest.approx(0.74, abs=1e-9)
    assert s2._scientific_evidence_hint["calibration_error_mean"] == pytest.approx(0.21, abs=1e-9)
    assert s2._scientific_evidence_hint["intervention_consistency_mean"] == pytest.approx(0.66, abs=1e-9)


def test_structural_scientist_diagnostic_trust_floor_is_gated_by_start_step_and_warmup():
    names = _variable_names()
    base = _DummyV1Scientist(law="latent_0 + latent_1")

    s2 = StructuralScientist(
        scientist=base,
        variable_names=names,
        warmup_steps=2,
        diagnostic_trust_floor=0.35,
        diagnostic_trust_floor_start_step=1,
    )
    features = np.ones(len(names), dtype=np.float32)

    out1 = s2.get_signal(features=features, reward=0.1, action=0.0, device="cpu")
    assert out1["in_warmup"] is True
    assert out1["trust"] == pytest.approx(0.0, abs=1e-9)
    assert out1["diagnostic_trust_floor_applied"] == pytest.approx(0.0, abs=1e-9)

    out2 = s2.get_signal(features=features, reward=0.1, action=0.0, device="cpu")
    assert out2["in_warmup"] is True
    assert out2["trust"] == pytest.approx(0.0, abs=1e-9)
    assert out2["diagnostic_trust_floor_applied"] == pytest.approx(0.0, abs=1e-9)

    out3 = s2.get_signal(features=features, reward=0.1, action=0.0, device="cpu")
    assert out3["in_warmup"] is False
    assert out3["trust"] >= 0.35
    assert out3["diagnostic_trust_floor_applied"] == pytest.approx(0.35, abs=1e-9)


def test_structural_scientist_blends_raw_and_verified_trust_without_collapse(monkeypatch):
    monkeypatch.setenv("ATOM_GOV_MIN_NOVELTY", "0.0")
    names = _variable_names()
    base = _DummyV1Scientist(law="mean_speed*2.0 - turbulence")
    s2 = StructuralScientist(
        scientist=base,
        variable_names=names,
        warmup_steps=0,
    )
    features = np.ones(len(names), dtype=np.float32)

    out = None
    for _ in range(8):
        # Constant reward keeps variance tiny and used to collapse verified trust.
        out = s2.get_signal(features=features, reward=0.1, action=0.0, device="cpu")

    assert out is not None
    assert out["has_law"] is True
    assert out["raw_trust"] > 0.0
    assert out["verified_trust"] >= 0.0
    # With blended trust, the final trust should recover above zero.
    assert out["trust"] > 0.1
    assert "holdout_rmse" in out["verifier_metrics"]
    assert "holdout_correlation" in out["verifier_metrics"]
    assert "out_of_sample_trust" in out["verifier_metrics"]


def test_structural_scientist_signal_trust_scale_modes_are_explicit():
    names = _variable_names()
    features = np.ones(len(names), dtype=np.float32)

    base_off = _DummyV1Scientist(law="latent_3*(-0.2)")
    s2_off = StructuralScientist(
        scientist=base_off,
        variable_names=names,
        warmup_steps=0,
        signal_trust_scale_mode="off",
    )
    out_off = s2_off.get_signal(features=features, reward=0.1, action=0.0, device="cpu")
    t_off = out_off["tensor"].detach().cpu().numpy()
    assert out_off["signal_trust_scale_mode"] == "off"
    assert out_off["signal_content_scale"] == pytest.approx(1.0, abs=1e-9)

    base_full = _DummyV1Scientist(law="latent_3*(-0.2)")
    s2_full = StructuralScientist(
        scientist=base_full,
        variable_names=names,
        warmup_steps=0,
        signal_trust_scale_mode="full",
    )
    out_full = s2_full.get_signal(features=features, reward=0.1, action=0.0, device="cpu")
    t_full = out_full["tensor"].detach().cpu().numpy()
    assert out_full["signal_trust_scale_mode"] == "full"
    assert out_full["signal_content_scale"] == pytest.approx(out_full["trust"], rel=1e-6, abs=1e-6)

    base_sqrt = _DummyV1Scientist(law="latent_3*(-0.2)")
    s2_sqrt = StructuralScientist(
        scientist=base_sqrt,
        variable_names=names,
        warmup_steps=0,
        signal_trust_scale_mode="sqrt",
    )
    out_sqrt = s2_sqrt.get_signal(features=features, reward=0.1, action=0.0, device="cpu")
    t_sqrt = out_sqrt["tensor"].detach().cpu().numpy()
    assert out_sqrt["signal_trust_scale_mode"] == "sqrt"
    assert out_sqrt["signal_content_scale"] == pytest.approx(
        math.sqrt(max(0.0, float(out_sqrt["trust"]))),
        rel=1e-6,
        abs=1e-6,
    )

    # Content dims are [prediction + 8 jacobian]; trust is final dim.
    off_content_norm = float(np.linalg.norm(t_off[0, :9]))
    full_content_norm = float(np.linalg.norm(t_full[0, :9]))
    sqrt_content_norm = float(np.linalg.norm(t_sqrt[0, :9]))
    assert full_content_norm <= off_content_norm + 1e-6
    assert full_content_norm <= sqrt_content_norm + 1e-6
    assert sqrt_content_norm <= off_content_norm + 1e-6
