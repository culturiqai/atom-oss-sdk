"""Unit tests for discovery governance and hypothesis registry."""

from atom.mind.governance import (
    ActiveExperimentScheduler,
    DiscoveryGovernance,
    GovernanceDecision,
    HypothesisRegistry,
)


def test_registry_novelty_and_support_stability(tmp_path):
    reg = HypothesisRegistry(path=str(tmp_path / "hypotheses.jsonl"))

    assert reg.novelty_score("x + y") == 1.0
    assert reg.support_stability([1, 2]) == 0.5

    decision = GovernanceDecision(
        approved=True,
        novelty_score=1.0,
        stability_score=0.5,
        intervention_consistency=0.5,
        same_support_refinement=False,
        reasons=[],
    )
    reg.record_decision(
        equation="x + y",
        support=[1, 2],
        status="accepted",
        decision=decision,
    )

    assert reg.novelty_score("x + y") < 0.05
    assert reg.same_support_as_last_accepted([1, 2]) is True
    assert reg.support_stability([1, 2]) == 1.0


def test_governance_rejects_low_intervention_when_verified(tmp_path):
    reg = HypothesisRegistry(path=str(tmp_path / "hypotheses_reject.jsonl"))
    gov = DiscoveryGovernance(reg)

    out = gov.evaluate(
        equation="latent_3*(-0.2)",
        support=[3],
        verified_trust=0.8,
        interventional_metric=0.0,
    )
    assert out.approved is False
    assert "intervention_inconsistency" in out.reasons


def test_governance_allows_same_support_refinement(tmp_path):
    reg = HypothesisRegistry(path=str(tmp_path / "hypotheses_refine.jsonl"))
    gov = DiscoveryGovernance(reg)

    first = gov.evaluate(
        equation="latent_3*(-0.2)",
        support=[3],
        verified_trust=0.1,
        interventional_metric=0.5,
    )
    reg.record_decision(
        equation="latent_3*(-0.2)",
        support=[3],
        status="accepted",
        decision=first,
    )

    refined = gov.evaluate(
        equation="latent_3*(-0.201)",
        support=[3],
        verified_trust=0.6,
        interventional_metric=0.5,
    )
    assert refined.same_support_refinement is True
    assert refined.approved is True


def test_active_scheduler_prioritizes_extreme_probe_under_intervention_uncertainty():
    scheduler = ActiveExperimentScheduler(action_bounds=(-1.0, 1.0), exploration_stride=0.4)
    plan = scheduler.propose(
        verified_trust=0.85,
        novelty_score=0.8,
        stability_score=0.8,
        intervention_consistency=0.05,
        same_support_refinement=False,
        approved=False,
        last_action=0.0,
        support=[3],
    )

    assert abs(plan.target_action) >= 0.4
    assert any(abs(v) >= 0.9 for v in plan.candidate_actions)
    assert plan.expected_information_gain >= 0.49
    assert plan.priority > 0.5


def test_active_scheduler_deprioritizes_when_uncertainty_is_low():
    scheduler = ActiveExperimentScheduler()
    plan = scheduler.propose(
        verified_trust=0.95,
        novelty_score=0.9,
        stability_score=0.95,
        intervention_consistency=0.95,
        same_support_refinement=True,
        approved=True,
        last_action=0.1,
        support=[3],
    )

    assert plan.priority < 0.25
    assert plan.expected_information_gain < 0.25
    assert isinstance(plan.uncertainty_breakdown, dict)
    assert plan.rationale
    assert plan.support == [3]


def test_governance_rejects_high_null_false_discovery_rate(tmp_path):
    reg = HypothesisRegistry(path=str(tmp_path / "hypotheses_fdr.jsonl"))
    gov = DiscoveryGovernance(reg)

    out = gov.evaluate(
        equation="latent_2 + mean_speed",
        support=[2],
        verified_trust=0.7,
        interventional_metric=0.8,
        scientific_evidence={"null_false_discovery_rate": 0.8, "seed_stability": 0.9},
    )
    assert out.approved is False
    assert "null_false_discovery_rate_above_threshold" in out.reasons


def test_governance_requires_scientific_evidence_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOM_GOV_REQUIRE_SCIENTIFIC_EVIDENCE", "1")
    reg = HypothesisRegistry(path=str(tmp_path / "hypotheses_required.jsonl"))
    gov = DiscoveryGovernance(reg)

    out = gov.evaluate(
        equation="mean_speed - turbulence",
        support=[1, 2],
        verified_trust=0.7,
        interventional_metric=0.8,
        scientific_evidence=None,
    )
    assert out.approved is False
    assert "scientific_evidence_missing" in out.reasons


def test_governance_rejects_high_calibration_error_mean(tmp_path):
    reg = HypothesisRegistry(path=str(tmp_path / "hypotheses_calib.jsonl"))
    gov = DiscoveryGovernance(reg)

    out = gov.evaluate(
        equation="latent_2 + mean_speed",
        support=[2],
        verified_trust=0.7,
        interventional_metric=0.8,
        scientific_evidence={
            "null_false_discovery_rate": 0.1,
            "seed_perturbation_stability": 0.9,
            "calibration_error_mean": 0.95,
        },
    )
    assert out.approved is False
    assert "calibration_error_mean_above_threshold" in out.reasons


def test_governance_rejects_low_intervention_consistency_mean(tmp_path):
    reg = HypothesisRegistry(path=str(tmp_path / "hypotheses_intervention.jsonl"))
    gov = DiscoveryGovernance(reg)

    out = gov.evaluate(
        equation="latent_2 + mean_speed",
        support=[2],
        verified_trust=0.7,
        interventional_metric=0.8,
        scientific_evidence={
            "null_false_discovery_rate": 0.1,
            "seed_perturbation_stability": 0.9,
            "intervention_consistency_mean": 0.10,
        },
    )
    assert out.approved is False
    assert "intervention_consistency_mean_below_threshold" in out.reasons
