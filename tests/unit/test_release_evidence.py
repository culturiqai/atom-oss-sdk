"""Unit tests for release evidence report synthesis."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import generate_release_evidence as gre


def test_release_evidence_passes_with_valid_inputs(tmp_path):
    reliability = tmp_path / "reliability.json"
    reliability.write_text(json.dumps({"overall_pass": True, "steps": 20, "seed": 1}))

    scientific = tmp_path / "scientific.json"
    scientific.write_text(
        json.dumps(
            {
                "summary": {
                    "gates": {"overall_pass": True},
                    "recovery_rate": 0.9,
                    "calibration_error_mean": 0.1,
                    "overfit_rate": 0.1,
                    "false_discovery_rate": 0.1,
                    "seed_perturbation_stability": 0.8,
                }
            }
        )
    )

    safety = tmp_path / "safety.json"
    safety.write_text(json.dumps({"status": "passed"}))

    ablation = tmp_path / "ablation.json"
    ablation.write_text(
        json.dumps(
            {
                "summary": [
                    {"agent": "atom", "reward_mean": 0.5},
                    {"agent": "baseline", "reward_mean": 0.1},
                ]
            }
        )
    )

    args = argparse.Namespace(
        reliability=str(reliability),
        scientific=str(scientific),
        safety=str(safety),
        ablation=str(ablation),
        output_json=str(tmp_path / "release.json"),
        output_md=str(tmp_path / "release.md"),
        min_atom_vs_baseline_reward_delta=0.0,
        profile="dev",
        allow_missing=False,
        check=False,
    )
    out = gre.generate_release_evidence(args)
    assert out["overall_pass"] is True
    assert out["profile"] == "dev"
    assert (tmp_path / "release.json").exists()
    assert (tmp_path / "release.md").exists()


def test_release_evidence_resolves_to_valid_ablation_when_input_invalid(tmp_path, monkeypatch):
    reliability = tmp_path / "reliability.json"
    reliability.write_text(json.dumps({"overall_pass": True, "steps": 20, "seed": 1}))

    scientific = tmp_path / "scientific.json"
    scientific.write_text(
        json.dumps(
            {
                "summary": {
                    "gates": {"overall_pass": True},
                    "recovery_rate": 0.9,
                    "calibration_error_mean": 0.1,
                    "intervention_consistency_mean": 0.8,
                    "overfit_rate": 0.1,
                    "false_discovery_rate": 0.1,
                    "seed_perturbation_stability": 0.8,
                }
            }
        )
    )

    safety = tmp_path / "safety.json"
    safety.write_text(json.dumps({"status": "passed"}))

    benchmark_root = tmp_path / "benchmark_results"
    invalid_dir = benchmark_root / "platform_ablations"
    invalid_dir.mkdir(parents=True, exist_ok=True)
    (invalid_dir / "summary.json").write_text("", encoding="utf-8")

    valid_dir = benchmark_root / "platform_ablations_rc1"
    valid_dir.mkdir(parents=True, exist_ok=True)
    valid_ablation = valid_dir / "summary.json"
    valid_ablation.write_text(
        json.dumps(
            {
                "summary": [
                    {"agent": "atom", "reward_mean": 0.5},
                    {"agent": "baseline", "reward_mean": 0.1},
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(gre, "ROOT", tmp_path)

    args = argparse.Namespace(
        reliability=str(reliability),
        scientific=str(scientific),
        safety=str(safety),
        ablation=str(invalid_dir / "summary.json"),
        output_json=str(tmp_path / "release_fallback.json"),
        output_md=str(tmp_path / "release_fallback.md"),
        min_atom_vs_baseline_reward_delta=0.0,
        profile="ci",
        allow_missing=False,
        check=False,
    )
    out = gre.generate_release_evidence(args)
    assert out["overall_pass"] is True
    assert out["inputs"]["ablation_resolved"] == str(valid_ablation)
    assert out["profile"] == "ci"


def test_release_profile_rejects_allow_missing(tmp_path):
    reliability = tmp_path / "reliability.json"
    reliability.write_text(json.dumps({"overall_pass": True, "steps": 20, "seed": 1}))

    scientific = tmp_path / "scientific.json"
    scientific.write_text(json.dumps({"summary": {"gates": {"overall_pass": True}}}))

    safety = tmp_path / "safety.json"
    safety.write_text(json.dumps({"status": "passed"}))

    ablation = tmp_path / "ablation.json"
    ablation.write_text(
        json.dumps(
            {
                "summary": [
                    {"agent": "atom", "reward_mean": 0.5},
                    {"agent": "baseline", "reward_mean": 0.1},
                ]
            }
        )
    )

    args = argparse.Namespace(
        reliability=str(reliability),
        scientific=str(scientific),
        safety=str(safety),
        ablation=str(ablation),
        output_json=str(tmp_path / "release.json"),
        output_md=str(tmp_path / "release.md"),
        min_atom_vs_baseline_reward_delta=0.0,
        profile="release",
        allow_missing=True,
        check=False,
    )

    try:
        gre.generate_release_evidence(args)
    except ValueError as exc:
        assert "allow_missing is not permitted" in str(exc)
    else:
        raise AssertionError("Expected release profile to reject allow_missing")


def test_release_profile_clamps_ablation_threshold_to_nonnegative(tmp_path):
    reliability = tmp_path / "reliability.json"
    reliability.write_text(json.dumps({"overall_pass": True, "steps": 20, "seed": 1}))

    scientific = tmp_path / "scientific.json"
    scientific.write_text(
        json.dumps(
            {
                "summary": {
                    "gates": {"overall_pass": True},
                    "recovery_rate": 0.9,
                    "calibration_error_mean": 0.1,
                    "intervention_consistency_mean": 0.8,
                    "overfit_rate": 0.1,
                    "false_discovery_rate": 0.1,
                    "seed_perturbation_stability": 0.8,
                }
            }
        )
    )

    safety = tmp_path / "safety.json"
    safety.write_text(json.dumps({"status": "passed"}))

    ablation = tmp_path / "ablation.json"
    ablation.write_text(
        json.dumps(
            {
                "summary": [
                    {"agent": "atom", "reward_mean": 0.0},
                    {"agent": "baseline", "reward_mean": 0.1},
                ]
            }
        )
    )

    args = argparse.Namespace(
        reliability=str(reliability),
        scientific=str(scientific),
        safety=str(safety),
        ablation=str(ablation),
        output_json=str(tmp_path / "release_strict.json"),
        output_md=str(tmp_path / "release_strict.md"),
        min_atom_vs_baseline_reward_delta=-0.5,
        profile="release",
        allow_missing=False,
        check=False,
    )

    out = gre.generate_release_evidence(args)
    assert out["requested_min_atom_vs_baseline_reward_delta"] == -0.5
    assert out["effective_min_atom_vs_baseline_reward_delta"] == 0.0
    assert out["details"]["ablation_vs_baseline"]["overall_pass"] is False
    assert out["overall_pass"] is False
