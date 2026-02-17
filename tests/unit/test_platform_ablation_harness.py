"""Unit tests for platform ablation harness helpers."""

import argparse
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_platform_ablations as rpa


def test_normalize_metrics_atom_and_baseline_payloads():
    atom_payload = {
        "summary": {
            "final_reward_50": 0.25,
            "final_stress_50": 0.13,
            "final_divergence_50": 0.031,
            "total_steps": 50,
            "laws_discovered": 2,
            "safety_interventions": 3,
            "safety_fallback_uses": 1,
        },
        "history": {"avg_step_time": 0.01},
    }
    baseline_payload = {
        "final_reward_last_50": -0.5,
        "final_stress_50": 0.22,
        "final_divergence_50": 0.014,
        "safety_interventions": 4,
        "safety_fallback_uses": 1,
        "steps": 50,
    }

    atom_metrics = rpa._normalize_metrics(atom_payload)
    baseline_metrics = rpa._normalize_metrics(baseline_payload)

    assert atom_metrics["final_reward_50"] == 0.25
    assert atom_metrics["laws_discovered"] == 2
    assert atom_metrics["safety_interventions"] == 3
    assert "safety_fallback_rate" in atom_metrics
    assert "safety_mode_safe_hold_rate" in atom_metrics
    assert atom_metrics["final_divergence_50"] == 0.031
    assert baseline_metrics["final_reward_50"] == -0.5
    assert baseline_metrics["final_stress_50"] == 0.22
    assert baseline_metrics["final_divergence_50"] == 0.014
    assert baseline_metrics["safety_interventions"] == 4
    assert baseline_metrics["safety_fallback_uses"] == 1
    assert baseline_metrics["steps_reported"] == 50


def test_aggregate_groups_by_agent_variant():
    records = [
        rpa.RunRecord(
            agent="atom",
            variant="full",
            safety_mode="off",
            seed=0,
            steps=10,
            run_dir=".",
            success=True,
            result_path=None,
            metrics={
                "final_reward_50": 0.1,
                "final_stress_50": 0.2,
                "final_divergence_50": 0.01,
                "safety_interventions": 1,
            },
        ),
        rpa.RunRecord(
            agent="atom",
            variant="full",
            safety_mode="off",
            seed=1,
            steps=10,
            run_dir=".",
            success=True,
            result_path=None,
            metrics={
                "final_reward_50": 0.3,
                "final_stress_50": 0.4,
                "final_divergence_50": 0.03,
                "safety_interventions": 3,
            },
        ),
    ]

    rows = rpa._aggregate(records, bootstrap_samples=256)
    assert len(rows) == 1
    assert rows[0]["agent"] == "atom"
    assert rows[0]["variant"] == "full"
    assert rows[0]["safety_mode"] == "off"
    assert rows[0]["n_runs"] == 2
    assert abs(float(rows[0]["divergence_mean"]) - 0.02) < 1e-12


def test_atom_variant_matrix_includes_production_modes():
    matrix = rpa._atom_variant_matrix("production")
    assert "no_scientist" in matrix
    assert "v1_raw" in matrix
    assert "v2_raw" in matrix
    assert "v2_residual" in matrix
    assert "v2_hybrid" in matrix
    assert "v2_advantage" in matrix


def test_apply_profile_overrides_fast_jax_defaults():
    args = argparse.Namespace(
        profile="fast_jax",
        world=rpa.DEFAULT_WORLD,
        grid=list(rpa.DEFAULT_GRID),
        steps=rpa.DEFAULT_STEPS,
        seeds=list(rpa.DEFAULT_SEEDS),
        safety_shift_warmup=rpa.DEFAULT_SAFETY_SHIFT_WARMUP,
    )
    out = rpa._apply_profile_overrides(args)
    assert out.world == rpa.FAST_JAX_PROFILE["world"]
    assert out.grid == rpa.FAST_JAX_PROFILE["grid"]
    assert out.steps == rpa.FAST_JAX_PROFILE["steps"]
    assert out.seeds == rpa.FAST_JAX_PROFILE["seeds"]
    assert out.safety_shift_warmup == rpa.FAST_JAX_PROFILE["safety_shift_warmup"]


def test_apply_profile_overrides_keeps_explicit_values():
    args = argparse.Namespace(
        profile="fast_jax",
        world="lbm:cylinder",
        grid=[48, 24, 8],
        steps=40,
        seeds=[9],
        safety_shift_warmup=30,
    )
    out = rpa._apply_profile_overrides(args)
    assert out.world == "lbm:cylinder"
    assert out.grid == [48, 24, 8]
    assert out.steps == 40
    assert out.seeds == [9]
    assert out.safety_shift_warmup == 30


def test_atom_command_includes_safety_shift_warmup():
    cmd, result_path = rpa._atom_command(
        world="lbm2d:cylinder",
        grid=(64, 64, 1),
        steps=12,
        seed=3,
        run_dir=Path("/tmp/atom-test"),
        variant_flags=["--ablate-scientist"],
        device="cpu",
        safety_shift_warmup=7,
        disable_safety=True,
        scientist_warmup_steps=0,
        scientist_diag_trust_floor=0.2,
        scientist_diag_trust_floor_start_step=1,
        safety_min_theory_trust=0.01,
        world_u_inlet=0.05,
        world_tau=0.56,
    )
    assert "--safety-shift-warmup" in cmd
    idx = cmd.index("--safety-shift-warmup")
    assert cmd[idx + 1] == "7"
    assert "--disable-safety" in cmd
    assert "--scientist-warmup-steps" in cmd
    assert cmd[cmd.index("--scientist-warmup-steps") + 1] == "0"
    assert "--scientist-diag-trust-floor" in cmd
    assert cmd[cmd.index("--scientist-diag-trust-floor") + 1] == "0.2"
    assert "--scientist-diag-trust-floor-start-step" in cmd
    assert cmd[cmd.index("--scientist-diag-trust-floor-start-step") + 1] == "1"
    assert "--safety-min-theory-trust" in cmd
    assert cmd[cmd.index("--safety-min-theory-trust") + 1] == "0.01"
    assert "--world-u-inlet" in cmd
    assert cmd[cmd.index("--world-u-inlet") + 1] == "0.05"
    assert "--world-tau" in cmd
    assert cmd[cmd.index("--world-tau") + 1] == "0.56"
    assert result_path.name == "results.json"


def test_baseline_command_includes_world_overrides():
    cmd, result_path = rpa._baseline_command(
        script=Path("/tmp/baseline.py"),
        world="lbm2d:cylinder",
        grid=(64, 64, 1),
        steps=20,
        seed=2,
        run_dir=Path("/tmp/base-test"),
        world_u_inlet=0.045,
        world_tau=0.54,
    )
    assert "--world-u-inlet" in cmd
    assert cmd[cmd.index("--world-u-inlet") + 1] == "0.045"
    assert "--world-tau" in cmd
    assert cmd[cmd.index("--world-tau") + 1] == "0.54"
    assert result_path.name == "results.json"


def test_build_paired_safety_report():
    records = [
        rpa.RunRecord(
            agent="atom",
            variant="v2_hybrid",
            safety_mode="off",
            seed=0,
            steps=20,
            run_dir=".",
            success=True,
            result_path=None,
            metrics={
                "final_reward_50": -0.1,
                "final_stress_50": 0.03,
                "final_theory_action_delta_l2_50": 0.2,
                "safety_interventions": 0,
            },
        ),
        rpa.RunRecord(
            agent="atom",
            variant="v2_hybrid",
            safety_mode="on",
            seed=0,
            steps=20,
            run_dir=".",
            success=True,
            result_path=None,
            metrics={
                "final_reward_50": -0.08,
                "final_stress_50": 0.02,
                "final_theory_action_delta_l2_50": 0.1,
                "safety_interventions": 2,
            },
        ),
    ]
    payload = rpa._build_paired_safety_report(records)
    assert len(payload["pairs"]) == 1
    assert payload["pairs"][0]["reward_delta_off_minus_on"] == pytest.approx(-0.02, abs=1e-9)
    assert payload["summary"][0]["n_pairs"] == 1
