"""Unit tests for platform ablation plotting pipeline."""

import argparse
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("matplotlib")

from scripts import plot_platform_ablations as ppa


def test_plot_pipeline_writes_outputs(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "agent": "atom",
                        "variant": "v2_hybrid",
                        "reward_mean": 0.4,
                        "reward_std": 0.1,
                        "stress_mean": 0.2,
                        "stress_std": 0.02,
                        "safety_interventions_mean": 1.0,
                    },
                    {
                        "agent": "atom",
                        "variant": "no_scientist",
                        "reward_mean": 0.2,
                        "reward_std": 0.1,
                        "stress_mean": 0.3,
                        "stress_std": 0.03,
                        "safety_interventions_mean": 2.0,
                    },
                    {
                        "agent": "baseline",
                        "variant": "cleanrl_ppo",
                        "reward_mean": 0.1,
                        "reward_std": 0.05,
                        "stress_mean": 0.4,
                        "stress_std": 0.04,
                        "safety_interventions_mean": 3.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "plots"
    args = argparse.Namespace(
        summary=str(summary_path),
        output_dir=str(out_dir),
        title="test",
        runs="",
        bootstrap_samples=400,
        bootstrap_seed=123,
        alpha=0.05,
    )
    payload = ppa.run_plot_pipeline(args)
    assert "plot_meta" in payload
    assert (out_dir / "platform_ablation_dashboard.png").exists()
    assert (out_dir / "platform_ablation_dashboard.pdf").exists()
    assert (out_dir / "platform_ablation_claims.md").exists()
    assert (out_dir / "platform_ablation_significance.json").exists()
    assert (out_dir / "platform_ablation_plots.json").exists()
    assert payload["significance"]["available"] is False


def test_plot_pipeline_reports_significance_from_runs(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "agent": "atom",
                        "variant": "v2_hybrid",
                        "reward_mean": 0.40,
                        "reward_std": 0.01,
                        "stress_mean": 0.20,
                        "stress_std": 0.02,
                        "safety_interventions_mean": 1.0,
                    },
                    {
                        "agent": "baseline",
                        "variant": "cleanrl_ppo",
                        "reward_mean": 0.20,
                        "reward_std": 0.01,
                        "stress_mean": 0.30,
                        "stress_std": 0.02,
                        "safety_interventions_mean": 0.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "agent": "atom",
                        "variant": "v2_hybrid",
                        "success": True,
                        "metrics": {"final_reward_50": 0.41},
                    },
                    {
                        "agent": "atom",
                        "variant": "v2_hybrid",
                        "success": True,
                        "metrics": {"final_reward_50": 0.38},
                    },
                    {
                        "agent": "baseline",
                        "variant": "cleanrl_ppo",
                        "success": True,
                        "metrics": {"final_reward_50": 0.19},
                    },
                    {
                        "agent": "baseline",
                        "variant": "cleanrl_ppo",
                        "success": True,
                        "metrics": {"final_reward_50": 0.21},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "plots"
    args = argparse.Namespace(
        summary=str(summary_path),
        output_dir=str(out_dir),
        title="significance-test",
        runs=str(runs_path),
        bootstrap_samples=300,
        bootstrap_seed=7,
        alpha=0.05,
    )
    payload = ppa.run_plot_pipeline(args)
    assert payload["significance"]["available"] is True
    stats = payload["significance"]["variants"]["v2_hybrid"]
    assert stats["available"] is True
    assert stats["n_atom_runs"] == 2
    assert stats["n_baseline_runs"] == 2
