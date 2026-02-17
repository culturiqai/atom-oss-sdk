"""Unit tests for scientific integrity benchmark harness."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_scientific_integrity_bench as sib


def test_scientific_integrity_benchmark_payload_structure(tmp_path):
    args = argparse.Namespace(
        seeds=[0, 1],
        noise_levels=[0.0],
        samples=96,
        force_sparse=True,
        support_jaccard_threshold=0.3,
        recovery_nrmse_threshold=1.0,
        overfit_margin=0.5,
        action_perturbation=0.25,
        null_trials_per_seed=1,
        null_acceptance_margin=0.05,
        null_min_trust=0.2,
        min_recovery_rate=0.0,
        max_calibration_error=1.0,
        min_intervention_consistency=0.0,
        max_overfit_rate=1.0,
        max_false_discovery_rate=1.0,
        min_seed_stability=0.0,
        output=str(tmp_path / "scientific_integrity.json"),
        check=False,
    )
    payload = sib.run_benchmark(args)
    assert "summary" in payload
    assert "known_cases" in payload
    assert "null_trials" in payload
    summary = payload["summary"]
    assert "gates" in summary
    assert "recovery_rate" in summary
    assert "intervention_consistency_mean" in summary
    assert "false_discovery_rate" in summary
    assert "intervention_consistency_mean" in summary["gates"]


def test_scientific_profile_overrides_fast_jax_defaults():
    args = argparse.Namespace(
        profile="fast_jax",
        seeds=list(sib.DEFAULT_SCI_SEEDS),
        noise_levels=list(sib.DEFAULT_SCI_NOISE),
        samples=sib.DEFAULT_SCI_SAMPLES,
        force_sparse=False,
        null_trials_per_seed=sib.DEFAULT_NULL_TRIALS_PER_SEED,
    )
    out = sib._apply_profile_overrides(args)
    assert out.seeds == sib.FAST_JAX_SCI_PROFILE["seeds"]
    assert out.noise_levels == sib.FAST_JAX_SCI_PROFILE["noise_levels"]
    assert out.samples == sib.FAST_JAX_SCI_PROFILE["samples"]
    assert out.null_trials_per_seed == sib.FAST_JAX_SCI_PROFILE["null_trials_per_seed"]
    assert out.force_sparse is True


def test_scientific_profile_preserves_explicit_values():
    args = argparse.Namespace(
        profile="fast_jax",
        seeds=[7],
        noise_levels=[0.0],
        samples=99,
        force_sparse=True,
        null_trials_per_seed=6,
    )
    out = sib._apply_profile_overrides(args)
    assert out.seeds == [7]
    assert out.noise_levels == [0.0]
    assert out.samples == 99
    assert out.null_trials_per_seed == 6
