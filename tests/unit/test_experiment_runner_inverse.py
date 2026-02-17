"""Tests for inverse-design wiring in atom_experiment_runner."""

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import atom_experiment_runner as aer


class _DummyBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_mode = "linear"
        self.eyes = nn.Linear(1, 1)
        self.liquid = nn.Linear(1, 1)
        self.actor_mu = nn.Linear(1, 1)
        self._actor_log_std = nn.Parameter(torch.zeros(1, 1))
        self.critic = nn.Linear(1, 1)

    @property
    def actor_log_std(self):
        return self._actor_log_std

    def apply_hebbian_growth(self):
        return None


class _DummyStructuralScientist:
    def __init__(self, scientist, variable_names, **kwargs):
        self.scientist = scientist
        self.variable_names = variable_names
        self.kwargs = dict(kwargs)

    def shutdown(self):
        return None


class _FakeWorld:
    def __init__(self, kwargs):
        self.kwargs = dict(kwargs)
        self.action_dim = 1
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        obs = np.zeros((1, 4, 6, 6, 2), dtype=np.float32)
        mask = np.zeros((1, 6, 6, 2), dtype=np.float32)
        return obs, mask

    def step(self, action):
        self.step_count += 1
        a = float(np.asarray(action).reshape(-1)[0])
        obs = np.zeros((1, 4, 6, 6, 2), dtype=np.float32)
        obs[:, 0] = a
        obs[:, 1] = 0.5 * a
        obs[:, 2] = 0.0
        obs[:, 3] = 1.0 + 0.1 * a
        reward = 0.2 - abs(a)
        done = self.step_count >= 5
        return obs, reward, done, {"mask": np.zeros((1, 6, 6, 2), dtype=np.float32)}


def test_inverse_design_uses_world_adapter_rollouts(monkeypatch, tmp_path):
    world_calls = []

    def _fake_create_world(world_spec, **kwargs):
        world_calls.append((world_spec, dict(kwargs)))
        return _FakeWorld(kwargs)

    monkeypatch.setattr(aer, "create_world", _fake_create_world)
    monkeypatch.setattr(aer, "create_brain_from_config", lambda **_: _DummyBrain())
    monkeypatch.setattr(aer, "create_memory_from_config", lambda *_: object())
    monkeypatch.setattr(aer, "AtomScientist", lambda **_: object())
    monkeypatch.setattr(aer, "StructuralScientist", _DummyStructuralScientist)

    config = aer.ExperimentConfig(
        name="inverse_design_test",
        world_spec="analytical:taylor_green",
        grid_shape=(8, 8, 4),
        max_steps=4,
        output_dir=str(tmp_path),
        device="cpu",
    )

    experiment = aer.ATOMExperiment(config)
    report = experiment.run_inverse_design(
        backend="evolutionary",
        iterations=2,
        population=4,
        top_k=2,
        rollout_steps=6,
    )

    assert report["backend"] == "evolutionary"
    assert len(report["candidates"]) == 8
    assert len(report["top_candidates"]) == 2
    assert world_calls, "Expected create_world to be called for candidate simulation"
    assert all(call[0] == "analytical:taylor_green" for call in world_calls)
    assert all(call[1]["nx"] == 8 and call[1]["ny"] == 8 and call[1]["nz"] == 4 for call in world_calls)
    candidate_calls = world_calls[1:]
    assert candidate_calls
    assert all("dt" in call[1] for call in candidate_calls)

    report_path = Path(tmp_path) / "inverse_design_test" / "inverse_design_report.json"
    assert report_path.exists()
