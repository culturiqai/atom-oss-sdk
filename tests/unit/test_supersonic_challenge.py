"""Tests for professional supersonic challenge orchestration wiring."""

import json
from pathlib import Path

import atom.challenges.supersonic_wedge_challenge as challenge


class _DummyHistory:
    def __init__(self):
        self.reward = [0.3, 0.2, 0.25]
        self.stress = [0.05, 0.04, 0.04]
        self.theory_score = [0.1, 0.2, 0.3]
        self.theory_trust = [0.2, 0.3, 0.35]
        self.laws_discovered = ["reward ~ a+b"]
        self.best_law = "reward ~ a+b"
        self.step_times = [0.01, 0.01, 0.01]

    def to_dict(self):
        return {
            "reward": self.reward,
            "stress": self.stress,
            "theory_score": self.theory_score,
            "theory_trust": self.theory_trust,
            "laws_discovered": self.laws_discovered,
        }


class _DummyRunner:
    class ExperimentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ATOMExperiment:
        def __init__(self, config, on_step=None, on_law_discovered=None):
            self.config = config
            self.on_step = on_step
            self.on_law_discovered = on_law_discovered

        def run(self):
            hist = _DummyHistory()
            if self.on_step is not None:
                for step in range(3):
                    self.on_step(
                        step,
                        None,
                        None,
                        0.2 + 0.01 * step,
                        {
                            "shock_strength": 0.15 - 0.01 * step,
                            "shock_reduction": 0.10 + 0.05 * step,
                            "jet_power": 0.003 * step,
                        },
                        hist,
                    )
            if self.on_law_discovered is not None:
                self.on_law_discovered("reward ~ a+b", 2)
            return hist


def test_run_challenge_writes_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(challenge, "_resolve_runner_module", lambda: _DummyRunner)
    report = challenge.run_challenge(max_steps=3, headless=True, output_dir=str(tmp_path))

    assert report["challenge"] == "supersonic_wedge_control"
    assert report["world_spec"] == "supersonic:wedge_d2q25"
    assert report["summary"]["steps_executed"] == 3
    assert report["summary"]["best_law"] == "reward ~ a+b"
    assert report["summary"]["laws_discovered"] == 1

    out_path = Path(tmp_path) / "challenge_audit.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["summary"]["steps_executed"] == 3
