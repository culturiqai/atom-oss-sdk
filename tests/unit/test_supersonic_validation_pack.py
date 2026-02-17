"""Unit tests for supersonic validation pack orchestration."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import supersonic_validation_pack as svp


class _FakeSupersonicWorld:
    def __init__(self, nx=16, ny=8):
        self.nx = int(nx)
        self.ny = int(ny)
        self.step_count = 0
        self._rho = np.ones((self.nx, self.ny), dtype=np.float32)

    def _build_obs(self, control: float) -> np.ndarray:
        obs = np.zeros((1, 4, self.nx, self.ny, 1), dtype=np.float32)
        obs[0, 0, :, :, 0] = control
        obs[0, 1, :, :, 0] = 0.5 * control
        obs[0, 3, :, :, 0] = self._rho
        return obs

    def reset(self):
        self.step_count = 0
        self._rho = np.ones((self.nx, self.ny), dtype=np.float32)
        mask = np.zeros((1, self.nx, self.ny, 1), dtype=np.float32)
        return self._build_obs(0.0), mask

    def step(self, action):
        control = float(np.asarray(action).reshape(-1)[0])
        self.step_count += 1
        self._rho = self._rho + 0.002 + 0.0001 * control
        info = {
            "shock_strength": 0.2 + 0.01 * self.step_count,
            "shock_reduction": 0.05 - 0.001 * abs(control),
        }
        return self._build_obs(control), 0.1 - abs(control), False, info

    def get_replay_state(self):
        return {
            "step_count": int(self.step_count),
            "rho": np.asarray(self._rho, dtype=np.float32).copy(),
        }

    def set_replay_state(self, state):
        self.step_count = int(state["step_count"])
        self._rho = np.asarray(state["rho"], dtype=np.float32).copy()


def test_validation_pack_passes_on_deterministic_world(monkeypatch):
    monkeypatch.setattr(
        svp,
        "create_world",
        lambda world_spec, **kwargs: _FakeSupersonicWorld(nx=kwargs["nx"], ny=kwargs["ny"]),
    )

    result = svp.run_validation(steps=10, nx=16, ny=8, seed=7, allow_skip=False)
    assert result["status"] == "passed"
    assert result["checks"]["deterministic_reset_replay"] is True
    assert result["checks"]["deterministic_state_replay"] is True
    assert result["checks"]["mass_drift_within_limit"] is True
