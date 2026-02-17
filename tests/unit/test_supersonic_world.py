"""Contract tests for the supersonic wedge world adapter."""

from importlib.util import find_spec
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atom_worlds import create_world, list_available_worlds


HAS_JAX = find_spec("jax") is not None


def _trace(world, actions):
    out = []
    for action in actions:
        obs, reward, done, _info = world.step(action)
        out.append((float(reward), float(np.mean(obs)), bool(done)))
    return out


def _trace_matches(a, b, atol=1e-6):
    if len(a) != len(b):
        return False
    for (ra, sa, da), (rb, sb, db) in zip(a, b):
        if not np.isclose(ra, rb, atol=atol):
            return False
        if not np.isclose(sa, sb, atol=atol):
            return False
        if da != db:
            return False
    return True


def test_registry_exposes_supersonic_world_when_jax_available():
    worlds = list_available_worlds()
    if HAS_JAX:
        assert "supersonic:wedge_d2q25" in worlds
    else:
        assert "supersonic:wedge_d2q25" not in worlds


@pytest.mark.skipif(not HAS_JAX, reason="requires jax")
def test_supersonic_world_contract_and_replay():
    world = create_world(
        "supersonic:wedge_d2q25",
        nx=48,
        ny=24,
        nz=1,
        warmup_steps=2,
        max_steps=16,
        seed=123,
        noise_amp=0.0,
    )

    obs, mask = world.reset()
    assert obs.shape == (1, 4, 48, 24, 1)
    assert mask.shape == (1, 48, 24, 1)
    assert np.isfinite(obs).all()
    assert np.isfinite(mask).all()

    actions = [np.array([v], dtype=np.float32) for v in (0.25, -0.1, 0.3, 0.0, -0.2)]

    world.reset()
    trace_a = _trace(world, actions)

    world.reset()
    trace_b = _trace(world, actions)
    assert _trace_matches(trace_a, trace_b)

    world.reset()
    _ = _trace(world, actions[:3])
    replay_state = world.get_replay_state()
    tail_a = _trace(world, actions[3:])
    world.set_replay_state(replay_state)
    tail_b = _trace(world, actions[3:])
    assert _trace_matches(tail_a, tail_b)
