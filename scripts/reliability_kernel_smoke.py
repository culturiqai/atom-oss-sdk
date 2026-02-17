#!/usr/bin/env python3
"""Phase-0 reliability smoke checks for ATOM world/runtime contracts.

Checks:
- WorldAdapter schema metadata contract
- 20-step runtime smoke (no crashes)
- deterministic reset+replay with fixed action trace
- deterministic mid-trajectory state restore via get/set replay hooks
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atom_worlds import WorldAdapter, create_world, list_available_worlds


@dataclass
class WorldSmokeResult:
    world_spec: str
    passed: bool
    steps: int
    reset_replay_match: bool
    state_restore_match: bool
    first_trace: List[Tuple[float, float, bool]]
    second_trace: List[Tuple[float, float, bool]]
    replay_trace_a: List[Tuple[float, float, bool]]
    replay_trace_b: List[Tuple[float, float, bool]]
    error: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "world_spec": self.world_spec,
            "passed": self.passed,
            "steps": self.steps,
            "reset_replay_match": self.reset_replay_match,
            "state_restore_match": self.state_restore_match,
            "first_trace": self.first_trace,
            "second_trace": self.second_trace,
            "replay_trace_a": self.replay_trace_a,
            "replay_trace_b": self.replay_trace_b,
            "error": self.error,
        }


def _obs_signature(obs: np.ndarray) -> float:
    arr = np.asarray(obs, dtype=np.float64)
    # Stable scalar signature (small + deterministic).
    return float(np.mean(arr) + 0.123 * np.std(arr) + 0.017 * np.max(arr))


def _trace_rollout(world: WorldAdapter, actions: Sequence[np.ndarray]) -> List[Tuple[float, float, bool]]:
    out: List[Tuple[float, float, bool]] = []
    for action in actions:
        obs, reward, done, _info = world.step(action)
        out.append((float(reward), _obs_signature(obs), bool(done)))
    return out


def _compare_traces(a: Sequence[Tuple[float, float, bool]], b: Sequence[Tuple[float, float, bool]], atol: float = 1e-6) -> bool:
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


def _assert_schema(world: WorldAdapter) -> None:
    schema = world.get_schema_metadata()
    required = {"obs_shape", "obs_dtype", "action_shape", "action_dtype", "reward_dtype"}
    missing = sorted(required - set(schema.keys()))
    if missing:
        raise RuntimeError(f"Schema metadata missing keys: {missing}")


def run_world_smoke(world_spec: str, steps: int, seed: int, nx: int, ny: int, nz: int) -> WorldSmokeResult:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    world = create_world(world_spec, nx=nx, ny=ny, nz=nz)
    _assert_schema(world)

    # Fixed action trace.
    actions = [rng.uniform(-0.75, 0.75, size=(world.action_dim,)).astype(np.float32) for _ in range(steps)]

    # Run 1.
    world.reset()
    first_trace = _trace_rollout(world, actions)

    # Run 2 from reset with same action trace.
    world.reset()
    second_trace = _trace_rollout(world, actions)

    reset_replay_match = _compare_traces(first_trace, second_trace)

    # Mid-trajectory replay restore.
    world.reset()
    split = steps // 2
    _ = _trace_rollout(world, actions[:split])
    replay_state = world.get_replay_state()
    replay_trace_a = _trace_rollout(world, actions[split:])

    world.set_replay_state(replay_state)
    replay_trace_b = _trace_rollout(world, actions[split:])

    state_restore_match = _compare_traces(replay_trace_a, replay_trace_b)

    return WorldSmokeResult(
        world_spec=world_spec,
        passed=reset_replay_match and state_restore_match,
        steps=steps,
        reset_replay_match=reset_replay_match,
        state_restore_match=state_restore_match,
        first_trace=first_trace,
        second_trace=second_trace,
        replay_trace_a=replay_trace_a,
        replay_trace_b=replay_trace_b,
    )


def _select_sim_world(simulation_spec: str) -> str:
    if simulation_spec != "auto":
        return simulation_spec

    available = list_available_worlds()
    for candidate in ("lbm:cylinder", "lbm:fluid", "lbm2d:cylinder"):
        if candidate in available:
            return candidate
    raise RuntimeError("No simulation-backed world available (expected lbm:* or lbm2d:*).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reliability-kernel smoke checks")
    parser.add_argument("--analytical", default="analytical:taylor_green")
    parser.add_argument("--simulation", default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--out", default="validation_outputs/reliability_kernel_smoke.json")
    args = parser.parse_args()

    results: Dict[str, Dict[str, object]] = {}
    overall_pass = True

    checks = [args.analytical, _select_sim_world(args.simulation)]

    for world_spec in checks:
        try:
            res = run_world_smoke(
                world_spec=world_spec,
                steps=args.steps,
                seed=args.seed,
                nx=args.nx,
                ny=args.ny,
                nz=args.nz,
            )
            results[world_spec] = res.to_dict()
            overall_pass = overall_pass and res.passed
        except Exception as exc:  # noqa: BLE001
            overall_pass = False
            results[world_spec] = {
                "world_spec": world_spec,
                "passed": False,
                "error": str(exc),
            }

    payload = {
        "overall_pass": overall_pass,
        "steps": args.steps,
        "seed": args.seed,
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
