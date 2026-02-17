#!/usr/bin/env python3
"""Supersonic solver validation pack (determinism, replay, conservation)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atom_worlds import create_world


@dataclass
class TracePoint:
    reward: float
    shock_strength: float
    shock_reduction: float
    rho_mean: float
    rho_min: float
    rho_max: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "reward": float(self.reward),
            "shock_strength": float(self.shock_strength),
            "shock_reduction": float(self.shock_reduction),
            "rho_mean": float(self.rho_mean),
            "rho_min": float(self.rho_min),
            "rho_max": float(self.rho_max),
        }


def _trace(world: Any, actions: Sequence[np.ndarray]) -> List[TracePoint]:
    points: List[TracePoint] = []
    for action in actions:
        obs, reward, _done, info = world.step(action)
        obs_arr = np.asarray(obs, dtype=np.float32)
        rho = obs_arr[0, 3]
        points.append(
            TracePoint(
                reward=float(np.asarray(reward).reshape(-1)[0]),
                shock_strength=float(info.get("shock_strength", 0.0)),
                shock_reduction=float(info.get("shock_reduction", 0.0)),
                rho_mean=float(np.mean(rho)),
                rho_min=float(np.min(rho)),
                rho_max=float(np.max(rho)),
            )
        )
    return points


def _trace_max_abs_delta(a: Sequence[TracePoint], b: Sequence[TracePoint]) -> float:
    if len(a) != len(b):
        return float("inf")
    worst = 0.0
    for pa, pb in zip(a, b):
        da = abs(pa.reward - pb.reward)
        db = abs(pa.shock_strength - pb.shock_strength)
        dc = abs(pa.rho_mean - pb.rho_mean)
        worst = max(worst, da, db, dc)
    return float(worst)


def run_validation(
    *,
    steps: int = 16,
    nx: int = 64,
    ny: int = 32,
    seed: int = 123,
    allow_skip: bool = False,
) -> Dict[str, Any]:
    n_steps = max(6, int(steps))
    world_kwargs = {
        "nx": int(nx),
        "ny": int(ny),
        "nz": 1,
        "seed": int(seed),
        "noise_amp": 0.0,
        "warmup_steps": 4,
        "max_steps": max(n_steps + 4, 24),
    }

    try:
        world = create_world("supersonic:wedge_d2q25", **world_kwargs)
    except Exception as exc:
        if allow_skip:
            return {
                "status": "skipped",
                "reason": f"{type(exc).__name__}: {exc}",
                "checks": {},
                "metrics": {},
            }
        raise

    actions = [
        np.asarray([0.30 * np.sin(0.27 * i)], dtype=np.float32)
        for i in range(n_steps)
    ]

    obs0, _mask0 = world.reset()
    rho0 = np.asarray(obs0, dtype=np.float32)[0, 3]
    mass0 = float(np.mean(rho0))

    trace_a = _trace(world, actions)
    mass_last = float(trace_a[-1].rho_mean)
    mass_drift_rel = abs(mass_last - mass0) / (abs(mass0) + 1e-8)

    world.reset()
    trace_b = _trace(world, actions)
    deterministic_delta = _trace_max_abs_delta(trace_a, trace_b)

    split = max(2, n_steps // 2)
    world.reset()
    _ = _trace(world, actions[:split])
    replay_state = world.get_replay_state()
    tail_a = _trace(world, actions[split:])
    world.set_replay_state(replay_state)
    tail_b = _trace(world, actions[split:])
    replay_delta = _trace_max_abs_delta(tail_a, tail_b)

    rho_mins = [p.rho_min for p in trace_a]
    rho_maxs = [p.rho_max for p in trace_a]
    shocks = [p.shock_strength for p in trace_a]
    reductions = [p.shock_reduction for p in trace_a]

    checks = {
        "finite_trace": bool(np.isfinite(rho_mins).all() and np.isfinite(rho_maxs).all()),
        "density_positive": bool(min(rho_mins) > 0.0),
        "deterministic_reset_replay": bool(deterministic_delta <= 1e-6),
        "deterministic_state_replay": bool(replay_delta <= 1e-6),
        "mass_drift_within_limit": bool(mass_drift_rel <= 0.15),
        "shock_metric_finite": bool(np.isfinite(shocks).all()),
    }
    passed = all(bool(v) for v in checks.values())

    return {
        "status": "passed" if passed else "failed",
        "checks": checks,
        "metrics": {
            "steps": int(n_steps),
            "seed": int(seed),
            "grid": [int(nx), int(ny), 1],
            "mass_initial": mass0,
            "mass_final": mass_last,
            "mass_drift_rel": mass_drift_rel,
            "deterministic_delta": deterministic_delta,
            "replay_delta": replay_delta,
            "shock_mean": float(np.mean(shocks)),
            "shock_reduction_mean": float(np.mean(reductions)),
            "shock_reduction_peak": float(np.max(reductions)),
            "rho_min": float(min(rho_mins)),
            "rho_max": float(max(rho_maxs)),
            "trace": [p.to_dict() for p in trace_a],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supersonic physics validation pack")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--allow-skip", action="store_true", help="Return skipped status instead of raising")
    parser.add_argument("--output", type=str, default="validation_outputs/supersonic_validation.json")
    args = parser.parse_args()

    result = run_validation(
        steps=args.steps,
        nx=args.nx,
        ny=args.ny,
        seed=args.seed,
        allow_skip=args.allow_skip,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result.get("status"), "output": str(out_path)}, indent=2))
    if result.get("status") == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
