"""Production-grade ATOM supersonic wedge control challenge.

This challenge is intentionally thin and auditable:
- Uses canonical `ATOMExperiment` (same pipeline as platform runner)
- Uses `supersonic:wedge_d2q25` world adapter
- Emits deterministic audit artifacts under `challenge_results/`
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SupersonicWedgeChallengeConfig:
    """Configuration for the professional supersonic wedge challenge."""

    max_steps: int = 1000
    headless: bool = True
    world_spec: str = "supersonic:wedge_d2q25"
    grid_shape: Tuple[int, int, int] = (192, 96, 1)
    output_dir: str = "challenge_results"
    experiment_name: str = "supersonic_wedge_challenge"
    batch_size: int = 4
    seed: int = 42
    device: str = "auto"


def _resolve_runner_module() -> Any:
    """Resolve canonical experiment runner from repository root."""
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        import atom_experiment_runner as runner
    except Exception as exc:  # pragma: no cover - runtime import failure
        raise RuntimeError(
            "Unable to import atom_experiment_runner.py. "
            "Run from repository root or ensure project root is on PYTHONPATH."
        ) from exc
    return runner


def run_challenge(
    max_steps: int = 1000,
    headless: bool = False,
    output_dir: str = "challenge_results",
    on_step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_law_callback: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """Run the professional supersonic wedge challenge."""
    cfg = SupersonicWedgeChallengeConfig(
        max_steps=int(max_steps),
        headless=bool(headless),
        output_dir=str(output_dir),
    )
    runner = _resolve_runner_module()

    rewards: List[float] = []
    shocks: List[float] = []
    reductions: List[float] = []
    jet_power: List[float] = []
    discovered_laws: List[str] = []

    def _on_step(
        step: int,
        obs: Any,
        action: Any,
        reward: float,
        info: Dict[str, Any],
        history: Any,
    ) -> None:
        _ = step, obs, action, history
        rewards.append(float(reward))
        shocks.append(float(info.get("shock_strength", 0.0)))
        reductions.append(float(info.get("shock_reduction", 0.0)))
        jet_power.append(float(info.get("jet_power", 0.0)))
        if on_step_callback is not None:
            on_step_callback(
                {
                    "step": int(step),
                    "reward": float(reward),
                    "shock_strength": float(info.get("shock_strength", 0.0)),
                    "shock_reduction": float(info.get("shock_reduction", 0.0)),
                    "jet_power": float(info.get("jet_power", 0.0)),
                    "headless": bool(cfg.headless),
                }
            )

    def _on_law_discovered(law_str: str, step: int) -> None:
        discovered_laws.append(f"step={int(step)} :: {law_str}")
        if on_law_callback is not None:
            on_law_callback(str(law_str), int(step))

    exp_cfg = runner.ExperimentConfig(
        name=cfg.experiment_name,
        seed=cfg.seed,
        world_spec=cfg.world_spec,
        grid_shape=cfg.grid_shape,
        max_steps=cfg.max_steps,
        batch_size=cfg.batch_size,
        output_dir=cfg.output_dir,
        device=cfg.device,
        sleep_interval=80,
        log_interval=10,
        save_interval=400,
    )
    experiment = runner.ATOMExperiment(
        exp_cfg,
        on_step=_on_step,
        on_law_discovered=_on_law_discovered,
    )
    history = experiment.run()

    history_dict = history.to_dict() if hasattr(history, "to_dict") else {}
    final_reward_50 = float(np.mean(rewards[-50:])) if rewards else 0.0
    final_shock_50 = float(np.mean(shocks[-50:])) if shocks else 0.0
    final_reduction_50 = float(np.mean(reductions[-50:])) if reductions else 0.0

    report = {
        "challenge": "supersonic_wedge_control",
        "world_spec": cfg.world_spec,
        "grid_shape": list(cfg.grid_shape),
        "summary": {
            "steps_executed": len(rewards),
            "final_reward_50": final_reward_50,
            "final_shock_strength_50": final_shock_50,
            "final_shock_reduction_50": final_reduction_50,
            "peak_shock_reduction": float(max(reductions)) if reductions else 0.0,
            "mean_jet_power": float(np.mean(np.abs(jet_power))) if jet_power else 0.0,
            "best_law": getattr(history, "best_law", None),
            "laws_discovered": int(len(getattr(history, "laws_discovered", []) or [])),
        },
        "discoveries": discovered_laws,
        "history": {
            "reward": rewards,
            "shock_strength": shocks,
            "shock_reduction": reductions,
            "jet_power": jet_power,
            "atom_history": history_dict,
        },
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "challenge_audit.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ATOM supersonic wedge challenge")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No local OpenCV display (recommended in server/runtime environments)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="challenge_results",
        help="Directory for challenge audit artifact",
    )
    args = parser.parse_args()

    report = run_challenge(
        max_steps=args.steps,
        headless=args.headless,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "audit_path": str(Path(args.output_dir) / "challenge_audit.json"),
                "summary": report["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
