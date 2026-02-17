#!/usr/bin/env python3
"""Production ablation harness for ATOM vs PPO baselines.

Outputs:
- <out>/manifest.json
- <out>/runs.json
- <out>/summary.json
- <out>/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ATOM_RUNNER = ROOT / "atom_experiment_runner.py"
BASELINE_CLEANRL = ROOT / "baseline_cleanrl_ppo.py"
BASELINE_SB3 = ROOT / "baseline_sb3_ppo.py"
VENV_PYTHON = ROOT / "venv" / "bin" / "python"
DEFAULT_WORLD = "analytical:taylor_green"
DEFAULT_GRID = [32, 32, 16]
DEFAULT_STEPS = 200
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_SAFETY_SHIFT_WARMUP = 24
FAST_JAX_PROFILE = {
    "world": "lbm2d:cylinder",
    "grid": [64, 64, 1],
    "steps": 72,
    "seeds": [0, 1],
    "safety_shift_warmup": 8,
}


@dataclass
class RunRecord:
    agent: str
    variant: str
    safety_mode: str
    seed: int
    steps: int
    run_dir: str
    success: bool
    result_path: Optional[str]
    metrics: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "variant": self.variant,
            "safety_mode": self.safety_mode,
            "seed": self.seed,
            "steps": self.steps,
            "run_dir": self.run_dir,
            "success": self.success,
            "result_path": self.result_path,
            "metrics": self.metrics,
            "error": self.error,
        }


def _run_cmd(cmd: Sequence[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _run_cmd_env(cmd: Sequence[str], cwd: Path, env: Dict[str, str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _python_bin() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def _atom_variant_matrix(variant_set: str) -> Dict[str, Dict[str, Any]]:
    if variant_set == "legacy":
        return {
            "full": {"flags": [], "env": {}},
            "no_scientist": {"flags": ["--ablate-scientist"], "env": {}},
            "no_symplectic": {"flags": ["--ablate-symplectic"], "env": {}},
            "no_trust_gate": {"flags": ["--ablate-trust-gate"], "env": {}},
        }
    return {
        "v2_hybrid": {
            "flags": [],
            "env": {
                "ATOM_SCIENTIST_SIGNAL_MODE": "v2",
                "ATOM_SCIENTIST_DISCOVERY_TARGET_MODE": "hybrid",
            },
        },
        "v2_residual": {
            "flags": [],
            "env": {
                "ATOM_SCIENTIST_SIGNAL_MODE": "v2",
                "ATOM_SCIENTIST_DISCOVERY_TARGET_MODE": "residual",
            },
        },
        "v2_raw": {
            "flags": [],
            "env": {
                "ATOM_SCIENTIST_SIGNAL_MODE": "v2",
                "ATOM_SCIENTIST_DISCOVERY_TARGET_MODE": "raw",
            },
        },
        "v2_advantage": {
            "flags": [],
            "env": {
                "ATOM_SCIENTIST_SIGNAL_MODE": "v2",
                "ATOM_SCIENTIST_DISCOVERY_TARGET_MODE": "advantage",
            },
        },
        "v1_raw": {
            "flags": [],
            "env": {
                "ATOM_SCIENTIST_SIGNAL_MODE": "v1",
            },
        },
        "no_scientist": {
            "flags": ["--ablate-scientist"],
            "env": {},
        },
    }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(payload.get("summary", {}))
    history = dict(payload.get("history", {}))
    history_divergence = history.get("divergence", [])
    if isinstance(history_divergence, list) and history_divergence:
        history_div_arr = np.asarray(history_divergence, dtype=np.float64)
        history_div_last50 = (
            history_div_arr[-50:] if len(history_div_arr) >= 50 else history_div_arr
        )
        history_divergence_50 = float(np.mean(history_div_last50))
    else:
        history_divergence_50 = 0.0

    if "final_reward_50" in summary:
        final_reward = float(summary.get("final_reward_50", 0.0))
        final_stress = float(summary.get("final_stress_50", 0.0))
        final_divergence = float(summary.get("final_divergence_50", history_divergence_50))
        safety_interventions = int(summary.get("safety_interventions", 0))
        safety_fallback_uses = int(summary.get("safety_fallback_uses", 0))
        safety_intervention_rate = float(summary.get("safety_intervention_rate", 0.0))
        safety_fallback_rate = float(summary.get("safety_fallback_rate", 0.0))
        safety_hard_event_rate = float(summary.get("safety_hard_event_rate", 0.0))
        safety_mode_normal_rate = float(summary.get("safety_mode_normal_rate", 0.0))
        safety_mode_cautious_rate = float(summary.get("safety_mode_cautious_rate", 0.0))
        safety_mode_safe_hold_rate = float(summary.get("safety_mode_safe_hold_rate", 0.0))
        final_theory_action_delta_l2_50 = float(summary.get("final_theory_action_delta_l2_50", 0.0))
        final_theory_action_delta_frac_50 = float(
            summary.get("final_theory_action_delta_frac_50", 0.0)
        )
        final_theory_verifier_oos_trust_50 = float(
            summary.get("final_theory_verifier_oos_trust_50", 0.0)
        )
    else:
        final_reward = float(payload.get("final_reward_last_50", 0.0))
        final_stress = float(payload.get("final_stress_50", 0.0))
        final_divergence = float(payload.get("final_divergence_50", history_divergence_50))
        safety_interventions = int(payload.get("safety_interventions", 0))
        safety_fallback_uses = int(payload.get("safety_fallback_uses", 0))
        steps_rep = int(summary.get("total_steps", payload.get("steps", 0)))
        denom = float(max(1, steps_rep))
        safety_intervention_rate = float(safety_interventions) / denom
        safety_fallback_rate = float(safety_fallback_uses) / denom
        safety_hard_event_rate = float(payload.get("safety_hard_event_rate", 0.0))
        safety_mode_normal_rate = float(payload.get("safety_mode_normal_rate", 0.0))
        safety_mode_cautious_rate = float(payload.get("safety_mode_cautious_rate", 0.0))
        safety_mode_safe_hold_rate = float(payload.get("safety_mode_safe_hold_rate", 0.0))
        final_theory_action_delta_l2_50 = float(payload.get("final_theory_action_delta_l2_50", 0.0))
        final_theory_action_delta_frac_50 = float(
            payload.get("final_theory_action_delta_frac_50", 0.0)
        )
        final_theory_verifier_oos_trust_50 = float(
            payload.get("final_theory_verifier_oos_trust_50", 0.0)
        )

    return {
        "final_reward_50": final_reward,
        "final_stress_50": final_stress,
        "final_divergence_50": final_divergence,
        "steps_reported": int(summary.get("total_steps", payload.get("steps", 0))),
        "laws_discovered": int(summary.get("laws_discovered", 0)),
        "safety_interventions": safety_interventions,
        "safety_fallback_uses": safety_fallback_uses,
        "safety_intervention_rate": safety_intervention_rate,
        "safety_fallback_rate": safety_fallback_rate,
        "safety_hard_event_rate": safety_hard_event_rate,
        "safety_mode_normal_rate": safety_mode_normal_rate,
        "safety_mode_cautious_rate": safety_mode_cautious_rate,
        "safety_mode_safe_hold_rate": safety_mode_safe_hold_rate,
        "final_theory_action_delta_l2_50": final_theory_action_delta_l2_50,
        "final_theory_action_delta_frac_50": final_theory_action_delta_frac_50,
        "final_theory_verifier_oos_trust_50": final_theory_verifier_oos_trust_50,
        "avg_step_time": float(history.get("avg_step_time", 0.0)),
    }


def _mean_std_sem_ci(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    rng_seed: int,
) -> Tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    sem = float(std / np.sqrt(max(1, arr.size)))
    if arr.size <= 1:
        return mean, std, sem, mean, mean
    n_boot = max(100, int(bootstrap_samples))
    rng = np.random.default_rng(int(rng_seed))
    indices = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = np.mean(arr[indices], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return mean, std, sem, float(ci_low), float(ci_high)


def _aggregate(records: Sequence[RunRecord], *, bootstrap_samples: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[RunRecord]] = {}
    for rec in records:
        if not rec.success:
            continue
        grouped.setdefault((rec.agent, rec.variant, rec.safety_mode), []).append(rec)

    rows: List[Dict[str, Any]] = []
    for (agent, variant, safety_mode), items in sorted(grouped.items()):
        rewards = np.asarray([float(x.metrics["final_reward_50"]) for x in items], dtype=np.float64)
        stresses = np.asarray([float(x.metrics["final_stress_50"]) for x in items], dtype=np.float64)
        divergences = np.asarray(
            [float(x.metrics.get("final_divergence_50", 0.0)) for x in items],
            dtype=np.float64,
        )
        interventions = np.asarray([float(x.metrics["safety_interventions"]) for x in items], dtype=np.float64)
        action_delta_l2 = np.asarray(
            [float(x.metrics.get("final_theory_action_delta_l2_50", 0.0)) for x in items],
            dtype=np.float64,
        )
        action_delta_frac = np.asarray(
            [float(x.metrics.get("final_theory_action_delta_frac_50", 0.0)) for x in items],
            dtype=np.float64,
        )
        oos_trust = np.asarray(
            [float(x.metrics.get("final_theory_verifier_oos_trust_50", 0.0)) for x in items],
            dtype=np.float64,
        )
        fallback_rates = np.asarray(
            [float(x.metrics.get("safety_fallback_rate", 0.0)) for x in items],
            dtype=np.float64,
        )
        safe_hold_rates = np.asarray(
            [float(x.metrics.get("safety_mode_safe_hold_rate", 0.0)) for x in items],
            dtype=np.float64,
        )
        reward_mean, reward_std, reward_sem, reward_ci95_low, reward_ci95_high = _mean_std_sem_ci(
            rewards,
            bootstrap_samples=int(bootstrap_samples),
            rng_seed=abs(hash((agent, variant, safety_mode, "reward"))) % (2**32),
        )
        rows.append(
            {
                "agent": agent,
                "variant": variant,
                "safety_mode": safety_mode,
                "n_runs": int(len(items)),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_sem": reward_sem,
                "reward_ci95_low": reward_ci95_low,
                "reward_ci95_high": reward_ci95_high,
                "stress_mean": float(np.mean(stresses)),
                "stress_std": float(np.std(stresses)),
                "divergence_mean": float(np.mean(divergences)),
                "divergence_std": float(np.std(divergences)),
                "safety_interventions_mean": float(np.mean(interventions)),
                "theory_action_delta_l2_mean": float(np.mean(action_delta_l2)),
                "theory_action_delta_frac_mean": float(np.mean(action_delta_frac)),
                "theory_verifier_oos_trust_mean": float(np.mean(oos_trust)),
                "safety_fallback_rate_mean": float(np.mean(fallback_rates)),
                "safety_mode_safe_hold_rate_mean": float(np.mean(safe_hold_rates)),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_paired_safety_report(records: Sequence[RunRecord]) -> Dict[str, Any]:
    by_key: Dict[Tuple[str, str, int], Dict[str, RunRecord]] = {}
    for rec in records:
        if not rec.success or rec.safety_mode not in {"on", "off"}:
            continue
        by_key.setdefault((rec.agent, rec.variant, int(rec.seed)), {})[rec.safety_mode] = rec

    pair_rows: List[Dict[str, Any]] = []
    for (agent, variant, seed), pair in sorted(by_key.items()):
        if "on" not in pair or "off" not in pair:
            continue
        on = pair["on"].metrics
        off = pair["off"].metrics
        pair_rows.append(
            {
                "agent": agent,
                "variant": variant,
                "seed": int(seed),
                "reward_on": float(on.get("final_reward_50", 0.0)),
                "reward_off": float(off.get("final_reward_50", 0.0)),
                "reward_delta_off_minus_on": float(off.get("final_reward_50", 0.0))
                - float(on.get("final_reward_50", 0.0)),
                "stress_on": float(on.get("final_stress_50", 0.0)),
                "stress_off": float(off.get("final_stress_50", 0.0)),
                "stress_delta_off_minus_on": float(off.get("final_stress_50", 0.0))
                - float(on.get("final_stress_50", 0.0)),
                "fallback_rate_on": float(on.get("safety_fallback_rate", 0.0)),
                "fallback_rate_off": float(off.get("safety_fallback_rate", 0.0)),
                "fallback_rate_delta_off_minus_on": float(off.get("safety_fallback_rate", 0.0))
                - float(on.get("safety_fallback_rate", 0.0)),
                "safe_hold_rate_on": float(on.get("safety_mode_safe_hold_rate", 0.0)),
                "safe_hold_rate_off": float(off.get("safety_mode_safe_hold_rate", 0.0)),
                "safe_hold_rate_delta_off_minus_on": float(
                    off.get("safety_mode_safe_hold_rate", 0.0)
                )
                - float(on.get("safety_mode_safe_hold_rate", 0.0)),
                "theory_action_delta_l2_on": float(on.get("final_theory_action_delta_l2_50", 0.0)),
                "theory_action_delta_l2_off": float(off.get("final_theory_action_delta_l2_50", 0.0)),
                "theory_action_delta_l2_delta_off_minus_on": float(
                    off.get("final_theory_action_delta_l2_50", 0.0)
                )
                - float(on.get("final_theory_action_delta_l2_50", 0.0)),
                "safety_interventions_on": int(on.get("safety_interventions", 0)),
                "safety_interventions_off": int(off.get("safety_interventions", 0)),
            }
        )

    summary_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in pair_rows:
        grouped.setdefault((str(row["agent"]), str(row["variant"])), []).append(row)

    for (agent, variant), items in sorted(grouped.items()):
        reward_deltas = np.asarray(
            [float(x["reward_delta_off_minus_on"]) for x in items],
            dtype=np.float64,
        )
        stress_deltas = np.asarray(
            [float(x["stress_delta_off_minus_on"]) for x in items],
            dtype=np.float64,
        )
        action_deltas = np.asarray(
            [float(x["theory_action_delta_l2_delta_off_minus_on"]) for x in items],
            dtype=np.float64,
        )
        fallback_deltas = np.asarray(
            [float(x["fallback_rate_delta_off_minus_on"]) for x in items],
            dtype=np.float64,
        )
        safe_hold_deltas = np.asarray(
            [float(x["safe_hold_rate_delta_off_minus_on"]) for x in items],
            dtype=np.float64,
        )
        summary_rows.append(
            {
                "agent": agent,
                "variant": variant,
                "n_pairs": int(len(items)),
                "reward_delta_off_minus_on_mean": float(np.mean(reward_deltas)),
                "reward_delta_off_minus_on_std": float(np.std(reward_deltas)),
                "stress_delta_off_minus_on_mean": float(np.mean(stress_deltas)),
                "stress_delta_off_minus_on_std": float(np.std(stress_deltas)),
                "fallback_rate_delta_off_minus_on_mean": float(np.mean(fallback_deltas)),
                "fallback_rate_delta_off_minus_on_std": float(np.std(fallback_deltas)),
                "safe_hold_rate_delta_off_minus_on_mean": float(np.mean(safe_hold_deltas)),
                "safe_hold_rate_delta_off_minus_on_std": float(np.std(safe_hold_deltas)),
                "theory_action_delta_l2_delta_off_minus_on_mean": float(np.mean(action_deltas)),
                "theory_action_delta_l2_delta_off_minus_on_std": float(np.std(action_deltas)),
            }
        )

    return {
        "pairs": pair_rows,
        "summary": summary_rows,
    }


def _apply_profile_overrides(args: argparse.Namespace) -> argparse.Namespace:
    profile = str(getattr(args, "profile", "standard")).strip().lower()
    if profile != "fast_jax":
        return args

    if str(args.world) == DEFAULT_WORLD:
        args.world = str(FAST_JAX_PROFILE["world"])
    if [int(v) for v in args.grid] == DEFAULT_GRID:
        args.grid = list(FAST_JAX_PROFILE["grid"])
    if int(args.steps) == DEFAULT_STEPS:
        args.steps = int(FAST_JAX_PROFILE["steps"])
    if [int(v) for v in args.seeds] == DEFAULT_SEEDS:
        args.seeds = list(FAST_JAX_PROFILE["seeds"])
    if int(args.safety_shift_warmup) == DEFAULT_SAFETY_SHIFT_WARMUP:
        args.safety_shift_warmup = int(FAST_JAX_PROFILE["safety_shift_warmup"])

    return args


def _atom_command(
    *,
    world: str,
    grid: Tuple[int, int, int],
    steps: int,
    seed: int,
    run_dir: Path,
    variant_flags: Sequence[str],
    device: str,
    safety_shift_warmup: int,
    disable_safety: bool,
    scientist_warmup_steps: Optional[int],
    scientist_diag_trust_floor: Optional[float],
    scientist_diag_trust_floor_start_step: Optional[int],
    scientist_signal_trust_scale_mode: Optional[str] = None,
    safety_min_theory_trust: Optional[float] = None,
    safety_cautious_hard_rate: Optional[float] = None,
    safety_safe_hold_hard_rate: Optional[float] = None,
    safety_cautious_action_scale: Optional[float] = None,
    safety_recovery_hysteresis: Optional[float] = None,
    safety_low_trust_hard_ratio: Optional[float] = None,
    safety_low_trust_hard_warmup: Optional[int] = None,
    world_u_inlet: Optional[float] = None,
    world_tau: Optional[float] = None,
) -> Tuple[List[str], Path]:
    cmd = [
        _python_bin(),
        str(ATOM_RUNNER),
        "--name",
        "run",
        "--world",
        world,
        "--grid",
        str(grid[0]),
        str(grid[1]),
        str(grid[2]),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--device",
        device,
        "--safety-shift-warmup",
        str(int(safety_shift_warmup)),
        "--output-dir",
        str(run_dir),
    ]
    if world_u_inlet is not None:
        cmd.extend(["--world-u-inlet", str(float(world_u_inlet))])
    if world_tau is not None:
        cmd.extend(["--world-tau", str(float(world_tau))])
    if disable_safety:
        cmd.append("--disable-safety")
    if scientist_warmup_steps is not None:
        cmd.extend(["--scientist-warmup-steps", str(int(scientist_warmup_steps))])
    if scientist_diag_trust_floor is not None:
        cmd.extend(["--scientist-diag-trust-floor", str(float(scientist_diag_trust_floor))])
    if scientist_diag_trust_floor_start_step is not None:
        cmd.extend(
            [
                "--scientist-diag-trust-floor-start-step",
                str(int(scientist_diag_trust_floor_start_step)),
            ]
        )
    if scientist_signal_trust_scale_mode is not None:
        cmd.extend(
            [
                "--scientist-signal-trust-scale-mode",
                str(scientist_signal_trust_scale_mode).strip().lower(),
            ]
        )
    if safety_min_theory_trust is not None:
        cmd.extend(["--safety-min-theory-trust", str(float(safety_min_theory_trust))])
    if safety_cautious_hard_rate is not None:
        cmd.extend(["--safety-cautious-hard-rate", str(float(safety_cautious_hard_rate))])
    if safety_safe_hold_hard_rate is not None:
        cmd.extend(["--safety-safe-hold-hard-rate", str(float(safety_safe_hold_hard_rate))])
    if safety_cautious_action_scale is not None:
        cmd.extend(["--safety-cautious-action-scale", str(float(safety_cautious_action_scale))])
    if safety_recovery_hysteresis is not None:
        cmd.extend(["--safety-recovery-hysteresis", str(float(safety_recovery_hysteresis))])
    if safety_low_trust_hard_ratio is not None:
        cmd.extend(["--safety-low-trust-hard-ratio", str(float(safety_low_trust_hard_ratio))])
    if safety_low_trust_hard_warmup is not None:
        cmd.extend(["--safety-low-trust-hard-warmup", str(int(safety_low_trust_hard_warmup))])
    cmd.extend(list(variant_flags))
    result_path = run_dir / "run" / "results.json"
    return cmd, result_path


def _baseline_command(
    *,
    script: Path,
    world: str,
    grid: Tuple[int, int, int],
    steps: int,
    seed: int,
    run_dir: Path,
    world_u_inlet: Optional[float],
    world_tau: Optional[float],
) -> Tuple[List[str], Path]:
    cmd = [
        _python_bin(),
        str(script),
        "--world",
        world,
        "--grid",
        str(grid[0]),
        str(grid[1]),
        str(grid[2]),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--output_dir",
        str(run_dir),
    ]
    if world_u_inlet is not None:
        cmd.extend(["--world-u-inlet", str(float(world_u_inlet))])
    if world_tau is not None:
        cmd.extend(["--world-tau", str(float(world_tau))])
    result_path = run_dir / "results.json"
    return cmd, result_path


def run_harness(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    atom_variants = _atom_variant_matrix(str(args.atom_variant_set))
    if bool(args.paired_safety):
        safety_runs: List[Tuple[str, bool]] = [("off", True), ("on", False)]
    else:
        safety_runs = [("off", True)] if bool(args.disable_safety) else [("on", False)]
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.replace("\n", " "),
        "python_bin": _python_bin(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "world": args.world,
        "grid": list(args.grid),
        "steps": int(args.steps),
        "seeds": [int(s) for s in args.seeds],
        "device": args.device,
        "include_sb3": bool(args.include_sb3),
        "skip_baselines": bool(args.skip_baselines),
        "atom_variant_set": str(args.atom_variant_set),
        "atom_variants": {
            name: {"flags": list(spec["flags"]), "env": dict(spec["env"])}
            for name, spec in atom_variants.items()
        },
        "force_sparse_scientist": bool(args.force_sparse_scientist),
        "profile": str(getattr(args, "profile", "standard")),
        "safety_shift_warmup": int(args.safety_shift_warmup),
        "disable_safety": bool(args.disable_safety),
        "paired_safety": bool(args.paired_safety),
        "safety_modes": [mode for mode, _ in safety_runs],
        "bootstrap_samples": int(args.bootstrap_samples),
        "safety_min_theory_trust": (
            float(args.safety_min_theory_trust)
            if args.safety_min_theory_trust is not None
            else None
        ),
        "safety_cautious_hard_rate": (
            float(args.safety_cautious_hard_rate)
            if args.safety_cautious_hard_rate is not None
            else None
        ),
        "safety_safe_hold_hard_rate": (
            float(args.safety_safe_hold_hard_rate)
            if args.safety_safe_hold_hard_rate is not None
            else None
        ),
        "safety_cautious_action_scale": (
            float(args.safety_cautious_action_scale)
            if args.safety_cautious_action_scale is not None
            else None
        ),
        "safety_recovery_hysteresis": (
            float(args.safety_recovery_hysteresis)
            if args.safety_recovery_hysteresis is not None
            else None
        ),
        "safety_low_trust_hard_ratio": (
            float(args.safety_low_trust_hard_ratio)
            if args.safety_low_trust_hard_ratio is not None
            else None
        ),
        "safety_low_trust_hard_warmup": (
            int(args.safety_low_trust_hard_warmup)
            if args.safety_low_trust_hard_warmup is not None
            else None
        ),
        "scientist_warmup_steps": (
            int(args.scientist_warmup_steps)
            if args.scientist_warmup_steps is not None
            else None
        ),
        "scientist_diag_trust_floor": (
            float(args.scientist_diag_trust_floor)
            if args.scientist_diag_trust_floor is not None
            else None
        ),
        "scientist_diag_trust_floor_start_step": (
            int(args.scientist_diag_trust_floor_start_step)
            if args.scientist_diag_trust_floor_start_step is not None
            else None
        ),
        "scientist_signal_trust_scale_mode": (
            str(args.scientist_signal_trust_scale_mode).strip().lower()
            if args.scientist_signal_trust_scale_mode is not None
            else None
        ),
        "world_u_inlet": (
            float(args.world_u_inlet) if args.world_u_inlet is not None else None
        ),
        "world_tau": float(args.world_tau) if args.world_tau is not None else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    records: List[RunRecord] = []

    for seed in args.seeds:
        for safety_mode, disable_safety in safety_runs:
            for variant, spec in atom_variants.items():
                if bool(args.paired_safety):
                    run_dir = out_dir / f"safety_{safety_mode}" / "atom" / variant / f"seed_{int(seed)}"
                else:
                    run_dir = out_dir / "atom" / variant / f"seed_{int(seed)}"
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd, result_path = _atom_command(
                    world=args.world,
                    grid=tuple(args.grid),
                    steps=int(args.steps),
                    seed=int(seed),
                    run_dir=run_dir,
                    variant_flags=spec["flags"],
                    device=args.device,
                    safety_shift_warmup=int(args.safety_shift_warmup),
                    disable_safety=disable_safety,
                    safety_min_theory_trust=args.safety_min_theory_trust,
                    safety_cautious_hard_rate=args.safety_cautious_hard_rate,
                    safety_safe_hold_hard_rate=args.safety_safe_hold_hard_rate,
                    safety_cautious_action_scale=args.safety_cautious_action_scale,
                    safety_recovery_hysteresis=args.safety_recovery_hysteresis,
                    safety_low_trust_hard_ratio=args.safety_low_trust_hard_ratio,
                    safety_low_trust_hard_warmup=args.safety_low_trust_hard_warmup,
                    scientist_warmup_steps=args.scientist_warmup_steps,
                    scientist_diag_trust_floor=args.scientist_diag_trust_floor,
                    scientist_diag_trust_floor_start_step=args.scientist_diag_trust_floor_start_step,
                    scientist_signal_trust_scale_mode=args.scientist_signal_trust_scale_mode,
                    world_u_inlet=args.world_u_inlet,
                    world_tau=args.world_tau,
                )
                try:
                    env = dict(os.environ)
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    if str(getattr(args, "profile", "standard")).strip().lower() == "fast_jax":
                        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
                    if args.force_sparse_scientist:
                        env["ATOM_SCIENTIST_FORCE_SPARSE"] = "1"
                    for k, v in dict(spec["env"]).items():
                        env[str(k)] = str(v)
                    _run_cmd_env(cmd, cwd=ROOT, env=env)
                    payload = _load_json(result_path)
                    metrics = _normalize_metrics(payload)
                    records.append(
                        RunRecord(
                            agent="atom",
                            variant=variant,
                            safety_mode=safety_mode,
                            seed=int(seed),
                            steps=int(args.steps),
                            run_dir=str(run_dir),
                            success=True,
                            result_path=str(result_path),
                            metrics=metrics,
                        )
                    )
                except Exception as exc:
                    records.append(
                        RunRecord(
                            agent="atom",
                            variant=variant,
                            safety_mode=safety_mode,
                            seed=int(seed),
                            steps=int(args.steps),
                            run_dir=str(run_dir),
                            success=False,
                            result_path=str(result_path),
                            metrics={},
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    if not args.continue_on_error:
                        raise

        if not args.skip_baselines:
            baseline_specs = [("cleanrl_ppo", BASELINE_CLEANRL)]
            if args.include_sb3:
                baseline_specs.append(("sb3_ppo", BASELINE_SB3))

            for variant, script in baseline_specs:
                run_dir = out_dir / "baseline" / variant / f"seed_{int(seed)}"
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd, result_path = _baseline_command(
                    script=script,
                    world=args.world,
                    grid=tuple(args.grid),
                    steps=int(args.steps),
                    seed=int(seed),
                    run_dir=run_dir,
                    world_u_inlet=args.world_u_inlet,
                    world_tau=args.world_tau,
                )
                try:
                    env = dict(os.environ)
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    _run_cmd_env(cmd, cwd=ROOT, env=env)
                    payload = _load_json(result_path)
                    metrics = _normalize_metrics(payload)
                    records.append(
                        RunRecord(
                            agent="baseline",
                            variant=variant,
                            safety_mode="na",
                            seed=int(seed),
                            steps=int(args.steps),
                            run_dir=str(run_dir),
                            success=True,
                            result_path=str(result_path),
                            metrics=metrics,
                        )
                    )
                except Exception as exc:
                    records.append(
                        RunRecord(
                            agent="baseline",
                            variant=variant,
                            safety_mode="na",
                            seed=int(seed),
                            steps=int(args.steps),
                            run_dir=str(run_dir),
                            success=False,
                            result_path=str(result_path),
                            metrics={},
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    if not args.continue_on_error:
                        raise

    runs_payload = {"runs": [r.to_dict() for r in records]}
    (out_dir / "runs.json").write_text(json.dumps(runs_payload, indent=2), encoding="utf-8")

    summary_rows = _aggregate(records, bootstrap_samples=int(args.bootstrap_samples))
    summary_payload = {"manifest": manifest, "summary": summary_rows}
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_csv(out_dir / "summary.csv", summary_rows)
    if bool(args.paired_safety):
        paired_payload = _build_paired_safety_report(records)
        (out_dir / "paired_safety_summary.json").write_text(
            json.dumps(paired_payload, indent=2),
            encoding="utf-8",
        )
        _write_csv(out_dir / "paired_safety_summary.csv", paired_payload.get("summary", []))
        summary_payload["paired_safety"] = paired_payload
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ATOM ablations with PPO baseline references")
    parser.add_argument(
        "--profile",
        type=str,
        default="standard",
        choices=["standard", "fast_jax"],
        help="Ablation runtime preset. fast_jax defaults to lbm2d:cylinder with reduced budget.",
    )
    parser.add_argument("--world", type=str, default=DEFAULT_WORLD)
    parser.add_argument(
        "--world-u-inlet",
        type=float,
        default=None,
        help="Optional inlet velocity forwarded to lbm/lbm2d worlds.",
    )
    parser.add_argument(
        "--world-tau",
        type=float,
        default=None,
        help="Optional relaxation time forwarded to lbm/lbm2d worlds.",
    )
    parser.add_argument("--grid", type=int, nargs=3, default=list(DEFAULT_GRID))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--out", type=str, default=str(ROOT / "benchmark_results" / "platform_ablations"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--atom-variant-set",
        type=str,
        default="production",
        choices=["production", "legacy"],
        help="Ablation matrix preset for ATOM variants",
    )
    parser.add_argument("--include-sb3", action="store_true", help="Include SB3 baseline runs")
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip PPO baseline runs (for quick local smoke only)",
    )
    parser.add_argument(
        "--force-sparse-scientist",
        action="store_true",
        help="Force sparse law discovery backend for deterministic, faster ablation runs",
    )
    parser.add_argument(
        "--safety-shift-warmup",
        type=int,
        default=DEFAULT_SAFETY_SHIFT_WARMUP,
        help="Forwarded to atom_experiment_runner for earlier/later safety shift checks.",
    )
    parser.add_argument(
        "--disable-safety",
        action="store_true",
        help="Forward --disable-safety to ATOM runs (keeps baselines unchanged unless --paired-safety is used).",
    )
    parser.add_argument(
        "--paired-safety",
        action="store_true",
        help="Run ATOM variants in both safety OFF and safety ON modes and emit paired comparison artifacts.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap samples used to estimate reward confidence intervals in summary aggregation.",
    )
    parser.add_argument(
        "--safety-min-theory-trust",
        type=float,
        default=None,
        help="Optional safety trust threshold forwarded to ATOM runs.",
    )
    parser.add_argument(
        "--safety-cautious-hard-rate",
        type=float,
        default=None,
        help="Optional hard-event rate threshold for cautious mode.",
    )
    parser.add_argument(
        "--safety-safe-hold-hard-rate",
        type=float,
        default=None,
        help="Optional hard-event rate threshold for safe-hold mode.",
    )
    parser.add_argument(
        "--safety-cautious-action-scale",
        type=float,
        default=None,
        help="Optional action scale applied in cautious mode.",
    )
    parser.add_argument(
        "--safety-recovery-hysteresis",
        type=float,
        default=None,
        help="Optional hysteresis ratio for safety-mode recovery.",
    )
    parser.add_argument(
        "--safety-low-trust-hard-ratio",
        type=float,
        default=None,
        help="Optional fraction of trust threshold treated as hard low-trust.",
    )
    parser.add_argument(
        "--safety-low-trust-hard-warmup",
        type=int,
        default=None,
        help="Optional warmup steps before low-trust can trigger hard intervention.",
    )
    parser.add_argument(
        "--scientist-warmup-steps",
        type=int,
        default=None,
        help="Optional scientist warmup override forwarded to ATOM runs.",
    )
    parser.add_argument(
        "--scientist-diag-trust-floor",
        type=float,
        default=None,
        help="Optional diagnostic trust floor forwarded to ATOM runs.",
    )
    parser.add_argument(
        "--scientist-diag-trust-floor-start-step",
        type=int,
        default=None,
        help="Optional diagnostic trust floor start step forwarded to ATOM runs.",
    )
    parser.add_argument(
        "--scientist-signal-trust-scale-mode",
        type=str,
        default=None,
        choices=["off", "sqrt", "full"],
        help="Optional scientist signal content scaling mode forwarded to ATOM runs.",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Continue when one run fails")
    args = parser.parse_args()
    args = _apply_profile_overrides(args)

    summary = run_harness(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
