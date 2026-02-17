#!/usr/bin/env python3
"""
ATOM PLATFORM: CANONICAL EXPERIMENT RUNNER
==========================================
The correct, complete implementation of the ATOM neuro-symbolic pipeline.

This file replaces cylinder.py, super_sonic_1.py, and diag_shapes.py with
an implementation that uses EVERY ATOM module through its REAL API:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  World.step(action)                                                │
  │    ↓ obs (B,4,X,Y,Z)                                              │
  │  Brain.eyes.embed(obs) → visual_latent (B, vision_dim)            │
  │    ↓ [action, mean_speed, turbulence, emb[:8]] → f_vec (11D)      │
  │  Scientist.predict_theory(f_vec) → (theory_val, theory_trust)     │
  │    ↓                                                               │
  │  Brain(obs, theory, prev_action, hx, theory_confidence=trust)     │
  │    → (mu, std), value, hx_new, stress                             │
  │    ↓                                                               │
  │  TanhNormal(mu, std).sample() → (action_phys, raw_action)         │
  │  TanhNormal.log_prob(action, pre_tanh_value=raw) → log_prob       │
  │    ↓                                                               │
  │  Memory.push(obs, action, reward, done, hx, log_prob, value,      │
  │              theory, trust)                                        │
  │  Scientist.observe(f_vec, reward)                                  │
  │    ↓ (when enough data)                                            │
  │  _update_brain() — PPO with GAE-λ, clipped surrogate, cached      │
  │                     theory/trust, visual reconstruction loss       │
  │    ↓ (periodic)                                                    │
  │  Scientist.ponder() — symbolic law discovery                       │
  └─────────────────────────────────────────────────────────────────────┘

Usage:
    python atom_experiment_runner.py --world analytical:taylor_green --steps 500
    python atom_experiment_runner.py --world lbm:cylinder --steps 1000

Author: ATOM Platform
Date: February 2026
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

# ---------------------------------------------------------------------------
# ATOM IMPORTS — Using REAL package paths (identical to training_loop.py)
# ---------------------------------------------------------------------------
from atom.config import AtomConfig
from atom.core.brain import AtomBrain, create_brain_from_config
from atom.core.distributions import TanhNormal
from atom.logging import setup_logging, get_logger
from atom.mind.memory import AtomMemory, create_memory_from_config
from atom.mind.scientist import AtomScientist
from atom.mind.scientist_v2 import StructuralScientist
from atom.platform import (
    ControlEnvelope,
    InverseDesignEngine,
    ObjectiveSpec,
    RuntimeSafetySupervisor,
)

# Local world adapters
from atom.sim.atom_worlds import WorldAdapter, create_world, AnalyticalWorld, list_available_worlds


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Identity
    name: str = "atom_experiment"
    seed: int = 42

    # World
    world_spec: str = "analytical:taylor_green"
    grid_shape: Tuple[int, int, int] = (32, 32, 16)
    world_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Brain
    vision_dim: int = 128
    internal_neurons: int = 32
    action_dim: int = 1
    use_symplectic: bool = True
    vision_mode: str = "fno"

    # Training
    max_steps: int = 500
    batch_size: int = 4
    learning_rate_actor: float = 1e-4
    learning_rate_eyes: float = 1e-4
    learning_rate_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 3
    entropy_coef: float = 0.01
    critic_coef: float = 0.5
    stress_coef: float = 0.1
    visual_coef: float = 1.0
    max_grad_norm: float = 0.5

    # Memory
    memory_capacity: int = 2000
    sequence_length: int = 4

    # Scientist
    sleep_interval: int = 100
    scientist_signal: Literal["v1", "v2"] = "v2"
    scientist_warmup_steps: int = 50
    scientist_diag_trust_floor: float = 0.0
    scientist_diag_trust_floor_start_step: Optional[int] = None
    scientist_signal_trust_scale_mode: Literal["off", "sqrt", "full"] = "off"

    # Eyes
    fno_modes: int = 6
    fno_width: int = 16

    # Ablation toggles (for benchmark suite)
    ablate_scientist: bool = False    # Disable symbolic regression
    ablate_symplectic: bool = False   # Disable Hamiltonian constraint
    ablate_trust_gate: bool = False   # Force trust=0 (pure neural)

    # Output
    output_dir: str = "experiment_results"
    log_interval: int = 10
    save_interval: int = 200

    # Hardware
    device: str = "auto"

    # Runtime safety envelope (Phase 3)
    safety_enable: bool = True
    safety_min_theory_trust: float = 0.08
    safety_max_stress: float = 8.0
    safety_shift_z_threshold: float = 8.0
    safety_shift_warmup: int = 24
    safety_density_min: float = 0.05
    safety_density_max: float = 8.0
    safety_speed_max: float = 3.0
    safety_cautious_hard_rate: float = 0.30
    safety_safe_hold_hard_rate: float = 0.60
    safety_cautious_action_scale: float = 0.50
    safety_recovery_hysteresis: float = 0.80
    safety_low_trust_hard_ratio: float = 0.20
    safety_low_trust_hard_warmup: int = 24

    def get_device(self) -> str:
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    def to_atom_config(self) -> AtomConfig:
        """Convert to AtomConfig for ATOM module factories."""
        return AtomConfig(
            experiment_name=self.name,
            seed=self.seed,
            hardware={"device": self.get_device(), "enable_x64_precision": True},
            physics={"grid_shape": list(self.grid_shape)},
            brain={
                "vision_dim": self.vision_dim,
                "internal_neurons": self.internal_neurons,
                "action_dim": self.action_dim,
                "batch_size": self.batch_size,
                "learning_rate_actor": self.learning_rate_actor,
                "learning_rate_eyes": self.learning_rate_eyes,
                "learning_rate_critic": self.learning_rate_critic,
                "use_symplectic": self.use_symplectic and not self.ablate_symplectic,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "ppo_epochs": self.ppo_epochs,
            },
            eyes={"fno_modes": self.fno_modes, "fno_width": self.fno_width},
            memory={"capacity": self.memory_capacity, "sequence_length": self.sequence_length},
            training={"max_steps": self.max_steps, "sleep_interval": self.sleep_interval},
        )


# =============================================================================
# GRADIENT CLIPPING (MPS-safe, from training_loop.py)
# =============================================================================

def _safe_clip_grads(parameters, max_norm: float) -> float:
    """MPS-safe gradient clipping (handles complex spectral grads)."""
    params = [p for p in parameters if p.grad is not None]
    total_norm_sq = 0.0

    for p in params:
        g = p.grad
        if g.is_complex():
            g_real = torch.view_as_real(g)
            param_norm = g_real.reshape(-1).norm(2)
        else:
            param_norm = g.norm(2)
        total_norm_sq += float(param_norm.item()) ** 2

    total_norm = total_norm_sq ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    return total_norm


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def _compute_divergence_rms(obs: np.ndarray) -> float:
    """Compute RMS divergence from an observation tensor.

    Supports channel-first (C, X, Y, Z) and channel-last (X, Y, Z, C) layouts,
    using the first velocity components aligned to active spatial axes.
    """
    try:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim < 2:
            return 0.0

        # Prefer channel-first for ATOM worlds (C, X, Y, Z)
        if 2 <= arr.shape[0] <= 8 and np.prod(arr.shape[1:]) > 0:
            spatial_shape = arr.shape[1:]
            active_axes = [i for i, n in enumerate(spatial_shape) if int(n) > 1]
            max_comp = min(len(active_axes), int(arr.shape[0]), 3)
            if max_comp <= 0:
                return 0.0
            div = np.zeros(spatial_shape, dtype=np.float64)
            for comp_idx in range(max_comp):
                axis = active_axes[comp_idx]
                comp = np.asarray(arr[comp_idx], dtype=np.float64)
                div += np.gradient(comp, axis=axis, edge_order=1)
            return float(np.sqrt(np.mean(div * div)))

        # Fallback: channel-last (X, Y, Z, C)
        if 2 <= arr.shape[-1] <= 8 and np.prod(arr.shape[:-1]) > 0:
            spatial_shape = arr.shape[:-1]
            active_axes = [i for i, n in enumerate(spatial_shape) if int(n) > 1]
            max_comp = min(len(active_axes), int(arr.shape[-1]), 3)
            if max_comp <= 0:
                return 0.0
            div = np.zeros(spatial_shape, dtype=np.float64)
            for comp_idx in range(max_comp):
                axis = active_axes[comp_idx]
                comp = np.asarray(arr[..., comp_idx], dtype=np.float64)
                div += np.gradient(comp, axis=axis, edge_order=1)
            return float(np.sqrt(np.mean(div * div)))
    except Exception:
        return 0.0

    return 0.0


def _extract_divergence_metric(info: Optional[Mapping[str, Any]], obs: np.ndarray) -> Tuple[float, str]:
    info_dict = dict(info or {}) if isinstance(info, Mapping) else {}
    for key in ("divergence_rms", "divergence", "div_rms", "mean_divergence"):
        if key in info_dict:
            return _safe_float(info_dict.get(key), default=0.0), f"info:{key}"
    return _compute_divergence_rms(obs), "proxy:obs_divergence_rms"


# =============================================================================
# HISTORY TRACKER
# =============================================================================

@dataclass
class ExperimentHistory:
    """Tracks all metrics during an experiment run."""
    reward: List[float] = field(default_factory=list)
    stress: List[float] = field(default_factory=list)
    divergence: List[float] = field(default_factory=list)
    theory_score: List[float] = field(default_factory=list)
    theory_trust: List[float] = field(default_factory=list)
    theory_trust_raw: List[float] = field(default_factory=list)
    theory_trust_verified: List[float] = field(default_factory=list)
    theory_trust_structural_floor: List[float] = field(default_factory=list)
    theory_trust_diag_floor_applied: List[float] = field(default_factory=list)
    theory_verifier_oos_trust: List[float] = field(default_factory=list)
    theory_action_delta_l2: List[float] = field(default_factory=list)
    theory_action_delta_frac: List[float] = field(default_factory=list)
    ppo_loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    safety_intervention: List[float] = field(default_factory=list)
    safety_fallback: List[float] = field(default_factory=list)
    best_law: Optional[str] = None
    laws_discovered: List[str] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward": self.reward,
            "stress": self.stress,
            "divergence": self.divergence,
            "theory_score": self.theory_score,
            "theory_trust": self.theory_trust,
            "theory_trust_raw": self.theory_trust_raw,
            "theory_trust_verified": self.theory_trust_verified,
            "theory_trust_structural_floor": self.theory_trust_structural_floor,
            "theory_trust_diag_floor_applied": self.theory_trust_diag_floor_applied,
            "theory_verifier_oos_trust": self.theory_verifier_oos_trust,
            "theory_action_delta_l2": self.theory_action_delta_l2,
            "theory_action_delta_frac": self.theory_action_delta_frac,
            "ppo_loss": self.ppo_loss,
            "safety_intervention": self.safety_intervention,
            "safety_fallback": self.safety_fallback,
            "best_law": self.best_law,
            "laws_discovered": self.laws_discovered,
            "avg_step_time": float(np.mean(self.step_times)) if self.step_times else 0.0,
        }


class _V1ScientistAdapter:
    """Compatibility wrapper exposing StructuralScientist-like API for V1 mode."""

    def __init__(self, scientist: AtomScientist):
        self.scientist = scientist

    @property
    def theory_dim(self) -> int:
        return 1

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if torch.is_tensor(value):
                return float(value.detach().cpu().reshape(-1)[0].item())
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return float(default)
            v = float(arr[0])
            if not np.isfinite(v):
                return float(default)
            return v
        except Exception:
            return float(default)

    def get_signal(
        self,
        features: Any,
        reward: Optional[float] = None,
        action: Optional[float] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        _ = (reward, action)
        pred, trust = self.scientist.predict_theory(features)
        pred_f = self._to_float(pred, default=0.0)
        trust_f = float(np.clip(self._to_float(trust, default=0.0), 0.0, 1.0))

        signal_np = np.array([np.clip(pred_f, -1.0, 1.0)], dtype=np.float32)
        signal_tensor = torch.tensor(signal_np, dtype=torch.float32, device=device).unsqueeze(0)
        return {
            "tensor": signal_tensor,
            "prediction": float(pred_f),
            "jacobian": np.zeros(8, dtype=np.float32),
            "jacobian_raw_norm": 0.0,
            "trust": float(trust_f),
            "raw_trust": float(trust_f),
            "verified_trust": float(trust_f),
            "trust_structural_floor": 0.0,
            "packet": None,
            "packet_version": "theory_packet.v1_compat",
            "feature_schema_hash": "",
            "hypothesis_record": None,
            "governance": {
                "approved": True,
                "reasons": [],
                "mode": "v1_signal",
            },
            "experiment_plan": None,
            "has_law": self.scientist.best_law is not None,
            "in_warmup": False,
            "baseline_ready": False,
            "baseline_slope": 0.0,
            "baseline_r2": 0.0,
            "discovery_target_mode": "raw",
            "discovery_hybrid_alpha": 0.0,
            "discovery_target_value": (
                float(reward) if reward is not None and np.isfinite(float(reward)) else 0.0
            ),
            "verifier_metrics": {
                "rmse": float("nan"),
                "correlation": 0.0,
                "interventional": 0.0,
            },
        }

    def observe(self, features: Any, target_metric: float) -> None:
        self.scientist.observe(features, target_metric)

    def observe_and_verify(
        self,
        features: Any,
        reward: float,
        prediction: float,
        action: float,
    ) -> None:
        _ = (prediction, action)
        self.scientist.observe(features, reward)

    def ponder(self) -> Optional[str]:
        return self.scientist.ponder()

    def fit_offline(self, X: Any, y: Any) -> None:
        self.scientist.fit_offline(X, y)

    def shutdown(self) -> None:
        self.scientist.shutdown()

    @property
    def best_law(self) -> Optional[str]:
        return self.scientist.best_law

    @property
    def theory_archive(self) -> List[Tuple[str, float]]:
        return self.scientist.theory_archive

    @property
    def new_discovery_alert(self) -> Optional[str]:
        return self.scientist.new_discovery_alert

    @new_discovery_alert.setter
    def new_discovery_alert(self, value: Optional[str]) -> None:
        self.scientist.new_discovery_alert = value


# =============================================================================
# THE EXPERIMENT RUNNER
# =============================================================================

class ATOMExperiment:
    """
    Canonical ATOM experiment runner.

    Uses the EXACT same pipeline as training_loop.py:AtomOrchestrator.run()
    but with pluggable worlds, ablation toggles, and callback hooks.
    """

    def __init__(self, config: ExperimentConfig,
                 on_step: Optional[Callable] = None,
                 on_law_discovered: Optional[Callable] = None):
        """
        Args:
            config: Experiment configuration
            on_step: Callback(step, obs, action, reward, info, history) per step
            on_law_discovered: Callback(law_str, step) when scientist finds a law
        """
        self.config = config
        self.on_step = on_step
        self.on_law_discovered = on_law_discovered

        self.device = config.get_device()
        self.atom_config = config.to_atom_config()
        self.scientist_signal_mode = str(config.scientist_signal).strip().lower()
        if self.scientist_signal_mode not in {"v1", "v2"}:
            self.scientist_signal_mode = "v2"

        setup_logging(self.atom_config)
        self.log = get_logger("experiment")

        # Output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. WORLD ---
        self.log.info(f"Creating world: {config.world_spec}")
        base_world_kwargs = (
            dict(config.world_kwargs)
            if isinstance(config.world_kwargs, dict)
            else {}
        )
        self.base_world_kwargs = dict(base_world_kwargs)
        self.world = create_world(
            config.world_spec,
            nx=config.grid_shape[0],
            ny=config.grid_shape[1],
            nz=config.grid_shape[2],
            **base_world_kwargs,
        )
        # Sync action_dim from world
        actual_action_dim = self.world.action_dim
        if actual_action_dim != config.action_dim:
            self.log.info(f"World action_dim={actual_action_dim}, updating config")
            config.action_dim = actual_action_dim
            self.atom_config = config.to_atom_config()

        self.control_envelope = self._build_control_envelope()
        self.safety_supervisor = (
            RuntimeSafetySupervisor(
                envelope=self.control_envelope,
                action_dim=config.action_dim,
                min_theory_trust=config.safety_min_theory_trust,
                max_stress=config.safety_max_stress,
                shift_z_threshold=config.safety_shift_z_threshold,
                shift_warmup=config.safety_shift_warmup,
                cautious_hard_rate=config.safety_cautious_hard_rate,
                safe_hold_hard_rate=config.safety_safe_hold_hard_rate,
                cautious_action_scale=config.safety_cautious_action_scale,
                recovery_hysteresis=config.safety_recovery_hysteresis,
                low_trust_hard_ratio=config.safety_low_trust_hard_ratio,
                low_trust_hard_warmup=config.safety_low_trust_hard_warmup,
            )
            if config.safety_enable
            else None
        )

        # --- 2. BRAIN (Eyes + LTC/GRU + Skeleton + Actor/Critic) ---
        use_symp = config.use_symplectic and not config.ablate_symplectic
        # Scientist signal mode controls theory signal dimension.
        self.atom_config.brain.theory_dim = 1 if self.scientist_signal_mode == "v1" else 10
        self.brain = create_brain_from_config(
            config=self.atom_config,
            use_symplectic=use_symp,
            vision_mode=config.vision_mode,
        ).to(self.device)

        # --- 3. MEMORY (Ring buffer with temporal integrity) ---
        self.memory = create_memory_from_config(self.atom_config)

        # --- 4. SCIENTIST (Symbolic regression + conformal trust) ---
        # Feature vector: [action(D), mean_speed(1), turbulence(1), emb[:8]]
        # Total dims = action_dim + 2 + 8 = action_dim + 10
        var_names = (
            [f"action_{i}" for i in range(config.action_dim)]
            + ["mean_speed", "turbulence"]
            + [f"latent_{i}" for i in range(8)]
        )
        base_scientist = AtomScientist(variable_names=var_names)
        if self.scientist_signal_mode == "v1":
            self.scientist = _V1ScientistAdapter(base_scientist)
        else:
            self.scientist = StructuralScientist(
                scientist=base_scientist,
                variable_names=var_names,
                warmup_steps=int(config.scientist_warmup_steps),
                diagnostic_trust_floor=float(config.scientist_diag_trust_floor),
                diagnostic_trust_floor_start_step=config.scientist_diag_trust_floor_start_step,
                signal_trust_scale_mode=str(config.scientist_signal_trust_scale_mode),
            )

        # --- 5. OPTIMIZER (Same structure as training_loop.py) ---
        self.optimizer = optim.Adam([
            {"params": self.brain.eyes.parameters(),
             "lr": config.learning_rate_eyes},
            {"params": self.brain.liquid.parameters(),
             "lr": config.learning_rate_actor},
            {"params": list(self.brain.actor_mu.parameters())
                       + [self.brain.actor_log_std],
             "lr": config.learning_rate_actor},
            {"params": self.brain.critic.parameters(),
             "lr": config.learning_rate_critic},
        ])

        # --- 6. HISTORY ---
        self.history = ExperimentHistory()
        self._divergence_source_counts: Dict[str, int] = {}

        self.log.info(f"ATOM Experiment initialized: {config.name}")
        self.log.info(f"  World: {config.world_spec} | Grid: {config.grid_shape}")
        self.log.info(f"  Brain: vision={config.vision_dim}, neurons={config.internal_neurons}")
        self.log.info(f"  Ablations: scientist={'OFF' if config.ablate_scientist else 'ON'}, "
                      f"symplectic={'OFF' if config.ablate_symplectic else 'ON'}, "
                      f"trust_gate={'OFF' if config.ablate_trust_gate else 'ON'}")
        self.log.info(f"  Scientist signal mode: {self.scientist_signal_mode}")
        if self.scientist_signal_mode == "v2":
            diag_start = (
                str(config.scientist_diag_trust_floor_start_step)
                if config.scientist_diag_trust_floor_start_step is not None
                else "auto"
            )
            self.log.info(
                "  Scientist diagnostics: "
                f"warmup_steps={int(config.scientist_warmup_steps)} "
                f"diag_trust_floor={float(config.scientist_diag_trust_floor):.3f} "
                f"diag_trust_floor_start={diag_start} "
                f"trust_scale_mode={str(config.scientist_signal_trust_scale_mode)}"
            )
        self.log.info(f"  Device: {self.device}")
        self.log.info(f"  Safety supervisor: {'ON' if self.safety_supervisor is not None else 'OFF'}")

    # -----------------------------------------------------------------
    # FEATURE VECTOR CONSTRUCTION (Exact match to training_loop.py:run)
    # -----------------------------------------------------------------

    def _build_feature_vector(self, obs_t: torch.Tensor,
                              last_action: torch.Tensor) -> torch.Tensor:
        """
        Construct the 11D feature vector for the Scientist.
        Matches training_loop.py lines 494-506 EXACTLY.

        Returns: (D,) tensor on self.device
        """
        with torch.no_grad():
            vel = obs_t[:, :3]  # (B, 3, X, Y, Z)
            speed = torch.norm(vel, dim=1)  # (B, X, Y, Z)
            mean_speed = speed.mean().item()
            turb_int = speed.std().item()
            emb = self.brain.eyes.embed(obs_t)[0, :8]  # (8,)

            f_vec = torch.cat([
                last_action.flatten(),
                torch.tensor([mean_speed, turb_int], device=obs_t.device),
                emb,
            ])

        return f_vec

    def _build_control_envelope(self) -> ControlEnvelope:
        action_bounds = {
            f"action_{i}": (-1.0, 1.0)
            for i in range(int(self.config.action_dim))
        }
        state_bounds = {
            "rho_min": (float(self.config.safety_density_min), 100.0),
            "rho_max": (0.0, float(self.config.safety_density_max)),
            "mean_speed": (0.0, float(self.config.safety_speed_max)),
        }
        return ControlEnvelope(
            action_bounds=action_bounds,
            state_bounds=state_bounds,
            intervention_policy="runtime_safety_supervisor.v1",
            fallback_policy_id="fallback:zero_action",
        )

    def _build_safety_diagnostics(
        self,
        obs_t: torch.Tensor,
        f_vec: torch.Tensor,
        theory_conf_t: torch.Tensor,
        stress: torch.Tensor,
        *,
        scientist_warmup_active: bool = False,
        info: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with torch.no_grad():
            vel = obs_t[:, :3]
            speed = torch.norm(vel, dim=1)
            rho = obs_t[:, 3]
            diagnostics = {
                "mean_speed": float(speed.mean().item()),
                "turbulence": float(speed.std().item()),
                "rho_min": float(rho.min().item()),
                "rho_max": float(rho.max().item()),
                "theory_trust": float(theory_conf_t.mean().item()),
                "stress": float(stress.mean().item()),
                "divergence_rms": _compute_divergence_rms(
                    np.asarray(obs_t[0].detach().cpu().numpy(), dtype=np.float32)
                ),
                "scientist_warmup_active": bool(scientist_warmup_active),
                "feature_vector": f_vec.detach().cpu().numpy().tolist(),
            }
        if info:
            for key in ("min_density", "max_density", "rho_mean", "rho_std"):
                if key in info:
                    diagnostics[key] = float(info[key])
        return diagnostics

    def _coerce_theory_batch(self, theory_batch: torch.Tensor) -> torch.Tensor:
        """Match cached theory tensor width to the Brain's configured theory_dim."""
        expected = int(self.brain.theory_dim)
        if theory_batch.ndim < 2:
            return theory_batch
        current = int(theory_batch.shape[-1])
        if current == expected:
            return theory_batch
        if current > expected:
            return theory_batch[..., :expected]
        pad_shape = list(theory_batch.shape[:-1]) + [expected]
        padded = torch.zeros(*pad_shape, dtype=theory_batch.dtype, device=theory_batch.device)
        padded[..., :current] = theory_batch
        return padded

    # -----------------------------------------------------------------
    # PPO UPDATE (Exact match to training_loop.py:_update_brain)
    # -----------------------------------------------------------------

    def _update_brain(self) -> Dict[str, Any]:
        """
        PPO update with GAE-λ, clipped surrogate, cached theory/trust.
        This is an exact reimplementation of training_loop.py:_update_brain().
        """
        batch = self.memory.sample(self.config.batch_size)
        if batch is None:
            return {"loss": 0.0}

        device = self.device

        # 1. UNPACK SEQUENCE
        obs = torch.as_tensor(batch["obs"], device=device, dtype=torch.float32)
        actions = torch.as_tensor(batch["action"], device=device, dtype=torch.float32)
        rewards = torch.as_tensor(batch["reward"], device=device, dtype=torch.float32)
        dones = torch.as_tensor(batch["done"], device=device, dtype=torch.float32)
        hx_batch = torch.as_tensor(batch["hx"], device=device, dtype=torch.float32)
        old_log_probs = torch.as_tensor(batch["old_log_prob"], device=device, dtype=torch.float32)
        old_values = torch.as_tensor(batch["old_value"], device=device, dtype=torch.float32)
        batch_theory = torch.as_tensor(batch["theory"], device=device, dtype=torch.float32)
        batch_theory = self._coerce_theory_batch(batch_theory)
        batch_trust = torch.as_tensor(batch["trust"], device=device, dtype=torch.float32)

        B, T, D = actions.shape
        prev_actions = torch.cat(
            [torch.zeros(B, 1, D, device=device), actions[:, :-1, :]], dim=1
        )

        # 2. COMPUTE GAE-λ ADVANTAGES
        next_obs = torch.as_tensor(batch["next_obs"], device=device, dtype=torch.float32)
        next_hx = torch.as_tensor(batch["next_hx"], device=device, dtype=torch.float32)

        # Bootstrap with cached theory/trust from last step
        last_theory = batch_theory[:, -1]
        last_trust = batch_trust[:, -1]
        if last_theory.ndim == 1:
            last_theory = last_theory.unsqueeze(1)
        if last_trust.ndim == 1:
            last_trust = last_trust.unsqueeze(1)

        with torch.no_grad():
            last_action = actions[:, -1]
            _, last_value, _, _ = self.brain(
                next_obs, last_theory, last_action, next_hx,
                theory_confidence=last_trust,
            )

            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards)
            gae = 0.0

            for t in reversed(range(T)):
                next_non_terminal = 1.0 - dones[:, t]
                next_val = last_value if (t == T - 1) else old_values[:, t + 1]
                delta = (rewards[:, t]
                         + self.config.gamma * next_val * next_non_terminal
                         - old_values[:, t])
                gae = (delta
                       + self.config.gamma * self.config.gae_lambda
                       * next_non_terminal * gae)
                advantages[:, t] = gae
                returns[:, t] = gae + old_values[:, t]

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. PPO UPDATE LOOP
        total_loss = 0.0
        clip_eps = self.config.clip_epsilon

        for epoch in range(self.config.ppo_epochs):
            mus, stds, values, stresses, rec_losses = [], [], [], [], []
            curr_hx = hx_batch

            for t in range(T):
                o_t = obs[:, t]
                a_prev_t = prev_actions[:, t]

                # Use CACHED theory/trust (no racy Scientist calls during PPO)
                th_t = batch_theory[:, t]
                trust_t = batch_trust[:, t]
                if th_t.ndim == 1:
                    th_t = th_t.unsqueeze(1)
                if trust_t.ndim == 1:
                    trust_t = trust_t.unsqueeze(1)

                (mu, std), value, new_hx, stress = self.brain(
                    o_t, th_t, a_prev_t, curr_hx, theory_confidence=trust_t
                )

                mus.append(mu)
                stds.append(std)
                values.append(value)
                stresses.append(stress)

                # Visual reconstruction loss (Eyes)
                if self.brain.vision_mode == "fno" and (t % 8 == 0):
                    recon_field = self.brain.eyes(o_t)
                    rec_loss = torch.nn.functional.mse_loss(recon_field, o_t[:, :3])
                else:
                    rec_loss = torch.tensor(0.0, device=device)

                rec_losses.append(rec_loss)
                curr_hx = new_hx

            mus = torch.stack(mus, dim=1)
            stds = torch.stack(stds, dim=1)

            # Numerical safety
            if not torch.isfinite(mus).all() or not torch.isfinite(stds).all():
                mus = torch.nan_to_num(mus, nan=0.0, posinf=1.0, neginf=-1.0)
                stds = torch.nan_to_num(stds, nan=1.0, posinf=1.0, neginf=0.1)

            new_values = torch.stack(values, dim=1)
            stress_mean = torch.stack(stresses).mean()
            visual_loss = torch.stack(rec_losses).mean()

            # TanhNormal — the REAL distribution
            dist = TanhNormal(mus, stds)
            new_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            actor_loss = actor_loss - self.config.entropy_coef * entropy

            # Clipped value loss (dual clipping)
            value_clipped = old_values + torch.clamp(
                new_values - old_values, -clip_eps, clip_eps
            )
            critic_loss1 = (returns - new_values).pow(2)
            critic_loss2 = (returns - value_clipped).pow(2)
            critic_loss = 0.5 * torch.max(critic_loss1, critic_loss2).mean()

            loss = (actor_loss
                    + self.config.critic_coef * critic_loss
                    + self.config.stress_coef * stress_mean
                    + self.config.visual_coef * visual_loss)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = _safe_clip_grads(self.brain.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss += float(loss.item())

        # Delayed Hebbian growth
        self.brain.apply_hebbian_growth()

        avg_loss = total_loss / float(self.config.ppo_epochs)
        self.history.ppo_loss.append(avg_loss)
        self.history.grad_norm.append(grad_norm)

        return {"loss": avg_loss, "grad_norm": grad_norm}

    # -----------------------------------------------------------------
    # INVERSE DESIGN (Phase 2)
    # -----------------------------------------------------------------

    def _default_inverse_parameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Default candidate parameter bounds for world-backed inverse design."""
        space: Dict[str, Tuple[float, float]] = {
            "action_bias": (-1.0, 1.0),
            "action_gain": (0.0, 1.0),
            "action_frequency": (0.0, 0.35),
            "action_phase": (-math.pi, math.pi),
        }

        if self.config.world_spec.startswith("analytical:"):
            space["world_dt"] = (0.02, 0.30)
        elif self.config.world_spec.startswith("lbm2d:"):
            space["world_u_inlet"] = (0.01, 0.12)
            space["world_tau"] = (0.52, 1.20)
        elif self.config.world_spec.startswith("supersonic:"):
            space["world_tau"] = (0.65, 1.05)
            space["world_inflow_velocity"] = (0.08, 0.28)
            space["world_jet_gain"] = (0.002, 0.03)
            space["world_reward_control_penalty"] = (0.01, 0.2)

        return space

    def _load_inverse_parameter_space(
        self,
        parameter_space_path: Optional[str],
    ) -> Dict[str, Tuple[float, float]]:
        if parameter_space_path is None:
            return self._default_inverse_parameter_space()

        raw = json.loads(Path(parameter_space_path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Inverse-design parameter space JSON must be an object")

        parsed: Dict[str, Tuple[float, float]] = {}
        for name, bounds in raw.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"Parameter '{name}' must have [low, high] bounds")
            low = float(bounds[0])
            high = float(bounds[1])
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError(f"Invalid bounds for '{name}': {bounds}")
            parsed[str(name)] = (low, high)

        if not parsed:
            raise ValueError("Inverse-design parameter space cannot be empty")
        return parsed

    def _load_inverse_objective_spec(
        self,
        objective_spec_path: Optional[str],
        parameter_space: Mapping[str, Tuple[float, float]],
        iterations: Optional[int],
        population: Optional[int],
    ) -> ObjectiveSpec:
        if objective_spec_path is None:
            targets = {"reward_mean": 0.0, "turbulence": 0.0}
            constraints = {
                "reward_mean": {"min": -1.0},
                "action_energy": {"max": 1.0},
                "termination_ratio": {"max": 0.5},
            }
            penalties = {
                "action_energy": 0.20,
                "reward_std": 0.10,
                "density_var": 0.05,
            }
            solver_budget: Dict[str, Any] = {
                "iterations": 8,
                "population": 12,
                "seed": int(self.config.seed),
            }
            hard_bounds = {k: (float(v[0]), float(v[1])) for k, v in parameter_space.items()}
        else:
            raw = json.loads(Path(objective_spec_path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Inverse-design objective JSON must be an object")

            targets = {str(k): float(v) for k, v in dict(raw.get("targets", {})).items()}
            constraints = dict(raw.get("constraints", {}))
            penalties = {str(k): float(v) for k, v in dict(raw.get("penalties", {})).items()}
            solver_budget = dict(raw.get("solver_budget", {}))

            hard_bounds_raw = dict(raw.get("hard_bounds", {}))
            hard_bounds: Dict[str, Tuple[float, float]] = {}
            for k, v in hard_bounds_raw.items():
                if not isinstance(v, (list, tuple)) or len(v) != 2:
                    raise ValueError(f"Hard bound '{k}' must be [low, high]")
                low = float(v[0])
                high = float(v[1])
                if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                    raise ValueError(f"Invalid hard bound for '{k}': {v}")
                hard_bounds[str(k)] = (low, high)

            # Ensure candidate space has explicit bounds even when JSON omits them.
            for name, bounds in parameter_space.items():
                hard_bounds.setdefault(str(name), (float(bounds[0]), float(bounds[1])))

        if iterations is not None:
            solver_budget["iterations"] = int(iterations)
        if population is not None:
            solver_budget["population"] = int(population)
        solver_budget.setdefault("seed", int(self.config.seed))

        return ObjectiveSpec(
            targets=targets,
            constraints=constraints,
            penalties=penalties,
            hard_bounds=hard_bounds,
            solver_budget=solver_budget,
        )

    def _candidate_action_vector(
        self,
        params: Mapping[str, float],
        step: int,
        action_dim: int,
    ) -> np.ndarray:
        vec = np.zeros(action_dim, dtype=np.float32)

        default_bias = float(params.get("action_bias", 0.0))
        default_gain = abs(float(params.get("action_gain", 0.0)))
        default_freq = abs(float(params.get("action_frequency", 0.0)))
        default_phase = float(params.get("action_phase", 0.0))

        for i in range(action_dim):
            bias = float(params.get(f"action_bias_{i}", default_bias))
            gain = abs(float(params.get(f"action_gain_{i}", default_gain)))
            freq = abs(float(params.get(f"action_frequency_{i}", default_freq)))
            phase = float(params.get(f"action_phase_{i}", default_phase))
            signal = bias + gain * np.sin(2.0 * np.pi * freq * float(step) + phase)
            vec[i] = np.float32(np.clip(signal, -1.0, 1.0))

        return vec

    def _candidate_world_kwargs(self, params: Mapping[str, float]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "nx": int(self.config.grid_shape[0]),
            "ny": int(self.config.grid_shape[1]),
            "nz": int(self.config.grid_shape[2]),
        }
        for key, value in self.base_world_kwargs.items():
            k = str(key)
            if k in {"nx", "ny", "nz"}:
                continue
            kwargs[k] = value
        for key, value in params.items():
            if str(key).startswith("world_"):
                kwargs[str(key)[6:]] = float(value)
        return kwargs

    @staticmethod
    def _failed_candidate_metrics() -> Dict[str, float]:
        return {
            "reward_mean": -1e6,
            "reward_std": 1e6,
            "reward_min": -1e6,
            "reward_max": -1e6,
            "mean_speed": 1e6,
            "turbulence": 1e6,
            "density_var": 1e6,
            "action_energy": 1e6,
            "steps_executed": 0.0,
            "termination_ratio": 1.0,
            "drag_proxy": 1e6,
            "lift_proxy": 0.0,
            "drag": 1e6,
            "lift": 0.0,
            "mass": 1e6,
        }

    def _simulate_design_candidate(
        self,
        params: Mapping[str, float],
        rollout_steps: int,
    ) -> Dict[str, float]:
        """Evaluate one candidate by rolling out a real WorldAdapter instance."""
        n_steps = max(4, int(rollout_steps))
        try:
            world = create_world(self.config.world_spec, **self._candidate_world_kwargs(params))
            obs, _ = world.reset()
        except Exception as exc:
            self.log.warning(f"Candidate world init failed: {exc}")
            return self._failed_candidate_metrics()

        rewards: List[float] = []
        mean_speeds: List[float] = []
        turbulences: List[float] = []
        density_vars: List[float] = []
        action_energies: List[float] = []
        terminated = 0

        try:
            for step in range(n_steps):
                action_vec = self._candidate_action_vector(params, step, world.action_dim)
                obs, reward, done, _ = world.step(action_vec)
                reward_f = float(np.asarray(reward).reshape(-1)[0])
                rewards.append(reward_f)

                obs_arr = np.asarray(obs, dtype=np.float32)
                if obs_arr.ndim == 5:
                    field = obs_arr[0]
                elif obs_arr.ndim == 4:
                    field = obs_arr
                else:
                    raise ValueError(f"Unexpected observation shape: {obs_arr.shape}")

                vel = field[:3]
                speed = np.linalg.norm(vel, axis=0)
                mean_speeds.append(float(np.mean(speed)))
                turbulences.append(float(np.std(speed)))
                density_vars.append(float(np.var(field[3])))
                action_energies.append(float(np.mean(np.square(action_vec))))

                if done:
                    terminated = 1
                    break
        except Exception as exc:
            self.log.warning(f"Candidate rollout failed: {exc}")
            return self._failed_candidate_metrics()

        if not rewards:
            return self._failed_candidate_metrics()

        reward_mean = float(np.mean(rewards))
        reward_std = float(np.std(rewards))
        mean_speed = float(np.mean(mean_speeds))
        turbulence = float(np.mean(turbulences))
        density_var = float(np.mean(density_vars))
        action_energy = float(np.mean(action_energies))
        steps_executed = float(len(rewards))

        return {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "mean_speed": mean_speed,
            "turbulence": turbulence,
            "density_var": density_var,
            "action_energy": action_energy,
            "steps_executed": steps_executed,
            "termination_ratio": float(terminated),
            # Compatibility aliases for common inverse-design vocabularies.
            "drag_proxy": max(0.0, -reward_mean),
            "lift_proxy": mean_speed,
            "drag": max(0.0, -reward_mean),
            "lift": mean_speed,
            "mass": action_energy,
        }

    def run_inverse_design(
        self,
        *,
        backend: str = "evolutionary",
        iterations: Optional[int] = None,
        population: Optional[int] = None,
        top_k: int = 5,
        rollout_steps: int = 64,
        objective_spec_path: Optional[str] = None,
        parameter_space_path: Optional[str] = None,
        report_name: str = "inverse_design_report.json",
    ) -> Dict[str, Any]:
        """Run inverse design using real world-adapter simulations as evaluator."""
        parameter_space = self._load_inverse_parameter_space(parameter_space_path)
        objective_spec = self._load_inverse_objective_spec(
            objective_spec_path,
            parameter_space,
            iterations,
            population,
        )

        self.log.info(
            f"Inverse design start | backend={backend} | world={self.config.world_spec} "
            f"| rollout_steps={rollout_steps}"
        )

        engine = InverseDesignEngine(
            objective_spec=objective_spec,
            simulator=lambda p: self._simulate_design_candidate(p, rollout_steps),
            parameter_space=parameter_space,
            seed=self.config.seed,
        )
        report = engine.run(
            backend=backend,
            iterations=iterations,
            population=population,
            top_k=top_k,
        )

        report_path = self.output_dir / report_name
        engine.export_report(report, report_path)
        self.log.info(f"Inverse design report saved to {report_path}")

        print(f"\n{'='*70}")
        print(f"  INVERSE DESIGN COMPLETE: {self.config.name}")
        print(f"  Backend: {backend} | Candidates: {len(report.candidates)}")
        print(f"  Report: {report_path}")
        if report.top_candidates:
            best = report.top_candidates[0]
            print(
                f"  Best: {best.candidate_id} | score={best.objective_score:.4f} "
                f"| feasible={best.feasible}"
            )
        print(f"{'='*70}\n")

        return report.to_dict()

    # -----------------------------------------------------------------
    # MAIN RUN LOOP (Exact match to training_loop.py:run)
    # -----------------------------------------------------------------

    def run(self) -> ExperimentHistory:
        """
        Execute the full ATOM Wake/Sleep training loop.
        Returns the experiment history.
        """
        self.log.info("ATOM WAKE CYCLE INITIATED")

        # Initialize environment
        obs_np, _ = self.world.reset()
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).to(self.device)

        # Ensure correct shape (B, 4, X, Y, Z)
        if obs_t.dim() == 4:
            obs_t = obs_t.unsqueeze(0)

        # Brain state
        hx = None
        act_dim = self.config.action_dim
        last_action = torch.zeros(1, act_dim, device=self.device)

        total_steps = 0
        start_time = time.time()

        print(f"\n{'='*70}")
        print(f"  ATOM EXPERIMENT: {self.config.name}")
        print(f"  World: {self.config.world_spec} | Grid: {self.config.grid_shape}")
        print(f"  Steps: {self.config.max_steps} | Device: {self.device}")
        print(f"{'='*70}\n")

        try:
            while total_steps < self.config.max_steps:
                step_start = time.time()

                # --- 1. SENSE & THINK ---

                # A. Build feature vector for Scientist
                f_vec = self._build_feature_vector(obs_t, last_action)

                # B. Scientist prediction (System 1 intuition + V2 Structural Signal)
                sci_out = None
                if self.config.ablate_scientist:
                    theory_t = torch.zeros(1, int(self.brain.theory_dim), device=self.device)
                    theory_conf_t = torch.zeros(1, 1, device=self.device)
                    theory_val_log = 0.0
                    raw_trust_log = 0.0
                    verified_trust_log = 0.0
                    structural_floor_log = 0.0
                    diag_floor_applied_log = 0.0
                    oos_trust_log = 0.0
                else:
                    sci_out = self.scientist.get_signal(
                        features=f_vec,
                        reward=None,
                        action=None,
                        device=self.device
                    )
                    theory_t = sci_out["tensor"]  # (1, 10)
                    theory_conf_t = torch.tensor([[sci_out["trust"]]], dtype=torch.float32, device=self.device)
                    theory_val_log = sci_out["prediction"]
                    raw_trust_log = float(sci_out.get("raw_trust", sci_out.get("trust", 0.0)))
                    verified_trust_log = float(sci_out.get("verified_trust", sci_out.get("trust", 0.0)))
                    structural_floor_log = float(sci_out.get("trust_structural_floor", 0.0))
                    diag_floor_applied_log = float(sci_out.get("diagnostic_trust_floor_applied", 0.0))
                    oos_trust_log = float(
                        sci_out.get("verifier_metrics", {}).get(
                            "out_of_sample_trust", verified_trust_log
                        )
                    )


                # Ablation: force trust=0 (pure neural, no constraint blending)
                if self.config.ablate_trust_gate:
                    theory_conf_t = torch.zeros_like(theory_conf_t)

                # C. Brain forward
                (mu, std), value, hx_new, stress = self.brain(
                    obs_t, theory_t, last_action, hx,
                    theory_confidence=theory_conf_t,
                )

                theory_action_delta_l2 = 0.0
                theory_action_delta_frac = 0.0
                if not self.config.ablate_scientist:
                    with torch.no_grad():
                        zero_theory_t = torch.zeros_like(theory_t)
                        zero_conf_t = torch.zeros_like(theory_conf_t)
                        (mu_no_theory, _), _, _, _ = self.brain(
                            obs_t,
                            zero_theory_t,
                            last_action,
                            hx,
                            theory_confidence=zero_conf_t,
                        )
                        mean_action = torch.tanh(mu.detach())
                        mean_action_no_theory = torch.tanh(mu_no_theory.detach())
                        delta_action = mean_action - mean_action_no_theory
                        delta_l2 = float(torch.linalg.vector_norm(delta_action).item())
                        base_l2 = float(torch.linalg.vector_norm(mean_action_no_theory).item())
                        theory_action_delta_l2 = delta_l2
                        theory_action_delta_frac = delta_l2 / (base_l2 + 1e-6)

                # D. Sample action with TanhNormal (NOT torch.distributions.Normal!)
                dist = TanhNormal(mu, std)
                action_phys, raw_action = dist.sample()
                action_phys = torch.clamp(action_phys, -1.0, 1.0)

                # E. Compute log_prob for PPO (must use pre_tanh_value for correctness)
                log_prob = dist.log_prob(action_phys, pre_tanh_value=raw_action).sum(-1, keepdim=True)

                # --- 2. ACT & FEEL ---
                safety_decision = None
                action_np = action_phys.detach().cpu().numpy().reshape(-1)
                if self.safety_supervisor is not None:
                    in_scientist_warmup = bool(sci_out.get("in_warmup", False)) if sci_out is not None else False
                    safety_diag = self._build_safety_diagnostics(
                        obs_t=obs_t,
                        f_vec=f_vec,
                        theory_conf_t=theory_conf_t,
                        stress=stress,
                        scientist_warmup_active=in_scientist_warmup,
                    )
                    safety_decision = self.safety_supervisor.review(action_np, safety_diag)
                    action_np = safety_decision.action.astype(np.float32).reshape(-1)
                    action_phys = torch.as_tensor(action_np, dtype=torch.float32, device=self.device).view(1, -1)
                    log_prob = dist.log_prob(action_phys).sum(-1, keepdim=True)

                next_obs_np, reward, done, info = self.world.step(action_np)
                info = dict(info or {})
                if safety_decision is not None:
                    info["safety"] = safety_decision.to_dict()
                step_divergence, div_source = _extract_divergence_metric(info, next_obs_np)
                self._divergence_source_counts[div_source] = (
                    self._divergence_source_counts.get(div_source, 0) + 1
                )
                info["divergence_rms"] = float(step_divergence)

                next_obs_t = torch.as_tensor(
                    next_obs_np, dtype=torch.float32
                ).to(self.device)
                if next_obs_t.dim() == 4:
                    next_obs_t = next_obs_t.unsqueeze(0)

                reward = float(np.asarray(reward).reshape(-1)[0])

                # --- 3. MEMORY & SCIENCE ---
                self.memory.push(
                    obs_t, action_phys, reward, done, hx_new,
                    log_prob=log_prob, value=value,
                    theory=theory_t, trust=theory_conf_t,
                )

                if not self.config.ablate_scientist:
                    self.scientist.observe_and_verify(
                        f_vec, reward, theory_val_log,
                        float(action_phys.flatten()[0].item()),
                    )

                # --- 4. LEARN (Online PPO) ---
                if self.memory.size > self.memory.seq_len + self.config.batch_size:
                    _ = self._update_brain()

                # --- 5. SLEEP CHECK (Symbolic Discovery) ---
                if (not self.config.ablate_scientist
                        and total_steps > 0
                        and total_steps % self.config.sleep_interval == 0):
                    new_law = self.scientist.ponder()
                    if new_law:
                        self.log.info(f"Law discovered: {new_law}")
                        self.history.laws_discovered.append(new_law)
                        self.history.best_law = new_law
                        if self.on_law_discovered:
                            self.on_law_discovered(new_law, total_steps)

                # --- 6. RECORD ---
                self.history.reward.append(reward)
                self.history.stress.append(float(stress.mean().item()))
                self.history.divergence.append(float(step_divergence))
                self.history.theory_score.append(theory_val_log)

                trust_log = float(theory_conf_t.item())
                self.history.theory_trust.append(trust_log)
                self.history.theory_trust_raw.append(raw_trust_log)
                self.history.theory_trust_verified.append(verified_trust_log)
                self.history.theory_trust_structural_floor.append(structural_floor_log)
                self.history.theory_trust_diag_floor_applied.append(diag_floor_applied_log)
                self.history.theory_verifier_oos_trust.append(oos_trust_log)
                self.history.theory_action_delta_l2.append(float(theory_action_delta_l2))
                self.history.theory_action_delta_frac.append(float(theory_action_delta_frac))
                if safety_decision is None:
                    self.history.safety_intervention.append(0.0)
                    self.history.safety_fallback.append(0.0)
                else:
                    self.history.safety_intervention.append(1.0 if safety_decision.intervened else 0.0)
                    self.history.safety_fallback.append(1.0 if safety_decision.fallback_used else 0.0)

                # V2 metrics for logging
                v2_trust = 0.0
                j_norm = 0.0
                j_raw = 0.0
                s_prior = 0.0
                in_warmup = False
                bl_r2 = 0.0
                if not self.config.ablate_scientist and sci_out is not None:
                    v2_trust = sci_out.get("verified_trust", 0.0)
                    j_norm = float(np.linalg.norm(sci_out.get("jacobian", np.zeros(8))))
                    j_raw = sci_out.get("jacobian_raw_norm", j_norm)
                    in_warmup = sci_out.get("in_warmup", False)
                    bl_r2 = sci_out.get("baseline_r2", 0.0)
                    if hasattr(self.scientist, 'verifier'):
                        s_prior = self.scientist.verifier._structural_prior

                step_dt = time.time() - step_start
                self.history.step_times.append(step_dt)

                # --- 7. LOG ---
                if total_steps % self.config.log_interval == 0:
                    avg_r = float(np.mean(self.history.reward[-50:])) if self.history.reward else 0.0
                    fps = 1.0 / (step_dt + 1e-9)
                    law_str = (self.scientist.best_law or "searching...")[:40] if not self.config.ablate_scientist else "ABLATED"
                    warmup_tag = " WARM" if in_warmup else ""
                    # Show both normalized J (what Brain sees) and raw J (what PySR produced)
                    j_display = f"J={j_norm:.3f}" if j_raw <= 1.0 else f"J={j_norm:.3f}({j_raw:.1f})"
                    # Show baseline R² when fitted (how much trivial variance removed)
                    bl_display = f" R²={bl_r2:.2f}" if bl_r2 > 0.01 else ""
                    safety_i50 = float(np.mean(self.history.safety_intervention[-50:])) if self.history.safety_intervention else 0.0
                    safety_f50 = float(np.mean(self.history.safety_fallback[-50:])) if self.history.safety_fallback else 0.0
                    delta_a50 = (
                        float(np.mean(self.history.theory_action_delta_l2[-50:]))
                        if self.history.theory_action_delta_l2
                        else 0.0
                    )
                    print(
                        f"  Step {total_steps:4d} | "
                        f"R={avg_r:+8.4f} | "
                        f"Stress={stress.mean().item():.4f} | "
                        f"Trust={trust_log:.3f} | "
                        f"SafeI={safety_i50:.2f} F={safety_f50:.2f} | "
                        f"V2={v2_trust:.2f} SP={s_prior:.2f} dA={delta_a50:.3f} {j_display}{bl_display}{warmup_tag} | "
                        f"Law: {law_str} | "
                        f"{fps:.1f} FPS"
                    )

                # --- 8. CALLBACK ---
                if self.on_step:
                    self.on_step(total_steps, obs_t, action_phys, reward, info, self.history)

                # --- 9. SAVE CHECKPOINT ---
                if total_steps > 0 and total_steps % self.config.save_interval == 0:
                    self._save_checkpoint(total_steps)

                # Advance
                obs_t = next_obs_t
                hx = hx_new.detach()
                last_action = action_phys
                total_steps += 1

                if done:
                    # Reset environment, keep Brain/Scientist state
                    obs_np, _ = self.world.reset()
                    obs_t = torch.as_tensor(obs_np, dtype=torch.float32).to(self.device)
                    if obs_t.dim() == 4:
                        obs_t = obs_t.unsqueeze(0)
                    hx = None

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
        finally:
            self.scientist.shutdown()

        elapsed = time.time() - start_time
        self._save_results(elapsed)

        print(f"\n{'='*70}")
        print(f"  EXPERIMENT COMPLETE: {self.config.name}")
        print(f"  Steps: {total_steps} | Time: {elapsed:.1f}s | "
              f"Avg FPS: {total_steps/(elapsed+1e-9):.1f}")
        if self.history.reward:
            print(f"  Final Reward (last 50): {np.mean(self.history.reward[-50:]):.4f}")
        if self.history.best_law:
            print(f"  Best Law: {self.history.best_law}")
        print(f"{'='*70}\n")

        return self.history

    # -----------------------------------------------------------------
    # SAVE / LOAD
    # -----------------------------------------------------------------

    def _save_checkpoint(self, step: int):
        path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save({
            "step": step,
            "brain_state_dict": self.brain.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def _save_results(self, elapsed: float):
        safety_snapshot = (
            self.safety_supervisor.snapshot()
            if self.safety_supervisor is not None
            else {
                "steps": len(self.history.reward),
                "interventions": 0,
                "fallback_uses": 0,
                "mode": "off",
                "hard_event_rate": 0.0,
                "degrade_window": 0,
                "mode_transitions": {},
                "mode_step_counts": {"normal": 0, "cautious": 0, "safe_hold": 0},
                "mode_step_fraction": {"normal": 0.0, "cautious": 0.0, "safe_hold": 0.0},
                "reason_histogram": {},
                "envelope": self.control_envelope.to_dict(),
                "policy": None,
            }
        )
        total_steps = max(1, len(self.history.reward))
        safety_interventions = int(sum(self.history.safety_intervention))
        safety_fallback_uses = int(sum(self.history.safety_fallback))
        mode_fraction = dict(safety_snapshot.get("mode_step_fraction", {}))

        results = {
            "config": {
                "name": self.config.name,
                "world": self.config.world_spec,
                "grid_shape": list(self.config.grid_shape),
                "max_steps": self.config.max_steps,
                "ablate_scientist": self.config.ablate_scientist,
                "ablate_symplectic": self.config.ablate_symplectic,
                "ablate_trust_gate": self.config.ablate_trust_gate,
                "scientist_signal": self.scientist_signal_mode,
                "scientist_warmup_steps": int(self.config.scientist_warmup_steps),
                "scientist_diag_trust_floor": float(self.config.scientist_diag_trust_floor),
                "scientist_diag_trust_floor_start_step": (
                    int(self.config.scientist_diag_trust_floor_start_step)
                    if self.config.scientist_diag_trust_floor_start_step is not None
                    else None
                ),
                "scientist_signal_trust_scale_mode": str(
                    self.config.scientist_signal_trust_scale_mode
                ),
                "discovery_target_mode": str(
                    os.getenv("ATOM_SCIENTIST_DISCOVERY_TARGET_MODE", "hybrid")
                ).strip().lower(),
                "device": self.device,
                "safety_enable": self.config.safety_enable,
                "safety_min_theory_trust": self.config.safety_min_theory_trust,
                "safety_max_stress": self.config.safety_max_stress,
                "safety_shift_z_threshold": self.config.safety_shift_z_threshold,
                "safety_shift_warmup": self.config.safety_shift_warmup,
                "safety_density_min": self.config.safety_density_min,
                "safety_density_max": self.config.safety_density_max,
                "safety_speed_max": self.config.safety_speed_max,
                "safety_cautious_hard_rate": self.config.safety_cautious_hard_rate,
                "safety_safe_hold_hard_rate": self.config.safety_safe_hold_hard_rate,
                "safety_cautious_action_scale": self.config.safety_cautious_action_scale,
                "safety_recovery_hysteresis": self.config.safety_recovery_hysteresis,
                "safety_low_trust_hard_ratio": self.config.safety_low_trust_hard_ratio,
                "safety_low_trust_hard_warmup": self.config.safety_low_trust_hard_warmup,
            },
            "summary": {
                "total_steps": len(self.history.reward),
                "elapsed_s": elapsed,
                "final_reward_50": float(np.mean(self.history.reward[-50:])) if self.history.reward else 0.0,
                "final_stress_50": float(np.mean(self.history.stress[-50:])) if self.history.stress else 0.0,
                "final_divergence_50": (
                    float(np.mean(self.history.divergence[-50:]))
                    if self.history.divergence
                    else 0.0
                ),
                "final_theory_trust_50": (
                    float(np.mean(self.history.theory_trust[-50:]))
                    if self.history.theory_trust
                    else 0.0
                ),
                "final_theory_trust_raw_50": (
                    float(np.mean(self.history.theory_trust_raw[-50:]))
                    if self.history.theory_trust_raw
                    else 0.0
                ),
                "final_theory_trust_verified_50": (
                    float(np.mean(self.history.theory_trust_verified[-50:]))
                    if self.history.theory_trust_verified
                    else 0.0
                ),
                "final_theory_trust_structural_floor_50": (
                    float(np.mean(self.history.theory_trust_structural_floor[-50:]))
                    if self.history.theory_trust_structural_floor
                    else 0.0
                ),
                "final_theory_trust_diag_floor_applied_50": (
                    float(np.mean(self.history.theory_trust_diag_floor_applied[-50:]))
                    if self.history.theory_trust_diag_floor_applied
                    else 0.0
                ),
                "final_theory_verifier_oos_trust_50": (
                    float(np.mean(self.history.theory_verifier_oos_trust[-50:]))
                    if self.history.theory_verifier_oos_trust
                    else 0.0
                ),
                "final_theory_action_delta_l2_50": (
                    float(np.mean(self.history.theory_action_delta_l2[-50:]))
                    if self.history.theory_action_delta_l2
                    else 0.0
                ),
                "final_theory_action_delta_frac_50": (
                    float(np.mean(self.history.theory_action_delta_frac[-50:]))
                    if self.history.theory_action_delta_frac
                    else 0.0
                ),
                "best_law": self.history.best_law,
                "laws_discovered": len(self.history.laws_discovered),
                "safety_interventions": safety_interventions,
                "safety_fallback_uses": safety_fallback_uses,
                "safety_intervention_rate": float(safety_interventions) / float(total_steps),
                "safety_fallback_rate": float(safety_fallback_uses) / float(total_steps),
                "safety_hard_event_rate": float(safety_snapshot.get("hard_event_rate", 0.0)),
                "safety_mode_normal_rate": float(mode_fraction.get("normal", 0.0)),
                "safety_mode_cautious_rate": float(mode_fraction.get("cautious", 0.0)),
                "safety_mode_safe_hold_rate": float(mode_fraction.get("safe_hold", 0.0)),
                "divergence_source": (
                    max(self._divergence_source_counts.items(), key=lambda item: item[1])[0]
                    if self._divergence_source_counts
                    else "unavailable"
                ),
            },
            "history": self.history.to_dict(),
            "safety": safety_snapshot,
        }
        path = self.output_dir / "results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.log.info(f"Results saved to {path}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ATOM Canonical Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analytical world (no JAX needed, fast)
  python atom_experiment_runner.py --world analytical:taylor_green --steps 300

  # LBM cylinder (requires JAX)
  python atom_experiment_runner.py --world lbm:cylinder --steps 1000 --grid 64 32 24

  # Ablation: disable scientist
  python atom_experiment_runner.py --world analytical:taylor_green --ablate-scientist

  # Ablation: disable trust gate
  python atom_experiment_runner.py --world analytical:taylor_green --ablate-trust-gate

  # Inverse design using real world adapters as evaluator
  python atom_experiment_runner.py --inverse-design --world analytical:taylor_green \
      --inverse-backend evolutionary --inverse-iterations 6 --inverse-population 10

Available worlds:
""" + "\n".join(f"  {k:35s} {v}" for k, v in list_available_worlds().items())
    )

    parser.add_argument("--world", type=str, default="analytical:taylor_green",
                        help="World specification (backend:scenario)")
    parser.add_argument("--name", type=str, default="atom_run",
                        help="Experiment name")
    parser.add_argument("--steps", type=int, default=500,
                        help="Maximum training steps")
    parser.add_argument("--grid", type=int, nargs=3, default=[32, 32, 16],
                        help="Grid shape (X Y Z)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="PPO batch size")
    parser.add_argument("--neurons", type=int, default=32,
                        help="Brain internal neurons")
    parser.add_argument("--vision-dim", type=int, default=128,
                        help="Visual embedding dimension")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, mps)")
    parser.add_argument(
        "--scientist-signal",
        type=str,
        default=str(os.getenv("ATOM_SCIENTIST_SIGNAL_MODE", "v2")).strip().lower(),
        choices=["v1", "v2"],
        help="Scientist signal mode: v1 (scalar) or v2 (structural 10D)",
    )
    parser.add_argument(
        "--scientist-warmup-steps",
        type=int,
        default=50,
        help="Warmup steps for V2 scientist signal injection (0 disables warmup gate).",
    )
    parser.add_argument(
        "--scientist-diag-trust-floor",
        type=float,
        default=0.0,
        help="Optional diagnostic trust floor [0,1] for V2 signal (default: disabled).",
    )
    parser.add_argument(
        "--scientist-diag-trust-floor-start-step",
        type=int,
        default=None,
        help="Step at which diagnostic trust floor starts (default: warmup_steps+1).",
    )
    parser.add_argument(
        "--scientist-signal-trust-scale-mode",
        type=str,
        default=str(os.getenv("ATOM_SCIENTIST_SIGNAL_TRUST_SCALE_MODE", "off")).strip().lower(),
        choices=["off", "sqrt", "full"],
        help="How scientist trust scales prediction/Jacobian content before Brain gating.",
    )
    parser.add_argument("--ablate-scientist", action="store_true",
                        help="Disable symbolic regression")
    parser.add_argument("--ablate-symplectic", action="store_true",
                        help="Disable Hamiltonian constraint")
    parser.add_argument("--ablate-trust-gate", action="store_true",
                        help="Force trust=0 (pure neural)")
    parser.add_argument("--disable-safety", action="store_true",
                        help="Disable runtime safety envelope + fallback arbitration")
    parser.add_argument("--safety-min-theory-trust", type=float, default=0.08,
                        help="Trust threshold below which fallback action is used")
    parser.add_argument("--safety-max-stress", type=float, default=8.0,
                        help="Stress threshold above which fallback action is used")
    parser.add_argument("--safety-shift-z-threshold", type=float, default=8.0,
                        help="Feature z-score threshold for distribution shift intervention")
    parser.add_argument("--safety-shift-warmup", type=int, default=24,
                        help="Warmup steps before distribution shift checks activate")
    parser.add_argument("--safety-density-min", type=float, default=0.05,
                        help="Minimum allowed density proxy in safety supervisor")
    parser.add_argument("--safety-density-max", type=float, default=8.0,
                        help="Maximum allowed density proxy in safety supervisor")
    parser.add_argument("--safety-speed-max", type=float, default=3.0,
                        help="Maximum allowed mean speed in safety supervisor")
    parser.add_argument("--safety-cautious-hard-rate", type=float, default=0.30,
                        help="Hard-event rate threshold to enter cautious mode")
    parser.add_argument("--safety-safe-hold-hard-rate", type=float, default=0.60,
                        help="Hard-event rate threshold to enter safe-hold mode")
    parser.add_argument("--safety-cautious-action-scale", type=float, default=0.50,
                        help="Action scaling factor in cautious mode")
    parser.add_argument("--safety-recovery-hysteresis", type=float, default=0.80,
                        help="Hysteresis factor for conservative-mode recovery")
    parser.add_argument("--safety-low-trust-hard-ratio", type=float, default=0.20,
                        help="Fraction of effective trust threshold treated as hard low-trust")
    parser.add_argument("--safety-low-trust-hard-warmup", type=int, default=24,
                        help="Steps before low-trust can trigger hard intervention")
    parser.add_argument("--output-dir", type=str, default="experiment_results",
                        help="Output directory")
    parser.add_argument(
        "--world-u-inlet",
        type=float,
        default=None,
        help="Optional world inlet velocity override (lbm/lbm2d worlds).",
    )
    parser.add_argument(
        "--world-tau",
        type=float,
        default=None,
        help="Optional world relaxation-time override (lbm/lbm2d worlds).",
    )
    parser.add_argument("--inverse-design", action="store_true",
                        help="Run inverse-design mode instead of training loop")
    parser.add_argument("--inverse-backend", type=str, default="evolutionary",
                        choices=["evolutionary", "gradient", "bayesian"],
                        help="Inverse-design optimizer backend")
    parser.add_argument("--inverse-iterations", type=int, default=None,
                        help="Override inverse-design iteration count")
    parser.add_argument("--inverse-population", type=int, default=None,
                        help="Override inverse-design population size")
    parser.add_argument("--inverse-top-k", type=int, default=5,
                        help="Top-K candidates to preserve in output report")
    parser.add_argument("--inverse-rollout-steps", type=int, default=64,
                        help="World rollout steps per candidate evaluation")
    parser.add_argument("--inverse-objective", type=str, default=None,
                        help="Path to objective JSON (ObjectiveSpec-compatible)")
    parser.add_argument("--inverse-param-space", type=str, default=None,
                        help="Path to parameter-space JSON {name:[low,high], ...}")
    parser.add_argument("--inverse-report", type=str, default="inverse_design_report.json",
                        help="Inverse-design report filename (inside output dir)")

    args = parser.parse_args()

    world_kwargs: Dict[str, Any] = {}
    if args.world_u_inlet is not None:
        world_kwargs["u_inlet"] = float(args.world_u_inlet)
    if args.world_tau is not None:
        world_kwargs["tau"] = float(args.world_tau)

    config = ExperimentConfig(
        name=args.name,
        seed=args.seed,
        world_spec=args.world,
        grid_shape=tuple(args.grid),
        world_kwargs=world_kwargs,
        max_steps=args.steps,
        batch_size=args.batch_size,
        internal_neurons=args.neurons,
        vision_dim=args.vision_dim,
        device=args.device,
        scientist_signal=args.scientist_signal,
        scientist_warmup_steps=max(0, int(args.scientist_warmup_steps)),
        scientist_diag_trust_floor=float(np.clip(args.scientist_diag_trust_floor, 0.0, 1.0)),
        scientist_diag_trust_floor_start_step=(
            None
            if args.scientist_diag_trust_floor_start_step is None
            else max(1, int(args.scientist_diag_trust_floor_start_step))
        ),
        scientist_signal_trust_scale_mode=str(args.scientist_signal_trust_scale_mode).strip().lower(),
        ablate_scientist=args.ablate_scientist,
        ablate_symplectic=args.ablate_symplectic,
        ablate_trust_gate=args.ablate_trust_gate,
        safety_enable=not args.disable_safety,
        safety_min_theory_trust=args.safety_min_theory_trust,
        safety_max_stress=args.safety_max_stress,
        safety_shift_z_threshold=args.safety_shift_z_threshold,
        safety_shift_warmup=args.safety_shift_warmup,
        safety_density_min=args.safety_density_min,
        safety_density_max=args.safety_density_max,
        safety_speed_max=args.safety_speed_max,
        safety_cautious_hard_rate=float(np.clip(args.safety_cautious_hard_rate, 0.01, 0.99)),
        safety_safe_hold_hard_rate=float(np.clip(args.safety_safe_hold_hard_rate, 0.01, 0.99)),
        safety_cautious_action_scale=float(np.clip(args.safety_cautious_action_scale, 0.05, 1.0)),
        safety_recovery_hysteresis=float(np.clip(args.safety_recovery_hysteresis, 0.10, 0.99)),
        safety_low_trust_hard_ratio=float(np.clip(args.safety_low_trust_hard_ratio, 0.05, 1.0)),
        safety_low_trust_hard_warmup=max(0, int(args.safety_low_trust_hard_warmup)),
        output_dir=args.output_dir,
    )

    experiment = ATOMExperiment(config)
    if args.inverse_design:
        experiment.run_inverse_design(
            backend=args.inverse_backend,
            iterations=args.inverse_iterations,
            population=args.inverse_population,
            top_k=args.inverse_top_k,
            rollout_steps=args.inverse_rollout_steps,
            objective_spec_path=args.inverse_objective,
            parameter_space_path=args.inverse_param_space,
            report_name=args.inverse_report,
        )
    else:
        experiment.run()


if __name__ == "__main__":
    main()
