"""
ATOM SIM: TRAINING LOOP (The Orchestrator)
------------------------------------------
Connects Body (World), Eyes, Brain, and Mind.
Runs the Wake/Sleep Cycle to evolve the Neuro-Symbolic Agent.

Architecture:
- Phase 1: Wake (Interaction & Fast Learning)
- Phase 2: Sleep (Symbolic Distillation & Memory Replay)
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
import torch
import torch.optim as optim

# Force non-interactive backend for stability
matplotlib.use('Agg')

from atom.config import get_config
from atom.core.brain import create_brain_from_config
from atom.core.world import FluidWorld, CylinderWorld
from atom.exceptions import RuntimeError as AtomRuntimeError, TrainingError
from atom.logging import get_logger
from atom.mind.memory import create_memory_from_config
from atom.mind.scientist import AtomScientist, create_scientist_from_config
from atom.mind.scientist_v2 import StructuralScientist
from atom.resources import get_memory_manager
import jax.dlpack
from torch.utils.dlpack import from_dlpack
from atom.core.distributions import TanhNormal
from atom.platform.contracts import ControlEnvelope
from atom.platform.safety import RuntimeSafetySupervisor

logger = get_logger("training")


def _to_scalar_reward(x: Any) -> float:
    """
    Robust conversion of reward to Python float.

    JAX often returns:
      - scalar ()
      - shape (1,)
      - shape (1,1)
    float(jnp.array([[...]])) will throw. This won't.

    If you ever run vectorized envs and reward becomes (B,1) or (B,),
    this will default to mean unless you explicitly force strict mode.

    Control via env:
      ATOM_REWARD_REDUCE = {"strict", "mean", "sum", "first"}
    """
    if isinstance(x, (float, int)):
        return float(x)

    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)

    if arr.size == 1:
        return float(arr.reshape(()))

    mode = str(os.getenv("ATOM_REWARD_REDUCE", "strict")).lower()
    if mode == "mean":
        return float(arr.mean())
    if mode == "sum":
        return float(arr.sum())
    if mode == "first":
        return float(arr.reshape(-1)[0])

    raise TrainingError(f"Reward must be scalar/size-1. Got shape={arr.shape} dtype={arr.dtype}")


class AtomOrchestrator:
    """
    Main training orchestrator for the ATOM system.

    Coordinates the interaction between perception, reasoning, memory, and physics
    to train a neuro-symbolic agent capable of learning physical laws.
    """

    def __init__(self, config=None, use_symplectic: Optional[bool] = None, vision_mode: Optional[str] = None):
        """
        Initialize the ATOM orchestrator.

        Args:
            config: Optional configuration override
            use_symplectic: Ablation toggle for symplectic manifold
            vision_mode: Ablation toggle for vision (fno vs linear)
        """
        if config is None:
            config = get_config()

        logger.info("Initializing ATOM Orchestrator")

        # Runtime knobs (do NOT change imports / API; only runtime behavior)
        # DLPack is zero-copy. If you store tensors derived from it into replay,
        # you can pin JAX buffers and blow up RAM. Default is safe.
        self._dlpack_clone = bool(int(os.getenv("ATOM_DLPACK_CLONE", "1")))
        # Visual recon loss is expensive; compute only every N timesteps in sequence unroll.
        self._visual_recon_stride = int(os.getenv("ATOM_VISUAL_RECON_STRIDE", "8"))
        if self._visual_recon_stride < 1:
            self._visual_recon_stride = 1
        self._diagnostics_enabled = bool(int(os.getenv("ATOM_DIAGNOSTIC_ENABLE", "1")))
        self._diagnostic_interval = max(1, int(os.getenv("ATOM_DIAGNOSTIC_INTERVAL", "8")))
        self._diagnostic_map_max_side = max(
            24, int(os.getenv("ATOM_DIAGNOSTIC_MAP_MAX_SIDE", "96"))
        )
        eyes_target = str(os.getenv("ATOM_EYES_SALIENCY_TARGET", "energy")).strip().lower()
        self._eyes_saliency_target: str = eyes_target
        self._eyes_saliency_latent_index: Optional[int] = None
        if eyes_target != "energy":
            try:
                latent_idx = int(eyes_target)
                if latent_idx >= 0:
                    self._eyes_saliency_latent_index = latent_idx
            except ValueError:
                logger.warning(
                    "Invalid ATOM_EYES_SALIENCY_TARGET=%s, defaulting to embedding-energy saliency.",
                    eyes_target,
                )
                self._eyes_saliency_target = "energy"
        self._theory_feature_labels = (
            ["prediction"]
            + [f"jacobian_latent_{i}" for i in range(8)]
            + ["verified_trust"]
        )
        self._last_runtime_diagnostics: Optional[Dict[str, Any]] = None

        # 1. THE BODY (JAX Physics)
        world_name = getattr(config.physics, "world_type", "fluid")

        if world_name == "cylinder":
            world_cls = CylinderWorld
            world_args = {}
        elif world_name == "custom":
            from atom.core.world import MeshWorld
            world_cls = MeshWorld
            # Ensure geometry path is in config
            geo_path = getattr(config.physics, "geometry_path", None)
            if not geo_path:
                raise ValueError("World type is 'custom' but 'geometry_path' is missing in config.physics")
            world_args = {"stl_path": geo_path}
        else:
            world_cls = FluidWorld
            world_args = {}

        self.world = world_cls(
            nx=config.physics.grid_shape[0],
            ny=config.physics.grid_shape[1],
            nz=config.physics.grid_shape[2],
            **world_args
        )

        # 2. THE BRAIN (Controller + Eyes + Skeleton)
        # Scientist V2 (Structural Coupling) requires 10D signal
        logger.info("Scientist V2 Active: Scaling Brain Theory Dimension to 10D")
        config.brain.theory_dim = 10
        self.brain = create_brain_from_config(
            config=config,
            use_symplectic=use_symplectic,
            vision_mode=vision_mode
        ).to(config.get_device())

        # 3. THE MIND (Memory + Scientist)
        self.memory = create_memory_from_config(config)

        # Scientist feature schema:
        # [action, mean_speed, turbulence, rho_mean, rho_std, latent_0..latent_7]
        self.scientist_vars = [
            "action",
            "mean_speed",
            "turbulence",
            "rho_mean",
            "rho_std",
        ] + [f"latent_{i}" for i in range(8)]
        base_scientist = AtomScientist(variable_names=self.scientist_vars)
        
        # Upgrade to Scientist V2 (Structural Coupling)
        self.scientist = StructuralScientist(
            scientist=base_scientist,
            variable_names=self.scientist_vars
        )

        # 4. OPTIMIZERS
        # We optimize Eyes and Brain together (End-to-End)
        self.optimizer = optim.Adam([
            {'params': self.brain.eyes.parameters(), 'lr': config.brain.learning_rate_eyes},
            {'params': self.brain.liquid.parameters(), 'lr': config.brain.learning_rate_actor},
            {'params': list(self.brain.actor_mu.parameters()) + [self.brain.actor_log_std],
             'lr': config.brain.learning_rate_actor},
            {'params': self.brain.critic.parameters(), 'lr': config.brain.learning_rate_critic}
        ])

        # Training state
        self.training_state = TrainingState()

        # Logging
        self.history = {"reward": [], "stress": [], "theory_score": []}
        self.config = config

        # Runtime safety supervisor (Phase 3 runtime assurance)
        self.safety_enabled = bool(int(os.getenv("ATOM_SAFETY_ENABLE", "1")))
        self.control_envelope = self._build_control_envelope()
        self.safety_supervisor = (
            RuntimeSafetySupervisor(
                envelope=self.control_envelope,
                action_dim=int(getattr(self.config.brain, "action_dim", 1)),
                min_theory_trust=float(os.getenv("ATOM_SAFETY_MIN_TRUST", "0.08")),
                max_stress=float(os.getenv("ATOM_SAFETY_MAX_STRESS", "8.0")),
                shift_z_threshold=float(os.getenv("ATOM_SAFETY_SHIFT_Z", "8.0")),
                shift_warmup=int(os.getenv("ATOM_SAFETY_SHIFT_WARMUP", "24")),
                degrade_window=int(os.getenv("ATOM_SAFETY_DEGRADE_WINDOW", "64")),
                cautious_hard_rate=float(os.getenv("ATOM_SAFETY_CAUTIOUS_HARD_RATE", "0.15")),
                safe_hold_hard_rate=float(os.getenv("ATOM_SAFETY_SAFE_HOLD_HARD_RATE", "0.35")),
                cautious_action_scale=float(os.getenv("ATOM_SAFETY_CAUTIOUS_ACTION_SCALE", "0.5")),
                recovery_hysteresis=float(os.getenv("ATOM_SAFETY_RECOVERY_HYSTERESIS", "0.8")),
            )
            if self.safety_enabled
            else None
        )
        self.history["safety_intervention"] = []
        self.history["safety_fallback"] = []

        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

        logger.info("ATOM Orchestrator initialized successfully")

    def _safe_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        """Best-effort JSON write that never crashes the training loop."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        except PermissionError as exc:
            logger.warning(f"Skipping write to {path} due to permission error: {exc}")
        except Exception as exc:
            logger.warning(f"Skipping write to {path} due to IO error: {exc}")

    def _safe_export_world(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.world.export_to_web(str(path))
        except PermissionError as exc:
            logger.warning(f"Skipping world export to {path} due to permission error: {exc}")
        except Exception as exc:
            logger.warning(f"Skipping world export to {path} due to error: {exc}")

    def _safe_render_world(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.world.render(str(path))
        except PermissionError as exc:
            logger.warning(f"Skipping render to {path} due to permission error: {exc}")
        except Exception as exc:
            logger.warning(f"Skipping render to {path} due to error: {exc}")

    def _detach_recurrent_state(self, state: Optional[Any]) -> Optional[Any]:
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            return state.detach()
        if isinstance(state, tuple):
            return tuple(self._detach_recurrent_state(item) for item in state)
        if isinstance(state, list):
            return [self._detach_recurrent_state(item) for item in state]
        return state

    def _serialize_map_2d(
        self,
        map_2d: torch.Tensor,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        arr = map_2d.detach().cpu().float().numpy()
        if arr.ndim != 2:
            arr = np.asarray(arr, dtype=np.float32).reshape(-1, 1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        raw_min = float(np.min(arr)) if arr.size else 0.0
        raw_max = float(np.max(arr)) if arr.size else 0.0

        values = arr
        if normalize and arr.size:
            q_lo = float(np.percentile(arr, 1.0))
            q_hi = float(np.percentile(arr, 99.0))
            if q_hi > q_lo + 1e-8:
                values = np.clip(arr, q_lo, q_hi)
                values = (values - q_lo) / (q_hi - q_lo)
            else:
                values = np.zeros_like(arr, dtype=np.float32)

        h, w = values.shape
        max_side = max(24, int(self._diagnostic_map_max_side))
        stride = max(1, int(np.ceil(max(h, w) / float(max_side))))
        values_ds = values[::stride, ::stride]

        return {
            "map_xy": values_ds.tolist(),
            "shape": [int(values_ds.shape[0]), int(values_ds.shape[1])],
            "stride": int(stride),
            "normalized": bool(normalize),
            "raw_min": raw_min,
            "raw_max": raw_max,
        }

    def _compute_live_view_maps(self, obs_t: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            obs0 = obs_t[0]
            ux = obs0[0]
            uy = obs0[1]
            uz = obs0[2]
            rho = obs0[3]

            speed = torch.sqrt(torch.clamp(ux * ux + uy * uy + uz * uz, min=0.0))
            speed_xy = speed.mean(dim=-1)
            rho_xy = rho.mean(dim=-1)
            ux_xy = ux.mean(dim=-1)
            uy_xy = uy.mean(dim=-1)

            try:
                dux_dx = torch.gradient(ux_xy, dim=0)[0]
                duy_dy = torch.gradient(uy_xy, dim=1)[0]
                div_xy = dux_dx + duy_dy
            except Exception:
                div_xy = torch.zeros_like(speed_xy)

        return {
            "projection": "xy_mean",
            "speed_xy": self._serialize_map_2d(speed_xy, normalize=True),
            "density_xy": self._serialize_map_2d(rho_xy, normalize=True),
            "divergence_xy": self._serialize_map_2d(div_xy, normalize=True),
        }

    def _extract_obstacle_overlay(
        self,
        info: Optional[Dict[str, Any]],
        obs_t: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(info, dict):
            return None
        mask_raw = info.get("mask")
        if mask_raw is None:
            return None

        try:
            mask_np = np.asarray(mask_raw, dtype=np.float32)
        except Exception:
            return None

        if mask_np.size == 0:
            return None

        if mask_np.ndim == 4:
            # (B, X, Y, Z) -> (X, Y)
            mask_xy = mask_np[0].mean(axis=-1)
        elif mask_np.ndim == 3:
            # (B, X, Y) or (X, Y, Z)
            if mask_np.shape[0] == 1:
                mask_xy = mask_np[0]
            else:
                mask_xy = mask_np.mean(axis=-1)
        elif mask_np.ndim == 2:
            mask_xy = mask_np
        else:
            return None

        try:
            mask_t = torch.as_tensor(mask_xy, dtype=torch.float32, device=obs_t.device)
        except Exception:
            return None
        mask_t = torch.clamp(mask_t, min=0.0, max=1.0)
        return self._serialize_map_2d(mask_t, normalize=False)

    def _compute_eyes2_saliency(self, obs_t: torch.Tensor) -> Optional[Dict[str, Any]]:
        was_training = bool(self.brain.training)
        obs_in = obs_t.detach().clone().requires_grad_(True)
        try:
            self.brain.eval()
            with torch.enable_grad():
                embed = self.brain.eyes.embed(obs_in)
                target_index = self._eyes_saliency_latent_index
                target_label = "embedding_energy"
                method = "grad_embedding_energy"
                if target_index is not None and target_index < int(embed.shape[-1]):
                    objective = torch.mean(embed[:, target_index])
                    target_label = f"latent_{target_index}"
                    method = "grad_embedding_latent"
                else:
                    objective = torch.mean(embed.pow(2))
                grad = torch.autograd.grad(
                    objective,
                    obs_in,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
            if grad is None:
                return None
            saliency_volume = grad.detach().abs().mean(dim=1)[0]
            saliency_xy = saliency_volume.mean(dim=-1)
            return {
                "method": method,
                "objective": float(objective.detach().cpu().item()),
                "projection": "xy_mean",
                "target_label": target_label,
                "target_index": (
                    int(target_index)
                    if target_index is not None and target_index < int(embed.shape[-1])
                    else None
                ),
                "map": self._serialize_map_2d(saliency_xy, normalize=True),
            }
        except Exception as exc:
            logger.debug(f"Eyes2 saliency computation skipped: {exc}")
            return None
        finally:
            if was_training:
                self.brain.train()

    def _compute_brain_saliency(
        self,
        obs_t: torch.Tensor,
        theory_t: torch.Tensor,
        last_action: torch.Tensor,
        hx: Optional[Any],
        theory_confidence: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        was_training = bool(self.brain.training)
        obs_in = obs_t.detach().clone().requires_grad_(True)
        theory_in = theory_t.detach().clone().requires_grad_(True)
        action_in = last_action.detach().clone()
        hx_in = self._detach_recurrent_state(hx)
        confidence_in = theory_confidence.detach().clone()

        try:
            self.brain.eval()
            with torch.enable_grad():
                (mu, _), _, _, _ = self.brain(
                    obs_in,
                    theory_in,
                    action_in,
                    hx_in,
                    theory_confidence=confidence_in,
                )
                objective = torch.mean(mu.pow(2))
                obs_grad, theory_grad = torch.autograd.grad(
                    objective,
                    [obs_in, theory_in],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

            if obs_grad is None:
                return None

            obs_saliency = obs_grad.detach().abs().mean(dim=1)[0]
            obs_saliency_xy = obs_saliency.mean(dim=-1)
            theory_importance: List[float] = []
            if theory_grad is not None:
                imp = theory_grad.detach().abs().mean(dim=0).cpu().float().numpy()
                imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
                max_imp = float(np.max(imp)) if imp.size else 0.0
                if max_imp > 1e-8:
                    imp = imp / max_imp
                theory_importance = [float(v) for v in imp.tolist()]

            labels = list(self._theory_feature_labels)
            if len(labels) != len(theory_importance):
                labels = [f"theory_{i}" for i in range(len(theory_importance))]

            return {
                "method": "grad_policy_energy",
                "objective": float(objective.detach().cpu().item()),
                "projection": "xy_mean",
                "map": self._serialize_map_2d(obs_saliency_xy, normalize=True),
                "theory_feature_importance": theory_importance,
                "theory_feature_labels": labels,
            }
        except Exception as exc:
            logger.debug(f"Brain saliency computation skipped: {exc}")
            return None
        finally:
            if was_training:
                self.brain.train()

    def _collect_runtime_diagnostics(
        self,
        *,
        step: int,
        obs_t: torch.Tensor,
        theory_t: torch.Tensor,
        last_action: torch.Tensor,
        hx: Optional[Any],
        theory_confidence: torch.Tensor,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        live_view = self._compute_live_view_maps(obs_t)
        obstacle_overlay = self._extract_obstacle_overlay(info, obs_t)
        if obstacle_overlay is not None:
            live_view["obstacle_xy"] = obstacle_overlay
        if not self._diagnostics_enabled:
            return {
                "enabled": False,
                "updated_step": int(step),
                "stale_steps": 0,
                "live_view": live_view,
            }

        should_refresh = (
            self._last_runtime_diagnostics is None
            or (int(step) % int(self._diagnostic_interval) == 0)
        )
        if not should_refresh and self._last_runtime_diagnostics is not None:
            cached = dict(self._last_runtime_diagnostics)
            cached["stale_steps"] = int(step) - int(cached.get("updated_step", step))
            cached["live_view"] = live_view
            return cached

        eyes_saliency = self._compute_eyes2_saliency(obs_t)
        brain_saliency = self._compute_brain_saliency(
            obs_t=obs_t,
            theory_t=theory_t,
            last_action=last_action,
            hx=hx,
            theory_confidence=theory_confidence,
        )
        snapshot = {
            "enabled": True,
            "updated_step": int(step),
            "stale_steps": 0,
            "interval": int(self._diagnostic_interval),
            "live_view": live_view,
            "eyes2_saliency": eyes_saliency,
            "brain_saliency": brain_saliency,
        }
        self._last_runtime_diagnostics = dict(snapshot)
        return snapshot

    def _tensorify(self, jax_array) -> torch.Tensor:
        """
        Convert JAX array to PyTorch Tensor using DLPack.

        Critical nuance:
        - DLPack is zero-copy. If you keep references around (replay buffer, history),
          you can pin JAX buffers and leak memory.
        - Default: clone to detach ownership from JAX backing storage.
        """
        try:
            dlpack_tensor = jax.dlpack.to_dlpack(jax_array)
            t = from_dlpack(dlpack_tensor)

            # Enforce float32 for network consistency
            if t.dtype != torch.float32:
                t = t.float()

            # Move to target device (may copy; that's fine)
            t = t.to(self.config.get_device())

            # Break JAX ownership tie if requested (default ON)
            if self._dlpack_clone:
                t = t.contiguous().clone()
            else:
                t = t.contiguous()

            return t

        except Exception:
            # Fallback if DLPack fails (incompatible dtypes/devices)
            np_array = np.array(jax_array)
            return torch.tensor(np_array, dtype=torch.float32).to(self.config.get_device())

    def _safe_clip_grads(self, parameters, max_norm: float) -> float:
        """
        MPS SAFE CLIPPER:
        Avoids 'norm ops not supported for complex' crash on Apple Silicon.

        Use view_as_real (safe) rather than .view(float32) (layout-dependent hack).
        """
        params = [p for p in parameters if p.grad is not None]
        total_norm_sq = 0.0

        for p in params:
            g = p.grad
            if g is None:
                continue

            if g.is_complex():
                # (..., 2) real/imag; safe and explicit
                g_realimag = torch.view_as_real(g)
                param_norm = g_realimag.reshape(-1).norm(2)
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

    def _build_control_envelope(self) -> ControlEnvelope:
        action_dim = int(getattr(self.config.brain, "action_dim", 1))
        action_bounds = {f"action_{i}": (-1.0, 1.0) for i in range(action_dim)}
        density_min = float(os.getenv("ATOM_SAFETY_DENSITY_MIN", "0.05"))
        density_max = float(os.getenv("ATOM_SAFETY_DENSITY_MAX", "8.0"))
        speed_max = float(os.getenv("ATOM_SAFETY_SPEED_MAX", "3.0"))
        return ControlEnvelope(
            action_bounds=action_bounds,
            state_bounds={
                "rho_min": (density_min, 100.0),
                "rho_max": (0.0, density_max),
                "mean_speed": (0.0, speed_max),
            },
            intervention_policy="runtime_safety_supervisor.v1",
            fallback_policy_id="fallback:zero_action",
        )

    def _build_safety_diagnostics(
        self,
        obs_t: torch.Tensor,
        f_vec: torch.Tensor,
        theory_conf_t: torch.Tensor,
        stress: torch.Tensor,
        info: Optional[Dict[str, Any]] = None,
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
                "feature_vector": f_vec.detach().cpu().numpy().tolist(),
            }
        if info:
            for key in ("min_density", "max_density", "rho_mean", "rho_std"):
                if key in info:
                    diagnostics[key] = float(info[key])
        return diagnostics

    def _update_brain(self) -> Dict[str, Any]:
        """
        PPO Update Step with:
        - GAE-λ Advantage Estimation
        - Clipped Surrogate Objective
        - Multiple Epochs per Batch

        Trains Eyes (Reconstruction) and Brain (Policy) simultaneously.
        """
        try:
            batch = self.memory.sample(self.config.brain.batch_size)
            if not batch:
                return {"loss": 0.0}

            device = self.config.get_device()

            # 1. UNPACK SEQUENCE
            obs = torch.as_tensor(batch["obs"], device=device, dtype=torch.float32)
            actions = torch.as_tensor(batch["action"], device=device, dtype=torch.float32)
            rewards = torch.as_tensor(batch["reward"], device=device, dtype=torch.float32)
            dones = torch.as_tensor(batch["done"], device=device, dtype=torch.float32)
            hx_batch = torch.as_tensor(batch["hx"], device=device, dtype=torch.float32)
            old_log_probs = torch.as_tensor(batch["old_log_prob"], device=device, dtype=torch.float32)
            old_values = torch.as_tensor(batch["old_value"], device=device, dtype=torch.float32)

            # CACHED THEORY & TRUST (Crucial for consistent PPO)
            batch_theory = torch.as_tensor(batch["theory"], device=device, dtype=torch.float32)
            batch_trust = torch.as_tensor(batch["trust"], device=device, dtype=torch.float32)

            if actions.ndim != 3:
                raise TrainingError(f"Expected actions shape [B,T,D], got {tuple(actions.shape)}")

            B, T, D = actions.shape

            # prev_actions[t] = actions[t-1], with zeros at t=0
            prev_actions = torch.cat([torch.zeros(B, 1, D, device=device), actions[:, :-1, :]], dim=1)

            # 2. COMPUTE GAE-λ ADVANTAGES (Before update loop)
            next_obs = torch.as_tensor(batch["next_obs"], device=device, dtype=torch.float32)
            next_hx = torch.as_tensor(batch["next_hx"], device=device, dtype=torch.float32)

            # Bootstrap must be consistent with conditioning used during rollout:
            # use cached theory/trust from last step, NOT zeros.
            last_theory = batch_theory[:, -1]
            last_trust = batch_trust[:, -1]
            if last_theory.ndim == 1:
                last_theory = last_theory.unsqueeze(1)
            if last_trust.ndim == 1:
                last_trust = last_trust.unsqueeze(1)

            with torch.no_grad():
                last_action = actions[:, -1]  # proprioception at end of sequence
                _, last_value, _, _ = self.brain(
                    next_obs,
                    last_theory,
                    last_action,
                    next_hx,
                    theory_confidence=last_trust,
                )

                advantages = torch.zeros_like(rewards)
                returns = torch.zeros_like(rewards)
                gae = 0.0

                for t in reversed(range(T)):
                    next_non_terminal = 1.0 - dones[:, t]
                    next_value = last_value if (t == T - 1) else old_values[:, t + 1]
                    delta = rewards[:, t] + self.config.brain.gamma * next_value * next_non_terminal - old_values[:, t]
                    gae = delta + self.config.brain.gamma * self.config.brain.gae_lambda * next_non_terminal * gae
                    advantages[:, t] = gae
                    returns[:, t] = gae + old_values[:, t]

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 3. PPO UPDATE LOOP (Multiple Epochs)
            total_loss = 0.0
            clip_eps = self.config.brain.clip_epsilon

            # Use config if present; otherwise defaults preserve prior behavior.
            entropy_coef = float(getattr(self.config.brain, "entropy_coef", 0.01))
            critic_coef = float(getattr(self.config.brain, "critic_coef", 0.5))
            stress_coef = float(getattr(self.config.brain, "stress_coef", 0.1))
            visual_coef = float(getattr(self.config.brain, "visual_coef", 1.0))
            max_grad_norm = float(getattr(self.config.brain, "max_grad_norm", 0.5))

            recon_stride = int(self._visual_recon_stride)
            if recon_stride < 1:
                recon_stride = 1

            for epoch in range(self.config.brain.ppo_epochs):
                mus, stds, values, stresses, rec_losses = [], [], [], [], []
                curr_hx = hx_batch

                for t in range(T):
                    o_t = obs[:, t]
                    a_prev_t = prev_actions[:, t]

                    # use cached theory/trust (no racy Scientist calls during PPO)
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

                    # Reconstruction Loss (Eyes) - throttle for laptop survivability
                    if self.brain.vision_mode == "fno" and (t % recon_stride == 0):
                        recon_field = self.brain.eyes(o_t)
                        rec_loss = torch.nn.functional.mse_loss(recon_field, o_t[:, :3])
                    else:
                        rec_loss = torch.tensor(0.0, device=device)

                    rec_losses.append(rec_loss)
                    curr_hx = new_hx

                mus = torch.stack(mus, dim=1)
                stds = torch.stack(stds, dim=1)

                # Numerical stability
                if not torch.isfinite(mus).all() or not torch.isfinite(stds).all():
                    mus = torch.nan_to_num(mus, nan=0.0, posinf=1.0, neginf=-1.0)
                    stds = torch.nan_to_num(stds, nan=1.0, posinf=1.0, neginf=0.1)

                new_values = torch.stack(values, dim=1)
                stress_mean = torch.stack(stresses).mean()
                visual_loss = torch.stack(rec_losses).mean()

                dist = TanhNormal(mus, stds)
                new_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - entropy_coef * entropy

                value_clipped = old_values + torch.clamp(new_values - old_values, -clip_eps, clip_eps)
                critic_loss1 = (returns - new_values).pow(2)
                critic_loss2 = (returns - value_clipped).pow(2)
                critic_loss = 0.5 * torch.max(critic_loss1, critic_loss2).mean()

                loss = actor_loss + critic_coef * critic_loss + stress_coef * stress_mean + visual_coef * visual_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._safe_clip_grads(self.brain.parameters(), max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())

            # Apply Delayed Hebbian Growth (Safe Skeleton)
            self.brain.apply_hebbian_growth()

            return {"loss": total_loss / float(self.config.brain.ppo_epochs)}

        except Exception as e:
            logger.error(f"Brain update failed: {e}")
            raise TrainingError(f"Brain update failed: {e}") from e

    def run_dream_mode(self) -> None:
        """Run symbolic distillation (dream mode)."""
        logger.info("Starting symbolic distillation (dream mode)")

        if self.memory.size < 100:
            logger.warning("Not enough memories to dream. Run training first.")
            return

        # Extract Data for Scientist
        batch_size = min(len(self.memory.obs_buf), 5000)
        batch = self.memory.sample(batch_size)

        if batch is None:
            return

        actions = batch['action']  # (B, T, 1)
        obs = batch['obs']         # (B, T, 4, X, Y, Z)
        rewards = batch['reward']  # (B, T, 1)

        device = self.config.get_device()
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)

        # Flatten Time dimension
        actions_flat = actions.reshape(-1, 1)
        rewards_flat = rewards.reshape(-1)
        obs_flat = obs.reshape(-1, 4, *self.config.physics.grid_shape)

        # COMPUTE PHYSICS FEATURES
        with torch.no_grad():
            # Mean Velocity Magnitude
            u = obs_flat[:, 0]  # Ux component
            v = obs_flat[:, 1]  # Uy component
            w = obs_flat[:, 2]  # Uz component
            speed = torch.sqrt(u**2 + v**2 + w**2)
            mean_speed = torch.mean(speed, axis=(1, 2, 3)).reshape(-1, 1)

            # Turbulence (Standard Deviation of speed spatial distribution)
            turb_int = torch.std(speed, axis=(1, 2, 3)).reshape(-1, 1)

            # Density observables (compressibility and shock/instability cues)
            rho = obs_flat[:, 3]
            rho_mean = torch.mean(rho, axis=(1, 2, 3)).reshape(-1, 1)
            rho_std = torch.std(rho, axis=(1, 2, 3)).reshape(-1, 1)

        # Get visual embeddings
        with torch.no_grad():
            visual_embed = self.brain.eyes.embed(obs_flat)  # (N, 256)
            embed_reduced = visual_embed[:, :8]  # Take first 8 dimensions

        # Construct Context Vector X
        # X = [Action, MeanSpeed, Turbulence, RhoMean, RhoStd, Embeddings...]
        X_features = torch.cat([
            actions_flat,
            mean_speed,
            turb_int,
            rho_mean,
            rho_std,
            embed_reduced
        ], dim=1).cpu().numpy()

        y_target = rewards_flat.detach().cpu().numpy()

        # Wake the Scientist (V3: Now includes Calibration)
        self.scientist.fit_offline(X_features, y_target)


    def run(self) -> None:
        """Main training loop."""
        logger.info("ATOM WAKE CYCLE INITIATED")

        # Initialize environment
        obs, _ = self.world.reset()
        obs_t = self._tensorify(obs)

        # Brain State (LTC Hidden)
        hx = None
        act_dim = self.config.brain.action_dim
        last_action = torch.zeros(1, act_dim).to(self.config.get_device())

        total_steps = 0
        start_time = time.time()

        try:
            while total_steps < self.config.training.max_steps:
                # --- 1. SENSE & THINK ---
                # A. Scientist Intuition (System 1)
                with torch.no_grad():
                    vel = obs_t[:, :3]
                    speed = torch.norm(vel, dim=1)
                    mean_speed = speed.mean().item()
                    turb_int = speed.std().item()
                    rho = obs_t[:, 3]
                    rho_mean = rho.mean().item()
                    rho_std = rho.std().item()
                    emb = self.brain.eyes.embed(obs_t)[0, :8]

                    # [Action, Speed, Turb, RhoMean, RhoStd, Emb...]
                    f_vec = torch.cat([
                        last_action.flatten(),
                        torch.tensor(
                            [mean_speed, turb_int, rho_mean, rho_std],
                            device=obs_t.device,
                        ),
                        emb
                    ])

                # Get Prediction AND Structural Signal (10D Jacobian + verified trust)
                sci_out = self.scientist.get_signal(
                    features=f_vec,
                    reward=None,  # Not known yet
                    action=None,  # Taken below
                    device=self.config.get_device()
                )
                theory_t = sci_out["tensor"]  # (1, 10)
                theory_conf_t = torch.tensor([[sci_out["trust"]]], dtype=torch.float32, device=self.config.get_device())
                theory_val_log = sci_out["prediction"]
                theory_packet = sci_out.get("packet")
                hypothesis_record = sci_out.get("hypothesis_record")
                experiment_plan = sci_out.get("experiment_plan")
                governance_decision = sci_out.get("governance")
                verifier_metrics = sci_out.get("verifier_metrics")
                baseline_context = {
                    "ready": bool(sci_out.get("baseline_ready", False)),
                    "slope": float(sci_out.get("baseline_slope", 0.0)),
                    "r2": float(sci_out.get("baseline_r2", 0.0)),
                    "discovery_target_mode": str(
                        sci_out.get("discovery_target_mode", "hybrid")
                    ),
                    "discovery_hybrid_alpha": float(
                        sci_out.get("discovery_hybrid_alpha", 0.0)
                    ),
                    "discovery_target_value": float(
                        sci_out.get("discovery_target_value", 0.0)
                    ),
                }

                # B. Brain Forward
                (mu, std), value, hx_new, stress = self.brain(
                    obs_t, theory_t, last_action, hx, theory_confidence=theory_conf_t
                )

                # C. Sample Action (Stochastic) with TanhNormal
                dist = TanhNormal(mu, std)
                action_phys, raw_action = dist.sample()
                action_phys = torch.clamp(action_phys, -1.0, 1.0)

                # Compute log_prob for PPO (rollout must match update path)
                log_prob = dist.log_prob(action_phys, pre_tanh_value=raw_action).sum(-1, keepdim=True)

                # --- 2. ACT & FEEL ---
                safety_decision = None
                action_np = action_phys.detach().cpu().numpy().reshape(-1)
                if self.safety_supervisor is not None:
                    safety_diag = self._build_safety_diagnostics(
                        obs_t=obs_t,
                        f_vec=f_vec,
                        theory_conf_t=theory_conf_t,
                        stress=stress,
                    )
                    safety_decision = self.safety_supervisor.review(action_np, safety_diag)
                    action_np = safety_decision.action.astype(np.float32).reshape(-1)
                    action_phys = torch.as_tensor(
                        action_np, dtype=torch.float32, device=self.config.get_device()
                    ).view(1, -1)
                    log_prob = dist.log_prob(action_phys).sum(-1, keepdim=True)

                next_obs, reward_jax, done, info = self.world.step(action_np)
                info = dict(info or {})
                if safety_decision is not None:
                    info["safety"] = safety_decision.to_dict()

                next_obs_t = self._tensorify(next_obs)
                reward = _to_scalar_reward(reward_jax)

                # --- 3. MEMORY & SCIENCE ---
                self.memory.push(
                    obs_t, action_phys, reward, done, hx_new,
                    log_prob=log_prob, value=value,
                    theory=theory_t, trust=theory_conf_t
                )

                # Scientist observation (residual-consistent V1+V2 path)
                if hasattr(self.scientist, "observe_and_verify"):
                    self.scientist.observe_and_verify(
                        f_vec,
                        reward,
                        theory_val_log,
                        float(action_phys.flatten()[0].item()),
                    )
                else:
                    self.scientist.observe(f_vec, reward)

                # --- 4. LEARN (Online Update) ---
                if self.memory.size > self.memory.seq_len + self.config.brain.batch_size:
                    _ = self._update_brain()

                # --- 5. SLEEP CHECK (System 2 Update) ---
                if total_steps % self.config.training.sleep_interval == 0:
                    new_law = self.scientist.ponder()
                    if new_law:
                        logger.info(f"Brain received new Insight: {new_law}")

                # --- 6. LOGGING ---
                self.history["reward"].append(reward)
                self.history["stress"].append(stress.mean().item())

                self.history["theory_score"].append(float(theory_val_log))
                if safety_decision is None:
                    self.history["safety_intervention"].append(0.0)
                    self.history["safety_fallback"].append(0.0)
                else:
                    self.history["safety_intervention"].append(1.0 if safety_decision.intervened else 0.0)
                    self.history["safety_fallback"].append(1.0 if safety_decision.fallback_used else 0.0)

                diagnostics = self._collect_runtime_diagnostics(
                    step=total_steps,
                    obs_t=obs_t,
                    theory_t=theory_t,
                    last_action=last_action,
                    hx=hx,
                    theory_confidence=theory_conf_t,
                    info=info,
                )

                telem_data = {
                    "step": total_steps,
                    "reward": float(reward),
                    "stress": float(stress.mean().item()),
                    "theory": float(theory_val_log),
                    "grid": f"{self.world.nx}x{self.world.ny}x{self.world.nz}",
                    "safety": (
                        safety_decision.to_dict()
                        if safety_decision is not None
                        else None
                    ),
                    "safety_summary": (
                        self.safety_supervisor.snapshot()
                        if self.safety_supervisor is not None
                        else {
                            "steps": len(self.history["reward"]),
                            "interventions": 0,
                            "fallback_uses": 0,
                            "mode": "normal",
                            "hard_event_rate": 0.0,
                            "degrade_window": 0,
                            "mode_transitions": {},
                            "reason_histogram": {},
                            "envelope": self.control_envelope.to_dict(),
                            "policy": None,
                        }
                    ),
                    "theory_packet": theory_packet.to_dict() if theory_packet is not None else None,
                    "hypothesis_record": (
                        hypothesis_record.to_dict()
                        if hypothesis_record is not None and hasattr(hypothesis_record, "to_dict")
                        else None
                    ),
                    "governance_decision": governance_decision,
                    "verifier_metrics": verifier_metrics,
                    "baseline_context": baseline_context,
                    "experiment_plan": experiment_plan,
                    "diagnostics": diagnostics,
                }
                self._safe_write_json(self.logs_dir / "telemetry.json", telem_data)

                # Volumetric export
                if total_steps % 10 == 0:
                    self._safe_export_world(self.logs_dir / "volumetric_data.json")

                if total_steps % self.config.training.render_interval == 0:
                    avg_r = np.mean(self.history["reward"][-50:]) if len(self.history["reward"]) > 0 else 0
                    safety_i50 = (
                        np.mean(self.history["safety_intervention"][-50:])
                        if len(self.history["safety_intervention"]) > 0
                        else 0.0
                    )
                    safety_f50 = (
                        np.mean(self.history["safety_fallback"][-50:])
                        if len(self.history["safety_fallback"]) > 0
                        else 0.0
                    )
                    logger.info(
                        f"Step {total_steps:04d} | Rew: {avg_r:.3f} | Stress: {stress.mean().item():.3f} "
                        f"| Theory: {theory_val_log:.2f} | SafeI: {safety_i50:.2f} | Fallback: {safety_f50:.2f}"
                    )
                    self._safe_render_world(self.logs_dir / f"atom_step_{total_steps:04d}.png")

                # Advance
                obs_t = next_obs_t
                hx = hx_new.detach()
                last_action = action_phys
                total_steps += 1

        except KeyboardInterrupt:
            logger.info("Manual shutdown requested")
        except Exception as e:
            logger.error(f"Training loop crashed: {e}")
            raise AtomRuntimeError(f"Training loop failed: {e}") from e
        finally:
            # Cleanup
            self.scientist.shutdown()

            # Memory cleanup
            memory_manager = get_memory_manager()
            memory_manager.cleanup_memory()

            logger.info("ATOM training completed")


class TrainingState:
    """Tracks training progress and state."""

    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.best_metric = float('-inf')
        self.metrics_history = []
        self.lr_schedulers_state = {}
        self.optimizers_state = {}


def create_orchestrator_from_config() -> AtomOrchestrator:
    """Create an AtomOrchestrator instance from the global configuration."""
    config = get_config()
    return AtomOrchestrator(config)


if __name__ == "__main__":
    # Create and run orchestrator
    orchestrator = create_orchestrator_from_config()
    orchestrator.run()
