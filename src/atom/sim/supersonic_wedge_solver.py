"""Deterministic D2Q25 solver for supersonic wedge-flow control experiments.

This module provides a production-oriented supersonic-like 2D solver with:
- D2Q25 lattice (tensor-product construction)
- wedge obstacle + localized jet actuation
- deterministic reset/step/replay behavior
- shock-strength reward signal for control optimization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except Exception:  # pragma: no cover - runtime dependency guard
    HAS_JAX = False
    jax = None
    jnp = None


@dataclass
class SupersonicWedgeSolverConfig:
    """Configuration for the supersonic wedge D2Q25 solver."""

    nx: int = 192
    ny: int = 96
    tau: float = 0.80
    inflow_velocity: float = 0.16
    noise_amp: float = 0.0
    warmup_steps: int = 20
    max_steps: int = 1000
    wedge_start_x: int = 36
    wedge_length: int = 72
    wedge_half_angle_deg: float = 14.0
    jet_gain: float = 0.012
    jet_radius: float = 4.0
    reward_control_penalty: float = 0.05
    instability_density_cap: float = 8.0
    seed: int = 42


def _build_d2q25_lattice() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return D2Q25 velocity set, weights, and opposite-index map."""
    c1 = np.array([0, 1, -1, 2, -2], dtype=np.int32)
    w0 = 9.0 / 16.0
    w1 = 5.0 / 24.0
    w2 = 1.0 / 96.0
    w1d = np.array([w0, w1, w1, w2, w2], dtype=np.float64)

    cx: list[int] = []
    cy: list[int] = []
    w: list[float] = []
    for i in range(5):
        for j in range(5):
            cx.append(int(c1[i]))
            cy.append(int(c1[j]))
            w.append(float(w1d[i] * w1d[j]))

    cx_arr = np.asarray(cx, dtype=np.int32)
    cy_arr = np.asarray(cy, dtype=np.int32)
    w_arr = np.asarray(w, dtype=np.float64)

    inv_idx = []
    for i in range(cx_arr.shape[0]):
        matches = np.where((cx_arr == -cx_arr[i]) & (cy_arr == -cy_arr[i]))[0]
        inv_idx.append(int(matches[0]))

    return cx_arr, cy_arr, w_arr, np.asarray(inv_idx, dtype=np.int32)


def _equilibrium(
    rho: "jnp.ndarray",
    ux: "jnp.ndarray",
    uy: "jnp.ndarray",
    cx: "jnp.ndarray",
    cy: "jnp.ndarray",
    w: "jnp.ndarray",
) -> "jnp.ndarray":
    """Third-order equilibrium used for the hot-lattice D2Q25 model."""
    cu = cx * ux + cy * uy
    u_sq = ux**2 + uy**2
    t1 = 2.0 * cu
    t2 = 2.0 * cu**2 - u_sq
    t3 = (4.0 / 3.0) * cu**3 - 2.0 * cu * u_sq
    return rho * w * (1.0 + t1 + 0.5 * t2 + (1.0 / 6.0) * t3)


class SupersonicWedgeD2Q25Solver:
    """Deterministic supersonic-like wedge solver with control actuation."""

    def __init__(self, config: Optional[SupersonicWedgeSolverConfig] = None):
        if not HAS_JAX:  # pragma: no cover - dependency-guard path
            raise RuntimeError("JAX is required for SupersonicWedgeD2Q25Solver")

        self.config = config or SupersonicWedgeSolverConfig()
        if self.config.nx < 24 or self.config.ny < 16:
            raise ValueError("Supersonic wedge solver requires at least nx>=24, ny>=16")
        if self.config.tau <= 0.5:
            raise ValueError("tau must be > 0.5 for BGK stability")

        jax.config.update("jax_enable_x64", True)

        self._cx_np, self._cy_np, self._w_np, self._inv_idx_np = _build_d2q25_lattice()
        self._q = int(self._cx_np.shape[0])

        self._cx = jnp.asarray(self._cx_np, dtype=jnp.float64).reshape(self._q, 1, 1)
        self._cy = jnp.asarray(self._cy_np, dtype=jnp.float64).reshape(self._q, 1, 1)
        self._w = jnp.asarray(self._w_np, dtype=jnp.float64).reshape(self._q, 1, 1)
        self._inv_idx = jnp.asarray(self._inv_idx_np, dtype=jnp.int32)

        self._obstacle_mask_np = self._build_wedge_mask()
        self._obstacle_mask = jnp.asarray(self._obstacle_mask_np)

        self._jet_mask_np = self._build_jet_mask()
        self._jet_mask = jnp.asarray(self._jet_mask_np, dtype=jnp.float64)

        self._inlet_profile_np = self._build_inlet_profile()
        self._inlet_profile = jnp.asarray(self._inlet_profile_np, dtype=jnp.float64)

        self._step_fn = self._build_step_function()

        self.step_count = 0
        self.baseline_shock: Optional[float] = None
        self.last_shock = 0.0
        self._key = jax.random.PRNGKey(int(self.config.seed))
        self._f_pop = None
        self._rho = None
        self._ux = None
        self._uy = None
        self.reset()

    def _build_wedge_mask(self) -> np.ndarray:
        y_idx, x_idx = np.meshgrid(
            np.arange(self.config.ny, dtype=np.float64),
            np.arange(self.config.nx, dtype=np.float64),
            indexing="ij",
        )
        center_y = float(self.config.ny // 2)
        dx = x_idx - float(self.config.wedge_start_x)
        slope = np.tan(np.deg2rad(self.config.wedge_half_angle_deg))

        return (
            (dx >= 0.0)
            & (dx < float(self.config.wedge_length))
            & (np.abs(y_idx - center_y) <= (slope * dx))
        )

    def _build_jet_mask(self) -> np.ndarray:
        y_idx, x_idx = np.meshgrid(
            np.arange(self.config.ny, dtype=np.float64),
            np.arange(self.config.nx, dtype=np.float64),
            indexing="ij",
        )
        jet_x = max(2.0, float(self.config.wedge_start_x) - 6.0)
        jet_y = float(self.config.ny // 2)
        sigma2 = max(1.0, float(self.config.jet_radius) ** 2)
        r2 = (x_idx - jet_x) ** 2 + (y_idx - jet_y) ** 2
        mask = np.exp(-0.5 * r2 / sigma2)
        mask /= (mask.max() + 1e-12)
        return mask.astype(np.float64)

    def _build_inlet_profile(self) -> np.ndarray:
        y = np.linspace(-1.0, 1.0, self.config.ny, dtype=np.float64)
        profile = self.config.inflow_velocity * (1.0 - 0.15 * y**2)
        return np.clip(profile, 0.0, None)

    def _build_step_function(self):
        tau = float(self.config.tau)
        omega = 1.0 / tau
        noise_amp = float(self.config.noise_amp)
        cy_int = tuple(int(v) for v in self._cy_np.tolist())
        cx_int = tuple(int(v) for v in self._cx_np.tolist())

        @jax.jit
        def _advance(f_pop: "jnp.ndarray", key: "jnp.ndarray", jet_scalar: "jnp.ndarray"):
            rho = jnp.sum(f_pop, axis=0)
            rho = jnp.clip(rho, 1e-8, 1e6)
            ux = jnp.sum(f_pop * self._cx, axis=0) / rho
            uy = jnp.sum(f_pop * self._cy, axis=0) / rho

            fx = jet_scalar * self._jet_mask
            fy = jnp.zeros_like(fx)

            ux_eq = ux + 0.5 * fx / rho
            uy_eq = uy + 0.5 * fy / rho
            feq = _equilibrium(rho, ux_eq, uy_eq, self._cx, self._cy, self._w)

            source = self._w * (1.0 - 0.5 * omega) * (2.0 * (self._cx * fx + self._cy * fy))
            f_post = f_pop * (1.0 - omega) + feq * omega + source

            if noise_amp > 0.0:
                key, noise_key = jax.random.split(key)
                noise = jax.random.normal(noise_key, shape=f_post.shape) * noise_amp
                noise = noise - jnp.mean(noise, axis=0, keepdims=True)
                f_post = f_post + noise

            f_post = jnp.clip(f_post, 1e-12, 1e6)

            f_stream = jnp.zeros_like(f_post)
            for i in range(self._q):
                f_stream = f_stream.at[i, :, :].set(
                    jnp.roll(f_post[i, :, :], shift=(cy_int[i], cx_int[i]), axis=(0, 1))
                )

            # Wedge bounce-back
            f_bounced = f_post[self._inv_idx, :, :]
            f_stream = jnp.where(self._obstacle_mask[None, :, :], f_bounced, f_stream)

            # Channel boundaries (weak-slip)
            f_stream = f_stream.at[:, 0, :].set(f_stream[:, 1, :])
            f_stream = f_stream.at[:, -1, :].set(f_stream[:, -2, :])

            # Inlet (left boundary): fixed inflow profile
            rho_in = jnp.ones((self.config.ny, 1), dtype=jnp.float64)
            ux_in = self._inlet_profile[:, None]
            uy_in = jnp.zeros_like(ux_in)
            feq_in = _equilibrium(rho_in, ux_in, uy_in, self._cx, self._cy, self._w)
            f_stream = f_stream.at[:, :, 0:1].set(feq_in)

            # Outlet (right boundary): zero-gradient copy
            f_stream = f_stream.at[:, :, -1].set(f_stream[:, :, -2])

            rho_n = jnp.sum(f_stream, axis=0)
            rho_n = jnp.clip(rho_n, 1e-8, 1e6)
            ux_n = jnp.sum(f_stream * self._cx, axis=0) / rho_n
            uy_n = jnp.sum(f_stream * self._cy, axis=0) / rho_n
            return f_stream, key, rho_n, ux_n, uy_n

        return _advance

    def _compute_shock_metric(self, rho: np.ndarray) -> float:
        grad_y, grad_x = np.gradient(rho)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        x0 = max(2, int(self.config.wedge_start_x) - 8)
        x1 = min(self.config.nx - 2, int(self.config.wedge_start_x + self.config.wedge_length + 28))
        y_mid = int(self.config.ny // 2)
        band = max(8, int(self.config.ny * 0.18))
        y0 = max(1, y_mid - band)
        y1 = min(self.config.ny - 1, y_mid + band)

        roi = grad_mag[y0:y1, x0:x1]
        if roi.size == 0:
            return float(np.mean(grad_mag))
        return float(np.mean(roi))

    def _snapshot(self) -> Dict[str, np.ndarray]:
        return {
            "rho": np.asarray(self._rho, dtype=np.float32),
            "ux": np.asarray(self._ux, dtype=np.float32),
            "uy": np.asarray(self._uy, dtype=np.float32),
            "mask": self._obstacle_mask_np.astype(np.float32),
        }

    def reset(self) -> Dict[str, np.ndarray]:
        self.step_count = 0
        self.baseline_shock = None
        self.last_shock = 0.0
        self._key = jax.random.PRNGKey(int(self.config.seed))

        rho0 = jnp.ones((self.config.ny, self.config.nx), dtype=jnp.float64)
        ux0 = jnp.broadcast_to(self._inlet_profile[:, None], (self.config.ny, self.config.nx))
        uy0 = jnp.zeros_like(ux0)
        self._f_pop = _equilibrium(rho0, ux0, uy0, self._cx, self._cy, self._w)

        zero_jet = jnp.asarray(0.0, dtype=jnp.float64)
        for _ in range(max(0, int(self.config.warmup_steps))):
            self._f_pop, self._key, self._rho, self._ux, self._uy = self._step_fn(
                self._f_pop,
                self._key,
                zero_jet,
            )

        if self._rho is None:
            self._rho = rho0
            self._ux = ux0
            self._uy = uy0

        self.baseline_shock = self._compute_shock_metric(np.asarray(self._rho))
        self.last_shock = float(self.baseline_shock)
        return self._snapshot()

    def step(self, control_signal: float) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.step_count += 1
        control = float(np.clip(control_signal, -1.0, 1.0))
        jet_scalar = jnp.asarray(control * self.config.jet_gain, dtype=jnp.float64)

        self._f_pop, self._key, self._rho, self._ux, self._uy = self._step_fn(
            self._f_pop,
            self._key,
            jet_scalar,
        )

        rho_np = np.asarray(self._rho, dtype=np.float64)
        shock = self._compute_shock_metric(rho_np)
        self.last_shock = float(shock)

        baseline = float(self.baseline_shock if self.baseline_shock is not None else shock)
        shock_reduction = (baseline - shock) / (baseline + 1e-8)
        reward = (
            float(shock_reduction)
            - float(self.config.reward_control_penalty) * abs(control)
        )

        unstable = (
            (not np.isfinite(rho_np).all())
            or float(np.max(rho_np)) > float(self.config.instability_density_cap)
            or float(np.min(rho_np)) <= 0.0
        )
        done = bool(unstable or self.step_count >= int(self.config.max_steps))
        if unstable:
            reward = -10.0

        info = {
            "shock_strength": float(shock),
            "shock_reduction": float(shock_reduction),
            "baseline_shock": float(baseline),
            "jet_power": float(control * self.config.jet_gain),
            "rho_mean": float(np.mean(rho_np)),
            "rho_std": float(np.std(rho_np)),
            "max_density": float(np.max(rho_np)),
            "min_density": float(np.min(rho_np)),
            "instability": bool(unstable),
            "step_count": int(self.step_count),
        }
        return self._snapshot(), float(reward), done, info

    def get_replay_state(self) -> Dict[str, Any]:
        return {
            "f_pop": np.asarray(self._f_pop, dtype=np.float64),
            "key": np.asarray(self._key),
            "step_count": int(self.step_count),
            "baseline_shock": float(self.baseline_shock if self.baseline_shock is not None else 0.0),
            "last_shock": float(self.last_shock),
        }

    def set_replay_state(self, state: Dict[str, Any]) -> None:
        self._f_pop = jnp.asarray(state["f_pop"], dtype=jnp.float64)
        self._key = jnp.asarray(state["key"])
        self.step_count = int(state.get("step_count", 0))
        self.baseline_shock = float(state.get("baseline_shock", 0.0))
        self.last_shock = float(state.get("last_shock", self.baseline_shock))

        self._rho = jnp.sum(self._f_pop, axis=0)
        self._rho = jnp.clip(self._rho, 1e-8, 1e6)
        self._ux = jnp.sum(self._f_pop * self._cx, axis=0) / self._rho
        self._uy = jnp.sum(self._f_pop * self._cy, axis=0) / self._rho
