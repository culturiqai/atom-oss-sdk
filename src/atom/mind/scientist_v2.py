#!/usr/bin/env python3
"""
ATOM SCIENTIST V2: STRUCTURAL COUPLING
========================================
Upgrades the Scientist from scalar-drip to structural feedback.

V1 Problem:
  Scientist discovers: reward ≈ -0.045 / latent_6
  Brain receives:      scalar_prediction = -0.12, trust = 0.3
  Brain can't use:     the STRUCTURE of the equation

V2 Solution:
  Scientist discovers: reward ≈ -0.045 / latent_6
  Brain receives:      [prediction, ∂law/∂latent_0..7, verified_trust] = 10-dim
  Now Brain knows:     "latent_6 is the only thing that matters,
                        sensitivity is 0.045/latent_6², and this law
                        has been verified against recent data"

Architecture:
  ┌────────────────┐
  │  PySR Scientist │ ← discovers symbolic law on RESIDUAL reward
  │  (existing V1)  │   (raw reward − baseline removes trivial
  └───────┬────────┘    mean_speed correlation)
          │ law_str (sympy expression)
          ▼
  ┌────────────────┐
  │  SymbolicJacob  │ ← computes ∂law/∂latent_i analytically via sympy.diff
  │  (new)          │
  └───────┬────────┘
          │ jacobian: (n_latent,) sensitivity vector
          ▼
  ┌────────────────┐
  │  Verifier       │ ← rolling check: does the law actually predict?
  │  (new)          │
  └───────┬────────┘
          │ verified_trust: scalar in [0,1]
          ▼
  ┌────────────────┐
  │  Rich Signal    │ → [pred, J_0..J_7, v_trust] = 10-dim tensor
  │  to Brain       │
  └────────────────┘

The Brain's theory_adapter changes from Linear(1→16) to Linear(10→16).
Same Tanh activation. Same concatenation with visual_latent + last_action.
But now the LTC receives gradient-prior information about which latent
dimensions the Scientist thinks matter.

Author: ATOM Platform
Date: February 2026
"""

from __future__ import annotations

import os
import time
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from atom.mind.governance import (
    ActiveExperimentScheduler,
    DiscoveryGovernance,
    GovernanceDecision,
    HypothesisRegistry,
)
from atom.platform.contracts import HypothesisRecord, TheoryPacket, schema_hash_from_names

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sympy import symbols, sympify, diff, lambdify, Float, oo, zoo, nan as sp_nan
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

logger = logging.getLogger("atom.scientist_v2")


def _load_scientific_evidence_hint_from_file() -> Dict[str, float]:
    """Best-effort load of scientific evidence hints from benchmark artifact."""
    path = Path(
        str(os.getenv("ATOM_SCI_EVIDENCE_PATH", "validation_outputs/scientific_integrity.json"))
    )
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary", {})
    source = summary if isinstance(summary, dict) else payload

    out: Dict[str, float] = {}
    fdr = source.get("false_discovery_rate")
    if fdr is not None:
        try:
            v = float(fdr)
            if np.isfinite(v):
                out["null_false_discovery_rate"] = float(v)
        except Exception:
            pass
    seed_stability = source.get("seed_perturbation_stability")
    if seed_stability is not None:
        try:
            v = float(seed_stability)
            if np.isfinite(v):
                out["seed_perturbation_stability"] = float(v)
        except Exception:
            pass
    calibration_error = source.get("calibration_error_mean")
    if calibration_error is not None:
        try:
            v = float(calibration_error)
            if np.isfinite(v):
                out["calibration_error_mean"] = float(v)
        except Exception:
            pass
    intervention_consistency = source.get("intervention_consistency_mean")
    if intervention_consistency is not None:
        try:
            v = float(intervention_consistency)
            if np.isfinite(v):
                out["intervention_consistency_mean"] = float(v)
        except Exception:
            pass
    return out


# =============================================================================
# SYMBOLIC JACOBIAN ENGINE
# =============================================================================

class SymbolicJacobian:
    """
    Computes ∂law/∂feature_i analytically from a sympy expression.

    Given a discovered law like:
        reward = -0.045 / latent_6

    Produces a callable that maps feature vectors to Jacobian vectors:
        J(features) = [∂law/∂action, ∂law/∂mean_speed, ..., ∂law/∂latent_7]

    We only expose the latent dimensions (indices 3..10) to the Brain,
    since action and physics features (mean_speed, turbulence) are not
    direct inputs to the policy.
    """

    def __init__(self, variable_names: List[str], latent_indices: Optional[List[int]] = None):
        """
        Args:
            variable_names: Full list of scientist feature names
                           e.g. ["action", "mean_speed", "turbulence", "latent_0", ...]
            latent_indices: Which indices in variable_names are latent dims.
                           If None, auto-detect "latent_*" names.
        """
        self.variable_names = list(variable_names)
        self.n_features = len(variable_names)

        if latent_indices is not None:
            self.latent_indices = list(latent_indices)
        else:
            self.latent_indices = [
                i for i, name in enumerate(variable_names)
                if name.startswith("latent_")
            ]

        self.n_latent = len(self.latent_indices)
        self.sym_vars = [symbols(n) for n in self.variable_names]

        # Compiled Jacobian function (set by compile_law)
        self._jacobian_func: Optional[Callable] = None
        self._law_expr = None

    def compile_law(self, law_str: str) -> bool:
        """
        Parse a law string and compile its Jacobian w.r.t. latent dimensions.

        Returns True on success, False on failure (graceful — never crashes agent).
        """
        if not HAS_SYMPY:
            logger.warning("sympy not available; Jacobian disabled")
            return False

        try:
            expr = sympify(law_str)
            self._law_expr = expr

            # Smooth Abs(x) → sqrt(x² + ε) for differentiability
            # Abs produces sign(x) derivatives that can't be lambdified
            try:
                from sympy import Wild, Abs as SymAbs, sqrt as sym_sqrt
                _w = Wild('_w')
                expr_smooth = expr.replace(SymAbs(_w), sym_sqrt(_w**2 + 1e-12))
            except Exception:
                expr_smooth = expr

            # Compute symbolic partial derivatives w.r.t. each latent variable
            partials = []
            for idx in self.latent_indices:
                var = self.sym_vars[idx]
                partial = diff(expr_smooth, var)

                # Safety: check for singularities (zoo, oo, nan)
                partial = partial.replace(zoo, Float(0))
                partial = partial.replace(oo, Float(0))

                # Force evaluation of Derivative objects (Abs, sign, Piecewise)
                try:
                    partial = partial.rewrite("Piecewise").doit()
                except Exception:
                    pass

                partials.append(partial)

            # Compile to numpy-callable (most portable backend).
            # We ALWAYS evaluate in numpy-land and wrap to torch if needed,
            # avoiding backend mismatch between compile and evaluate.
            for backend in ["numpy", "math"]:
                try:
                    self._jacobian_func = lambdify(
                        self.sym_vars,
                        partials,
                        modules=backend
                    )
                    break
                except Exception:
                    continue
            else:
                logger.warning("All lambdify backends failed for Jacobian")
                self._jacobian_func = None
                return False

            logger.info(
                f"Compiled Jacobian for {len(partials)} latent dims. "
                f"Law: {law_str[:60]}..."
            )
            return True

        except Exception as e:
            logger.error(f"Jacobian compilation failed: {e}")
            self._jacobian_func = None
            return False

    def evaluate(self, features) -> "np.ndarray":
        """
        Evaluate the Jacobian at a given feature vector.

        Args:
            features: (D,) or (B, D) feature tensor/array (torch.Tensor or np.ndarray)

        Returns:
            jacobian: (n_latent,) or (B, n_latent) — same type as input
                      ∂law/∂latent_i evaluated at the given features
        """
        if self._jacobian_func is None:
            # No law compiled — return zeros
            if HAS_TORCH and isinstance(features, torch.Tensor):
                if features.dim() == 1:
                    return torch.zeros(self.n_latent, device=features.device)
                return torch.zeros(features.shape[0], self.n_latent, device=features.device)
            x = np.asarray(features, dtype=np.float32)
            if x.ndim == 1:
                return np.zeros(self.n_latent, dtype=np.float32)
            return np.zeros((x.shape[0], self.n_latent), dtype=np.float32)

        try:
            is_torch = HAS_TORCH and isinstance(features, torch.Tensor)

            # ALWAYS evaluate in numpy (jacobian_func is numpy-compiled)
            if is_torch:
                x = features.detach().cpu().float().numpy()
            else:
                x = np.asarray(features, dtype=np.float32)

            was_1d = x.ndim == 1
            if was_1d:
                x = x.reshape(1, -1)

            # Unpack columns as numpy float64 arrays
            cols = [x[:, i].astype(np.float64) for i in range(min(x.shape[1], self.n_features))]
            while len(cols) < self.n_features:
                cols.append(np.zeros_like(cols[0]))

            result = self._jacobian_func(*cols)

            # Assemble (B, n_latent) numpy array
            if isinstance(result, (list, tuple)):
                parts = []
                for r in result:
                    if isinstance(r, np.ndarray):
                        parts.append(r.astype(np.float32))
                    elif isinstance(r, (int, float, np.floating, np.integer)):
                        parts.append(np.full(x.shape[0], float(r), dtype=np.float32))
                    else:
                        try:
                            parts.append(np.full(x.shape[0], float(r), dtype=np.float32))
                        except (TypeError, ValueError):
                            parts.append(np.zeros(x.shape[0], dtype=np.float32))
                J = np.stack(parts, axis=-1)
            else:
                J = np.asarray(result, dtype=np.float32)
                if J.ndim == 1:
                    J = J.reshape(-1, 1)

            # Numerical safety
            J = np.clip(J, -10.0, 10.0)
            J = np.where(np.isfinite(J), J, 0.0).astype(np.float32)

            if was_1d:
                J = J.squeeze(0)

            # Convert back to torch if input was torch
            if is_torch:
                return torch.tensor(J, dtype=torch.float32, device=features.device)
            return J

        except Exception as e:
            logger.debug(f"Jacobian evaluation failed: {e}")
            if HAS_TORCH and isinstance(features, torch.Tensor):
                if features.dim() == 1:
                    return torch.zeros(self.n_latent, device=features.device)
                return torch.zeros(features.shape[0], self.n_latent, device=features.device)
            x = np.asarray(features, dtype=np.float32)
            if x.ndim == 1:
                return np.zeros(self.n_latent, dtype=np.float32)
            return np.zeros((x.shape[0], self.n_latent), dtype=np.float32)


# =============================================================================
# VERIFICATION ENGINE
# =============================================================================

class LawVerifier:
    """
    Rolling verification of discovered laws against actual observations.

    Tracks three independent verification signals:
      1. Predictive accuracy: rolling RMSE of (predicted_reward, actual_reward)
      2. Correlation stability: rolling Pearson r — does the relationship hold?
      3. Interventional consistency: when action changes significantly,
         does the prediction error stay bounded?

    These combine into a verified_trust ∈ [0,1] that's harder to game than
    the raw conformal score (which only measures calibration set fit).
    
    Supports a structural_prior: when the law's structure (which latents matter)
    is consistent across sleep cycles, a trust floor is maintained even when
    the verifier resets for new predictions.
    """

    def __init__(
        self,
        window_size: int = 20,
        intervention_threshold: float = 0.3,
        holdout_fraction: float = 0.35,
        min_holdout_points: int = 4,
    ):
        """
        Args:
            window_size: Number of recent (pred, actual) pairs to track
            intervention_threshold: Action change magnitude that counts as "intervention"
        """
        self.window_size = window_size
        self.intervention_threshold = intervention_threshold
        self.holdout_fraction = float(np.clip(float(holdout_fraction), 0.1, 0.8))
        self.min_holdout_points = max(2, int(min_holdout_points))

        # Rolling buffers
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.actions = deque(maxlen=window_size)

        # Structural prior: accumulates when same structure persists
        self._structural_prior = 0.0
        self._structural_continuity_count = 0

        # Cached metrics
        self._last_rmse = float("inf")
        self._last_corr = 0.0
        self._last_interv_acc = 0.0
        self._last_holdout_rmse = float("inf")
        self._last_holdout_corr = 0.0
        self._last_oos_trust = 0.0

    def _safe_accuracy_trust(self, preds: np.ndarray, acts: np.ndarray) -> Tuple[float, float]:
        rmse = float(np.sqrt(np.mean((preds - acts) ** 2)))
        act_std = float(np.std(acts) + 1e-8)
        pred_std = float(np.std(preds) + 1e-8)
        act_level = float(np.mean(np.abs(acts)))
        pred_level = float(np.mean(np.abs(preds)))
        scale = max(
            act_std,
            0.25 * act_level,
            0.25 * pred_level,
            1e-3,
        )
        nrmse = min(rmse / scale, 2.0)
        return float(max(0.0, 1.0 - nrmse)), rmse

    def _safe_correlation_trust(
        self,
        preds: np.ndarray,
        acts: np.ndarray,
        accuracy_trust: float,
    ) -> float:
        pred_std = float(np.std(preds) + 1e-8)
        act_std = float(np.std(acts) + 1e-8)
        if pred_std < 1e-10 or act_std < 1e-10:
            return float(np.clip(0.25 + 0.5 * accuracy_trust, 0.0, 1.0))
        corr = float(np.corrcoef(preds, acts)[0, 1])
        if np.isfinite(corr):
            return float(np.clip(0.5 * (corr + 1.0), 0.0, 1.0))
        return float(np.clip(0.25 + 0.5 * accuracy_trust, 0.0, 1.0))

    def observe(self, prediction: float, actual: float, action: float) -> None:
        """Record a (prediction, actual, action) triple."""
        if not (np.isfinite(prediction) and np.isfinite(actual)):
            return
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.actions.append(action)

    @property
    def has_data(self) -> bool:
        return len(self.predictions) >= 5

    def note_structural_continuity(self) -> None:
        """
        Called when a new law has the SAME Jacobian structure as the old one.
        Builds a structural prior that provides a trust floor.
        """
        self._structural_continuity_count += 1
        # Prior ramps up: 0.2 → 0.3 → 0.35 → ... (asymptotes ~0.4)
        self._structural_prior = min(0.4, 0.2 + 0.05 * min(self._structural_continuity_count, 4))

    def note_structural_change(self) -> None:
        """Called when a new law has a DIFFERENT Jacobian structure."""
        self._structural_continuity_count = 0
        self._structural_prior = 0.0

    def compute_verified_trust(self) -> float:
        """
        Compute a verified trust score combining three signals + structural prior.

        Returns:
            trust ∈ [0, 1] where:
                1.0 = law perfectly predicts reward across all conditions
                0.0 = law is useless or anti-correlated
        """
        if not self.has_data:
            # No predictive data yet — return structural prior as floor
            return self._structural_prior

        preds_all = np.array(self.predictions, dtype=np.float64)
        acts_all = np.array(self.actuals, dtype=np.float64)
        actions_all = np.array(self.actions, dtype=np.float64)

        n = len(preds_all)
        holdout_n = int(round(float(n) * float(self.holdout_fraction)))
        holdout_n = max(self.min_holdout_points, holdout_n)
        holdout_n = min(holdout_n, max(1, n - 3))
        split_idx = max(3, n - holdout_n)

        preds = preds_all[:split_idx]
        acts = acts_all[:split_idx]
        actions = actions_all[:split_idx]

        preds_holdout = preds_all[split_idx:]
        acts_holdout = acts_all[split_idx:]

        # 1. Predictive accuracy (RMSE -> trust)
        accuracy_trust, rmse = self._safe_accuracy_trust(preds, acts)

        # 2. Correlation stability (Pearson r -> trust)
        corr_trust = self._safe_correlation_trust(preds, acts, accuracy_trust)

        # 3. Interventional consistency
        # When action changes significantly, does prediction error stay bounded?
        if len(actions) >= 15:
            action_diffs = np.abs(np.diff(actions))
            interventions = action_diffs > self.intervention_threshold

            if interventions.sum() >= 3:
                errors = np.abs(preds[1:] - acts[1:])
                interv_errors = errors[interventions]
                non_interv_errors = errors[~interventions]

                if len(non_interv_errors) > 0 and non_interv_errors.mean() > 1e-10:
                    ratio = interv_errors.mean() / (non_interv_errors.mean() + 1e-10)
                    interv_trust = max(0.0, 1.0 - max(0.0, ratio - 1.0))
                else:
                    interv_trust = 0.5
            else:
                interv_trust = 0.5
        else:
            interv_trust = 0.5

        # In-sample component.
        geo = (max(accuracy_trust, 1e-6) * max(corr_trust, 1e-6) * max(interv_trust, 1e-6)) ** (1 / 3)
        in_sample_trust = float(np.clip(0.7 * geo + 0.3 * accuracy_trust, 0.0, 1.0))

        # Out-of-sample component from a chronological holdout slice.
        if len(preds_holdout) >= self.min_holdout_points:
            holdout_acc, holdout_rmse = self._safe_accuracy_trust(preds_holdout, acts_holdout)
            holdout_corr = self._safe_correlation_trust(preds_holdout, acts_holdout, holdout_acc)
            oos_trust = float(np.clip(0.65 * holdout_acc + 0.35 * holdout_corr, 0.0, 1.0))
        else:
            holdout_acc = accuracy_trust
            holdout_corr = corr_trust
            holdout_rmse = rmse
            oos_trust = in_sample_trust

        verified_trust = float(np.clip(0.6 * in_sample_trust + 0.4 * oos_trust, 0.0, 1.0))

        # Final trust = max(structural prior, verified trust)
        # Structural prior provides a floor when predictions are still noisy
        final = max(self._structural_prior, float(np.clip(verified_trust, 0.0, 1.0)))

        # Cache for logging
        self._last_rmse = float(rmse)
        self._last_corr = float(corr_trust)
        self._last_interv_acc = float(interv_trust)
        self._last_holdout_rmse = float(holdout_rmse)
        self._last_holdout_corr = float(holdout_corr)
        self._last_oos_trust = float(oos_trust)

        return final

    def reset(self):
        """Clear verification buffers (call when a new law is compiled).
        Does NOT reset structural prior — that persists across law changes.
        """
        self.predictions.clear()
        self.actuals.clear()
        self.actions.clear()
        self._last_rmse = float("inf")
        self._last_corr = 0.0
        self._last_interv_acc = 0.0
        self._last_holdout_rmse = float("inf")
        self._last_holdout_corr = 0.0
        self._last_oos_trust = 0.0

    def full_reset(self):
        """Clear everything including structural prior (new structure)."""
        self.reset()
        self._structural_prior = 0.0
        self._structural_continuity_count = 0


# =============================================================================
# REWARD BASELINE (RESIDUAL FITTING)
# =============================================================================

class RewardBaseline:
    """
    Adaptive linear baseline: reward ≈ slope · mean_speed + intercept

    Problem diagnosed: PySR converges on `mean_speed*(-23) + 1.1` because 
    that's the dominant correlation in raw reward.  But the Brain's critic 
    already learns this through backprop.  The Scientist should find what 
    the critic CAN'T — latent structure beyond the obvious.

    Solution: fit a rolling-window OLS baseline on (mean_speed → reward),
    then feed PySR the RESIDUAL.  Since ∂baseline/∂latent_i = 0, the 
    Jacobian is purely about latent structure.  PySR is forced to discover
    `latent_3*(-0.15)` instead of the trivial `mean_speed*(-23)`.

    Properties:
      - Adapts as the policy evolves (rolling window)
      - Stable within a sleep cycle (PySR sees consistent residuals)
      - Only 2 parameters (slope + intercept) — can't overfit
      - Reports R² so we can see how much trivial variance was removed
    """

    def __init__(self, mean_speed_idx: int, window_size: int = 200,
                 min_samples: int = 30):
        """
        Args:
            mean_speed_idx: Index of mean_speed in the feature vector
            window_size: Rolling window for OLS fit
            min_samples: Minimum observations before fitting
        """
        self._ms_idx = mean_speed_idx
        self._window = window_size
        self._min_n = min_samples

        # Rolling buffers
        self._x = deque(maxlen=window_size)
        self._y = deque(maxlen=window_size)

        # Current fit: reward ≈ slope * mean_speed + intercept
        self.slope = 0.0
        self.intercept = 0.0
        self._fitted = False
        self._fit_count = 0
        self._r_squared = 0.0

    def _extract_ms(self, features) -> float:
        """Extract mean_speed from feature vector (handles torch/numpy)."""
        val = features[self._ms_idx]
        if HAS_TORCH and isinstance(val, torch.Tensor):
            return float(val.detach().cpu().item())
        return float(val)

    def update(self, features, reward: float) -> None:
        """Add (mean_speed, reward) observation and refit if ready."""
        x = self._extract_ms(features)
        y = float(reward)

        if not (np.isfinite(x) and np.isfinite(y)):
            return

        self._x.append(x)
        self._y.append(y)

        if len(self._x) >= self._min_n:
            self._refit()

    def _refit(self) -> None:
        """Ordinary least squares: y = a*x + b."""
        x = np.array(self._x, dtype=np.float64)
        y = np.array(self._y, dtype=np.float64)

        x_mean = x.mean()
        y_mean = y.mean()
        ss_xx = ((x - x_mean) ** 2).sum()

        if ss_xx > 1e-12:
            self.slope = float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)
            self.intercept = float(y_mean - self.slope * x_mean)
            self._fitted = True
            self._fit_count += 1

            # Compute R² for diagnostics
            y_pred = self.slope * x + self.intercept
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y_mean) ** 2).sum()
            self._r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    def predict(self, features) -> float:
        """Return baseline prediction (0 if not yet fitted)."""
        if not self._fitted:
            return 0.0
        x = self._extract_ms(features)
        return self.slope * x + self.intercept

    def residual(self, features, reward: float) -> float:
        """Return reward minus baseline prediction."""
        return float(reward) - self.predict(features)

    @property
    def is_ready(self) -> bool:
        return self._fitted

    @property
    def r_squared(self) -> float:
        return self._r_squared


# =============================================================================
# STRUCTURAL SCIENTIST (V2)
# =============================================================================

class StructuralScientist:
    """
    Wraps AtomScientist (V1) and adds structural coupling.

    Drop-in replacement in the training loop:
        # Old:
        theory_val, theory_trust = self.scientist.predict_theory(f_vec)
        theory_t = tensor([[theory_val]])

        # New:
        signal = self.structural_scientist.get_signal(f_vec, reward, action)
        theory_t = signal["tensor"]  # (1, 10) instead of (1, 1)

    The 10-dim signal is:
        [0]     scalar prediction from law
        [1..8]  Jacobian: ∂law/∂latent_0 .. ∂law/∂latent_7
        [9]     verified trust score
    """

    # Signal dimensions
    N_PREDICTION = 1
    N_JACOBIAN = 8   # matches latent_0..latent_7
    N_TRUST = 1
    SIGNAL_DIM = N_PREDICTION + N_JACOBIAN + N_TRUST  # = 10

    def __init__(
        self,
        scientist,  # AtomScientist instance (V1)
        variable_names: List[str],
        verification_window: int = 20,
        trust_gate_threshold: float = 0.25,
        warmup_steps: int = 50,
        diagnostic_trust_floor: float = 0.0,
        diagnostic_trust_floor_start_step: Optional[int] = None,
        signal_trust_scale_mode: Optional[str] = None,
    ):
        """
        Args:
            scientist: Existing AtomScientist instance
            variable_names: Feature names (must match scientist's variable_names)
            verification_window: Rolling window for law verification
            trust_gate_threshold: Don't adopt structurally different law if
                                  current verified trust exceeds this
            warmup_steps: Don't inject signal into Brain for this many steps.
                         PySR still observes and discovers, but the policy isn't
                         destabilized by early noisy laws.  Critical for chaotic
                         worlds (cylinder wake) where first laws have Jacobian
                         norms ~1000x larger than converged laws.
            diagnostic_trust_floor: Optional trust floor (0-1) for controlled
                         diagnostics. Disabled by default.
            diagnostic_trust_floor_start_step: Step to start applying the
                         diagnostic trust floor. Defaults to warmup_steps + 1.
            signal_trust_scale_mode: How trust scales theory content dimensions.
                         "off" keeps a single trust gate in Brain (recommended).
                         "sqrt" applies sqrt(trust) scaling.
                         "full" applies direct trust scaling (legacy behavior).
        """
        self.scientist = scientist
        self.variable_names = list(variable_names)
        self._feature_schema_hash = schema_hash_from_names(self.variable_names)
        self._packet_version = "theory_packet.v1"
        self.last_theory_packet: Optional[TheoryPacket] = None
        self.last_hypothesis_record: Optional[HypothesisRecord] = None
        self.last_governance_decision: Optional[Dict[str, Any]] = None

        # Phase-1 governance: persistent lifecycle + law-adoption checks.
        self.registry = HypothesisRegistry()
        self.scheduler = ActiveExperimentScheduler()
        self.governance = DiscoveryGovernance(self.registry)
        self.last_experiment_plan: Optional[Dict[str, Any]] = None

        # Structural components
        self.jacobian = SymbolicJacobian(variable_names)
        self.verifier = LawVerifier(window_size=verification_window)

        # Reward baseline: removes trivial mean_speed→reward correlation
        # so PySR must find latent structure instead
        ms_idx = variable_names.index("mean_speed") if "mean_speed" in variable_names else 1
        self.baseline = RewardBaseline(
            mean_speed_idx=ms_idx,
            window_size=200,
            min_samples=30,
        )

        # Track which law we've compiled the Jacobian for
        self._last_compiled_law: Optional[str] = None
        self._last_jacobian_nonzero: set = set()
        self._trust_gate_threshold = trust_gate_threshold

        # Warmup: zero signal for first N steps
        self._warmup_steps = warmup_steps
        self._diagnostic_trust_floor = float(
            np.clip(float(diagnostic_trust_floor), 0.0, 1.0)
        )
        if diagnostic_trust_floor_start_step is None:
            diagnostic_trust_floor_start_step = int(self._warmup_steps) + 1
        self._diagnostic_trust_floor_start_step = max(
            1, int(diagnostic_trust_floor_start_step)
        )
        if signal_trust_scale_mode is None:
            signal_trust_scale_mode = os.getenv(
                "ATOM_SCIENTIST_SIGNAL_TRUST_SCALE_MODE", "off"
            )
        scale_mode = str(signal_trust_scale_mode).strip().lower()
        if scale_mode not in {"off", "sqrt", "full"}:
            logger.warning(
                "Invalid signal_trust_scale_mode=%s, falling back to off",
                scale_mode,
            )
            scale_mode = "off"
        self._signal_trust_scale_mode = scale_mode
        self._step_count = 0
        self._last_action = 0.0
        self._last_discovery_target = 0.0

        null_fdr_hint_raw = os.getenv("ATOM_GOV_NULL_FDR_HINT")
        seed_stability_hint_raw = os.getenv("ATOM_GOV_SEED_STABILITY_HINT")
        calibration_error_hint_raw = os.getenv("ATOM_GOV_CALIBRATION_ERROR_HINT")
        intervention_consistency_hint_raw = os.getenv(
            "ATOM_GOV_INTERVENTION_CONSISTENCY_HINT"
        )
        self._scientific_evidence_hint: Dict[str, float] = {}
        try:
            if null_fdr_hint_raw is not None:
                v = float(null_fdr_hint_raw)
                if np.isfinite(v):
                    self._scientific_evidence_hint["null_false_discovery_rate"] = float(v)
        except Exception:
            pass
        try:
            if seed_stability_hint_raw is not None:
                v = float(seed_stability_hint_raw)
                if np.isfinite(v):
                    self._scientific_evidence_hint["seed_perturbation_stability"] = float(v)
        except Exception:
            pass
        try:
            if calibration_error_hint_raw is not None:
                v = float(calibration_error_hint_raw)
                if np.isfinite(v):
                    self._scientific_evidence_hint["calibration_error_mean"] = float(v)
        except Exception:
            pass
        try:
            if intervention_consistency_hint_raw is not None:
                v = float(intervention_consistency_hint_raw)
                if np.isfinite(v):
                    self._scientific_evidence_hint["intervention_consistency_mean"] = float(v)
        except Exception:
            pass
        # Fallback: pull scientific hints from latest integrity artifact.
        file_hint = _load_scientific_evidence_hint_from_file()
        if "null_false_discovery_rate" not in self._scientific_evidence_hint:
            if "null_false_discovery_rate" in file_hint:
                self._scientific_evidence_hint["null_false_discovery_rate"] = float(
                    file_hint["null_false_discovery_rate"]
                )
        if "seed_perturbation_stability" not in self._scientific_evidence_hint:
            if "seed_perturbation_stability" in file_hint:
                self._scientific_evidence_hint["seed_perturbation_stability"] = float(
                    file_hint["seed_perturbation_stability"]
                )
        if "calibration_error_mean" not in self._scientific_evidence_hint:
            if "calibration_error_mean" in file_hint:
                self._scientific_evidence_hint["calibration_error_mean"] = float(
                    file_hint["calibration_error_mean"]
                )
        if "intervention_consistency_mean" not in self._scientific_evidence_hint:
            if "intervention_consistency_mean" in file_hint:
                self._scientific_evidence_hint["intervention_consistency_mean"] = float(
                    file_hint["intervention_consistency_mean"]
                )

        target_mode = str(
            os.getenv("ATOM_SCIENTIST_DISCOVERY_TARGET_MODE", "hybrid")
        ).strip().lower()
        if target_mode not in {"residual", "raw", "hybrid", "advantage"}:
            logger.warning(
                "Invalid ATOM_SCIENTIST_DISCOVERY_TARGET_MODE=%s, falling back to hybrid",
                target_mode,
            )
            target_mode = "hybrid"
        self._discovery_target_mode = target_mode
        self._hybrid_alpha = float(
            np.clip(
                float(os.getenv("ATOM_SCIENTIST_DISCOVERY_HYBRID_ALPHA", "0.35")),
                0.0,
                1.0,
            )
        )
        self._raw_trust_weight = float(
            np.clip(
                float(os.getenv("ATOM_SCIENTIST_RAW_TRUST_WEIGHT", "0.35")),
                0.0,
                1.0,
            )
        )
        self._reward_ema_beta = float(
            np.clip(
                float(os.getenv("ATOM_SCIENTIST_REWARD_EMA_BETA", "0.92")),
                0.5,
                0.999,
            )
        )
        self._reward_ema: Optional[float] = None

        # Stats
        self.stats = {
            "jacobian_compiles": 0,
            "jacobian_failures": 0,
            "signals_emitted": 0,
            "verification_resets": 0,
            "trust_gated_rejections": 0,
            "governance_rejections": 0,
            "governance_approved": 0,
            "warmup_zeros": 0,
            "jacobian_clipped": 0,
            "experiment_plans": 0,
        }

    @property
    def theory_dim(self) -> int:
        """Signal dimensionality for Brain's theory_adapter."""
        return self.SIGNAL_DIM

    def _build_hypothesis_record(
        self, equation: str, decision: Optional[Dict[str, Any]] = None
    ) -> HypothesisRecord:
        fit_dataset_id = f"wake_buffer:{len(getattr(self.scientist, 'short_term_X', []))}"
        has_cal = bool(
            getattr(self.scientist, "conformal", None)
            and getattr(self.scientist.conformal, "is_calibrated", False)
        )
        confidence = {
            "verified_trust": float(self.verifier.compute_verified_trust()),
            "structural_floor": float(self.verifier._structural_prior),
            "baseline_r2": float(self.baseline.r_squared),
            "equation_score": float(getattr(self.scientist, "best_law_score", float("nan"))),
            "out_of_sample_trust": float(self.verifier._last_oos_trust),
        }
        if decision is not None:
            confidence.update(
                {
                    "novelty_score": float(decision.get("novelty_score", 0.0)),
                    "stability_score": float(decision.get("stability_score", 0.0)),
                    "intervention_consistency": float(
                        decision.get("intervention_consistency", 0.0)
                    ),
                    "governance_approved": float(bool(decision.get("approved", False))),
                }
            )

        failure_modes = [
            "low_verified_trust",
            "structural_shift_detected",
            "interventional_instability",
        ]
        if decision is not None and decision.get("reasons"):
            failure_modes = list(decision.get("reasons"))

        return HypothesisRecord(
            equation=str(equation),
            fit_dataset_id=fit_dataset_id,
            calibration_dataset_id="conformal_holdout" if has_cal else "none",
            validity_window={
                "verification_window": int(self.verifier.window_size),
                "warmup_steps": int(self._warmup_steps),
                "governance_thresholds": {
                    "min_novelty": float(self.governance.min_novelty),
                    "min_stability": float(self.governance.min_stability),
                    "min_intervention": float(self.governance.min_intervention),
                },
            },
            failure_modes=failure_modes,
            confidence_components=confidence,
            scientist_version="scientist_v2",
            feature_schema_hash=self._feature_schema_hash,
            provenance={
                "packet_version": self._packet_version,
                "discovery_target_mode": self._discovery_target_mode,
                "hybrid_alpha": float(self._hybrid_alpha),
                "reward_ema_beta": float(self._reward_ema_beta),
                "governance_decision": dict(decision or {}),
                "scientific_evidence_hint": dict(self._scientific_evidence_hint),
                "diagnostic_trust_floor": float(self._diagnostic_trust_floor),
                "diagnostic_trust_floor_start_step": int(
                    self._diagnostic_trust_floor_start_step
                ),
                "raw_trust_weight": float(self._raw_trust_weight),
                "signal_trust_scale_mode": str(self._signal_trust_scale_mode),
            },
        )

    def _build_theory_packet(
        self,
        prediction: float,
        jacobian: np.ndarray,
        raw_trust: float,
        verified_trust: float,
        structural_floor: float,
    ) -> TheoryPacket:
        return TheoryPacket(
            prediction=float(prediction),
            jacobian=tuple(float(v) for v in jacobian[: self.N_JACOBIAN]),
            trust_raw=float(raw_trust),
            trust_verified=float(verified_trust),
            trust_structural_floor=float(structural_floor),
            version=self._packet_version,
            feature_schema_hash=self._feature_schema_hash,
        )

    def _update_experiment_plan(
        self,
        final_trust: float,
        action: Optional[float],
    ) -> Dict[str, Any]:
        if action is not None:
            try:
                action_float = float(action)
                if np.isfinite(action_float):
                    self._last_action = action_float
            except Exception:
                pass

        decision = self.last_governance_decision or {}
        approved_raw = decision.get("approved")
        approved: Optional[bool]
        if isinstance(approved_raw, bool):
            approved = approved_raw
        elif isinstance(approved_raw, (int, np.integer)):
            approved = bool(int(approved_raw))
        else:
            approved = None

        plan = self.scheduler.propose(
            verified_trust=float(final_trust),
            novelty_score=float(decision.get("novelty_score", 0.5)),
            stability_score=float(decision.get("stability_score", 0.5)),
            intervention_consistency=float(
                decision.get("intervention_consistency", self.verifier._last_interv_acc)
            ),
            same_support_refinement=bool(decision.get("same_support_refinement", False)),
            approved=approved,
            last_action=float(self._last_action),
            support=sorted(self._last_jacobian_nonzero),
        )
        self.last_experiment_plan = plan.to_dict()
        self.stats["experiment_plans"] += 1
        return self.last_experiment_plan

    def _sync_jacobian(self) -> None:
        """
        Check if the Scientist has a new law and recompile the Jacobian.
        Called internally before each get_signal().
        
        Key insight: predictions must ALWAYS be cleared on law change (even 
        same structure) because stored (prediction, actual) pairs used the 
        OLD law's predictions. Mixing old and new predictions corrupts
        the verifier's correlation and RMSE.
        
        BUT: if the Jacobian structure (which latent dims matter) is the same,
        we note structural continuity. This builds a trust floor that persists
        across law refinements, so the Brain always gets a nonzero signal
        when the Scientist has converged on a stable structure.
        
        Trust gate: if we have high structural trust AND PySR proposes a
        completely different structure, we block the switch.
        """
        current_law = getattr(self.scientist, "best_law", None)

        if current_law is not None and current_law != self._last_compiled_law:
            # Probe new law's structure without adopting yet
            probe_jac = SymbolicJacobian(self.variable_names)
            if not probe_jac.compile_law(current_law):
                self.stats["jacobian_failures"] += 1
                return

            test_feat = np.zeros(len(self.variable_names), dtype=np.float32) + 0.5
            new_J = probe_jac.evaluate(test_feat)
            new_nonzero = set(np.where(np.abs(new_J) > 1e-6)[0])

            # Check if structure changed
            structure_changed = (new_nonzero != self._last_jacobian_nonzero
                                 and len(self._last_jacobian_nonzero) > 0)

            # Trust gate: block structural changes when current structure is verified
            current_trust = self.verifier.compute_verified_trust()
            if structure_changed and current_trust > self._trust_gate_threshold:
                self.stats["trust_gated_rejections"] += 1
                self.stats["governance_rejections"] += 1
                gate_decision = GovernanceDecision(
                    approved=False,
                    novelty_score=0.0,
                    stability_score=1.0,
                    intervention_consistency=float(self.verifier._last_interv_acc),
                    same_support_refinement=False,
                    reasons=["trust_gate_rejection"],
                )
                self.last_governance_decision = gate_decision.to_dict()
                self.registry.record_decision(
                    equation=current_law,
                    support=sorted(new_nonzero),
                    status="rejected",
                    decision=gate_decision,
                    extra={"verified_trust": float(current_trust)},
                )
                logger.info(
                    f"Trust gate BLOCKED law switch: "
                    f"{self._last_jacobian_nonzero} → {new_nonzero} "
                    f"(current trust={current_trust:.3f} > {self._trust_gate_threshold})"
                )
                return

            intervention_metric = (
                0.5 if not self.verifier.has_data else float(self.verifier._last_interv_acc)
            )
            governance_decision = self.governance.evaluate(
                equation=current_law,
                support=sorted(new_nonzero),
                verified_trust=float(current_trust),
                interventional_metric=intervention_metric,
                scientific_evidence=self._scientific_evidence_hint or None,
            )
            self.last_governance_decision = governance_decision.to_dict()

            if not governance_decision.approved:
                self.stats["governance_rejections"] += 1
                self.registry.record_decision(
                    equation=current_law,
                    support=sorted(new_nonzero),
                    status="rejected",
                    decision=governance_decision,
                    extra={"verified_trust": float(current_trust)},
                )
                logger.info(
                    f"Governance rejected law adoption: reasons={governance_decision.reasons}"
                )
                return

            self.stats["governance_approved"] += 1

            # Accept the new law
            success = self.jacobian.compile_law(current_law)
            if success:
                self.stats["jacobian_compiles"] += 1
                self._last_compiled_law = current_law

                if structure_changed:
                    # Completely new structure → full reset
                    self.verifier.full_reset()
                    self.verifier.note_structural_change()
                    self.stats["verification_resets"] += 1
                    logger.info(
                        f"Structure changed: {self._last_jacobian_nonzero} → {new_nonzero}, "
                        f"full verifier reset"
                    )
                elif len(self._last_jacobian_nonzero) == 0:
                    # First ever law — no prior structure to compare against.
                    # Don't award structural continuity; just record the structure.
                    self.verifier.reset()
                    logger.info(
                        f"First law compiled, structure: {new_nonzero}"
                    )
                else:
                    # Same structure, different coefficients → clear predictions
                    # (they're from old law) but note structural continuity
                    self.verifier.reset()  # clears predictions, keeps structural prior
                    self.verifier.note_structural_continuity()
                    logger.info(
                        f"Law refined (same structure {new_nonzero}), "
                        f"predictions cleared, structural prior → "
                        f"{self.verifier._structural_prior:.3f} "
                        f"(continuity #{self.verifier._structural_continuity_count})"
                    )

                self._last_jacobian_nonzero = new_nonzero
                self.last_hypothesis_record = self._build_hypothesis_record(
                    current_law, decision=governance_decision.to_dict()
                )
                self.registry.record_decision(
                    equation=current_law,
                    support=sorted(new_nonzero),
                    status="accepted",
                    decision=governance_decision,
                    extra={"verified_trust": float(current_trust)},
                )
            else:
                self.stats["jacobian_failures"] += 1
                compile_fail_decision = GovernanceDecision(
                    approved=False,
                    novelty_score=float(governance_decision.novelty_score),
                    stability_score=float(governance_decision.stability_score),
                    intervention_consistency=float(
                        governance_decision.intervention_consistency
                    ),
                    same_support_refinement=bool(
                        governance_decision.same_support_refinement
                    ),
                    reasons=list(governance_decision.reasons) + ["jacobian_compile_failed"],
                )
                self.last_governance_decision = compile_fail_decision.to_dict()
                self.registry.record_decision(
                    equation=current_law,
                    support=sorted(new_nonzero),
                    status="rejected",
                    decision=compile_fail_decision,
                    extra={"verified_trust": float(current_trust)},
                )

    def _compute_discovery_target(
        self,
        features: Union[torch.Tensor, np.ndarray],
        reward: float,
    ) -> Tuple[float, float]:
        """
        Build the target used by Scientist discovery and verifier.
        Returns (target_value, residual_reward).
        """
        reward_float = float(reward)
        residual_reward = self.baseline.residual(features, reward_float)
        if self._discovery_target_mode == "residual":
            target = residual_reward
        elif self._discovery_target_mode == "raw":
            target = reward_float
        elif self._discovery_target_mode == "advantage":
            if self._reward_ema is None or not np.isfinite(self._reward_ema):
                self._reward_ema = reward_float
            target = reward_float - float(self._reward_ema)
        else:
            # Hybrid keeps physical law signal while preserving residual structure.
            alpha = float(self._hybrid_alpha)
            target = (1.0 - alpha) * residual_reward + alpha * reward_float
        if self._reward_ema is None or not np.isfinite(self._reward_ema):
            self._reward_ema = reward_float
        else:
            beta = float(self._reward_ema_beta)
            self._reward_ema = beta * float(self._reward_ema) + (1.0 - beta) * reward_float
        self._last_discovery_target = float(target)
        return float(target), float(residual_reward)

    def get_signal(
        self,
        features: Union[torch.Tensor, np.ndarray],
        reward: Optional[float] = None,
        action: Optional[float] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Compute the full structural signal for the Brain.

        Args:
            features: (D,) feature vector [action, mean_speed, turb, latent_0..7]
            reward: actual reward (for verification; None during inference-only)
            action: action taken (for interventional verification; None ok)
            device: torch device for output tensor

        Returns:
            dict with:
                "tensor":    (1, 10) tensor for Brain's theory_adapter
                "prediction": float — scalar law prediction
                "jacobian":  (8,) numpy — ∂law/∂latent_i (normalized)
                "jacobian_raw_norm": float — pre-normalization ||J||
                "trust":     float — verified trust (0 during warmup)
                "raw_trust": float — scientist's original trust
                "has_law":   bool
                "in_warmup": bool
        """
        self._step_count += 1
        self._sync_jacobian()

        # 1. Get scalar prediction + raw trust from V1 Scientist
        pred_val, raw_trust = self.scientist.predict_theory(features)

        # Convert to float
        if HAS_TORCH and isinstance(pred_val, torch.Tensor):
            pred_float = float(pred_val.detach().cpu().item() if pred_val.numel() == 1 else pred_val.detach().cpu().mean().item())
        else:
            pred_float = float(pred_val) if np.isfinite(float(pred_val)) else 0.0

        if HAS_TORCH and isinstance(raw_trust, torch.Tensor):
            raw_trust_float = float(raw_trust.detach().cpu().item() if raw_trust.numel() == 1 else raw_trust.detach().cpu().mean().item())
        else:
            raw_trust_float = float(raw_trust) if np.isfinite(float(raw_trust)) else 0.0

        # 2. Update verifier if we have ground truth
        #    Verifier compares prediction (which is a residual prediction)
        #    against actual residual (reward - baseline), not raw reward.
        if reward is not None and self._last_compiled_law is not None:
            action_float = float(action) if action is not None else 0.0
            discovery_target, _ = self._compute_discovery_target(features, float(reward))
            self.verifier.observe(pred_float, discovery_target, action_float)

        # 3. Compute Jacobian
        J = self.jacobian.evaluate(features)
        if HAS_TORCH and isinstance(J, torch.Tensor):
            J_np = J.detach().cpu().numpy().flatten()
        else:
            J_np = np.asarray(J, dtype=np.float32).flatten()

        # Pad or truncate to exactly N_JACOBIAN dims
        if len(J_np) < self.N_JACOBIAN:
            J_np = np.pad(J_np, (0, self.N_JACOBIAN - len(J_np)))
        elif len(J_np) > self.N_JACOBIAN:
            J_np = J_np[:self.N_JACOBIAN]

        # 3b. NORMALIZE Jacobian to unit ball
        #     Raw J norms: Taylor-Green ~0.01, cylinder wake ~10.0
        #     Without normalization, the 1000x scale difference destabilizes
        #     the Brain's theory_adapter (Linear(10→16)) which was calibrated
        #     for small values but suddenly sees huge inputs.
        #     Fix: divide by max(||J||, 1.0) — preserves direction, bounds magnitude.
        j_raw_norm = float(np.linalg.norm(J_np))
        if j_raw_norm > 1.0:
            J_np = J_np / j_raw_norm
            self.stats["jacobian_clipped"] += 1

        # 4. Compute verified trust (includes structural prior as floor)
        verified_trust = self.verifier.compute_verified_trust()
        structural_floor = self.verifier._structural_prior

        # V2 trust blend:
        #   The structural prior represents 2-10+ consecutive sleep cycles
        #   producing the SAME Jacobian structure.  It's the most reliable
        #   signal we have — "latent_1 keeps showing up."
        #
        #   The V1 conformal score (raw_trust) measures prediction accuracy
        #   on a 20-step window of RESIDUALS, which are inherently noisy.
        #   On cylinder wake, raw_trust ≈ 0.10-0.15 even for good laws
        #   because residuals are small and noisy.
        #
        #   Old logic: final = min(raw_trust, verified) → 0.15 crushes 0.40
        #   New logic: structural prior is a FLOOR.  Raw trust can suppress
        #   the predictive component (correlation/RMSE based) but not the
        #   structural floor.  This means:
        #     SP=0.40, raw=0.15 → final=0.40 (structure is trusted)
        #     SP=0.00, raw=0.80 → final=0.80 (pure predictive, no structure)
        #     SP=0.25, raw=0.50 → final=0.50 (predictive exceeds floor)
        if self.verifier.has_data and raw_trust_float > 0.01:
            # Conservative blend between V1 conformal trust and V2 verified trust.
            # This avoids trust collapse when one estimator is temporarily noisy.
            raw_w = float(self._raw_trust_weight)
            predictive_trust = raw_w * raw_trust_float + (1.0 - raw_w) * verified_trust
            final_trust = max(structural_floor, predictive_trust)
        else:
            final_trust = verified_trust

        diagnostic_floor_applied = 0.0
        if (
            self._diagnostic_trust_floor > 0.0
            and self._step_count >= self._diagnostic_trust_floor_start_step
        ):
            final_trust = max(final_trust, self._diagnostic_trust_floor)
            diagnostic_floor_applied = float(self._diagnostic_trust_floor)

        # 5. Assemble 10-dim signal
        signal_np = np.zeros(self.SIGNAL_DIM, dtype=np.float32)
        signal_np[0] = np.clip(pred_float, -1.0, 1.0)
        signal_np[1:1+self.N_JACOBIAN] = J_np  # already normalized
        signal_np[-1] = np.clip(final_trust, 0.0, 1.0)

        # 5b. Optional trust scaling of content dims (prediction + Jacobian).
        # "off" keeps trust gating centralized in Brain to avoid double attenuation.
        signal_content_scale = 1.0
        if self._signal_trust_scale_mode == "full":
            signal_content_scale = float(signal_np[-1])
        elif self._signal_trust_scale_mode == "sqrt":
            signal_content_scale = float(np.sqrt(max(0.0, float(signal_np[-1]))))
        signal_np[: self.N_PREDICTION + self.N_JACOBIAN] *= float(signal_content_scale)

        # 6. WARMUP GATE: zero the signal for the first N steps
        #    PySR still observes/discovers, verifier still tracks structure,
        #    but the Brain doesn't receive noisy early-law signals that
        #    destabilize the policy (especially on chaotic worlds like cylinder).
        in_warmup = self._step_count <= self._warmup_steps
        if in_warmup:
            signal_np[:] = 0.0
            final_trust = 0.0
            diagnostic_floor_applied = 0.0
            self.stats["warmup_zeros"] += 1

        # Convert to tensor (production uses torch; testing can use numpy)
        if HAS_TORCH:
            signal_tensor = torch.tensor(
                signal_np, dtype=torch.float32, device=device
            ).unsqueeze(0)  # (1, 10)
        else:
            signal_tensor = signal_np.reshape(1, -1)  # (1, 10) numpy

        self.stats["signals_emitted"] += 1

        packet = self._build_theory_packet(
            prediction=pred_float,
            jacobian=J_np,
            raw_trust=raw_trust_float,
            verified_trust=final_trust,
            structural_floor=structural_floor,
        )
        self.last_theory_packet = packet
        experiment_plan = self._update_experiment_plan(final_trust=final_trust, action=action)

        return {
            "tensor": signal_tensor,
            "prediction": pred_float,
            "jacobian": J_np.copy(),
            "jacobian_raw_norm": j_raw_norm,
            "trust": final_trust,
            "raw_trust": raw_trust_float,
            "verified_trust": verified_trust,
            "trust_structural_floor": structural_floor,
            "diagnostic_trust_floor": float(self._diagnostic_trust_floor),
            "diagnostic_trust_floor_start_step": int(
                self._diagnostic_trust_floor_start_step
            ),
            "diagnostic_trust_floor_applied": float(diagnostic_floor_applied),
            "raw_trust_weight": float(self._raw_trust_weight),
            "signal_trust_scale_mode": str(self._signal_trust_scale_mode),
            "signal_content_scale": float(signal_content_scale),
            "packet": packet,
            "packet_version": self._packet_version,
            "feature_schema_hash": self._feature_schema_hash,
            "hypothesis_record": self.last_hypothesis_record,
            "governance": self.last_governance_decision,
            "experiment_plan": experiment_plan,
            "has_law": self._last_compiled_law is not None,
            "in_warmup": in_warmup,
            "baseline_ready": self.baseline.is_ready,
            "baseline_slope": self.baseline.slope,
            "baseline_r2": self.baseline.r_squared,
            "discovery_target_mode": self._discovery_target_mode,
            "discovery_hybrid_alpha": self._hybrid_alpha,
            "discovery_target_value": self._last_discovery_target,
            "verifier_metrics": {
                "rmse": self.verifier._last_rmse,
                "correlation": self.verifier._last_corr,
                "interventional": self.verifier._last_interv_acc,
                "holdout_rmse": self.verifier._last_holdout_rmse,
                "holdout_correlation": self.verifier._last_holdout_corr,
                "out_of_sample_trust": self.verifier._last_oos_trust,
            },
        }

    def get_theory_packet(self) -> Optional[TheoryPacket]:
        """Return the latest emitted TheoryPacket (if any)."""
        return self.last_theory_packet

    def get_hypothesis_record(self) -> Optional[HypothesisRecord]:
        """Return the most recent accepted hypothesis metadata."""
        return self.last_hypothesis_record

    def get_governance_decision(self) -> Optional[Dict[str, Any]]:
        """Return the most recent governance decision for a candidate law."""
        return self.last_governance_decision

    def get_experiment_plan(self) -> Optional[Dict[str, Any]]:
        """Return the most recent active experiment recommendation."""
        return self.last_experiment_plan

    # ─── Pass-through to V1 Scientist ───

    def observe(self, features, target_metric: float) -> None:
        """
        Update baseline and forward configured discovery target to V1 Scientist.

        The baseline captures reward ≈ slope·mean_speed + intercept.
        Discovery target modes:
          residual: reward - baseline
          raw:      reward
          hybrid:   (1-a)*residual + a*reward
        """
        self.baseline.update(features, target_metric)
        discovery_target, _ = self._compute_discovery_target(features, float(target_metric))
        self.scientist.observe(features, discovery_target)

    def observe_and_verify(
        self,
        features,
        reward: float,
        prediction: float,
        action: float,
    ) -> None:
        """
        Single entry point after receiving reward.  Handles:
          1. Baseline update  (mean_speed → reward OLS)
          2. Residual compute (reward - baseline)
          3. V1 observe       (features, residual) → PySR buffer
          4. Verifier observe  (prediction, residual, action) → trust

        This replaces the old two-call pattern:
            scientist.observe(f_vec, reward)
            scientist.verifier.observe(pred, reward, action)

        The prediction should be from the CURRENT law (already a residual
        prediction since the law was fit on residuals).
        """
        try:
            action_float = float(action)
            if np.isfinite(action_float):
                self._last_action = action_float
        except Exception:
            pass

        # 1. Update baseline with (mean_speed, raw_reward)
        self.baseline.update(features, reward)

        # 2. Compute configured discovery target
        discovery_target, _ = self._compute_discovery_target(features, reward)

        # 3. V1 Scientist observes (features, target) for law fitting
        self.scientist.observe(features, discovery_target)

        # 4. Verifier compares prediction vs actual configured target
        if self._last_compiled_law is not None:
            self.verifier.observe(prediction, discovery_target, action)

    def ponder(self):
        """Forward to V1 Scientist."""
        return self.scientist.ponder()

    def fit_offline(self, X, y) -> None:
        """Forward to V1 Scientist, then recompile Jacobian."""
        self.scientist.fit_offline(X, y)
        self._sync_jacobian()

    def shutdown(self) -> None:
        """Forward to V1 Scientist."""
        self.scientist.shutdown()

    @property
    def best_law(self):
        return self.scientist.best_law

    @property
    def theory_archive(self):
        return self.scientist.theory_archive

    @property
    def new_discovery_alert(self):
        return self.scientist.new_discovery_alert

    @new_discovery_alert.setter
    def new_discovery_alert(self, value):
        self.scientist.new_discovery_alert = value


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def upgrade_scientist(
    scientist,
    variable_names: List[str],
    verification_window: int = 20,
    warmup_steps: int = 50,
) -> StructuralScientist:
    """
    Wrap an existing AtomScientist with structural coupling.

    Usage in training_loop.py:
        # After creating scientist:
        self.scientist = AtomScientist(variable_names=self.scientist_vars)
        self.structural_scientist = upgrade_scientist(
            self.scientist, self.scientist_vars
        )
    """
    return StructuralScientist(
        scientist=scientist,
        variable_names=variable_names,
        verification_window=verification_window,
        warmup_steps=warmup_steps,
    )


def get_theory_dim() -> int:
    """
    Return the signal dimensionality for Brain's theory_adapter.

    Usage in brain.py or experiment runner:
        from scientist_v2 import get_theory_dim
        brain = AtomBrain(theory_dim=get_theory_dim(), ...)
    """
    return StructuralScientist.SIGNAL_DIM


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  SCIENTIST V2: STRUCTURAL COUPLING SELF-TEST")
    print("=" * 60)

    # ── Test 1: Symbolic Jacobian ──
    print("\n  Test 1: Symbolic Jacobian")
    var_names = ["action", "mean_speed", "turbulence"] + [f"latent_{i}" for i in range(8)]
    jac = SymbolicJacobian(var_names)

    success = jac.compile_law("-0.045 / latent_6")
    assert success, "Jacobian compilation failed"

    features = np.array([0.1, 0.05, 0.01, 0.5, -0.3, 0.2, 0.1, -0.4, 0.6, 0.3, -0.1], dtype=np.float32)
    J = jac.evaluate(features)
    print(f"    Law: -0.045 / latent_6")
    print(f"    Features[latent_6] = {features[9]:.3f}")
    print(f"    Jacobian: {J}")
    # latent_6 is at index 6 in Jacobian output
    expected = 0.045 / features[9]**2
    assert abs(J[6] - expected) < 0.01, f"Jacobian value wrong: {J[6]} vs {expected}"
    print("    ✅ PASS")

    # ── Test 2: Linear law ──
    print("\n  Test 2: Linear law Jacobian")
    jac2 = SymbolicJacobian(var_names)
    jac2.compile_law("mean_speed * 7.0 + latent_3 * (-0.2) - 0.45")
    J2 = jac2.evaluate(features)
    print(f"    Law: mean_speed*7.0 + latent_3*(-0.2) - 0.45")
    print(f"    Jacobian (latent dims only): {J2}")
    assert abs(J2[3] - (-0.2)) < 0.001, f"Expected -0.2 at index 3, got {J2[3]}"
    print("    ✅ PASS")

    # ── Test 3: Verification Engine ──
    print("\n  Test 3: Law Verifier")
    verifier = LawVerifier(window_size=50)

    np.random.seed(42)
    for i in range(50):
        actual = np.sin(i * 0.1) * 0.1
        pred = actual + np.random.randn() * 0.01
        verifier.observe(pred, actual, action=np.sin(i * 0.05))

    good_trust = verifier.compute_verified_trust()
    print(f"    Good law trust: {good_trust:.3f} (expect > 0.5)")
    assert good_trust > 0.4, f"Good law should have decent trust, got {good_trust}"
    print("    ✅ PASS")

    verifier2 = LawVerifier(window_size=50)
    for i in range(50):
        actual = np.sin(i * 0.1) * 0.1
        pred = np.random.randn() * 0.1
        verifier2.observe(pred, actual, action=np.sin(i * 0.05))

    bad_trust = verifier2.compute_verified_trust()
    print(f"    Bad law trust:  {bad_trust:.3f} (expect < 0.3)")
    assert bad_trust < good_trust, "Bad law should have lower trust"
    print("    ✅ PASS")

    # ── Test 4: Signal dimensions ──
    print("\n  Test 4: Signal dimensions")
    assert get_theory_dim() == 10, f"Expected 10, got {get_theory_dim()}"
    print(f"    theory_dim = {get_theory_dim()}")
    print("    ✅ PASS")

    # ── Test 5: Batch evaluation ──
    print("\n  Test 5: Batch Jacobian")
    batch_features = np.random.randn(4, 11).astype(np.float32)
    J_batch = jac.evaluate(batch_features)
    assert J_batch.shape == (4, 8), f"Expected (4, 8), got {J_batch.shape}"
    print(f"    Batch shape: {J_batch.shape}")
    print("    ✅ PASS")

    # ── Test 6: All 18 real discovered laws ──
    print("\n  Test 6: All real discovered laws")
    real_laws = [
        "mean_speed*6.992453 - 0.44956324",
        "-turbulence + Abs(mean_speed + 5.5098243)**2 - 1*30.97593",
        "-0.09667613/latent_1",
        "latent_3*(-0.5583588)*cos((-1.61142729932819*turbulence**2 + 0.51223636*Abs(latent_5*mean_speed))*119.86806)",
        "(-0.063803315)*latent_5 - 0.08504382",
        "(latent_4 + latent_5 - 1*(-0.32852462) - 1.0183885/latent_0)*0.1474691",
        "latent_1*(-0.055769704)/latent_6",
        "latent_3*(-0.19828272)",
        "(-latent_1 + latent_5 + 0.19025771)*0.09497672",
        "(latent_1 - latent_5)*(-0.082053095)",
        "mean_speed + sin(latent_1 - latent_5)*(-0.66695136) + 0.48680314",
        "latent_6*0.12395711 - 1*0.15855405",
        "-0.04490153/latent_6",
        "-(-9.424582)*mean_speed - 0.56262124",
        "(latent_1 - latent_5)*(-0.07624834)",
        "latent_3*(-0.19608139)",
        "(-latent_1 + latent_5)*0.080665104",
        "latent_1*(-0.1295893)",
    ]
    compiled = 0
    for law in real_laws:
        jt = SymbolicJacobian(var_names)
        if jt.compile_law(law):
            J = jt.evaluate(features)
            assert J.shape == (8,)
            compiled += 1
    print(f"    {compiled}/{len(real_laws)} laws compiled and differentiated")
    assert compiled == len(real_laws), f"Expected all {len(real_laws)}, got {compiled}"
    print("    ✅ PASS")

    # ── Test 7: RewardBaseline ──
    print("\n  Test 7: RewardBaseline")
    bl = RewardBaseline(mean_speed_idx=1, window_size=100, min_samples=20)
    assert not bl.is_ready, "Should not be fitted yet"

    # Generate data: reward = -23 * mean_speed + 1.1 + noise
    np.random.seed(42)
    for i in range(50):
        ms = 0.04 + 0.005 * np.random.randn()
        reward = -23.0 * ms + 1.1 + 0.002 * np.random.randn()
        fake_features = np.array([0.0, ms, 0.01] + [0.0]*8, dtype=np.float32)
        bl.update(fake_features, reward)

    assert bl.is_ready, "Should be fitted after 50 samples"
    assert abs(bl.slope - (-23.0)) < 1.0, f"Slope should be ≈-23, got {bl.slope:.2f}"
    assert abs(bl.intercept - 1.1) < 0.1, f"Intercept should be ≈1.1, got {bl.intercept:.3f}"
    assert bl.r_squared > 0.95, f"R² should be > 0.95, got {bl.r_squared:.3f}"
    print(f"    Slope: {bl.slope:.2f} (expect ≈-23)")
    print(f"    Intercept: {bl.intercept:.3f} (expect ≈1.1)")
    print(f"    R²: {bl.r_squared:.3f} (expect > 0.95)")

    # Residual should be ≈0 for data on the line
    ms_test = 0.045
    r_test = -23.0 * ms_test + 1.1
    f_test = np.array([0.0, ms_test, 0.01] + [0.0]*8, dtype=np.float32)
    resid = bl.residual(f_test, r_test)
    assert abs(resid) < 0.05, f"Residual should be ≈0 on baseline, got {resid:.4f}"
    print(f"    Residual (on baseline): {resid:.4f} (expect ≈0)")
    print("    ✅ PASS")

    # ── Test 8: Residual fitting produces latent-dependent laws ──
    print("\n  Test 8: Residual target simulation")
    # Simulate: true reward = -23*ms + 1.1 + 0.15*latent_3
    # Without baseline: PySR finds -23*ms + 1.1 (J=0 for all latents)
    # With baseline: PySR target = residual ≈ 0.15*latent_3 (J[3] ≠ 0)
    bl2 = RewardBaseline(mean_speed_idx=1, window_size=100, min_samples=20)
    residuals = []
    latent3_vals = []
    for i in range(60):
        ms = 0.04 + 0.005 * np.random.randn()
        l3 = np.random.randn() * 0.5
        reward = -23.0 * ms + 1.1 + 0.15 * l3
        fake_f = np.array([0.0, ms, 0.01, 0.0, 0.0, l3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        bl2.update(fake_f, reward)
        if bl2.is_ready:
            residuals.append(bl2.residual(fake_f, reward))
            latent3_vals.append(l3)

    residuals = np.array(residuals)
    latent3_vals = np.array(latent3_vals)
    # Residuals should correlate with latent_3
    if len(residuals) > 10:
        corr = np.corrcoef(residuals, latent3_vals)[0, 1]
        print(f"    Correlation(residual, latent_3) = {corr:.3f} (expect > 0.8)")
        assert corr > 0.5, f"Residuals should correlate with latent_3, got r={corr:.3f}"
    print("    ✅ PASS")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print(f"  Signal dim: {get_theory_dim()} = [1 pred + 8 jacobian + 1 trust]")
    print("  Torch available:", HAS_TORCH)
    print("=" * 60)
