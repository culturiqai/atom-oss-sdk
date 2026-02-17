"""
ATOM MIND: CONFORMAL PREDICTION (Safety Layer)
----------------------------------------------
Wraps the Symbolic Scientist with statistical error bounds.

Implements *normalized split conformal prediction*:

- Base model gives f(x)
- Error estimator gives r(x) ≈ E[|y - f(x)| | x]
- Calibration scores: s_i = |y_i - f(x_i)| / max(r(x_i), eps)
- Quantile Q picked using canonical split-conformal finite-sample rule:
    k = ceil((n_cal + 1) * (1 - alpha))
    Q = k-th smallest score (1-indexed), clamped to [1, n_cal]

Prediction interval width:
    width(x) = 2 * Q * r(x)

Trust:
    trust(x) = exp(-lambda * width(x))

No bullshit constraints:
- If calibration is not valid, we fail-closed: is_calibrated stays False.
- KNN needs feature scaling; we standardize X using train stats (or cal stats in fallback).
- This provides *marginal* coverage under exchangeability assumptions, not magical online-RL guarantees.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Tuple, Optional, Any

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from atom.logging import get_logger

logger = get_logger("conformal")


def _as_2d_float32(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim >= 3:
        # Safety: conformal operates on feature vectors, not high-rank tensors.
        X = X.reshape(X.shape[0], -1)
    return X


def _as_1d_float32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return y


def _predict_or_raise(model: Any, X: np.ndarray, context: str) -> np.ndarray:
    """
    Base model must support .predict(X)->(N,) or (N,1).
    If it doesn't, calibration must fail-closed.
    """
    try:
        pred = model.predict(X)
    except Exception as e:
        raise RuntimeError(f"{context}: base_model.predict failed: {e}") from e

    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred.reshape(-1)
    elif pred.ndim != 1:
        pred = pred.reshape(-1)

    if pred.shape[0] != X.shape[0]:
        raise RuntimeError(
            f"{context}: base_model.predict returned wrong shape "
            f"(pred={pred.shape}, X={X.shape})."
        )
    return pred


def _canonical_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Canonical split-conformal finite-sample quantile.
    scores: (n_cal,)
    Returns Q >= 0.

    k = ceil((n+1)(1-alpha))
    Q = k-th order statistic (1-indexed), clamped to n
    """
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    s = np.nan_to_num(s, nan=np.inf, posinf=np.inf, neginf=0.0)
    n = s.shape[0]
    if n == 0:
        return float("inf")

    s_sorted = np.sort(s)
    # k is 1-indexed
    k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    k = max(1, min(k, n))
    Q = float(s_sorted[k - 1])
    if not np.isfinite(Q):
        # If calibration is degenerate, do not pretend.
        return float("inf")
    if Q < 0.0:
        Q = 0.0
    return Q


class ConformalWrapper:
    """
    Wraps a regressor (Scientist) to add uncertainty quantification.

    Uses normalized split conformal prediction:
      - fit error estimator r(x) on residuals
      - calibrate Q on normalized scores
      - produce width = 2 * Q * r(x)
      - map width -> trust
    """

    def __init__(self, base_model: Any, calibration_fraction: float = 0.2, alpha: float = 0.1):
        """
        Args:
            base_model: The symbolic model adapter (must implement .predict(X)).
            calibration_fraction: Fraction of data to reserve for calibration (kept for API compatibility).
            alpha: Miscoverage level (0.1 => 90% marginal coverage).
        """
        self.model = base_model
        self.cal_fraction = float(calibration_fraction)
        self.alpha = float(alpha)

        # Residual Error Model (estimates local difficulty)
        # KNN is simple and decent locally, but MUST be used with scaled features.
        n_neighbors = int(os.getenv("ATOM_CONFORMAL_KNN_K", "10"))
        self.error_estimator = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")

        # Feature scaler for KNN distance sanity (critical for mixed feature spaces).
        self.scaler = StandardScaler(with_mean=True, with_std=True)

        # Numerical safety
        self.eps = float(os.getenv("ATOM_CONFORMAL_EPS", "1e-6"))

        # Trust mapping temperature (don’t hardcode “5.0” forever).
        self.trust_lambda = float(os.getenv("ATOM_CONFORMAL_TRUST_LAMBDA", "5.0"))

        self.q_score = float("inf")
        self.is_calibrated = False

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> None:
        """
        Calibrates the error model using a dedicated calibration set.
        Does NOT refit the base model.

        Args:
            X_cal: Calibration features (must not be seen by base model fitting).
            y_cal: Calibration targets.
            X_train: Optional training features used to fit the error estimator on residuals.
            y_train: Optional training targets used to compute residuals.
        """
        # Preserve prior calibration state in case recalibration fails
        prev_calibrated = self.is_calibrated
        prev_q_score = self.q_score

        # Fail-closed by default (will be restored if we bail early)
        self.is_calibrated = False
        self.q_score = float("inf")

        def _restore_prior():
            self.is_calibrated = prev_calibrated
            self.q_score = prev_q_score

        X_cal = _as_2d_float32(X_cal)
        y_cal = _as_1d_float32(y_cal)

        n_cal = X_cal.shape[0]
        if n_cal < 10:
            logger.warning(f"Not enough data for conformal calibration ({n_cal} < 10). Skipping.")
            _restore_prior()
            return
        if y_cal.shape[0] != n_cal:
            logger.error(f"Conformal calibration: X_cal/y_cal mismatch (X_cal={X_cal.shape}, y_cal={y_cal.shape}).")
            _restore_prior()
            return
        if not (0.0 < self.alpha < 1.0):
            logger.error(f"Conformal calibration: alpha must be in (0,1). Got {self.alpha}.")
            _restore_prior()
            return
        if self.model is None:
            logger.error("Conformal calibration: base_model is None.")
            _restore_prior()
            return

        # Predict on calibration set (must succeed; otherwise calibration is meaningless)
        try:
            preds_cal = _predict_or_raise(self.model, X_cal, context="calibrate(X_cal)")
        except Exception as e:
            logger.error(str(e))
            _restore_prior()
            return

        # Fit scaler + error estimator
        if X_train is not None and y_train is not None:
            X_train = _as_2d_float32(X_train)
            y_train = _as_1d_float32(y_train)

            if X_train.shape[0] != y_train.shape[0]:
                logger.error(
                    f"Conformal calibration: X_train/y_train mismatch (X_train={X_train.shape}, y_train={y_train.shape})."
                )
                _restore_prior()
                return
            if X_train.shape[1] != X_cal.shape[1]:
                logger.error(
                    f"Conformal calibration: feature dim mismatch (X_train={X_train.shape[1]}, X_cal={X_cal.shape[1]})."
                )
                _restore_prior()
                return

            try:
                preds_train = _predict_or_raise(self.model, X_train, context="calibrate(X_train)")
            except Exception as e:
                logger.error(str(e))
                _restore_prior()
                return

            residuals_train = np.abs(y_train - preds_train).astype(np.float32)

            # Scale features for KNN
            try:
                self.scaler.fit(X_train)
                X_train_s = self.scaler.transform(X_train)
                X_cal_s = self.scaler.transform(X_cal)
            except Exception as e:
                logger.error(f"Conformal calibration: scaler fit/transform failed: {e}")
                _restore_prior()
                return

            try:
                self.error_estimator.fit(X_train_s, residuals_train)
            except Exception as e:
                logger.error(f"Failed to fit error estimator (train residuals): {e}")
                _restore_prior()
                return

        else:
            # Fallback: fit on calibration itself (less rigorous, but functional). Warn loudly.
            logger.warning("No X_train provided for Error Model fitting. Using X_cal (Suboptimal).")

            residuals_cal_fit = np.abs(y_cal - preds_cal).astype(np.float32)
            try:
                self.scaler.fit(X_cal)
                X_cal_s = self.scaler.transform(X_cal)
            except Exception as e:
                logger.error(f"Conformal calibration: scaler fit/transform failed: {e}")
                _restore_prior()
                return

            try:
                self.error_estimator.fit(X_cal_s, residuals_cal_fit)
            except Exception as e:
                logger.error(f"Failed to fit error estimator (cal residuals): {e}")
                _restore_prior()
                return

        # Compute nonconformity scores on calibration set
        residuals_cal = np.abs(y_cal - preds_cal).astype(np.float32)

        try:
            est_err_cal = self.error_estimator.predict(X_cal_s).astype(np.float32)
        except Exception as e:
            logger.error(f"Conformal calibration: error_estimator.predict failed: {e}")
            _restore_prior()
            return

        est_err_cal = np.maximum(est_err_cal, self.eps)
        scores = residuals_cal / est_err_cal

        # Canonical conformal quantile (finite-sample correct)
        Q = _canonical_conformal_quantile(scores, alpha=self.alpha)
        if not np.isfinite(Q) or Q == float("inf"):
            logger.error("Conformal calibration failed: Q is not finite. Leaving is_calibrated=False.")
            _restore_prior()
            return

        self.q_score = float(Q)
        self.is_calibrated = True

        logger.info(f"Conformal Calibration Complete. Q-Score: {self.q_score:.6g} (N_cal={n_cal}, alpha={self.alpha})")

    def predict_with_trust(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts value, interval width, and trust score.

        Returns:
            y_pred: (N,) prediction
            interval_width: (N,) total uncertainty width
            trust_score: (N,) trust in [0,1]
        """
        X = _as_2d_float32(X)

        if self.model is None:
            # No base model. Fail-closed.
            preds = np.zeros((X.shape[0],), dtype=np.float32)
            width = np.ones_like(preds) * 10.0
            trust = np.zeros_like(preds)
            return preds, width, trust

        # Base prediction must work even if not calibrated.
        try:
            y_pred = _predict_or_raise(self.model, X, context="predict_with_trust")
        except Exception as e:
            logger.error(str(e))
            preds = np.zeros((X.shape[0],), dtype=np.float32)
            width = np.ones_like(preds) * 10.0
            trust = np.zeros_like(preds)
            return preds, width, trust

        if not self.is_calibrated:
            # Default to huge uncertainty and zero trust.
            width = np.ones_like(y_pred, dtype=np.float32) * 10.0
            trust = np.zeros_like(y_pred, dtype=np.float32)
            return y_pred, width, trust

        # Estimate local difficulty r(x) in scaled space
        try:
            X_s = self.scaler.transform(X)
            est_error = self.error_estimator.predict(X_s).astype(np.float32)
        except Exception as e:
            # If error model is broken, fail-closed at inference time too.
            logger.error(f"Conformal predict: error model failed: {e}")
            width = np.ones_like(y_pred, dtype=np.float32) * 10.0
            trust = np.zeros_like(y_pred, dtype=np.float32)
            return y_pred, width, trust

        est_error = np.maximum(est_error, self.eps)

        # Conformal width = 2 * Q * r(x)
        half_width = est_error * float(self.q_score)
        total_width = (2.0 * half_width).astype(np.float32)

        # Trust mapping: exp(-lambda * width) -> [0,1]
        lam = float(self.trust_lambda)
        if lam < 0:
            lam = 0.0
        trust = np.exp(-lam * total_width).astype(np.float32)
        trust = np.clip(trust, 0.0, 1.0)

        return y_pred.astype(np.float32), total_width, trust
