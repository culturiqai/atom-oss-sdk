"""
ATOM MIND: SCIENTIST (Symbolic Reasoner)
----------------------------------------
The System 2 Engine.
Uses Symbolic Regression (PySR) to derive mathematical laws from experience.

Modes:
- Wake: Async background process finding laws in real-time.
- Sleep: Heavy batch processing on Ground Truth data.

NVIDIA-grade notes (no bullshit):
- Wake-mode must not DOS your box. We require new-data deltas + backoff + compute budgeting.
- Trust semantics must be consistent across torch/numpy and across conformal/non-conformal paths.
- Shape contracts are enforced (single vs batch, returns scalar or (B,1)).
- Failures are surfaced via structured counters + logs (no silent death spiral).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple, Union, Any, Dict, Sequence
import os
import threading
import time
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, Future
from pathlib import Path

import numpy as np
import torch
from sympy import symbols, sympify, lambdify

from atom.config import get_config
from atom.exceptions import ScientistError
from atom.logging import get_logger

# Suppress Julia/PySR warnings for clean logs (PySR pulls in Julia under the hood)
warnings.filterwarnings("ignore")

logger = get_logger("scientist")


# ----------------------------
# Helpers (internal)
# ----------------------------

def _now_s() -> float:
    return time.time()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _as_numpy_2d(features: Union[np.ndarray, torch.Tensor, List[float]]) -> np.ndarray:
    """
    Convert features into a 2D numpy array (B, D).
    """
    if isinstance(features, torch.Tensor):
        x = features.detach().cpu().float().numpy()
    elif isinstance(features, np.ndarray):
        x = features
    else:
        x = np.array(features, dtype=np.float32)

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 0:
        # Scalar -> (1,1)
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        # (D,) -> (1,D)
        x = x.reshape(1, -1)
    elif x.ndim >= 3:
        # We do not support high-rank here; caller must flatten.
        # This is a "System 2" feature vector, not an image/field tensor.
        x = x.reshape(x.shape[0], -1)
    return x


def _as_torch_2d(features: torch.Tensor) -> torch.Tensor:
    """
    Ensure torch tensor is float32 2D (B, D).
    """
    x = features
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim >= 3:
        x = x.reshape(x.shape[0], -1)
    if x.dtype != torch.float32:
        x = x.float()
    return x


def _trust_from_score(score: float, k: float = 10.0) -> float:
    """
    Convert a (non-negative, lower-is-better) score into a [0,1] trust scalar.

    Important: PySR exposes multiple columns (loss/score). Your original code treats
    best_law_score as "lower is better" (init inf). Keep that contract.

    trust = exp(-k * score), clamped.
    """
    s = _safe_float(score, default=float("inf"))
    if not np.isfinite(s):
        return 0.0
    if s < 0:
        # Don't let negative scores create trust > 1.0. Clamp.
        s = 0.0
    t = float(np.exp(-k * s))
    if t < 0.0:
        return 0.0
    if t > 1.0:
        return 1.0
    return t


@dataclass
class _DiscoveryCandidate:
    """Internal ranking record for a discovered law candidate."""

    law: str
    source: str
    score_nrmse: float
    adjusted_score: float
    latent_terms: int
    physics_terms: int
    support_size: int


def _clip_score(score: float, default: float = float("inf")) -> float:
    """Coerce score to a finite non-negative float."""
    s = _safe_float(score, default=default)
    if not np.isfinite(s):
        return default
    return max(0.0, float(s))


def _split_train_val_indices(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic train/validation split used for candidate ranking.
    """
    if n_samples <= 1:
        idx = np.arange(max(n_samples, 0), dtype=np.int64)
        return idx, idx

    rng = np.random.default_rng(17)
    idx = rng.permutation(n_samples).astype(np.int64)
    n_val = max(4, int(round(0.2 * n_samples)))
    n_val = min(n_val, max(1, n_samples - 1))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if train_idx.size == 0:
        train_idx = val_idx[:1]
    return train_idx, val_idx


def _law_support_metrics(law_str: str, variable_names: Sequence[str]) -> Tuple[int, int, int]:
    """
    Return (latent_terms, physics_terms, support_size) for a symbolic law.
    """
    support: set[str] = set()
    try:
        expr = sympify(law_str)
        support = {str(s) for s in expr.free_symbols}
    except Exception:
        text = str(law_str)
        for name in variable_names:
            if str(name) in text:
                support.add(str(name))

    latent_terms = sum(1 for n in support if str(n).startswith("latent_"))
    physics_terms = sum(1 for n in support if not str(n).startswith("latent_"))
    return latent_terms, physics_terms, len(support)


def _eval_symbolic_law(
    law_str: str,
    X: np.ndarray,
    variable_names: Sequence[str],
) -> Optional[np.ndarray]:
    """
    Evaluate a symbolic expression on a feature matrix.
    Returns a vector shape (N,) or None when evaluation fails.
    """
    try:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(variable_names):
            return None

        sym_vars = [symbols(str(n)) for n in variable_names]
        expr = sympify(law_str)
        func = lambdify(sym_vars, expr, modules="numpy")

        cols = [X[:, i] for i in range(X.shape[1])]
        pred = func(*cols)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.ndim == 0:
            pred = np.full((X.shape[0],), float(pred), dtype=np.float64)
        else:
            pred = pred.reshape(-1)
        if pred.shape[0] != X.shape[0]:
            return None

        pred = np.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)
        pred = np.clip(pred, -10.0, 10.0)
        return pred.astype(np.float32)
    except Exception:
        return None


def _score_law_candidate(
    law_str: str,
    X: np.ndarray,
    y: np.ndarray,
    variable_names: Sequence[str],
    source: str,
) -> Optional[_DiscoveryCandidate]:
    """
    Compute normalized RMSE and a physics-aware adjusted objective.
    """
    if X.shape[0] < 8:
        return None

    _, val_idx = _split_train_val_indices(X.shape[0])
    X_val = X[val_idx]
    y_val = y[val_idx]

    pred = _eval_symbolic_law(law_str, X_val, variable_names)
    if pred is None:
        return None

    y_std = float(np.std(y_val) + 1e-6)
    rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
    nrmse = max(0.0, rmse / y_std)

    latent_terms, physics_terms, support_size = _law_support_metrics(law_str, variable_names)
    support_den = max(1, support_size)
    latent_frac = float(latent_terms) / float(support_den)
    physics_frac = float(physics_terms) / float(support_den)

    latent_penalty = _safe_float(
        os.getenv("ATOM_SCIENTIST_LATENT_SUPPORT_PENALTY", "0.035"),
        default=0.035,
    )
    physics_bonus = _safe_float(
        os.getenv("ATOM_SCIENTIST_PHYSICS_SUPPORT_BONUS", "0.020"),
        default=0.020,
    )
    complexity_penalty = _safe_float(
        os.getenv("ATOM_SCIENTIST_SUPPORT_COMPLEXITY_PENALTY", "0.002"),
        default=0.002,
    )

    adjusted = (
        float(nrmse)
        + float(latent_penalty) * latent_frac
        - float(physics_bonus) * physics_frac
        + float(complexity_penalty) * float(support_size)
    )
    if source == "pysr":
        # Small tie-break preference for PySR when fits are essentially equal.
        adjusted -= 1e-4

    return _DiscoveryCandidate(
        law=str(law_str),
        source=str(source),
        score_nrmse=float(nrmse),
        adjusted_score=float(adjusted),
        latent_terms=int(latent_terms),
        physics_terms=int(physics_terms),
        support_size=int(support_size),
    )


def _discover_sparse_library_law(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: Sequence[str],
) -> Tuple[Optional[str], Optional[float]]:
    """
    Deterministic sparse-library discovery fallback.
    Produces explicit symbolic equations without PySR dependency.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0] or X.shape[0] < 16 or X.shape[1] == 0:
        return None, None

    max_samples = int(_safe_float(os.getenv("ATOM_SCIENTIST_LIBRARY_MAX_SAMPLES", "2048"), 2048))
    if X.shape[0] > max_samples and max_samples > 32:
        rng = np.random.default_rng(23)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    n, d = X.shape
    train_idx, val_idx = _split_train_val_indices(n)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if X_train.shape[0] < 4 or X_val.shape[0] < 2:
        return None, None

    terms: List[Tuple[str, np.ndarray]] = []

    def _add_term(expr: str, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.shape[0] != n:
            return
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.std(arr)) < 1e-10:
            return
        terms.append((expr, arr))

    # Linear and quadratic terms for all features.
    for i, name in enumerate(variable_names):
        x_i = X[:, i]
        _add_term(str(name), x_i)
        _add_term(f"({name})**2", x_i * x_i)

    # Focus nonlinear and interaction terms on strongest correlations.
    corr_scores: List[Tuple[float, int]] = []
    y_std = float(np.std(y) + 1e-8)
    for i in range(d):
        x_i = X[:, i]
        x_std = float(np.std(x_i) + 1e-8)
        if x_std < 1e-8 or y_std < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x_i, y)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        corr_scores.append((abs(corr), i))
    corr_scores.sort(key=lambda t: t[0], reverse=True)
    top = [idx for _, idx in corr_scores[: min(5, d)]]

    for i in top:
        name = str(variable_names[i])
        x_i = X[:, i]
        _add_term(f"sin({name})", np.sin(x_i))
        _add_term(f"cos({name})", np.cos(x_i))
        _add_term(f"1/(Abs({name}) + 0.05)", 1.0 / (np.abs(x_i) + 0.05))

    for p in range(len(top)):
        for q in range(p + 1, len(top)):
            i, j = top[p], top[q]
            ni, nj = str(variable_names[i]), str(variable_names[j])
            _add_term(f"({ni})*({nj})", X[:, i] * X[:, j])

    if not terms:
        return None, None

    max_terms = int(_safe_float(os.getenv("ATOM_SCIENTIST_LIBRARY_MAX_TERMS", "4"), 4))
    min_improve = _safe_float(os.getenv("ATOM_SCIENTIST_LIBRARY_MIN_IMPROVE", "1e-4"), 1e-4)

    def _fit_eval(selected: List[int]) -> Tuple[float, np.ndarray]:
        cols_train = [np.ones(X_train.shape[0], dtype=np.float64)]
        cols_val = [np.ones(X_val.shape[0], dtype=np.float64)]
        for idx in selected:
            cols_train.append(terms[idx][1][train_idx])
            cols_val.append(terms[idx][1][val_idx])

        phi_train = np.column_stack(cols_train)
        phi_val = np.column_stack(cols_val)
        try:
            coef, _, _, _ = np.linalg.lstsq(phi_train, y_train, rcond=None)
        except Exception:
            coef = np.zeros(phi_train.shape[1], dtype=np.float64)
        pred_val = phi_val @ coef
        rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        nrmse = max(0.0, rmse / float(np.std(y_val) + 1e-6))
        return nrmse, coef

    selected: List[int] = []
    current_score, _ = _fit_eval(selected)
    for _ in range(max_terms):
        best_idx: Optional[int] = None
        best_score = current_score
        for idx in range(len(terms)):
            if idx in selected:
                continue
            score, _ = _fit_eval(selected + [idx])
            if score + float(min_improve) < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        current_score = best_score

    if not selected:
        return None, None

    # Final fit on all sampled data.
    cols_full = [np.ones(n, dtype=np.float64)] + [terms[idx][1] for idx in selected]
    phi_full = np.column_stack(cols_full)
    try:
        coef, _, _, _ = np.linalg.lstsq(phi_full, y, rcond=None)
    except Exception:
        return None, None

    # Build a symbolic string.
    pieces: List[str] = [f"{float(coef[0]):.8g}"]
    for c, idx in zip(coef[1:], selected):
        if abs(float(c)) < 1e-8:
            continue
        expr = terms[idx][0]
        pieces.append(f"({float(c):.8g})*({expr})")
    law = " + ".join(pieces) if pieces else "0.0"

    candidate = _score_law_candidate(law, X.astype(np.float32), y.astype(np.float32), variable_names, "sparse")
    if candidate is None:
        return None, None
    return law, float(candidate.score_nrmse)


def _select_best_candidate(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: Sequence[str],
    candidates: Sequence[Tuple[str, str]],
) -> Optional[_DiscoveryCandidate]:
    """
    Rank candidate equations by holdout fit + support-aware objective.

    Additional policy:
    - If a latent-only equation wins by a small margin, prefer a physics-bearing
      equation that has comparable fit. This reduces brittle "latent shortcut"
      laws in production without forcing hard failures when only latent signal exists.
    """
    scored: List[_DiscoveryCandidate] = []
    for source, law in candidates:
        row = _score_law_candidate(law, X, y, variable_names, source)
        if row is not None:
            scored.append(row)
    if not scored:
        return None
    scored.sort(key=lambda c: c.adjusted_score)

    best = scored[0]
    if best.physics_terms > 0:
        return best

    enforce_physics_priority = (
        str(os.getenv("ATOM_SCIENTIST_ENFORCE_PHYSICS_PRIORITY", "1")).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if not enforce_physics_priority:
        return best

    physics_rows = [row for row in scored if row.physics_terms > 0]
    if not physics_rows:
        return best

    best_physics = min(physics_rows, key=lambda row: row.adjusted_score)
    adjusted_margin = _safe_float(
        os.getenv("ATOM_SCIENTIST_PHYSICS_PRIORITY_MARGIN", "0.03"),
        default=0.03,
    )
    nrmse_margin = _safe_float(
        os.getenv("ATOM_SCIENTIST_PHYSICS_PRIORITY_NRMSE_MARGIN", "0.015"),
        default=0.015,
    )
    adjusted_is_close = (
        float(best_physics.adjusted_score)
        <= float(best.adjusted_score) + max(0.0, float(adjusted_margin))
    )
    nrmse_is_close = (
        float(best_physics.score_nrmse)
        <= float(best.score_nrmse) + max(0.0, float(nrmse_margin))
    )
    if adjusted_is_close and nrmse_is_close:
        return best_physics
    return best


class _SymbolicTorchAdapter:
    """
    Wraps a torch-compiled lambdify function into a sklearn-ish .predict(X)->(N,)
    interface expected by ConformalWrapper.

    ConformalWrapper is CPU/scikit-based in your stack; this adapter
    keeps that boundary explicit and safe.
    """
    def __init__(self, func: Callable[..., Any]):
        self.func = func

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Conformal passes numpy arrays. We run torch on CPU for compatibility.
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_t = torch.from_numpy(X)  # CPU tensor
        # Evaluate: func(*X.T) expects one argument per feature dimension.
        try:
            if X_t.ndim == 1:
                y_t = self.func(*X_t)
            else:
                y_t = self.func(*X_t.T)
        except Exception as e:
            # Hard failure -> return zeros; caller trust should collapse anyway.
            logger.error(f"SymbolicTorchAdapter.predict failed: {e}")
            return np.zeros((X.shape[0],), dtype=np.float32)

        if isinstance(y_t, torch.Tensor):
            y = y_t.detach().cpu().numpy()
        else:
            y = np.array(y_t)

        y = np.asarray(y)
        if y.ndim == 0:
            y = np.full((X.shape[0],), float(y), dtype=np.float32)
        elif y.ndim >= 2:
            # Flatten any (N,1) etc -> (N,)
            y = y.reshape(-1)

        y = np.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        return y


# ----------------------------
# Main class
# ----------------------------

class AtomScientist:
    """Symbolic reasoning engine using PySR for mathematical law discovery."""

    def __init__(
        self,
        variable_names: Optional[List[str]] = None,
        memory_limit: Optional[int] = None,
        wake_interval_s: Optional[float] = None,
    ):
        # --- MEMORY SYSTEMS ---
        if memory_limit is None:
            self.memory_limit = int(os.getenv("ATOM_SCIENTIST_MEMORY_LIMIT", "10000"))
        else:
            self.memory_limit = int(memory_limit)
        self.data_lock = threading.Lock()

        # Short-term buffer for the Wake Cycle
        self.short_term_X: Deque[np.ndarray] = deque(maxlen=self.memory_limit)
        self.short_term_y: Deque[float] = deque(maxlen=self.memory_limit)

        # --- KNOWLEDGE BASE ---
        self.theory_archive: List[Tuple[str, float]] = []
        self.best_law: Optional[str] = None
        self.best_law_score: float = float("inf")

        # The "Reflex": Compiled function for System 1 (Fast)
        self.current_law_func: Optional[Callable[..., Any]] = None

        # Communication Flag
        self.new_discovery_alert: Optional[str] = None

        # --- SYMBOLIC CONFIG ---
        if variable_names:
            self.variable_names = list(variable_names)
        else:
            # Default Atom Variables: Action(1) + FlowStats(3)
            self.variable_names = ["action_z", "mean_vel", "turb_ke", "pressure"]

        self.pysr_initialized = False
        self.model: Optional[Any] = None
        self.conformal: Optional[Any] = None  # initialized lazily

        # --- Wake scheduling controls (fixes your DOS behavior) ---
        self._min_samples = int(os.getenv("ATOM_SCIENTIST_MIN_SAMPLES", "10"))
        self._min_new_samples = int(os.getenv("ATOM_SCIENTIST_MIN_NEW_SAMPLES", "25"))
        if wake_interval_s is None:
            self._cooldown_s_min = float(os.getenv("ATOM_SCIENTIST_COOLDOWN_S_MIN", "2.0"))
        else:
            self._cooldown_s_min = max(float(wake_interval_s), 0.1)
        self._cooldown_s_max = float(os.getenv("ATOM_SCIENTIST_COOLDOWN_S_MAX", "60.0"))
        if self._cooldown_s_max < self._cooldown_s_min:
            self._cooldown_s_max = self._cooldown_s_min
        self._cooldown_s = self._cooldown_s_min
        self._last_submit_time_s = 0.0
        self._last_submit_sample_count = 0
        self._last_success_time_s = 0.0

        # Optional budget: do not run wake-discovery more than N times/hour
        self._budget_per_hour = int(os.getenv("ATOM_SCIENTIST_BUDGET_PER_HOUR", "120"))
        self._budget_window_s = 3600.0
        self._budget_timestamps: Deque[float] = deque(maxlen=max(1, self._budget_per_hour * 2))

        # Where discovered laws are appended (avoid collisions)
        self._laws_path = Path(os.getenv("ATOM_LAWS_PATH", "atom_laws.txt"))

        # --- Health / Metrics ---
        self._stats: Dict[str, float] = {
            "wake_jobs_submitted": 0,
            "wake_jobs_completed": 0,
            "wake_jobs_failed": 0,
            "wake_jobs_skipped_cooldown": 0,
            "wake_jobs_skipped_budget": 0,
            "wake_jobs_skipped_insufficient_data": 0,
            "compile_success": 0,
            "compile_fail": 0,
            "predict_fail": 0,
            "offline_fit_calls": 0,
            "offline_fit_fail": 0,
            "last_wake_score": float("nan"),
            "last_wake_duration_s": float("nan"),
        }

        # --- ASYNC CORE ---
        logger.info("Scientist V3 (Neuro-Symbolic) Online.")
        self.running = True

        # ProcessPool to bypass GIL. PySR is CPU-heavy.
        self.executor = None
        self.current_future: Optional[Future] = None
        self._executor_mode = "process_pool"
        self._executor_error: Optional[str] = None

        try:
            self.executor = ProcessPoolExecutor(max_workers=1)
        except Exception as e:
            self.executor = None
            self._executor_mode = "sync_fallback"
            self._executor_error = str(e)
            logger.warning(
                f"ProcessPool unavailable ({e}). Falling back to synchronous wake jobs."
            )

        # Monitor Thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    # ----------------------------
    # Initialization
    # ----------------------------

    def _init_pysr(self) -> None:
        """Lazy load PySR + ConformalWrapper for offline sleeping."""
        if self.pysr_initialized:
            return

        try:
            from pysr import PySRRegressor
            from atom.mind.conformal import ConformalWrapper

            logger.info("Initializing Symbolic Engine (Stable Mode)...")
            self.model = PySRRegressor(
                niterations=20,
                binary_operators=["+", "*", "-", "/"],
                unary_operators=["abs", "square", "sin", "cos"],
                complexity_of_operators={"sin": 3, "cos": 3, "div": 2, "square": 2},
                constraints={"/": (-1, 9), "square": 6},
                nested_constraints={"square": {"square": 1}},
                parsimony=0.01,
                verbosity=0,
                warm_start=True,
                update=True,
                temp_equation_file=True,
                delete_tempfiles=True,
            )

            self.conformal = ConformalWrapper(base_model=None)
            self.pysr_initialized = True

        except ImportError as e:
            # Keep agent alive; symbolic is optional.
            logger.warning(f"PySR or Conformal missing: {e}. Symbolic reasoning disabled.")
            self.pysr_initialized = False
            self.model = None
            self.conformal = None

    # ----------------------------
    # Wake mode: data ingestion
    # ----------------------------

    def observe(self, features: Union[np.ndarray, torch.Tensor, List[float]], target_metric: float) -> None:
        """WAKE MODE: Record experience (Thread-Safe)."""
        # Convert to numpy 1D feature vector. Do not store GPU tensors in shared state.
        if isinstance(features, torch.Tensor):
            f = features.detach().cpu().float().numpy()
        elif isinstance(features, np.ndarray):
            f = features.astype(np.float32, copy=False)
        else:
            f = np.array(features, dtype=np.float32)

        f = np.asarray(f, dtype=np.float32).reshape(-1)
        y = _safe_float(target_metric, default=0.0)

        with self.data_lock:
            self.short_term_X.append(f)
            self.short_term_y.append(y)

    def ponder(self) -> Optional[str]:
        """NON-BLOCKING Check for new laws."""
        if self.new_discovery_alert:
            discovery = self.new_discovery_alert
            self.new_discovery_alert = None
            return discovery
        return None

    # ----------------------------
    # Wake mode: scheduling + background jobs
    # ----------------------------

    def _budget_allows(self, t_now: float) -> bool:
        # Purge timestamps outside the window
        while self._budget_timestamps and (t_now - self._budget_timestamps[0]) > self._budget_window_s:
            self._budget_timestamps.popleft()
        return len(self._budget_timestamps) < self._budget_per_hour

    def _should_submit_job(self, sample_count: int, t_now: float) -> Tuple[bool, str]:
        if sample_count < self._min_samples:
            return False, "insufficient_data"

        # Require new samples since last submit, otherwise you just rediscover the same crap.
        if (sample_count - self._last_submit_sample_count) < self._min_new_samples and self._last_submit_sample_count > 0:
            return False, "insufficient_new_data"

        # Cooldown/backoff
        if (t_now - self._last_submit_time_s) < self._cooldown_s:
            return False, "cooldown"

        # Budget
        if not self._budget_allows(t_now):
            return False, "budget"

        return True, "ok"

    def _on_job_success(self) -> None:
        self._cooldown_s = self._cooldown_s_min
        self._last_success_time_s = _now_s()

    def _on_job_failure(self) -> None:
        # Exponential backoff on failures.
        self._cooldown_s = min(self._cooldown_s * 2.0, self._cooldown_s_max)

    def _run_wake_job_sync(self, X: np.ndarray, y: np.ndarray) -> None:
        # Run a wake discovery job synchronously when ProcessPool is unavailable.
        law, score, duration_s = _heavy_think_global(X, y, self.variable_names)
        self._stats["wake_jobs_completed"] += 1
        self._stats["last_wake_score"] = _safe_float(score, default=float("nan"))
        self._stats["last_wake_duration_s"] = _safe_float(duration_s, default=float("nan"))

        if law:
            self._process_discovery(law, score)
            self._on_job_success()
        else:
            self._cooldown_s = min(max(self._cooldown_s, self._cooldown_s_min), self._cooldown_s_max)

    def _monitor_loop(self) -> None:
        """
        Lightweight thread that submits jobs to the ProcessPool.

        Critical: do not submit continuously. The old version would run every ~1s forever once N>=10.
        We enforce:
        - min_new_samples delta
        - cooldown/backoff
        - a per-hour budget
        """
        while self.running:
            time.sleep(0.5)  # tighter than 1s, but submissions are gated
            if not self.running:
                return

            # 1) If previous thought is running, do nothing.
            if self.current_future is not None and not self.current_future.done():
                continue

            # 2) If previous thought finished, harvest it.
            if self.current_future is not None and self.current_future.done():
                try:
                    law, score, duration_s = self.current_future.result()
                    self._stats["wake_jobs_completed"] += 1
                    self._stats["last_wake_score"] = _safe_float(score, default=float("nan"))
                    self._stats["last_wake_duration_s"] = _safe_float(duration_s, default=float("nan"))
                    if law:
                        self._process_discovery(law, score)
                        self._on_job_success()
                    else:
                        # No discovery is not a failure; still relax cooldown modestly.
                        self._cooldown_s = min(max(self._cooldown_s, self._cooldown_s_min), self._cooldown_s_max)
                except Exception as e:
                    self._stats["wake_jobs_failed"] += 1
                    logger.error(f"Wake thought failed: {e}")
                    self._on_job_failure()
                finally:
                    self.current_future = None

            # 3) Snapshot data for a new job
            with self.data_lock:
                sample_count = len(self.short_term_X)
                if sample_count < self._min_samples:
                    self._stats["wake_jobs_skipped_insufficient_data"] += 1
                    continue
                X = np.array(self.short_term_X, dtype=np.float32)
                y = np.array(self.short_term_y, dtype=np.float32)

            t_now = _now_s()
            ok, reason = self._should_submit_job(sample_count, t_now)
            if not ok:
                if reason == "cooldown":
                    self._stats["wake_jobs_skipped_cooldown"] += 1
                elif reason == "budget":
                    self._stats["wake_jobs_skipped_budget"] += 1
                else:
                    # insufficient_new_data or insufficient_data (already counted)
                    self._stats["wake_jobs_skipped_insufficient_data"] += 1
                continue

            # 4) Submit job to GLOBAL function (MacOS spawn compatible)
            try:
                if not self.running:
                    continue

                self._budget_timestamps.append(t_now)
                self._last_submit_time_s = t_now
                self._last_submit_sample_count = sample_count
                self._stats["wake_jobs_submitted"] += 1

                if self.executor is None:
                    self._run_wake_job_sync(X, y)
                else:
                    self.current_future = self.executor.submit(
                        _heavy_think_global, X, y, self.variable_names
                    )
            except Exception as e:
                logger.error(f"Wake submit failed: {e}")
                self._stats["wake_jobs_failed"] += 1
                self._on_job_failure()

    # ----------------------------
    # Discovery -> compilation
    # ----------------------------

    def _process_discovery(self, law_str: str, score: float) -> None:
        """Process a new law discovery."""
        if score is None:
            return
        if law_str != self.best_law:
            self._compile_new_law(law_str, score, len(self.variable_names))

    def _evaluate_discovery(self, feature_count: int) -> None:
        """Evaluate the discovered equations (called from fit_offline)."""
        try:
            if self.model is None or (not hasattr(self.model, "equations_")) or self.model.equations_ is None:
                return

            best_row = self.model.get_best()
            new_law_str = str(best_row["sympy_format"])
            score = float(best_row["score"])

            if new_law_str != self.best_law:
                self._compile_new_law(new_law_str, score, feature_count)

        except Exception as e:
            logger.error(f"Offline eval error: {e}")

    def _compile_new_law(self, law_str: str, score: float, feature_count: int) -> None:
        """
        Compile a new mathematical law into executable form.

        Contract:
        - We compile into a torch-compatible callable for fast path inference.
        - We DO NOT assume dimensional correctness beyond feature_count; caller must supply meaningful features.
        """
        try:
            # Ensure variable names match feature_count (otherwise sympify/lambdify will mis-bind)
            if self.variable_names and len(self.variable_names) == feature_count:
                sym_vars = [symbols(n) for n in self.variable_names]
            else:
                # Fall back to x0..x{D-1} if mismatch
                sym_vars = [symbols(f"x{i}") for i in range(feature_count)]

            sym_eq = sympify(law_str)

            # Compile to torch for GPU-to-GPU execution (when called with torch tensors)
            func = lambdify(sym_vars, sym_eq, modules="torch")

            self.current_law_func = func
            self.best_law = law_str
            self.best_law_score = float(score)
            self.theory_archive.append((law_str, float(score)))
            self.new_discovery_alert = law_str

            # Persist discovery (namespaced path via env var)
            try:
                self._laws_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._laws_path, "a") as f:
                    f.write(f"[{time.ctime()}] Score {float(score):.6g}: {law_str}\n")
            except Exception as e_io:
                logger.warning(f"Could not append laws to {self._laws_path}: {e_io}")

            self._stats["compile_success"] += 1
            logger.info(f"Compiled new law: {law_str} (score: {float(score):.6g})")

        except Exception as e:
            self._stats["compile_fail"] += 1
            logger.error(f"Compile error: {e}")
            # Don't raise here; keep the wake thread alive.

    # ----------------------------
    # Sleep mode: offline fit + conformal calibration
    # ----------------------------

    def fit_offline(self, X: np.ndarray, y: np.ndarray) -> None:
        """SLEEP MODE: Deep Thinking on Ground Truth Data."""
        self._stats["offline_fit_calls"] += 1
        logger.info(f"Entering REM Sleep. Dreaming on {len(X)} samples...")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not self.pysr_initialized:
            self._init_pysr()
            if not self.pysr_initialized:
                # PySR unavailable: fallback to deterministic sparse discovery.
                law, score = _discover_sparse_library_law(X, y, self.variable_names)
                if law is not None and score is not None:
                    self._process_discovery(law, score)
                return

        if self.model is None:
            return

        try:
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y mismatch: X={X.shape}, y={y.shape}")

            # Heavier search offline
            self.model.niterations = 50

            # Strict train/cal split for trust calibration
            n_samples = X.shape[0]
            indices = np.random.permutation(n_samples)
            split_idx = int(n_samples * 0.8)

            train_idx, cal_idx = indices[:split_idx], indices[split_idx:]
            if len(cal_idx) < 5:
                # If too small, just don't calibrate trust.
                cal_idx = indices[:0]

            X_train, y_train = X[train_idx], y[train_idx]
            X_cal, y_cal = X[cal_idx], y[cal_idx]

            # Fit on TRAIN only
            # If variable_names mismatch dimensionality, PySR can crash; enforce alignment.
            var_names = self.variable_names
            if var_names is None or len(var_names) != X_train.shape[1]:
                var_names = [f"x{i}" for i in range(X_train.shape[1])]

            self.model.fit(X_train, y_train, variable_names=var_names)
            self._evaluate_discovery(X_train.shape[1])

            # Conformal calibration (optional)
            if self.current_law_func is not None and self.conformal is not None and len(cal_idx) > 0:
                logger.info("Calibrating Conformal Trust Model...")
                self.conformal.model = _SymbolicTorchAdapter(self.current_law_func)

                # Split CP: calibrate on holdout; use train set to fit internal error estimator.
                self.conformal.calibrate(X_cal, y_cal, X_train, y_train)

            logger.info(f"Woke up with theory: {self.best_law}")

        except Exception as e:
            self._stats["offline_fit_fail"] += 1
            logger.error(f"Offline fitting failed: {e}")
            raise ScientistError(f"Offline fitting failed: {e}") from e

    # ----------------------------
    # Inference API
    # ----------------------------

    def predict_theory(self, features: Union[np.ndarray, torch.Tensor, List[float]]) -> Tuple[Any, Any]:
        """
        Generate theory-informed intuition and LOCAL TRUST SCORE.

        Contract:
        - Input: feature vector (D,) or batch (B,D); torch or numpy or list.
        - Output:
            - If single: (scalar_pred, scalar_trust)
            - If batch: (preds(B,1), trust(B,1))
        """
        # No theory -> zeros (shape-correct)
        if self.current_law_func is None:
            if isinstance(features, torch.Tensor):
                x = _as_torch_2d(features)
                dev = x.device
                if features.ndim == 1:
                    return torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)
                return torch.zeros((x.shape[0], 1), device=dev), torch.zeros((x.shape[0], 1), device=dev)

            x = _as_numpy_2d(features)
            if np.asarray(features).ndim == 1:
                return 0.0, 0.0
            return np.zeros((x.shape[0], 1), dtype=np.float32), np.zeros((x.shape[0], 1), dtype=np.float32)

        try:
            # If conformal is calibrated, use it. That’s the only path that has any claim to "local" trust.
            if self.conformal is not None and getattr(self.conformal, "is_calibrated", False):
                X_np = _as_numpy_2d(features)
                preds, _, trust = self.conformal.predict_with_trust(X_np)

                preds = np.asarray(preds, dtype=np.float32).reshape(-1, 1)
                trust = np.asarray(trust, dtype=np.float32).reshape(-1, 1)

                preds = np.clip(preds, -10.0, 10.0)
                trust = np.clip(trust, 0.0, 1.0)

                if isinstance(features, torch.Tensor):
                    dev = features.device
                    th = torch.tensor(preds, dtype=torch.float32, device=dev)
                    tr = torch.tensor(trust, dtype=torch.float32, device=dev)
                    if features.ndim == 1:
                        return th[0, 0], tr[0, 0]
                    return th, tr

                if np.asarray(features).ndim == 1:
                    return float(preds[0, 0]), float(trust[0, 0])
                return preds, trust

            # Otherwise: fast path using compiled torch function.
            global_trust = _trust_from_score(self.best_law_score, k=10.0)

            if isinstance(features, torch.Tensor):
                x = _as_torch_2d(features)
                # Feature dimensionality must match; if not, refuse safely.
                if x.shape[1] != len(self.variable_names):
                    # Don’t crash the agent; return 0 trust and 0 prediction.
                    dev = x.device
                    if features.ndim == 1:
                        return torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)
                    return torch.zeros((x.shape[0], 1), device=dev), torch.zeros((x.shape[0], 1), device=dev)

                if x.ndim == 1:
                    y = self.current_law_func(*x)
                else:
                    y = self.current_law_func(*x.T)

                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, dtype=torch.float32, device=x.device)

                if y.ndim == 0:
                    y = y.reshape(1)
                if y.ndim == 1 and features.ndim > 1:
                    y = y.unsqueeze(-1)

                y = torch.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0)
                y = torch.clamp(y, -10.0, 10.0)

                trust_t = torch.full_like(y, float(global_trust))
                if features.ndim == 1:
                    return y[0], trust_t[0]
                return y, trust_t

            # Numpy path
            X = _as_numpy_2d(features)
            if X.shape[1] != len(self.variable_names):
                if np.asarray(features).ndim == 1:
                    return 0.0, 0.0
                return np.zeros((X.shape[0], 1), dtype=np.float32), np.zeros((X.shape[0], 1), dtype=np.float32)

            # Evaluate; current_law_func expects one argument per feature.
            val = self.current_law_func(*X.T)
            val = np.asarray(val, dtype=np.float32)

            if val.ndim == 0:
                val = np.full((X.shape[0], 1), float(val), dtype=np.float32)
            elif val.ndim == 1:
                val = val.reshape(-1, 1)

            val = np.nan_to_num(val, nan=0.0, posinf=10.0, neginf=-10.0)
            val = np.clip(val, -10.0, 10.0)

            trust = np.full_like(val, float(global_trust), dtype=np.float32)

            if np.asarray(features).ndim == 1:
                return float(val[0, 0]), float(trust[0, 0])
            return val, trust

        except Exception as e:
            self._stats["predict_fail"] += 1
            logger.debug(f"Theory prediction failed: {e}")

            # Safe fallback
            if isinstance(features, torch.Tensor):
                dev = features.device
                if features.ndim == 1:
                    return torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)
                x = _as_torch_2d(features)
                return torch.zeros((x.shape[0], 1), device=dev), torch.zeros((x.shape[0], 1), device=dev)

            X = _as_numpy_2d(features)
            if np.asarray(features).ndim == 1:
                return 0.0, 0.0
            return np.zeros((X.shape[0], 1), dtype=np.float32), np.zeros((X.shape[0], 1), dtype=np.float32)

    # ----------------------------
    # Persistence + status
    # ----------------------------

    def save_laws(self, filepath: Union[str, Path]) -> None:
        """Save discovered laws to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write("# ATOM Discovered Laws\n")
            f.write(f"# Generated at: {time.ctime()}\n\n")

            if self.best_law:
                f.write(f"Best Law (Score: {self.best_law_score:.6g}):\n")
                f.write(f"{self.best_law}\n\n")

            f.write("Theory Archive:\n")
            for i, (law, score) in enumerate(self.theory_archive):
                f.write(f"{i+1}. Score {float(score):.6g}: {law}\n")

        logger.info(f"Laws saved to {filepath}")

    def get_stats(self) -> dict:
        """Get scientist statistics."""
        return {
            "best_law": self.best_law,
            "best_score": self.best_law_score,
            "archive_size": len(self.theory_archive),
            "short_term_samples": len(self.short_term_X),
            "pysr_initialized": self.pysr_initialized,
            "running": self.running,
            "wake_cooldown_s": self._cooldown_s,
            "executor_mode": self._executor_mode,
            "executor_error": self._executor_error,
            **self._stats,
        }

    def get_best_theory(self) -> str:
        """Return the current best mathematical law discovered."""
        return self.best_law if self.best_law else "Searching..."

    def shutdown(self) -> None:
        """Shutdown the scientist cleanly."""
        logger.info("Scientist shutting down...")
        self.running = False

        # Stop monitor thread
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

        # Stop executor
        if hasattr(self, "executor") and self.executor:
            try:
                # Python 3.9+: cancel_futures exists; safe if not.
                self.executor.shutdown(wait=False, cancel_futures=True)  # type: ignore
            except TypeError:
                self.executor.shutdown(wait=False)


def create_scientist_from_config(config: Any = None) -> AtomScientist:
    """Create an AtomScientist instance from the configuration."""
    if config is None:
        config = get_config()
    return AtomScientist(
        variable_names=config.scientist.variable_names,
        memory_limit=getattr(config.scientist, "memory_limit", None),
        wake_interval_s=getattr(config.scientist, "wake_interval", None),
    )


# ----------------------------
# Worker (must be top-level for MacOS spawn compatibility)
# ----------------------------

def _heavy_think_global(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: List[str],
) -> Tuple[Optional[str], Optional[float], float]:
    """
    Standalone global function for multiprocessing.
    Safe for pickling on MacOS/Linux.

    Returns:
        (law_str, score, duration_s)
    """
    t0 = _now_s()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0] or X.shape[0] < 8:
        return None, None, _now_s() - t0

    # Ensure variable_names is clean and aligned.
    if variable_names is not None:
        if hasattr(variable_names, "tolist"):
            variable_names = variable_names.tolist()
        if not isinstance(variable_names, list):
            variable_names = list(variable_names)
    if variable_names is None or len(variable_names) != X.shape[1]:
        variable_names = [f"x{i}" for i in range(X.shape[1])]

    candidate_laws: List[Tuple[str, str]] = []

    # Candidate 1: PySR (if available)
    force_sparse = str(os.getenv("ATOM_SCIENTIST_FORCE_SPARSE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not force_sparse:
        try:
            from pysr import PySRRegressor

            warnings.filterwarnings("ignore")

            # Fresh model each time (process isolation). This is not true warm-start,
            # but it is safe. If you want real incremental, you need a persistent worker.
            model = PySRRegressor(
                niterations=10,
                binary_operators=["+", "*", "-", "/"],
                unary_operators=["abs", "square", "sin", "cos"],
                complexity_of_operators={"sin": 3, "cos": 3, "div": 2, "square": 2},
                verbosity=0,
                warm_start=False,  # don't pretend
                temp_equation_file=False,
            )

            model.fit(X, y, variable_names=variable_names)

            if hasattr(model, "equations_") and model.equations_ is not None:
                best_row = model.get_best()
                law = str(best_row["sympy_format"])
                if law:
                    candidate_laws.append(("pysr", law))
        except Exception:
            # PySR path is optional; we still run deterministic fallback below.
            pass

    # Candidate 2: deterministic sparse library (always available).
    try:
        sparse_law, _ = _discover_sparse_library_law(X, y, variable_names)
        if sparse_law:
            candidate_laws.append(("sparse", sparse_law))
    except Exception:
        pass

    if not candidate_laws:
        return None, None, _now_s() - t0

    best = _select_best_candidate(X, y, variable_names, candidate_laws)
    if best is None:
        return None, None, _now_s() - t0
    return best.law, _clip_score(best.score_nrmse, default=1.0), _now_s() - t0


if __name__ == "__main__":
    sci = AtomScientist(variable_names=["x"])
    X = np.random.rand(100, 1).astype(np.float32) * 10
    y = (X[:, 0] ** 2).astype(np.float32)

    for i in range(100):
        sci.observe(X[i], float(y[i]))

    print("   Waiting for thought...")
    time.sleep(6)

    if sci.best_law:
        print(f"   Discovery: {sci.best_law}")
    sci.shutdown()
