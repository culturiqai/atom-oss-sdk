#!/usr/bin/env python3
"""Scientific integrity benchmark harness for ATOM law discovery.

Evaluates discovery quality on:
- Synthetic known-law datasets (with noise)
- Synthetic null datasets (false-discovery control)
- Seed perturbation stability across repeated trials

Outputs:
- JSON report with per-case diagnostics and gate status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atom.mind.scientist import _heavy_think_global, _trust_from_score  # noqa: E402

try:
    from sympy import lambdify, sympify, symbols
except Exception as exc:  # pragma: no cover - dependency issue
    raise RuntimeError("sympy is required for scientific integrity benchmark") from exc


FEATURE_NAMES = ["action", "mean_speed", "turbulence"] + [f"latent_{i}" for i in range(8)]
DEFAULT_SCI_SEEDS = [0, 1, 2]
DEFAULT_SCI_NOISE = [0.0, 0.03, 0.06]
DEFAULT_SCI_SAMPLES = 256
DEFAULT_NULL_TRIALS_PER_SEED = 4
FAST_JAX_SCI_PROFILE = {
    "seeds": [0, 1],
    "noise_levels": [0.0, 0.03],
    "samples": 128,
    "null_trials_per_seed": 2,
}

LAW_LIBRARY = [
    {
        "name": "physics_linear",
        "equation": "0.55*action + 3.2*mean_speed - 1.8*turbulence",
        "support": ["action", "mean_speed", "turbulence"],
    },
    {
        "name": "mixed_interaction",
        "equation": "1.2*mean_speed + 0.35*latent_2 - 0.25*latent_2*turbulence",
        "support": ["mean_speed", "latent_2", "turbulence"],
    },
    {
        "name": "latent_nonlinear",
        "equation": "0.8*mean_speed + 0.7*sin(latent_1) - 0.35*latent_4",
        "support": ["mean_speed", "latent_1", "latent_4"],
    },
]


@dataclass
class BenchCase:
    law_name: str
    seed: int
    noise: float
    recovered: bool
    true_support: List[str]
    discovered_law: str
    discovered_support: List[str]
    support_jaccard: float
    train_nrmse: float
    val_nrmse: float
    trust: float
    quality: float
    calibration_error: float
    intervention_nrmse: float
    intervention_consistency: float
    overfit: bool
    solver_score: float
    discovery_source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "law_name": self.law_name,
            "seed": int(self.seed),
            "noise": float(self.noise),
            "recovered": bool(self.recovered),
            "true_support": list(self.true_support),
            "discovered_law": str(self.discovered_law),
            "discovered_support": list(self.discovered_support),
            "support_jaccard": float(self.support_jaccard),
            "train_nrmse": float(self.train_nrmse),
            "val_nrmse": float(self.val_nrmse),
            "trust": float(self.trust),
            "quality": float(self.quality),
            "calibration_error": float(self.calibration_error),
            "intervention_nrmse": float(self.intervention_nrmse),
            "intervention_consistency": float(self.intervention_consistency),
            "overfit": bool(self.overfit),
            "solver_score": float(self.solver_score),
            "discovery_source": str(self.discovery_source),
        }


def _split_indices(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 1:
        idx = np.arange(max(n, 0), dtype=np.int64)
        return idx, idx
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n).astype(np.int64)
    n_val = max(8, int(round(0.2 * n)))
    n_val = min(n_val, max(1, n - 1))
    return idx[n_val:], idx[:n_val]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = {str(v) for v in a}
    sb = {str(v) for v in b}
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb) / len(union))


def _extract_support(law: Optional[str]) -> List[str]:
    if not law:
        return []
    try:
        expr = sympify(str(law))
        out = sorted(str(s) for s in expr.free_symbols)
        return out
    except Exception:
        out: List[str] = []
        text = str(law)
        for name in FEATURE_NAMES:
            if name in text:
                out.append(name)
        return sorted(set(out))


def _compile_truth_laws() -> Dict[str, Any]:
    sym_vars = [symbols(name) for name in FEATURE_NAMES]
    compiled: Dict[str, Any] = {}
    for entry in LAW_LIBRARY:
        expr = sympify(entry["equation"])
        compiled[str(entry["name"])] = lambdify(sym_vars, expr, modules="numpy")
    return compiled


def _evaluate_law(law: str, X: np.ndarray) -> Optional[np.ndarray]:
    try:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(FEATURE_NAMES):
            return None
        sym_vars = [symbols(name) for name in FEATURE_NAMES]
        fn = lambdify(sym_vars, sympify(law), modules="numpy")
        pred = fn(*[X[:, i] for i in range(X.shape[1])])
        pred = np.asarray(pred, dtype=np.float64)
        if pred.ndim == 0:
            pred = np.full((X.shape[0],), float(pred), dtype=np.float64)
        pred = pred.reshape(-1)
        if pred.shape[0] != X.shape[0]:
            return None
        pred = np.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)
        pred = np.clip(pred, -10.0, 10.0)
        return pred.astype(np.float32)
    except Exception:
        return None


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape[0] == 0 or y_pred.shape[0] != y_true.shape[0]:
        return float("inf")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = float(np.std(y_true) + 1e-6)
    return max(0.0, rmse / denom)


def _intervention_consistency(
    *,
    discovered_law: str,
    X_eval: np.ndarray,
    truth_fn: Any,
    action_perturbation: float,
) -> Tuple[float, float]:
    if not discovered_law:
        return float("inf"), 0.0
    X_eval = np.asarray(X_eval, dtype=np.float32)
    if X_eval.ndim != 2 or X_eval.shape[0] == 0 or X_eval.shape[1] != len(FEATURE_NAMES):
        return float("inf"), 0.0

    pred_base = _evaluate_law(discovered_law, X_eval)
    if pred_base is None:
        return float("inf"), 0.0

    X_shift = np.array(X_eval, copy=True)
    X_shift[:, 0] = np.clip(
        X_shift[:, 0] + float(action_perturbation),
        -1.0,
        1.0,
    )
    pred_shift = _evaluate_law(discovered_law, X_shift)
    if pred_shift is None:
        return float("inf"), 0.0

    try:
        true_base = np.asarray(
            truth_fn(*[X_eval[:, i] for i in range(X_eval.shape[1])]),
            dtype=np.float64,
        ).reshape(-1)
        true_shift = np.asarray(
            truth_fn(*[X_shift[:, i] for i in range(X_shift.shape[1])]),
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        return float("inf"), 0.0

    true_delta = (true_shift - true_base).astype(np.float32)
    pred_delta = (pred_shift - pred_base).astype(np.float32)
    delta_nrmse = _nrmse(true_delta, pred_delta)
    consistency = float(np.clip(1.0 - delta_nrmse, 0.0, 1.0))
    return float(delta_nrmse), consistency


def _make_features(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    action = rng.uniform(-1.0, 1.0, size=n)
    mean_speed = np.clip(rng.normal(0.08, 0.02, size=n), 0.0, None)
    turbulence = np.clip(rng.normal(0.03, 0.01, size=n), 0.0, None)
    latents = rng.normal(0.0, 0.6, size=(n, 8))
    return np.column_stack([action, mean_speed, turbulence, latents]).astype(np.float32)


def _run_known_law_cases(
    *,
    seeds: Sequence[int],
    noise_levels: Sequence[float],
    samples: int,
    support_jaccard_threshold: float,
    recovery_nrmse_threshold: float,
    overfit_margin: float,
    action_perturbation: float,
) -> Tuple[List[BenchCase], Dict[Tuple[str, float], List[List[str]]]]:
    compiled = _compile_truth_laws()
    cases: List[BenchCase] = []
    supports_by_group: Dict[Tuple[str, float], List[List[str]]] = {}

    for seed in seeds:
        for law_entry in LAW_LIBRARY:
            law_name = str(law_entry["name"])
            law_fn = compiled[law_name]
            true_support = [str(v) for v in law_entry["support"]]
            for noise in noise_levels:
                case_seed = int(seed * 1000 + abs(hash((law_name, float(noise)))) % 997)
                X = _make_features(case_seed, samples)
                y_clean = np.asarray(
                    law_fn(*[X[:, i] for i in range(X.shape[1])]),
                    dtype=np.float64,
                ).reshape(-1)
                rng = np.random.default_rng(case_seed + 17)
                noise_sigma = float(noise) * float(np.std(y_clean) + 1e-6)
                y = y_clean + rng.normal(0.0, noise_sigma, size=y_clean.shape[0])
                y = y.astype(np.float32)

                law, score, _ = _heavy_think_global(X, y, FEATURE_NAMES)
                discovered_law = str(law) if law else ""
                discovered_support = _extract_support(discovered_law)
                supports_by_group.setdefault((law_name, float(noise)), []).append(
                    list(discovered_support)
                )

                train_idx, val_idx = _split_indices(X.shape[0], seed=case_seed + 31)
                if discovered_law:
                    pred_train = _evaluate_law(discovered_law, X[train_idx])
                    pred_val = _evaluate_law(discovered_law, X[val_idx])
                else:
                    pred_train = None
                    pred_val = None

                if pred_train is None or pred_val is None:
                    train_nrmse = float("inf")
                    val_nrmse = float("inf")
                else:
                    train_nrmse = _nrmse(y[train_idx], pred_train)
                    val_nrmse = _nrmse(y[val_idx], pred_val)

                intervention_nrmse, intervention_consistency = _intervention_consistency(
                    discovered_law=discovered_law,
                    X_eval=X[val_idx],
                    truth_fn=law_fn,
                    action_perturbation=float(action_perturbation),
                )

                solver_score = float(score) if score is not None else float("inf")
                trust = _trust_from_score(solver_score if np.isfinite(solver_score) else 1.0)
                quality = float(np.clip(1.0 - val_nrmse, 0.0, 1.0))
                calibration_error = float(abs(trust - quality))
                support_jaccard = _jaccard(true_support, discovered_support)
                recovered = bool(
                    support_jaccard >= support_jaccard_threshold
                    and val_nrmse <= recovery_nrmse_threshold
                )
                overfit = bool((val_nrmse - train_nrmse) > overfit_margin)

                cases.append(
                    BenchCase(
                        law_name=law_name,
                        seed=int(seed),
                        noise=float(noise),
                        recovered=recovered,
                        true_support=true_support,
                        discovered_law=discovered_law,
                        discovered_support=discovered_support,
                        support_jaccard=float(support_jaccard),
                        train_nrmse=float(train_nrmse),
                        val_nrmse=float(val_nrmse),
                        trust=float(trust),
                        quality=float(quality),
                        calibration_error=float(calibration_error),
                        intervention_nrmse=float(intervention_nrmse),
                        intervention_consistency=float(intervention_consistency),
                        overfit=overfit,
                        solver_score=float(solver_score),
                        discovery_source=("none" if not discovered_law else "worker"),
                    )
                )

    return cases, supports_by_group


def _run_null_trials(
    *,
    seeds: Sequence[int],
    samples: int,
    null_trials_per_seed: int,
    null_acceptance_margin: float,
    null_min_trust: float,
) -> Dict[str, Any]:
    trials: List[Dict[str, Any]] = []
    false_discoveries = 0
    total = 0

    for seed in seeds:
        for trial in range(null_trials_per_seed):
            total += 1
            case_seed = int(seed * 100 + trial)
            X = _make_features(case_seed, samples)
            rng = np.random.default_rng(case_seed + 101)
            y = rng.normal(0.0, 1.0, size=(samples,)).astype(np.float32)

            law, score, _ = _heavy_think_global(X, y, FEATURE_NAMES)
            discovered_law = str(law) if law else ""
            accepted = False
            val_nrmse = float("inf")
            trust = _trust_from_score(float(score) if score is not None else 1.0)

            if discovered_law:
                train_idx, val_idx = _split_indices(samples, seed=case_seed + 7)
                pred_val = _evaluate_law(discovered_law, X[val_idx])
                if pred_val is not None:
                    val_nrmse = _nrmse(y[val_idx], pred_val)
                    baseline = _nrmse(
                        y[val_idx],
                        np.full((len(val_idx),), float(np.mean(y[train_idx])), dtype=np.float32),
                    )
                    accepted = bool(
                        val_nrmse + null_acceptance_margin < baseline
                        and trust >= float(null_min_trust)
                    )

            if accepted:
                false_discoveries += 1

            trials.append(
                {
                    "seed": int(seed),
                    "trial": int(trial),
                    "discovered_law": discovered_law,
                    "solver_score": (
                        float(score) if score is not None and np.isfinite(float(score)) else None
                    ),
                    "trust": float(trust),
                    "val_nrmse": float(val_nrmse),
                    "accepted_discovery": bool(accepted),
                }
            )

    rate = float(false_discoveries / max(1, total))
    return {
        "trials": trials,
        "false_discoveries": int(false_discoveries),
        "total_trials": int(total),
        "false_discovery_rate": rate,
    }


def _seed_stability(
    grouped_supports: Mapping[Tuple[str, float], Sequence[Sequence[str]]],
) -> float:
    scores: List[float] = []
    for _, supports in grouped_supports.items():
        uniq = [set(s) for s in supports]
        if len(uniq) < 2:
            continue
        for a, b in combinations(uniq, 2):
            scores.append(_jaccard(a, b))
    if not scores:
        return 0.0
    return float(np.mean(np.asarray(scores, dtype=np.float64)))


def _apply_profile_overrides(args: argparse.Namespace) -> argparse.Namespace:
    profile = str(getattr(args, "profile", "standard")).strip().lower()
    if profile != "fast_jax":
        return args

    if [int(v) for v in args.seeds] == DEFAULT_SCI_SEEDS:
        args.seeds = list(FAST_JAX_SCI_PROFILE["seeds"])
    if [float(v) for v in args.noise_levels] == DEFAULT_SCI_NOISE:
        args.noise_levels = list(FAST_JAX_SCI_PROFILE["noise_levels"])
    if int(args.samples) == DEFAULT_SCI_SAMPLES:
        args.samples = int(FAST_JAX_SCI_PROFILE["samples"])
    if int(args.null_trials_per_seed) == DEFAULT_NULL_TRIALS_PER_SEED:
        args.null_trials_per_seed = int(FAST_JAX_SCI_PROFILE["null_trials_per_seed"])
    if not bool(args.force_sparse):
        args.force_sparse = True
    return args


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    if args.force_sparse:
        os.environ["ATOM_SCIENTIST_FORCE_SPARSE"] = "1"

    known_cases, grouped_supports = _run_known_law_cases(
        seeds=args.seeds,
        noise_levels=args.noise_levels,
        samples=int(args.samples),
        support_jaccard_threshold=float(args.support_jaccard_threshold),
        recovery_nrmse_threshold=float(args.recovery_nrmse_threshold),
        overfit_margin=float(args.overfit_margin),
        action_perturbation=float(args.action_perturbation),
    )
    null_result = _run_null_trials(
        seeds=args.seeds,
        samples=int(args.samples),
        null_trials_per_seed=int(args.null_trials_per_seed),
        null_acceptance_margin=float(args.null_acceptance_margin),
        null_min_trust=float(args.null_min_trust),
    )

    if known_cases:
        recovery_rate = float(np.mean([1.0 if c.recovered else 0.0 for c in known_cases]))
        calibration_error_mean = float(
            np.mean([float(c.calibration_error) for c in known_cases])
        )
        intervention_consistency_mean = float(
            np.mean([float(c.intervention_consistency) for c in known_cases])
        )
        overfit_rate = float(np.mean([1.0 if c.overfit else 0.0 for c in known_cases]))
    else:
        recovery_rate = 0.0
        calibration_error_mean = float("inf")
        intervention_consistency_mean = 0.0
        overfit_rate = 1.0

    false_discovery_rate = float(null_result["false_discovery_rate"])
    seed_stability = _seed_stability(grouped_supports)

    gates = {
        "recovery_rate": recovery_rate >= float(args.min_recovery_rate),
        "calibration_error_mean": calibration_error_mean <= float(args.max_calibration_error),
        "intervention_consistency_mean": intervention_consistency_mean
        >= float(args.min_intervention_consistency),
        "overfit_rate": overfit_rate <= float(args.max_overfit_rate),
        "false_discovery_rate": false_discovery_rate <= float(args.max_false_discovery_rate),
        "seed_perturbation_stability": seed_stability >= float(args.min_seed_stability),
    }
    gates["overall_pass"] = bool(all(gates.values()))

    summary = {
        "recovery_rate": float(recovery_rate),
        "calibration_error_mean": float(calibration_error_mean),
        "intervention_consistency_mean": float(intervention_consistency_mean),
        "overfit_rate": float(overfit_rate),
        "false_discovery_rate": float(false_discovery_rate),
        "seed_perturbation_stability": float(seed_stability),
        "n_known_cases": int(len(known_cases)),
        "n_null_trials": int(null_result["total_trials"]),
        "gates": gates,
        "thresholds": {
            "min_recovery_rate": float(args.min_recovery_rate),
            "max_calibration_error": float(args.max_calibration_error),
            "min_intervention_consistency": float(args.min_intervention_consistency),
            "max_overfit_rate": float(args.max_overfit_rate),
            "max_false_discovery_rate": float(args.max_false_discovery_rate),
            "min_seed_stability": float(args.min_seed_stability),
        },
    }

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "profile": str(getattr(args, "profile", "standard")),
            "seeds": [int(v) for v in args.seeds],
            "noise_levels": [float(v) for v in args.noise_levels],
            "samples": int(args.samples),
            "force_sparse": bool(args.force_sparse),
            "support_jaccard_threshold": float(args.support_jaccard_threshold),
            "recovery_nrmse_threshold": float(args.recovery_nrmse_threshold),
            "overfit_margin": float(args.overfit_margin),
            "action_perturbation": float(args.action_perturbation),
            "null_trials_per_seed": int(args.null_trials_per_seed),
            "null_acceptance_margin": float(args.null_acceptance_margin),
            "null_min_trust": float(args.null_min_trust),
        },
        "summary": summary,
        "known_cases": [c.to_dict() for c in known_cases],
        "null_trials": null_result["trials"],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scientific integrity benchmark for ATOM law discovery")
    parser.add_argument(
        "--profile",
        type=str,
        default="standard",
        choices=["standard", "fast_jax"],
        help="Benchmark preset. fast_jax reduces synthetic benchmark budget for quicker iteration.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SCI_SEEDS))
    parser.add_argument("--noise-levels", type=float, nargs="+", default=list(DEFAULT_SCI_NOISE))
    parser.add_argument("--samples", type=int, default=DEFAULT_SCI_SAMPLES)
    parser.add_argument("--force-sparse", action="store_true", help="Force sparse discovery backend for reproducibility")
    parser.add_argument("--support-jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--recovery-nrmse-threshold", type=float, default=0.85)
    parser.add_argument("--overfit-margin", type=float, default=0.25)
    parser.add_argument(
        "--action-perturbation",
        type=float,
        default=0.25,
        help="Perturbation size applied to action for interventional consistency checks",
    )
    parser.add_argument("--null-trials-per-seed", type=int, default=DEFAULT_NULL_TRIALS_PER_SEED)
    parser.add_argument("--null-acceptance-margin", type=float, default=0.20)
    parser.add_argument("--null-min-trust", type=float, default=0.20)
    parser.add_argument("--min-recovery-rate", type=float, default=0.55)
    parser.add_argument("--max-calibration-error", type=float, default=0.55)
    parser.add_argument("--min-intervention-consistency", type=float, default=0.35)
    parser.add_argument("--max-overfit-rate", type=float, default=0.50)
    parser.add_argument("--max-false-discovery-rate", type=float, default=0.35)
    parser.add_argument("--min-seed-stability", type=float, default=0.35)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "validation_outputs" / "scientific_integrity.json"),
    )
    parser.add_argument("--check", action="store_true", help="Exit non-zero if any gate fails")
    args = parser.parse_args()
    args = _apply_profile_overrides(args)

    payload = run_benchmark(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    if args.check and not bool(payload["summary"]["gates"]["overall_pass"]):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
