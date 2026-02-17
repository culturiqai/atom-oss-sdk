"""Inverse-design DSL, optimizers, and candidate lifecycle orchestration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .contracts import ObjectiveSpec


EPS = 1e-8
LARGE_VIOLATION = 1e6


def _as_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive type conversion
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _normalize_bounds(bounds: Mapping[str, Any]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for name, raw in bounds.items():
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError(f"Bound for '{name}' must be a 2-tuple/list, got {raw!r}")
        low = _as_float(raw[0], f"{name}.low")
        high = _as_float(raw[1], f"{name}.high")
        if low >= high:
            raise ValueError(f"Bound for '{name}' requires low < high, got {raw!r}")
        out[str(name)] = (low, high)
    return out


@dataclass(frozen=True)
class ConstraintClause:
    """Normalized constraint atom for the Objective DSL."""

    metric: str
    op: str
    value: Union[float, Tuple[float, float]]
    weight: float = 1.0


@dataclass
class ConstraintResult:
    """Constraint evaluation output for a metrics dictionary."""

    feasible: bool
    violations: Dict[str, float]
    total_violation: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": bool(self.feasible),
            "violations": {k: float(v) for k, v in self.violations.items()},
            "total_violation": float(self.total_violation),
        }


@dataclass
class DesignCandidate:
    """End-to-end lifecycle object for inverse-design candidates."""

    candidate_id: str
    parameters: Dict[str, float]
    backend: str
    generation: int
    status: str = "generated"
    metrics: Dict[str, float] = field(default_factory=dict)
    raw_objective_score: float = math.inf
    objective_score: float = math.inf
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    total_constraint_violation: float = math.inf
    feasible: bool = False
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": str(self.candidate_id),
            "parameters": {k: float(v) for k, v in self.parameters.items()},
            "backend": str(self.backend),
            "generation": int(self.generation),
            "status": str(self.status),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "raw_objective_score": float(self.raw_objective_score),
            "objective_score": float(self.objective_score),
            "constraint_violations": {
                k: float(v) for k, v in self.constraint_violations.items()
            },
            "total_constraint_violation": float(self.total_constraint_violation),
            "feasible": bool(self.feasible),
            "rank": None if self.rank is None else int(self.rank),
        }


@dataclass
class DesignRunReport:
    """Top-level report for a complete inverse-design run."""

    backend: str
    iterations: int
    population: int
    objective_spec: ObjectiveSpec
    parameter_space: Dict[str, Tuple[float, float]]
    candidates: List[DesignCandidate]
    top_candidates: List[DesignCandidate]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": str(self.backend),
            "iterations": int(self.iterations),
            "population": int(self.population),
            "objective_spec": self.objective_spec.to_dict(),
            "parameter_space": {
                k: [float(v[0]), float(v[1])] for k, v in self.parameter_space.items()
            },
            "candidates": [c.to_dict() for c in self.candidates],
            "top_candidates": [c.to_dict() for c in self.top_candidates],
        }


class ObjectiveDSL:
    """Parser/evaluator for ObjectiveSpec constraints and scalar objective."""

    _OPS = {"<=", ">=", "<", ">", "==", "between"}

    @classmethod
    def normalize_spec(cls, spec: Union[ObjectiveSpec, Mapping[str, Any]]) -> ObjectiveSpec:
        if isinstance(spec, ObjectiveSpec):
            return spec
        if not isinstance(spec, Mapping):
            raise TypeError(f"Objective spec must be ObjectiveSpec or mapping, got {type(spec)}")

        required = ("targets", "constraints", "penalties", "hard_bounds", "solver_budget")
        missing = [k for k in required if k not in spec]
        if missing:
            raise ValueError(f"Objective spec missing required fields: {missing}")

        targets = {str(k): _as_float(v, f"targets.{k}") for k, v in spec["targets"].items()}
        penalties = {
            str(k): _as_float(v, f"penalties.{k}") for k, v in spec["penalties"].items()
        }
        hard_bounds = _normalize_bounds(spec["hard_bounds"])
        constraints = dict(spec["constraints"])
        solver_budget = dict(spec["solver_budget"])

        return ObjectiveSpec(
            targets=targets,
            constraints=constraints,
            penalties=penalties,
            hard_bounds=hard_bounds,
            solver_budget=solver_budget,
        )

    @classmethod
    def compile_constraints(cls, spec: ObjectiveSpec) -> List[ConstraintClause]:
        clauses: List[ConstraintClause] = []
        for metric, raw_clause in spec.constraints.items():
            metric_name = str(metric)
            if isinstance(raw_clause, (float, int)):
                clauses.append(
                    ConstraintClause(
                        metric=metric_name,
                        op="<=",
                        value=_as_float(raw_clause, f"constraints.{metric_name}"),
                    )
                )
                continue

            if not isinstance(raw_clause, Mapping):
                raise ValueError(
                    f"Constraint '{metric_name}' must be numeric or mapping, got {raw_clause!r}"
                )

            weight = _as_float(raw_clause.get("weight", 1.0), f"constraints.{metric_name}.weight")
            if weight < 0.0:
                raise ValueError(f"Constraint '{metric_name}' weight must be non-negative")

            if "op" in raw_clause:
                op = str(raw_clause["op"])
                if op not in cls._OPS:
                    raise ValueError(f"Constraint '{metric_name}' has unsupported op '{op}'")
                if op == "between":
                    between_raw = raw_clause.get("value")
                    if (
                        not isinstance(between_raw, (tuple, list))
                        or len(between_raw) != 2
                    ):
                        raise ValueError(
                            f"Constraint '{metric_name}' between op requires value=[low, high]"
                        )
                    low = _as_float(between_raw[0], f"constraints.{metric_name}.value[0]")
                    high = _as_float(between_raw[1], f"constraints.{metric_name}.value[1]")
                    if low > high:
                        raise ValueError(
                            f"Constraint '{metric_name}' requires low <= high for between"
                        )
                    clauses.append(
                        ConstraintClause(metric=metric_name, op="between", value=(low, high), weight=weight)
                    )
                else:
                    value = _as_float(raw_clause.get("value"), f"constraints.{metric_name}.value")
                    clauses.append(
                        ConstraintClause(metric=metric_name, op=op, value=value, weight=weight)
                    )
                continue

            if "between" in raw_clause:
                between = raw_clause["between"]
                if not isinstance(between, (tuple, list)) or len(between) != 2:
                    raise ValueError(
                        f"Constraint '{metric_name}' between requires [low, high]"
                    )
                low = _as_float(between[0], f"constraints.{metric_name}.between[0]")
                high = _as_float(between[1], f"constraints.{metric_name}.between[1]")
                if low > high:
                    raise ValueError(
                        f"Constraint '{metric_name}' requires low <= high for between"
                    )
                clauses.append(
                    ConstraintClause(metric=metric_name, op="between", value=(low, high), weight=weight)
                )
                continue

            clause_count_before = len(clauses)
            if "min" in raw_clause:
                clauses.append(
                    ConstraintClause(
                        metric=metric_name,
                        op=">=",
                        value=_as_float(raw_clause["min"], f"constraints.{metric_name}.min"),
                        weight=weight,
                    )
                )
            if "max" in raw_clause:
                clauses.append(
                    ConstraintClause(
                        metric=metric_name,
                        op="<=",
                        value=_as_float(raw_clause["max"], f"constraints.{metric_name}.max"),
                        weight=weight,
                    )
                )

            for op in ("<=", ">=", "<", ">", "=="):
                if op in raw_clause:
                    clauses.append(
                        ConstraintClause(
                            metric=metric_name,
                            op=op,
                            value=_as_float(raw_clause[op], f"constraints.{metric_name}.{op}"),
                            weight=weight,
                        )
                    )

            if len(clauses) == clause_count_before:
                raise ValueError(
                    f"Constraint '{metric_name}' has no supported bounds/operators"
                )

        return clauses

    @classmethod
    def evaluate_constraints(
        cls,
        clauses: Sequence[ConstraintClause],
        metrics: Mapping[str, float],
    ) -> ConstraintResult:
        violations: Dict[str, float] = {}
        total = 0.0

        for clause in clauses:
            metric = str(clause.metric)
            if metric not in metrics:
                violation = LARGE_VIOLATION
            else:
                value = _as_float(metrics[metric], f"metrics.{metric}")
                violation = cls._clause_violation(clause, value)

            weighted = max(0.0, float(violation)) * max(float(clause.weight), 0.0)
            if weighted > 0.0:
                violations[metric] = violations.get(metric, 0.0) + weighted
                total += weighted

        return ConstraintResult(
            feasible=(total <= EPS),
            violations=violations,
            total_violation=float(total),
        )

    @staticmethod
    def _clause_violation(clause: ConstraintClause, observed: float) -> float:
        op = str(clause.op)
        value = clause.value
        if op == "<=":
            return max(0.0, observed - float(value))
        if op == "<":
            return max(0.0, observed - float(value) + EPS)
        if op == ">=":
            return max(0.0, float(value) - observed)
        if op == ">":
            return max(0.0, float(value) - observed + EPS)
        if op == "==":
            return abs(observed - float(value))
        if op == "between":
            low, high = value  # type: ignore[misc]
            if observed < low:
                return low - observed
            if observed > high:
                return observed - high
            return 0.0
        raise ValueError(f"Unsupported constraint op: {op}")

    @classmethod
    def score_objective(cls, spec: ObjectiveSpec, metrics: Mapping[str, float]) -> float:
        score = 0.0

        for metric, target in spec.targets.items():
            if metric not in metrics:
                return math.inf
            obs = _as_float(metrics[metric], f"metrics.{metric}")
            target_f = _as_float(target, f"targets.{metric}")
            denom = max(abs(target_f), 1.0)
            score += abs(obs - target_f) / denom

        for metric, weight in spec.penalties.items():
            if metric not in metrics:
                continue
            obs = _as_float(metrics[metric], f"metrics.{metric}")
            score += max(0.0, obs) * _as_float(weight, f"penalties.{metric}")

        return float(score)


class _BaseBackend:
    name = "base"

    def propose(
        self,
        engine: "InverseDesignEngine",
        history: Sequence[DesignCandidate],
        generation: int,
        population: int,
    ) -> List[Dict[str, float]]:
        raise NotImplementedError


class _EvolutionaryBackend(_BaseBackend):
    name = "evolutionary"

    def propose(
        self,
        engine: "InverseDesignEngine",
        history: Sequence[DesignCandidate],
        generation: int,
        population: int,
    ) -> List[Dict[str, float]]:
        if not history:
            return [engine.sample_random_parameters() for _ in range(population)]

        ordered = engine.rank_like(history)
        elite_count = max(2, min(len(ordered), max(2, population // 3)))
        elites = ordered[:elite_count]
        mutation_scale = float(engine.objective_spec.solver_budget.get("mutation_scale", 0.12))
        mutation_scale = max(0.01, mutation_scale * (0.95 ** generation))

        proposals: List[Dict[str, float]] = []
        while len(proposals) < population:
            a = elites[int(engine.rng.integers(0, len(elites)))]
            b = elites[int(engine.rng.integers(0, len(elites)))]
            child: Dict[str, float] = {}
            for name in engine.parameter_names:
                low, high = engine.parameter_space[name]
                span = high - low
                mixed = 0.5 * (a.parameters[name] + b.parameters[name])
                noise = float(engine.rng.normal(0.0, mutation_scale * span))
                child[name] = float(np.clip(mixed + noise, low, high))
            proposals.append(child)

            if engine.rng.random() < 0.20 and len(proposals) < population:
                proposals.append(engine.sample_random_parameters())

        return proposals[:population]


class _GradientBackend(_BaseBackend):
    name = "gradient"

    def propose(
        self,
        engine: "InverseDesignEngine",
        history: Sequence[DesignCandidate],
        generation: int,
        population: int,
    ) -> List[Dict[str, float]]:
        if len(history) < 3:
            return [engine.sample_random_parameters() for _ in range(population)]

        ordered = engine.rank_like(history)
        anchor = ordered[0]
        x_anchor = engine.params_to_vector(anchor.parameters)

        gradient = self._estimate_gradient(engine, ordered[: min(12, len(ordered))], x_anchor)
        norm = float(np.linalg.norm(gradient))

        lr = float(engine.objective_spec.solver_budget.get("learning_rate", 0.25))
        lr = max(0.01, lr * (0.90 ** generation))
        if norm > EPS:
            direction = gradient / norm
        else:
            direction = engine.rng.normal(0.0, 1.0, size=len(engine.parameter_names))
            direction /= max(float(np.linalg.norm(direction)), EPS)

        base = x_anchor - lr * direction
        proposals: List[Dict[str, float]] = [engine.vector_to_params(base)]

        for _ in range(population - 1):
            jitter = engine.rng.normal(0.0, 1.0, size=len(engine.parameter_names))
            jitter /= max(float(np.linalg.norm(jitter)), EPS)
            scaled = x_anchor - lr * direction + 0.30 * lr * jitter
            proposals.append(engine.vector_to_params(scaled))

        return proposals

    @staticmethod
    def _estimate_gradient(
        engine: "InverseDesignEngine",
        candidates: Sequence[DesignCandidate],
        anchor: np.ndarray,
    ) -> np.ndarray:
        if len(candidates) < 2:
            return np.zeros_like(anchor)

        rows: List[np.ndarray] = []
        ys: List[float] = []
        anchor_score = float(candidates[0].objective_score)

        for cand in candidates[1:]:
            rows.append(engine.params_to_vector(cand.parameters) - anchor)
            ys.append(float(cand.objective_score - anchor_score))

        x = np.stack(rows, axis=0)
        y = np.asarray(ys, dtype=np.float64)
        try:
            grad, *_ = np.linalg.lstsq(x, y, rcond=None)
            return grad.astype(np.float64)
        except Exception:
            return np.zeros_like(anchor, dtype=np.float64)


class _BayesianBackend(_BaseBackend):
    name = "bayesian"

    def propose(
        self,
        engine: "InverseDesignEngine",
        history: Sequence[DesignCandidate],
        generation: int,
        population: int,
    ) -> List[Dict[str, float]]:
        if len(history) < 4:
            return [engine.sample_random_parameters() for _ in range(population)]

        x_hist = np.stack([engine.params_to_vector(c.parameters) for c in history], axis=0)
        y_hist = np.asarray([c.objective_score for c in history], dtype=np.float64)
        y_var = float(np.var(y_hist) + EPS)

        pool = max(64, population * 8)
        beta = float(engine.objective_spec.solver_budget.get("bayes_beta", 1.5))
        length_scale = float(engine.objective_spec.solver_budget.get("bayes_length_scale", 0.30))
        length_scale = max(0.05, length_scale)

        acquisitions: List[Tuple[float, Dict[str, float]]] = []
        for _ in range(pool):
            params = engine.sample_random_parameters()
            x = engine.params_to_vector(params)
            mean, var = self._predict(x_hist, y_hist, x, y_var=y_var, length_scale=length_scale)
            lcb = mean - beta * math.sqrt(max(var, EPS))
            acquisitions.append((float(lcb), params))

        acquisitions.sort(key=lambda item: item[0])
        return [params for _, params in acquisitions[:population]]

    @staticmethod
    def _predict(
        x_hist: np.ndarray,
        y_hist: np.ndarray,
        x: np.ndarray,
        *,
        y_var: float,
        length_scale: float,
    ) -> Tuple[float, float]:
        deltas = x_hist - x.reshape(1, -1)
        d2 = np.sum(deltas * deltas, axis=1)
        weights = np.exp(-0.5 * d2 / (length_scale ** 2 + EPS))
        z = float(np.sum(weights))
        if z <= EPS:
            return float(np.mean(y_hist)), float(y_var)

        mean = float(np.sum(weights * y_hist) / z)
        var = float(np.sum(weights * (y_hist - mean) ** 2) / z)
        return mean, max(var, EPS)


class InverseDesignEngine:
    """Coordinates DSL parsing, optimization, verification, ranking, and export."""

    _BACKENDS = {
        "evolutionary": _EvolutionaryBackend,
        "gradient": _GradientBackend,
        "bayesian": _BayesianBackend,
    }

    def __init__(
        self,
        objective_spec: Union[ObjectiveSpec, Mapping[str, Any]],
        simulator: Callable[[Mapping[str, float]], Mapping[str, float]],
        parameter_space: Optional[Mapping[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None,
    ):
        self.objective_spec = ObjectiveDSL.normalize_spec(objective_spec)
        self.simulator = simulator
        self.parameter_space = self._resolve_parameter_space(parameter_space)
        self.parameter_names = list(self.parameter_space.keys())
        self.constraint_clauses = ObjectiveDSL.compile_constraints(self.objective_spec)

        seed_raw = (
            seed
            if seed is not None
            else int(self.objective_spec.solver_budget.get("seed", 0))
        )
        self.rng = np.random.default_rng(int(seed_raw))

        self._candidate_counter = 0
        self.last_report: Optional[DesignRunReport] = None

    def _resolve_parameter_space(
        self, parameter_space: Optional[Mapping[str, Tuple[float, float]]]
    ) -> Dict[str, Tuple[float, float]]:
        if parameter_space is not None:
            return _normalize_bounds(parameter_space)
        if self.objective_spec.hard_bounds:
            return _normalize_bounds(self.objective_spec.hard_bounds)
        raise ValueError(
            "InverseDesignEngine requires parameter_space or ObjectiveSpec.hard_bounds"
        )

    def sample_random_parameters(self) -> Dict[str, float]:
        return {
            name: float(self.rng.uniform(low, high))
            for name, (low, high) in self.parameter_space.items()
        }

    def params_to_vector(self, params: Mapping[str, float]) -> np.ndarray:
        return np.asarray(
            [_as_float(params[name], f"parameters.{name}") for name in self.parameter_names],
            dtype=np.float64,
        )

    def vector_to_params(self, vector: Sequence[float]) -> Dict[str, float]:
        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vec.shape[0] != len(self.parameter_names):
            raise ValueError(
                f"Expected vector of size {len(self.parameter_names)}, got {vec.shape[0]}"
            )
        out: Dict[str, float] = {}
        for idx, name in enumerate(self.parameter_names):
            low, high = self.parameter_space[name]
            out[name] = float(np.clip(vec[idx], low, high))
        return out

    def rank_like(self, candidates: Sequence[DesignCandidate]) -> List[DesignCandidate]:
        return sorted(
            candidates,
            key=lambda c: (
                0 if c.feasible else 1,
                float(c.objective_score),
                float(c.total_constraint_violation),
                str(c.candidate_id),
            ),
        )

    def _evaluate_candidate(
        self,
        params: Mapping[str, float],
        *,
        backend: str,
        generation: int,
    ) -> DesignCandidate:
        self._candidate_counter += 1
        candidate = DesignCandidate(
            candidate_id=f"cand_{self._candidate_counter:06d}",
            parameters=self.vector_to_params(self.params_to_vector(params)),
            backend=backend,
            generation=int(generation),
            status="generated",
        )

        try:
            metrics_raw = self.simulator(candidate.parameters)
            if not isinstance(metrics_raw, Mapping):
                raise TypeError("Simulator must return a mapping of metric -> float")

            metrics = {
                str(k): _as_float(v, f"simulator.metric.{k}")
                for k, v in metrics_raw.items()
            }
            candidate.metrics = metrics
            candidate.status = "simulated"

            c_result = ObjectiveDSL.evaluate_constraints(self.constraint_clauses, metrics)
            raw_score = ObjectiveDSL.score_objective(self.objective_spec, metrics)
            candidate.raw_objective_score = float(raw_score)
            candidate.constraint_violations = dict(c_result.violations)
            candidate.total_constraint_violation = float(c_result.total_violation)
            candidate.feasible = bool(c_result.feasible and math.isfinite(raw_score))
            if candidate.feasible:
                candidate.objective_score = float(raw_score)
            else:
                penalty = 100.0 * candidate.total_constraint_violation
                candidate.objective_score = float(raw_score + penalty)
            candidate.status = "verified"
        except Exception:
            candidate.metrics = {}
            candidate.raw_objective_score = math.inf
            candidate.objective_score = math.inf
            candidate.constraint_violations = {"runtime_failure": float(LARGE_VIOLATION)}
            candidate.total_constraint_violation = float(LARGE_VIOLATION)
            candidate.feasible = False
            candidate.status = "failed"
        return candidate

    def run(
        self,
        *,
        backend: str = "evolutionary",
        iterations: Optional[int] = None,
        population: Optional[int] = None,
        top_k: int = 5,
    ) -> DesignRunReport:
        backend_name = str(backend).strip().lower()
        if backend_name not in self._BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Supported: {sorted(self._BACKENDS.keys())}"
            )

        n_iter = int(
            iterations
            if iterations is not None
            else int(self.objective_spec.solver_budget.get("iterations", 16))
        )
        n_pop = int(
            population
            if population is not None
            else int(self.objective_spec.solver_budget.get("population", 12))
        )
        if n_iter < 1 or n_pop < 1:
            raise ValueError("iterations and population must be >= 1")

        backend_impl = self._BACKENDS[backend_name]()
        history: List[DesignCandidate] = []

        for generation in range(n_iter):
            proposals = backend_impl.propose(self, history, generation, n_pop)
            for params in proposals[:n_pop]:
                candidate = self._evaluate_candidate(
                    params=params,
                    backend=backend_name,
                    generation=generation,
                )
                history.append(candidate)

        ranked = self.rank_like(history)
        k = max(1, int(top_k))
        top = ranked[:k]
        for idx, candidate in enumerate(ranked, start=1):
            candidate.rank = idx
            if candidate.status != "failed":
                candidate.status = "ranked"

        report = DesignRunReport(
            backend=backend_name,
            iterations=n_iter,
            population=n_pop,
            objective_spec=self.objective_spec,
            parameter_space=dict(self.parameter_space),
            candidates=ranked,
            top_candidates=top,
        )
        self.last_report = report
        return report

    def export_report(
        self,
        report: Optional[DesignRunReport],
        path: Union[str, Path],
    ) -> Path:
        payload = report or self.last_report
        if payload is None:
            raise ValueError("No design report available to export")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload.to_dict(), indent=2), encoding="utf-8")

        for candidate in payload.top_candidates:
            if candidate.status != "failed":
                candidate.status = "exported"

        return out_path
