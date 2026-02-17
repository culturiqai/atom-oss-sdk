"""Unit tests for inverse-design DSL and optimization lifecycle."""

from pathlib import Path

import pytest

from atom.platform.contracts import ObjectiveSpec
from atom.platform.inverse_design import InverseDesignEngine, ObjectiveDSL


def _toy_simulator(params):
    x = float(params["shape_a"])
    y = float(params["shape_b"])
    drag = (x - 0.2) ** 2 + 0.1 * abs(y)
    lift = 1.2 - (x - 0.7) ** 2 - 0.2 * (y ** 2)
    mass = 0.4 + 0.3 * x + 0.1 * abs(y)
    return {"drag": drag, "lift": lift, "mass": mass}


def _objective_spec():
    return ObjectiveSpec(
        targets={"drag": 0.08, "lift": 1.05},
        constraints={
            "lift": {"min": 0.80},
            "drag": {"max": 0.40},
            "mass": {"between": [0.35, 0.85]},
        },
        penalties={"mass": 0.05},
        hard_bounds={"shape_a": (0.0, 1.0), "shape_b": (-1.0, 1.0)},
        solver_budget={"iterations": 4, "population": 8, "seed": 7},
    )


def test_objective_dsl_compiles_and_evaluates_mixed_constraints():
    spec = _objective_spec()
    clauses = ObjectiveDSL.compile_constraints(spec)
    assert len(clauses) >= 3

    passing = {"drag": 0.2, "lift": 0.95, "mass": 0.5}
    result_ok = ObjectiveDSL.evaluate_constraints(clauses, passing)
    assert result_ok.feasible is True
    assert result_ok.total_violation == 0.0

    failing = {"drag": 0.5, "lift": 0.5, "mass": 1.0}
    result_bad = ObjectiveDSL.evaluate_constraints(clauses, failing)
    assert result_bad.feasible is False
    assert result_bad.total_violation > 0.0
    assert "drag" in result_bad.violations
    assert "lift" in result_bad.violations
    assert "mass" in result_bad.violations


def test_inverse_design_engine_evolutionary_lifecycle_and_export(tmp_path):
    spec = _objective_spec()
    engine = InverseDesignEngine(spec, simulator=_toy_simulator)
    report = engine.run(backend="evolutionary", top_k=3)

    assert report.backend == "evolutionary"
    assert report.candidates
    assert report.top_candidates
    assert report.top_candidates[0].rank == 1
    assert report.top_candidates[0].status == "ranked"

    export_path = engine.export_report(report, tmp_path / "inverse_design_report.json")
    assert export_path == Path(tmp_path / "inverse_design_report.json")
    assert export_path.exists()
    assert report.top_candidates[0].status == "exported"


def test_inverse_design_engine_runs_gradient_and_bayesian_backends():
    spec = _objective_spec()
    engine = InverseDesignEngine(spec, simulator=_toy_simulator)

    gradient_report = engine.run(backend="gradient", iterations=3, population=6, top_k=2)
    bayes_report = engine.run(backend="bayesian", iterations=3, population=6, top_k=2)

    assert gradient_report.top_candidates
    assert bayes_report.top_candidates

    for report in (gradient_report, bayes_report):
        for cand in report.top_candidates:
            low_a, high_a = spec.hard_bounds["shape_a"]
            low_b, high_b = spec.hard_bounds["shape_b"]
            assert low_a <= cand.parameters["shape_a"] <= high_a
            assert low_b <= cand.parameters["shape_b"] <= high_b


def test_objective_dsl_rejects_constraint_without_operator():
    spec = ObjectiveSpec(
        targets={"drag": 0.1},
        constraints={"drag": {"weight": 2.0}},
        penalties={},
        hard_bounds={"shape_a": (0.0, 1.0)},
        solver_budget={"iterations": 2, "population": 2, "seed": 3},
    )
    with pytest.raises(ValueError, match="no supported bounds/operators"):
        ObjectiveDSL.compile_constraints(spec)


def test_inverse_design_engine_deterministic_with_fixed_seed():
    spec = _objective_spec()
    run_kwargs = {
        "backend": "evolutionary",
        "iterations": 3,
        "population": 6,
        "top_k": 3,
    }
    engine_a = InverseDesignEngine(spec, simulator=_toy_simulator, seed=17)
    engine_b = InverseDesignEngine(spec, simulator=_toy_simulator, seed=17)

    report_a = engine_a.run(**run_kwargs)
    report_b = engine_b.run(**run_kwargs)

    assert [c.parameters for c in report_a.top_candidates] == [
        c.parameters for c in report_b.top_candidates
    ]
    assert [c.objective_score for c in report_a.top_candidates] == [
        c.objective_score for c in report_b.top_candidates
    ]


def test_inverse_design_engine_isolates_candidate_failures():
    spec = _objective_spec()
    calls = {"n": 0}

    def _flaky_simulator(params):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("synthetic simulator failure")
        return _toy_simulator(params)

    engine = InverseDesignEngine(spec, simulator=_flaky_simulator, seed=5)
    report = engine.run(backend="evolutionary", iterations=3, population=6, top_k=3)

    assert len(report.candidates) == 18
    assert any(c.status == "failed" for c in report.candidates)
    assert all(c.rank is not None for c in report.candidates)
    assert report.top_candidates
    assert all(c.status in {"ranked", "failed"} for c in report.top_candidates)
