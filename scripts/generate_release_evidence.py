#!/usr/bin/env python3
"""Synthesize production evidence into a single release-readiness artifact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
_VALID_PROFILES = {"dev", "ci", "release"}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_valid_ablation_payload(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False
    rows = payload.get("summary", [])
    if not isinstance(rows, list) or not rows:
        return False
    has_atom = any(str(r.get("agent", "")) == "atom" for r in rows if isinstance(r, dict))
    has_baseline = any(
        str(r.get("agent", "")) == "baseline" for r in rows if isinstance(r, dict)
    )
    return bool(has_atom and has_baseline)


def _default_ablation_path() -> Path:
    benchmark_root = ROOT / "benchmark_results"
    preferred = benchmark_root / "platform_ablations" / "summary.json"
    candidates = [
        preferred,
        benchmark_root / "platform_ablations_rc1" / "summary.json",
        benchmark_root / "platform_ablations_gate" / "summary.json",
    ]
    if benchmark_root.exists():
        dynamic = sorted(
            benchmark_root.glob("platform_ablations*/summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in dynamic:
            if path not in candidates:
                candidates.append(path)
    for path in candidates:
        if _is_valid_ablation_payload(_load_json(path)):
            return path
    return preferred


def _resolve_ablation_payload(path: Path) -> Tuple[Optional[Dict[str, Any]], Path]:
    payload = _load_json(path)
    if _is_valid_ablation_payload(payload):
        return payload, path

    fallback_path = _default_ablation_path()
    fallback_payload = _load_json(fallback_path)
    if _is_valid_ablation_payload(fallback_payload):
        return fallback_payload, fallback_path

    return payload, path


def _evaluate_reliability(payload: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    if payload is None:
        return False, {"reason": "missing_reliability_artifact"}
    passed = bool(payload.get("overall_pass", False))
    return passed, {
        "overall_pass": passed,
        "steps": int(payload.get("steps", 0)),
        "seed": int(payload.get("seed", 0)),
    }


def _evaluate_safety(payload: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    if payload is None:
        return False, {"reason": "missing_supersonic_validation_artifact"}
    status = str(payload.get("status", "")).lower()
    passed = status == "passed"
    return passed, {
        "status": status,
        "output": payload.get("output"),
    }


def _evaluate_scientific(payload: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    if payload is None:
        return False, {"reason": "missing_scientific_integrity_artifact"}
    summary = dict(payload.get("summary", {}))
    gates = dict(summary.get("gates", {}))
    passed = bool(gates.get("overall_pass", False))
    return passed, {
        "overall_pass": passed,
        "recovery_rate": float(summary.get("recovery_rate", 0.0)),
        "calibration_error_mean": float(summary.get("calibration_error_mean", 1.0)),
        "intervention_consistency_mean": float(
            summary.get("intervention_consistency_mean", 0.0)
        ),
        "overfit_rate": float(summary.get("overfit_rate", 1.0)),
        "false_discovery_rate": float(summary.get("false_discovery_rate", 1.0)),
        "seed_perturbation_stability": float(
            summary.get("seed_perturbation_stability", 0.0)
        ),
    }


def _evaluate_ablations(
    payload: Optional[Dict[str, Any]],
    min_reward_delta: float,
) -> Tuple[bool, Dict[str, Any]]:
    if payload is None:
        return False, {"reason": "missing_ablation_summary_artifact"}
    rows = list(payload.get("summary", []))
    atom_rows = [r for r in rows if str(r.get("agent", "")) == "atom"]
    baseline_rows = [r for r in rows if str(r.get("agent", "")) == "baseline"]
    if not atom_rows or not baseline_rows:
        return False, {"reason": "ablation_summary_missing_atom_or_baseline_rows"}

    best_atom = max(float(r.get("reward_mean", -1e9)) for r in atom_rows)
    best_baseline = max(float(r.get("reward_mean", -1e9)) for r in baseline_rows)
    reward_delta = float(best_atom - best_baseline)
    passed = reward_delta >= float(min_reward_delta)
    return passed, {
        "overall_pass": passed,
        "best_atom_reward_mean": float(best_atom),
        "best_baseline_reward_mean": float(best_baseline),
        "reward_delta": float(reward_delta),
        "min_reward_delta": float(min_reward_delta),
        "rows": int(len(rows)),
    }


def _normalize_profile(raw: Any) -> str:
    token = str(raw or "dev").strip().lower()
    return token if token in _VALID_PROFILES else "dev"


def _markdown_report(payload: Dict[str, Any]) -> str:
    gates = payload["gates"]
    details = payload["details"]
    lines: List[str] = []
    lines.append("# ATOM Release Evidence")
    lines.append("")
    lines.append(f"- generated_utc: {payload['generated_utc']}")
    lines.append(f"- overall_pass: {payload['overall_pass']}")
    lines.append("")
    lines.append("| gate | pass |")
    lines.append("|---|---|")
    for gate_name in ("reliability", "scientific_integrity", "safety", "ablation_vs_baseline"):
        lines.append(f"| {gate_name} | {bool(gates.get(gate_name, False))} |")
    lines.append("")
    lines.append("## Details")
    for key in ("reliability", "scientific_integrity", "safety", "ablation_vs_baseline"):
        lines.append(f"### {key}")
        lines.append("```json")
        lines.append(json.dumps(details.get(key, {}), indent=2))
        lines.append("```")
    return "\n".join(lines) + "\n"


def generate_release_evidence(args: argparse.Namespace) -> Dict[str, Any]:
    profile = _normalize_profile(getattr(args, "profile", "dev"))
    allow_missing_requested = bool(getattr(args, "allow_missing", False))
    if profile == "release" and allow_missing_requested:
        raise ValueError("allow_missing is not permitted when --profile=release")
    allow_missing_effective = bool(allow_missing_requested and profile != "release")
    requested_min_delta = float(getattr(args, "min_atom_vs_baseline_reward_delta", -0.25))
    effective_min_delta = max(0.0, requested_min_delta) if profile == "release" else requested_min_delta

    reliability_payload = _load_json(Path(args.reliability))
    safety_payload = _load_json(Path(args.safety))
    scientific_payload = _load_json(Path(args.scientific))
    ablation_input_path = Path(args.ablation)
    ablation_payload, ablation_path_used = _resolve_ablation_payload(ablation_input_path)

    reliability_ok, reliability_details = _evaluate_reliability(reliability_payload)
    safety_ok, safety_details = _evaluate_safety(safety_payload)
    scientific_ok, scientific_details = _evaluate_scientific(scientific_payload)
    ablation_ok, ablation_details = _evaluate_ablations(
        ablation_payload,
        min_reward_delta=effective_min_delta,
    )

    gates = {
        "reliability": bool(reliability_ok),
        "scientific_integrity": bool(scientific_ok),
        "safety": bool(safety_ok),
        "ablation_vs_baseline": bool(ablation_ok),
    }

    overall_pass = bool(all(gates.values()))
    if allow_missing_effective:
        # If missing is explicitly allowed, only evaluate available artifacts.
        availability = {
            "reliability": reliability_payload is not None,
            "scientific_integrity": scientific_payload is not None,
            "safety": safety_payload is not None,
            "ablation_vs_baseline": ablation_payload is not None,
        }
        considered = [k for k, v in availability.items() if v]
        if considered:
            overall_pass = bool(all(gates[k] for k in considered))
        else:
            overall_pass = False

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": profile,
        "allow_missing_effective": bool(allow_missing_effective),
        "requested_min_atom_vs_baseline_reward_delta": float(requested_min_delta),
        "effective_min_atom_vs_baseline_reward_delta": float(effective_min_delta),
        "overall_pass": overall_pass,
        "gates": gates,
        "details": {
            "reliability": reliability_details,
            "scientific_integrity": scientific_details,
            "safety": safety_details,
            "ablation_vs_baseline": ablation_details,
        },
        "inputs": {
            "reliability": str(args.reliability),
            "scientific": str(args.scientific),
            "safety": str(args.safety),
            "ablation": str(args.ablation),
            "ablation_resolved": str(ablation_path_used),
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out_md.write_text(_markdown_report(out), encoding="utf-8")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ATOM release evidence report")
    parser.add_argument(
        "--reliability",
        type=str,
        default=str(ROOT / "validation_outputs" / "reliability_kernel_smoke.json"),
    )
    parser.add_argument(
        "--scientific",
        type=str,
        default=str(ROOT / "validation_outputs" / "scientific_integrity.json"),
    )
    parser.add_argument(
        "--safety",
        type=str,
        default=str(ROOT / "validation_outputs" / "supersonic_validation.json"),
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=str(_default_ablation_path()),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(ROOT / "validation_outputs" / "release_evidence.json"),
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(ROOT / "validation_outputs" / "release_evidence.md"),
    )
    parser.add_argument(
        "--min-atom-vs-baseline-reward-delta",
        type=float,
        default=-0.25,
        help="Minimum required (best atom reward_mean - best baseline reward_mean)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="dev",
        choices=sorted(_VALID_PROFILES),
        help="Release profile policy for strictness.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    try:
        out = generate_release_evidence(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "overall_pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "profile": _normalize_profile(getattr(args, "profile", "dev")),
                },
                indent=2,
            )
        )
        return 3
    print(json.dumps(out, indent=2))
    if args.check and not bool(out["overall_pass"]):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
