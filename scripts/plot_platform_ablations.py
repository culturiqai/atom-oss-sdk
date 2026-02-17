#!/usr/bin/env python3
"""Generate publication-grade plots and claim summary for platform ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _load_summary(summary_path: Path) -> Dict[str, Any]:
    if not summary_path.exists():
        raise FileNotFoundError(f"summary json not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if "summary" not in payload or not isinstance(payload["summary"], list):
        raise ValueError(f"invalid summary payload at {summary_path}")
    return payload


def _load_runs(runs_path: Optional[Path]) -> List[Dict[str, Any]]:
    if runs_path is None or not runs_path.exists():
        return []
    payload = json.loads(runs_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in runs:
        if isinstance(row, dict):
            out.append(dict(row))
    return out


def _variant_order(rows: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = [
        "v2_hybrid",
        "v2_residual",
        "v2_raw",
        "v1_raw",
        "no_scientist",
        "full",
        "no_symplectic",
        "no_trust_gate",
    ]
    existing = [str(r.get("variant", "")) for r in rows]
    out: List[str] = [v for v in preferred if v in existing]
    for v in sorted(set(existing)):
        if v and v not in out:
            out.append(v)
    return out


def _row_by_variant(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("variant", "")): dict(r) for r in rows}


def _get_metric(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _cohen_d(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    import numpy as np

    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_num = (len(a) - 1) * var_a + (len(b) - 1) * var_b
    pooled_den = len(a) + len(b) - 2
    if pooled_den <= 0:
        return 0.0
    pooled_std = float(np.sqrt(max(pooled_num / pooled_den, 0.0)))
    if pooled_std <= 1e-12:
        return 0.0
    return float((float(np.mean(a)) - float(np.mean(b))) / pooled_std)


def _reward_samples_by_agent_variant(
    runs_rows: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str], List[float]]:
    out: Dict[Tuple[str, str], List[float]] = {}
    for row in runs_rows:
        if not bool(row.get("success", False)):
            continue
        agent = str(row.get("agent", ""))
        variant = str(row.get("variant", ""))
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        try:
            reward = float(metrics.get("final_reward_50", 0.0))
        except Exception:
            continue
        out.setdefault((agent, variant), []).append(reward)
    return out


def _bootstrap_delta_stats(
    atom_rewards: Sequence[float],
    baseline_rewards: Sequence[float],
    *,
    n_samples: int,
    seed: int,
    alpha: float,
) -> Dict[str, Any]:
    import numpy as np

    a = np.asarray(atom_rewards, dtype=np.float64)
    b = np.asarray(baseline_rewards, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return {
            "available": False,
            "reason": "missing_run_samples",
            "n_atom_runs": int(len(a)),
            "n_baseline_runs": int(len(b)),
        }

    n_boot = max(int(n_samples), 200)
    alpha_clamped = min(max(float(alpha), 1e-4), 0.2)
    rng = np.random.default_rng(int(seed))
    diffs = np.empty((n_boot,), dtype=np.float64)
    for idx in range(n_boot):
        a_draw = rng.choice(a, size=len(a), replace=True)
        b_draw = rng.choice(b, size=len(b), replace=True)
        diffs[idx] = float(np.mean(a_draw) - np.mean(b_draw))

    lower = float(np.quantile(diffs, alpha_clamped / 2.0))
    upper = float(np.quantile(diffs, 1.0 - alpha_clamped / 2.0))
    p_improve = float(np.mean(diffs > 0.0))
    p_two_sided = float(2.0 * min(p_improve, 1.0 - p_improve))
    mean_delta = float(np.mean(a) - np.mean(b))
    return {
        "available": True,
        "n_atom_runs": int(len(a)),
        "n_baseline_runs": int(len(b)),
        "mean_delta": mean_delta,
        "ci95_low": lower,
        "ci95_high": upper,
        "p_improve": p_improve,
        "p_two_sided": p_two_sided,
        "cohen_d": float(_cohen_d(a, b)),
        "significant_ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
    }


def _reward_significance(
    *,
    summary_rows: Sequence[Dict[str, Any]],
    runs_rows: Sequence[Dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
    alpha: float,
) -> Dict[str, Any]:
    atom_rows = [r for r in summary_rows if str(r.get("agent", "")) == "atom"]
    baseline_rows = [r for r in summary_rows if str(r.get("agent", "")) == "baseline"]
    if not atom_rows:
        return {"available": False, "reason": "missing_atom_rows"}
    if not baseline_rows:
        return {"available": False, "reason": "missing_baseline_rows"}
    if not runs_rows:
        return {"available": False, "reason": "missing_runs_rows"}

    best_baseline = max(baseline_rows, key=lambda r: _get_metric(r, "reward_mean", -1e9))
    baseline_variant = str(best_baseline.get("variant", ""))
    samples = _reward_samples_by_agent_variant(runs_rows)
    baseline_rewards = list(samples.get(("baseline", baseline_variant), []))
    if not baseline_rewards:
        return {
            "available": False,
            "reason": "missing_baseline_reward_samples",
            "baseline_variant": baseline_variant,
        }

    by_variant: Dict[str, Any] = {}
    for row in atom_rows:
        variant = str(row.get("variant", ""))
        atom_rewards = list(samples.get(("atom", variant), []))
        stats = _bootstrap_delta_stats(
            atom_rewards=atom_rewards,
            baseline_rewards=baseline_rewards,
            n_samples=int(bootstrap_samples),
            seed=int(bootstrap_seed),
            alpha=float(alpha),
        )
        stats["baseline_variant"] = baseline_variant
        by_variant[variant] = stats
    return {
        "available": True,
        "baseline_variant": baseline_variant,
        "variants": by_variant,
        "bootstrap_samples": int(max(int(bootstrap_samples), 200)),
        "alpha": float(alpha),
    }


def _claims_markdown(
    *,
    atom_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
    significance: Optional[Dict[str, Any]] = None,
) -> str:
    atom_map = _row_by_variant(atom_rows)
    lines: List[str] = []
    lines.append("# Platform Ablation Claims")
    lines.append("")

    if atom_rows:
        best_atom = max(atom_rows, key=lambda r: _get_metric(r, "reward_mean", -1e9))
        lines.append(
            f"- Best ATOM variant by reward: `{best_atom.get('variant')}` "
            f"(reward_mean={_get_metric(best_atom, 'reward_mean'):.4f})"
        )
    else:
        lines.append("- No ATOM rows found in summary.")

    if baseline_rows:
        best_base = max(baseline_rows, key=lambda r: _get_metric(r, "reward_mean", -1e9))
        lines.append(
            f"- Best baseline by reward: `{best_base.get('variant')}` "
            f"(reward_mean={_get_metric(best_base, 'reward_mean'):.4f})"
        )
        if atom_rows:
            best_atom = max(atom_rows, key=lambda r: _get_metric(r, "reward_mean", -1e9))
            delta = _get_metric(best_atom, "reward_mean") - _get_metric(best_base, "reward_mean")
            lines.append(f"- ATOM vs baseline reward delta: `{delta:+.4f}`")
    else:
        lines.append("- No baseline rows found (comparison to PPO baseline not established).")

    lines.append("")
    lines.append("## What This Ablation Proves")
    lines.append("- Relative contribution of configured ATOM subsystems under identical world/seed/step budgets.")
    lines.append("- Sensitivity of observed reward/stress/safety metrics to scientist signal design (`v1` vs `v2`, target modes).")
    if baseline_rows:
        lines.append("- Relative performance of ATOM variants against included baselines on the measured simulator envelope.")
    else:
        lines.append("- Baseline-comparison claim is unavailable in this run because no baseline rows were provided.")
    lines.append("")
    lines.append("## What This Does Not Prove")
    lines.append("- Certification-grade safety in real deployment conditions.")
    lines.append("- Generalization beyond tested worlds, seeds, and step budgets.")
    lines.append("- Causal guarantees outside controlled simulator assumptions.")
    lines.append("")

    # Additional concrete checks
    if "no_scientist" in atom_map and any(v in atom_map for v in ("v2_hybrid", "v2_residual", "v2_raw", "full")):
        ref_key = "v2_hybrid" if "v2_hybrid" in atom_map else ("full" if "full" in atom_map else "v2_raw")
        ref = atom_map[ref_key]
        ns = atom_map["no_scientist"]
        reward_delta = _get_metric(ref, "reward_mean") - _get_metric(ns, "reward_mean")
        stress_delta = _get_metric(ref, "stress_mean") - _get_metric(ns, "stress_mean")
        safety_delta = _get_metric(ref, "safety_interventions_mean") - _get_metric(ns, "safety_interventions_mean")
        lines.append(
            f"- Scientist contribution (`{ref_key}` - `no_scientist`): "
            f"reward `{reward_delta:+.4f}`, stress `{stress_delta:+.4f}`, safety interventions `{safety_delta:+.4f}`."
        )
    if "v1_raw" in atom_map and any(v in atom_map for v in ("v2_hybrid", "v2_residual", "v2_raw")):
        ref_key = "v2_hybrid" if "v2_hybrid" in atom_map else "v2_raw"
        reward_delta = _get_metric(atom_map[ref_key], "reward_mean") - _get_metric(atom_map["v1_raw"], "reward_mean")
        stress_delta = _get_metric(atom_map[ref_key], "stress_mean") - _get_metric(atom_map["v1_raw"], "stress_mean")
        safety_delta = _get_metric(atom_map[ref_key], "safety_interventions_mean") - _get_metric(atom_map["v1_raw"], "safety_interventions_mean")
        lines.append(
            f"- Structural signal contribution (`{ref_key}` - `v1_raw`): "
            f"reward `{reward_delta:+.4f}`, stress `{stress_delta:+.4f}`, safety interventions `{safety_delta:+.4f}`."
        )
    if all(v in atom_map for v in ("v2_hybrid", "v2_residual", "v2_raw")):
        rh = _get_metric(atom_map["v2_hybrid"], "reward_mean")
        rr = _get_metric(atom_map["v2_residual"], "reward_mean")
        rw = _get_metric(atom_map["v2_raw"], "reward_mean")
        lines.append(
            f"- Discovery target sensitivity (reward): "
            f"`hybrid={rh:.4f}`, `residual={rr:.4f}`, `raw={rw:.4f}`."
        )
    if atom_rows:
        n_runs_values = [_get_metric(r, "n_runs", 0.0) for r in atom_rows]
        min_runs = int(min(n_runs_values)) if n_runs_values else 0
        max_runs = int(max(n_runs_values)) if n_runs_values else 0
        if max_runs <= 1:
            lines.append("- Statistical power warning: this output is from a single-run smoke regime; treat conclusions as directional.")
        else:
            lines.append(
                f"- Replication coverage: ATOM variants were evaluated with n_runs in the range `{min_runs}`-`{max_runs}`."
            )

    lines.append("")
    lines.append("## Reward Significance vs Baseline")
    if significance and bool(significance.get("available", False)):
        variant_stats = dict(significance.get("variants", {}))
        order = _variant_order(atom_rows)
        for variant in order:
            stats = variant_stats.get(variant)
            if not isinstance(stats, dict):
                continue
            if not bool(stats.get("available", False)):
                lines.append(
                    f"- `{variant}`: significance unavailable ({stats.get('reason', 'unknown_reason')})."
                )
                continue
            lines.append(
                f"- `{variant}` vs `{stats.get('baseline_variant')}`: "
                f"delta `{float(stats.get('mean_delta', 0.0)):+.4f}`, "
                f"95% CI [`{float(stats.get('ci95_low', 0.0)):+.4f}`, `{float(stats.get('ci95_high', 0.0)):+.4f}`], "
                f"Cohen's d `{float(stats.get('cohen_d', 0.0)):+.3f}`, "
                f"P(delta>0) `{float(stats.get('p_improve', 0.0)):.3f}`, "
                f"sig={bool(stats.get('significant_ci_excludes_zero', False))}."
            )
    else:
        reason = (
            str(significance.get("reason", "missing_significance_inputs"))
            if isinstance(significance, dict)
            else "missing_significance_inputs"
        )
        lines.append(f"- Significance unavailable: `{reason}`.")

    return "\n".join(lines) + "\n"


def _plot_dashboard(
    *,
    summary_rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    title: str,
) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    atom_rows = [r for r in summary_rows if str(r.get("agent", "")) == "atom"]
    baseline_rows = [r for r in summary_rows if str(r.get("agent", "")) == "baseline"]

    order = _variant_order(atom_rows + baseline_rows)
    rows = _row_by_variant(atom_rows + baseline_rows)
    labels = order
    x = np.arange(len(labels), dtype=np.float64)

    reward_means = [_get_metric(rows[v], "reward_mean") for v in labels]
    reward_stds = [_get_metric(rows[v], "reward_std") for v in labels]
    stress_means = [_get_metric(rows[v], "stress_mean") for v in labels]
    stress_stds = [_get_metric(rows[v], "stress_std") for v in labels]
    safety_means = [_get_metric(rows[v], "safety_interventions_mean") for v in labels]

    if not labels:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "No summary rows available for plotting.", ha="center", va="center")
        ax.set_axis_off()
        out_png = out_dir / "platform_ablation_dashboard.png"
        out_pdf = out_dir / "platform_ablation_dashboard.pdf"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        return {
            "dashboard_png": str(out_png),
            "dashboard_pdf": str(out_pdf),
            "n_rows": int(len(summary_rows)),
            "n_atom_rows": int(len(atom_rows)),
            "n_baseline_rows": int(len(baseline_rows)),
        }

    colors = []
    for v in labels:
        agent = str(rows[v].get("agent", ""))
        if agent == "baseline":
            colors.append("#7b8794")
        elif v.startswith("v2_"):
            colors.append("#1f77b4")
        elif v.startswith("v1_"):
            colors.append("#ff7f0e")
        elif v == "no_scientist":
            colors.append("#d62728")
        else:
            colors.append("#2ca02c")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.bar(x, reward_means, yerr=reward_stds, capsize=4, color=colors, edgecolor="black", linewidth=1.0)
    ax1.set_title("Reward Mean ± Std (higher is better)")
    ax1.set_ylabel("reward_mean")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(x, stress_means, yerr=stress_stds, capsize=4, color=colors, edgecolor="black", linewidth=1.0)
    ax2.set_title("Stress Mean ± Std (lower is better)")
    ax2.set_ylabel("stress_mean")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.grid(axis="y", alpha=0.25)

    ax3.bar(x, safety_means, color=colors, edgecolor="black", linewidth=1.0)
    ax3.set_title("Safety Interventions Mean (lower is better)")
    ax3.set_ylabel("safety_interventions_mean")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=25, ha="right")
    ax3.grid(axis="y", alpha=0.25)

    atom_only = [r for r in atom_rows]
    if baseline_rows and atom_only:
        best_base = max(baseline_rows, key=lambda r: _get_metric(r, "reward_mean", -1e9))
        base_reward = _get_metric(best_base, "reward_mean")
        atom_order = _variant_order(atom_only)
        atom_deltas = [_get_metric(rows[v], "reward_mean") - base_reward for v in atom_order]
        x_atom = np.arange(len(atom_order), dtype=np.float64)
        ax4.bar(x_atom, atom_deltas, color="#2ca02c", edgecolor="black", linewidth=1.0)
        ax4.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax4.set_title(f"ATOM Reward Delta vs Best Baseline ({best_base.get('variant')})")
        ax4.set_ylabel("reward_mean_delta")
        ax4.set_xticks(x_atom)
        ax4.set_xticklabels(atom_order, rotation=25, ha="right")
        ax4.grid(axis="y", alpha=0.25)
    else:
        ax4.text(
            0.5,
            0.5,
            "Baseline rows missing.\nRun ablations with baselines for delta chart.",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax4.set_axis_off()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = out_dir / "platform_ablation_dashboard.png"
    out_pdf = out_dir / "platform_ablation_dashboard.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "dashboard_png": str(out_png),
        "dashboard_pdf": str(out_pdf),
        "n_rows": int(len(summary_rows)),
        "n_atom_rows": int(len(atom_rows)),
        "n_baseline_rows": int(len(baseline_rows)),
    }


def run_plot_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    summary_path = Path(args.summary)
    summary_payload = _load_summary(summary_path)
    summary_rows = list(summary_payload.get("summary", []))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_path: Optional[Path] = None
    runs_arg = str(getattr(args, "runs", "") or "").strip()
    if runs_arg:
        runs_path = Path(runs_arg)
    else:
        inferred = summary_path.parent / "runs.json"
        if inferred.exists():
            runs_path = inferred
    runs_rows = _load_runs(runs_path)

    plot_meta = _plot_dashboard(
        summary_rows=summary_rows,
        out_dir=out_dir,
        title=str(args.title),
    )

    atom_rows = [r for r in summary_rows if str(r.get("agent", "")) == "atom"]
    baseline_rows = [r for r in summary_rows if str(r.get("agent", "")) == "baseline"]
    significance = _reward_significance(
        summary_rows=summary_rows,
        runs_rows=runs_rows,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        alpha=float(args.alpha),
    )
    claims_md = _claims_markdown(
        atom_rows=atom_rows,
        baseline_rows=baseline_rows,
        significance=significance,
    )

    claims_path = out_dir / "platform_ablation_claims.md"
    claims_path.write_text(claims_md, encoding="utf-8")
    significance_path = out_dir / "platform_ablation_significance.json"
    significance_path.write_text(json.dumps(significance, indent=2), encoding="utf-8")

    payload = {
        "summary_source": str(args.summary),
        "runs_source": str(runs_path) if runs_path is not None else None,
        "plot_meta": plot_meta,
        "significance": significance,
        "claims_markdown": str(claims_path),
        "significance_json": str(significance_path),
    }
    (out_dir / "platform_ablation_plots.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot platform ablation results")
    parser.add_argument(
        "--summary",
        type=str,
        default="benchmark_results/platform_ablations/summary.json",
        help="Path to ablation summary JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/platform_ablations",
        help="Directory to write plot artifacts",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="ATOM Platform Ablation Dashboard",
        help="Figure title",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="",
        help="Optional runs.json for per-seed significance stats (auto-inferred beside summary if omitted)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=4000,
        help="Bootstrap draws per variant for reward delta confidence intervals",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=123,
        help="Seed for bootstrap significance estimation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided significance alpha for CI reporting",
    )
    args = parser.parse_args()

    payload = run_plot_pipeline(args)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
