"""Unit tests for release packet assembly."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_release_packet as brp


def test_build_release_packet_copies_required_artifacts(tmp_path):
    ablation_dir = tmp_path / "ablations"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    for rel in (
        "summary.csv",
        "platform_ablation_dashboard.png",
        "platform_ablation_dashboard.pdf",
        "platform_ablation_claims.md",
        "summary.json",
        "runs.json",
    ):
        (ablation_dir / rel).write_text("x", encoding="utf-8")

    evidence_json = tmp_path / "release_evidence.json"
    evidence_md = tmp_path / "release_evidence.md"
    evidence_json.write_text("{}", encoding="utf-8")
    evidence_md.write_text("# evidence\n", encoding="utf-8")

    out_dir = tmp_path / "packet"
    args = argparse.Namespace(
        ablation_dir=str(ablation_dir),
        release_evidence_json=str(evidence_json),
        release_evidence_md=str(evidence_md),
        output_dir=str(out_dir),
        profile="dev",
        allow_missing=False,
    )
    payload = brp.build_packet(args)
    assert payload["ok"] is True
    assert payload["profile"] == "dev"
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "platform_ablation_dashboard.png").exists()
    assert (out_dir / "platform_ablation_dashboard.pdf").exists()
    assert (out_dir / "platform_ablation_claims.md").exists()
    assert (out_dir / "release_evidence.json").exists()
    assert (out_dir / "release_evidence.md").exists()
    assert (out_dir / "release_packet_manifest.json").exists()


def test_build_release_packet_rejects_allow_missing_in_release_profile(tmp_path):
    ablation_dir = tmp_path / "ablations"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    # Intentionally omit required ablation artifacts; this should not be maskable in release profile.
    evidence_json = tmp_path / "release_evidence.json"
    evidence_md = tmp_path / "release_evidence.md"
    evidence_json.write_text("{}", encoding="utf-8")
    evidence_md.write_text("# evidence\n", encoding="utf-8")

    args = argparse.Namespace(
        ablation_dir=str(ablation_dir),
        release_evidence_json=str(evidence_json),
        release_evidence_md=str(evidence_md),
        output_dir=str(tmp_path / "packet"),
        profile="release",
        allow_missing=True,
    )

    try:
        brp.build_packet(args)
    except ValueError as exc:
        assert "allow_missing is not permitted" in str(exc)
    else:
        raise AssertionError("Expected release profile to reject allow_missing")
