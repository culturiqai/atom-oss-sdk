#!/usr/bin/env python3
"""Assemble a release packet from ablation and evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

_VALID_PROFILES = {"dev", "ci", "release"}

REQUIRED_ABLATION_ARTIFACTS = [
    "summary.csv",
    "platform_ablation_dashboard.png",
    "platform_ablation_dashboard.pdf",
    "platform_ablation_claims.md",
]

OPTIONAL_ABLATION_ARTIFACTS = [
    "summary.json",
    "manifest.json",
    "runs.json",
    "platform_ablation_plots.json",
    "platform_ablation_significance.json",
]


def _normalize_profile(raw: Any) -> str:
    token = str(raw or "dev").strip().lower()
    return token if token in _VALID_PROFILES else "dev"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_packet(args: argparse.Namespace) -> Dict[str, Any]:
    profile = _normalize_profile(getattr(args, "profile", "dev"))
    allow_missing_requested = bool(getattr(args, "allow_missing", False))
    if profile == "release" and allow_missing_requested:
        raise ValueError("allow_missing is not permitted when --profile=release")

    ablation_dir = Path(args.ablation_dir).resolve()
    release_json = Path(args.release_evidence_json).resolve()
    release_md = Path(args.release_evidence_md).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: List[str] = []
    copied: List[Dict[str, Any]] = []

    def copy_checked(src: Path, rel_name: str, required: bool) -> None:
        if not src.exists():
            if required:
                errors.append(f"missing required artifact: {src}")
            return
        dst = out_dir / rel_name
        _copy_file(src, dst)
        copied.append(
            {
                "name": rel_name,
                "source": str(src),
                "dest": str(dst),
                "size_bytes": int(dst.stat().st_size),
                "sha256": _sha256(dst),
            }
        )

    if not ablation_dir.exists():
        errors.append(f"ablation directory not found: {ablation_dir}")

    for rel in REQUIRED_ABLATION_ARTIFACTS:
        copy_checked(ablation_dir / rel, rel, required=True)
    for rel in OPTIONAL_ABLATION_ARTIFACTS:
        copy_checked(ablation_dir / rel, rel, required=False)

    copy_checked(release_json, "release_evidence.json", required=True)
    copy_checked(release_md, "release_evidence.md", required=True)

    ok = len(errors) == 0
    packet_manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": bool(ok),
        "ablation_dir": str(ablation_dir),
        "release_evidence_json": str(release_json),
        "release_evidence_md": str(release_md),
        "output_dir": str(out_dir),
        "copied_artifacts": copied,
        "errors": errors,
    }
    (out_dir / "release_packet_manifest.json").write_text(
        json.dumps(packet_manifest, indent=2),
        encoding="utf-8",
    )

    if errors and not bool(allow_missing_requested):
        raise RuntimeError("; ".join(errors))
    packet_manifest["profile"] = profile
    return packet_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ATOM release packet")
    parser.add_argument("--ablation-dir", type=str, required=True)
    parser.add_argument("--release-evidence-json", type=str, required=True)
    parser.add_argument("--release-evidence-md", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--profile",
        type=str,
        default="dev",
        choices=sorted(_VALID_PROFILES),
        help="Release profile policy for strictness.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    try:
        payload = build_packet(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 3
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
