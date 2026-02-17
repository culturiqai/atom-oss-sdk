#!/usr/bin/env python3
"""Documentation quality gate for command and API freshness checks."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DOC_FILES = [
    ROOT / "README.md",
    ROOT / "docs" / "PLATFORM_API_UI.md",
]

REQUIRED_ENDPOINT_MARKERS = [
    "/healthz",
    "/api/v1/worlds",
    "/api/v1/studio/catalog",
    "/api/v1/studio/director-pack",
    "/api/v1/runtime/telemetry",
    "/api/v1/inverse-design/jobs",
    "/api/v1/challenges/supersonic/jobs",
    "/api/v1/challenges/supersonic/jobs/{job_id}/telemetry",
    "/api/v1/assistant/query",
]


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _extract_bash_blocks(text: str) -> List[str]:
    return re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL)


def _split_commands(block: str) -> List[str]:
    commands: List[str] = []
    buf = ""
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            buf += line[:-1].rstrip() + " "
            continue
        full = (buf + line).strip()
        buf = ""
        if full:
            commands.append(full)
    if buf.strip():
        commands.append(buf.strip())
    return commands


def _extract_command_tokens(cmd: str) -> List[str]:
    tokens = shlex.split(cmd, posix=True)
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
        tokens = tokens[1:]
    return tokens


def _check_command_blocks(doc_path: Path, text: str, errors: List[str]) -> None:
    for block in _extract_bash_blocks(text):
        for cmd in _split_commands(block):
            if cmd.startswith("--"):
                # Some docs include option-only snippets as reference; skip parse gate.
                continue
            try:
                tokens = _extract_command_tokens(cmd)
            except Exception as exc:
                errors.append(f"{doc_path}: shell parse failed for command `{cmd}` ({exc})")
                continue
            if not tokens:
                continue
            for token in tokens:
                if token.startswith("scripts/") or token.startswith("docs/") or token.startswith("src/"):
                    if not (ROOT / token).exists():
                        errors.append(f"{doc_path}: referenced path does not exist `{token}`")


def _check_no_user_absolute_paths(doc_path: Path, text: str, errors: List[str]) -> None:
    if "/Users/tiger/" in text:
        errors.append(
            f"{doc_path}: contains machine-specific path '/Users/tiger/'. "
            "Use workspace-relative paths in docs."
        )


def _check_endpoint_markers(doc_path: Path, text: str, errors: List[str]) -> None:
    if doc_path.name != "PLATFORM_API_UI.md":
        return
    for marker in REQUIRED_ENDPOINT_MARKERS:
        if marker not in text:
            errors.append(f"{doc_path}: missing endpoint marker `{marker}`")


def _check_openapi_snapshot(errors: List[str]) -> None:
    openapi_path = ROOT / "docs" / "openapi.json"
    if not openapi_path.exists():
        errors.append(f"{openapi_path}: missing OpenAPI artifact. Run scripts/export_openapi.py.")
        return
    try:
        from atom.platform.webapp import create_app
    except Exception as exc:
        errors.append(
            f"OpenAPI import failed ({exc}). Install web dependencies and ensure PYTHONPATH=src."
        )
        return

    try:
        app = create_app()
        current = app.openapi()
        existing = json.loads(openapi_path.read_text(encoding="utf-8"))
        if _canonical(existing) != _canonical(current):
            errors.append(
                f"{openapi_path}: stale snapshot. Regenerate with `python3 scripts/export_openapi.py`."
            )
    except Exception as exc:
        errors.append(f"OpenAPI snapshot check failed: {exc}")


def run_docs_gate() -> Tuple[bool, List[str]]:
    errors: List[str] = []
    for path in DOC_FILES:
        if not path.exists():
            errors.append(f"missing required documentation file: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        _check_no_user_absolute_paths(path, text, errors)
        _check_command_blocks(path, text, errors)
        _check_endpoint_markers(path, text, errors)
    _check_openapi_snapshot(errors)
    return (len(errors) == 0), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Run documentation gate checks")
    parser.parse_args()

    ok, errors = run_docs_gate()
    if ok:
        print("DOCS_GATE_OK")
        return 0
    print("DOCS_GATE_FAILED")
    for err in errors:
        print(f"- {err}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
