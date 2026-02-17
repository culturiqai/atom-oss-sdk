#!/usr/bin/env python3
"""Export the current FastAPI OpenAPI schema to a versioned JSON artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _canonical(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_openapi() -> Dict[str, Any]:
    try:
        from atom.platform.webapp import create_app
    except Exception as exc:
        raise RuntimeError(
            "Unable to import atom.platform.webapp:create_app. "
            "Install API dependencies first (fastapi, uvicorn, jinja2)."
        ) from exc

    app = create_app()
    return app.openapi()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OpenAPI schema for ATOM platform API")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "docs" / "openapi.json"),
        help="Destination OpenAPI JSON path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify destination already matches current schema (no write)",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = _load_openapi()
    if args.check:
        if not output_path.exists():
            print(f"OPENAPI_MISMATCH: missing file {output_path}")
            return 2
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if _canonical(existing) != _canonical(spec):
            print(f"OPENAPI_MISMATCH: {output_path} is stale")
            return 3
        print(f"OPENAPI_OK: {output_path}")
        return 0

    output_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    print(f"OPENAPI_EXPORTED: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
