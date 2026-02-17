# ATOM OSS SDK Release Runbook

## Scope
This runbook defines the release procedure for the OSS SDK (library + CLI + validation tooling).

## Versioning Policy
- Use SemVer (`MAJOR.MINOR.PATCH`).
- `MAJOR`: breaking API/CLI or contract changes.
- `MINOR`: backward-compatible features.
- `PATCH`: backward-compatible fixes.

## Preconditions
1. `docs/SDK_RELEASE.md` gates updated with current state.
2. `CHANGELOG.md` updated for the target version.
3. No open P0/P1 release bugs.
4. CI quality gates green.

## Gate Checklist
1. G-01 Packaging and CLI: pass.
2. G-02 Security baseline: pass.
3. G-03 Config reproducibility: pass.
4. G-04 Scientific integrity: pass.
5. G-05 Reliability kernel: pass.
6. G-06 Safety validation: pass.
7. G-07 Ablation validity: pass.
8. G-08 Release evidence synthesis (release profile): pass.
9. G-09 Docs and OpenAPI freshness: pass.
10. G-10 RC regression: pass.

## Build + Validate
From repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/check_docs_gate.py
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/generate_release_evidence.py --profile release --check --output-json validation_outputs/release_evidence.json --output-md validation_outputs/release_evidence.md
```

## Assemble Release Packet
```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/build_release_packet.py \
  --ablation-dir benchmark_results/platform_ablations \
  --release-evidence-json validation_outputs/release_evidence.json \
  --release-evidence-md validation_outputs/release_evidence.md \
  --output-dir release_packets/v1.0.0 \
  --profile release
```

## Tag + Publish
1. Create annotated tag: `git tag -a vX.Y.Z -m "ATOM OSS SDK vX.Y.Z"`.
2. Push tag.
3. Build artifacts (`sdist`, `wheel`) and publish.
4. Attach release packet and evidence artifacts to release notes.

## Rollback
1. Mark release as revoked in release notes.
2. Publish `PATCH` version with fix or restore previous stable tag.
3. Document root cause and remediation in `CHANGELOG.md`.
