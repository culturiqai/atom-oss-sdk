# Contributing to ATOM

This project is run as a reliability-first scientific systems program.
Contributions are welcome when they improve correctness, reproducibility, and operator safety.

## License + contribution terms (Apache-2.0)

- This project is licensed under **Apache-2.0**.
- By contributing, you agree that your contribution is licensed under Apache-2.0 ("inbound=outbound").
- No CLA is required. For maximum industry adoption, we use a DCO-style sign-off:
  - Please sign commits: `git commit -s` (adds a `Signed-off-by:` trailer).
  - You must have the right to submit the code (authorship or permission).

## 1. Development Setup

From repo root:

```bash
./scripts/setup_venv.sh
venv/bin/python --version
```

## 2. Required Checks Before PR

Run all three:

```bash
PYTHONDONTWRITEBYTECODE=1 scripts/run_phase0_gate.sh
```

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/export_openapi.py --check --output docs/openapi.json
```

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python scripts/check_docs_gate.py
```

If you changed ablation or release-evidence logic, include updated artifacts from:
- `benchmark_results/...`
- `validation_outputs/release_evidence.json`
- `validation_outputs/release_evidence.md`

## 3. Contribution Rules

1. Keep changes deterministic under fixed seeds.
2. Preserve typed contracts (`TheoryPacket`, `HypothesisRecord`, `ControlEnvelope`, `ObjectiveSpec`).
3. Do not bypass runtime safety defaults unless your change is explicitly safety-reviewed.
4. Add/adjust tests for any behavior change.
5. Keep docs and OpenAPI in sync with implementation.

## 4. Commit and PR Guidance

1. Keep PRs focused and reviewable.
2. Include:
- Problem statement
- Design/implementation summary
- Validation evidence (tests + gates)
- Risk assessment and rollback plan

## 5. High-Stakes Use Notice

ATOM is simulation-first software and is not a certified flight/medical/industrial controller.
Do not deploy in high-stakes production environments without independent validation and
domain-specific assurance processes.
