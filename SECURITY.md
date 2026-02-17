# Security Policy

## Supported Scope

Security review currently covers the repository code and default runtime paths in this project.
No warranty is provided for third-party integrations or modified deployment environments.

## Reporting a Vulnerability

Report vulnerabilities privately to maintainers before public disclosure.

Include:
1. Affected files/components
2. Reproduction steps
3. Impact assessment
4. Suggested fix or mitigation (if available)

Do not open public issues for active vulnerabilities.

## Response Expectations

1. Initial triage acknowledgment target: 72 hours.
2. Severity classification and mitigation plan target: 7 days.
3. Coordinated disclosure once a patch or mitigation is available.

## Safe Deployment Guidance

1. Run with least privilege and isolated credentials.
2. Keep dependency versions pinned and audited.
3. Enable runtime logging and evidence artifact generation.
4. Do not expose experimental endpoints directly to the public internet without authentication.

## High-Stakes Domains

This project is not certified for regulated or life-critical control by default.
Independent safety, verification, and compliance processes are required before such use.
