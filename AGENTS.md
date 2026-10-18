# InferWeave — Engineering Instructions for Coding Agents

These rules apply to changes in this repository, regardless of which coding agent is used.
Read the affected implementation, its callers, and its tests before editing.

## Project and scope

- InferWeave is a Python 3.11/3.12 SDK for GPU inference deployments across providers. Source: `src/inferweave/`; tests: `tests/`; documentation: `docs/`.
- Follow the existing architecture: model profiles and registry, runtime templates / `RuntimeSpec`, deployment lifecycle, persistence, transport, and provider adapters.
- Keep tasks small and focused. **Complete the assigned slice end-to-end**: implementation, edge cases, regression tests, and verification. Do not begin the next slice without being asked.
- Prefer targeted changes and existing utilities. Avoid unrelated formatting, refactors, dependency upgrades, generated-file churn, or public API changes.
- InferWeave changes are additive unless a breaking change is explicitly approved. Preserve existing behavior, public imports, serialization, SQLite/JSON records, and legacy deployments. Add defaults for new persistence fields and test old data.

## Security and cloud safety

- Never embed or persist cloud credentials, proxy tokens, passwords, private keys, or secret-bearing headers in profiles, runtime metadata, deployment records, dry-run output, logs, tests, or commit messages.
- Apply existing credential checks across every relevant input path; verify sanitization at persistence and diagnostic boundaries. Use synthetic dummy values in tests.
- Treat URLs, redirects, shell commands, container argv, environment variables, and account identity as security boundaries. Preserve structured argv and refuse unsafe cross-origin forwarding of credentials.
- Cloud provisioning, destructive operations, and billable live integration tests require explicit owner authorization. Use mocks and dry-runs by default.

## Tests and verification

Use the repository's CI configuration (`.github/workflows/ci.yml`) as the authority for exact commands. Where the corresponding dependencies are installed:

```bash
uv lock --check
uv run --locked --only-group dev ruff check .
uv sync --locked --extra modal --extra workers --extra lightning --group dev
uv run --no-sync mypy src
uv run --no-sync pytest -m "not integration" --strict-markers -ra
```

- At minimum, run focused tests for changed behavior plus affected regression suites. Run full non-integration tests and type/lint checks before declaring a completed feature whenever feasible.
- Add negative, boundary, compatibility, and security tests for relevant changes. Unit tests must remain cloud-independent.
- Do not weaken tests, skip assertions, hide failures, or alter test expectations merely to produce a green result. Diagnose and fix the underlying problem.
- Inspect `git diff --check`, `git diff`, and `git status` before committing. Ensure the final diff contains only intended files.
- If a check is unavailable or fails, report its exact status and reason. Never claim tests, live deployments, CI, or reviews succeeded without evidence.

## Change delivery

- Respect the user's requested branch and commit workflow. Never force-push, merge to a protected branch, publish a release, or deploy to a paid cloud account without explicit authorization.
- Report the files changed, decisions, tests with actual results, outstanding risks, and commit SHA when applicable. Clearly distinguish verified results from untested assumptions.
- Stop at the requested task boundary. Do not silently expand the task or describe unimplemented future work as completed.
