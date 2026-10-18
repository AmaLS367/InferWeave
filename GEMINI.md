# Gemini / Antigravity — Definition of Done

Follow the common project rules in `AGENTS.md`. This file adds stricter execution discipline for Gemini working in Antigravity.

## Small task, fully finished

The owner gives you **small, targeted tasks intentionally**. Your responsibility is to finish **every part of the current task**, including edge cases and validation. Do not expand it into adjacent milestones.

A passing happy-path test or a plausible implementation does **not** mean the task is finished. Anticipate the problems that an independent reviewer would discover. Resolve them **before** committing.

## Mandatory workflow for every coding task

1. **Inspect first.** Read the requested acceptance criteria and affected source files. Locate upstream callers, downstream consumers, validation paths, persistence, error handling, and relevant tests. Identify existing behavior that must remain unchanged.
2. **Make a short checklist.** Cover the task's acceptance criteria, likely failure modes, security implications, compatibility, and tests. Keep it internal or succinct; do not spend the task budget writing long plans.
3. **Implement the complete slice.** Make minimal, coherent edits. Reuse the existing abstractions and conventions. Do not skip an inconvenient acceptance criterion, leave a stub, or silently replace the requested semantics.
4. **Write meaningful tests.** Cover successful use, invalid input, boundaries, security and credential leakage, relevant provider/consumer integration on mocks, and backward compatibility. A test that only checks the object you just constructed is often insufficient.
5. **Execute verification.** Run focused tests, affected regression tests, Ruff, and mypy when applicable. For work that touches shared models, transports, persistence, or runtime specifications, also run the full non-integration suite where available. Use the exact commands in `AGENTS.md` / CI. Fix failures introduced by your changes and repeat affected checks.
6. **Independent adversarial self-review.** Read the final diff as if reviewing another developer's PR. Search for missing requirements, unsafe inputs, secrets in nested metadata or headers, inconsistent fields, broken old APIs/records, mismatched ports, unintended command execution, missing tests, and unrelated edits. Trace important values through every affected layer, not just the new unit test. Fix what you find, then rerun affected checks.
7. **Commit only after the quality gate.** Review `git diff --check`, `git status`, and changed files. If the owner requested a commit/push, do so only after the checks. Do not start the next task.

## Non-negotiable quality gate

Before calling the task complete, verify all of these:

- [ ] Every acceptance criterion is implemented and has evidence.
- [ ] No known bug from the current change remains unaddressed.
- [ ] Relevant existing APIs, historical records, and provider contracts still work.
- [ ] Sensitive information cannot leak through logs, errors, metadata, snapshots, or dry-run output.
- [ ] Edge cases and negative cases are tested; no test was weakened to pass.
- [ ] Required checks were actually run, with passing results, or blockers are honestly documented.
- [ ] Final diff is minimal and reviewed.

If something cannot be verified, **say exactly what is missing**. Do not announce “all done,” “all tests passed,” or “production-ready” based on assumptions. Never invent command output, test counts, or CI status.

## Response after completing a task

Use a concise evidence-based report:

- **Implemented:** what changed and which acceptance criteria it covers.
- **Verification:** exact commands and outcomes (pass/fail/skipped/not run).
- **Self-review:** important issues discovered and fixed; remaining risks.
- **Git:** branch, commit SHA if created; whether changes were pushed.

A reviewer should receive a clean, independently reviewed change instead of discovering elementary issues after each commit. Your goal is first-pass quality, even when the task is deliberately narrow.
