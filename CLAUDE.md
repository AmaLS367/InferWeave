# Claude Code — InferWeave

@AGENTS.md

Apply the repository-wide engineering, safety, testing, and compatibility rules above.

For each requested change:

- Inspect relevant interfaces, callers, and existing tests before editing.
- Complete the narrowly assigned task, including regression tests and a final adversarial diff review.
- For shared models, runtimes, persistence, or providers, verify downstream consumers rather than assuming the new unit tests are enough.
- Run relevant CI-equivalent checks and report exact outcomes. Never present unexecuted checks as passing.
- Keep commits focused and avoid unrelated changes. Respect the user's branch, push, and cloud-execution instructions.
- Report unresolved risks explicitly and stop at the requested task boundary.
