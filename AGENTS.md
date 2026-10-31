# Town Model Agent Entrypoint

For every task, start here:

1. Read `docs/CURRENT_PROJECT_STATUS.md` and `docs/ACCEPTANCE_INDEX.md` to establish the accepted baseline and hold boundaries.
2. Read `docs/agents/issue-tracker.md`, `docs/agents/EXECUTION_PROTOCOL.md`, and `docs/agents/DECISION_AUTHORITY.md`.
3. Verify `git status`, branch, and `git rev-parse HEAD` against the task's starting SHA and approved base; do not discard or overwrite existing work.
4. Find work assigned to you first, otherwise inspect unassigned `workflow:ready` GitHub issues and verify blockers. Never treat an author-decision issue as an implementation task.
5. Check the issue's approved specification/decision links. The classifier inventory is descriptive evidence, not approval to reconstruct a model.
6. Claim before implementation, use a dedicated branch/worktree, and stay within the ticket's scope.
7. Journal claim, investigation, checkpoints, verification, and interruptions as concise issue comments.
8. Submit the exact implementation SHA, test evidence, acceptance-criteria mapping, limitations, and independent-review request.
9. End with a reproducible issue-comment handoff. Only an independent reviewer may accept work; do not close unfinished issues.

Classifier reconstruction is planning-only until the author approves a mathematical specification. Phase 4 remains on hold. Load only the scientific-spec sections a task needs.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues; use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Domain docs use the single-context layout. See `docs/agents/domain.md`.
