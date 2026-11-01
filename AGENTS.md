# Town Model Agent Entrypoint

For every task, start here:

1. Read `docs/CURRENT_PROJECT_STATUS.md`, `docs/ACCEPTANCE_INDEX.md`, and `reviews/classifier_teardown_t5_final_acceptance.md` to establish the current classifier-free baseline, accepted teardown, and hold boundaries. For reconstruction work, use the [canonical A1 ledger](docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) and [readiness frontier](docs/reconstruction/A1_IMPLEMENTATION_READINESS.md); the [completeness audit](docs/reconstruction/A1_DECISION_COMPLETENESS_AUDIT.md) reproduces its source hash and decision locations.
2. Read `docs/agents/issue-tracker.md`, `docs/agents/EXECUTION_PROTOCOL.md`, and `docs/agents/DECISION_AUTHORITY.md`.
3. Verify `git status`, branch, and `git rev-parse HEAD` against the task's starting SHA and approved base; do not discard or overwrite existing work.
4. Find work assigned to you first, otherwise inspect unassigned `workflow:ready` GitHub issues and verify blockers. Never treat an author-decision issue as an implementation task.
5. Check the issue's approved specification/decision links. The classifier inventory is descriptive evidence, not approval to reconstruct a model.
6. Claim before implementation, use a dedicated branch/worktree, and stay within the ticket's scope.
7. Journal claim, investigation, checkpoints, verification, and interruptions as concise issue comments.
8. Submit the exact implementation SHA, test evidence, acceptance-criteria mapping, limitations, and independent-review request.
9. End with a reproducible issue-comment handoff. Only an independent reviewer may accept work; do not close unfinished issues.

The accepted T0–T5 teardown leaves a classifier-free reconstruction foundation; historical Phase 1–3 classifier behavior is not operational. The original canonical A1 source records 70 active approvals for F0–F10; its exact identity and active-ID provenance are indexed in `docs/reconstruction/A1_DECISION_COMPLETENESS_AUDIT.md`. A separate post-R1 addendum records twelve author-reported F11 approvals, for a combined source-level frontier of 82; R1 was independently accepted and integrated at `100d04d4ecae4f58f525bbdac76b87bf28026d5c`, but its review did not cover F11. The F11 repository addendum is tracked by issue #12 and becomes the separately versioned repository authority for F11 only after independent review and integration. Treat the original ledger as authority for F0–F10 and the addendum as the F11 record subject to that condition; the readiness report is a derived dependency map. Issue #11 is a separate post-R1 lifecycle-status task. Unresolved details remain author-controlled. Phase 4, detector development, and later scientific training remain on hold/unimplemented. Load only the scientific-spec sections a task needs.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues; use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Domain docs use the single-context layout. See `docs/agents/domain.md`.
