# Agent Workflow Bootstrap Validation

This file records validation evidence for GitHub issue [#1 — Bootstrap persistent agent workflow for town_model](https://github.com/midhunpradeep/town_model/issues/1). GitHub issue comments remain the append-only execution journal and authoritative work status; this report is evidence, not a second task tracker.

## Baseline and access

- Expected and verified remote baseline: `bb2002f3ef4e4a232816556bcb0d83cb52430a1e` on `maintenance/accepted-phase3-cleanup`.
- Work branch: `infrastructure/agent-workflow`, created directly from that baseline.
- `gh auth status`: authenticated as `midhunpradeep`; `gh repo view` reports Issues enabled and `viewerPermission: ADMIN`.
- Existing tracker had no issues or issue templates. The five canonical Matt Pocock triage labels existed.
- GitHub `gh` is 2.55.0. GraphQL exposes native `Issue.blockedBy` / `blocking` and `parent` / `subIssues`. The native parent-child API was exercised successfully. The newer `gh issue --parent` / `--blocked-by` convenience flags are unavailable in this CLI, so API-compatible commands are documented in `docs/agents/issue-tracker.md`.

## Dry-run path exercised

The bootstrap task itself is the harmless documentation-only workflow example:

1. Issue [#1](https://github.com/midhunpradeep/town_model/issues/1) was created open, unassigned, and labeled `workflow:ready`, with scope, dependencies, observable deliverables, acceptance criteria, and independent-review requirements.
2. It was claimed by assigning `midhunpradeep`, transitioning the single lifecycle label to `workflow:in-progress`, and posting a claim comment with the starting SHA, branch/worktree, scope, dependency check, and criteria. The investigation-complete checkpoint is also in the issue journal.
3. The fresh-agent distinction is represented by the tracker: issue [#3 — `[AUTHOR DECISION] Define classifier membership semantics and town set operations`](https://github.com/midhunpradeep/town_model/issues/3) is open, unassigned, and labeled only `wayfinder:grilling`; it is not an implementation-ready task. It is a child of planning map [#2 — Classifier Reconstruction — Author-Led Mathematical Design](https://github.com/midhunpradeep/town_model/issues/2). No classifier implementation issue was created or marked ready.
4. The native `Issue.blockedBy` query returned an empty blocker list for the bootstrap task and decision ticket. The map's native `subIssues` query returned issue #3. No artificial blocker edge was created just to make the fixture nonempty.
5. Label creation, issue creation, assignee transition, issue comments, native sub-issue linking, and GraphQL reads were executed successfully. All five wayfinder labels and seven mutually exclusive workflow labels were verified on the repo.
6. The final checkpoint and review handoff are recorded in issue #1. Its required final state is `workflow:in-review`, still open and unaccepted until an independent reviewer verifies it. No issue is closed by this validation.

## Scope and limitation

This validates tracker discovery/claim/journaling/hand-off behavior for documentation-only setup. It does not claim that an independent review has occurred. The current harness cannot dispatch the two fresh parallel subagents required by the installed `code-review` skill; review remains a human or separately invoked-agent action. No tests are represented as run unless their commands/results are stated in the issue journal. No model code, scientific artifacts, or historical reports were modified.
