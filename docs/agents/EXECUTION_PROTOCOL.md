# Execution Protocol

GitHub Issues are the sole durable task tracker. Issue comments are the append-only execution journal; commits, test output summaries, and reports are linked evidence. `docs/CURRENT_PROJECT_STATUS.md` remains milestone-level state and is not updated after every task.

## Start and discover

1. Read `docs/CURRENT_PROJECT_STATUS.md`, `docs/ACCEPTANCE_INDEX.md`, the tracker guidance, and `docs/agents/DECISION_AUTHORITY.md`. Read only the scientific-spec sections relevant to the ticket.
2. Verify the repository, working tree, branch, current SHA, and task's recorded starting SHA. Reconcile existing files/worktrees before touching them.
3. Find work assigned to you first. Otherwise list unassigned `workflow:ready` issues and verify each issue has no open native blockers. Do not treat Wayfinder decision tickets as implementation work.
4. Check the approved specification/decision links and the issue's acceptance criteria. If authority or scope is unclear, stop and ask; do not silently substitute a different task.
5. Confirm all prerequisites are resolved. A task is READY only when it is narrow, implementable from approved requirements, has falsifiable acceptance criteria, and has no open blockers.

Useful discovery commands:

```bash
gh issue list --repo midhunpradeep/town_model --state open --label workflow:ready \
  --json number,title,assignees,url
# For each candidate, query GitHub's native Issue.blockedBy connection using the
# GraphQL query documented in docs/agents/issue-tracker.md.
```

`workflow:ready` is the only implementation-work frontier. A candidate with an assignee or an open blocker is not eligible. The `ready-for-agent` triage label is not a substitute for this check.

## Issue lifecycle and state authority

An implementation issue has exactly one `workflow:*` label at a time. That label is its authoritative lifecycle state; its assignee is its current owner. Do not encode state in multiple labels, free-text status fields, project boards, or a Markdown task list. GitHub open/closed status is a terminal projection: accepted tasks are closed; rejected tasks remain open unless explicitly superseded.

| State | Meaning | Required evidence / next action |
|---|---|---|
| **DEFINED** (`workflow:defined`) | Work is recorded but not eligible to start. | Objective, scope, dependencies, and falsifiable acceptance criteria are being clarified. |
| **READY** (`workflow:ready`) | Approved requirements are clear, prerequisites resolved, and no open blockers remain. | Coordinator verifies the readiness checklist; unassigned work can be claimed. |
| **IN PROGRESS** (`workflow:in-progress`) | One implementer has claimed the task and is working within scope. | Assignee records claim and checkpoints. |
| **IN REVIEW** (`workflow:in-review`) | Implementer submitted a fixed revision for independent review. | Review handoff identifies exact SHA, evidence, criteria, limitations, and reviewer. |
| **ACCEPTED** (`workflow:accepted`) | Independent reviewer verified the acceptance criteria. | Reviewer records verdict/evidence; issue may then be closed. |
| **BLOCKED** (`workflow:blocked`) | Work cannot proceed until a stated condition is met. | Open blocker link(s), prior state, and exact unblock condition are recorded. |
| **REJECTED** (`workflow:rejected`) | The task or submitted approach was rejected and is not accepted. | Record reason and disposition. It remains open unless explicitly superseded by linked work. |

Wayfinder maps and decision tickets are a distinct planning flow: a map carries `wayfinder:map`; each child carries exactly one `wayfinder:<type>` label. Open/unassigned means unclaimed; assignment claims a ticket; native blockers establish its frontier; closing a decision ticket requires an author-approved resolution. Per the Wayfinder skill, these issues do not carry `workflow:*` labels. Use the `[AUTHOR DECISION]` title prefix and explicit approval fields to make unresolved author input visible.

### Allowed transitions and roles

Roles are **author**, **coordinator** (may also be the dispatcher), **implementer**, and **independent reviewer**. The implementer and accepting reviewer must be different people/agents. GitHub permissions do not grant scientific decision authority.

| Transition | Who may perform it | Conditions |
|---|---|---|
| Create → DEFINED | Coordinator or issue proposer | Use the appropriate template; link evidence and dependencies. |
| DEFINED → READY | Coordinator | Readiness checklist passes; all author-owned decisions have explicit approval; blockers are closed/removed. |
| DEFINED / READY / IN PROGRESS / IN REVIEW → BLOCKED | Any agent may flag; coordinator records the transition | Add native blocker(s), prior state, and exact unblock condition. Pause dependent work. |
| READY → IN PROGRESS | Implementer claiming the issue | Issue is open, unassigned, has `workflow:ready`, and has no open blockers. Claim before implementation. |
| IN PROGRESS → IN REVIEW | Implementer | Post the verification/review handoff with a fixed SHA and evidence. Stop modifying the submitted revision while it is under review. |
| IN REVIEW → IN PROGRESS | Independent reviewer | Record specific changes required and evidence; return work to an implementer. |
| IN REVIEW → ACCEPTED | Independent reviewer only | Every acceptance criterion is demonstrated; reviewer is not the implementer. Record evidence, then close. |
| Any nonterminal state → REJECTED | Author or coordinator; reviewer may recommend | Record why it is rejected and whether it is superseded. A reviewer cannot approve a scientific alternative. |
| BLOCKED → recorded prior state | Coordinator or current owner | Verify unblock condition and recheck readiness; if requirements changed, return to DEFINED instead. |
| ACCEPTED → IN PROGRESS / DEFINED | Author or coordinator | Reopen only with a comment explaining new evidence or scope; preserve the acceptance history and re-review changed work. |
| REJECTED → DEFINED | Author or coordinator | Reopen only with a changed scope or new evidence, recorded in a comment. |

`workflow:accepted` issues may be closed. `workflow:rejected` issues may be closed only when the issue explicitly links the superseding decision/task; otherwise leave them open and inert. No other state may be closed. Reopening never erases prior comments or review verdicts.

### Claim and concurrency

Before starting, inspect state, assignees, blockers, branch/worktree, and current files. For a ready implementation task:

1. Confirm it is open, unassigned, `workflow:ready`, and has zero open blockers.
2. Assign yourself first (`gh issue edit <n> --add-assignee @me`), then change the sole state label to `workflow:in-progress`.
3. Re-read the issue. If another owner appeared, multiple agents claim it, or the state changed, stop and coordinate; do not both proceed.
4. Post the claim comment with starting SHA, branch/worktree, scope, dependencies, and acceptance criteria.

GitHub assignment is not a transactional lock. The re-read and explicit coordination rule are mandatory. One issue has one active implementer unless the issue records a coordination agreement. Concurrent code-writing agents use separate branches/worktrees; never share a mutable worktree or silently overwrite another agent's changes.

## Execution journal

Post concise GitHub Issue comments at each boundary. Summarize evidence; do not paste tool logs or conversation transcripts. Use these fields and adapt only when a field truly does not apply.

### A. Claim

- Issue number and title.
- Assigned scope.
- Starting commit SHA.
- Work branch/worktree.
- Dependencies checked and status.
- Initial acceptance criteria.

### B. Investigation complete

- What was inspected.
- Current behavior or reproduced defect (or relevant baseline for a feature).
- Technical approach within approved constraints.
- Unresolved questions.
- Tests/observations that would falsify the intended result.

### C. Implementation checkpoint

- Changes made.
- Current commit SHA or working-tree status.
- Tests/commands executed and actual results.
- Remaining steps.
- Unexpected discoveries/blockers.

### D. Verification and review submission

- Final implementation SHA.
- Exact test commands and actual results (never report unexecuted tests as passing).
- Evidence mapped to each acceptance criterion.
- Known limitations.
- Independent reviewer handoff/request.

### E. Completion or interruption

- Current status.
- Last known-good revision.
- What remains unfinished.
- Exact next action.
- Blockers/artifacts needing recovery.

Post a checkpoint before ending a session where possible. A task is not complete because an agent ran out of context; use `IN PROGRESS`, `BLOCKED`, or `IN REVIEW` honestly and leave it open.

## Blockers and author decisions

Use GitHub native issue dependencies. The child issue is blocked by the issue that must finish first. A blocker must be open and specific; a blocked issue comment/body must state `Blocked by`, `Unblock when`, and the state to resume. For unresolved scientific decisions, link the `[AUTHOR DECISION]` Wayfinder issue as a native blocker and use `workflow:blocked`. Never mark dependent work READY merely because a decision ticket exists or an agent recommends an answer.

When all blockers close, the coordinator/current owner verifies the stated condition, removes obsolete dependency edges if appropriate, posts an unblock comment, and restores the recorded prior state or returns the task to DEFINED if its scope changed. Do not infer unblocking from a comment alone while the native issue edge remains open.

## Verification, review, and handoff

The implementer runs the tests warranted by the task, reports only executed results, maps evidence to every criterion, commits on the task branch, and submits the fixed SHA for independent review. For code tasks, invoke the installed review workflow where the harness supports its two fresh parallel subagents; otherwise obtain an independent reviewer explicitly. Documentation-only tasks still receive independent review of links, semantics, and stated evidence. The implementer never marks their own work ACCEPTED or closes the issue.

Reviewers record **ACCEPTED**, **CHANGES REQUIRED**, or **REJECTED**, with evidence and criterion mapping. Changes required return the issue to IN PROGRESS. Acceptance closes the issue. Rejection does not imply acceptance and does not close the issue unless explicitly superseded.

Finish with an issue-comment handoff another agent can act on: current state, last known-good SHA, exact next action, commands/results, remaining work, and blockers. The `handoff` skill writes to a temporary directory; that file is supplementary only. The issue journal and linked repository artifacts remain authoritative.

## Abandoned claims and recovery

A timeout, silence, lost session, or missing conversation does not automatically release a claim. A coordinator first checks the assignee, issue journal, Git branches/worktrees, commit history, staged/unstaged changes, and untracked files. Ask the current owner when reachable. Reassign only after explicit abandonment or a coordinator-recorded recovery decision; never assume another agent's state is valid.

A recovering agent must preserve all uncommitted work. Do not run destructive cleanup/reset commands, delete a worktree, or overwrite a branch as a recovery shortcut. Record a filesystem snapshot/status and copy or retain the original worktree before continuing. Review every prior diff and commit against the issue, rerun relevant checks, and post a recovery checkpoint. Continue the same branch only after coordination; otherwise use a separate worktree/branch and reconcile explicitly.

## Installed skill boundaries

Installed skill definitions were inspected for this repository setup. `setup-matt-pocock-skills`, `wayfinder`, `grill-with-docs`, `to-spec`, `to-tickets`, `implement`, and `handoff` declare `disable-model-invocation: true`; invoke them only when the user/harness explicitly supports slash-skill invocation. `grill-with-docs` is a wrapper asking for separate grilling and domain-modeling skill calls. `to-spec` and `to-tickets` require user confirmation before publishing. `wayfinder` is planning-first and must not turn its decision map into implementation tickets without resolved decisions.

`code-review` is installed, but its two-axis process requires two fresh parallel subagents. If the active harness cannot dispatch them, do not pretend the skill ran; request a separate independent reviewer or state that review remains pending. These repository protocols are the fallback workflow; do not edit upstream skill definitions or rely on nested skill calls that the runtime does not provide.
