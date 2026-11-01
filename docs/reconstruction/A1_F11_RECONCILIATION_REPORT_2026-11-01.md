# A1-F11 Scientific Addendum Reconciliation Report

**Date:** 2026-11-01
**Implementation issue:** [#12 — F11: Reconcile twelve post-R1 scientific approvals](https://github.com/midhunpradeep/town_model/issues/12), native child of planning map [#2](https://github.com/midhunpradeep/town_model/issues/2)
**Starting `main`:** `100d04d4ecae4f58f525bbdac76b87bf28026d5c`
**Exact implementation SHA:** recorded in the fixed-SHA review-submission comment on issue #12; do not follow a moving branch head.

## Purpose and provenance boundary

This report records a documentation-only reconciliation of twelve F11 approvals supplied as author-approved summaries in a post-R1 mathematics-session handoff dated 2026-11-01. The author explicitly approved this task contract on that date. The original mathematics-session transcript, formal derivations, detailed proof hypotheses, algorithms, experimental protocols, and source URL were not supplied. The [versioned F11 addendum](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md) preserves the supplied statements and identifies the gaps without inventing missing science.

The original [A1 ledger](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) remains canonical for F0–F10. It is not edited or rewritten. The separate F11 addendum is not a second 82-decision ledger. The source-level combined frontier is 70 original active approvals plus twelve F11 approvals (=82); the accepted R1 review did not independently review all 82. F11 becomes repository authority only after this addendum's independent review and integration.

## Identity and decision inventory

| Artifact | SHA-256 | Bytes | Status / scope |
|---|---|---:|---|
| Original A1 ledger | `f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9` | 34,910 | Preserved unchanged; authority for F0–F10. |
| F11 addendum | `3a417f74dbfdbee0771dbdf353c9348801a1c35467c64cdd6b9975caabf91c86` | 21,547 | Twelve supplied author-reported approvals; submitted separately for independent review. |

The twelve unique F11 headings are exactly: F11.1, F11.2, F11.3, F11.4, F11.5, F11.6, F11.7, F11.8, F11.9, F11.10, F11.11, and F11.12. The original ledger continues to contain 70 active IDs; its existing audit remains unchanged. No original decision is superseded by F11, and the existing F3.6 → F3.6-R supersession remains in force. The F11 addendum records all twelve source statements verbatim as approval summaries, not as transcript quotations.

### Reproduce source hashes and F11 count

Run from the repository root using Python 3 standard library only:

```python
from hashlib import sha256
from pathlib import Path
import re

original = Path("docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md").read_bytes()
assert len(original) == 34910
assert sha256(original).hexdigest() == "f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9"

path = Path("docs/reconstruction/A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md")
raw = path.read_bytes()
assert len(raw) == 21547
assert sha256(raw).hexdigest() == "3a417f74dbfdbee0771dbdf353c9348801a1c35467c64cdd6b9975caabf91c86"
text = raw.decode("utf-8")
expected = [f"F11.{i}" for i in range(1, 13)]
ids = re.findall(r"(?m)^### (F11\.\d+) — ", text)
assert ids == expected, ids
assert len(ids) == len(set(ids)) == 12
for letter in "ABCDEFGHIJ":
    assert re.search(rf"(?m)^### {letter} — ", text), letter
print("original_sha256", sha256(original).hexdigest(), "bytes", len(original))
print("F11_ids", ids)
print("addendum_sha256", sha256(raw).hexdigest(), "bytes", len(raw))
```

To inspect the source-linked prior-decision crosswalk, open addendum §3 and compare its A1 decision IDs with the linked line numbers in the unchanged original ledger. The A–J source topics link directly to the original audit lines; the addendum's §4 lists, for each, the approved principle, missing details, consequence, and actual dependent-task boundary.

## Crosswalk and readiness reconciliation

The addendum §3 source-links F11.1–F11.12 to relevant F0–F10 decisions without rewriting them. Its §4 maps each original foundational audit topic A–J to the supplied F11 contribution and separately records what remains missing, implementation/evaluation consequences, and whether a dependent scope is actually blocked. These mappings are partial: no blanket “A–J resolved” claim is made.

The derived [readiness frontier](A1_IMPLEMENTATION_READINESS.md) now separates four practical categories:

- **Implementation-critical:** P0-A bounded/unbounded raw-to-physical parameterization/transforms, finite-precision reserves, feasibility and initialization; P0-B transactional invalid optimizer-update rejection/rollback/recovery including optimizer and scheduler state; P0-C training-stage transitions, optimizer-state policy, numerical defaults and gradient-failure handling. These block corresponding trainable modules, not unrelated pure evaluators whose contracts exclude those concerns. Class ordering, point-domain behavior, empty inputs and zero-window classifier-integration edges remain separately unresolved.
- **Evaluation-critical:** relation references and provenance, exact certificate algorithms, statistical units, experiment populations, generalization protocols, and remaining RQ1/RQ2 decision details. Foundational approval does not supply a complete evaluation protocol.
- **Optional experiments:** compactness; auxiliary point organization; fixed-point diagnostics; higher-order intersection and capacity benchmarks; supplementary affine structural diagnostics; additional reference losses and ablations. None is a primary prerequisite.
- **Deferred theory:** broad topology characterization and general ontology-satisfiability work not needed for the immediate specified experiments remains in the scientific record without becoming an implementation blocker.

The readiness frontier retains planning candidates for physical-corner Boolean box membership; house/window membership and town union; zero-window/single-box equivalence; signed coordinate clearance; exact min/max composition; signed logistic/hinge kernels; flat point-only affine behavior subject to scoped integration edges; and deterministic boundary, degeneracy and fixed-point fixtures. These are not implemented or automatically ready. Pure evaluators are not blocked on P0-A/B/C if their contract excludes trainable parameterization, optimization and optimizer state.

## Primary invariants retained

The five primary conditions remain exactly:

1. Affine baseline + BCE-with-logits.
2. Box + signed logistic.
3. Box + signed hinge.
4. Town + signed logistic.
5. Town + signed hinge.

Primary objectives remain unregularized. F3.6-R remains authoritative: a house is a closed frame minus ambient open window interiors; a town is the union of houses. Exact Boolean membership determines geometric predictions. Signed coordinate clearance remains a separate numerical training measurement composed with the approved exact min/max rules and no hidden normalization. No retired classifier mechanics are inferred.

## Current authority and issue reconciliation

- The original A1 ledger and completeness audit are unchanged; the audit still covers only the original 70 decisions.
- The R1 implementation `be4bbc067d235cbd896665ba33f6585b2cabc4e6` and accepted independent-review report/`main` revision `100d04d4ecae4f58f525bbdac76b87bf28026d5c` are unchanged. R1 is accepted and closed under #10; that review did not cover F11.
- Issue #2 remains open `wayfinder:map`; after the addendum revision was fixed, its current planning map was updated to cite the exact addendum commit and distinguish the F11 author-reported source approval from pending repository review/integration.
- Issue #3 remains open `wayfinder:grilling`; the membership/town-set proposal remains subject to explicit author confirmation.
- Issue #11 remains a separate post-R1 lifecycle-status task and was not implemented under #12. It was OPEN, unassigned and `workflow:defined` at investigation and was rechecked before submission; the overlap was recorded in the #12 journal and a [coordination-only comment on #11](https://github.com/midhunpradeep/town_model/issues/11#issuecomment-6562119453). No concurrent owner/edit was observed; #11 body, labels, owner and lifecycle were not changed.
- Issue #12 is the implementation/review tracker. At submission it is `workflow:in-review`; only an independent reviewer may accept or close it.

## Changed-file inventory

The submitted implementation changes exactly these nine documentation files:

1. `AGENTS.md`
2. `README.md`
3. `docs/ACCEPTANCE_INDEX.md`
4. `docs/CURRENT_PROJECT_STATUS.md`
5. `docs/agents/DECISION_AUTHORITY.md`
6. `docs/agents/EXECUTION_PROTOCOL.md`
7. `docs/reconstruction/A1_IMPLEMENTATION_READINESS.md`
8. `docs/reconstruction/A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md`
9. `docs/reconstruction/A1_F11_RECONCILIATION_REPORT_2026-11-01.md`

Issue #2's body is separately updated after the fixed implementation SHA exists. No production code, tests, datasets, snapshots, checkpoints, experiments, historical reports/reviews, original A1 ledger/audit, R1 completion report, or accepted review evidence is changed. The issue journal records the #11 overlap; #11 itself remains separate.

## Verification and acceptance mapping

- **AC1 — Original authority preserved:** verified original ledger SHA-256/size above. The diff inventory excludes the original ledger, original completeness audit, R1 implementation/report/review, T5/R0 evidence and all production artifacts.
- **AC2 — F11 completeness/provenance:** reproducible heading extraction asserts the exact ordered 12 unique IDs. Exact supplied summaries and author-reported date/status are in addendum §2; the unavailable transcript and missing derivations are expressly disclosed.
- **AC3 — Combined authority:** addendum §1 and current entrypoints state 70 original + 12 F11 = 82 source-level approvals, with no replacement ledger and explicit statement that R1 did not review F11.
- **AC4 — Audit reconciliation:** addendum §4 maps all A–J topics, identifies missing detail and dependent scope, and does not mark the audit resolved.
- **AC5 — Priority frontier:** readiness report presents implementation-critical, evaluation-critical, optional-experiment, and deferred-theory categories; P0-A/B/C and classifier integration edges remain distinct.
- **AC6 — Primary invariants:** addendum §5 and readiness report retain five exact primary conditions, unregularized objectives, F3.6-R, Boolean-vs-clearance separation and optional-study boundary.
- **AC7 — Current authority consistency:** current entrypoints distinguish original A1 from F11, point to #10/#12 and state conditional F11 repository authority; #2 pins the exact addendum revision after it exists.
- **AC8 — Workflow separation:** #2/#3 Wayfinder states, #10 accepted/closed, #11 separate, and #12 in review are verified in GitHub; no author decision is self-approved or closed by this work.
- **AC9 — Scope isolation:** exact nine-file inventory above; protected code/data/experiment/historical-authority paths unchanged.
- **AC10 — Reproducibility:** embedded hash/count commands, source-linked crosswalk, issue links, actual path inventory, and `git diff --check` are available at the fixed SHA. No runtime tests were run or claimed; no executable paths changed.

## Verification commands and limitations

Executed for this documentation-only task: original and addendum SHA-256/byte-count checks; standard-library F11 heading/count and audit-topic checks; current status/issue-state and remote-main verification; local Markdown target checks; changed-path/protected-path diff inspection; and `git diff --check`. The complete command/results and exact implementation SHA are recorded in the issue #12 fixed-SHA submission comment.

No runtime model regression tests were run because no executable code changed. This report does not claim access to the underlying mathematics-session transcript, formal proof details, or unprovided experiment protocols. The addendum is submitted for independent review and is not self-accepted. Stop here: no R2, classifier, detector, training, or unrelated task is authorized by this reconciliation.
