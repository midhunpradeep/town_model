# Independent Review — R1 Authoritative A1 Scientific Decision Ledger

## Verdict

**ACCEPT R1**

All ten acceptance criteria pass. The canonical author-uploaded source is byte-identical at the submitted revision; the independent extraction confirms all 70 active decisions and source locations; derivative documentation preserves the approved science and its unresolved boundaries; tracker state and scope are consistent. No blocking finding remains.

## Review identity and exact target

- **Repository:** `midhunpradeep/town_model`
- **Issue:** [#10 — R1: Authoritative A1 Scientific Decision Ledger](https://github.com/midhunpradeep/town_model/issues/10)
- **Base / R0-integrated `main`:** `b44304a09f387d767f55abe3db1b3f86cbae4c36`
- **Reviewed implementation:** `be4bbc067d235cbd896665ba33f6585b2cabc4e6`
- **Implementation branch:** `docs/r1-authoritative-a1-ledger` (advertised branch HEAD matched the reviewed SHA)
- **Review worktree:** `/tmp/town-model-r1-independent-review`, detached at the exact reviewed SHA; clean before review-report creation.
- **Ancestry:** the reviewed SHA descends from the exact base through `0fa19f98cbf0d3acde0276deed18136862736bec` (canonical source addition) and `be4bbc067d235cbd896665ba33f6585b2cabc4e6` (current authority-status alignment). The base is an ancestor of the reviewed SHA. R0 implementation `40f362b6cf7e3969b0ea61c9b2a52f968a32bce2`, final R0 review/report `1259859fcc29e422a95e1b8291516c2b0b3f155f`, and T5 review/report `2b9df576aac2fd95f093bbb3b046813620a822a5` are ancestors of the base.
- **Review environment:** Linux x86_64, Git 2.34.1, Python 3.12.0, GitHub CLI 2.55.0. The embedded audit and independent parsers use only Python's standard library. No runtime model/test environment was needed.
- **Independence limitation:** I did not implement R1 and reviewed it in a separate agent session/worktree. GitHub operations and commits use the repository-owner authenticated account shared with the implementation journal; this is not a claim of a separately authenticated human account.

The fixed target was fetched/checked as the remote R1 branch head. `main` and `origin/main` remained at the stated base during review. Issue #10 was OPEN with `workflow:in-review` at review start; its native parent is #2 and its native open-blocker list is empty.

## Baseline and actual base-to-submission diff

The complete `git diff b44304a09f387d767f55abe3db1b3f86cbae4c36 be4bbc067d235cbd896665ba33f6585b2cabc4e6` was generated and inspected. It contains exactly these eleven documentation files:

1. `AGENTS.md`
2. `README.md`
3. `docs/ACCEPTANCE_INDEX.md`
4. `docs/CURRENT_PROJECT_STATUS.md`
5. `docs/agents/DECISION_AUTHORITY.md`
6. `docs/agents/EXECUTION_PROTOCOL.md`
7. `docs/reconstruction/CLASSIFIER_TEARDOWN_MAP.md`
8. `docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md`
9. `docs/reconstruction/A1_DECISION_COMPLETENESS_AUDIT.md`
10. `docs/reconstruction/A1_IMPLEMENTATION_READINESS.md`
11. `docs/reconstruction/R1_COMPLETION_REPORT.md`

The actual diff is 586 insertions and 15 deletions. There are no production source, tests, datasets/examples, model/checkpoint, experiment artifact, review, R0, Phase 1–3 acceptance, expanded-specification, or `T3_TEST_RETIREMENT_MANIFEST.md` changes. No classifier, detector, loss, optimizer, training, or R2 implementation was added. The file inventory is exact; no unexpected changes were found.

R0's historically correct 50-decision account remains in its preserved R0 records and comments, not current authority. The current entrypoints instead identify A1's 70 active approvals. The R0 report/reviews are unchanged. The accepted T0–T5 classifier-free baseline remains explicit; historical Phase 1–3 acceptance is not represented as current classifier availability. Phase 4's historical authorization is distinguished from its present hold. The expanded v1.1 draft remains historical material. Current status and readiness language do not claim RQ1 or RQ2 has been scientifically established.

## Canonical-source identity and provenance

The repository file `docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md` independently hashes to:

- **SHA-256:** `f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9`
- **Size:** `34,910` bytes

The original supplied author file was available at `/home/midhun/Downloads/town_model_A1_scientific_decision_ledger_70.md`. Its independently computed hash and byte count match, and `cmp -s` against the repository copy returned 0 (byte-for-byte equal). The canonical source was not altered during review.

The complete source is visibly categorized: active decisions in §1; superseded F3.6 in §0/F3.6-R; mathematical consequences in §2; historical engineering context in §3; twelve unresolved gates in §4; unresolved foundational questions A–J in §5; and future implementation-session recommendations in §6. The audit labels these categories separately and does not count consequences, historical context, gates, A–J, or recommendations as approvals.

## Independent decision extraction and audit-script assessment

I extracted active decision headings independently from the canonical source's §1 using a separate parser, then compared `(ID, source line, heading)` tuples to the audit table. The embedded script in `A1_DECISION_COMPLETENESS_AUDIT.md` was also extracted and executed exactly with Python 3.12.0. Both checks agree:

| Family | Verified active IDs |
|---|---:|
| F0 | 1 |
| F1 | 1 |
| F2 | 8 |
| F3 | 11 |
| F4 | 4 |
| F5 | 6 |
| F6 | 5 |
| F7 | 8 |
| F8 | 5 |
| F9 | 5 |
| F10 | 16 |
| **Total** | **70 unique IDs** |

The verified ID set is F0.1; F1.1; F2.1–F2.8; F3.1–F3.5, F3.6-R, F3.7–F3.11; F4.1–F4.4; F5.1–F5.6; F6.1–F6.5; F7.1–F7.8; F8.1–F8.5; F9.1–F9.5; and F10.1–F10.16. Every one of the 70 audit table rows matches the extracted ID, source line, and heading. F3.6-R at line 43 is active; F3.6 is not active and is explicitly superseded. The source retains all 12 numbered open gates and all ten A–J foundational questions.

The embedded script is effective against the requested completeness/index faults:

- Its exact source SHA/size guard fails on any changed source bytes, including a missing or duplicated source decision heading.
- It asserts the exact 70-ID set, uniqueness, family counts and F3.6-R supersession.
- Its ordered equality check between extracted source tuples and audit rows detects a missing, duplicate, wrong-line, wrong-ID, or wrong-heading index entry.
- It verifies numbered gates 1–12 and audit letters A–J.

I tested these properties only on temporary copies: removing or duplicating a source heading failed the hash assertion; removing or duplicating an index row and changing an indexed line all failed the tuple equality assertion. The canonical source and audit were not modified. This is a reproducible index of the hash-pinned authority, not a substitute specification.

## Scientific-fidelity findings

The canonical source is unchanged, and the derivative readiness/authority documents accurately preserve the requirements relevant to this task:

- The primary comparison is exactly **affine + BCE-with-logits; box + signed logistic; box + signed hinge; town + signed logistic; town + signed hinge** (`A1_IMPLEMENTATION_READINESS.md`, “Approved comparison and semantic guardrails”; canonical ledger F6.1–F6.3 and primary-comparison statement).
- Primary geometry/classification objectives remain unregularized. The readiness document excludes compactness penalties, auxiliary point attraction, ontology-axiom constraints and supplementary reference losses from the five-way primary configurations. ChebiBox and original-town-inspired town references remain supplementary; the source-corrected town reference formula under F6.5 is explicitly not approved.
- Membership retains active F3.6-R: `H = F \ ⋃ int_{ℝ^d}(W)`, a closed frame minus strict ambient window interiors; boundaries remain included unless another window's interior excludes them; town membership is the union of houses. Boolean membership remains authoritative and signed coordinate clearance a separate measurement. No current document revives F3.6.
- F10.3 fixed-point diagnostics and F10.13–F10.16 remain in the immutable source as diagnostic/optional work. The readiness report keeps fixed-point benchmarks, compactness and point-organization work separate from the primary comparison, says they do not authorize primary regularization or global ontology claims, and identifies optional P2 studies as non-blocking.
- The unresolved numerical and integration details remain unresolved: raw-to-physical transforms; exact epsilon/delta and float32 reserves/ranges; atomic optimizer rollback; optimizer/scheduler transitions and numeric defaults; reference-loss source conformance; detector/ROI-matching policy; relation-certificate details; and final statistical/evaluation protocol. Where the source approves a principle (for example F7.8 rejection of invalid updates, F9.4 constant-LR AdamW initially, or F10.6 sound certificates/explicit unknown), the readiness report distinguishes that principle from the missing implementation decision.

No derivative document turns the historical implementation into authority, implies the approved ledger is already implemented, or reports scientific evidence establishing RQ1/RQ2.

## Readiness frontier assessment

`A1_IMPLEMENTATION_READINESS.md` maps the complete P0/P1 frontier to approved bases, the missing decision, earliest affected task, consequence of premature selection, and a focused author mathematics-session question. The report splits source gates where distinct implementation dependencies are useful; its P0-1 through P0-6 and P1-1 through P1-5 cover the source's gates 1–9 without silently answering them.

The mapping correctly distinguishes:

- **P0:** raw/physical transforms and initialization; numerical operating range/reserves; invalid-step state transaction; phase/optimizer transitions and numeric values; supplementary source-reference objective conformance; classifier integration boundaries.
- **P1:** global relation population/references; sound certificate assumptions; comparison matrix/monitors; detector and ROI matching; statistical units and uncertainty.
- **P2:** compactness, point organization and additional diagnostics/studies, explicitly not blockers for the five-way primary comparison.
- **A–J:** all ten foundational questions explicitly marked unresolved, with affected scopes distinguished from local evaluator work.

The candidate scopes are bounded by supplied physical inputs and approved semantics. Pure physical-corner Boolean membership and raw-unit signed-clearance evaluation do not authorize raw transforms, optimizers, trainable geometry, or training. The report states the required preconditions and defers unresolved domain/integration edges. No readiness claim stronger than the canonical source was identified; no scientific question was resolved during review.

## GitHub issue reconciliation

### Issue #2

Issue #2 is OPEN and retains only `wayfinder:map`. Its current body pins the source, audit and readiness report to the fixed R1 SHA; states 70 decisions and F3.6-R supersession; distinguishes the readiness frontier from the authority source; separates P0/P1, optional P2 and A–J; and says R1 remains under independent review. Its unresolved frontier is not represented as approved science. R1's current map reconciliation is recorded in [comment 6561623422](https://github.com/midhunpradeep/town_model/issues/2#issuecomment-6561623422), with fixed-revision status in [comment 6561662122](https://github.com/midhunpradeep/town_model/issues/2#issuecomment-6561662122).

### Issue #3

Issue #3 is OPEN and retains only `wayfinder:grilling`. The corrected full proposal is [comment 6561634838](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6561634838). It accurately covers F3.2, F3.5, F3.6-R and superseded F3.6, F3.11, F4.1–F4.4, and applicable F5.1/F5.3/F5.6. The final exact-revision line pointers are in [comment 6561664377](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6561664377) and [comment 6561667693](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6561667693). The ledger blob at the earlier pinned commit `0fa19f98cbf0d3acde0276deed18136862736bec` is identical to the blob at the submitted SHA. The final pinned lines (39, 42, 43, 48, 54–57, 61, 63 and 66) resolve to the stated decisions.

I inspected the complete comment history and all source-line citations. An earlier preserved proposal comment contains the reported formula-escaping error and one misspelled F4 source path (`A1_SCIENT_DECISION_LEDGER...`). The subsequent correction comments and, most importantly, the complete corrected proposal explicitly say they supersede the earlier formula-transcription and F4-link errors. The later exact-revision pointers are valid. The old erroneous link remains in append-only history but is clearly superseded; the current proposal is discoverable by its “Corrected proposed author resolution” heading. No further clarification is needed for fresh-agent recovery.

The proposal repeatedly states that it is agent-prepared, not author-signed, and awaits explicit author confirmation. Despite the shared GitHub owner login, it does not claim to be author confirmation. Issue #3 was not closed or modified by this review.

### Issue #10

At review start, issue #10 was OPEN with `workflow:in-review`, assigned to the implementer, a native child of #2 and with zero open native blockers. The fixed-SHA handoff matches the exact target reviewed here. Following publication of this report and linked verdict, I moved #10 to `workflow:accepted` and closed it under the documented reviewer-only lifecycle. Issues #2 and #3 remain open with their Wayfinder labels.

## Scope, Markdown links, whitespace, and tests

- **Scope:** the exact 11-file list above is documentation only. A protected-path diff for `src/`, `tests/`, `examples/`, `artifacts/`, `reviews/`, the expanded v1.1 specification, and `docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` was empty. No tests, datasets, models, checkpoints or experiments changed.
- **Links:** an independent local Markdown scan checked 101 local inline targets across the 11 changed documents; all resolve. Issue #3 source-line citations were separately checked against the pinned commit trees and line contents as described above. One historical typo remains only in the explicitly superseded comment.
- **Whitespace exception:** the canonical source contains exactly three trailing-whitespace lines, lines 3–5, each ending in exactly two ASCII spaces. The author-uploaded file has the same bytes, so preserving the three Markdown hard breaks is necessary for the required source identity. Full `git diff --check` reports exactly those three findings and no others; the check excluding only the canonical ledger passes. No repository-wide whitespace settings or configuration were changed.
- **Runtime verification:** no pytest, compile, model, or scientific evaluation tests were run. This is documentation-only and no executable code/test/data path changed. The executed checks were source SHA/size and `cmp`; the embedded audit; an independent source parser and audit-table comparison; temporary-copy audit mutation checks; local Markdown link validation; issue/comment/label/native-parent/blocker queries; Git ancestry/ref/inventory/protected-path checks; and both full and path-filtered `git diff --check`.

## Acceptance matrix

| Criterion | Verdict | Concrete evidence and independent verification |
|---|---|---|
| **AC1 — Source identity** | **PASS** | Target source hashes to `f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9`, is 34,910 bytes, and `cmp` against the available original author-uploaded file succeeds. No source edits made. |
| **AC2 — Decision completeness** | **PASS** | Embedded script executed successfully; separate extraction verifies 70 unique exact IDs, family counts, all 70 line/heading mappings, 12 gates and A–J. Missing/duplicate/source-index mutation cases failed in temporary copies. |
| **AC3 — Supersession** | **PASS** | F3.6-R is the active source decision at line 43; F3.6 is explicitly superseded and excluded from active IDs. The audit and derivatives preserve this. |
| **AC4 — Scientific fidelity** | **PASS** | Readiness guardrails preserve the exact five-way comparison, unregularized primary objectives, supplementary references/F6.5 unresolved state, F3.6-R Boolean semantics, separate clearance, and optional F10.3/F10.13–F10.16 studies. No conflicting derivative claim found. |
| **AC5 — Approval separation** | **PASS** | Audit categories distinguish active approvals, superseded choice, mathematical consequences, historical context, gates, A–J and recommendations. Source approval, implementation state, independent review and scientific evaluation remain separate. |
| **AC6 — Authority consistency** | **PASS** | `AGENTS.md`, README, status, acceptance index, decision authority, execution protocol and teardown navigation point to A1. Current R1 is not accepted; the T0–T5 classifier-free baseline, Phase 4 hold, historical v1.1 status and non-evaluation of RQ1/RQ2 are clear. Historical R0 50-count records are untouched. |
| **AC7 — Tracker consistency** | **PASS** | Live GitHub state confirms #2 OPEN/`wayfinder:map`, #3 OPEN/`wayfinder:grilling`, and #10 initially OPEN/`workflow:in-review`. #2 reflects 70 decisions and distinct blockers. #3's corrected proposal and final source pointers are accurate, discoverable, and explicitly pending author confirmation. |
| **AC8 — Actionable dependency mapping** | **PASS** | Readiness report maps P0/P1 bases, missing decisions, earliest tasks, premature-choice consequences and author questions; P2/A–J remain distinct. Candidate physical-input evaluator scopes exclude unresolved transforms, optimizer and training choices. |
| **AC9 — Scope isolation** | **PASS** | Actual base-to-SHA inventory contains exactly the eleven expected Markdown files; protected source/test/data/artifact/review/specification/manifest diff is empty. No implementation or R2 work. |
| **AC10 — Fresh-agent reproducibility** | **PASS** | Hash-pinned source, executable audit, exact line table, all local Markdown targets, exact branch SHA/ancestry, and live issue identities/states/comments reproduce the handoff. Prior erroneous #3 text is explicitly superseded and final pointers are pinned. |

## Limitations and final disposition

Runtime model regression tests were not executed because no executable code changed; no RQ1/RQ2 evaluation was performed or is claimed. The local link scan validates file targets, while GitHub source-line pointers were validated against commit trees and line contents. The old incorrect #3 citation is preserved in append-only history but superseded by the corrected proposal and final valid pointers. GitHub account authentication is shared with the implementation journal; the independent review was performed in a separate agent session and exact-SHA worktree.

**Final verdict: ACCEPT R1.** No blocking findings. Issue #10 is now `workflow:accepted` and CLOSED. Issues #2 and #3 remain open; no merge to `main`, R2 work, classifier implementation, or scientific decision was undertaken.
