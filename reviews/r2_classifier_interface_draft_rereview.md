# Independent Re-review — R2 Classifier Interface Draft

## Verdict

**VERIFIED AS FAITHFUL DRAFT**

All three blocking findings from the first review are corrected. The revised text preserves the prior accepted content and adds no material contradiction. It is suitable for explicit author review only. This verdict does **not** approve D1, D2, D3, any other author-controlled choice, implementation, route reconnection, or scientific evaluation.

## Exact revisions and provenance

- Repository: `midhunpradeep/town_model`.
- Starting `main`: `7ebc2b62c175288b0673524fbfae9f9d6f15a4ff`.
- Original reviewed draft: `c48b4c7f275ea70d92462e7e20cda008039640aa`.
- Original independent review commit/report: `b0636d234bcbe31b80f39b30c0967da7171bd6bb`; report blob SHA-1 `990ee50ad2fdc501777fe2f25abe5f6f01f8e2d0`.
- Corrected draft reviewed here: `fb432286113f9d11d433ab9e3499e97c7275dba2`.
- Draft path: [`docs/reconstruction/R2_CLASSIFIER_INTERFACE_CONTRACT_DRAFT.md`](https://github.com/midhunpradeep/town_model/blob/fb432286113f9d11d433ab9e3499e97c7275dba2/docs/reconstruction/R2_CLASSIFIER_INTERFACE_CONTRACT_DRAFT.md).
- Re-review branch is based directly on the corrected SHA; this report is committed separately from the draft.

The corrected commit's sole parent is the original draft SHA. The original review commit is a sibling descendant of that draft, not an ancestor of the correction. The corrected commit is a descendant of starting `main`. Local and remote `docs/r2-classifier-interface-contract` refs resolved to the corrected SHA; live remote `main` remained at starting `main`. The original review branch remained at `b0636d…`; its report blob is unchanged.

Reviewer: fresh Pi session `01a19a08-6073-770f-815a-d44267c426f3`, separate from the implementation session represented in the #14 journal. The authenticated GitHub/Git principal is the shared repository-owner account `midhunpradeep`; independence is at the separate agent-session level, not separate human credentials. No second-human or parallel-subagent review is claimed.

## Authority and issue boundaries

Read the [execution protocol](../docs/agents/EXECUTION_PROTOCOL.md) and [decision-authority protocol](../docs/agents/DECISION_AUTHORITY.md); #14's implementation and review journals; the [original independent review report](https://github.com/midhunpradeep/town_model/blob/b0636d234bcbe31b80f39b30c0967da7171bd6bb/reviews/r2_classifier_interface_draft_independent_review.md); corrected draft; [original A1 ledger](../docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md); accepted [F11](../docs/reconstruction/A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md) and [F12](../docs/reconstruction/A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md) addenda; [readiness frontier](../docs/reconstruction/A1_IMPLEMENTATION_READINESS.md); [definitive T5 teardown review](classifier_teardown_t5_final_acceptance.md); and live [issues #2](https://github.com/midhunpradeep/town_model/issues/2), [#3](https://github.com/midhunpradeep/town_model/issues/3), and [#14](https://github.com/midhunpradeep/town_model/issues/14).

The draft remains a prominently marked proposal, not an approved scientific authority. It distinguishes the original A1 source (F0–F10), separate F11 and F12 addenda, repository-document acceptance, and implementation/evaluation. The author-reported 94-source approval frontier is not used as implementation authorization; F12.1–F12.8 remain grouped without per-ID mapping. T5 remains the operational classifier-free boundary.

Live GitHub checks showed [#2](https://github.com/midhunpradeep/town_model/issues/2) OPEN (`wayfinder:map`), [#3](https://github.com/midhunpradeep/town_model/issues/3) OPEN (`wayfinder:grilling`, pending explicit author confirmation), and [#14](https://github.com/midhunpradeep/town_model/issues/14) OPEN (`wayfinder:task`). No issue state, source authority, or author decision was changed.

## Three-finding resolution matrix

| Original blocking finding | Re-review evidence | Verdict |
|---|---|---|
| **1 — Class identity persistence** | §§4.1, 11 D1, and 13 require the exact ordered trainable class-ID tuple and canonical vocabulary/manifest identity to be bound to training identity and persisted with the bound model/run/checkpoint. Training, inference, and restoration require exact binding validation; same-width permutations and incompatible manifest identities reject before model/optimizer/checkpoint state mutation. Silent relabeling/reordering is prohibited. A class-count-only kernel is allowed only behind this enforcing binding layer. Future tests include valid restore, permutation mismatch, manifest mismatch, and mutation-free rejection. Output metadata alone is explicitly insufficient. Classifier-owned versus immutable-wrapper-owned binding remains for author choice. Fixed `DomainThing` representation remains a separate author choice. | **RESOLVED** |
| **2 — D2 author-decision completeness** | §§4.1–5, 11 D2, and 14 individually retain D2-A `N=0`, D2-B `C_train=0`, D2-C `d=0`, D2-D bounded external queries versus adapter violations, and D2-E validation/numerical policy as unresolved. Detailed discussion and final register agree that alternatives remain open, consequences and proposed defaults are stated, and author approval is requested. `d=0` discussion distinguishes the unsupported geometric coordinate minimum from the mathematically possible bias-only affine expression; retained `ImagePointModel` behavior is not author approval. D2-E covers rank/shape, finiteness, dtype/device, numerical envelope, affine policy, and rejection without silent casting, broadcasting, clipping, or repair. Float32 geometry under F7.3 remains approved; float32 affine behavior remains proposed. `ImageBatch B=0` remains a separate restriction, `K=0` remains supported, `W=0` semantics remain approved, and no-supervision training remains undefined under F8.4. | **RESOLVED** |
| **3 — Fail-closed coverage** | §§10 and 13 identify the accepted T5 §J unavailable-operation matrix as the complete regression reference, not an illustrative subset. `export-geometry` is named. Its unauthorized future invocation must raise `ClassifierBackendUnavailableError`, create no files/artifacts, and leave existing outputs/state unchanged. The same fail-closed/no-side-effect rule covers applicable T5-disabled training, evaluation, resume, checkpoint reconstruction/evaluation, interpretation/snapshot, relation, worker, CLI/API, and dashboard surfaces. | **RESOLVED** |

## Regression checks on previously accepted content

The original review found the rest of the draft substantially sound. The correction diff is confined to D1/D2 clarification, explicit T5 matrix coverage, and the deferred F5.5 note; the following boundaries remain accurate and unchanged in substance:

- **Scientific attribution:** F0–F12 are attributed to their separately versioned sources; F3.6-R remains active and supersedes F3.6; no additional approval is inferred from this draft.
- **Mathematical interface:** flat `[N,d]` inputs; affine logits `w·x+b` and hard `z≥0`; ordered Boolean membership remains separately typed from affine logits or geometry signed-coordinate measurements.
- **Geometry and objectives:** F3.6-R frame/window boundary and ambient open-interior semantics; exact signed-coordinate clearance and min/max composition; the five unregularized primary configurations; F8 ownership and reducer boundaries are preserved.
- **Class vocabulary and retained infrastructure:** canonical manifest order, fixed/unsupported class distinctions, point-only `ImagePointModel`/`ImagePointResult`, retained valid-occurrence identity and `K=0`, and generic caller-value reducers remain engineering facts rather than new scientific requirements.
- **D3–D5:** bounded `tanh` versus proposed unbounded identity-head analysis remains unapproved with deferral available; D4 preserves point identity, valid-row flattening, and gradient flow; D5 does not claim existing reducers complete F8 conformance.
- **Sequencing and hold boundaries:** independent pure-evaluator slices remain separated from image integration, objectives, training, and evaluation. The F5.5 tie-subgradient/branch-frequency item is explicitly deferred to differentiable clearance and is not a dependency for non-trainable membership tests. Issue #3's pending author-confirmation boundary is preserved.

No new material contradiction or blocking finding was identified. The draft does not reconnect or modify any T5-disabled route.

## Changed-file, source-protection, and link verification

Exact changed-file inventories:

```text
Original draft → corrected draft:
M docs/reconstruction/R2_CLASSIFIER_INTERFACE_CONTRACT_DRAFT.md

Starting main → corrected draft:
A docs/reconstruction/R2_CLASSIFIER_INTERFACE_CONTRACT_DRAFT.md
```

Thus no executable source, tests, model artifact, dataset, scientific source, accepted review, current authority document, or protected T5 review changed. The original review is retained unchanged on its separate branch; its report blob remains `990ee50…`.

Independent Markdown audit of the corrected draft parsed **92 links**: 89 local targets, including 75 line-anchor checks and one heading-anchor check, plus three external GitHub issue links. All local files/anchors resolved; no failures. The three live issue destinations (#2/#3/#14) were verified present and their states checked. A repository-relative inline-link parser was used; it does not claim validation of arbitrary external websites.

Commands and results:

```text
git diff --check c48b4c7f275ea70d92462e7e20cda008039640aa fb432286113f9d11d433ab9e3499e97c7275dba2  PASS
git diff --check 7ebc2b62c175288b0673524fbfae9f9d6f15a4ff fb432286113f9d11d433ab9e3499e97c7275dba2  PASS
```

Ancestry checks confirmed `7ebc2…` and `c48b4…` are ancestors of `fb432…`, while `b0636…` is not. Remote refs were queried with `git ls-remote`; `main` and the corrected implementation branch matched the specified SHAs.

## Remaining author choices

D1 binding ownership and fixed `DomainThing` presentation; each D2-A–E alternative/default; and D3 bounded/unbounded image-head policy remain proposed or unresolved for author review. D4/D5 remain later engineering boundaries. #3 still requires explicit author confirmation. This review chooses none of them and authorizes no pure evaluator, flat classifier, image integration, or training implementation.

## Verification performed and limitations

- Performed exact SHA/parent/ancestry checks, both diff inventories, protected-path comparison, report-blob/ref comparison, Markdown link/anchor audit, `git diff --check`, and live issue-state checks.
- No runtime tests were run; the correction is documentation-only and introduces no executable changes. The original review's 29 focused passing tests remain prior review evidence, not tests rerun in this re-review.
- No T5 route was invoked or reconnected. T5's accepted matrix was inspected and used as the required future regression reference.
- This review verifies documentation fidelity and retained-source attribution, not mathematical proofs, implementation, numerical behavior, learning, or scientific outcomes. Reviewer identity shares the repository-owner GitHub principal as disclosed above.

## Final verdict

**VERIFIED AS FAITHFUL DRAFT — suitable for author review.** All three original blocking findings are corrected, with no new material defect found. This is not approval of D1, D2, D3, resolution of #3, authorization to implement, acceptance of a classifier backend, or permission to change the current fail-closed baseline. Issues #14 and #3 remain OPEN.
