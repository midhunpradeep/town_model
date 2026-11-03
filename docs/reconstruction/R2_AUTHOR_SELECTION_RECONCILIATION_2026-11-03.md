# R2 Author-Selection Reconciliation — 2026-11-03

> **Lifecycle status update after source revision `af0865439be085bc59434b6264e4bd5d2df91c59`:** The author subsequently confirmed the R2 selections, including the D2-D1 bounded-affine interpretation, in [#15 comment 6597917380](https://github.com/midhunpradeep/town_model/issues/15#issuecomment-6597917380), and confirmed the existing #3 mathematical proposal in [#3 comment 6597917347](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6597917347). The pending-confirmation language in the preserved source text below describes the status at the original reconciliation revision; consult the [current R2 decision index](R2_CONFIRMED_INTERFACE_DECISIONS_2026-11-03.md) for current authority. These confirmations approve specifications only, not implementation. The underlying session transcript was not supplied or independently inspected.

> **PROPOSED TRANSCRIPTION — AUTHOR CONFIRMATION PENDING.** This document reconciles author-reported selections from the 2026-11-03 ChatGPT session against the independently reviewed R2 interface draft. It is not an accepted specification, does not amend the A1/F11/F12 scientific sources, and does not authorize implementation, route reconnection, or scientific training. The selection report was supplied in the task request; no underlying ChatGPT transcript or session URL was provided or independently inspected.

**Decision record:** [#15 — R2 interface selections](https://github.com/midhunpradeep/town_model/issues/15) (OPEN, `wayfinder:grilling`), child of [#2](https://github.com/midhunpradeep/town_model/issues/2); related draft task [#14](https://github.com/midhunpradeep/town_model/issues/14) remains OPEN. Existing mathematical confirmation ticket [#3](https://github.com/midhunpradeep/town_model/issues/3) remains OPEN.

## 1. Provenance, starting refs, and authority boundaries

### Fixed source and review revisions

The live refs were checked before reconciliation. At that check, local `main` and remote `origin/main` both resolved to **`7ebc2b62c175288b0673524fbfae9f9d6f15a4ff`**. The following remote refs also matched the task's verified references:

| Evidence | Exact revision | Scope/status |
|---|---|---|
| Starting/current `main` | [`7ebc2b62c175288b0673524fbfae9f9d6f15a4ff`](https://github.com/midhunpradeep/town_model/commit/7ebc2b62c175288b0673524fbfae9f9d6f15a4ff) | Current classifier-free baseline at reconciliation start. |
| Corrected R2 draft | [`fb432286113f9d11d433ab9e3499e97c7275dba2`](https://github.com/midhunpradeep/town_model/blob/fb432286113f9d11d433ab9e3499e97c7275dba2/docs/reconstruction/R2_CLASSIFIER_INTERFACE_CONTRACT_DRAFT.md) | Exact draft reviewed for author consideration; its parent is the earlier draft `c48b4c7f275ea70d92462e7e20cda008039640aa`. |
| First independent report | [`b0636d234bcbe31b80f39b30c0967da7171bd6bb`](https://github.com/midhunpradeep/town_model/blob/b0636d234bcbe31b80f39b30c0967da7171bd6bb/reviews/r2_classifier_interface_draft_independent_review.md) | Reviewed the earlier draft `c48b4c7f275ea70d92462e7e20cda008039640aa`; requested three corrections. |
| Independent re-review | [`0cc2680a67d7ad198cfa16afb5f69dcbd4e99b59`](https://github.com/midhunpradeep/town_model/blob/0cc2680a67d7ad198cfa16afb5f69dcbd4e99b59/reviews/r2_classifier_interface_draft_rereview.md) | Reviewed corrected draft `fb432286113f9d11d433ab9e3499e97c7275dba2`; verdict **VERIFIED AS FAITHFUL DRAFT**. |

The reports establish draft/source fidelity for their bounded review scope. They do **not** record the 2026-11-03 selections, durably confirm an author decision, prove the mathematics, accept implementation, or authorize any code or route. The first report's REQUEST DRAFT CORRECTIONS is superseded only as a review verdict by the independent re-review of the corrected draft; neither report is an author approval.

### Immutable scientific sources

The separately versioned sources were verified at the starting revision and remain unmodified by this task:

| Authority | Source identity at `7ebc2…` | Scope |
|---|---|---|
| [Original A1 Scientific Decision Ledger](https://github.com/midhunpradeep/town_model/blob/7ebc2b62c175288b0673524fbfae9f9d6f15a4ff/docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md) | SHA-256 `f552b60f10f96d9e76585331b99e8543d8b7a81535c261386a93aa697b0489b9`; 34,910 bytes. | 70 active approvals, F0–F10; F3.6-R supersedes F3.6. The [completeness audit](https://github.com/midhunpradeep/town_model/blob/7ebc2b62c175288b0673524fbfae9f9d6f15a4ff/docs/reconstruction/A1_DECISION_COMPLETENESS_AUDIT.md) audits this original source only. |
| [F11 Scientific Addendum](https://github.com/midhunpradeep/town_model/blob/7ebc2b62c175288b0673524fbfae9f9d6f15a4ff/docs/reconstruction/A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md) | SHA-256 `3a417f74dbfdbee0771dbdf353c9348801a1c35467c64cdd6b9975caabf91c86`; 21,547 bytes. | Twelve author-reported additions, independently accepted/integrated under #12 at `a050234ae490d650e28391f180670d1c38183192`. |
| [F12 Scientific Addendum](https://github.com/midhunpradeep/town_model/blob/7ebc2b62c175288b0673524fbfae9f9d6f15a4ff/docs/reconstruction/A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md) | SHA-256 `a4fd095dd92143c2b70ef6491b97b74a8c47f3e9a874ec71e4ca085115de30cc`; 15,631 bytes. | Twelve author-reported additions, independently accepted/integrated under #13 at `5686d60bdb3e309924d37eb600aa75fb93855340`. **F12.1–F12.8 are one grouped geometry block with no per-ID formula mapping.** |

The preserved frontier is **70 + 12 + 12 = 94 author-reported approvals across three unchanged scientific sources**. This record adds interface decisions; it neither rewrites the 94-source frontier nor validates proofs, code, numerical performance, or learning. The original A1 source remains the authority for F0–F10; the accepted separate addenda remain the F11 and F12 records. The derived [readiness frontier](A1_IMPLEMENTATION_READINESS.md) is planning, not a competing specification.

### Distinct decision statuses

| Status | Reconciliation state |
|---|---|
| **Author selected in session** | Reported by the author in the 2026-11-03 task instructions. This agent did not inspect a session transcript. |
| **Transcribed into repository/tracker** | Transcribed here and submitted to #15 as an agent-prepared proposal; exact fixed documentation SHA is provided in the #14 handoff. |
| **Durably confirmed by author** | **PENDING.** The session report and this agent-authored transcription are not confirmation. |
| **Independently reviewed as faithful** | **PENDING for this reconciliation.** The R2 draft's earlier independent review is not a review of this transcription. |
| **Authorized for a specific implementation task** | **NO.** No implementation task is authorized or marked READY by this document or by session selections alone. |

## 2. Decision-by-decision reconciliation

Each selection below is recorded as author-reported and is separately awaiting durable confirmation. “Previously approved” identifies only the applicable A1/F11/F12 mathematical source. A session-selected interface or architecture policy is not silently attributed to those sources.

### D1.1 — Immutable class binding

**Author selection — A, Immutable bound wrapper**

> The mathematical classifier remains class-ID agnostic.
>
> An immutable bound wrapper owns the exact ordered trainable-class ID tuple and canonical vocabulary/manifest identity.
>
> Persist the binding with the model, run and checkpoints. Require exact identity checks before training, inference and state restoration.
>
> Reject same-width permutations, incompatible manifests and mismatches without silently relabeling parameters or mutating restored state.

**Corresponding R2 draft:** §§3.1, 4.1, 11 D1, 13 “Class-axis binding,” and 14 D1. The draft had proposed persistence and exact checking but left classifier-state ownership versus immutable-wrapper ownership unresolved. The selection chooses the wrapper.

**Sources/status:** A1 F2.3/F4.1/F4.4 ([F2.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L29), [F4.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L54), [F4.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L57)) already establish a shared semantic domain, flat point interface, and ordered typed trainable outputs. F11.1 adds shared global predicates/evidence-supported masks but does not specify parameter-row persistence ([F11.1](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L31)); F12 §7 explicitly leaves class-order ownership unresolved ([F12 interface boundary](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L213)). The retained dataset's canonical class order and manifest identity are engineering facts (§3.1 of the draft). **New interface policy:** ownership by an immutable bound wrapper and persistence/check timing are the reported session selection, not another F0–F12 scientific approval.

**Finding and consequence:** This resolves the draft's open ownership choice. The mathematical kernel can remain class-ID agnostic; a wrapper must bind its parameter rows to the exact canonical order and identity. An output carrying today's IDs cannot prove which order trained a checkpoint. Exact comparison is required even when class counts match; mismatch must be rejected before restored model or optimizer state is mutated.

**Rejected/deferred alternatives:** Classifier-state ownership is not selected. Output-only metadata, count-only checkpoint matching, silent parameter permutation/relabeling, and mutation-then-reject restoration are rejected. A class-count-only pure kernel remains allowed only behind the enforcing wrapper.

**Affected scope/prerequisites:** Future bound classifier/result DTO, dataset-to-classifier boundary, run identity, checkpoint serialization/restoration, and training/inference preflight. No class-free pure geometry evaluator is affected. Prerequisites are a durable confirmation, a canonical serialized identity representation, and separately reviewed persistence/validation contract; none exists in active classifier code.

**Remaining ambiguity:** The selected owner and equality invariant are explicit. Engineering must specify the canonical manifest identity encoding/version and migration/error format without weakening exact identity. No scientific ambiguity is inferred from that serialization detail.

**Falsifiable future tests:** Save/restore a correctly bound checkpoint; reject a same-width permuted class-ID tuple; reject an incompatible vocabulary/manifest identity; reject before any model/optimizer state changes; prove no silent row relabeling; and show the pure kernel itself remains independent of ontology/dataset objects.

### D1.2 — Separate Boolean `DomainThing` output

**Author selection — A, Separate Boolean output**

> Expose fixed `DomainThing` membership as a distinct Boolean `[N]` field, outside the trainable `[N,C_train]` columns.
>
> It has no trainable parameters.
>
> For valid in-domain points it is true. Reconcile external mathematical queries consistently with D2-D1.
>
> Do not create a trainable `DomainThing` class row.

**Corresponding R2 draft:** §§2, 4.1, 5 D2-D1, 11 D1, 13 “Typed output contract,” and 14 D1/D2. The draft explicitly left the representation unresolved.

**Sources/status:** A1 F4.3/F4.4 already fixes `DomainThing` as parameter-free and distinguishes ordered Boolean trainable columns from family-specific measurements ([F4.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L56), [F4.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L57)). F11.1's shared-domain and evidence-supported-supervision rule does not choose this output shape; F12 §7 expressly leaves typed outputs/`DomainThing` unresolved ([F11.1](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L31), [F12 interface boundary](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L213)). **New interface policy:** a separate Boolean vector, not a row or an outer-only presentation choice, is selected in session. This does not change trainable class order or affine/geometry parameters.

**Finding and consequence:** The output now has two typed axes: trainable `contains:[N,C_train]` and fixed `DomainThing:[N]`. On valid points in the selected learned domain, the fixed field is true. Under selected D2-D1, an intentional bounded-domain query outside `Delta` is nonmembership, so the fixed field is false there; this is membership relative to the selected domain, not a claim that `DomainThing` is universally true over all of `R^d` or an unrestricted OWL universe. A bounded image adapter's invalid point is instead rejected under D2-D2 before scoring.

**Rejected/deferred alternatives:** Do not append a trainable `DomainThing` row, mix it into trainable class parameters, or collapse it into an untyped generic score. A separate outer materialization adapter is not the selected output contract.

**Affected scope/prerequisites:** Result DTOs, flat classifier evaluators, and the domain-check boundary. Requires durable confirmation of D1.2 and consistent application of D2-D1/D2-D2. It does not alter the domain's scientific set semantics.

**Remaining ambiguity:** The external-query relationship is selected as a cross-reference, but the affine interpretation of a bounded-domain query still requires the explicit scope confirmation recorded under D2-D1. Do not let `DomainThing`'s separate field bypass that guard.

**Falsifiable future tests:** For every valid in-domain row, assert fixed Boolean true and no trainable parameter/column for `DomainThing`; verify output shapes `[N]` and `[N,C_train]`; for a pure bounded external query verify false fixed membership; for a bounded adapter invalid input verify rejection before any scoring call.

### D2-A — Empty point rows

**Author selection — A, Accept empty rows**

> Pure evaluators accept `[0,d]` and return appropriately shaped empty results.
>
> This does not authorize empty `ImageBatch` inputs.

**Corresponding R2 draft:** §§4.1–5 D2-A, 11 D2-A, 12 evaluator sequence, 13 cardinality policies, and 14 D2-A.

**Sources/status:** F4.1 supplies a flat `[N,d]` mathematical boundary but does not state whether `N=0` is accepted. The draft verifies that retained `ImageBatch` validation rejects `B=0`; this is an existing image-batch engineering constraint, not the pure-evaluator rule. F11/F12 do not select pure `N=0`; F12 §7 identifies empty-input behavior as an unresolved interface boundary ([F12 interface boundary](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L213)). **New interface policy:** accept zero point rows for pure evaluation.

**Finding and consequence:** Empty point rows are a valid pure-evaluator input and must preserve the other declared axes and output dtypes. This does not relax `ImageBatch B>=1`, create empty-image batching, or imply an eligible training objective.

**Rejected/deferred alternatives:** Rejecting `N=0` at pure evaluators is not selected. Empty image batches remain disallowed; adapters may not use this choice to manufacture empty `ImageBatch` values.

**Affected scope/prerequisites:** Pure affine/box/house/town evaluators and their adapter call sites. Requires durable confirmation and exact shape/dtype contracts; does not require image pipeline changes.

**Remaining ambiguity:** “Appropriately shaped” follows each typed result variant and declared `C_train`; D2-B independently permits `C_train=0`. No training behavior follows from empty outputs.

**Falsifiable future tests:** For valid `d>=1`, pass `[0,d]` through each pure evaluator and assert correctly shaped empty membership/measurement arrays, stable dtype/device, no reductions over a nonexistent row, and no exception. Separately prove `ImageBatch` still rejects `B=0`.

### D2-B — Empty trainable class axis

**Author selection — A, Permit in pure evaluation**

> Pure evaluators may produce `[N,0]` trainable results, with fixed `DomainThing` separate.
>
> An enabled training task with no eligible supervision remains undefined under F8.

**Corresponding R2 draft:** §§4.1–5 D2-B, 11 D2-B, 13 cardinality/F8 policies, and 14 D2-B.

**Sources/status:** F4.3/F4.4 separates the fixed predicate from trainable classes. F8.4 already says an enabled task with no defined contribution remains undefined and all-undefined enabled work fails ([F8.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L98)). The draft's `[N,0]` reducer probe is only an engineering observation. F11.1 does not define zero classifier columns; F12 §7 leaves empty-input and typed-output boundaries open ([F11.1](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L31), [F12 interface boundary](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L213)). **New interface policy:** allow a zero trainable class axis for pure evaluation only.

**Finding and consequence:** Pure result shapes may have zero trainable columns while the fixed Boolean remains `[N]`. This tensor shape does not make training supervision present, convert undefined loss to zero, or satisfy an enabled objective.

**Rejected/deferred alternatives:** Requiring at least one trainable class in pure evaluation is not selected. Treating zero columns/no labels as a defined trainable task is expressly rejected by the preserved F8.4 boundary.

**Affected scope/prerequisites:** Pure result DTOs and evaluators; later objective/supervision adapters must still use known masks, eligibility, and F8.4. Requires durable confirmation before code. It does not authorize empty `ImageBatch`.

**Remaining ambiguity:** The separately fixed output exists even with no trainable IDs; D1.2/D2-D1 defines its domain-relative values. No weighting or optimizer policy is selected here.

**Falsifiable future tests:** Produce `[N,0]` trainable results plus a separate `[N]` fixed field; verify no trainable parameter is created. In a future enabled training-task fixture with no eligible supervision, verify the task is undefined and the all-undefined case fails rather than publishing a fabricated zero objective.

### D2-C — Latent dimension

**Author selection — A, Reject `d=0`**

> Require `d>=1` for classifier and geometry evaluators.
>
> Do not add artificial zero-dimensional geometric reduction conventions.

**Corresponding R2 draft:** §§4.2, 5 D2-C, 11 D2-C, 12, 13, and 14 D2-C.

**Sources/status:** A1 F2.8 makes dimension configurable, F3.10 permits useful degenerate fixed-corner fixtures, and F5.1/F5.3 use coordinate reductions ([F2.8](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L34), [F3.10](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L47), [F5.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L61), [F5.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L63)). Those fixed-corner allowances do not approve `d=0`. F11.3 distinguishes feasibility/optimization/generalization questions but supplies no zero-dimension convention; F12 §7 leaves evaluator input boundaries to separate decisions ([F11.3](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L39), [F12 interface boundary](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L213)). **New interface policy:** reject zero latent dimension for classifier and geometry evaluators.

**Finding and consequence:** Require `d>=1`; no empty-coordinate `min`/`max`, identity geometry, or artificial zero-dimensional reduction semantics are introduced. This is independent of accepting `N=0`, `C_train=0`, or degenerate fixed endpoints when otherwise valid.

**Rejected/deferred alternatives:** Defining an affine-only bias model or a new geometric reduction convention at `d=0` is not selected. The existing image model's dimension validation is not treated as the approval source.

**Affected scope/prerequisites:** Every pure classifier/geometry evaluator, transform/evaluation API, and later image adapter's dimension handshake. Requires durable confirmation; no production change is made here.

**Remaining ambiguity:** None in the selected lower bound. Per-evaluator maximum supported dimension and measured float32 range remain engineering declarations under D2-E, not inferred numerical guarantees.

**Falsifiable future tests:** Reject `d=0` at each public mathematical evaluator boundary before arithmetic; accept declared positive dimensions in supported test fixtures; prove no zero-coordinate reduction fallback exists.

### D2-D1 — External mathematical query outside bounded `Delta`

**Author selection — A, Return nonmembership**

> An intentional pure mathematical query outside bounded `Delta` returns false membership.
>
> Reconcile the precise semantics with F3.1/F4.2.
>
> The affine classifier presents a possible interpretation issue: a raw positive logit outside `Delta` does not imply membership in a class region restricted to `Delta`.
>
> Proposed consistent definition:
>
> `contains_c(x) = in_domain(x) AND (affine_logit_c(x) >= 0)`
>
> In-domain behavior remains exactly the approved affine threshold.
>
> **Important:** Treat this as a mathematical interpretation requiring an explicitly recorded scope confirmation, not as an automatically authorized alteration to F4.2.
>
> Do not silently change the underlying affine scientific definition.
>
> Geometric signed clearances remain separately typed measurements, not Boolean substitutes.

**Corresponding R2 draft:** §§2 F3.1/F4.2 semantics, 4.2 domain rule, 5 D2-D, 11 D2-D, 13 D2 policies, and 14 D2-D. The draft had left pure-query policy and adapter-invalid-input policy entangled; this record keeps them separate.

**Sources/status:** A1 F3.1 says affine baseline class decisions correspond to halfspaces restricted to `Delta`; F3.2 requires exact Boolean membership; F4.2 records the affine `z_c(x)=w_c·x+b_c` and `z_c>=0` threshold ([F3.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L38), [F3.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L39), [F4.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L55)). F11.1 establishes a shared domain for class predicates but does not define an external-query extension ([F11.1](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L31)); F12 §7 leaves point-domain behavior unresolved ([F12 interface boundary](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L213)). The selected outside-query policy is an additional interface/query policy. The exact affine extension remains an **author-controlled interpretation requiring scope confirmation**; it is not silently promoted into F4.2.

**Finding and consequence:** For intentional pure bounded-domain queries, all class membership results outside `Delta` are false, even if an affine raw logit is positive. Inside `Delta`, the approved affine threshold remains unchanged. The displayed `in_domain AND threshold` equation is the proposed consistency expression for confirming this scope; it must not be implemented or represented as an amendment to F4.2 until the author explicitly confirms that reading. For `DomainThing`, D1.2 is likewise true on valid in-domain points and false for an outside-domain pure query. A bounded image adapter's invalid encoder output follows D2-D2 and is rejected, not scored as false.

**Rejected/deferred alternatives:** Do not treat a positive outside-domain affine logit as membership in a bounded class region; do not clip/repair the query; do not threshold signed geometric clearance to invent Boolean membership; do not silently rewrite F4.2. Rejection as the pure mathematical query policy was not selected.

**Affected scope/prerequisites:** Pure bounded affine/box/town membership queries and fixed `DomainThing` output; requires an explicit, durable scope-confirmation answer (recorded under #15 and, if the author chooses, by amending/confirming the exact #3 proposal) before the affine expression is specified for implementation. This does not change `R^d` behavior, where finite points are in the selected mathematical domain subject to declared numerical validation.

**Remaining ambiguity — substantive conflict preserved:** The source can be read as (i) a restricted halfspace set whose membership is false outside `Delta`, or (ii) an unqualified raw-logit threshold for arbitrary mathematical queries. The selection chooses false for outside bounded queries but expressly requires scope confirmation for the affine interpretation. No agent resolves that tension by changing the underlying affine definition.

**Falsifiable future tests (blocked on scope confirmation for affine):** For geometry and fixed membership, any finite outside-`Delta` query returns false without altering the supplied point. After explicit affine scope confirmation, test an in-domain point exactly at `z=0` remains true, and test outside points with both positive and negative raw logits return false while exposing unchanged raw logits. Until confirmation, mark the latter as a pending contract test, not an accepted criterion.

### D2-D2 — Invalid bounded-adapter input

**Author selection — A, Reject input**

> A bounded image/classifier adapter must reject unexpected out-of-domain points before scoring.
>
> No clipping, silent repair or fabricated negative prediction.
>
> Distinguish this from D2-D1's intentional mathematical query.

**Corresponding R2 draft:** §§4.2, 5 D2-D, 7 image adapter, 11 D2-D, 13 adapter/domain checks, and 14 D2-D.

**Sources/status:** A1 F2.7 approves bounded and unbounded domains as distinct conditions; F2.5 restricts point evidence to visual inputs and authorized pixel-derived features ([F2.5](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L31), [F2.7](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L33)). R2 verified retained `ImagePointModel` outputs are bounded by existing `tanh` heads, an engineering fact that does not define future bounded adapter error handling. F11.1's shared-domain rule does not select adapter error behavior, and F12 §7 leaves point-domain integration open ([F11.1](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L31), [F12 interface boundary](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L213)). **New interface policy:** reject an unexpected out-of-domain adapter point before classifier scoring.

**Finding and consequence:** A pure mathematical query intentionally outside bounded `Delta` returns false under D2-D1; an encoder/adapter violating a bounded-run contract is malformed input and must fail before scoring. This prevents a point-head/domain failure from being hidden as a negative class result.

**Rejected/deferred alternatives:** Pass-through nonmembership with a warning, clipping, silent repair, and fabricated negative predictions are rejected. This does not authorize changes to the retained point architecture or any route.

**Affected scope/prerequisites:** Future bounded image adapter and any bounded classifier orchestration. Requires durable confirmation of the session transcription, declared domain/run identity, and pre-score validation. No effect on explicit pure set queries.

**Remaining ambiguity:** Engineering must define how domain failure is surfaced and prove that checks precede scoring; the fail-closed task lifecycle stays governed by T5. No permitted tolerance/repair is implied.

**Falsifiable future tests:** Provide an out-of-`Delta` encoder point to a bounded adapter and assert a typed rejection before classifier invocation, with no clipping, scoring, state mutation, or output publication. In the same test suite, show the separate pure evaluator returns false for an intentional query and does not mutate the input.

### D2-E — Strict input validation and numerical policy

**Author selection — A, Strict validated inputs; initial float32 affine**

> Require strict rank/shape, finite-value, dimension, dtype, device and declared numerical-envelope checks.
>
> No silent casting, broadcasting, clipping or physical geometry repair.
>
> Geometry float32 was already approved under F7.3.
>
> Initial float32 affine calculations are newly selected here.
>
> Actual finite operating limits and hardware conformance remain engineering validation tasks, not invented constants or universal numerical guarantees.

**Corresponding R2 draft:** §§4.2, 5 D2-E, 11 D2-E, 13 validation/numerical tests, and 14 D2-E.

**Sources/status:** A1 F7.3 already requires geometry transforms/evaluation/loss in float32 and an explicit tested finite operating range ([F7.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L84)); F7.4 distinguishes exact boundary fixtures from transform limits. The grouped F12.1–F12.8 geometry block also records float32 and tested-envelope obligations ([F12.1–F12.8](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L36)), without mapping formulas per individual ID. F11.8 distinguishes exact stored-endpoint certificates from tolerance estimates, but does not supply universal input limits ([F11.8](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L59)). **New interface policy:** strict shape/value/dtype/device/envelope validation and initial float32 affine calculations are session selections; float32 affine was not already an F7.3 approval.

**Finding and consequence:** Exact declared rank/shape and compatible dimensions must be checked; nonfinite or out-of-declared-envelope values, unsupported dtype/device, and incompatible device placement reject. Inputs are not silently cast, broadcast, clipped, physically repaired, or transferred. Geometry remains float32 as approved. Initial affine calculations are float32 as newly selected, pending durable confirmation. The finite operating limits, supported hardware/device set, and demonstrated conformance are not assigned invented constants here.

**Rejected/deferred alternatives:** Implicit framework casting/broadcasting, clipping, coordinate repair, silently moving devices, and a universal “all finite float32 inputs are safe” guarantee are rejected. No arbitrary numeric limit is selected by this document.

**Affected scope/prerequisites:** Every pure evaluator/classifier validator, geometry input boundary, later affine calculation, and image adapter. Requires durable confirmation and engineering-owned, declared test envelope plus hardware-specific conformance evidence; no implementation is ready from this record.

**Remaining ambiguity/obligations:** Actual finite operating limits, affine conditioning bounds, supported device/hardware matrix, and exact failure types/messages require engineering validation. Do not present F7.3/F12 float32 as proof of behavior on all magnitudes or devices; do not infer an affine dtype approval from F7.3.

**Falsifiable future tests:** For each public input, test rank/shape and dimension mismatches, NaN/±Inf, each unsupported dtype/device, cross-device tensors, and values immediately inside/outside the declared envelope. Assert explicit rejection before computation, no cast/broadcast/clip/repair/transfer, geometry dtype exactly float32, initial affine dtype exactly float32 after durable confirmation, and expected boundary results across the declared hardware conformance matrix.

### D3 — Identity terminal heads for unbounded image points

**Author selection — A, Identity terminal heads**

> Preserve existing bounded `tanh` point-producing pathways.
>
> For unbounded domain runs, use identity terminal activations in the relevant component, global, composition and hybrid point-producing heads.
>
> Do not introduce a finite surrogate for `R^d`.
>
> Record domain/activation identity in run configuration and checkpoints.
>
> Do not silently reinterpret bounded checkpoints as unbounded checkpoints.
>
> Keep the shared point architecture, identities, masks and composition machinery unchanged except for the separately authorized activation behavior.
>
> This choice authorizes specification of the future implementation contract, not the implementation itself.

**Corresponding R2 draft:** §6, §11 D3, §13 “Domain-specific image heads,” and §14 D3. The draft correctly distinguished F2.7's approved domains from the retained all-`tanh` engineering fact, but left the unbounded point-head policy unresolved.

**Sources/status:** A1 F2.7 already approves bounded `[-1,1]^d` and unbounded `R^d` as distinct scientific conditions; F2.4/F2.5 preserve point identity and visual-input boundaries ([F2.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L30), [F2.5](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L31), [F2.7](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L33)). Retained terminal `tanh` heads are an existing engineering fact. The grouped F12.1–F12.8 approval specifies geometry transforms, not the image point-head terminal activation ([F12.1–F12.8](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L36)); F12 §7 leaves bounded/unbounded point behavior unresolved ([F12 interface boundary](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L213)). **New interface/architecture policy:** identity terminal heads for unbounded image runs are the reported session selection, not an F12 geometry approval or implementation authorization.

**Finding and consequence:** Bounded outputs remain exactly on the current `tanh` path. Unbounded image runs replace only the relevant final point-producing activations (component, global, composition/DeepSets, and hybrid) with identity. Shared architecture, parameters, identities, masks, and composition remain unchanged except for that activation behavior. A bounded checkpoint with compatible tensor shapes is semantically incompatible with identity-head interpretation and must not be silently relabeled or reused as unbounded.

**Rejected/deferred alternatives:** Finite rescaling, clipping, or a finite surrogate for `R^d` is rejected. Leaving all unbounded image integration deferred was an option in the draft but is not the session-selected direction. This selection still requires durable confirmation before an implementation task.

**Affected scope/prerequisites:** `src/town_model/models/image_points.py` terminal heads and domain-aware configuration/run/checkpoint identity; later image integration and bounded/unbounded comparisons. Prerequisites include durable decision confirmation, a separate narrowly scoped implementation issue, exact checkpoint compatibility policy, and independent review. No point code is changed here.

**Remaining ambiguity:** “Relevant” heads are the point-producing paths that can feed selected global/compositional/hybrid mode; the future contract must test each path and state whether inactive heads are constructed or evaluated. Matching initialization protocol and numerical operating range remain engineering work. The selected domain/activation identity is not optional metadata.

**Falsifiable future tests:** Bounded outputs/checkpoint behavior remain unchanged; each unbounded output head has identity terminal activation and no hidden clip/scale; all modes preserve IDs/masks/composition/gradients; the run/checkpoint identity records domain and activation; restoring bounded state into an unbounded run and vice versa rejects before state mutation; demonstrate points outside `[-1,1]^d` can be emitted without finite surrogate clipping.

### D4 — Separate component and pizza classifier calls

**Author selection — A, Separate component/pizza calls**

> Use separately organized component and pizza flat-point classifier calls initially.
>
> Both calls use the same shared classifier state and global class geometry.
>
> Preserve row identity, occurrence multiplicity, masks, padding exclusion and gradient flow.
>
> Do not introduce separate class-region parameters for components and pizzas.

**Corresponding R2 draft:** §7, §11 D4, §13 “Point adapter,” and §14 D4. The draft left combined versus separate calls unresolved and recommended separate calls as a starting engineering choice.

**Sources/status:** A1 F2.3/F2.4/F2.5/F4.1 establishes a shared semantic domain, one point per observed entity/occurrence, visual inference boundaries, and a flat mathematical interface ([F2.3](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L29), [F2.4](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L30), [F2.5](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L31), [F4.1](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L54)). F11.1 constrains shared predicates and evidence-supported supervision but does not choose call organization ([F11.1](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L31)); F12 §7 likewise leaves adapter integration boundaries separate ([F12 interface boundary](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L213)). **New adapter policy:** two separately organized calls, sharing one classifier state/global class geometry, are selected; they do not create distinct component and pizza class-region models.

**Finding and consequence:** Component occurrences are flattened/scored in one adapter path and pizza points in another, each through the same bound classifier and global class regions. Row maps preserve sample/occurrence identity and multiplicity. Invalid/padded rows are excluded; both calls remain differentiable. Separate calls organize the adapter only and do not alter F4.1's point-only mathematical classifier.

**Rejected/deferred alternatives:** A single combined component+pizza flat call is not the initial selection. Separate classifier instances, component-specific/pizza-specific trainable class regions, dropping repeated occurrences, scoring padded entries, or detaching points are rejected.

**Affected scope/prerequisites:** Later image-point adapter, target/output reconstruction, shared class binding from D1.1, and gradient integration. Requires durable confirmation and a separate adapter contract; no classifier/image route is connected now.

**Remaining ambiguity:** The selection fixes separate call organization and shared classifier state; batching, call order, and exact row-map DTO are engineering details. Neither may duplicate or split classifier parameters.

**Falsifiable future tests:** Use repeated occurrence IDs/rows and verify multiplicity/identity survives; prove padding and masked slots are never scored; verify component and pizza calls reference the same classifier parameters and geometry; backpropagate both outputs and verify gradients reach the shared state and valid point rows without detach; verify class outputs map back to correct source rows.

### D5 — Reuse reducers with explicit objective/supervision adapters

**Author selection — A, Reuse existing reducers with adapter**

> Preserve retained generic reducers.
>
> Use explicit objective and supervision adapters for classifier-specific penalties, known labels, trainable class projection, positive weights, entity/image aggregation and undefined-task handling.
>
> Require end-to-end F8 conformance tests.
>
> Do not claim the current reducers alone already implement the full approved training objective.

**Corresponding R2 draft:** §8, §11 D5, §13 “Reducer/objective adapter,” and §14 D4/D5 scope.

**Sources/status:** A1 F5.2 separates geometry, objective, reducer, optimization, and evaluation ownership; F6 defines the primary objective families; F8.1–F8.5 defines masks, weights, entity/image hierarchy, task coefficients, undefined tasks, and accumulation ([F5.2](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L62), [F6](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L68), [F8.1–F8.5](A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L93)). R2's code inspection shows current reducers accept caller values; they do not construct classifier penalties or all F8 adapters. F11.1 also requires evidence-supported known/unknown masks and no automatic cross-type negatives, without defining the objective adapter ([F11.1](A1_F11_SCIENTIFIC_ADDENDUM_2026-11-01.md#L31)); F12's optimizer lifecycle approvals do not mean the F8 objective/reducer adapter exists ([F12 readiness](A1_F12_SCIENTIFIC_ADDENDUM_2026-11-01.md#L194)). **New implementation boundary:** preserve the generic reducers and add explicit objective/supervision adapters with end-to-end F8 tests.

**Finding and consequence:** Existing reducers are reusable infrastructure, not a complete classifier objective. A future adapter must project canonical trainable classes and known masks, compute only the approved family-specific penalties, apply only configured positive weights, reduce eligible entities, aggregate components within images then eligible images, handle pizza task separately, and preserve F8.4 undefined-task behavior. Generic reducer semantics need not be rewritten unless a falsifiable conformance test shows the API cannot support the approved contract.

**Rejected/deferred alternatives:** Replacing retained generic reducers preemptively is not selected. Claiming their current caller-value APIs alone implement the full objective is expressly rejected. No implicit loss, weighting, task renormalization, or optimizer behavior is selected.

**Affected scope/prerequisites:** Later objective/supervision adapter, dataset class/mask projection, component/pizza image aggregation, positive-weight computation, diagnostics, and training integration. Requires durable confirmation, accepted typed classifier outputs, exact D1 binding, and a bounded F8 conformance issue. This does not authorize optimizer, training, or route reconnection.

**Remaining ambiguity:** The selected adapter boundary is clear. Exact module/API placement and diagnostics representation are engineering details. F8 numerical formulas and undefined-task rules remain governed by A1; no complete implementation exists.

**Falsifiable future tests:** End-to-end fixtures for all five primary objective conditions; known/unknown masking and trainable projection; unit and configured positive weights; component mean within each eligible image then eligible-image mean; independent pizza aggregation; no coefficient renormalization; undefined enabled task omission/logging and all-undefined failure; gradients and effective-batch behavior where in scope. Also assert that generic reducers continue to consume caller values and are not silently converted into classifier-specific objectives.

## 3. Issue #3 — exact existing corrected proposal for author confirmation

**Status: confirmation pending.** The author-reported session selection for #3 is **A — Confirm as recorded**. The corrected proposal below is reproduced from the existing [complete corrected proposed resolution, comment 6561634838](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6561634838), rather than the older comments with escaping or source-link transcription errors. Its wording is an agent-prepared proposal, not an author-signed resolution. The later source-pointer comments [6561664377](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6561664377) and [6561667693](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6561667693) pin its unchanged decision citations to the final R1 source revision `be4bbc067d235cbd896665ba33f6585b2cabc4e6`.

## Corrected proposed author resolution — confirmation pending

This complete proposal supersedes the earlier agent comments' formula-transcription and F4-link errors; those comments remain preserved in the journal. It is prepared by the R1 implementation agent, **not an author-signed issue resolution**. The A1 source itself is author-approved, but issue-level confirmation remains pending. Please confirm or amend this wording; issue #3 remains open.

**Canonical source:** [A1 Scientific Decision Ledger — 70 Active Approvals at fixed R1 revision `0fa19f98cbf0d3acde0276deed18136862736bec`](https://github.com/midhunpradeep/town_model/blob/0fa19f98cbf0d3acde0276deed18136862736bec/docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md).

### Proposed resolution within issue #3's scope

1. **Membership is an exact Boolean set test.** F3.2 makes membership in the exact geometric class set the prediction. A signed score or tolerance is not a substitute for this Boolean result.
2. **Town set operations use F3.5 and active F3.6-R.** A house is its closed frame set-differenced by the union of strict ambient `ℝ^d` window interiors: `H_h = F_h \ ⋃_w int_{ℝ^d}(W_hw)`. The town is the union of its houses: `T = ⋃_h H_h`. Frame-boundary points are retained. Window-boundary points are retained unless they lie in another window's strict interior. Independent windows outside a frame do not remove out-of-frame points.
3. **F3.6 is superseded.** F3.6-R replaces the previous closed-window-interior exclusion; it is the one active boundary convention, not an alternative to F3.6.
4. **Zero windows are defined.** F3.11 requires `W=0` to work and `H=1, W=0` to be exactly one box, with no fake window state.
5. **Boolean predictions and numerical measurements remain separately typed.** F4.1 takes flat `[N,d]` points; F4.2 defines affine logits `z_c(x)=w_c·x+b_c` and the baseline's hard prediction `z_c≥0`; F4.3 makes `DomainThing` fixed-positive relative to the selected learned domain; F4.4 separates ordered Boolean `contains` from family-specific numerical outputs. For geometry, `contains` follows exact set membership. Signed coordinate clearance is a separate measurement under F5.1/F5.3/F5.6, not a thresholded replacement for membership.

### Direct decision references

- [F3.2 — exact set membership](https://github.com/midhunpradeep/town_model/blob/0fa19f98cbf0d3acde0276deed18136862736bec/docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L39)
- [F3.5 — town construction](https://github.com/midhunpradeep/town_model/blob/0fa19f98cbf0d3acde0276deed18136862736bec/docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L42)
- [F3.6-R — open window interiors and retained boundaries; supersedes F3.6](https://github.com/midhunpradeep/town_model/blob/0fa19f98cbf0d3acde0276deed18136862736bec/docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L43)
- [F3.11 — `W=0` / single-box equivalence](https://github.com/midhunpradeep/town_model/blob/0fa19f98cbf0d3acde0276deed18136862736bec/docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L48)
- [F4.1–F4.4 — point interface, affine baseline, fixed `DomainThing`, typed outputs](https://github.com/midhunpradeep/town_model/blob/0fa19f98cbf0d3acde0276deed18136862736bec/docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L54)
- [F5.1/F5.3/F5.6 — signed coordinate clearance, min/max composition and raw physical units](https://github.com/midhunpradeep/town_model/blob/0fa19f98cbf0d3acde0276deed18136862736bec/docs/reconstruction/A1_SCIENTIFIC_DECISION_LEDGER_2026-11-01.md#L61)

This proposal adds no approval beyond the cited A1 decisions and resolves no question outside this ticket. **Author confirmation is pending; do not close #3 based on an agent comment.**

**Reconciliation finding:** This proposal faithfully restates the cited existing approvals within its declared scope and adds no mathematical approval. The separate D2-D1 external-query issue is not smuggled into #3 item 5: the proposal states the in-domain affine threshold and `DomainThing` relation but does not settle how raw affine logits outside bounded `Delta` are extended. Keep that interpretation as an explicit author-controlled scope confirmation under D2-D1. Do not mark #3 resolved until the author confirms or amends the corrected proposal in a durable author-attributable comment. No agent comment in this task is such confirmation.

## 4. Cross-cutting contradiction, numerical, and fail-closed audits

### F3.1/F4.2 and out-of-domain affine membership

The selected pure-query policy says false outside bounded `Delta`; F3.1 describes affine halfspaces restricted to `Delta`, while F4.2 states the raw affine threshold. A positive raw logit outside `Delta` therefore exposes a genuine scope/interpretation question. `contains_c(x) = in_domain(x) AND (affine_logit_c(x) >= 0)` is a consistent proposed membership extension that preserves the in-domain threshold, but it is **not** silently treated as an approved change to F4.2. The decision issue must obtain explicit author scope confirmation before any future affine evaluator encodes that extension.

### F4.3 `DomainThing` and external queries

The selection keeps `DomainThing` fixed and true on valid points in the selected learned domain, separate from all trainable columns. Consistency with D2-D1 makes an intentional query outside bounded `Delta` false for this domain-relative membership. This is not a trainable class row or a universal assertion about every point in `R^d`/an unrestricted ontology universe. A bounded adapter's unexpected out-of-domain input follows D2-D2 rejection instead.

### D1 exact checkpoint-to-column identity

The corrected R2 draft already remedied the first review finding by requiring a persistent tuple/manifest binding and pre-mutation exact checks. The 2026-11-03 selection resolves the remaining owner to the immutable bound wrapper. Required future failure cases include same-width permutations, incompatible manifests, checkpoint restore/inference identity mismatch, and proof that the rejected restore leaves parameters and optimizer/restored state untouched. Output metadata alone is insufficient.

### D2 empty axes versus undefined training supervision

`N=0` pure evaluator rows and `C_train=0` pure result columns are selected as shape policies only. They do not authorize `ImageBatch B=0`; they do not change `K=0` retained point-path behavior or F3.11's `W=0` set semantics; and they do not make an enabled task with no eligible supervision defined. F8.4 remains controlling for objective/task eligibility.

### D3 bounded/unbounded point and checkpoint semantics

Bounded image points retain `tanh`; unbounded-domain point heads use identity only after the future D3 implementation contract is approved. Configuration/checkpoint identity must distinguish domain and activation. Matching tensor shapes do not establish semantic compatibility. No bounded checkpoint is silently interpreted as unbounded, and no finite surrogate for `R^d` is introduced.

### D4 shared classifier across separate calls

The two adapter calls are separate organization, not separate classifiers. They share the exact same immutable binding, trainable class parameters, and global class geometry. There must be no component/pizza-specific region parameters. Row identity, occurrence multiplicity, valid masks, padding exclusion, and gradients must survive both calls.

### D5 partial versus complete F8 conformance

The retained reducers are generic caller-value reducers. They cover some masking, weighting, eligibility, and aggregation mechanics; they do **not** alone produce the approved family-specific penalties, canonical known-label projection, all F8 weighting, component-within-image/entity hierarchy, full diagnostics, or task-specific undefined handling. Future end-to-end F8 tests are required; do not claim the reducers alone already implement the full objective.

### F12/F7.3 numerical obligations

F7.3 and grouped F12.1–F12.8 already require float32 geometry and a finite, explicitly tested operating range. The session newly selects initial float32 affine calculations and strict validation policies. None supplies universal numerical guarantees or actual safe limits. Later engineering must declare and empirically validate operating envelopes, supported device/hardware behavior, boundary/ULP budgets where applicable, and rejection behavior. This reconciliation invents no constants, silently extends no affine scientific definition, and makes no hardware-conformance claim.

### Complete T5 fail-closed route matrix

The [accepted T5 review §J](../../reviews/classifier_teardown_t5_final_acceptance.md#j-unavailable-operation-matrix-and-artifact-non-publication) independently verified this complete 12-route CLI matrix. Every future pure evaluator/reconstruction task must retain the entire unavailable-route/no-side-effect boundary; this documentation task does not invoke or reconnect any route.

| T5 operation | Route that remains unavailable |
|---|---|
| Synthetic fixed experiment/training | `experiment fixed` |
| Synthetic point-mapping experiment | `experiment mapping` |
| Image experiment preflight | `experiment image preflight` |
| Image training | `experiment image run` |
| Image experiment matrix | `experiment image matrix` |
| Image resume | `experiment image resume` |
| Historical image checkpoint evaluation | `experiment image evaluate` |
| Image worker stop | `experiment image stop` |
| Image relation diagnosis | `experiment image relations` |
| Synthetic evaluation | `evaluate` |
| Synthetic resume | `resume` |
| Synthetic geometry export | **`export-geometry`** |

The future verification contract must also preserve the accepted direct-API rejection matrix (including construction, training/evaluation, checkpoint reconstruction, run/resume, worker/matrix/stop, snapshot generation and relation diagnosis) and disabled dashboard surfaces. Unauthorized operations must fail with `ClassifierBackendUnavailableError`/the existing CLI presentation before side effects. Across the complete matrix, test both absent outputs and pre-existing sentinels: no files/artifacts created, no existing output/state mutated. `export-geometry` is explicitly included. The matrix here is reviewed against T5 evidence, not executed by the reconciliation agent.

## 5. Remaining engineering obligations and implementation boundary

### Numerical work not settled by selection

- Declare actual float32 operating limits for geometry and the selected initial affine path; test finite values, exact shape/dtype/device, conditioning, and boundary behavior. Do not invent universal constants in this record.
- Establish hardware/device conformance for the declared envelope. F7.3/F12 float32 approval is not a blanket CPU/GPU/platform guarantee.
- Define canonical manifest identity serialization/version and pre-mutation checkpoint validation under D1.1.
- Verify D3 activation/domain persistence and checkpoint rejection without automatic bounded-to-unbounded reinterpretation.
- Test D4 gradient and row-identity preservation over separate calls sharing one state.
- Test complete F8 objective/supervision adapter behavior; retained reducers alone are not that evidence.

### Candidate later implementation slices — none READY

These candidates are planning only; each requires a separate issue, exact confirmed scope, native blocker checks, a claim, dedicated implementation/review, and the complete applicable T5 route matrix. **No candidate is marked READY here.**

1. Pure physical-corner box membership and clearance evaluator; decide/encode only the confirmed D2 input and query semantics; no transform or optimizer.
2. Pure house/town membership and clearance evaluator; preserve active F3.6-R, exact boundaries and `W=0`; no trainable geometry.
3. Pure affine point evaluator with exact F4.2 in-domain threshold, selected D2 shapes/dtypes and immutable binding boundary. Keep the external affine scope blocked until D2-D1 is durably confirmed.
4. Immutable bound-wrapper serialization/validation for class-column identity and separate fixed `DomainThing` output, after D1 confirmation.
5. Domain-specific point-head activation contract/implementation for D3 in a separate task; preserve bounded behavior and checkpoint distinction.
6. Separate component/pizza adapter calls using shared classifier state, after D1/D2/D3 prerequisites and D4 confirmation.
7. Objective/supervision adapter with end-to-end F8 conformance, after typed outputs and D5 confirmation.
8. Training, optimizer lifecycle, detector, Phase 4, scientific evaluation, and operational route reconnection only in later separately approved tasks. They are not implied by this record.

### Exact non-authorization boundary

This task does not implement pure evaluators, classifier code, point-head changes, geometry transforms, objectives, optimizers, training, detector work, or route reconnection. It does not promote this transcription, either proposed ADR, or the corrected R2 draft into accepted scientific authority. A source-reported session selection is not durable confirmation; a durable decision is not independent review; independent review is not authorization for a specific implementation task. No dependent issue is READY solely because of this document or #15's existence.

## 6. Proposed ADRs — not accepted

The decisions have architectural consequences, so two ADR drafts accompany this reconciliation. Both are explicitly **PROPOSED**, link to #15, and remain non-authoritative until the author confirms the exact scope and the repository authority/review procedure is satisfied. They do not authorize work.

- [ADR-0001 — Immutable bound wrapper for classifier class-column identity](../adr/0001-immutable-class-binding.md)
- [ADR-0002 — Domain-specific terminal activations for image point heads](../adr/0002-domain-specific-image-point-heads.md)

## 7. Verification record and review handoff

### Pre-submission checks

- Rechecked remote `main`, R2 draft, first review, and re-review refs against the exact SHAs in §1.
- Verified A1/F11/F12 file SHA-256 identities at the fixed starting revision and preserved the 94-source frontier and grouped F12.1–F12.8 caveat.
- Read the execution protocol, decision-authority protocol, original A1 ledger, completeness audit, readiness frontier, accepted F11/F12 addenda, R2 draft, both independent R2 reports, T5 review, and live #2/#3/#14 states.
- Reproduced and linked the corrected #3 proposal only; earlier typo-bearing comments are not used as authority.
- Created #15 as an OPEN, unassigned `[AUTHOR DECISION]` Wayfinder child of #2, linked to #14, after checking that no duplicate R2 interface decision ticket existed. #3 remains separate.
- No source ledger, accepted review report, production source, test, dataset, artifact, current-status/acceptance authority, or T5 operation was modified or run.

### Required independent review

Request a fresh independent reviewer to verify the exact documentation commit, including:

1. every session selection is transcribed without omission or reversal and remains marked author-reported/pending confirmation;
2. R2 section/report/source links and A1/F11/F12 source identity/frontier are exact;
3. the F3.1/F4.2 external affine ambiguity and F4.3 `DomainThing` scope are explicitly preserved without silently changing science;
4. D1 checkpoint-to-class identity and D3 bounded/unbounded checkpoint semantics are enforceable and correctly scoped;
5. D2 empty axes remain distinct from F8 undefined supervision; D4 shares one classifier; D5 does not overclaim F8;
6. float32 and numerical-envelope engineering obligations are not converted into invented guarantees;
7. complete T5 failure coverage includes `export-geometry` and no-side-effect checks;
8. only documentation files changed, ADRs are PROPOSED, no T0–T5 protected code boundary changed, and #2/#3/#14/#15 remain in the stated states.

The author-confirmation gate remains explicit. Keep #15, #3, and #14 open. Do not mark any future implementation slice READY or close an author-decision ticket on an agent's authority.
