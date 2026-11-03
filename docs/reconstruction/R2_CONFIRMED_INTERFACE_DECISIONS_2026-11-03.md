# R2 Confirmed Interface Decisions — 2026-11-03

> **Current authority:** the author-confirmed R2 specifications below. This index reconciles the fixed draft, confirmation comments, ADRs, and independent review history. It approves specifications only—not implementation, route reconnection, training, or scientific evaluation.

## Authoritative confirmations

- **R2 selections D1.1–D5, including bounded-affine query scope:** author-confirmed in [#15 comment 6597917380](https://github.com/midhunpradeep/town_model/issues/15#issuecomment-6597917380). That comment expressly confirms the reported selections in [#15 comment 6597555391](https://github.com/midhunpradeep/town_model/issues/15#issuecomment-6597555391), the reconciliation at [`af0865439be085bc59434b6264e4bd5d2df91c59`](https://github.com/midhunpradeep/town_model/blob/af0865439be085bc59434b6264e4bd5d2df91c59/docs/reconstruction/R2_AUTHOR_SELECTION_RECONCILIATION_2026-11-03.md), and the architectural choices represented by ADR-0001 and ADR-0002.
- **Membership/set resolution:** author-confirmed in [#3 comment 6597917347](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6597917347), for the corrected proposal at [#3 comment 6561634838](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6561634838).
- **Planning-map pointer and current issue boundary:** [#2 comment 6597927869](https://github.com/midhunpradeep/town_model/issues/2#issuecomment-6597927869). #2 remains OPEN; #3 and #15 remain CLOSED; #14 remains OPEN for integration review. None is reopened by this integration.

The original session transcript was not supplied or independently inspected. These durable author-attributable GitHub comments are the confirmation authority. The R2 decisions are additional interface/architecture specifications; they do not amend A1/F11/F12 or alter their original **70 + 12 + 12 = 94 author-reported approvals**. F12.1–F12.8 remain a grouped block without per-ID formula mapping.

## Confirmed choices

| Decision | Confirmed specification |
|---|---|
| **D1.1 — class binding** | The mathematical kernel remains class-ID agnostic. An immutable bound wrapper owns and persists the exact ordered trainable class-ID tuple and canonical vocabulary/manifest identity with model, run, and checkpoint. Validate exact identity before training, inference, and restore; reject same-width permutations or mismatches before state mutation. See [ADR-0001](../adr/0001-immutable-class-binding.md). |
| **D1.2 — `DomainThing`** | A fixed, parameter-free Boolean `[N]` output, separate from trainable `[N,C_train]`; no trainable class row. It is true for valid in-domain points and false for intentional queries outside bounded `Δ`; an invalid bounded-adapter input is rejected under D2-D2. |
| **D2-A/B/C — empty axes/dimension** | Pure evaluators accept point rows `[0,d]` and may return zero trainable columns `[N,0]`; neither choice permits empty `ImageBatch` inputs or makes a no-supervision training task defined. Require `d >= 1`; introduce no zero-dimensional reduction convention. `DomainThing` remains separate. |
| **D2-D1 — bounded pure query** | For an intentional mathematical query outside bounded `Δ`, membership is false even if the raw affine logit is positive: `contains_c(x) = in_domain(x) AND (affine_logit_c(x) >= 0)`. On in-domain points the approved F4.2 `z_c(x) >= 0` threshold is unchanged. Preserve the raw logit unchanged; this is domain-restricted membership, not logit clipping or a new loss. |
| **D2-D2 — bounded adapter input** | A bounded image/classifier adapter rejects an unexpected out-of-domain point before scoring. This is distinct from a pure outside-domain query; do not clip, repair, or fabricate a negative prediction. |
| **D2-E — validation/numerics** | Require strict rank/shape, dimension, finite-value, dtype, device, and declared numerical-envelope validation; do not silently cast, broadcast, clip, transfer, or repair geometry. Geometry float32 remains the existing F7.3 approval; initial affine calculations use float32 by the R2 selection. Actual finite operating limits and hardware conformance must be separately declared and measured—none are invented or guaranteed here. |
| **D3 — image point heads** | Preserve bounded terminal `tanh`. For unbounded `ℝ^d` image runs, use identity terminal activations in the relevant component, global, composition, and hybrid point-producing heads. Record domain/activation in run and checkpoint identity; do not reinterpret bounded checkpoints as unbounded. See [ADR-0002](../adr/0002-domain-specific-image-point-heads.md). |
| **D4 — image adapter calls** | Organize component and pizza points as separate flat classifier calls, sharing the same classifier state and global class geometry. Preserve sample/occurrence identity and multiplicity, valid masks, padding exclusion, and gradient flow; do not create separate component/pizza class regions. |
| **D5 — reducers** | Reuse the retained generic reducers with explicit objective/supervision adapters. Require end-to-end F8 conformance for class projection, masks, penalties, weighting, entity/image aggregation, and undefined-task behavior. Existing reducers alone are not the complete classifier objective. |

## Confirmed mathematical membership and measurement contract

- Classifier inputs are flat `[N,d]` points. The affine baseline retains raw logits `z_c(x)=w_c·x+b_c`; in-domain hard membership is exactly `z_c(x) >= 0`. For bounded-domain pure queries, apply the domain restriction above and leave the raw logit unchanged.
- Geometric predictions are exact Boolean set membership, not a clearance threshold or tolerance test. Boxes are closed. Under active **F3.6-R**, a house is its closed frame minus the union of strict ambient `ℝ^d` window interiors, and a town is a union of houses. Frame boundaries are retained; window boundaries are retained unless they lie in another window's strict interior. A window outside a frame does not remove out-of-frame points. F3.6-R supersedes F3.6.
- `W=0` is defined; `H=1,W=0` is exactly one box, with no fake window state.
- Signed coordinate clearance is a separate numerical measurement in raw physical units, composed by the approved min/max rules. It is not Euclidean distance and never substitutes for Boolean membership.

These mathematical choices are also recorded in [#3's confirmed comment](https://github.com/midhunpradeep/town_model/issues/3#issuecomment-6597917347). No new formulas or set conventions are introduced by this index.

## Source and review provenance

The following history is preserved. Fixed source revisions remain identifiable; historical review verdicts are not rewritten.

| Evidence | Fixed revision | Scope and status |
|---|---|---|
| [Corrected R2 draft](R2_CLASSIFIER_INTERFACE_CONTRACT_DRAFT.md) | [`fb432286113f9d11d433ab9e3499e97c7275dba2`](https://github.com/midhunpradeep/town_model/blob/fb432286113f9d11d433ab9e3499e97c7275dba2/docs/reconstruction/R2_CLASSIFIER_INTERFACE_CONTRACT_DRAFT.md) | Preserved exactly as a historical draft; it was not rewritten to imply its proposals were approved when drafted. |
| [Earlier independent draft review](../../reviews/r2_classifier_interface_draft_independent_review.md) | [`b0636d234bcbe31b80f39b30c0967da7171bd6bb`](https://github.com/midhunpradeep/town_model/blob/b0636d234bcbe31b80f39b30c0967da7171bd6bb/reviews/r2_classifier_interface_draft_independent_review.md) | **REQUEST DRAFT CORRECTIONS** against original draft `c48b4c7f275ea70d92462e7e20cda008039640aa`; retained as historical evidence, not deleted. |
| [Corrective draft re-review](../../reviews/r2_classifier_interface_draft_rereview.md) | [`0cc2680a67d7ad198cfa16afb5f69dcbd4e99b59`](https://github.com/midhunpradeep/town_model/blob/0cc2680a67d7ad198cfa16afb5f69dcbd4e99b59/reviews/r2_classifier_interface_draft_rereview.md) | **VERIFIED AS FAITHFUL DRAFT** for corrected draft `fb432…`; supersedes the earlier review verdict for the corrected draft only. It is not implementation acceptance. |
| [Author-selection reconciliation](R2_AUTHOR_SELECTION_RECONCILIATION_2026-11-03.md) | [`af0865439be085bc59434b6264e4bd5d2df91c59`](https://github.com/midhunpradeep/town_model/blob/af0865439be085bc59434b6264e4bd5d2df91c59/docs/reconstruction/R2_AUTHOR_SELECTION_RECONCILIATION_2026-11-03.md) | Its original body records a proposed transcription pending confirmation at that revision. A dated lifecycle note now links the later durable confirmations; the historical body remains intact. |
| [Reconciliation independent review](../../reviews/r2_author_selection_reconciliation_independent_review.md) | [`823de6505167509602c7937b10844a4e402193e7`](https://github.com/midhunpradeep/town_model/blob/823de6505167509602c7937b10844a4e402193e7/reviews/r2_author_selection_reconciliation_independent_review.md) | **VERIFIED AS FAITHFUL RECONCILIATION** at source revision `af086…`; it did not accept implementation. |
| [ADR-0001](../adr/0001-immutable-class-binding.md) and [ADR-0002](../adr/0002-domain-specific-image-point-heads.md) | Source: `af0865439be085bc59434b6264e4bd5d2df91c59` | Both now record **author-accepted specifications** with #15 confirmation references. Their approval is not implementation acceptance/authorization. |

The earlier independent review, corrective re-review, and reconciliation review are complete and preserved. An independent **integration** review of the exact branch revision is still required; the integration agent must not self-accept.

## Implementation and operational boundary

The accepted T0–T5 classifier-free operational baseline remains in force ([T5 final acceptance](../../reviews/classifier_teardown_t5_final_acceptance.md)). This documentation integration does not implement or reconnect classifiers, geometry evaluators, trainable geometry, objectives, optimizers, point-head activation changes, training/evaluation, dataset/ROI/detector work, CLI/API/dashboard classifier operations, geometry export, or interpretation routes. Phase 4, detector development, and scientific training remain on hold. All disabled operations must remain disabled. Author approval of these specifications and review of their documentation do not constitute acceptance of an implementation.

No implementation issue is made READY by this integration. The live `workflow:ready` frontier was empty at the integration start; readiness must be checked again at handoff. #14 remains open pending independent integration review and coordinator action; #2 remains open. Do not reopen #3 or #15.

### Proposed next implementation ticket — not READY

Propose a small, pure **physical-corner box Boolean membership evaluator** as the first separately scoped implementation ticket. Follow it in separate tasks with signed-coordinate box clearance, then house/town membership and clearance. Keep trainable geometry/transforms, image integration, objective/reducer adapters, optimizer work, and training out of the first slice. This index creates no issue and authorizes no implementation.
