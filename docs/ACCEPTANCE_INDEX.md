# Acceptance index

This index identifies the accepted implementation revisions and the independent review records that establish their status. Implementation SHAs and later report-only review commits are intentionally listed separately. Links to historical documents preserve their original wording and verdicts; those files have not been rewritten to make them look current.

## Authoritative accepted revisions

| Milestone | Accepted implementation SHA | Independent acceptance record | Scope of acceptance |
|---|---|---|---|
| Phase 1 | [`39b2adb739475632e7183254c5c792a2569826de`](https://github.com/midhunpradeep/town_model/commit/39b2adb739475632e7183254c5c792a2569826de) | Review commit [`bf2e3080a07734c50b9c679aa812d1e1e933db91`](https://github.com/midhunpradeep/town_model/commit/bf2e3080a07734c50b9c679aa812d1e1e933db91); [`reviews/phase_01_final_closure_review.md`](../reviews/phase_01_final_closure_review.md) | Pizzaïolo dataset preparation, semantic labels/masks, validation, CLI and initial dashboard. The review records a minor nonblocking sparse-checkout diagnostic limitation. |
| Phase 2 | [`087b116c4d06cd08af65d775136986a413cc5eff`](https://github.com/midhunpradeep/town_model/commit/087b116c4d06cd08af65d775136986a413cc5eff) | Review commit [`fbe1063f2451270a89c9e0c110b988d6aafa4a27`](https://github.com/midhunpradeep/town_model/commit/fbe1063f2451270a89c9e0c110b988d6aafa4a27); [`reviews/phase_02_final_acceptance_review.md`](../reviews/phase_02_final_acceptance_review.md) | Synthetic neural-baseline, box and town classifiers, geometry/loss/reducer behavior, experiment persistence and Geometry Laboratory. |
| Phase 3 | [`a098a0a3c66a30200bd85906212cb07ef8bd19eb`](https://github.com/midhunpradeep/town_model/commit/a098a0a3c66a30200bd85906212cb07ef8bd19eb) | Definitive review commit [`2de9c5f29b436f727720e87132d0b4be25a40768`](https://github.com/midhunpradeep/town_model/commit/2de9c5f29b436f727720e87132d0b4be25a40768); [`reviews/phase_03_definitive_acceptance_review.md`](../reviews/phase_03_definitive_acceptance_review.md) | **Pre-detector, GT-ROI image-to-ontology implementation milestone.** It is not detector-pretrained stage 2, end-to-end detection, or evidence for final scientific claims. The definitive review records 292 passing tests. |

The current cleanup branch was created directly from the definitive Phase 3 review commit `2de9c5f29b436f727720e87132d0b4be25a40768`. In ancestry order, the direct Phase 3 sequence is `a098a0a` (accepted implementation) → `87a5215` (separate remediation report) → `2de9c5f` (definitive review record). Do not use `main` or the older `implementation/phase-3-image-embeddings` tip as the accepted Phase 3 baseline.

## Historical implementation and remediation records

### Phase 1

- Implementation/handoff: [`docs/PHASE_1_REPORT.md`](PHASE_1_REPORT.md)
- Remediation records: [`docs/PHASE_1_REMEDIATION_REPORT.md`](PHASE_1_REMEDIATION_REPORT.md), [`docs/PHASE_1_REMEDIATION_2_REPORT.md`](PHASE_1_REMEDIATION_2_REPORT.md)
- Independent reviews, in order: [`reviews/phase_01_independent_review.md`](../reviews/phase_01_independent_review.md), [`reviews/phase_01_remediation_review.md`](../reviews/phase_01_remediation_review.md), [`reviews/phase_01_remediation_2_review.md`](../reviews/phase_01_remediation_2_review.md), [`reviews/phase_01_final_closure_review.md`](../reviews/phase_01_final_closure_review.md)
- The first three review records retain their historical **REQUEST CHANGES** verdicts. The final closure review accepts Phase 1 and supersedes those earlier non-acceptance verdicts after the final source-checkout and handoff corrections. It reports a minor sparse-checkout limitation, not an acceptance blocker.

### Phase 2

- Scope: [`docs/PHASE_2_SCOPE.md`](PHASE_2_SCOPE.md)
- Implementation report: [`docs/PHASE_2_IMPLEMENTATION_REPORT.md`](PHASE_2_IMPLEMENTATION_REPORT.md)
- Remediation report: [`docs/PHASE_2_REMEDIATION_REPORT.md`](PHASE_2_REMEDIATION_REPORT.md)
- Independent reviews: [`reviews/phase_02_independent_review.md`](../reviews/phase_02_independent_review.md), [`reviews/phase_02_remediation_closure_review.md`](../reviews/phase_02_remediation_closure_review.md), [`reviews/phase_02_final_acceptance_review.md`](../reviews/phase_02_final_acceptance_review.md)
- The original review and an intermediate closure review retain **REQUEST CHANGES** verdicts. The final acceptance review verifies the last preflight correction at implementation `087b116` and accepts Phase 2; that verdict supersedes the earlier non-acceptance verdicts. The implementation report's initial test/outcome numbers remain historical; current remediation and acceptance evidence are in the later records.

### Phase 3

- Scope and sequencing: [`docs/PHASE_3_SCOPE.md`](PHASE_3_SCOPE.md)
- Architecture: [`docs/PHASE_3_ARCHITECTURE.md`](PHASE_3_ARCHITECTURE.md)
- Historical implementation report: [`docs/PHASE_3_IMPLEMENTATION_REPORT.md`](PHASE_3_IMPLEMENTATION_REPORT.md)
- Remediation reports: [`docs/PHASE_3_PRE_REVIEW_REMEDIATION_REPORT.md`](PHASE_3_PRE_REVIEW_REMEDIATION_REPORT.md), [`docs/PHASE_3_ACCEPTANCE_REMEDIATION_REPORT.md`](PHASE_3_ACCEPTANCE_REMEDIATION_REPORT.md), [`docs/PHASE_3_FINAL_CLOSURE_REMEDIATION_REPORT.md`](PHASE_3_FINAL_CLOSURE_REMEDIATION_REPORT.md), [`docs/PHASE_3_LAST_TWO_FINDINGS_REMEDIATION_REPORT.md`](PHASE_3_LAST_TWO_FINDINGS_REMEDIATION_REPORT.md)
- Independent reviews, in order: [`reviews/phase_03_independent_acceptance_review.md`](../reviews/phase_03_independent_acceptance_review.md), [`reviews/phase_03_final_remediation_closure_review.md`](../reviews/phase_03_final_remediation_closure_review.md), [`reviews/phase_03_final_acceptance_closure_review.md`](../reviews/phase_03_final_acceptance_closure_review.md), [`reviews/phase_03_definitive_acceptance_review.md`](../reviews/phase_03_definitive_acceptance_review.md)
- The first three reviews retain their historical **REQUEST CHANGES** verdicts and identify then-open findings. Those verdicts are superseded by the definitive review of the final D5 corrections, which accepts Phase 3 within the stated **pre-detector GT-ROI** scope. The implementation report and earlier remediation documents still say “not accepted” because they are historical evidence; the definitive review is authoritative for the final status.

## Read this before relying on a result

- **Accepted implementation does not mean scientifically evaluated.** The definitive Phase 3 review explicitly says acceptance is not evidence that towns outperform the baseline or box model. The accepted system has no implemented or trained image detector.
- The unchanged scientific authority is [`pizzaiolo_thesis_implementation_spec_v1_1_expanded.md`](../pizzaiolo_thesis_implementation_spec_v1_1_expanded.md). Phase-specific scope documents record implementation sequencing/deferrals and do not replace that specification.
- Current status and remaining work are summarized in [`docs/CURRENT_PROJECT_STATUS.md`](CURRENT_PROJECT_STATUS.md). Factual classifier details are in [`docs/CLASSIFIER_IMPLEMENTATION_INVENTORY.md`](CLASSIFIER_IMPLEMENTATION_INVENTORY.md).
