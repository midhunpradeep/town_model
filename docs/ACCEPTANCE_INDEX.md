# Acceptance index

This index identifies the accepted implementation revisions and the independent review records that establish their status. Implementation SHAs and later report-only review commits are intentionally listed separately. Links to historical documents preserve their original wording and verdicts; those files have not been rewritten to make them look current.

## Accepted historical Phase 1–3 implementation revisions

| Milestone | Accepted implementation SHA | Independent acceptance record | Scope of acceptance |
|---|---|---|---|
| Phase 1 | [`39b2adb739475632e7183254c5c792a2569826de`](https://github.com/midhunpradeep/town_model/commit/39b2adb739475632e7183254c5c792a2569826de) | Review commit [`bf2e3080a07734c50b9c679aa812d1e1e933db91`](https://github.com/midhunpradeep/town_model/commit/bf2e3080a07734c50b9c679aa812d1e1e933db91); [`reviews/phase_01_final_closure_review.md`](../reviews/phase_01_final_closure_review.md) | Pizzaïolo dataset preparation, semantic labels/masks, validation, CLI and initial dashboard. The review records a minor nonblocking sparse-checkout diagnostic limitation. |
| Phase 2 | [`087b116c4d06cd08af65d775136986a413cc5eff`](https://github.com/midhunpradeep/town_model/commit/087b116c4d06cd08af65d775136986a413cc5eff) | Review commit [`fbe1063f2451270a89c9e0c110b988d6aafa4a27`](https://github.com/midhunpradeep/town_model/commit/fbe1063f2451270a89c9e0c110b988d6aafa4a27); [`reviews/phase_02_final_acceptance_review.md`](../reviews/phase_02_final_acceptance_review.md) | Synthetic neural-baseline, box and town classifiers, geometry/loss/reducer behavior, experiment persistence and Geometry Laboratory. |
| Phase 3 | [`a098a0a3c66a30200bd85906212cb07ef8bd19eb`](https://github.com/midhunpradeep/town_model/commit/a098a0a3c66a30200bd85906212cb07ef8bd19eb) | Definitive review commit [`2de9c5f29b436f727720e87132d0b4be25a40768`](https://github.com/midhunpradeep/town_model/commit/2de9c5f29b436f727720e87132d0b4be25a40768); [`reviews/phase_03_definitive_acceptance_review.md`](../reviews/phase_03_definitive_acceptance_review.md) | **Pre-detector, GT-ROI image-to-ontology implementation milestone.** It is not detector-pretrained stage 2, end-to-end detection, or evidence for final scientific claims. The definitive review records 292 passing tests. |

The Phase 3 implementation/review pair above remains accepted for its historical pre-detector GT-ROI scope. It is not the current classifier-execution baseline. At R0 start, fetched `main` was `3bd7252f2232d86e190f71c4e4a29de40dedb396`; it contains both accepted teardown implementation `19222424f5ad8347deddc6dfb2d7187bd85609c0` and T5 final review/report commit `2b9df576aac2fd95f093bbb3b046813620a822a5` in its ancestry. The nine commits after the T5 report preserve earlier teardown review records; R0 found no intervening classifier implementation changes. See the [R0 report](reconstruction/R0_RECONCILIATION_REPORT.md) for exact ancestry and fresh checks.

## Accepted T0–T5 classifier teardown — current execution boundary

The staged teardown (issues #4–#8) is independently accepted. The definitive [T5 review](../reviews/classifier_teardown_t5_final_acceptance.md), report commit [`2b9df576aac2fd95f093bbb3b046813620a822a5`](https://github.com/midhunpradeep/town_model/commit/2b9df576aac2fd95f093bbb3b046813620a822a5), accepts the exact teardown implementation [`19222424f5ad8347deddc6dfb2d7187bd85609c0`](https://github.com/midhunpradeep/town_model/commit/19222424f5ad8347deddc6dfb2d7187bd85609c0). This acceptance supersedes classifier execution from Phases 2–3 without rewriting those historical records.

The current foundation retains dataset/ontology/supervision contracts, canonical vision/FPN/ROI features, `VisionFeatureBatch`, point-only image projections/composition, generic known-mask reducers, feature caching and run/storage identity, validated read-only historical metadata/snapshot inspection, and dataset CLI/dashboard surfaces. Component visual features remain `[B,K,260]`; global spatial features `[B,256,7,7]`; component points `[B,K,d]`; final pizza points `[B,d]`.

Historical neural/affine, box, and town classifiers; their membership/geometric algorithms and classifier-specific objectives; and model-dependent train/evaluate/resume/interpretation operations are removed or explicitly unavailable through `ClassifierBackendUnavailableError`. Rejected operations must not create or mutate model artifacts. The exact unavailable route matrix and retained-boundary evidence are in the T5 review and fresh R0 report.

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
- The expanded v1.1 document is a historical review draft, not current scientific authority or implementation authorization. The author-approved mathematical handoff supplied to R0 reports 50 active A1 decisions; durable ledger reconciliation is pending R1. R0 does not reconstruct that ledger, and unresolved scientific questions remain author-controlled. See [current project status](CURRENT_PROJECT_STATUS.md), the R0 report, and live planning issues [#2](https://github.com/midhunpradeep/town_model/issues/2)/[#3](https://github.com/midhunpradeep/town_model/issues/3).
- [`docs/CURRENT_PROJECT_STATUS.md`](CURRENT_PROJECT_STATUS.md) states the current operational boundary and separately labels its 2026-10-31 Phase 1–3 text as a historical snapshot. [`docs/CLASSIFIER_IMPLEMENTATION_INVENTORY.md`](CLASSIFIER_IMPLEMENTATION_INVENTORY.md) remains descriptive evidence only.
