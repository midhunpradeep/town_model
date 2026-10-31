# Current project status

**Recorded:** 2026-10-31

**Repository:** `midhunpradeep/town_model`

**Scientific authority:** [`pizzaiolo_thesis_implementation_spec_v1_1_expanded.md`](../pizzaiolo_thesis_implementation_spec_v1_1_expanded.md), unchanged.

**Accepted Phase 3 implementation:** `a098a0a3c66a30200bd85906212cb07ef8bd19eb`

**Definitive Phase 3 review/report commit:** `2de9c5f29b436f727720e87132d0b4be25a40768`

**Current cleanup branch:** `maintenance/accepted-phase3-cleanup`, created directly from the definitive review commit above.

See the [acceptance index](ACCEPTANCE_INDEX.md) for accepted implementation/report pairs and the preserved review history. The cleanup branch adds documentation only; it does not replace, amend, or merge the accepted implementation.

## Status vocabulary

- **Implemented:** present in the accepted repository code.
- **Independently accepted:** a named implementation revision has a passing final independent review for its stated scope.
- **Exploratory:** a diagnostic run or small learning demonstration; not comparison-quality evidence unless explicitly established as such.
- **Deferred / outstanding:** required or planned work not present in the accepted implementation.
- **Not yet scientifically evaluated:** no final experiment has answered the research question or supported a thesis conclusion.

Acceptance of an implementation milestone is not scientific acceptance of a hypothesis.

## Implementation milestone status

| Milestone | Implementation status | Independent acceptance | Evidence and boundary |
|---|---|---|---|
| Phase 1 — dataset preparation | Implemented | **Accepted** | Final implementation `39b2adb739475632e7183254c5c792a2569826de`; final review record `bf2e3080a07734c50b9c679aa812d1e1e933db91`. Provides the pinned Pizzaïolo data-preparation package, schema, ontology-aware targets/masks, validation, CLI and initial dashboard. The final review records 86 passing tests and one minor sparse-checkout diagnostic limitation. |
| Phase 2 — synthetic geometry/classification | Implemented | **Accepted** | Final implementation `087b116c4d06cd08af65d775136986a413cc5eff`; final review record `fbe1063f2451270a89c9e0c110b988d6aafa4a27`. Provides neural, pure-box and town point classifiers, synthetic experiments, persistence/resume and Geometry Laboratory. The final review records 143 passing tests. |
| Phase 3 — real-image-to-ontology embeddings | Implemented as a **pre-detector, GT-ROI milestone** | **Accepted within this bounded scope** | Accepted implementation `a098a0a3c66a30200bd85906212cb07ef8bd19eb`; definitive independent review `reviews/phase_03_definitive_acceptance_review.md` at report commit `2de9c5f29b436f727720e87132d0b4be25a40768`. The review records all five gates passing and 292 passed / 0 failed / 0 skipped. It does not accept detector training, predicted-ROI learning, joint training, complete RQ2 evaluation or final held-out scientific comparisons. |
| Phase 4 | Not started | Authorized, **on hold** | The definitive Phase 3 review authorizes Phase 4. Current author direction intentionally holds implementation pending the author's model-consolidation work. No detector development is part of this cleanup. |

Earlier **REQUEST CHANGES** verdicts are retained in their original review documents. The later final acceptance reviews supersede those earlier verdicts for Phases 1–3; see the [acceptance index](ACCEPTANCE_INDEX.md) for the sequence and review scope.

## Implemented subsystems in the accepted baseline

### Data and ontology preparation (Phase 1)

- Pinned-source Pizzaïolo generation, immutable dataset manifests/assets, deterministic type-stratified splits and validation.
- Ontology class inventory distinct from individuals, supported target closure, explicit known/unknown masks, component-occurrence records, support reports, CLI and dataset dashboard.
- Dataset generation is explicit; model/run/dashboard inspection does not implicitly generate datasets.

### Synthetic point classifiers (Phase 2)

- Shared flat-point classifier interface and trainable class ordering, fixed-positive `DomainThing` presentation output, and neural-baseline, box and town families.
- Exact box/town membership, coordinate-clearance margins, masked/weighted loss reduction, fixed-point and trainable synthetic point-mapping experiments.
- Checkpoint/evaluation/resume workflows and saved-geometry diagnostics/visualization.

### Pre-detector image-to-ontology path (Phase 3)

- Prepared-image loading/subsetting, validated experiment configuration, RGB letterbox preprocessing and a pinned Torchvision MobileNetV3-Large/FPN feature path.
- Ground-truth component ROIAlign, whole-image ROI features, bounded component/global projections, sum-based DeepSets composition, and global/compositional/hybrid pizza modes.
- The same baseline/box/town classifier families are applied to component and pizza points; masked image objectives, supported positive weighting, AdamW training, checkpoint/resume, frozen-feature caching, snapshots, local run lifecycle/CLI and Image Embeddings dashboard are implemented.
- Exact box and explicitly sampled town region-relation diagnostics are available. Sampled absence of a witness is reported as inconclusive, not as proof.

**No image detector has been implemented or trained in the accepted Phase 3 system.** The MobileNet/FPN path is a visual feature extractor, not a Faster R-CNN detector; the FPN initialization is not a detector-pretraining checkpoint. Phase 3 uses ground-truth ROIs and predates detector pretraining.

## Implementation phases are not scientific training stages

The expanded specification has two distinct kinds of “phase.” They must not be conflated:

| Implementation milestone | Actual meaning in this repository |
|---|---|
| Phase 1 | Dataset preparation and inspection. |
| Phase 2 | Synthetic point classifier/geometry experiments. This is **not** scientific training stage 2. |
| Phase 3 | Pre-detector image-to-ontology implementation using GT ROIs. It is **not** the detector-pretrained scientific stage 2 because no Pizzaïolo detector has been pretrained. |
| Phase 4 | Authorized but currently on hold; no implementation started on this branch. |

| Scientific training stage in the expanded specification | Required scientific operation | Current status |
|---|---|---|
| Stage 1 | Train a class-aware detector on Pizzaïolo component boxes/categories and select it by validation detection metrics. | **Not implemented / not trained.** |
| Stage 2 | Freeze a selected Stage-1 detector and train ontology embeddings on GT ROIs. | **Not scientifically performed.** The accepted Phase 3 model code has GT-ROI pathways, but lacks a detector-pretraining checkpoint and therefore is pre-detector exploratory work rather than the specified stage. |
| Stage 3 | Joint detector/ontology training on selected predicted ROIs, with deterministic matching, detached selected coordinates, detector and ontology gradient paths, and end-to-end evaluation. | **Not implemented / not trained.** |

## Research questions and current evidence status

### RQ1 — predictive quality

> Do town embeddings cause a relevant loss of predictive quality relative to pure box embeddings? The working hypothesis is that towns preserve comparable multilabel prediction performance despite greater geometric complexity.

**Status: not yet scientifically evaluated.** There are exploratory synthetic results and small pre-detector GT-ROI learning demonstrations. They are not detector-pretrained, multi-seed canonical comparisons and do not settle RQ1. Parameter counts/capacity differ; the scientific specification explicitly does not call the families capacity-matched.

### RQ2 — ontology structure

> Can the greater expressivity of regions with holes improve the representation of ontology relations, particularly overlap, exclusion and disjointness, while preserving membership and subsumption?

**Status: not yet scientifically evaluated.** The implementation provides exact box relation checks and explicitly sampled town diagnostics, with ontology metadata. It has no training-time ontology-axiom loss and no supervisor-approved final relation-level scoring protocol. Point-classification quality or a sampled diagnostic does not establish global ontology-relation preservation.

## Exploratory results — not final scientific claims

- Phase 2's fixed and mapping experiments use small synthetic fixtures and single-seed diagnostic runs. The Phase 2 implementation report records both baseline-competitive simple-fixture results and stronger town results on a nonconvex fixture. The report explicitly says those outcomes do not establish image generalization or resolve RQ1/RQ2.
- Phase 3 reports include nine family × composition real-image smoke configurations and reduced-subset learning demonstrations. The implementation/remediation reports preserve mixed outcomes, including poor box hard-membership F1 in one corrected run and selected-class failures. These are pre-detector GT-ROI engineering/exploratory evidence, not final detector-pretrained comparisons.
- The definitive Phase 3 acceptance review did not rerun a controlled learning experiment and did not inspect the sealed test partition. Its 292-test result is implementation regression evidence, not a model-performance result.

Historical results and artifacts remain unchanged; no new dataset or expensive experiment was created for this status update.

## Accepted verification evidence

The final independent acceptance reports are authoritative for their respective reviewed revisions:

- **Phase 1:** [`reviews/phase_01_final_closure_review.md`](../reviews/phase_01_final_closure_review.md) accepts implementation `39b2adb…` and reports **86 passed, 0 failed, 0 skipped**. It records the minor case where a clean sparse checkout can be detected late: generation then fails without publishing a dataset.
- **Phase 2:** [`reviews/phase_02_final_acceptance_review.md`](../reviews/phase_02_final_acceptance_review.md) accepts implementation `087b116…` and reports **143 passed, 0 failed, 0 skipped**, successful compile/dependency checks and validation of the committed sample dataset.
- **Phase 3:** [`reviews/phase_03_definitive_acceptance_review.md`](../reviews/phase_03_definitive_acceptance_review.md) accepts implementation `a098a0a…` as pre-detector GT-ROI and reports **292 passed, 0 failed, 0 skipped**, focused D5 tests, nine real-image smoke tests, compileall and `pip check`. Review runtime was Python 3.12 and CPU-only PyTorch; CUDA behavior was not claimed. The sealed test partition was not used.

The cleanup branch test suite was run in `.venv` with Python 3.12 and the installed CPU PyTorch environment:

- `.venv/bin/python -m pytest -q` — **292 passed, 0 failed, 0 skipped** in 166.92 seconds; 11 existing Streamlit/Pandas `Pandas4Warning` deprecations.
- README command checks: `town --help`, the Phase 2 fixed/mapping commands, Phase 3 image run/matrix/lifecycle commands, and the documented dataset/evaluation/dashboard commands expose the documented arguments.
- Local Markdown targets in `README.md` and the three new status/index/inventory documents resolve to existing files. `git diff --check` passes.

This is the observed test result for the cleanup checkout; it is separate from (and matches the count in) the historical definitive Phase 3 review.

## Remaining engineering requirements from the expanded specification

The following are **outstanding**, not implicitly satisfied by Phase 3 acceptance:

1. **Detector implementation and pretraining:** integrate the specified class-aware Faster R-CNN-style detector with the shared CNN/FPN path, 22 visual foreground categories, configured proposal/filtering behavior and training-split-derived anchor decisions. Train and select a detector checkpoint using validation detection metrics, including mAP/recall and relevant category/size diagnostics.
2. **Scientific stage lineage:** use a selected detector checkpoint for frozen-detector GT-ROI ontology stage 2; preserve detector state as frozen; then initialize stage 3 from its stage-2 descendant.
3. **Predicted-ROI/joint stage:** produce filtered predicted detections, perform deterministic one-to-one IoU matching, do not inject missed GT boxes, let unmatched detections enter pizza composition without fabricated component targets, detach selected box coordinates from ontology gradients, and train detector losses jointly through the shared feature computation.
4. **End-to-end component evaluation:** implement the specification's matched-only and end-to-end component metrics, including missed-GT false negatives, unmatched-prediction false positives, duplicate penalties, support/mask conventions, and the phase-specific validation monitor. Current image metrics are GT-ROI classification metrics, not end-to-end detector metrics.
5. **Canonical scientific comparisons:** execute the five-way classifier/objective core (baseline BCE; box signed BCE and hinge; town signed BCE and hinge) with the required controlled data/splits/initialization, and the paired independent-versus-contained window study. **Independent windows are not supported by current trainable `TownGeometry`; only contained windows are implemented.** Run the selected configurations with the specification's comparison-quality independent seeds, report per-seed outcomes and variation, parameter counts and relevant detection performance.
6. **Reference geometric losses:** `reference_box_bce` and `reference_town_distance` are not implemented. The expanded specification requires source-faithful adaptations and a branch-by-branch mathematical reference-port note/hand-computable cases before accepting the reference town loss; Appendix C says unresolved material choices must go to the author rather than be silently invented.
7. **Final held-out evaluation:** after configuration/model selection, run and report explicit final test evaluation under the specified leakage controls. No accepted report here contains final test-model outcomes.
8. **Full detector end-to-end CPU smoke and provenance:** the specification's required smoke includes detector forward/loss, GT and predicted ROIs, matching, all compatible classifier/loss paths, checkpoint/evaluation/snapshot export and dashboard load. Current nine-cell smokes exercise image models on real RGB data but not a detector. Comparison-quality experiment provenance is also expected to include exact source commit/dirty state and execution environment; review documents note that this has not been complete for all current runs.

Other explicitly deferred options in the expanded specification (for example cones, adaptive house/window gates, attention pooling, OWL/ALC reasoning, and automatic hyperparameter search) remain deferred and should not be treated as hidden Phase 3 deliverables.

## Pending scientific/methodology decisions

- The supervisor-approved RQ2 protocol remains pending: ontology-pair population, relation labels, aggregate metrics, handling of unknown relations, statistical comparison, and what evidence is sufficient for the thesis claim. Exact box and sampled town utilities are diagnostic foundations, not that final protocol.
- The reference town-distance adaptation is not fully mathematically settled in the expanded specification. The specification explicitly requires source tracing and author review for material ambiguity; no code or documentation here resolves it.
- The image weighting implementation uses positive-count formulas and a documented normalization convention (normalize across all positive-supported canonical in-scope classes so the most-supported class has weight 1, then cap and select columns). The remediation report calls this an implementation convention because the specification requires common normalization but does not dictate its absolute scale. It is not an experimentally established best choice.
- Provisional defaults (including weighting cap/β, initialization distributions, and selected comparison settings) are configuration choices, not scientific optima. The author-led classifier consolidation is expected to decide which model definitions and comparisons remain appropriate; this status document does not decide them.
- Baseline/box/town capacity differs, and the expanded specification names this as a confound to report rather than a solved fairness issue.

## Git branch structure and provenance

Before cleanup work, remote refs were checked with `git ls-remote`. The exact Phase 3 acceptance branch matched `2de9c5f29b436f727720e87132d0b4be25a40768`; accepted Phase 2 and Phase 1 review branches matched their records. The accepted implementation ancestry is:

```text
Phase 1 implementation 39b2adb… → Phase 1 acceptance record bf2e308…
    → Phase 2 accepted implementation 087b116… → Phase 2 acceptance record fbe1063…
        → Phase 3 remediations … → accepted implementation a098a0a…
            → separate remediation report 87a5215…
                → definitive Phase 3 acceptance record 2de9c5f…
                    → maintenance/accepted-phase3-cleanup
```

Relevant remote branch heads at cleanup start:

| Remote branch | Head at cleanup start | Interpretation |
|---|---|---|
| `main` | `e6f4050800e32b4c6d85b076b5a406a60f39398d` | Outdated relative to accepted Phase 1/2/3 work; explicitly not the cleanup base. |
| `remediation/phase-1-findings` | `bf2e3080a07734c50b9c679aa812d1e1e933db91` | Phase 1 final acceptance record. |
| `remediation/phase-2-review-findings` | `fbe1063f2451270a89c9e0c110b988d6aafa4a27` | Phase 2 final acceptance record. |
| `implementation/phase-2-geometry-classifiers` | `ac1e54b9962be38deeb9a4edfbc3264004d7826d` | Earlier Phase 2 implementation/review point; not the final Phase 2 acceptance record. |
| `implementation/phase-3-image-embeddings` | `241cac3619022d9ae7707e1bc18667c872359527` | Earlier Phase 3 implementation point; not the accepted implementation SHA. |
| `remediation/phase-3-pre-review-findings` | `5a23bb034c57719c870a18fed51349502e65f6a0` | Earlier Phase 3 remediation/review history. |
| `remediation/phase-3-acceptance-findings` | `9ed71818dbfffe70b5bcff107a29bd0866278b73` | Intermediate Phase 3 closure history. |
| `remediation/phase-3-final-closure-findings` | `43fa70fe2b2889451f81d23a1f51443cdb2889a6` | Review before final two fixes. |
| `remediation/phase-3-last-two-findings` | `2de9c5f29b436f727720e87132d0b4be25a40768` | Definitive Phase 3 acceptance branch and starting cleanup commit. |

This cleanup branch is based directly on the last row's exact SHA. The old `implementation/phase-3-image-embeddings` tip (`241cac…`) is an earlier development revision, not the accepted implementation. The maintenance branch is a separate documentation-only descendant; it is not a merge into `main` and does not move or replace any acceptance ref.

## Hold status

Phase 4 authorization is recorded in the definitive Phase 3 review. **Phase 4 remains intentionally on hold pending the author's model-consolidation work.** This handoff documents the accepted baseline and makes classifier code inspectable; it does not start reconstruction, choose model structures, resolve scientific methodology, or modify implementation.
