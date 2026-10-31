# T2 — Disconnect classifier-dependent model execution

**Status:** submitted for independent review; not accepted.
**Recorded:** 2026-10-31
**Branch:** `reconstruction/t2-disconnect-execution`
**Starting SHA:** `cefbbc7a4334c28bec1d6071d700a26bb4977d21`
**Implementation SHA (code and tests):** `4072dde7ca73a3d4d705002067577e3e970bc4de`

The starting revision is the accepted T0/T1 review commit. T0/T1 implementation `fbed1bc008127f7024aeabd1641519e2a333a840` was independently accepted in `reviews/classifier_teardown_t0_t1_independent_review.md`. The implementation SHA above predates this report-only commit. T2 acceptance remains exclusively with an independent reviewer.

## Result and unavailable-backend contract

T2 establishes a fail-closed operational boundary. The old classifier, training, and geometry source remains for T3's removal pass, but supported execution entrypoints now stop explicitly instead of constructing or invoking it. No replacement classifier, protocol, output representation, loss, score, membership rule, or scientific interpretation was introduced.

The centralized contract is `town_model.classifier_backend.ClassifierBackendUnavailableError`. Its operation-specific message is:

> `{operation} is unavailable during classifier reconstruction. Historical classifier execution has been intentionally disconnected, and no reconstructed classifier backend is currently approved. This operation cannot proceed until the author's model reconstruction is implemented and accepted.`

The exception is raised before side effects at operational boundaries. CLI catches it and exits nonzero; direct APIs raise the typed exception. The detached worker's CLI entry reports the same message and exits 2.

## Disabled execution paths and guards

- **Synthetic:** `run_experiment`, fixed synthetic matrix execution, the low-level family runner, saved-classifier loading, evaluation, resume, snapshot generation, loss computation, and saved geometry/export.
- **Image:** execution preflight, `ImageOntologyModel` construction/inference, `build_image_modules`, training, evaluation, optimizer construction, checkpoint loading/evaluation, synchronous run/resume, matrix execution, background launch/worker execution, stop requests, and classifier-dependent snapshot creation.
- **Geometry and ontology-region interpretation:** direct historical classifier construction/factory use; box/town parameterization construction and physical-geometry export; hard membership, signed margins, exact box/town relation computation, and relation diagnostics from classifiers or stored geometry.
- **CLI:** synthetic train/evaluate/resume/export and image preflight/run/matrix/resume/evaluate/stop/relations remain visible for compatibility but fail with an explicit unavailable error. Metadata-only `image inspect` and `inspect-run` remain available.
- **Dashboard:** Geometry Laboratory displays an unavailable notice without importing its geometry page. Image Embeddings displays the unavailable notice, exposes no launch/stop/resume/geometry/relation controls, and does not reconcile worker state. It retains only raw persisted run metadata and checksum-validated snapshot-header metadata; it does not load checkpoints or recompute/display classifier predictions, scores, plots, or geometric interpretations.

Key guard locations: `models/classifiers.py`, `models/geometry.py`, `models/image_embeddings.py`, `training/image_trainer.py`, `experiments/synthetic.py`, `experiments/image_runs.py`, `experiments/image_worker.py`, `inspection/image_snapshots.py`, `inspection/relation_diagnostics.py`, `cli.py`, and the two dashboard execution pages.

## Side-effect and preflight boundary

- `run_image_experiment`, `launch_image_experiment`, and `run_image_matrix` reject before config resolution, feature-cache creation, run-directory/manifest creation, locks, worker status/config/log files, or matrix publication.
- `train_image_model`, `build_image_modules`, evaluation/checkpoint loaders, and `save_image_snapshot` reject before visual-weight construction/download, optimizer creation, checkpoint directories/files, model inference, or snapshot staging.
- Synthetic run/fixed/evaluate/load/resume/geometry APIs reject before fixture/run-directory writes, checkpoint deserialization, optimizer restoration, or geometry export.
- Stop requests reject before reading/writing lifecycle markers. Image lifecycle `inspect_run` is now genuinely read-only; the historical `refresh_interrupted` keyword is retained for call compatibility but ignored. It no longer acquires a lock, repairs a manifest, or writes worker status.
- `resolve_image_experiment` no longer constructs `ImageOntologyModel`. It retains pure config, device, prepared-dataset, deterministic-subset, canonical-class-ID, and support validation, then raises at the explicit classifier-execution boundary. It does not return a resolved object that could imply the backend is executable.
- Historical JSON configuration parsing and `ImageExperimentConfig` validation remain available. Existing family-specific validation was not replaced by weaker approximations; successful parsing/validation does not indicate model availability. Persisted identity/config formats remain readable from stored metadata.
- `build_visual_feature_extractor` is the separate vision-only construction path for explicit feature extraction/cache use. It constructs no classifier. Unsupported run APIs do not call it. `ImagePointModel` remains classifier-free.

## Preserved infrastructure

- Pizzaïolo generation, dataset validation/inspection, ontology parsing/vocabulary order, canonical IDs, multilabel targets, known/unknown masks, manifests, and provenance/integrity checks.
- RGB preprocessing, MobileNetV3-Large/FPN, ROI providers/ROIAlign, independent visual feature extraction and frozen feature-cache APIs.
- `ImagePointModel`/`ImagePointResult`, point shapes and sample/component identities, valid masks, projections, DeepSets composition, and hybrid point fusion.
- Read-only image run metadata inspection, synthetic JSON metadata inspection, checksummed stored image-snapshot reading, and generic atomic relation-result publication (the classifier-dependent relation calculation route is unavailable).
- Dataset dashboard pages and dashboard startup.

The point-producing architecture is retained as infrastructure only; it is not approved as a final scientific design.

## Verification

All results below were executed on the implementation code SHA above with the repository `.venv`.

### T2 rejection and retained-path tests

- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_classifier_backend_unavailable.py` — **24 passed, 0 failed** (2 existing Streamlit/Pandas warnings). Covers direct API and CLI unavailability, construction/factory/optimizer/backbone instrumentation, new/existing artifact non-publication, worker rejection, read-only lifecycle/snapshot readers, fresh-process imports, point-only execution, and disabled dashboard pages.
- A defined surviving-infrastructure selection — **160 passed, 0 failed** (5 existing Streamlit/Pandas warnings). It comprises the complete T2 test module; T1 import-isolation test; independent cache identity/locking tests; complete Phase 3 vision, dataset, application/CLI/dashboard-data, and config modules; dataset, generator/worker, labels, and ontology modules; dataset dashboard pages; classifier-independent supervision reducer tests; import-architecture tests; ontology entailment/vocabulary tests excluding the geometry-execution test; and generic relation-publication tests excluding the CLI geometry-execution test. No tests were skipped or blanket-marked.
- `tests/test_phase3_cache.py::test_frozen_cache_has_strict_identity_and_only_deterministic_train_validation_panel` remains active with a visual-only extractor and direct cache-identity validation. The separate historical training-trajectory test remains active and is dispositioned below.

### Required command and runtime checks

- `PYTHONPATH=src .venv/bin/python -m compileall -q src/town_model tests` — passed.
- `git diff --check` — passed.
- `.venv/bin/town --help` — exit 0; compatibility model commands remain visible and describe their unavailable status.
- `.venv/bin/town inspect examples/pizza-48 --json` — exit 0; dataset `pizza-dcc148789aecf8af86a3`, 48 samples, `current_pass`, `current_artifact_state`.
- `.venv/bin/town dashboard --headless --port 8769` — Streamlit `/_stcore/health` returned `ok` (HTTP 200); the server process group was stopped after the check.
- `tests/test_classifier_backend_unavailable.py` separately verifies that the disabled dashboard pages render without execution controls, and that a fresh-process import of CLI/dashboard/image-point/snapshot/relation/worker modules does not import concrete classifier or geometry modules.

### Full suite — intentionally transitional

- `PYTHONPATH=src .venv/bin/python -m pytest -q` — **188 passed, 130 failed, 0 skipped**, 9 existing Streamlit/Pandas `Pandas4Warning` warnings.
- The full suite is **not green**. The 130 failures map to assertions that expect the intentionally disconnected historical classifier, objective, geometry, checkpoint, training, resume, relation, worker, or dashboard execution to succeed, plus one old lifecycle-inspection assertion that expected inspection to mutate stale run state. The matching T2 tests pass and demonstrate explicit rejection/no side effects. No test was deleted or broadly skipped to conceal the transition.

## Transitional test disposition for T3/T4

The historical assertions remain in the suite. T3 must produce the formal retirement manifest and reconcile them; T4 owns the presentation replacement where noted. Per-file full-suite failure counts:

| Test file | Failures | Invalidated expectation and planned disposition |
|---|---:|---|
| `test_classifier_teardown_boundary.py` | 1 | T1 point-vs-`ImageOntologyModel` compatibility parity is obsolete because the adapter is disabled. T3 retires the adapter assertion; T2's point-only test remains active. |
| `test_dashboard_app.py` | 5 | Old Image Embeddings preflight/launch/worker/resume/relation presentation is disabled. T3 records retired execution assertions; T4 may replace them with safe metadata-only presentation tests. |
| `test_phase2_artifact_errors.py` | 5 | Historical classifier checkpoint construction/loading/evaluation is unavailable. T3 retires model-specific loader expectations; retain generic non-model artifact integrity checks where separable. |
| `test_phase2_cli.py` | 2 | Synthetic train/evaluate/export CLI success is unavailable. T3 retires these execution expectations; the new CLI rejection tests stay active. |
| `test_phase2_dashboard_geometry.py` | 8 | Saved classifier restore, membership, geometry plots and parameter-motion displays execute historical mathematics. T3 retires those assertions; T4 owns any historical metadata-only presentation. |
| `test_phase2_experiments.py` | 3 | Synthetic classifier training, evaluation, reload and geometry results are unavailable. T3 retires execution/trajectory assertions. |
| `test_phase2_geometry.py` | 14 | Exact box/town membership, margins, parameterization, initialization and feasibility are historical mathematics. T3 retires those mathematical tests; do not rewrite them to a replacement model. |
| `test_phase2_gradients.py` | 3 | Classifier/geometry gradient assertions require the disconnected algorithms. T3 retires them. |
| `test_phase2_min_width_preflight.py` | 5 | Geometry-family feasibility/constructor and training-preflight success expectations are no longer supported. T3 retires the historical geometry tests; pure config parsing remains. |
| `test_phase2_models_losses.py` | 6 | Classifier constructors, thresholds, penalty formulas, and encoder-to-classifier gradient test are historical. T3 retires those cases; common reducer and all-unknown tests remain active and passed. |
| `test_phase2_resume.py` | 6 | Historical checkpoint load, evaluation and Adam resume trajectories are unavailable. T3 retires these expectations; read-only JSON metadata remains separate. |
| `test_phase3_cache.py` | 1 | Cache-vs-training trajectory comparison requires classifier training. T3 retires this trajectory assertion; cache generation, identity, integrity and locking tests remain active. |
| `test_phase3_cli.py` | 1 | Successful image run plus inspect flow is obsolete. T3 retires the run expectation; metadata-only inspect remains available and is covered by T2. |
| `test_phase3_d5a_ontology_entailments.py` | 1 | The test combines ontology entailment with a geometric violation query, which is unavailable. T3 retires the geometry call; independent ontology entailment/vocabulary tests remain active and passed. |
| `test_phase3_d5b_relation_publication.py` | 1 | The CLI relation-computation route is unavailable. T3 retires this route assertion; generic atomic publication/durability tests remain active and passed. |
| `test_phase3_models.py` | 29 | Historical image classifier construction/output/objective/gradient/snapshot behavior is unavailable. T3 retires those assertions; point/vision/target/mask/aggregation tests that are independent remain active where selected. |
| `test_phase3_relations.py` | 5 | Exact box and sampled town membership/relation outcomes are disconnected. T3 retires historical geometry outcomes; ontology metadata parsing remains independent. |
| `test_phase3_runs.py` | 15 | Image execution preflight/run/resume/matrix/worker/stop expectations are unavailable. The stale-process inspection assertion also expected mutation; T2 intentionally makes inspection read-only. T3 records obsolete execution tests; retain generic lock/manifest tests and the new read-only inspector test. |
| `test_phase3_training.py` | 18 | Image optimizer, evaluation, checkpoint, snapshot, resume, and family/composition training trajectories are intentionally unavailable. T3 retires these historical execution assertions. |
| `test_phase3_weighting.py` | 1 | Direct classifier checkpoint compatibility loading is unavailable. T3 retires loader execution; pure config/weight identity behavior remains independently testable. |

The 20 groups account for all 130 full-suite failures. Independent reviewer should verify this mapping against the actual full-suite log and T3 retirement manifest.

## Remaining dependencies and review limitations

- T3 (#6) remains separate and blocked by T2 (#5): it removes historical implementations and reconciles the retirement manifest. T4 (#7) remains separate and blocked by T3: it completes safe historical presentation. No T3/T4/T5 work began here.
- Issue #3 remains an author decision and is not resolved or needed for this disconnect. Bootstrap issue #1 remains open `workflow:in-review`; this report does not claim its independent acceptance.
- Historical algorithms remain physically in source behind explicit guards until T3. Independent review must verify every operational path, side-effect ordering, absence of factory/classifier fallback, and the test-disposition mapping.
- Image Embeddings currently permits raw persisted metadata and checksum-validated snapshot-header inspection only. Full read-only presentation reconciliation is T4 work.
- Verification ran on the available CPU environment; no CUDA training is claimed (training is disabled by this pass).
- No datasets, manifests, ontology assets, checkpoints, cached research outputs, accepted review reports, or scientific-specification files were changed. No replacement mathematical decisions were made.
