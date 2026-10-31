# Independent Review — T2 Classifier Execution Disconnection

## A. Verdict

**REQUEST CHANGES**

T2 closes the supported training, evaluation, construction, geometry, worker, CLI, and dashboard routes I traced. It does not close one directly callable image-objective route: the historical image scoring/loss formulas in `training/image_objectives.py` still execute and return loss values. This contradicts the fail-closed scoring contract. T3 must remain blocked until this T2 defect is corrected and independently reviewed.

No replacement classifier, representation, objective, or scientific decision was introduced by this review.

## B. Provenance

- Accepted T0/T1 starting SHA: `cefbbc7a4334c28bec1d6071d700a26bb4977d21`
- T2 implementation code/test SHA: `4072dde7ca73a3d4d705002067577e3e970bc4de`
- Submitted branch HEAD reviewed: `c4c6d9e70e0281ce12434f42c6a9b3ef390352cf`
- Remote `origin/reconstruction/t2-disconnect-execution` matched that HEAD at review start and again before preparing this report.
- `cefbbc7…` is an ancestor of `c4c6d9e…`; the submitted branch contains the expected three T2 commits. Working tree was clean before review. The report-only review commit is a descendant of the submitted HEAD; its exact SHA is recorded in the issue handoff and final review response.
- GitHub issue #5 was open with `workflow:in-review`; predecessor #4 was closed with `workflow:accepted`. Issue #1 remains open `workflow:in-review` and is not treated as accepted or as part of T2. Issues #6, #7, and #8 remain `workflow:defined`; native blockers are #5 for #6, #6 for #7, and #4/#5/#6/#7 for #8.

## C. Execution-boundary findings

### Classifier construction, inference, and deserialization

Direct `PointClassifier`, `NeuralBaseline`, `BoxClassifier`, `TownClassifier`, `make_classifier()`, and `ImageOntologyModel` construction/inference fail with `ClassifierBackendUnavailableError`. The inherited constructor guard runs before `nn.Module` initialization; each historical `forward` is also guarded. The lazy package exports do not provide an alternate implementation. Image checkpoint loading and synthetic saved-run loading/evaluation are guarded before deserialization or model reconstruction.

I imported nine modules in nine fresh processes, including `cli`, `dashboard.app`, `dashboard.image_embeddings`, `image_worker`, `image_runs`, image config, snapshot/relation readers, and `ImagePointModel`. Neither `town_model.models.classifiers` nor `town_model.models.geometry` was imported in those processes. A separate instrumentation check replaced the MobileNet/FPN constructor and `torch.optim.AdamW` with failure sentinels; guarded image preflight, module construction, training, evaluation, run, and worker-launch calls raised the typed unavailable error without reaching either sentinel.

Historical helper definitions and private transforms remain in source for T3, but supported model/geometry methods guard before using them. The module-private `_bounded_corners` and inverse/initialization helpers remain directly callable only by reaching into internal implementation details; I found no supported entrypoint that reaches them after construction/export/evaluation guards.

### Blocking bypass — image objective/scoring helpers

`src/town_model/training/image_objectives.py::_penalties` (line 38) and `image_objective_terms` (line 80) are not guarded. They still execute the historical baseline BCE / signed-margin BCE / hinge selection and return objective tensors. `image_trainer.train_image_model` and `evaluate_image_model` guard before their normal calls into these helpers, but direct callers can still invoke the module-level objective API. The old test `tests/test_phase3_models.py::test_default_family_objective_penalties_and_signed_distance_hinge_are_numerically_exact` still passes and directly exercises `_penalties`.

Independent reproduction: I constructed an `ImageModelResult` with a caller-supplied zero signed margin and a known positive target, then called `image_objective_terms(..., family="box", geometry_loss="signed_distance")`. It returned `pizza_per_image == [0.10000000149011612]`; `aggregate_effective_batch` returned total loss `0.10000000149011612`. No classifier/backend was constructed or approved. This is executable historical classifier-dependent loss mathematics, not merely dead source behind the new guards.

**Expected:** the direct scoring/objective operation fails with the same typed, actionable unavailable-backend error, rather than producing a historical loss value. Preserve score-agnostic aggregation only if it remains genuinely independent.

**Minimal correction:** guard the direct image scoring/objective entrypoints (including `_penalties`, so direct private-helper calls cannot bypass the boundary) and add tests for both direct calls. Do not replace or redesign the formula. T3 may later remove the historical formula and retire its old mathematical assertion.

### Geometry, scoring, and relation diagnostics

`BoxGeometry`/`TownGeometry` construction, `from_physical`, physical geometry export, membership/margin evaluation, and `evaluate_town_corners` fail closed. The separate NumPy relation paths (`_box_pair`, `_sampled_pair`, membership helpers), classifier/snapshot-to-region adapters, `diagnose_region_relation`, geometry exports, dashboard region plot functions, and snapshot generation also reject before evaluating geometry. I found no active duplicate NumPy membership path that bypasses these guards. Pure region-record constructors accept caller-provided corner data, but the membership/relation operations consuming those records are guarded; they do not infer or publish predictions by themselves.

The objective-helper bypass above is the exception to the otherwise consistent scoring boundary.

### Synthetic experiments

Synthetic run/fixed-matrix execution, private family runner, saved-classifier loading, evaluation, resume, snapshot generation, loss computation, and geometry/export reject before fixture/run publication, checkpoint deserialization, optimizer restoration, or geometry evaluation. CLI synthetic train/evaluate/resume/export routes produce nonzero errors. `inspect_saved_run` remains a JSON-only reader and does not load checkpoints or instantiate a classifier.

### Image experiments, configuration, and workers

`run_image_experiment`, resume, matrix execution, worker launch/execution, stop requests, training/evaluation/module construction/optimizer construction/checkpoint loading, and snapshot writing reject before filesystem publication or model construction. The worker module entrypoint prints the typed message and exits 2; no broad handler turns rejection into success.

Direct `resolve_image_experiment()` validates the typed config, device, prepared dataset, deterministic train/validation subsets, canonical ordered class IDs, and support requirements, then raises unavailable. It does not construct `ImageOntologyModel`, MobileNet/FPN, optimizer, cache, or a resolved executable run. CLI `image preflight` rejects immediately before parsing its config; historical config parsing remains separately available. Config identity/order validation remains intact in the independently tested config and dataset paths.

The visual-only `build_visual_feature_extractor()` is an explicit, separate vision entrypoint. It can construct the visual backbone when deliberately requested; rejected model operations do not call it or trigger pretrained-weight downloads. No CUDA behavior was tested; the environment is Python 3.12 with CPU-only PyTorch 2.5.1.

### CLI and dashboard

`town --help`, dataset inspection, and dataset validation succeed. Model-dependent CLI operations are rejected with `SystemExit(2)` and the operation-specific unavailable message. Metadata-only image-run inspection remains available.

The dashboard server started successfully: `/_stcore/health` returned `ok` and `/` returned HTTP 200; the process was stopped. Dataset overview, image explorer, and generation pages rendered. Geometry Laboratory and Image Embeddings showed the unavailable state without launch/stop/resume/geometry/relation buttons. A separate dashboard fixture with valid run metadata and a checksummed v2 snapshot rendered those persisted values with `torch.load` instrumented to fail; no checkpoint load occurred.

The local ignored `artifacts/phase3/runs` tree contains a v1 snapshot that the v2 reader rejects with an explicit schema error, so the default Image Embeddings AppTest displayed that integrity error while still rendering without an exception or controls. This is an existing incompatible historical artifact, not a T2-created file. The checksummed v2 reader and v2 dashboard path passed.

## D. Side-effect findings

For rejected API calls, I independently hashed an existing temporary image run containing a running manifest, writer-lock bytes, stop marker, and checkpoint. Image run, launch/resume, stop, synthetic resume/run calls returned the typed unavailable error; the existing tree hashes stayed unchanged and no new run roots were created. The focused T2 tests additionally instrument constructors/optimizer, check cache/run/matrix/worker/snapshot/relation output paths, and compare existing artifacts before/after rejection.

Source ordering supports those checks:

- Image run, launch, matrix, stop, training, evaluation, checkpoint loader, module builder, and snapshot writer raise at function entry, before output/cache/lock/status/checkpoint/snapshot operations.
- Synthetic run, load, evaluate, resume, snapshot, and saved geometry paths raise before fixture/run/checkpoint reads or writes.
- CLI relation/export/training routes reject before snapshot/config/checkpoint reads and before output publication.
- Worker rejection occurs before argument parsing, status-file reads/writes, process launch, or config/log creation.

### Lifecycle mutation audit

`inspect_run()` at `image_runs.py:544` now calls only `_read_manifest`; `refresh_interrupted=True` is ignored. I verified that calling it leaves both manifest and worker-status bytes unchanged.

`transition_run()` at line 434 and `_reconcile_worker_status_file()` at line 454 still write by design. In-repository `transition_run()` callers are within `run_image_experiment()`, whose unconditional unavailable guard at line 572 makes them unreachable through supported execution. `_reconcile_worker_status_file()` is reached only through explicit `reconcile_worker_status()` at line 519. The only in-repository caller of that function is the old `_render_image_run_controls()` helper; `render_image_embeddings()` now returns through its metadata-only path before that helper is called. `request_stop()` rejects before inspection or marker writes.

A direct, explicit call to `transition_run()` still changes a valid manifest state, and a direct explicit `reconcile_worker_status()` call rewrites worker-status JSON. These are retained generic lifecycle mutators, not read-only inspectors. They are not invoked by any supported classifier-disabled operation or by rejected calls. This is appropriately bounded retained infrastructure; no lifecycle rewrite is requested for T2.

## E. Preserved infrastructure

- Dataset CLI: `town inspect examples/pizza-48 --json` succeeded with dataset `pizza-dcc148789aecf8af86a3`, 48 samples, and `current_pass`; `town validate examples/pizza-48 --no-write-report --json` exited 0 without changing the report.
- Ontology, canonical class ordering/IDs, multilabel targets, known/unknown masks, dataset integrity, and dataset provenance: the dataset, ontology, labels, application/dashboard-data, and ontology-entailment selections passed.
- Vision and ROI: all selected Phase 3 preprocessing, MobileNetV3/FPN, ROI provider/ROIAlign, empty-ROI, and feature-path tests passed. Cache identity/locking tests passed using the visual-only builder.
- Point path: `ImagePointModel`/`ImagePointResult` remained usable with sample/component IDs, validity masks, and expected component/global/composed/final tensor shapes. The result contains no `contains` or `logits` fields. DeepSets point behavior was selected and passed.
- Historical metadata: `inspect_run()` and synthetic JSON inspection returned stored metadata without writes. Checksummed v2 snapshot header/array inspection passed, including dashboard rendering. No model/checkpoint was reconstructed.

## F. Test evidence

Environment: Python 3.12.0; `torch 2.5.1+cpu`; `torch.cuda.is_available()` was false.

- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_classifier_backend_unavailable.py` — **24 passed**, 2 existing Streamlit/Pandas warnings.
- The T2 report claims a surviving selection of 160 passed but does not record an exact command/node list. I ran an explicit selection matching its listed categories (complete T2 module; T1 import test; two cache identity/locking tests; complete Phase 3 vision/dataset/config, application/CLI/dashboard-data, dataset/generator/worker/labels/ontology, and architecture modules; dataset dashboard pages; reducer tests; ontology entailment tests except the geometry case; generic relation-publication tests except its classifier CLI route). Exact command:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_classifier_backend_unavailable.py \
  tests/test_classifier_teardown_boundary.py::test_dataset_and_point_imports_do_not_load_concrete_classifiers \
  tests/test_phase3_cache.py::test_frozen_cache_has_strict_identity_and_only_deterministic_train_validation_panel \
  tests/test_phase3_cache.py::test_feature_cache_lock_recovers_a_dead_local_writer_and_rejects_a_live_writer \
  tests/test_phase3_vision.py tests/test_phase3_dataset.py \
  tests/test_application_cli_dashboard.py tests/test_phase3_dashboard_data.py \
  tests/test_phase3_config.py tests/test_dataset.py tests/test_generator.py \
  tests/test_generator_worker.py tests/test_labels.py tests/test_ontology.py \
  tests/test_dashboard_app.py::test_dashboard_explorer_loads_shared_dataset_and_renders_annotations \
  tests/test_dashboard_app.py::test_dashboard_marks_corrupt_data_unverified_and_blocks_explorer \
  tests/test_dashboard_app.py::test_dashboard_generate_page_is_available_without_triggering_generation \
  tests/test_phase2_models_losses.py::test_common_reducer_masks_unknowns_weights_positives_and_normalizes_instances \
  tests/test_phase2_models_losses.py::test_all_unknown_instances_are_absent_from_objective_and_have_zero_gradient \
  tests/test_phase2_models_losses.py::test_reducer_rejects_invalid_target_and_mask_shapes \
  tests/test_phase3_architecture.py tests/test_phase3_d5a_ontology_entailments.py \
  tests/test_phase3_d5b_relation_publication.py \
  --deselect=tests/test_phase3_d5a_ontology_entailments.py::test_geometric_violation_does_not_change_ontology_entailment \
  --deselect=tests/test_phase3_d5b_relation_publication.py::test_cli_surfaces_actionable_post_publication_failure
```

Result: **163 passed, 2 deselected**, 5 existing warnings. The reported 160 cannot be reproduced exactly from the T2 report because its exact selection is not specified; the independently defined category-matching selection above is reproducible.
- `PYTHONPATH=src .venv/bin/python -m pytest -q` — **188 passed, 130 failed, 0 skipped**, 9 existing warnings, 67.98 seconds.
- `PYTHONPATH=src .venv/bin/python -m compileall -q src/town_model tests` — passed.
- `git diff --check cefbbc7a4334c28bec1d6071d700a26bb4977d21..c4c6d9e70e0281ce12434f42c6a9b3ef390352cf` — passed.
- `.venv/bin/town --help` — exit 0.
- `.venv/bin/town inspect examples/pizza-48 --json` — exit 0; 48-sample current-pass dataset.
- `.venv/bin/town validate examples/pizza-48 --no-write-report --json` — exit 0.
- Dashboard server health/root and AppTest page checks — passed as described above.
- Fresh-process import isolation — 9 processes passed.
- Independent artifact hash and no-backbone/no-optimizer instrumentation — passed for guarded entrypoints.
- `tests/test_phase3_models.py::test_default_family_objective_penalties_and_signed_distance_hinge_are_numerically_exact` — **1 passed**. This passing legacy-math test is direct evidence of the blocking objective bypass, not evidence of T2 compliance.

### Audit of the 130 full-suite failures

I parsed the actual full-suite log. Per-file counts exactly match the T2 report's 20 groups and sum to 130:

| Test file | Failed | Actual failure class and disposition |
|---|---:|---|
| `test_classifier_teardown_boundary.py` | 1 | Compatibility parity constructs disabled `ImageOntologyModel`; typed unavailable. Historical adapter assertion. |
| `test_dashboard_app.py` | 5 | Old Image Embeddings preflight/launch/worker/resume/relation controls; route/page no longer launches models. The saved-run UI test stops at its initial now-unavailable training call, so it does not itself verify metadata presentation; the independent valid-v2 dashboard fixture did. |
| `test_phase2_artifact_errors.py` | 5 | Historical classifier checkpoint/load/evaluation expectations are preempted by explicit unavailable guards. |
| `test_phase2_cli.py` | 2 | Synthetic train/evaluate/export expected success; CLI exits 2 with unavailable error. |
| `test_phase2_dashboard_geometry.py` | 8 | Old checkpoint restoration/region/motion plotting is disabled; typed unavailable. |
| `test_phase2_experiments.py` | 3 | Synthetic classifier train/reload/evaluation expectations are disabled; typed unavailable. |
| `test_phase2_geometry.py` | 14 | Historical box/town parameterization, initialization, membership, margin, and geometry behavior is guarded. |
| `test_phase2_gradients.py` | 3 | Historical classifier/geometry training and gradients are disabled. |
| `test_phase2_min_width_preflight.py` | 5 | Old geometry constructors/training preflight expectations are disabled; pure config validation tests remain passing. |
| `test_phase2_models_losses.py` | 6 | Classifier creation, classifier output/threshold, old penalty, and encoder-to-classifier expectations are disabled; the independent reducer tests pass. |
| `test_phase2_resume.py` | 6 | Saved classifier load/evaluation/resume expectations are disabled before deserialization. |
| `test_phase3_cache.py` | 1 | Cache-vs-classifier-training trajectory invokes disabled training. Cache identity/locking tests pass. |
| `test_phase3_cli.py` | 1 | Image run command exits 2 unavailable; metadata-only inspection remains available. |
| `test_phase3_d5a_ontology_entailments.py` | 1 | Combined ontology/geometric-violation test passes its ontology checks, then the explicitly disabled relation diagnosis raises. |
| `test_phase3_d5b_relation_publication.py` | 1 | CLI relation route rejects before reaching the mocked post-publication failure; generic publication tests pass. |
| `test_phase3_models.py` | 29 | Historical adapter construction/output, training/objective gradients, and snapshot/relation execution are disabled. The separate direct `_penalties` test passes, revealing the defect above. |
| `test_phase3_relations.py` | 5 | Four box/town relation computations are disabled. The fifth test's asserted/entailed/unknown ontology metadata checks pass before its later relation call fails; T3 must split/retain those independent checks. |
| `test_phase3_runs.py` | 15 | Run/train/worker/stop/matrix/preflight historical-success expectations are disabled. The lifecycle-inspection test now observes `running` instead of a written `interrupted` state because inspection is intentionally read-only. |
| `test_phase3_training.py` | 18 | Historical module construction, training, evaluation, checkpoint, snapshot and resume paths raise before side effects; pure weighting tests pass. |
| `test_phase3_weighting.py` | 1 | Direct checkpoint load is unavailable before deserialization. |

All **130 failing expectations** are attributable to deliberately disconnected historical operations or the intentional read-only lifecycle change; I found no unrelated functional regression among those failures. The mixed-purpose ontology metadata test must not be retired wholesale: T3 should preserve its metadata assertions as an independent test. Separately, the suite's passing `_penalties` assertion is a material bypass not represented by the failure counts.

## G. Defects and limitations

### Blocking defect — historical image loss remains callable

- **Severity:** High / blocking.
- **Location:** `src/town_model/training/image_objectives.py::_penalties` and `image_objective_terms`.
- **Reproduction:** Construct an `ImageModelResult` containing a zero `signed_margin` and a known positive target; call `image_objective_terms(result, batch, family="box", geometry_loss="signed_distance")`, then `aggregate_effective_batch([terms])`.
- **Actual:** Returns a historical hinge loss of approximately `0.1` without an approved classifier backend; the existing numerical `_penalties` test passes.
- **Expected:** Typed `ClassifierBackendUnavailableError` before any old classifier-dependent score/loss formula executes.
- **Minimal correction:** Guard the direct objective/scoring entrypoints, including the private penalty helper; test direct callers. Keep only score-agnostic aggregation if appropriate. No formula redesign is authorized.

### Nonblocking review limitations / retirement notes

- The implementer report states 160 surviving tests but provides no exact node selection. The exact category-matching selection run here produced 163 passed and 2 deselected; the 160 selection is therefore not independently reproducible from the report alone.
- The 20 per-file failure counts are accurate. `test_phase3_relations.py::test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations` combines retained ontology metadata checks with obsolete relation execution; the metadata portion passed before failure and should be split/preserved in T3.
- The old dashboard saved-run test fails before reaching inspection because it first invokes now-disabled training. A separate valid historical metadata/v2-snapshot fixture confirmed the metadata-only dashboard route works.
- The local default dashboard run tree includes a v1 snapshot rejected by the existing v2 checksum/schema reader; no file was modified. CUDA was unavailable and untested.

No other material bypass or independent-infrastructure regression was found.

## H. Issue #5 acceptance-criteria mapping

1. **No active training, evaluation, resume, preflight, or geometry path automatically constructs a historical classifier:** **Partial / blocked.** Supported model paths guard correctly, but the direct image scoring/objective path still executes legacy loss mathematics.
2. **Unsupported operation fails with specific actionable unavailability; no dummy output/fallback:** **Partial / blocked.** CLI/API/worker paths do; direct `image_objective_terms` and `_penalties` return a historical loss instead.
3. **Failure before new artifacts and no mutation of existing artifacts:** **Pass for guarded unavailable operations.** Source ordering, focused tests, and independent filesystem hashes confirmed no publication/mutation. Explicit generic lifecycle mutators remain separate and are not invoked by rejected operations.
4. **Dataset CLI/dashboard, point/vision, and validated historical metadata remain usable:** **Pass.** Dataset CLI, dashboard pages, vision/ROI/cache, point path, JSON metadata and checksummed v2 snapshot checks passed.
5. **Tests cover rejection/nonpublication and retained infrastructure:** **Partial.** The focused tests pass, but omit direct `image_objectives` entrypoints; the existing direct historical penalty test still passes. The test selection behind the report's 160 count is also unspecified.
6. **No new scientific interpretation/classifier math; independent review before T3:** **Pass.** No replacement math was introduced. T3 must remain blocked until the objective bypass is corrected and reviewed.

## I. Next-phase authorization

**REQUEST CHANGES — T2 is not accepted.** The blocking correction is to make direct image objective/scoring calls fail closed before they can compute historical loss values, with focused direct-call tests. The implementation report should also identify the exact surviving-test node selection or correct its reported selection evidence. Preserve independent ontology metadata assertions when T3 creates its retirement manifest.

Issue #5 should return to `workflow:in-progress` for correction. Issue #6 remains natively blocked by #5. **T3 is not eligible for activation.** No replacement classifier design has been approved.
