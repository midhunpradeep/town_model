# T3 — Historical Classifier Removal Report

**Status:** submitted for independent review
**Repository:** `midhunpradeep/town_model`
**Starting accepted T2 SHA:** `c6aad87a45ac75835f02670f32fdeca5167f18f7`
**Implementation branch:** `reconstruction/t3-remove-historical-classifiers`
**Implementation/source-and-test SHA:** `0d95073ec5245acb59ce803f3bbbd466366f74c2`
**Historical implementation retained in Git:** `2de9c5f29b436f727720e87132d0b4be25a40768`

The implementation was made in a clean worktree created directly from the exact accepted T2 SHA. The original review worktree was left untouched. The implementation commit changes active source/tests and the retirement manifest only; no accepted-history ref or canonical research artifact was modified.

## 1. Result and authority boundary

T3 removes the historical classifier implementations and executable mathematics from active reconstruction source. It does not reconstruct, rename, relocate, or comment out the old algorithms. No replacement classifier, output contract, membership rule, geometry, loss, initialization, or scientific interpretation was introduced. Historical behavior remains available through the accepted Git history only.

`ClassifierBackendUnavailableError` remains the single explicit error for supported model-dependent operations. Dataset, ontology, visual-feature, point-projection, generic reduction, identity/cache, JSON metadata, and checksum-validated snapshot boundaries remain available independently.

## 2. Modules deleted

The following historical implementation modules were deleted:

- `src/town_model/models/classifiers.py`
- `src/town_model/models/geometry.py`
- `src/town_model/models/image_embeddings.py`
- `src/town_model/models/classifier_transport.py`
- `src/town_model/models/image_model_validation.py`
- `src/town_model/training/image_objectives.py`
- `src/town_model/experiments/synthetic.py`
- `src/town_model/dashboard/geometry.py`

`models/__init__.py` now exports only classifier-independent point-projection/composition components and the generic supervision reducer. `contracts.ImageModelResult` and the legacy `ClassifierOutput` DTO were removed; `ImagePointResult` remains the point-only transport.

## 3. Historical algorithms removed

Active source no longer defines or executes:

- `NeuralBaseline`, `BoxClassifier`, `TownClassifier`, the old `PointClassifier`, `make_classifier()`, or `SyntheticPointEncoder`.
- Box/town trainable geometry, raw-to-physical transforms, inverse/initialization helpers, feasibility logic, hard membership, signed-margin algorithms, and old geometry-specific diagnostics.
- Historical classifier penalty formulas, baseline BCE-with-logits selection, signed-margin BCE, signed-distance hinge, score conversion, or classifier-specific objective selection.
- Image classifier adapters, reshaping of classifier outputs, model-dependent image training/evaluation/gradient/geometry-validity logic, and checkpoint reconstruction/inference.
- Synthetic classifier fixtures/encoders, training, evaluation, checkpoint restore/resume, snapshots, geometry extraction, and model-dependent CLI/dashboard execution.
- Snapshot-derived region reconstruction, exact/sample box/town relation calculations, and the duplicate NumPy membership implementations.

The only remaining historical loss formula text is serialized by `experiments/image_config.py` for archived configuration/run identity compatibility. It is explicitly documented as metadata; there are no executable loss bodies or model constructors using it. `models/geometry_diagnostics.py` remains a metadata-only name/axis vocabulary for interpreting stored arrays; it performs no geometry computation.

## 4. Refactored mixed modules

- **`training/image_trainer.py`:** reduced to explicit typed rejection entrypoints for historical model construction, training, evaluation, and checkpoint evaluation. It no longer contains training loops, metrics, scoring, optimizer groups, checkpoint reconstruction, or geometry checks.
- **Visual feature construction:** moved to `experiments/visual_features.py`; it constructs only the retained MobileNetV3/FPN visual feature pathway. Generic tensor-state hashing moved to `town_model/identity.py` and preserves the previous state-hash byte contract used by the frozen feature cache.
- **Dataset/subset validation:** moved to `experiments/image_validation.py`, preserving canonical class-ID ordering, original train/validation partition checks, positive-support validation, and device checks without constructing a model.
- **`training/supervision.py`:** contains only caller-supplied value reduction and effective-batch aggregation. Its documentation states that these operations do not convert classifier scores or define an approved objective. `models.losses.reduce_supervision()` and `SupervisionReduction` remain score-independent; `validate_signed_distance_margin()` was removed.
- **`experiments/image_runs.py`:** retains generic JSON run metadata, read-only manifest inspection, locking/process identity, and lifecycle metadata utilities. Preflight validates config/data then rejects; run, matrix, launch, resume, and stop operations contain only small centralized rejection entrypoints.
- **`experiments/image_worker.py`:** reduced to a fail-closed entrypoint that rejects before parsing arguments or touching status files.
- **`inspection/image_snapshots.py`:** retains checksummed JSON/NPZ schema, offset, axis, and stored-metadata integrity reading. Model-dependent snapshot generation is a small rejection entrypoint. Loading returns stored values only and never recomputes predictions, margins, or geometry.
- **`inspection/relation_diagnostics.py`:** geometry/region/membership/relation mathematics and adapters were removed. The independent ontology-to-snapshot class-ID metadata check and generic immutable JSON publication/durability helper remain; relation diagnosis explicitly rejects.
- **`experiments/synthetic.py`:** deleted. The independent JSON-only `inspect-run` reader is now `inspection/synthetic_metadata.py`; it never reads a checkpoint.
- **Dashboard and CLI:** `Geometry Laboratory` shows the explicit unavailable state without importing geometry code. `Image Embeddings` displays only persisted run JSON and checksum-validated snapshot metadata. Dataset dashboard pages remain active. Model-dependent CLI/worker commands fail with the centralized actionable error before file reads or side effects; dataset generation, validation, and inspection remain functional.

## 5. Preserved infrastructure

The following remain available and are covered by the final suite:

- `ImagePointModel`, `ImagePointResult`, `ComponentPointProjection`, `GlobalPointProjection`, `DeepSetsComposition`, and `HybridPointFusion`.
- Component and pizza point values, sample/component IDs, validity masks, and composition mode; no ontology predictions or classifier outputs are produced.
- MobileNetV3/FPN visual feature extraction, preprocessing, batching, ROI providers/ROIAlign, and frozen visual feature-cache creation/identity/integrity.
- Dataset generation/validation/inspection; ontology vocabulary and entailment metadata; canonical ordered class IDs, targets, known/unknown masks, provenance, and image validation.
- Generic `reduce_supervision`, caller-supplied weighted-instance reduction, effective-batch aggregation, and undefined-support handling. Their retention is not approval of a final training objective.
- Historical JSON configuration deserialization, stored run manifests/IDs/identity hashes, generic JSON inspection, atomic JSON publication, and checksum-validated historical snapshot headers/arrays.

## 6. Historical configuration and identity compatibility

Historical family/objective/geometry fields and formula strings remain only where required to deserialize and identify stored runs. The geometry-dependent constructor validation module was removed; `ImageModelConfig` now performs pure archived-metadata validation without instantiating or validating executable geometry. Dataset, vision, subset, target/mask, and provenance validation remain intact.

Identity compatibility was checked against accepted T2 before implementation. `configs/phase3_image_smoke.json` retains identity hash `394452225dc600041b25818f9edc3fcb0cccb7b8d79771b898ce88a4191109ff`; the default config retains `62d7998c19e006fc92d59ed08d8fcb77c56c202162fc83ebdec6b8b764cf5ad9`.

## 7. Public unavailable behavior and stored data

Supported image/synthetic train, evaluate, resume, matrix, preflight, checkpoint inference, snapshot generation, relation diagnosis, geometry export, stop, and background-worker operations raise `ClassifierBackendUnavailableError`. The CLI maps these to nonzero actionable errors. Focused tests verify no output/cache/worker/checkpoint publication and no mutation of existing run trees.

Read-only inspection remains deliberately limited:

- `town inspect-run` reads synthetic historical JSON metadata only.
- `town experiment image inspect` and the Image Embeddings page read stored run JSON.
- Snapshot readers verify checksums/schema/offsets and return stored numeric arrays/header values without model construction or recomputation.
- Historical `contains`, logits, margins, and geometry remain stored evidence only; they are not a future classifier interface.

T4 owns any final historical presentation reconciliation. No new plots or geometry demonstrations were added.

## 8. Test retirement and preservation

Formal per-node accounting is in [`T3_TEST_RETIREMENT_MANIFEST.md`](T3_TEST_RETIREMENT_MANIFEST.md).

The accepted T2 baseline was **191 passed, 132 failed, 0 skipped**. The manifest accounts for **176 original node IDs across 129 groups**: 96 deleted nodes, 77 replaced nodes, and 3 split nodes. It covers all 132 original failing node IDs plus 44 previously passing/mixed-purpose assertions whose old semantics or model dependencies were removed. Test collection changed from 323 original nodes to 180 current nodes: 147 original node IDs remain unchanged, 6 were retargeted/split under the same names, and 27 new node IDs were added.

The ontology metadata checks from `test_phase3_relations.py::test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations` remain independently executable; only its later geometry relation query/publication was retired. The independent entailment in `test_phase3_d5a_ontology_entailments.py::test_geometric_violation_does_not_change_ontology_entailment` is retained as `test_mushroom_vegetarian_ontology_entailment_is_independent_of_geometry`.

Point projection/empty-ROI tests were retargeted to `ImagePointModel`. Generic known-mask reducers, weighted aggregation, feature cache, run locks, dataset/ontology/vision, metadata readers, snapshot checksum validation, dashboard dataset pages, and atomic generic JSON publication remain tested. No historical math test was made to pass by returning zero, false, or an arbitrary exception.

## 9. Verification evidence

All commands below were run on implementation/source-and-test SHA `0d95073ec5245acb59ce803f3bbbd466366f74c2`, from the dedicated T3 worktree. The shared repository Python environment was used by absolute path because the worktree has no `.venv`.

- Full suite, `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q` — **180 passed, 0 failed, 0 skipped**, 6 existing Streamlit/Pandas `Pandas4Warning` warnings.
- Focused infrastructure selection covering dataset/ontology, vision/FPN/ROI, point/DeepSets/hybrid, target/mask contracts, generic reducers, feature cache, config/run identity, snapshots, CLI/dashboard, backend unavailability, source/import architecture, and manifest consistency — **180 passed, 0 failed, 0 skipped**, 6 existing warnings. Exact command is reproducible from the final issue journal.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests` — passed.
- `git diff --check HEAD^ HEAD` — passed.
- `.venv/bin/town --help` and `.venv/bin/town experiment image --help` — passed; help identifies model-dependent operations as unavailable.
- `.venv/bin/town inspect examples/pizza-48 --json` — passed; dataset `pizza-dcc148789aecf8af86a3`, 48 samples, `current_pass`.
- `.venv/bin/town validate examples/pizza-48 --no-write-report --json` — passed; 0 errors. Canonical `examples/pizza-48/reports/validation.json` SHA-256 was unchanged (`cd371bd190796552fdf831f084b365d48af42d363c448f9952fc5a2fafe07925`).
- Representative synthetic/image run, evaluation, and geometry-export CLI commands — exit 2 with the centralized actionable unavailable message, no `ModuleNotFoundError`/traceback, and no output paths created.
- Dashboard server — `/_stcore/health` returned `ok`; `/` returned HTTP 200. The T3 process group was stopped after the check. Dashboard AppTests for dataset pages and unavailable/model metadata pages pass in the suite.
- Fresh-process import tests cover `town_model.models`, `image_points`, `experiments.image_config`, CLI, dashboard app, and snapshot/metadata readers; removed implementation modules cannot be imported. AST/source tests find no old class/factory/geometry/objective definitions, score-conversion call bodies, or model checkpoint reconstruction calls.
- Repository diff does not touch `examples/`, `artifacts/`, canonical dataset manifests, ontology assets, checkpoints, the scientific specification, accepted review reports, `main`, or accepted-history refs.

Environment was CPU-only; no CUDA model execution is claimed. Historical model checkpoints are not reconstructed or loaded by T3.

## 10. Remaining responsibilities and independent review

- **T4 (#7):** decide final presentation of historical stored geometry/scores, keep dataset pages usable, and ensure no old interpretation is implied. T3 leaves only metadata-only inspection and unavailable geometry pages.
- **T5 (#8):** independent acceptance of the staged teardown, including source/import absence, no hidden fallback, side-effect ordering, preserved infrastructure, and manifest accuracy.
- **Author decisions #2/#3:** remain independent and unresolved; this removal makes no mathematical decision. Phase 4 remains on hold.

Independent review should inspect the exact implementation SHA and branch, source/import scans, active unavailable entrypoints, point/vision/dataset/cache/metadata boundaries, mixed-test replacements, the complete retirement manifest, and the actual 180-test runtime result. No self-acceptance is claimed.

## 11. T3 snapshot-integrity and retirement-accounting remediation (2026-11-01)

### Review finding and scope

The independent review at [`834b2f640fa970f661d0d5070f6928c8adb8695b`](https://github.com/midhunpradeep/town_model/blob/834b2f640fa970f661d0d5070f6928c8adb8695b/reviews/classifier_teardown_t3_independent_review.md) returned **REQUEST CHANGES** on submitted T3 HEAD `8386839e3c258e5f263dd7f07fa3aeaefe731f3d`. Its two blockers were (1) malformed but correctly checksummed geometry metadata escaping as raw `KeyError('box_low')` through the snapshot reader, discovery, and Image Embeddings; and (2) inaccurate/incomplete retirement-manifest replacement references and missing non-empty diagnostic-axis coverage. No classifier-removal defect was found; this remediation does not repeat or undo that teardown.

**Remediation branch:** `reconstruction/t3-snapshot-retirement-remediation` (created directly from `8386839e3c258e5f263dd7f07fa3aeaefe731f3d`). **Source/test/manifest commit:** `97bf379ff035b7bc2c7c1a4789c5474012eff9b6` (`fix: harden T3 snapshot integrity and retirement evidence`).

### Snapshot reader, discovery, and dashboard correction

`src/town_model/inspection/image_snapshots.py` now validates the JSON header container, persisted sample IDs, component-offset/count shapes, diagnostic axis declarations, and each stored `classifier_geometry_*` array's matching `geometry.<field>` before comparing values. A missing/non-mapping field, incompatible shape, malformed value, or disagreement raises a path- and field-specific `ValueError`. Optional geometry metadata remains optional when no corresponding geometry array exists. Known invalid NPZ containers are also normalized to the reader's integrity error; no catch-all exception handling was added. Discovery continues to skip only controlled `ValueError` integrity failures.

The Image Embeddings page retains its metadata-only behavior and states when no snapshots passed checksum-validated inspection, explicitly identifying invalid snapshots as unavailable. It does not add controls, construct models, or calculate geometry. Before the fix, I independently recreated the reviewer's checksummed v2 fixture and observed the raw `KeyError` from all three paths. After the fix, the direct reader reports `header.geometry.box_low`, discovery returns valid siblings while omitting corrupt snapshots (or an empty list when all fail), and AppTest renders a clear no-valid-snapshot state with no uncaught exception.

`tests/test_image_snapshot_integrity.py` adds manually authored reader-only fixtures. Positive coverage uses non-empty supported diagnostic axes (`class`, `coordinate`, `component_occurrence`, `house`), preserves array dtypes/values, sample IDs and component offsets, and checks stored geometry/header agreement. Negative coverage includes missing/malformed geometry, shape/value disagreement, malformed geometry values, wrong axis dimensions/names, missing declared arrays, checksum corruption, invalid NPZ bytes with a matching checksum, and scalar sample/offset shapes. Discovery covers all-valid, mixed valid/invalid, and all-invalid runs. The Image Embeddings AppTest uses the exact missing-`geometry.box_low` failure fixture. No model, checkpoint, membership, margin, prediction, or geometry value is constructed or recomputed.

### Retirement-manifest reconciliation

The manifest retains accepted T2 `c6aad87a45ac75835f02670f32fdeca5167f18f7`, accepted historical implementation `2de9c5f29b436f727720e87132d0b4be25a40768`, all original 132 T2 failure IDs, and the established disposition arithmetic: **176 original IDs** (96 deleted, 77 replaced, 3 split; 68 deleted groups, 58 replaced groups, 3 split groups). Every original ID was checked against actual T2 collection output: 323 nodes collected, no missing or duplicate accounted IDs. From detached worktree `/tmp/town-model-t2-manifest-audit` at the exact accepted T2 SHA, the commands were `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest --collect-only -q` (323 collected) and `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q`. The full rerun produced **191 passed, 132 failed, 0 skipped, 9 warnings**; the 132 actual failure IDs exactly equal the manifest's preserved list.

Replacement references were audited against actual current collection output, not guessed function names. Incorrect metadata-inspection and rejected-operation mappings now point to the read-only metadata, explicit unavailable/no-publication, and existing-run nonmutation tests that assert those guarantees. Point projections cite all four collected parameterized nodes (`[2]`, `[5]`, `[8]`, `[16]`). Snapshot-axis coverage points to the new hand-built reader fixtures and explicitly distinguishes stored-data assertions from retired model-generated geometry/activity/margin/relation assertions. Deleted entries carry no replacement references; all 58 replaced and 3 split groups identify current tests and describe partial preservation where applicable.

`tests/test_t3_architecture.py` now performs a bounded `pytest --collect-only -q -p no:cacheprovider tests` subprocess. This collects node IDs only and does not execute tests. It verifies every replacement resolves to an exact collected node, rejects malformed/duplicate references within an entry, requires non-empty replacement lists for replaced/split dispositions, forbids replacement claims for deleted entries, preserves baseline failure coverage and ontology split checks, and compares the manifest's current-test count to actual collection. The independent command `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest --collect-only -q` reported **196 tests collected**. The manifest records the original submitted-T3 count (**180**) separately from remediation (**196**). Final reconciliation: 147 original IDs remain unchanged, 6 accounted IDs retain their names with retargeted/split assertions, and 43 current IDs are new relative to accepted T2 (`147 + 6 + 43 = 196`). All **121 replacement references across 40 distinct node IDs** resolve; no replaced/split entry is empty and no deleted entry claims coverage.

### Verification and preserved boundaries

- Full suite: `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q` — **196 passed, 0 failed, 0 skipped, 7 existing Streamlit/Pandas warnings**.
- Focused integrity/architecture/metadata selection — **51 passed, 0 failed, 0 skipped, 7 existing warnings**:
  ```bash
  PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q \
    tests/test_image_snapshot_integrity.py tests/test_dashboard_app.py \
    tests/test_classifier_backend_unavailable.py tests/test_t3_architecture.py \
    tests/test_phase3_models.py \
    tests/test_phase3_relations.py::test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations \
    tests/test_phase3_runs.py::test_run_config_identity_and_original_subset_order_remain_metadata_only \
    tests/test_phase3_config.py::test_historical_objective_metadata_and_run_identity_compatibility_are_stable \
    tests/test_phase2_models_losses.py::test_all_unknown_instances_are_absent_from_reduction_and_have_zero_gradient \
    tests/test_phase3_architecture.py::test_image_dashboard_is_metadata_only_and_has_no_model_execution_or_geometry
  ```
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests` — passed; staged `git diff --cached --check` — passed.
- CLI commands were run with `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town`: `--help`; `inspect examples/pizza-48 --json`; `validate examples/pizza-48 --no-write-report --json`; and `experiment image inspect <temporary-run>`. Inspection returned `pizza-dcc148789aecf8af86a3`, 48 samples, `current_pass`; validation returned valid/zero errors and the canonical `examples/pizza-48/reports/validation.json` SHA-256 remained `cd371bd190796552fdf831f084b365d48af42d363c448f9952fc5a2fafe07925`. `experiment image run --config <missing-config> --output <temporary-output>` and `experiment mapping --output <temporary-output>` both exited 2 with the unavailable message and created no output paths.
- Dashboard startup command `town dashboard --headless --port 50621` returned `/_stcore/health=ok` and root HTTP 200; its process group was stopped. Dataset-page AppTests passed. A fresh process verified all eight removed implementation modules remain unimportable. Architecture/source tests still find no historical classifier definitions or checkpoint reconstruction.
- No canonical datasets, research artifacts, accepted history refs, or historical algorithms were changed or restored. The initial T3 evidence above remains historical evidence for the original submission.

### Remaining responsibilities, limitations, and handoff

Issue #7 / T4 remains **blocked, unassigned, and unstarted**. T4 still owns final historical presentation policy for stored geometry/scores and must keep dataset pages usable without implying old mathematical interpretation. T5/independent teardown acceptance and author-owned classifier decisions are also not started or implied by this remediation. Discovery intentionally omits invalid snapshots rather than listing every rejected path; the page clearly reports that no snapshots passed integrity inspection. CPU-only checks do not claim CUDA behavior. These boundaries and the exact implementation diff require independent verification; this report does not self-accept the work.
