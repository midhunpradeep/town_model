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
