# Classifier teardown dependency map (T0/T1)

## Scope and authority

This map records the accepted implementation at baseline `95438f2356f8415506dba93a45cb2908ff5cf875` and the compatibility-preserving T1 extraction on `reconstruction/classifier-teardown`. The old models remain active for this pass. The map is dependency evidence, not a replacement-model specification. No classifier output meaning, geometry, parameterization, initialization, objective, score, membership rule, or scientific interpretation is approved here.

The author-led classifier map is issue [#2](https://github.com/midhunpradeep/town_model/issues/2); its membership/set-operation decision [#3](https://github.com/midhunpradeep/town_model/issues/3) remains pending and is not required for T0/T1. The infrastructure bootstrap review [#1](https://github.com/midhunpradeep/town_model/issues/1) was `workflow:in-review` when this work began; its independent acceptance is pending. This map does not represent the agent workflow as independently accepted.

The accepted historical implementation remains at [`remediation/phase-3-last-two-findings`](https://github.com/midhunpradeep/town_model/tree/remediation/phase-3-last-two-findings), commit `2de9c5f29b436f727720e87132d0b4be25a40768`. `main` and that branch are unchanged.

## Target boundary after T1

```text
Authoritative dataset (ordered class IDs, targets, known masks)
    → image decoding/preprocessing and MobileNet/FPN/ROI feature extraction
    → classifier-independent image-to-point projection/composition
    → replaceable classifier integration boundary
```

T1 introduces `models.image_points.ImagePointModel` and the `contracts.ImagePointResult` transport. `ImagePointModel` takes `VisionFeatureBatch` plus the supplied sample identities and returns points and occurrence-validity metadata only. It does not receive/construct class vocabularies, emit classifier outputs, or require logits, margins, membership, or losses. Component/global projections, DeepSets composition, and hybrid fusion retain their existing registered parameter names, seed scope, calculations, values, and shapes.

`models.image_embeddings.ImageOntologyModel` remains a compatibility adapter. It subclasses the point path and imports `make_classifier()` only when the adapter is explicitly constructed. Thus training behavior is still the old behavior; T1 has not disconnected execution or removed any classifier. Dataset class selection remains in `image_trainer` from `PreparedDataset.canonical_class_ids` and manifest names.

The previous `ClassifierOutput` dataclass now lives in `models.classifier_transport`. It is retained as a **legacy transport DTO**, not established as a future classifier port; its old optional `logits` and `signed_margin` fields are not required of future outputs. Geometry-specific diagnostic-axis metadata now lives separately in `models.geometry_diagnostics`.

## Import-time dependency map

- Before T1, importing `town_model.models` eagerly imported all concrete classifiers and image model classes. `image_embeddings` also imported concrete classes/factory at module import, including for type declarations; image configuration imported that module for validation.
- Before T1, importing `town_model.cli` eagerly imported synthetic training, image run/training, and checkpoint evaluation even for dataset-only `generate`, `validate`, and `inspect` commands.
- Before T1, importing `dashboard.app` eagerly imported Geometry Laboratory and Image Embeddings. Those transitively imported synthetic/model training, image runs, snapshots, relation queries, and concrete classifier code before a dataset page was selected.
- T1 makes `models.__init__` exports lazy; `image_points`, classifier transport, image config validation, `cli`, `dashboard.app`, and snapshot loading import without loading `models.classifiers`. Synthetic/model and dashboard pages are imported when those commands/pages are selected. Snapshot geometry extraction and classifier-to-region adaptation import concrete classes only inside the active adapter functions.
- `relation_diagnostics` retains actual old geometry evaluation when a relation query is run. Its `PointClassifier` annotation is type-check-only, concrete types are imported only by `region_from_classifier`, and the town evaluator is imported only by town membership evaluation.
- This is not a T2 disconnection: the CLI still exposes and executes old model operations when selected, and `ImageOntologyModel` still constructs the old factory. A later removal must preserve the dataset-only import boundaries and fail model operations explicitly rather than silently falling back.

## Per-module responsibilities and disposition

Each entry distinguishes historical classifier semantics from classifier-independent data, feature, transport, and storage responsibilities. “Historical-only tests” means tests that assert or execute old model behavior after T3; such assertions must be retired explicitly, not rewritten to a replacement model.

### 1. `src/town_model/models/classifiers.py`

1. **Current responsibility:** flat-point neural baseline, box and town adapters; shared point classifier base; old factory; synthetic-only point encoder; Phase 2 penalty calculation.
2. **Classifier semantics:** `NeuralBaseline`, `BoxClassifier`, `TownClassifier`, `PointClassifier` validation/fixed-positive output behavior, `make_classifier`, and `unreduced_penalties` encode old model/output/objective behavior. `SyntheticPointEncoder` is not an image feature path but is used only by the model-dependent synthetic experiment.
3. **Independent code:** none of the three implementations or their training/output assumptions. `ClassifierOutput` is now imported from the separate legacy transport module.
4. **Callers/public API:** synthetic experiment, compatibility `ImageOntologyModel` constructor, `models` lazy exports, explicit classifier-to-region snapshot adapter; tests and historical imports may use package exports.
5. **Later disposition:** execution is first disconnected in T2; old class definitions, factory, synthetic encoder/training dependencies, and formulas are removed from active source in T3. Keep no factory or substitute until authorized separately.
6. **Tests:** `test_phase2_models_losses.py` reducer tests are independently useful if isolated from output construction; classifier interface, threshold, and penalty tests are historical-only. Point infrastructure is protected in `test_phase3_models.py` and `test_phase3_vision.py`.
7. **Silent-success risk:** factory fallback or retaining the classes under a different name would keep old semantics active; an absent backend must not fabricate predictions.

### 2. `src/town_model/models/geometry.py`

1. **Current responsibility:** trainable box/town tensors, raw-to-physical transforms, hard membership, signed margins, town corner evaluator, min-width feasibility, and geometric output diagnostics.
2. **Classifier semantics:** all `BoxGeometry`/`TownGeometry` algorithms, corner transforms, `evaluate_town_corners`, `GeometryEvaluation` values, and family-specific `validate_min_width` behavior are historical mathematics.
3. **Independent code:** no image-to-point, dataset, ROI, or target/mask logic. The diagnostic-axis name-to-shape metadata was moved to `geometry_diagnostics.py`; the old import path re-exports it for compatibility.
4. **Callers/public API:** old classifier adapters; image model configuration validation; saved relation and geometry visualization code; explicit physical geometry readers; package exports and phase tests.
5. **Later disposition:** disable active execution in T2, remove old parameterization/evaluation in T3, and disable/relabel presentation and relation interpretations in T4. Stored values may be read without recomputing them.
6. **Tests:** ontology/dataset target tests and point-projection tests do not depend on this module. `test_phase2_geometry.py`, `test_phase2_gradients.py`, family feasibility tests in `test_phase2_min_width_preflight.py`, and exact/sample relation tests are historical-only.
7. **Silent-success risk:** geometry diagnostic plots or relation queries could recompute historical membership from saved corners after the implementation is removed; do not call the evaluator as a presentation fallback.

### 3. `src/town_model/models/losses.py`

1. **Current responsibility:** `SupervisionReduction`, per-instance known-mask/trainable-mask/positive-weight reduction, and signed-distance-margin validation.
2. **Classifier semantics:** `validate_signed_distance_margin` is specific to the removed historical hinge objective. It is not a generic future output contract.
3. **Independent code:** `reduce_supervision` and its target/known-mask weighting and support handling are reusable data/objective aggregation infrastructure when detached from any chosen classifier penalty. Shared targets and masks themselves live in dataset/image contracts and remain authoritative.
4. **Callers/public API:** synthetic old objective, package export, tests. Image objectives currently have their own family-dependent score-to-penalty path.
5. **Later disposition:** retain the reducer if still independently used; retire the old margin validator with the formula in T3. Do not reinterpret a new classifier output through this reducer without an approved objective.
6. **Tests:** `test_common_reducer_masks_unknowns_weights_positives_and_normalizes_instances`, `test_all_unknown_instances_are_absent_from_objective_and_have_zero_gradient`, and reducer shape/gradient tests may remain if the reducer remains. Margin and old penalty tests are historical-only.
7. **Silent-success risk:** retaining a generic-looking helper must not imply that its old normalization/objective is an approved reconstructed loss.

### 4. `src/town_model/models/image_points.py` — T1 extraction

1. **Current responsibility:** `ComponentPointProjection`, `GlobalPointProjection`, `DeepSetsComposition`, `HybridPointFusion`, and trainable `ImagePointModel`.
2. **Classifier semantics:** none. The module produces points only; no class order, classifier construction, predictions, scoring, membership, or loss.
3. **Independent code:** all registered image-to-point layers and composition; `ImagePointResult` preserves sample/occurrence IDs, valid mask, point tensors, selected composition mode, and optional vision diagnostics.
4. **Callers/public API:** temporary compatibility adapter, lazy package exports, new boundary tests; callers can import the image point path without importing concrete classifiers.
5. **Later disposition:** retain as required feature/point infrastructure and as the future classifier integration input seam. `torch.nn.Module` is necessary for trainable point projection/composition; no placeholder classifier parameters were added.
6. **Tests:** `test_classifier_teardown_boundary.py::test_classifier_free_points_match_compatibility_adapter_exactly`; existing point shape, composition, empty-ROI, gradients, and vision tests in `test_phase3_models.py`/`test_phase3_vision.py`.
7. **Silent-success risk:** do not grow this point-only result into an implicit classifier output or label order. It has no classifier predictions and must not be mistaken for model success.

### 5. `src/town_model/models/image_embeddings.py`

1. **Current responsibility:** temporary `ImageOntologyModel` compatibility adapter that applies one old point classifier to component and pizza points; component output reshaping remains here.
2. **Classifier semantics:** old `family` selection and factory construction are in the adapter constructor; `_reshape_classifier_output` preserves the old classifier output and geometry diagnostic layout.
3. **Independent code:** compatibility wiring and transport adaptation only. Point projection/composition is now inherited from `ImagePointModel`; constructor validation is in `image_model_validation.py`.
4. **Callers/public API:** image trainer and experiment run preflight; snapshot writer type/compatibility adapter; existing `models.ImageOntologyModel` export and image model tests.
5. **Later disposition:** keep temporarily in T1; T2 must stop automatic construction/execution. T3 removes this historical adapter unless an author-approved integration is later designed. The package can still expose point infrastructure independently.
6. **Tests:** T1 point parity and projection/composition tests remain after changing their setup to `ImagePointModel`; model-family integration, objective, and gradient-through-old-classifier tests become historical-only.
7. **Silent-success risk:** keeping this adapter reachable as the automatic default would violate the disconnect. Its factory import is intentionally local to construction, not a future factory design.

### 6. `src/town_model/training/image_objectives.py`

1. **Current responsibility:** converts `ImageModelResult`, image targets/masks, and classifier scores into component/pizza objective terms and effective-batch aggregates.
2. **Classifier semantics:** `_penalties` chooses BCE/logit vs old signed-margin formulas by family; `image_objective_terms` assumes old output fields; this is not a generic classifier-independent training objective.
3. **Independent code:** target/mask packing and aggregation mechanics can be useful, but their score inputs and objective meaning are currently coupled. Preserve shared target/known-mask data contracts, not implicit loss semantics.
4. **Callers/public API:** `image_trainer` train/evaluate loops; Phase 3 numerical tests.
5. **Later disposition:** disable use for model execution in T2 and remove/retire old score-dependent formulas in T3. If future aggregation is needed, design it from an approved objective rather than retaining this as an implicit port.
6. **Tests:** known-mask records and dataset target projection remain; numerical old penalties, score selection, objective gradients, and weighted loss values in `test_phase3_models.py` become historical-only. Generic aggregation implementation may be tested separately only if retained without old scoring semantics.
7. **Silent-success risk:** `logits if present else signed_margin` is an explicit hard-coded two-representation assumption; it must not be used to define future classifier outputs.

### 7. `src/town_model/training/image_trainer.py`

1. **Current responsibility:** image visual/model module construction, training/evaluation, optimization, validation metrics, old objective calls, checkpoint persistence/resume, point/geometry diagnostics, and identity hashes.
2. **Classifier semantics:** construction of `ImageOntologyModel`, classifier parameter optimizer group, family-specific metrics/objective selection, geometry checks and family identities are active old-model execution.
3. **Independent code:** image batching, vision extraction, authoritative ordered class-ID selection, dataset support/known-mask handling, generic state hashing, environment/RNG metadata, safe serialization, and run/checkpoint identity structure where independent.
4. **Callers/public API:** `image_runs`, image worker/CLI, feature cache (`_hash_state`), snapshot writer, tests.
5. **Later disposition:** T2 must fail training/evaluation/resume before new artifacts when no approved backend is available. Preserve image/point feature creation and generic storage identities. Remove old geometric execution checks in T3.
6. **Tests:** preserve dataset/vision and feature identity, checksum, safe checkpoint and generic run metadata tests where they do not train/evaluate. `test_nine_real_image_training_smokes...`, old optimizer behavior, objective gradient, family matching and old classifier resume trajectories become historical-only.
7. **Silent-success risk:** current run path creates directories/checkpoints and can produce successful metrics from old code; unavailability must be detected before `checkpoints.mkdir`/manifest publication, not after a partial run.

### 8. `src/town_model/experiments/synthetic.py`

1. **Current responsibility:** synthetic fixture/teacher generation, old model-family training, evaluation, resume, checkpoint/snapshot persistence, and saved-geometry reload.
2. **Classifier semantics:** almost all run paths execute `make_classifier`, old losses, old membership/margins, family metrics, geometry activity, and old synthetic objectives. The fixture labels are model-dependent synthetic training, not required Pizzaïolo dataset generation.
3. **Independent code:** generic JSON/checksum/atomic storage patterns may inform retained storage but this module is not required for the Pizzaïolo dataset pipeline.
4. **Callers/public API:** synthetic CLI subcommands; Geometry Laboratory; tests for Phase 2 experiment/resume/artifacts.
5. **Later disposition:** T2 disconnects model training/evaluation/resume; T3 removes model-dependent synthetic training and model construction. Preserve any useful generic serialization independently, not under a misleading training command.
6. **Tests:** dataset generator/validator tests are separate and retained. `test_phase2_experiments.py`, `test_phase2_resume.py`, synthetic artifact/model error tests, and old synthetic CLI run tests are historical-only; only independently valuable storage parsing cases may be extracted.
7. **Silent-success risk:** restoring a checkpoint and evaluating it recomputes old classifier algorithms; do not describe it as read-only inspection merely because the optimizer is not updated.

### 9. `src/town_model/experiments/image_config.py`

1. **Current responsibility:** typed dataset, vision, model-family, objective, training, execution, diagnostics, visualization, run identity and validation config.
2. **Classifier semantics:** `ImageModelConfig.family` and old geometry settings, `ObjectiveConfig` family formulas, formula-bearing `objective_identity`, and model compatibility checks serialize old classifier choices.
3. **Independent code:** generic config parsing, dataset subset IDs, vision settings, execution controls, presentation preferences, and identity serialization can remain where not requiring a classifier. T1 moves constructor-equivalent validation to `image_model_validation.py`, which does not import concrete classifier classes.
4. **Callers/public API:** CLI/dashboard, image runs, feature cache, trainer; JSON configs are persisted run metadata.
5. **Later disposition:** retain generic config/identity and historical deserialization as appropriate; T2–T4 must mark old fields as historical or reject active execution without silently resolving them to a backend. Do not alter identity hashes during T1.
6. **Tests:** generic immutable config/hash/visualization identity and sequence parsing tests remain useful. Old family/objective formula validation and resolved old objective identities become historical-only after removal, though stored values remain readable.
7. **Silent-success risk:** a default family or loss in a config constructor could silently reactivate old mathematics; missing backend must be an explicit error.

### 10. `src/town_model/experiments/image_runs.py`

1. **Current responsibility:** dataset/config resolution, model preflight, run identity, lifecycle/locking, worker launch, run/checkpoint/snapshot publication, training orchestration and matrices.
2. **Classifier semantics:** `resolve_image_experiment` constructs `ImageOntologyModel`; launch/run/matrix/resume invoke old training. Family/objective identity and old snapshot creation are tied to the current model.
3. **Independent code:** JSON manifest storage, process/lock ownership, state reconciliation, atomic publication, immutable run ID/config metadata, and validated read-only manifest inspection where safe.
4. **Callers/public API:** image CLI and dashboard page; worker subprocess; feature cache; tests.
5. **Later disposition:** T2 removes active classifier preflight/run construction and makes unsupported calls fail before creating/reserving run artifacts. Preserve storage/readers and identity representations where independent; T4 reconciles historical presentation.
6. **Tests:** lock, process liveness, manifest reconciliation and generic immutable metadata tests are storage candidates to retain. Tests that run, resume, train, evaluate, or matrix old families become historical-only.
7. **Silent-success risk:** `resolve_image_experiment` currently does constructor preflight and `run_image_experiment` publishes a draft manifest/checkpoint; backend absence must be caught before publication. Never return a completed status after skipping model execution.

### 11. `src/town_model/inspection/image_snapshots.py`

1. **Current responsibility:** creates bounded numeric snapshots from a running old image model; loads checksum-validated JSON/NPZ snapshots and discovers valid historical snapshots.
2. **Classifier semantics:** `_classifier_geometry` derives baseline/box/town math from a live model; `save_image_snapshot` runs the model and stores old membership/logit/margin/geometry and diagnostic arrays.
3. **Independent code:** checksum/schema/offset/axis validation and reading already-stored arrays/header are storage operations. T1 moves the axis schema import away from geometry algorithms and makes concrete class imports local to old extraction.
4. **Callers/public API:** image run snapshot callback; dashboard snapshot reader; CLI relation command; tests.
5. **Later disposition:** retain checksum-validated read-only loading and stored values. T2/T3 disable/remove active old-model snapshot creation; T4 ensures snapshot display does not recompute old math.
6. **Tests:** add/preserve reader checksum, schema, offset, and stored-value fidelity tests. Snapshot generation, model-output capture, and old diagnostic-axis computations become historical-only.
7. **Silent-success risk:** a saved JSON/NPZ snapshot is safe to display as historical data, but recomputing membership/margins from its corners after teardown is not a read-only load.

### 12. `src/town_model/inspection/relation_diagnostics.py`

1. **Current responsibility:** converts regions/snapshots/classifiers to box/town region records, computes exact box and sampled town relations with ontology metadata, and atomically stores diagnostics.
2. **Classifier semantics:** box interval relation logic and town sampling validated by old exact town membership are old classifier geometry; conversion from live concrete classes is an old-specific adapter.
3. **Independent code:** generic JSON no-clobber publication and ontology metadata lookup are separable. T1 moves classifier imports to active adapter functions and defers town evaluator import to membership calls. Stored result serialization itself does not require a classifier.
4. **Callers/public API:** CLI `image relations`, Image Embeddings dashboard, image snapshots; tests.
5. **Later disposition:** T2 makes active geometry operations unavailable; T3 removes old membership/relation execution; T4 preserves generic result storage and any historical raw records while clearly disabling old interpretation.
6. **Tests:** all `test_phase3_d5b_relation_publication.py` writer/durability tests protect generic publication. Ontology entailment/unknown-state tests remain. `test_phase3_relations.py` exact/sample relation tests and relation outcomes that execute old geometry become historical-only.
7. **Silent-success risk:** sampled “no witness” is explicitly inconclusive today; after teardown it must not run using a hidden old evaluator or be presented as a future model relation.

### 13. `src/town_model/dashboard/geometry.py`

1. **Current responsibility:** discovers synthetic run groups, reloads actual checkpoints/snapshots, computes old predictions, and plots classifier regions, margins/scores and parameter motion.
2. **Classifier semantics:** every actual region/score/motion plot executes old model outputs or displays old parameter semantics; it is not generic dataset visualization.
3. **Independent code:** Matplotlib infrastructure is generic, but the module’s domain functions and saved-geometry dependency are not independently useful without the old semantics.
4. **Callers/public API:** Geometry Laboratory in `dashboard.app`, imported lazily in T1; Phase 2 dashboard tests.
5. **Later disposition:** T2 disables execution; T3 removes old model-dependent reader; T4 disables or marks historical-only geometry page. Preserve dataset pages separately.
6. **Tests:** dataset dashboard tests remain. `test_phase2_dashboard_geometry.py` plotting, saved-run, and model-state tests are historical-only; generic corrupted-file error display may be retired with the page.
7. **Silent-success risk:** importing the dashboard must not import/train a model, and plotting stored old geometry must not be presented as reconstructed geometry.

### 14. `src/town_model/dashboard/image_embeddings.py`

1. **Current responsibility:** image-run controls/launch/resume and read-only saved snapshot/image/point/geometry/relation/metric presentation.
2. **Classifier semantics:** run controls execute old models; `_exact_region`, geometry plot, old score labels, and relation actions recompute old classifier behavior from stored values.
3. **Independent code:** dataset-image lookup/overlay, checksum-validated snapshot arrays, raw stored point/mask tables, generic run metadata/worker state and presentation can survive when marked historical and detached from recomputation.
4. **Callers/public API:** `dashboard.app` Image Embeddings page, now imported only when selected; image dashboard tests.
5. **Later disposition:** T2 disables launch/resume/model operations; T3 removes active old interpretation; T4 retains dataset and safe historical storage views while disabling relation/geometry recomputation.
6. **Tests:** dataset explorer/overlay tests remain. Tests that launch/train/resume or execute relation/geometry calculation become historical-only; persisted metadata reader tests can be split to assert no checkpoint load and stored-value fidelity only.
7. **Silent-success risk:** read-only snapshot inspection currently includes old geometry and can run relation queries. “Read-only” with respect to checkpoints does not mean “does not execute old model mathematics.”

### 15. `src/town_model/dashboard/app.py`

1. **Current responsibility:** dataset overview/explorer/generation/validation plus Phase 2 Geometry Laboratory and Phase 3 Image Embeddings pages.
2. **Classifier semantics:** the dataset pages use none. The two model pages route to old geometry/run behaviors.
3. **Independent code:** dataset pages, authoritative ontology target/mask display, image validation, explicit generation and dashboard navigation.
4. **Callers/public API:** `town dashboard` launches Streamlit; tests use Streamlit AppTest. T1 lazily imports model pages only when selected.
5. **Later disposition:** retain dataset pages/startup; disable or historical-label old model pages through T2–T4. No import or startup path should trigger training.
6. **Tests:** `test_application_cli_dashboard.py` dataset service/integrity cases, `test_dashboard_app.py` explorer/corrupt-data/generation cases remain. Model page launch, relation and old geometry behavior become historical-only.
7. **Silent-success risk:** eager page imports can make a dataset-only dashboard fail when classifier modules are removed; the new import boundary must be regression-tested.

### 16. `src/town_model/cli.py`

1. **Current responsibility:** dataset generate/validate/inspect; synthetic train/evaluate/resume/export; image run/matrix/preflight/evaluate/resume/relations; dashboard launcher.
2. **Classifier semantics:** synthetic model operations and image training/evaluation/geometry commands run old algorithms. Parser/help declarations alone are not execution.
3. **Independent code:** dataset command dispatch and dashboard subprocess launcher. T1 defers imports of image and synthetic services to their selected command branches.
4. **Callers/public API:** console entry point, tests, dashboard worker subprocess.
5. **Later disposition:** keep dataset command behavior. T2 makes model-dependent commands fail explicitly before artifacts; T3/T4 remove retired commands or provide explicit historical-read-only routes without model recomputation.
6. **Tests:** dataset `generate`/`validate`/`inspect` and dashboard launch tests remain; Phase 2 training/evaluate/resume/export, image run/resume/evaluate/relations become historical-only or replaced by explicit unavailable tests.
7. **Silent-success risk:** `town --help`, dataset commands, or dashboard startup must not import removed model classes; model command errors must not claim successful completion or leave run directories.

### 17. `src/town_model/contracts.py`

1. **Current responsibility:** shared data transport for image geometry/batches, ROI identity/provider contracts, feature diagnostics, vision features, frozen feature cache, and image model results.
2. **Classifier semantics:** `ImageModelResult` contains historical classifier outputs as `Any`, but it does not construct or interpret them. T1 adds classifier-free `ImagePointResult` containing identity/validity and point tensors only.
3. **Independent code:** `ImageBatch` target/known-mask records, ROI provider interfaces, `VisionFeatureBatch`, `FrozenImageFeatureCache`, `ImageGeometry`, and point transport. These are retained contracts.
4. **Callers/public API:** dataset image data, vision batching/features/ROI/preprocessing, feature cache, image point model, training, inspection and tests.
5. **Later disposition:** retain shared dataset/vision/point contracts and targets/known masks. `ImageModelResult` is compatibility-only and can be retired with the old wrapper; do not promote its old classifier output fields as a future port.
6. **Tests:** `test_phase3_vision.py`, `test_phase3_dataset.py`, `test_phase3_models.py` point transport, `test_phase3_cache.py`, and new boundary tests. Old output DTO/model result tests become historical-only.
7. **Silent-success risk:** a generic `Any` classifier payload is storage transport, not evidence that any classifier backend exists or that a prediction was computed.

### 18. `src/town_model/models/__init__.py`

1. **Current responsibility:** package-level public exports for old classifiers, geometry/loss helpers, and image model/projection classes.
2. **Classifier semantics:** eager imports used to load concrete families just by importing the package. T1 replaces this with lazy attribute exports while retaining import compatibility.
3. **Independent code:** symbol-to-module export map and lazy lookup; point and transport symbols can be imported without loading `models.classifiers`.
4. **Callers/public API:** `from town_model.models import ...` in synthetic experiment, tests, and external users.
5. **Later disposition:** keep only safe retained exports; remove old classifier names/factory in T3. Avoid a new default factory or compatibility alias that revives retired implementations.
6. **Tests:** new import isolation test and `test_phase3_architecture.py`; existing direct package imports protect temporary T1 compatibility.
7. **Silent-success risk:** an eager import or alias can keep old implementations required by dataset-only code or silently make them active again.

## New separation modules and transport

- `models/image_points.py`: classifier-independent trainable points/composition; `nn.Module` remains because existing point projections are trainable.
- `contracts.ImagePointResult`: point-only transport with no score or membership fields.
- `models/classifier_transport.py`: existing `ClassifierOutput` DTO isolated from concrete implementations and explicitly marked legacy.
- `models/geometry_diagnostics.py`: historical geometry diagnostic axes, separate from geometric algorithms and generic transport.
- `models/image_model_validation.py`: current adapter/config validation without importing concrete classifier classes.

No dummy classifier, prediction fabrication, placeholder `nn.Parameter`, or new classifier factory/port was introduced.

## Test disposition index

### Independently valuable and to preserve

- Dataset, ontology, classes/order, targets and known masks: `tests/test_phase3_dataset.py`; `tests/test_phase3_d5a_ontology_entailments.py` ontology parsing/entailment/support tests; `tests/test_phase3_dashboard_data.py`; dataset/integrity/CLI tests in `tests/test_application_cli_dashboard.py`; dataset explorer and generation tests in `tests/test_dashboard_app.py`.
- Vision, preprocessing, feature extraction and ROI contracts: all `tests/test_phase3_vision.py` tests, including letterboxing, MobileNet/FPN feature shapes, replaceable ROI providers, ROIAlign, empty ROI and frozen/trainable feature paths.
- Point projection/composition: `tests/test_classifier_teardown_boundary.py`; point-specific projection, permutation/multiplicity, empty-set, hybrid point-path and common-state assertions in `tests/test_phase3_models.py`, retargeted to the classifier-free point model when old classifier setup is retired.
- Generic feature cache identity/locking and stored feature integrity in `tests/test_phase3_cache.py`; preserve feature value checks, not the old classifier training-trajectory coupling.
- Generic atomic result publication/durability: storage-only tests in `tests/test_phase3_d5b_relation_publication.py`.
- Architecture/import boundaries: `tests/test_phase3_architecture.py` plus the T1 test that checks CLI/dashboard/config/snapshot import without concrete classifiers.
- Generic run manifest/lock/process reconciliation and safe JSON/NPZ checksum/offset readers where separable from model execution.

### Historical-only after old algorithms are removed

- Classifier contracts, thresholds, margins, initialization and old mathematical assertions: `tests/test_phase2_models_losses.py` classifier cases; all old geometry/membership/gradient/parameterization tests in `tests/test_phase2_geometry.py` and `tests/test_phase2_gradients.py`; old-family min-width preflight in `tests/test_phase2_min_width_preflight.py`.
- Model-dependent synthetic fixtures/training/resume/checkpoint reload and CLI execution: `tests/test_phase2_experiments.py`, `tests/test_phase2_resume.py`, model checkpoint cases in `tests/test_phase2_artifact_errors.py`, and synthetic execution cases in `tests/test_phase2_cli.py`.
- Active synthetic Geometry Laboratory plots/readers that recompute model behavior: `tests/test_phase2_dashboard_geometry.py` and model-dependent portions of `tests/test_phase2_artifact_errors.py`.
- Old image classifier outputs, diagnostic geometry, family/objective formulas, and classifier gradients in `tests/test_phase3_models.py`; image training/evaluation/resume/objective/checkpoint/snapshot trajectory tests in `tests/test_phase3_training.py`; old family/objective checks in `tests/test_phase3_config.py` and model-preflight/training cases in `tests/test_phase3_runs.py`.
- Exact box/town relation outcomes and the test that invokes old geometry to contradict ontology entailment: geometry-execution cases in `tests/test_phase3_relations.py` and `test_geometric_violation_does_not_change_ontology_entailment` in `tests/test_phase3_d5a_ontology_entailments.py`. Independent ontology entailment and asserted/entailed/unknown metadata tests remain.
- Tests that launch/train/evaluate old models, recompute historical region membership, or create model outputs in `tests/test_dashboard_app.py`, `tests/test_phase3_cache.py`, and `tests/test_phase3_runs.py` must be split or retired explicitly. T3 must add a retirement manifest naming each removed assertion and rationale; do not edit an old assertion to fit a replacement.

## Failure modes and safeguards for later passes

1. A module import is not harmless merely because the selected command does not train: eager package, CLI, dashboard, snapshot, and type-only imports have previously reached concrete classifier code. Keep the tested import boundary.
2. Generic serialized run identity/configuration and checksummed snapshot arrays are not model mathematics. Preserve them as data where useful; do not re-run old model code while reading them.
3. `logits` and `signed_margin` are historical alternatives, not an exhaustive future scoring protocol. Never use `logits or margins` as the new classifier boundary.
4. A model command that skips classifier execution but returns success, zero scores, fabricated predictions, or a partial checkpoint is misleading. T2 must fail before new artifacts.
5. Geometry plots/relations and saved point masks can look like generic inspection while they recompute the historical membership rule. T4 must distinguish stored values from recomputation and clearly mark or disable old geometry interpretations.
6. Dataset class IDs/names, target closure, known masks, and unknown-state handling are dataset-owned. Do not infer class vocabulary from a classifier constructor or turn unknown targets into negatives.
7. The accepted historical branch and `main` are not cleanup targets. No generated dataset/checkpoint artifacts were changed during T0/T1.

## T2–T5 issue plan

These are related infrastructure implementation issues, not children of the mathematical decision map #2. Native issue dependencies enforce sequencing; all future tickets remain `workflow:defined` and unassigned.

| Pass | Issue | Scope | Native blocker(s) | Falsifiable acceptance focus |
|---|---|---|---|---|
| T2 — Disconnect model execution | [#5](https://github.com/midhunpradeep/town_model/issues/5) | Remove automatic construction/execution; explicit actionable unavailable errors for train/evaluate/resume/geometry before artifacts | T1 issue #4 | Each unsupported operation fails before run/checkpoint/snapshot/result publication; dataset/vision/point and metadata readers still work. |
| T3 — Remove implementations | [#6](https://github.com/midhunpradeep/town_model/issues/6) | Remove active old classifiers, parameterizations, losses, membership/margins, and synthetic model training; add retirement manifest | T2 issue #5 | Source/import scans find no active old algorithms; retained infrastructure tests pass; each retired assertion is listed, not rewritten. |
| T4 — Read-only/presentation reconciliation | [#7](https://github.com/midhunpradeep/town_model/issues/7) | Preserve checksum-validated historical metadata; disable/relabel old geometry computations and plots; preserve dataset CLI/dashboard | T3 issue #6 (and transitively #5) | Readers return stored values without model/checkpoint loading or old math; dataset surfaces start and work. |
| T5 — Independent teardown acceptance | [#8](https://github.com/midhunpradeep/town_model/issues/8) | Independent acceptance of the staged teardown only | T0/T1 #4, T2 #5, T3 #6, T4 #7 | Exact SHA review demonstrates no active old algorithms, retained boundaries pass, and unavailable operations fail honestly before artifacts. |

Issue #4 (T0/T1) is the only task in progress for this assignment. T2–T5 must not begin until their native blockers close; in particular T2 must wait for independent T1 review. The teardown does not approve any reconstructed classifier definition.
