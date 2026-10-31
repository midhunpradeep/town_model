# Independent Review — T3 Historical Classifier Removal

## A. Verdict

**REQUEST CHANGES**

The historical classifier algorithms and their executable scoring/geometry paths are removed from active source, and the reviewed point, vision, dataset, generic-reduction, identity, and ordinary historical-storage paths remain usable. The submitted revision is not acceptable yet for two material reasons:

1. A malformed but checksummed historical snapshot can escape the reader as an uncaught `KeyError` and crash the Image Embeddings dashboard.
2. The test-retirement manifest has missing and incorrect replacement references. In particular, the mixed snapshot test is marked `replaced` with no replacement tests, and its retained non-empty axis-metadata assertion is not covered by the current snapshot fixtures.

No replacement classifier mathematics or future output contract is approved by this review. No implementation correction was made.

## B. Provenance

- **Accepted T2 base:** `c6aad87a45ac75835f02670f32fdeca5167f18f7`.
- **T3 source/test/manifest commit:** `0d95073ec5245acb59ce803f3bbbd466366f74c2`; its parent is the exact accepted T2 SHA above.
- **Submitted T3 branch and HEAD reviewed:** `reconstruction/t3-remove-historical-classifiers` at `8386839e3c258e5f263dd7f07fa3aeaefe731f3d`. The remote branch matched this SHA at review start and after verification. Its parent is the source/test/manifest commit above.
- **Accepted historical implementation:** `2de9c5f29b436f727720e87132d0b4be25a40768`; it is an ancestor and its remote accepted-history ref remains unchanged.
- **Review worktree:** `/tmp/town-model-t3-independent-review`, based exactly on submitted HEAD. The report-only review commit SHA is published in the issue #6 review comment and report URL; a commit cannot include its own object ID in its contents.
- T0/T1 issue #4 is closed `workflow:accepted`; T2 issue #5 is closed `workflow:accepted`. T3 issue #6 was open `workflow:in-review` during review. T4 #7 remains open, unassigned, `workflow:defined`, and natively blocked by #6. Author decision #3 remains pending; no teardown issue approved classifier mathematics.

The exact T2-to-T3 diff contains the reported source/test changes plus the T3 report and retirement manifest. It does not change `examples/`, `artifacts/`, canonical dataset manifests, ontology assets, checkpoints, the scientific specification, earlier reports/reviews, `main`, or accepted-history refs. The source commit is a direct child of accepted T2; there are no unrelated commits between them.

## C. Algorithm-removal audit

All eight reported modules are absent at the reviewed HEAD:

- `src/town_model/models/classifiers.py`
- `src/town_model/models/geometry.py`
- `src/town_model/models/image_embeddings.py`
- `src/town_model/models/classifier_transport.py`
- `src/town_model/models/image_model_validation.py`
- `src/town_model/training/image_objectives.py`
- `src/town_model/experiments/synthetic.py`
- `src/town_model/dashboard/geometry.py`

The package exports only point-projection/composition and generic reduction symbols. `ImageModelResult` and `ClassifierOutput` are absent; `ImagePointResult` is point-only. The eight modules cannot be imported. Fresh supported imports in `tests/test_t3_architecture.py` and `tests/test_classifier_backend_unavailable.py` do not load them.

I manually searched the remaining Python source in addition to reviewing the source diff and AST/import tests. No equivalent renamed implementations were found for the old classifier classes/factory, trainable box/town geometry, corner transformations/inverses, initialization/feasibility, hard membership, signed-margin evaluation, old family-specific objective selection, score-to-penalty conversion, checkpoint reconstruction, or synthetic training/evaluation/resume. The trainer and worker modules are fail-closed entrypoints; image run and relation operations likewise reject. There is no active `torch.load`, `load_state_dict`, or equivalent model checkpoint reconstruction path.

`models/geometry_diagnostics.py` is an axis-name/shape vocabulary only. `image_config.py` retains formula and positive-weighting strings to preserve archived config identity, but those strings are not evaluated as formulas. Its `objective_identity()` builds metadata only. `image_snapshots.py` compares stored arrays to saved header values; it does not derive membership, margins, predictions, or relations. `relation_diagnostics.py` retains ontology metadata validation and generic JSON publication only. These are archival/storage operations, not hidden model execution.

## D. Preserved infrastructure audit

### Dataset, ontology, validation

The canonical dataset/ontology implementation is unchanged from accepted T2. Dataset generation, inspection, validation, class ordering, ontology entailment metadata, target/known-mask contracts, provenance, and integrity tests passed. The CLI independently reported dataset `pizza-dcc148789aecf8af86a3`, 48 samples, `current_pass`; validation reported `valid: true`, zero errors. The canonical validation report SHA-256 remained `cd371bd190796552fdf831f084b365d48af42d363c448f9952fc5a2fafe07925`.

`image_validation.py` retains canonical class ordering and support checks. Compared with the prior T2 validation routine, it retains the known-positive training-coverage and primary-validation-monitor rules and adds explicit dataset-identity and original-split checks; I found no weakened rule. The focused support-validation test and full suite pass.

### Vision, ROI and feature cache

`visual_features.py` is the prior visual-only construction path moved out of the image trainer. I independently ran it at accepted T2 and T3 with the same CPU config (`pretrained=False`, seed 317) and the same canonical image batch. Both produced the same visual state hash `2076328c4d1976fb6ee006ddeeae71718d9a6ee8186f7f083462bddd94e2e6f7`, the same combined feature hash `a0b9d757b165e0d8f3cdd7a36f0ce918c9aa4ae5e6d7294952c42bc18a5cdccb`, and identical shapes `[2,15,260]`, `[2,256,7,7]`, and `[2,15]`. The moved tensor-state hash function also produced exact parity. The full vision tests cover MobileNetV3-Large/FPN, preprocessing, ROIAlign, replaceable ROI providers, empty ROIs, frozen features, and trainable-FPN gradients.

Feature-cache identity, retrieval, integrity, and locking tests remain active. `test_phase3_cache.py` no longer claims classifier training-trajectory parity; it tests the independent cache contract.

### Image-to-point path

`ImagePointModel` remains a trainable `nn.Module` that returns component/global/composed/final point tensors, sample/component IDs, validity masks, and mode metadata only. It constructs no class vocabulary or classifier and fabricates no prediction. Retained tests cover projection shape/bounds, DeepSets permutation/multiplicity behavior, empty and padded component masks, point identity, and hybrid composition. The source was not modified from accepted T2; the full current suite passed these checks.

### Generic reductions

`reduce_supervision`, `SupervisionReduction`, `reduce_weighted_instances`, and `aggregate_effective_batch` accept caller-supplied values. Their numerical known-mask/weight behavior, eligibility, undefined-supervision handling, and gradients remain covered and passed. The reducer extraction preserves the former mask/weight arithmetic and effective-batch aggregation; no classifier-family score conversion was moved into it. Retaining these utilities does not approve a future objective.

## E. Operational safety

Representative direct APIs, the worker, CLI, and dashboard fail closed through `ClassifierBackendUnavailableError`. Unit tests exercise direct training/evaluation/module/checkpoint/snapshot/relation/run/worker operations, CLI errors, and unchanged existing-run files. The worker entrypoint reports the central error and exits nonzero. `town` model-dependent commands returned exit status 2 with the intentional-unavailability message, not an import error or generic traceback.

I independently ran:

- `town experiment fixed --output <temporary path>` — exit 2; no output path created.
- `town experiment image run --config <missing config> --output <temporary path>` — exit 2 with the unavailable message before reading the missing config; no output created.
- `town export-geometry <missing run> --output <temporary path>` — exit 2; no export created.
- Dataset-only `town --help`, `town inspect`, and `town validate` — exit 0.

The dashboard started on an unused temporary port: `/_stcore/health` returned `ok`, `/` returned HTTP 200, and the process was stopped. AppTests in the full suite cover dataset explorer/generation and metadata-only Image Embeddings. Geometry Laboratory is an explicit unavailable page. Supported imports do not load removed model modules. No rejected operation created run/checkpoint/cache/worker/export outputs in the test and CLI fixtures.

The one operational exception is the malformed-snapshot crash described in Section J. Normal valid snapshot loading is safe; malformed nested geometry metadata is not normalized into a reader error and can escape discovery into the dashboard.

## F. Historical compatibility and stored data

I independently loaded `configs/phase3_image_smoke.json` at accepted T2 and T3. Both produce the recorded identity hash:

`394452225dc600041b25818f9edc3fcb0cccb7b8d79771b898ce88a4191109ff`

Image run metadata and synthetic JSON metadata remain inspectable without checkpoint loading or writes. A valid hand-authored snapshot was read as stored data; its `pizza_contains` value `[[true]]` and `pizza_signed_margin` value `[[0.125]]` were returned unchanged. The valid reader checks the array checksum, schema, sample IDs, component offsets, declared axis dimensions, and stored-array/header geometry agreement; it does not recompute the values. The dashboard fixture and CLI metadata inspection passed.

However, a malformed but correctly checksummed snapshot with an array `classifier_geometry_box_low` and header `geometry: {}` causes `load_image_snapshot()` to raise a raw `KeyError('box_low')`. `discover_image_snapshots()` catches `ValueError` only, so the same error propagates. Selecting Image Embeddings with that persisted fixture produced an uncaught Streamlit AppTest exception at `dashboard/image_embeddings.py` while discovering snapshots. This is an invalid stored metadata record, not model reconstruction; it must be rejected safely.

## G. Retirement-manifest audit

I independently collected tests from exact accepted T2 (`c6aad87…`) and exact submitted T3 (`8386839…`): **323** baseline nodes and **180** current nodes, both unique. I also reran the exact T2 full suite: **191 passed, 132 failed, 0 skipped, 9 warnings**. Parsed failure node IDs match the manifest's 132 `baseline_failed_node_ids` exactly: no omission and no extra. The manifest has 129 groups and accounts for 176 unique original nodes: 96 deleted, 77 replaced, and 3 split. This consists of all 132 T2 failures plus 44 previously passing/mixed nodes. The count arithmetic reconciles: 147 original node IDs are unchanged and unlisted; six accounted IDs retain their names but were retargeted/split; 27 current IDs are new; `147 + 6 + 27 = 180`.

The manifest is not yet a complete/accurate replacement map:

1. `tests/test_phase3_models.py::test_snapshot_diagnostic_axes_roundtrip_preserve_class_geometry_and_instance_activity` is marked `replaced` with an empty `replacement_tests` array. The current `test_checksum_validated_snapshot_values_remain_readable_without_model_import` is meaningful basic stored-value/checksum-valid coverage, but the manifest misspells that node as `...remain_readable_without_model`. Both current snapshot fixtures use an empty `diagnostic_axes_schema`; no retained test exercises a non-empty diagnostic-axis declaration or its dimension validation. The original mixed test included an independent round-trip assertion of declared axes and stored array shape/value, in addition to historical model/geometry computations. Thus it was only partially replaced. The removed model-generated outputs should stay retired, but the reader/axis contract needs an exact replacement reference and a stored-data-only regression fixture (or a corrected disposition with the retained reader coverage explicitly documented).
2. `tests/test_phase3_models.py::test_model_projection_shapes_and_bounded_point_contract` is listed as a replacement without parameter IDs. The collected replacement nodes are `[2]`, `[8]`, `[16]`, and `[5]`; the bare function string is not a pytest node ID.
3. Other replacement mappings also need correction: `test_image_cli_metadata_inspection_remains_read_only` points to a CLI-rejection test rather than the current read-only metadata test; `test_image_execution_and_checkpoint_entrypoints_fail_before_model_construction` and `test_rejected_operations_do_not_create_runs_workers_checkpoints_or_exports` point to the unrelated point/visual-feature test; the rejected-operation entries should cite the actual current unavailable/no-side-effect tests. The architecture test currently checks counts and selected ontology entries but does not verify that replacement references exist, which is why these errors pass.

I reviewed the mixed-purpose ontology tests directly against accepted T2. The asserted/entailed/unknown metadata and snapshot class-ID checks from `test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations` remain in the current test; only the later geometry diagnosis/publication was removed. The MushroomTopping → VegetarianTopping entailment from `test_geometric_violation_does_not_change_ontology_entailment` remains as `test_mushroom_vegetarian_ontology_entailment_is_independent_of_geometry`. Generic reducer, cache identity/integrity, run-lock/identity, dataset dashboard, and generic atomic relation-publication assertions also remain. The retired synthetic fixture checks concern the removed historical synthetic teacher/experiment, not the canonical Pizzaïolo dataset. Retired weighting tests assert historical objective-specific weighting policies, not the generic caller-supplied value reduction retained here.

## H. Mixed-purpose assertion findings

- **Ontology metadata:** preserved as above; no geometry interpretation remains executable.
- **Generic losses/reductions:** retained value/mask/weight/gradient assertions are now run without score conversion. The old family-specific BCE/margin/hinge assertions were retired, not rewritten.
- **Feature cache:** cache identity/integrity and train/validation-only panel checks remain. Only classifier-trajectory comparison was retired.
- **Run lifecycle:** generic lock, process identity, and metadata identity tests remain. Rejected run/stop operations are checked for nonmutation; explicit generic lifecycle mutators are not called by those rejected paths.
- **Dashboard:** dataset overview/explorer/generation and persisted metadata-only rendering remain. Model/geometry controls and recomputation are disabled.
- **Relation publication:** atomic/no-clobber JSON storage tests remain; the removed CLI relation behavior depended on old geometry execution.

## I. Runtime test and integrity evidence

Environment: repository Python 3.12 virtual environment at `/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv`; CPU-only PyTorch.

Exact baseline and current suite commands/results:

```bash
cd /tmp/town-model-t2-objective-guard-remediation
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q
# 191 passed, 132 failed, 0 skipped, 9 warnings (63.59 s)

cd /tmp/town-model-t3-remove-historical-classifiers
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q
# 180 passed, 0 failed, 0 skipped, 6 warnings (43.40 s)
```

The accepted-T2 failure node set was parsed independently from its full-suite output and matches all 132 manifest failure IDs exactly. The current full suite had no skips.

The exact focused selection from the T3 issue handoff was:

```bash
cd /tmp/town-model-t3-remove-historical-classifiers
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q \
  tests/test_application_cli_dashboard.py tests/test_classifier_backend_unavailable.py \
  tests/test_classifier_teardown_boundary.py tests/test_config.py tests/test_dashboard_app.py \
  tests/test_dataset.py tests/test_generator.py tests/test_generator_worker.py tests/test_labels.py tests/test_ontology.py \
  tests/test_phase2_models_losses.py tests/test_phase3_architecture.py tests/test_phase3_cache.py \
  tests/test_phase3_config.py tests/test_phase3_d5a_ontology_entailments.py \
  tests/test_phase3_d5b_relation_publication.py tests/test_phase3_dashboard_data.py \
  tests/test_phase3_dataset.py tests/test_phase3_models.py tests/test_phase3_relations.py \
  tests/test_phase3_runs.py tests/test_phase3_vision.py tests/test_t3_architecture.py \
  tests/test_image_validation_infrastructure.py
# 180 passed, 0 failed, 0 skipped, 6 warnings (44.33 s)
```

This focused selection collected the entire current suite; it is not additional coverage beyond the full-suite run.

Other exact checks:

```bash
cd /tmp/town-model-t3-remove-historical-classifiers
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests
# passed
git diff --check c6aad87a45ac75835f02670f32fdeca5167f18f7..8386839e3c258e5f263dd7f07fa3aeaefe731f3d
# passed
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town --help
# exit 0
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town inspect examples/pizza-48 --json
# exit 0; pizza-dcc148789aecf8af86a3, 48 samples, current_pass
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town validate examples/pizza-48 --no-write-report --json
# exit 0; valid=true, zero errors
```

The exact rejected-operation commands and results were:

```bash
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town experiment fixed --output /tmp/t3-smoke.GbByHl/rejected-synthetic
# exit 2; unavailable message; output absent
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town experiment image run --config /tmp/t3-smoke.GbByHl/missing-config.json --output /tmp/t3-smoke.GbByHl/rejected-image
# exit 2; unavailable message precedes missing-config read; output absent
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town export-geometry /tmp/t3-smoke.GbByHl/missing-run --output /tmp/t3-smoke.GbByHl/rejected-geometry.json
# exit 2; unavailable message; output absent
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town experiment image inspect /tmp/t3-smoke.GbByHl/historical-image-run
# exit 0; stored manifest/metrics returned
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town inspect-run /tmp/t3-smoke.GbByHl/historical-synthetic-run
# exit 0; stored JSON returned
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town dashboard --headless --port 8769
# launched; /_stcore/health=ok and / returned HTTP 200; process stopped after check
```

These fixtures and tested output paths were under `/tmp/t3-smoke.GbByHl/`; rejected commands created no output. A valid checksummed snapshot returned its stored arrays unchanged. Dataset and metadata-only pages also passed AppTests in the full suite.

T2/T3 historical image-config identity parity and visual-feature state/output parity were exact, as detailed above.

## J. Defects and limitations

### 1. Blocking — corrupt historical geometry metadata crashes the metadata dashboard

- **Severity:** High; blocks safe historical-storage acceptance.
- **Location:** `src/town_model/inspection/image_snapshots.py::load_image_snapshot` (geometry-array/header comparison, around line 71), `discover_image_snapshots` (around line 86), and `src/town_model/dashboard/image_embeddings.py::render_image_embeddings` (snapshot discovery, around line 70).
- **Reproduction:** Create a valid `image-research-snapshot-v2` fixture with a correct `arrays_sha256`, `sample_ids=['s']`, valid component offsets, and an array `classifier_geometry_box_low`; set `diagnostic_axes_schema` to `{'classifier_geometry_box_low': ['class', 'coordinate']}` and `geometry` to `{}`. Call `load_image_snapshot(path)`, `discover_image_snapshots(run)`, or select that run in Image Embeddings.
- **Expected:** the reader reports a controlled invalid-snapshot error (`ValueError` or equivalent); discovery omits/marks the invalid snapshot; dashboard renders an integrity error or no-valid-snapshot state without an uncaught traceback.
- **Actual:** `load_image_snapshot` indexes `geometry['box_low']` outside its guarded parsing block and raises raw `KeyError`. Discovery catches `ValueError` only and propagates it. AppTest records an uncaught Streamlit exception.
- **Minimal correction:** validate the geometry header shape/required stored keys and normalize malformed geometry metadata to the reader's documented integrity error; ensure discovery/dashboard handle it. Add direct reader and dashboard corrupt-fixture regressions. No geometry computation is needed.

### 2. Blocking — test-retirement manifest has materially inaccurate replacement accounting

- **Severity:** Medium; blocks the required auditable retirement record and preservation criterion.
- **Location:** `docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` entries for the snapshot round-trip node and the listed T2 boundary tests; `tests/test_t3_architecture.py::test_retirement_manifest_accounts_for_each_original_node_and_mixed_assertions` does not validate replacement-node existence or coverage.
- **Reproduction:** Independently collect T3 tests and compare each `replacement_tests` value with pytest node IDs. Two references do not exist: the mistyped checksum-reader test name and the unparameterized point-projection test name. The snapshot round-trip entry is `replaced` with `replacement_tests: []`.
- **Expected:** every replacement reference resolves to an actual current node; each `replaced`/`split` entry documents meaningful coverage; the mixed snapshot reader/axis assertions are either explicitly and accurately replaced by reader-only tests or dispositioned as deleted with the retained reader scope documented separately.
- **Actual:** the current checksum fixture has a different node name and empty axis schema; no current fixture covers non-empty axis metadata. Other entries point to unrelated tests, so the manifest overstates what replacement tests prove.
- **Minimal correction:** fix all node IDs, map the affected T2 boundary tests to the current metadata/no-side-effect tests, and add the missing snapshot axis/integrity fixture (or correct the disposition and retained-coverage statement). Extend manifest consistency validation to reject nonexistent replacement IDs and inappropriate empty replacement lists for `replaced`/`split` entries.

No historical math bypass, fabricated prediction, unintended repository artifact change, or unrelated functional regression was found. CUDA behavior was not tested; no CUDA claim is made.

## K. Issue #6 acceptance-criteria mapping

| Issue #6 criterion | Finding |
|---|---|
| Search/import tests find no active old classes/factory/geometry/objective/trainer or fallback | **Pass.** Deleted modules absent; package exports and fresh-process imports verified; manual scans found no equivalent executable algorithms. |
| Obsolete classifier tests explicitly accounted for without adapting old math | **Partial / fail.** All 132 actual T2 failures and 176 total original IDs are accounted, but replacement references and the snapshot mixed-purpose disposition are materially inaccurate/incomplete. |
| Dataset/ontology/known masks, vision/ROI, points, generic reducers, identity/storage, validation, and checksummed readers remain usable | **Partial.** Normal paths and identity parity pass. Malformed geometry metadata in a checksummed historical snapshot crashes dashboard discovery; axis metadata preservation has no non-empty-axis replacement test. |
| No new predictions or scientific semantics; unsupported operations fail honestly | **Pass** for supported direct/API/CLI/worker/dashboard paths. No classifier or new scoring semantics was introduced. |
| No generated datasets/checkpoints or accepted history refs modified; independent review before T4 | **Pass.** Diff/ref audit is clean; #7 remains unassigned and blocked by #6. |

## L. Next-phase authorization

**T3 is not accepted.** Correct the snapshot reader's malformed-metadata handling and reconcile the retirement manifest/replacement coverage, then resubmit an exact fixed SHA for independent review. Issue #6 must remain open; issue #7 stays unclaimed, unstarted, and blocked. No T4 work is authorized, and no classifier-reconstruction mathematics has been approved.

**T3 NOT ACCEPTED — T4 REMAINS BLOCKED**
