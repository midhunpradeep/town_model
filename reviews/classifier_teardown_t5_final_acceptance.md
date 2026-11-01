# T5 — Final Independent Acceptance of the Classifier Teardown

## A. Final verdict

**ACCEPT TEARDOWN**

This verdict is against the exact accepted T4 implementation revision `19222424f5ad8347deddc6dfb2d7187bd85609c0`, not the separate T4 review-report commit. The teardown is internally consistent: historical classifier implementations and their executable mathematics are absent from active source; retained data, vision, point, reducer, identity, storage, and read-only inspection seams work; model-dependent operations reject explicitly without publishing or mutating model artifacts.

## B. Provenance, independence, and readiness

- Repository: `midhunpradeep/town_model`.
- Reviewed implementation: `19222424f5ad8347deddc6dfb2d7187bd85609c0`, branch `reconstruction/t4-historical-presentation`.
- T5 review branch: `review/classifier-teardown-t5-independent`, created directly from that exact SHA in `/tmp/town-model-t5-independent-review`.
- Exact T4 branch ref was verified at claim and after verification; `origin/reconstruction/t4-historical-presentation` still resolves to the target SHA. The separate T4 review report is at `94f602ec3740f3a383250243e1c36e7cde09753b` and was not merged into the implementation.
- The target is a descendant of accepted historical revision `2de9c5f29b436f727720e87132d0b4be25a40768`, accepted T2 `c6aad87a45ac75835f02670f32fdeca5167f18f7`, and accepted T3 `9a33ef291057c8b72565eaa28ef32f5e2fc66271`. Its parent is T4 source/test checkpoint `51a33ff449f6c13a1e26aff046890a096d43ff98`.
- Reviewer identity: fresh Pi agent session `01a1978c-df23-7741-b8e6-6d21f7d257af`. This session did not implement or review T0/T1–T4. The authenticated GitHub principal is the shared repository-owner account `midhunpradeep`, which also appears as the historical stage assignee/committer; this account-level limitation was disclosed in the T5 readiness and claim comments. Independence is at the separate agent/session level, consistent with the repository's recorded fresh-session handoffs, not a claim of a separately authenticated human account.
- Readiness: live GitHub state confirmed #4, #5, #6, and #7 CLOSED with `workflow:accepted`; native GraphQL `Issue.blockedBy` for #8 listed exactly those four issues, all CLOSED (zero open blockers). Issue #8 was OPEN, unassigned, `workflow:defined`, without a T5 journal or competing claim/worktree. The coordinator readiness transition and fresh-session claim were recorded before review. See [readiness comment](https://github.com/midhunpradeep/town_model/issues/8#issuecomment-6559427279) and [claim comment](https://github.com/midhunpradeep/town_model/issues/8#issuecomment-6559432520).
- Author-owned issues #2/#3 remain OPEN and unapproved; neither is a T5 implementation requirement. No author-owned classifier mathematics was used.

## C. Stage history and acceptance evidence

| Stage | Accepted implementation / evidence | Cross-stage result |
|---|---|---|
| T0/T1 — interface separation | Implementation `fbed1bc008127f7024aeabd1641519e2a333a840`; independent report at `cefbbc7a4334c28bec1d6071d700a26bb4977d21`; issue #4 CLOSED `workflow:accepted`. | Separated `ImagePointModel`/point transport from the historical adapter while preserving the old implementation temporarily. Dataset-owned class ordering and image-to-point seams were established without new classifier semantics. |
| T2 — execution disconnection | Remediation implementation `a95f2630bc67d4d0445b3fea2ac0038c3115cfac`; accepted T2 HEAD/report commit `c6aad87a45ac75835f02670f32fdeca5167f18f7`; issue #5 CLOSED accepted. | Centralized actionable `ClassifierBackendUnavailableError` and guarded direct objectives and model operations before model/artifact effects. The first independent review's direct-objective finding was corrected and re-reviewed. |
| T3 — implementation removal | Source/test checkpoint `589e4e0850e657b72cf81267310595a1e669bc77`; accepted T3 HEAD/report `9a33ef291057c8b72565eaa28ef32f5e2fc66271`; independent acceptance `1bc384207591e483ce616c53f919baa150526aea`; issue #6 CLOSED accepted. | Deleted historical definitions and algorithms; retained independent foundations and added the retirement manifest. Earlier snapshot/manifest/unsigned-offset findings were corrected and independently re-reviewed. |
| T4 — historical presentation | Source/test checkpoint `51a33ff449f6c13a1e26aff046890a096d43ff98`; exact implementation HEAD `19222424f5ad8347deddc6dfb2d7187bd85609c0`; independent report `94f602ec3740f3a383250243e1c36e7cde09753b`; issue #7 CLOSED accepted. | Required explicit snapshot-axis schema, labeled both CLI inspection payloads as archived metadata, proved read-only sidecar behavior, and preserved dataset presentation. |

T5 did not infer acceptance from these separate reports: it independently inspected the combined source/import/call paths, re-ran the current full and focused suites, exercised read-only and failure paths with temporary fixtures, and checked the exact target's ancestry and integrity.

## D. Historical algorithms and imports

Filesystem inspection confirms these eight active-source modules are absent:

```text
src/town_model/models/classifiers.py
src/town_model/models/geometry.py
src/town_model/models/image_embeddings.py
src/town_model/models/classifier_transport.py
src/town_model/models/image_model_validation.py
src/town_model/training/image_objectives.py
src/town_model/experiments/synthetic.py
src/town_model/dashboard/geometry.py
```

A fresh process imported all retained public paths requested by T5—`town_model.cli`, `town_model.application`, `town_model.dashboard.app`, `town_model.dashboard.image_embeddings`, `town_model.inspection.image_snapshots`, `town_model.inspection.synthetic_metadata`, `town_model.inspection.relation_diagnostics`, `town_model.models.image_points`, and `town_model.experiments.visual_features`—plus the retained generic reducer modules. None loaded a removed implementation. Explicit import attempts for all eight removed module names each raised `ModuleNotFoundError`. The T3 architecture test also checks imports, package exports, and executable definitions.

Independent active-source scans found no old classifier classes/factory, trainable box/town classifier geometries, hard membership, signed-margin calculation, classifier scoring/objective, model-dependent synthetic training, checkpoint reconstruction, or import of the removed modules. Searches for `torch.load`, `load_state_dict`, old score-to-loss calls, and old geometry evaluation found no active reconstruction path. The only loss-formula strings found are in `experiments/image_config.py`'s documented historical identity serializer/validation; these are archived configuration metadata and are not consumed as scoring functions. `models/geometry_diagnostics.py` is axis-name metadata only. `ImageGeometry` and box transforms in the vision preprocessing path describe image resize/padding and original-pixel ROIs, not classifier regions. Ontology relationship metadata remains ontology-only; `diagnose_region_relation()` itself rejects.

No hidden constructor/factory fallback or checkpoint-based model loading remains reachable through CLI dispatch, dashboard page selection, image configuration/preflight, point projection, historical run inspection, snapshot reading, or relation diagnosis.

## E. Dataset, ontology, and supervision preservation

From the T5 worktree, the installed CLI returned:

- `town --help` — exit 0.
- `town generate --help` — exit 0; generation remains explicit.
- `town inspect examples/pizza-48 --json` — exit 0; dataset `pizza-dcc148789aecf8af86a3`, 48 samples, `current_pass`, 16 samples in each of train/validation/test. Provenance, ontology/vocabulary identity, and per-split class support were present.
- `town validate examples/pizza-48 --no-write-report --json` — exit 0, valid, zero errors, 178 support warnings, nine reported integrity checks. The canonical report SHA-256 remained `cd371bd190796552fdf831f084b365d48af42d363c448f9952fc5a2fafe07925` before and after.

An independent `load_prepared_dataset(..., validate=True)` inspection confirmed 48 unique records; a 60-ID canonical trainable class ordering; one fixed-positive class and five unsupported classes; 66-wide pizza/component target and known-mask vectors; fixed-positive targets are known positive and unsupported values remain unknown; 803 component occurrences have unique IDs and valid original-pixel boxes; and the original splits remain 16/16/16. The pinned ontology loaded at its required hash; Germany remains an ontology individual, not a class. Manifest source provenance identifies the pinned Pizzaïolo repository and commit `6c0bcb221a957c184012beaf103bebcde4b1f793`.

No data generation or canonical data rewrite was performed. The example data remains usable without classifier training. Generation itself requires either an already clean checkout of the pinned external Pizzaïolo source or a fetch/checkout of that pinned source into the ignored cache. In this worktree `PIZZAIOLO_SOURCE_DIR` was unset and `artifacts/sources/pizzaiolo` was absent; only generation help and the existing generator/pinned-source tests were exercised. This is a documented generation dependency, not a teardown blocker.

## F. Vision, ROI, and image-to-point infrastructure

`tests/test_phase3_vision.py` passed all seven tests in the focused and full suites. Coverage includes RGB/ImageNet letterboxing and box round trips, MobileNetV3-Large/FPN feature maps/channels, ROIAlign, replaceable ROI providers, original-pixel box transforms, ragged and zero-component batches, and frozen/trainable FPN gradient behavior. The path emits reusable 256-channel FPN/ROI/global features; no classifier predictions are generated.

`ImagePointModel` remains an independent mapping from visual features and caller-supplied sample/occurrence identity to component/global/composed/hybrid points. Focused and full tests cover dimensions 2, 5, 8, and 16; point shapes, valid masks, identities, composition modes, empty ROIs, permutation behavior and deterministic construction. Its `ImagePointResult` has no membership, logits, classifier outputs, or predictions. The classifier-free point path and its composition remain importable without the retired classes.

Dataset ROI annotations remain original-pixel component boxes; these are not interpreted as classifier geometry.

## G. Generic reductions, identity, and storage

`models.losses.reduce_supervision`, `training.supervision.reduce_weighted_instances`, and `aggregate_effective_batch` remain caller-value reducers, not score converters or an approved classifier objective. The focused/full suite retains numerical known-mask/positive-weight behavior, eligible-instance counts, unknown exclusion, effective-batch aggregation, zero-gradient behavior where required, and gradient-flow tests. No signed-margin convention or historical classifier-specific objective remains.

Historical config/run identity remains serializable, while `experiments/visual_features.py` and `feature_cache.py` identify/cache only visual feature state and dataset/ROI/config provenance. Focused/full tests cover feature-cache integrity/identity, config identity compatibility, run metadata and locks. `inspect_run()` and `inspect_saved_run()` are read-only JSON readers. Explicit lifecycle transitions and generic atomic/no-clobber JSON publication remain separate callable storage utilities; read-only inspection does not invoke them. The relation-publication tests pass without restoring relation computation.

## H. Snapshot schema, integrity, and value fidelity

`image_snapshots.py` reads stored v2 header/NPZ values only. It validates schema, required explicit `diagnostic_axes_schema`, array checksum, sample-ID/header agreement, signed/unsigned component offsets, zero start, adjacent monotonicity (including the unsigned-safe `np.any(offsets[1:] < offsets[:-1])` check), total-component agreement, axis declarations/array dimensions, geometry/header consistency, and controlled corruption errors. It copies stored arrays and does not recompute or normalize classifier values.

In addition to the 20 focused snapshot integrity tests, I built independent checksum-valid temporary fixtures in `/tmp/t5-snapshot-fixtures-n78l9rt_` and observed:

- valid non-empty axes and exact stored numeric dtypes/values load unchanged;
- explicit empty axis schema loads; missing axis schema rejects with a `header.diagnostic_axes_schema` `ValueError`;
- decreasing unsigned offsets `[0,5,3]` reject; valid unsigned repeated offsets `[0,0,3]` load unchanged;
- missing geometry/header agreement rejects with a controlled field-specific error;
- corrupt NPZ bytes with a matching checksum reject through a controlled reader error; discovery keeps a valid sibling and omits the corrupt one.

The full integrity suite additionally covers signed offsets, empty intervals, geometry shape/value mismatch, malformed geometry/axes, invalid discovery, and checksum failures. No archived production snapshot `header.json` or `arrays.npz` exists in this checkout; `find` found none and Git tracks none. Thus schema-v2 compatibility is independently demonstrated with the documented writer contract and constructed fixtures, but artifact-by-artifact compatibility against a real production archive is not claimed. This limitation does not block acceptance of the testable schema contract.

## I. Historical inspection and dashboard presentation

Independent temporary image and synthetic run fixtures were inspected through the actual `town` executable using both supported routes:

```text
town experiment image inspect <run>
town inspect-run <run>
```

Both exited 0 and emitted top-level `evidence_status: historical_stored_metadata`. Archived nested manifest/config/metrics/environment values compared exactly with the fixture values. A stale `.image-workers` sidecar, run files, and surrounding directory inventory were byte-identical before and after. No model/checkpoint evaluation or lifecycle reconciliation occurred.

Dashboard AppTests exercised all five pages without exceptions:

- Dataset overview: displayed 48 samples and current integrity PASS; the overview has class support/provenance surfaces.
- Image explorer: loaded the prepared dataset, sample and component tables, target/mask presentation, and original-image annotation path. A direct canonical-data check loaded the original 224×224 image and verified that `_draw_annotations` painted the selected component's original-pixel box at `(124,155,173,178)` without changing the image dimensions.
- Generate dataset: explicit controls and validation form were visible; no generation future/output was started by opening the page.
- Image Embeddings and Geometry Laboratory: both showed the explicit intentionally-disconnected state and no model/action buttons. Existing AppTests exercised stored metadata, checksummed snapshots, and corrupt snapshot handling.

The actual Streamlit server started on temporary port 36219; `/_stcore/health` returned `ok`, `/` returned HTTP 200, and the server exited cleanly with status 0 after termination. Dashboard startup and page selection did not run training or generate data.

## J. Unavailable-operation matrix and artifact non-publication

Twelve actual CLI routes were invoked as subprocesses with temporary nonexistent paths or sentinel-bearing existing paths. Every route exited **2**, emitted its operation-specific `ClassifierBackendUnavailableError` presentation with the shared actionable “intentionally disconnected / no reconstructed classifier backend is currently approved” text, and produced no traceback or `ModuleNotFoundError`:

| Public operation | Verified route | Result |
|---|---|---|
| Synthetic fixed experiment/training | `experiment fixed` | Exit 2, unavailable |
| Synthetic point-mapping experiment | `experiment mapping` | Exit 2, unavailable |
| Image experiment preflight | `experiment image preflight` | Exit 2, unavailable |
| Image training | `experiment image run` | Exit 2, unavailable |
| Image experiment matrix | `experiment image matrix` | Exit 2, unavailable |
| Image resume | `experiment image resume` | Exit 2, unavailable |
| Historical image checkpoint evaluation | `experiment image evaluate` | Exit 2, unavailable |
| Image worker stop | `experiment image stop` | Exit 2, unavailable |
| Image relation diagnosis | `experiment image relations` | Exit 2, unavailable |
| Synthetic evaluation | `evaluate` | Exit 2, unavailable |
| Synthetic resume | `resume` | Exit 2, unavailable |
| Synthetic geometry export | `export-geometry` | Exit 2, unavailable |

Direct API calls independently raised `ClassifierBackendUnavailableError` for image module construction, training, evaluation, checkpoint evaluation/reconstruction, run/resume, background launch, matrix, stop, worker execution, snapshot generation, and relation diagnosis. A valid image preflight fixture validated its dataset/config then rejected before model construction.

The CLI run used a nonexistent output path and existing directories/files containing sentinels; direct API checks used an existing run with manifest, lock, stop request, checkpoint, snapshot, and worker sidecar sentinels. Full temporary tree SHA-256 inventories before and after were identical. No run manifest, model checkpoint, snapshot array, relation result, geometry export, worker status, or partial output was published; no existing data changed. The dashboard Geometry Laboratory is explicitly unavailable, and the retired geometry plotting module cannot import.

## K. Retirement manifest and test reconciliation

`docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` is byte-identical to the accepted T3 version at `1bc384207591e483ce616c53f919baa150526aea`: SHA-256 `b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3`.

Independent parsing confirmed:

- 176 unique original node IDs across 129 groups;
- 132 unique accepted-T2 historical failure IDs plus 44 disjoint additional accounted IDs;
- independently anchored node dispositions: 96 deleted, 77 replaced, 3 split;
- group dispositions: 68 deleted, 58 replaced, 3 split;
- 121 replacement references over 40 distinct current test nodes, all resolving against the exact T5 target's 202-node pytest collection.

The accepted-T3 architecture test preserves fixed disposition constants and rejects coordinated in-memory manifest/count mutations. It inventories five final T3 integrity regression nodes and the one T4 schema regression separately: `196 + 5 + 1 = 202`. The T4 node is not silently added to T3's manifest. The ontology metadata and independent entailment assertions remain covered by `test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations` and `test_mushroom_vegetarian_ontology_entailment_is_independent_of_geometry`. The accepted T3 independent report records the actual accepted-T2 collection (323 nodes) and full-suite baseline (191 passed/132 historical failures); this T5 review verified the manifest's complete unique accounting and all current references without running the old T2 suite from a different worktree.

## L. Cross-stage integration assessment

No cross-stage defect was found. The T0/T1 point seam still imports and runs after T2 disconnect/T3 deletion; all required public import paths remain free of retired modules. Dataset class order and target/mask vocabulary remain dataset-owned, while visual features and points do not emit classifier results. Generic reductions accept caller-supplied values without embedding margin conversion. Cache identity derives from visual state/config/data, not a classifier. Historical configuration formula strings remain archival identity metadata only. Snapshot loading returns checksummed stored values without geometric interpretation. CLI/dashboard history inspection does not load checkpoints or reconcile lifecycle/worker status. Model-dependent CLI/API operations reject before publication and preserve sentinels. No renamed/embedded equivalent classifier or implicit fallback was found.

## M. Verification and environment

All commands and inspections used `/tmp/town-model-t5-independent-review` as the current worktree. The isolated worktree has no `.venv`; the installed Python environment was used by absolute path: `/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python` (Python 3.12.0, PyTorch 2.5.1+cpu, CUDA unavailable).

- Required full suite: `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q` — **202 passed, 0 failed, 0 skipped**, 7 existing non-failing Streamlit/Pandas `Pandas4Warning` warnings, 48.12 seconds.
- Focused preserved-boundary selection — **133 passed, 0 failed, 0 skipped**, 7 warnings. Exact selection:

```bash
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q \
  tests/test_image_snapshot_integrity.py \
  tests/test_dashboard_app.py \
  tests/test_classifier_backend_unavailable.py \
  tests/test_classifier_teardown_boundary.py \
  tests/test_phase3_architecture.py \
  tests/test_phase3_config.py \
  tests/test_application_cli_dashboard.py \
  tests/test_phase3_d5b_relation_publication.py \
  tests/test_phase3_runs.py::test_run_config_identity_and_original_subset_order_remain_metadata_only \
  tests/test_phase3_relations.py \
  tests/test_t3_architecture.py \
  tests/test_phase3_vision.py \
  tests/test_phase3_dataset.py \
  tests/test_phase2_models_losses.py \
  tests/test_phase3_models.py \
  tests/test_generator.py \
  tests/test_generator_worker.py \
  tests/test_image_validation_infrastructure.py \
  tests/test_phase3_d5a_ontology_entailments.py \
  tests/test_phase3_cache.py
```

- Required collection: `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest --collect-only -q` — **202 collected**; every T3 replacement reference independently resolved.
- Required source compilation: `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests` — passed.
- Required stage diff check: `git diff --check 9a33ef291057c8b72565eaa28ef32f5e2fc66271..19222424f5ad8347deddc6dfb2d7187bd85609c0` — passed.
- Cumulative teardown diff from accepted historical implementation `2de9c5f29b436f727720e87132d0b4be25a40768` was inventoried: 75 changed paths, 6,436 insertions / 9,296 deletions, including removal of the historical algorithm modules and preservation/refactoring of the independent paths described above. It touches no `examples/`, `artifacts/`, scientific specification, prior phase review report, or ADR.
- CLI smoke, temporary direct API/CLI failure fixtures, custom v2 snapshot fixtures, dataset inspection, dashboard AppTests/server startup, source/import scans, manifest parsing, and exact remote-ref/ancestry checks are described in sections D–K.

The full suite emitted only the seven non-failing Pandas/Streamlit deprecation warnings. Direct AppTest invocations also emitted Streamlit's expected bare-mode `ScriptRunContext` warnings. No CUDA execution, historical classifier training, or model evaluation was performed or is claimed.

## N. Artifact and accepted-history integrity

- The T5 review worktree was clean at exact target before report creation. No implementation source, tests, canonical data, research artifact, checkpoint, or specification was edited.
- `examples/` is unchanged across the accepted historical-to-T4 diff; `artifacts/` is absent in the target checkout; no snapshot archive files are present. The canonical validation report hash stayed unchanged through no-write validation.
- The accepted historical commit remains an ancestor and its Git objects/refs were not rewritten. Accepted implementation branches and review commits were not merged, amended, or force-pushed. `main` was not touched; no promotion/merge was attempted.
- Issues #2/#3 and the expanded scientific draft were not changed. No phase acceptance history was rewritten.

## O. Limitations

- Verification is CPU-only, with PyTorch `2.5.1+cpu`; CUDA behavior is not established.
- No historical classifier training, inference, checkpoint reconstruction, or scientific evaluation was run; these are intentionally unavailable.
- No external Pizzaïolo checkout was present, so generation was not run. The committed example dataset was independently inspected and no-write validated; generation's pinned-source requirement is recorded above.
- No archived production snapshot files were available. T5 verifies the documented schema-v2 contract and independently constructed fixtures but does not claim a real-production-archive census.
- Seven existing non-failing Streamlit/Pandas deprecation warnings remain; AppTest bare-mode context warnings are also non-failing.
- No scientific semantics, replacement classifier, loss, or Phase 4 work is accepted or begun.

## P. Issue #8 acceptance-criteria mapping

| Issue #8 criterion | Independent evidence | Result |
|---|---|---|
| Static/import checks show no active historical definitions, factories, objectives, membership algorithms, or synthetic trainer | Filesystem absence; fresh-process imports; eight explicit import failures; source scans; T3 architecture test | Pass |
| Retained dataset/ontology/known-mask, vision/ROI, point/composition, generic identity/storage, and historical readers work | Dataset CLI/API and 48-record audit; focused/full dataset, vision, point, reducer, cache, identity, storage, and snapshot suites; custom snapshot fixtures | Pass |
| Unsupported execution/geometry operations fail explicitly before artifact publication | 12 CLI routes exit 2 with actionable central errors; direct API matrix raises typed errors; new-output and sentinel inventories unchanged | Pass |
| Dataset CLI/dashboard operate without training; stored historical values are read without recomputation | CLI help/inspect/no-write validate; live dashboard health/root; five-page AppTest; original-pixel annotation fixture; both historical CLI markers/payloads and sidecar hashes unchanged | Pass |
| T0/T1 compatibility and retirement manifest reviewed; no new scientific semantics | Point-only interface and import tests; accepted T0/T1 report; byte-identical manifest and independently reconciled counts/references; no active equivalent math found | Pass |
| Reviewer records exact SHA, commands/results, criteria, limitations, and verdict; reviewer alone accepts/closes | This report against exact `19222424…`; exact test/compile/diff/CLI/dashboard evidence and limitations; fresh-session reviewer handoff | Pass |

## Q. Final reconstruction-readiness statement

**ACCEPT TEARDOWN — The historical neural baseline, box and town classifier implementations, their membership/geometry calculations, classifier-specific objectives and model-dependent training paths have been removed from active execution. Unsupported operations fail explicitly without producing model artifacts. Dataset generation, ontology, known-mask supervision, vision/FPN/ROI, image-to-point projection, composition, generic reductions, identity, storage, checksum-validated historical inspection, and dataset presentation remain functional. Historical classifier outputs are preserved only as archived evidence, without recomputation or scientific reinterpretation. The staged teardown has passed independent acceptance.**

**This acceptance establishes infrastructure readiness for author-led mathematical reconstruction. It does not approve classifier semantics, loss functions, geometric regions, experiments, model correctness, learning performance, or Phase 4 implementation.**
