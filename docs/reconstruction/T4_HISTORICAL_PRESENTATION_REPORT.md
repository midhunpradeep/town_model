# T4 — Historical Inspection and Presentation Reconciliation

**Status:** implementation submitted for independent review
**Repository:** `midhunpradeep/town_model`
**Issue:** [#7 — T4: Reconcile read-only historical inspection and presentation](https://github.com/midhunpradeep/town_model/issues/7)
**Branch:** `reconstruction/t4-historical-presentation`
**Accepted T3 starting implementation SHA:** `9a33ef291057c8b72565eaa28ef32f5e2fc66271`
**Code/test checkpoint:** `51a33ff449f6c13a1e26aff046890a096d43ff98` (the final report commit and exact submitted SHA are recorded in the issue handoff).

## 1. Scope and authority

T4 preserves historical stored information as archived evidence without restoring classifier construction, inference, loss evaluation, membership, margins, region reconstruction, or relation calculations. It adds no replacement classifier, scientific interpretation, objective, output meaning, or Phase 4 work. Author-owned classifier decisions remain outside this infrastructure task.

## 2. Readiness and claim evidence

Issue #7 was verified OPEN, `workflow:defined`, unassigned, with only native blocker #6; #6 was CLOSED with `workflow:accepted`, leaving zero open blockers. Issues #4 and #5 were also verified CLOSED/accepted. Issue #8 remained defined and was not a T4 prerequisite. No local or remote T4 branch/worktree existed at the time of claim.

The issue was transitioned to `workflow:ready`, assigned to the implementer, then transitioned to `workflow:in-progress` and re-read. The dedicated worktree `/tmp/town-model-t4-historical-presentation` and branch `reconstruction/t4-historical-presentation` were created directly from the accepted T3 SHA above. Readiness, claim, investigation/gap inventory, and implementation checkpoints are recorded in [the issue journal](https://github.com/midhunpradeep/town_model/issues/7).

Relevant journal entries:

- [Readiness verification and transition](https://github.com/midhunpradeep/town_model/issues/7#issuecomment-6558988743)
- [Claim and exact starting SHA](https://github.com/midhunpradeep/town_model/issues/7#issuecomment-6558992799)
- [Pre-implementation gap inventory](https://github.com/midhunpradeep/town_model/issues/7#issuecomment-6559055298)
- [Implementation checkpoint](https://github.com/midhunpradeep/town_model/issues/7#issuecomment-6559088027)

## 3. Pre-implementation gap inventory

The inventory was posted before tracked source/test changes. No classifier mathematics was needed.

### Already satisfied by accepted T3

| Criterion | Source and executable evidence |
|---|---|
| Stored snapshot values are read without model construction or recomputation | `src/town_model/inspection/image_snapshots.py`; exact dtype/value round-trip in `tests/test_image_snapshot_integrity.py::test_reader_roundtrips_nonempty_diagnostic_axes_and_stored_values_without_reconstruction`. |
| Snapshot integrity and safe discovery | Schema/checksum/sample identity/offset/axis/geometry-header checks in `image_snapshots.py`; signed/unsigned offset, including unsigned-safe `np.any(offsets[1:] < offsets[:-1])`, and corruption/discovery tests in `test_image_snapshot_integrity.py`; corrupt dashboard fixture in `tests/test_dashboard_app.py`. |
| Read-only run metadata and retained historical configuration identity | `experiments/image_runs.inspect_run`, `inspection/synthetic_metadata.inspect_saved_run`, and `experiments/image_config.py`; read-only inspection tests in `test_classifier_backend_unavailable.py`, identity compatibility in `test_phase3_config.py`. |
| Image Embeddings has no model controls, inference, or geometric rendering | `dashboard/image_embeddings.py` is metadata-only, captions persisted values and unavailable operations, and loads checksummed snapshot metadata only; AppTests and AST boundary checks in `test_dashboard_app.py` and `test_phase3_architecture.py`. |
| Geometry Laboratory is explicitly unavailable | `dashboard/app.py::_render_geometry_laboratory`; AppTest/import checks in `test_classifier_backend_unavailable.py`. The former plotting module remains absent. |
| Relation calculation is unavailable and generic JSON publication remains independent | `inspection/relation_diagnostics.py::diagnose_region_relation` rejects centrally; `save_relation_diagnostic` remains a generic atomic/no-clobber JSON writer; `test_phase3_d5b_relation_publication.py` covers its publication/durability behavior. |
| Dataset CLI/dashboard and integrity gates remain active | `cli.py`, `application.py`, `dashboard/app.py`, and `dashboard/data.py`; dataset service/integrity and dashboard tests in `test_application_cli_dashboard.py`, `test_dashboard_app.py`, and `test_phase3_dashboard_data.py`. `_draw_annotations` reads authoritative `record.components[*].bbox_xyxy_original`. |
| Model-dependent operations fail before artifact publication; imports do not load removed implementations | `test_classifier_backend_unavailable.py`, `test_phase3_architecture.py`, and `test_t3_architecture.py`; T3 source still has no historical implementation modules. |
| Explicit lifecycle/storage writers are separate from inspection | `transition_run`, `reconcile_worker_status`, `RunWriterLock`, and `_atomic_json` remain explicit utilities in `image_runs.py`; `inspect_run` itself reads the manifest and does not invoke reconciliation. |

### Partially satisfied; bounded corrections made

1. **Missing snapshot axis-schema field.** `load_image_snapshot()` used `header.get("diagnostic_axes_schema", {})`, so a correctly checksummed v2 snapshot with matching stored geometry but no axis-schema field was accepted. A temporary fixture reproduced the acceptance. The reader now rejects the missing required field with a controlled `ValueError`; an explicit empty object remains valid. No persisted schema was rewritten.
2. **Machine-readable CLI historical status.** Dashboard presentation and CLI help supplied human context, but `town experiment image inspect` and `town inspect-run` JSON had no explicit status marker. Both now add top-level `evidence_status: "historical_stored_metadata"`; nested archived manifest/config/metrics/environment values are unchanged.
3. **Lifecycle sidecar nonmutation coverage.** Existing tests proved run-tree nonmutation, but not adjacent `.image-workers` metadata. CLI and dashboard regressions now include a deliberately stale worker status sidecar and verify the complete temporary tree remains byte-identical after inspection. No lifecycle implementation needed modification.

### Blocked by author decision—not implemented

Recomputing or visually/scientifically interpreting stored classifier values, membership, margins, geometry, or relations would require future author-approved mathematical semantics. Such operations remain explicitly unavailable. No decision from issues #2/#3 was used.

## 4. Implemented changes and rationale

- `src/town_model/inspection/image_snapshots.py`: require `diagnostic_axes_schema` in the existing v2 header and preserve the existing controlled validation of its object, keys, axis names, dimensionality, and declared arrays.
- `src/town_model/cli.py`: label the two JSON historical-run inspection outputs with a top-level evidence status without changing archived values or run files.
- `tests/test_image_snapshot_integrity.py`: add a reader-only regression for a valid-checksum v2 snapshot missing the axis-schema field.
- `tests/test_classifier_backend_unavailable.py` and `tests/test_dashboard_app.py`: extend read-only tests to cover worker-status sidecars and CLI evidence labels.
- `tests/test_t3_architecture.py`: account for the one new T4 pytest node separately from T3's five final-integrity regression nodes. This was required because the accepted T3 architecture test asserted exact collection reconciliation. The T3 retirement manifest itself is unchanged.

No page was rebuilt, no array visualization was introduced, and no old plotting or model code was restored.

## 5. Historical stored-data compatibility

- `load_image_snapshot()` returns copies of stored NPZ arrays without converting their dtypes/values or deriving model outputs. The checksum, schema version, sample identity, offset shape/order, diagnostic-axis declarations, and stored geometry/header agreement checks remain in place.
- The exact accepted unsigned-offset check remains `np.any(offsets[1:] < offsets[:-1])`; stored arrays are not sorted, repaired, or rewritten.
- Historical classifier fields in configuration remain serialization/identity metadata. Existing config identity compatibility tests pass; no field is translated into an executable model contract.
- `inspect_run()` and `inspect_saved_run()` remain read-only JSON readers. The CLI presentation marker is outside the archived payload; the archived values are not edited.
- Corrupt snapshots are rejected by the reader, omitted by discovery, and shown as unavailable by the dashboard rather than as successfully validated evidence.

## 6. Dataset CLI and dashboard preservation

Dataset `generate`, `validate`, and `inspect` dispatch remains separate from model operations. Dashboard overview/explorer retain provenance, generator-type distributions, class vocabularies, known/unknown supervision, current integrity gating, sample selection, original-image loading, component IDs, annotation boxes, targets/masks, and explicit generation controls/monitoring.

Opening the dashboard does not generate a dataset or start model execution. The explorer continues to block when current integrity is not `current_pass`; saved validation is labeled historical and does not override current artifact validation. Annotation overlays use dataset component `bbox_xyxy_original`, not classifier geometry.

## 7. Image Embeddings and Geometry Laboratory

**Image Embeddings:** remains metadata-only. It displays persisted manifest/configuration/metrics under an explicit historical/persisted caption, offers only checksum-validated snapshot metadata, and has no launch/train/resume/predict/score/geometry controls. It does not expose arrays as newly interpreted plots or predictions. Invalid snapshots are safely reported unavailable.

**Geometry Laboratory:** remains an explicit unavailable state via `ClassifierBackendUnavailableError`. It does not reconstruct regions, render saved geometry as future-model semantics, or report a simulated experiment as successful.

## 8. Relations, CLI boundaries, and storage separation

`diagnose_region_relation()` raises the centralized unavailable error before reading a snapshot or publishing output. It cannot calculate membership or relation outcomes. `validate_snapshot_ontology_mapping()` remains an independent check against the dataset-owned vocabulary. `save_relation_diagnostic()` remains a generic caller-supplied JSON storage operation with its accepted atomicity, no-clobber, error, and durability guarantees.

Read-only CLI/dashboard inspection does not invoke `reconcile_worker_status()`, write worker status, transition lifecycle state, create snapshots, load checkpoints, or acquire writer locks. Explicit lifecycle/state utilities remain available as separate mutation operations; they were not removed as a model-execution fallback.

Supported dataset CLI commands and dashboard startup remain functional. Representative model-dependent commands return exit status 2 with the actionable centralized error, no traceback, and no new output paths.

## 9. Import/source architecture

A fresh Python process imported all of:

- `town_model.cli`
- `town_model.dashboard.app`
- `town_model.dashboard.image_embeddings`
- `town_model.inspection.image_snapshots`
- `town_model.inspection.relation_diagnostics`
- `town_model.inspection.synthetic_metadata`
- `town_model.application`
- `town_model.models.image_points`

No removed classifier implementation module was loaded. The full suite's source/import architecture tests also pass. All eight T3-retired implementation modules remain absent. No lazy page/command path reintroduces an old algorithm.

## 10. Tests and verification

Environment: repository Python 3.12 virtual environment, CPU-only PyTorch. No model training or CUDA run was performed.

### Baseline at exact accepted T3 SHA

```bash
cd /tmp/town-model-t4-historical-presentation
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q
```

Result: **201 passed, 0 failed, 0 skipped, 7 existing Streamlit/Pandas warnings** (53.98 s).

An initial test invocation accidentally ran from the original T2 checkout and failed retired historical-model tests (188 passed / 130 failed). The working-directory error was identified and the command was rerun from the exact T4 worktree above. The T2 run is not used as the baseline or attributed to T4.

### Focused tests

```bash
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q \
  tests/test_image_snapshot_integrity.py tests/test_dashboard_app.py \
  tests/test_classifier_backend_unavailable.py tests/test_phase3_architecture.py \
  tests/test_phase3_config.py tests/test_application_cli_dashboard.py \
  tests/test_phase3_d5b_relation_publication.py \
  tests/test_phase3_runs.py::test_run_config_identity_and_original_subset_order_remain_metadata_only \
  tests/test_phase3_relations.py tests/test_t3_architecture.py
```

Result: **80 passed, 0 failed, 0 skipped, 7 existing warnings** (25.89 s). This includes historical value fidelity/checksum/corruption, CLI fail-closed behavior, read-only metadata and worker-sidecar checks, dataset integrity/dashboard pages, relation/storage, import/source architecture, and retirement-manifest reconciliation.

### Final full suite

```bash
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q
```

Result: **202 passed, 0 failed, 0 skipped, 7 existing Streamlit/Pandas warnings** (51.30 s). The current collected suite is 202 tests: the accepted T3 checkpoint remains 196, plus its five final-integrity nodes, plus one separately inventoried T4 regression node. The T3 retirement manifest remains unchanged.

### Compilation and diff

```bash
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests
git diff --check
```

Both passed. The required architecture test independently resolves every accepted T3 replacement reference and retains its fixed disposition-count invariants.

## 11. CLI and dashboard smoke results

With `PYTHONPATH=src` and the repository `.venv/bin/town`:

- `town --help` — exit 0; supported dataset commands remain listed.
- `town generate --help` — exit 0; explicit dataset-generation interface remains available (no dataset was generated for this task).
- `town inspect examples/pizza-48 --json` — exit 0; dataset `pizza-dcc148789aecf8af86a3`, 48 samples, `current_pass`.
- `town validate examples/pizza-48 --no-write-report --json` — exit 0; valid, zero errors.
- Temporary historical image/synthetic inspection fixtures pass `test_image_and_synthetic_metadata_inspection_are_read_only`; outputs identify archived metadata and the run tree plus worker sidecar remain byte-identical.
- Representative image training, image matrix, checkpoint evaluation, relation diagnosis, and geometry export commands each exited 2 with the centralized actionable unavailable message, no traceback, and no newly created run/result path.
- Dashboard started on temporary port 46753: `/_stcore/health` returned `ok`, root returned HTTP 200, and the process was stopped with SIGTERM after the check.
- AppTest smoke passed for Dataset overview, Image explorer, Generate dataset controls, Geometry Laboratory unavailable state, and Image Embeddings historical metadata state. The existing corrupt-snapshot AppTest also passes without an uncaught dashboard exception. Reproducible page tests are included in the focused pytest command above (`test_dashboard_app.py` and `test_classifier_backend_unavailable.py`).

## 12. Artifact nonmutation and preserved audit evidence

- The canonical example validation report retained SHA-256 `cd371bd190796552fdf831f084b365d48af42d363c448f9952fc5a2fafe07925` after the `--no-write-report` validation smoke.
- `docs/reconstruction/T3_TEST_RETIREMENT_MANIFEST.md` has the same SHA-256 at the accepted T3 base and T4 worktree: `b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3`. Its 176 original IDs, 132 accepted-T2 failures, 96/77/3 node dispositions, and 121 replacement references were not rewritten.
- No canonical datasets, research artifacts, checkpoints, scientific specification, author-owned decision records, accepted reviews, accepted implementation refs, or `main` were modified. The original T2 checkout and accepted T3 worktree remained clean.
- No dataset was generated and no model artifact was published.

## 13. Remaining limitations

- Historical classifier/geometry values remain inspectable only through validated storage readers; no current predictions, geometry plots, or relation results are provided.
- The Image Embeddings page intentionally does not render stored numeric arrays. It presents metadata and provenance only, which avoids implying scientific meaning.
- Historical snapshot headers without the required v2 diagnostic-axis schema now fail closed; no snapshot is repaired or rewritten.
- Verification used CPU-only Python 3.12. It makes no CUDA or classifier-training claim.
- The seven existing Streamlit/Pandas deprecation warnings remain unrelated and non-failing.

## 14. Independent-review handoff

Review the final pushed branch revision recorded in the issue #7 handoff. Independently verify the criteria listed in issue #7 and in the assignment, especially: historical values/checksums/provenance, the explicit historical CLI marker, missing-axis-schema rejection, no prediction/relation/geometry recomputation, dataset annotation/integrity behavior, no inspection lifecycle mutation, import absence, unavailable-operation side-effect ordering, unchanged retirement manifest, and the full 202-test suite.

The implementer does not accept or close issue #7. T5 remains blocked pending independent T4 acceptance; no Phase 4 work was started.
