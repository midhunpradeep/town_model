# Independent Review — T4 Historical Inspection and Presentation

## A. Verdict

**ACCEPT T4**

The submitted revision preserves read-only access to historical run and snapshot records without restoring classifier execution or giving stored values current mathematical authority. The v2 axis-schema validation correction is supported by the historical writer and documented schema contract; both CLI inspection routes mark their output as stored historical metadata; and independent CLI/dashboard checks found no run or worker-sidecar mutation. Dataset inspection and presentation remain usable, and the accepted T3 teardown evidence is unchanged.

## B. Provenance

- Accepted T3 implementation baseline: `9a33ef291057c8b72565eaa28ef32f5e2fc66271`.
- Accepted T3 independent review: `1bc384207591e483ce616c53f919baa150526aea` (`review/classifier-teardown-t3-final-integrity-independent`); the remote review ref resolves to this SHA.
- Submitted T4 branch: `reconstruction/t4-historical-presentation`.
- T4 code/test commit: `51a33ff449f6c13a1e26aff046890a096d43ff98`.
- Exact submitted T4 HEAD: `19222424f5ad8347deddc6dfb2d7187bd85609c0`; remote branch HEAD was independently verified equal before review and again before report creation.
- Ancestry is `9a33ef2` → `51a33ff` → `1922242`; the accepted T3 SHA is an ancestor of the exact T4 HEAD.
- Review worktree/branch: `/tmp/town-model-t4-independent-review`, `review/classifier-teardown-t4-independent`, created directly at the submitted T4 SHA. The exact review-report commit is linked in the issue #7 handoff.
- Exact T3-to-T4 diff contains only the seven expected paths: this report, `src/town_model/cli.py`, `src/town_model/inspection/image_snapshots.py`, and the four named test files (`test_classifier_backend_unavailable.py`, `test_dashboard_app.py`, `test_image_snapshot_integrity.py`, `test_t3_architecture.py`). No additional paths were present.
- Workflow before review: #4/#5/#6 CLOSED `workflow:accepted`; #7 OPEN `workflow:in-review`; #8 OPEN `workflow:defined`, natively blocked by #7 (with #4–#6 already closed). The complete #7 specification and issue journal, plus issues #4–#6/#8 and the author-led #2/#3 boundaries, were checked. Issues #2/#3 remain separate; neither authorizes replacement semantics.

## C. Audit-first assessment

The three gaps in the pre-edit inventory are legitimate and bounded:

1. **Absent `diagnostic_axes_schema`:** the accepted T3 loader defaulted a missing key to `{}` and therefore accepted a checksummed malformed v2 header. The T4 code changes this to a controlled integrity error identifying `header.diagnostic_axes_schema`. Independent reproduction is recorded in section D.
2. **CLI evidence marker:** the image `inspect` and synthetic `inspect-run` JSON outputs had no machine-readable indication that their values were archived metadata. The additive top-level `evidence_status` marker communicates that distinction without changing the stored payload. Repository consumers were searched; none assumes an exact top-level shape or parses these inspection responses in a way that conflicts with an additive key.
3. **Adjacent worker-status sidecars:** prior tests checked run-file nonmutation but not `.image-workers` metadata. The new stale-sidecar regressions cover the missing read-only guarantee for CLI and dashboard inspection.

The rest of the inventory is consistent with the accepted T3 source and tests: metadata inspection, metadata-only Image Embeddings, unavailable Geometry Laboratory and relation computation, dataset surfaces, model-operation rejection, and import boundaries were already present. Those behaviors were preserved rather than rebuilt. The only source changes are the axis-schema guard and two presentation markers; the remaining changes add regression coverage and account for the new test node. No unnecessary implementation or presentation redesign was found. No T4 acceptance criterion remains materially unverified; the absence of archived production snapshots in the checkout is disclosed below, not treated as proof of compatibility.

## D. Snapshot v2 schema compatibility

**Finding: requiring the field is consistent with the intended v2 contract; no evidence was found that a legitimate v2 writer omitted it.**

Evidence:

- `docs/PHASE_3_ACCEPTANCE_REMEDIATION_REPORT.md` §6 describes `image-research-snapshot-v2` as recording the diagnostic-axis schema and distinguishes unsupported v1 snapshots without trustworthy axes.
- `docs/PHASE_3_ARCHITECTURE.md` states that snapshot schema v2 declares diagnostic axes and that schema-v1 diagnostics lacking axes are rejected rather than guessed.
- The accepted T2 snapshot writer at `c6aad87a45ac75835f02670f32fdeca5167f18f7` constructs a `diagnostic_axes` dictionary and unconditionally includes `"diagnostic_axes_schema": diagnostic_axes` in every v2 header. The pre-teardown Phase 3 writer at `e1bdd41722c70160e56b4b1735f7ffd2fafb8684` does the same. Thus an explicitly empty object is representable, but omission is not an output of the v2 writer. The earlier v1 writer did not declare these axes and used the v1 schema.
- The accepted T3 reader's `header.get("diagnostic_axes_schema", {})` was permissive, but that implementation detail is weaker than the documented v2 contract. Its unsupported-schema message also says v2 expects explicit diagnostic axes. No historical writer or documented contract was found that defines omission as optional.
- Current tests and tracked files contain no archived production `header.json`/`arrays.npz` snapshots to census. The available historical v2 fixtures explicitly contain the field, including `{}`. This is a limitation on artifact-by-artifact compatibility evidence, not an unresolved contract ambiguity.

Independent checks at exact T4 HEAD:

- A hand-built, correctly checksummed v2 snapshot with valid sample IDs, `uint64` offsets `[0, 1, 1]`, matching scalar total, stored geometry array/header agreement, valid axis declarations, and no `diagnostic_axes_schema` raised exactly:
  `ValueError: invalid image research snapshot at <path>: header.diagnostic_axes_schema: is required by the snapshot schema`.
- A valid sibling with non-empty axis declarations remained readable and retained its stored arrays. A separately built valid v2 snapshot with explicit `diagnostic_axes_schema: {}` and no diagnostic arrays loaded successfully.
- `discover_image_snapshots()` excluded the missing-field snapshot and returned both valid siblings. It did not crash.
- AppTest with only a missing-field snapshot rendered the normal no-valid-snapshot/unavailable state without an uncaught exception.

## E. Stored-data integrity and value fidelity

The exact T4 reader still validates the header object and v2 schema, NPZ checksum, sample-ID shape/type and header agreement, integer component-offset shape/length/start/order, total-count scalar and final-offset agreement, diagnostic-axis object/name/declaration/array/dimensionality checks, and stored geometry/header shape and value agreement. Corruption is reported through controlled `ValueError`s; discovery omits invalid snapshots while retaining valid siblings.

The accepted T3 unsigned-offset fix remains exactly `np.any(offsets[1:] < offsets[:-1])`, avoiding unsigned subtraction underflow. Existing signed and unsigned valid offsets, repeated offsets/empty component intervals, non-empty diagnostic axes, and dtype/value round trips remain covered by the accepted tests and the passing suite. My independent valid/malformed fixtures additionally confirmed the checksum, identity, geometry and offset metadata were otherwise consistent. Arrays are copied from storage and checked, not recast or recalculated. Nothing in the reader derives classifier outputs, scores, margins, memberships, regions, or relations.

## F. Historical CLI JSON

At exact T4 HEAD, both `town experiment image inspect <run>` and `town inspect-run <run>` emit top-level:

```json
{"evidence_status": "historical_stored_metadata"}
```

The marker is presentation-only. Independent temporary-run CLI checks confirmed:

- Image inspection preserves the exact stored manifest and metrics under their existing nested keys.
- Synthetic inspection preserves stored config, metrics and environment under their existing keys; the returned `path` remains the inspected run path.
- No checkpoint is loaded, no synthetic evaluation/model reconstruction occurs, and no persisted JSON is amended to add the marker.
- The repository has no consumer relying on an exact response-key set; the marker is additive and does not overwrite nested archived fields.

Both CLI calls returned successfully and the complete temporary fixture trees remained byte-identical.

## G. Run and worker-sidecar nonmutation

I independently used an image run with a deliberately stale adjacent `.image-workers/<run>.json` status (`state=running`, nonexistent PID, stale timestamp). Before and after `town experiment image inspect`, the manifest, metrics, resolved config and sidecar bytes were identical; the surrounding fixture tree had the same file set and contents. The CLI did not reconcile the status, transition lifecycle state, publish worker metadata, create an artifact, or acquire a writer lock.

I repeated the check through Image Embeddings AppTest with the same kind of stale sidecar and a malformed snapshot. Manifest/config/metrics/snapshot/sidecar bytes and the fixture tree were unchanged. There was no writer/status publication. `inspect_run()` is a direct manifest read; the dashboard calls it and reads metrics/config directly. `transition_run()`, `reconcile_worker_status()`, `RunWriterLock`, and generic atomic writers remain separate explicit lifecycle/storage utilities; their existence is not an inspection violation.

## H. Dashboard and dataset presentation

- **Image Embeddings:** `dashboard/image_embeddings.py` is metadata-only, displays stored metrics in a `stored_metrics` field under an explicit persisted/historical caption, discovers checksum-validated snapshot records, and offers no launch/train/resume/prediction/score/geometry controls. Invalid snapshots are omitted and a safe no-valid-snapshot message is displayed. A custom AppTest for a missing-axis-schema snapshot had no exception and no mutations.
- **Geometry Laboratory:** `_render_geometry_laboratory()` renders the explicit centralized unavailable-backend message. It does not import `dashboard.geometry`, load a checkpoint, reconstruct a region, compute membership/margins, or publish geometry. The custom AppTest passed with no exception or file changes.
- **Dataset overview/explorer:** the accepted dataset dashboard remains separate and model-free. The focused/full suites cover overview integrity/provenance/support summaries, dataset explorer navigation, known/unknown target states, generation controls, current-integrity gating and image display. `test_dashboard_app.py::test_dashboard_explorer_loads_shared_dataset_and_renders_annotations` passed. The source draws from `component.bbox_xyxy_original`; this is authoritative dataset annotation geometry, not classifier geometry. I independently loaded `examples/pizza-48`, verified `current_pass`, and confirmed `_draw_annotations()` preserves the `(224,224)` source image dimensions and paints the original-coordinate bounding-box corner at `(66,154)` for the selected component.
- Dataset generation remains explicit; help and dashboard controls work without generation being triggered by inspection or startup.

## I. Relation and unavailable-operation boundaries

`diagnose_region_relation()` rejects immediately with `ClassifierBackendUnavailableError`; no region relation, membership, or margin is calculated and no fake result is returned. Snapshot class-ID/name validation remains ontology-backed. `save_relation_diagnostic()` is still a caller-supplied generic atomic/no-clobber JSON publisher; the relation-publication regression suite passed.

I independently invoked representative CLI routes for synthetic training, image training, image matrix execution, checkpoint evaluation, resume, relation diagnosis and geometry export. Each exited 2 with the actionable intentionally-disconnected backend message, no traceback/ImportError, and no output paths. The temporary output root remained empty. Existing API/CLI tests additionally cover synthetic evaluation/resume, image preflight/stop, snapshot generation, workers and no-side-effect behavior. No worker was launched and no status sidecar was published.

## J. Accepted T3 preservation

All eight required historical implementation modules remain absent and unimportable:

- `models/classifiers.py`
- `models/geometry.py`
- `models/image_embeddings.py`
- `models/classifier_transport.py`
- `models/image_model_validation.py`
- `training/image_objectives.py`
- `experiments/synthetic.py`
- `dashboard/geometry.py`

A fresh Python process imported retained dataset/CLI/dashboard/snapshot/relation/point paths and verified the retired modules were not loaded and could not be imported. T3 architecture tests also check forbidden definitions/calls and package exports. Historical formula strings retained in `experiments/image_config.py` are configuration identity metadata, not executable formula implementations. No equivalent classifier implementation or active checkpoint reconstruction was found; no old classifier, membership, signed-margin, objective, synthetic trainer, or relation algorithm was restored.

Dataset generation/validation, ontology class ordering, targets and known masks, image preprocessing, MobileNetV3/FPN/ROI features, image-to-point projections, DeepSets/hybrid composition, generic reducers, feature cache, run identity/metadata, snapshot readers and dataset dashboard remain covered by the passing suite. No future classifier output contract or scientific semantics were added.

The retirement manifest is byte-identical to accepted T3: SHA-256 `b3b283a4b7849cd5dd5d08b9aa9ecf66888f8435954479a6f3a7823a0802ded3` at both T3 and T4. Independent parsing/collection reconciliation confirmed 176 original nodes, 132 historical T2 failures, 96 deleted / 77 replaced / 3 split nodes, 121 replacement references across 40 distinct nodes, and all replacement references resolved. `test_t3_architecture.py` preserves the fixed T3 disposition invariants and accounts for the one T4 snapshot regression separately: 196 accepted T3 checkpoint nodes + 5 final-integrity nodes + 1 T4 node = 202 current nodes. The manifest itself was not edited.

## K. Independent verification evidence

All commands below ran from the exact submitted T4 worktree using the repository Python 3.12 environment at `/home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python` (CPU-only PyTorch).

- Focused selection (exact command from the T4 report) — **80 passed, 0 failed, 0 skipped**, 7 existing Pandas/Streamlit warnings (28.67 s):
  ```bash
  PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q \
    tests/test_image_snapshot_integrity.py tests/test_dashboard_app.py \
    tests/test_classifier_backend_unavailable.py tests/test_phase3_architecture.py \
    tests/test_phase3_config.py tests/test_application_cli_dashboard.py \
    tests/test_phase3_d5b_relation_publication.py \
    tests/test_phase3_runs.py::test_run_config_identity_and_original_subset_order_remain_metadata_only \
    tests/test_phase3_relations.py tests/test_t3_architecture.py
  ```
- Full suite `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q` — **202 passed, 0 failed, 0 skipped**, 7 existing Pandas/Streamlit warnings (56.30 s).
- Fresh `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest --collect-only -q -p no:cacheprovider tests` — 202 current nodes; all T3 replacement references resolve.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests` — passed.
- `git diff --check 9a33ef291057c8b72565eaa28ef32f5e2fc66271..19222424f5ad8347deddc6dfb2d7187bd85609c0` — passed.
- `town --help`, `town generate --help`, `town inspect examples/pizza-48 --json`, and `town validate examples/pizza-48 --no-write-report --json` — passed. Inspection returned `pizza-dcc148789aecf8af86a3`, 48 samples, `current_pass`; validation was valid with zero errors. The canonical validation report remained SHA-256 `cd371bd190796552fdf831f084b365d48af42d363c448f9952fc5a2fafe07925`; no generation was run.
- Dashboard temporary-port smoke — `/_stcore/health` returned `ok`, `/` returned HTTP 200; the process group was stopped after the check.
- AppTests exercised dataset explorer/annotations, generation controls, Image Embeddings historical metadata/corrupt-snapshot handling and Geometry Laboratory unavailable state; the independent custom invalid-v2-snapshot dashboard test also passed.
- Independent custom snapshot fixtures verified missing-axis error, empty schema acceptance, non-empty declarations, valid checksums/geometry/identity/offsets and valid-sibling discovery. Independent CLI and dashboard fixtures verified unchanged run/sidecar contents. Representative unavailable CLI commands exited as expected without output artifacts.

The implementation diff changes no canonical datasets, manifests, validation reports, research artifacts, checkpoints, scientific specification, author-owned mathematical decisions, prior review reports, accepted implementation refs, accepted branches, or `main`. The canonical validation report and T3 retirement-manifest hashes are unchanged. The original checkout remained on its pre-existing T2 branch and clean; review work used a separate worktree.

## L. Defects and limitations

No material T4 defect remains. No change is required.

Limitations: no tracked production snapshot artifacts were present, so historical snapshot compatibility was established from the normative documentation, accepted T2/pre-teardown v2 writer, and explicit-field fixtures rather than a census of archived production files. External CLI consumers outside this repository cannot be surveyed; the marker is an additive JSON key and no in-repository exact-shape dependency exists. Verification was CPU-only; no model training or CUDA behavior is claimed. Seven existing non-failing Streamlit/Pandas deprecation warnings remain. No parallel review subagents were available in this harness; this report records a direct review of the exact fixed revision and does not claim subagent findings.

## M. Issue #7 acceptance-criteria mapping

| Issue #7 criterion | Independent evidence |
|---|---|
| Checksummed snapshots and run metadata load without model/checkpoint execution or historical math | Snapshot reader source and custom valid/malformed fixtures; CLI payload inspection; fresh-process import checks; full snapshot/metadata suite. |
| Stored classifier/geometry values are marked historical and not recomputed/interpreted as future outputs | Both CLI markers verified with exact nested payloads; Image Embeddings caption and `stored_metrics`; no array plotting/model controls; formula strings remain metadata only. |
| Active region prediction, geometry plotting and old relation recomputation are unavailable | Geometry Laboratory AppTest; deleted plotting module; `diagnose_region_relation()` rejection; unavailable CLI routes and no artifacts. |
| Dataset generation, validation, CLI inspection, dashboard startup/overview/explorer remain functional without training | Help/inspect/no-write validation; dashboard health/root smoke; dashboard and application tests; annotation pixel check using `bbox_xyxy_original`. |
| Dataset CLI/dashboard imports do not load retired classifiers or model execution | Fresh Python import-boundary process and T3 architecture test; all eight modules absent/unimportable. |
| Tests cover checksum, value fidelity, disabled geometry and preserved dataset surfaces | Focused selection 80 passed; full suite 202 passed; custom independent checksum/axis/geometry checks. |

## N. Next-stage authorization

**ACCEPT T4 — Historical run and snapshot inspection remains usable without classifier reconstruction or mathematical recomputation. Archived evidence is explicitly identified as historical, snapshot integrity is enforced, and inspection leaves persisted runs and worker status unchanged. Image Embeddings, Geometry Laboratory, relation APIs and model-dependent commands correctly respect unavailable-backend boundaries. Dataset CLI/dashboard functionality remains operational, and accepted T3 classifier-removal and retirement evidence is preserved. No replacement classifier semantics have been approved. T5 may become READY after coordinator verification of all native blockers.**
