# Independent Re-Review — T3 Snapshot Integrity and Test Retirement Remediation

## A. Verdict

**REQUEST CHANGES**

Both findings from the first T3 review are corrected: malformed stored geometry now fails through a controlled reader error, and the previously defective manifest references and non-empty diagnostic-axis coverage are repaired. However, a new malformed-offset case is accepted by the reader, and the strengthened manifest test does not reject coordinated incorrect disposition counts. T3 is therefore not yet acceptable.

No classifier mathematics, replacement output semantics, or scientific requirements are approved by this review.

## B. Provenance

- Accepted T2 ancestor: `c6aad87a45ac75835f02670f32fdeca5167f18f7`.
- Original T3 source/test/manifest commit: `0d95073ec5245acb59ce803f3bbbd466366f74c2`.
- Original submitted T3 HEAD: `8386839e3c258e5f263dd7f07fa3aeaefe731f3d`.
- First independent T3 review (REQUEST CHANGES): `834b2f640fa970f661d0d5070f6928c8adb8695b`.
- Remediation source/test/manifest commit: `97bf379ff035b7bc2c7c1a4789c5474012eff9b6`.
- Exact submitted remediation HEAD reviewed: `5dc3f0937504abfdb61f8d35e669717324bb6d06`.
- Review worktree/branch: `/tmp/town-model-t3-remediation-independent`, `review/classifier-teardown-t3-remediation-independent`, created at the exact submitted HEAD. This report is the only intended change on that branch. Its final commit SHA is recorded in the issue #6 handoff and report URL; a commit cannot contain its own object ID.

At review start, `origin/reconstruction/t3-snapshot-retirement-remediation` matched `5dc3f0937504abfdb61f8d35e669717324bb6d06`. The accepted T2 SHA and original submitted T3 SHA are ancestors of the submitted remediation. The exact remediation diff changes the eight expected files only. The diff does not touch canonical data, checkpoints, accepted review/specification files, or accepted refs.

Live workflow state was verified: #5 is CLOSED `workflow:accepted`; #6 was OPEN `workflow:in-review`; #7 is OPEN `workflow:defined`, unassigned, and natively blocked by #6; #8 is OPEN `workflow:defined` and not activated. Classifier-design map #2 and author-decision issue #3 remain open; neither authorizes replacement mathematics.

## C. Snapshot integrity: original failure reproduction

I independently built a correctly checksummed `image-research-snapshot-v2` fixture with a string sample ID, valid `[0, 0]` component offsets and zero total, a float32 `classifier_geometry_box_low` array, a non-empty `class/coordinate` declaration, and `geometry: {}`.

Against the exact original T3 worktree at `8386839e3c258e5f263dd7f07fa3aeaefe731f3d`:

- `load_image_snapshot()` raised raw `KeyError('box_low')`.
- `discover_image_snapshots()` propagated the same `KeyError`.

Against exact remediation HEAD `5dc3f0937504abfdb61f8d35e669717324bb6d06`:

- The same fixture raises `ValueError` identifying `header.geometry.box_low` and the stored array that requires it.
- Discovery returns no invalid snapshots rather than raising.
- Image Embeddings AppTest has no uncaught exception and displays the no-valid-snapshot message.

This resolves the original snapshot-integrity finding.

## D. Reader, discovery, and dashboard findings

`src/town_model/inspection/image_snapshots.py` now validates the header object, sample ID values/shapes, component offset/count shapes and dtypes, diagnostic-axis declarations, checksum, NPZ readability, and matching stored-geometry metadata. Missing/non-mapping geometry, absent fields, malformed nested values, shape mismatch, and value mismatch produce controlled `ValueError`s. The checks are narrow; no catch-all exception handling was added. Geometry remains optional when no corresponding `classifier_geometry_*` arrays exist.

The new tests execute all three discovery cases: all valid snapshots are returned in sorted order; a valid sibling is retained while an invalid geometry snapshot is omitted; all-invalid candidates yield an empty list. The dashboard safely handles the all-invalid run. A separate valid-run AppTest confirmed stored run manifest/config/metrics and snapshot sample IDs remain visible, without uncaught exceptions or model controls. The dashboard shows an explicit unavailable state when no snapshot passes inspection.

Negative coverage exercises missing/non-mapping geometry, shape/value disagreement, malformed nested geometry, axis dimensionality/type/missing-array errors, checksum mismatch, a non-NPZ container with a correctly recomputed checksum, and malformed sample-ID/offset shapes. The focused and full suites pass these cases.

**Additional integrity defect — unsigned offsets:** at `src/town_model/inspection/image_snapshots.py:79`, monotonicity is checked using `np.diff(offsets) < 0`. For unsigned integer arrays, a decreasing difference wraps to a large positive value. I built a correctly checksummed snapshot with two sample IDs, `component_offsets=np.array([0, 5, 3], dtype=np.uint64)`, and `component_count_total=np.uint64(3)`. `load_image_snapshot()` accepts it, and discovery includes it, even though offsets decrease.

Expected behavior is a field-specific `ValueError` for `arrays.component_offsets`, and discovery should omit it. A minimal correction is to compare adjacent values directly in a dtype-safe manner (for example, `offsets[1:] < offsets[:-1]`) and add an unsigned decreasing-offset regression that has a matching final total. This is a stored-data integrity defect, not a model or geometry calculation.

## E. Diagnostic-axis preservation

The positive fixture in `tests/test_image_snapshot_integrity.py::test_reader_roundtrips_nonempty_diagnostic_axes_and_stored_values_without_reconstruction` is hand-authored and calls the reader directly; it does not instantiate a classifier or generate model outputs. It declares:

- `class` and `coordinate` dimensions for stored low/high geometry arrays;
- `component_occurrence`, `class`, and historical `house` dimensions for `component_activity_frame_margin`.

The test asserts exact axis declarations, sample IDs, component offsets and total, every array's dtype and numeric/Boolean contents, and geometry/header value agreement. The reader enforces declared dimensionality. The reader also accepts a metadata-only snapshot with no geometry arrays and `geometry: null`.

The suite includes negative axis declaration cases for wrong dimensionality, malformed axis entries, and declarations for absent arrays. The exact original missing-`geometry.box_low` fixture also carries a non-empty valid axis declaration. Thus the prior missing independent non-empty-axis evidence is restored without classifier reconstruction.

## F. Retirement-manifest reconciliation

I independently collected tests at both exact revisions:

- Accepted T2 `c6aad87a45ac75835f02670f32fdeca5167f18f7`: **323 nodes**.
- Submitted remediation `5dc3f0937504abfdb61f8d35e669717324bb6d06`: **196 nodes**.

The manifest accounts for **176 unique original IDs**; every ID exists in the accepted-T2 collection. It includes all **132 unique baseline failure IDs**, each present in that collection. The accepted T2 review records the full-suite result of 191 passed / 132 failed / 0 skipped; I relied on that accepted review's full-suite evidence as permitted and independently checked the IDs against the baseline collection.

The independently derived dispositions are:

| Disposition | Original nodes | Groups |
|---|---:|---:|
| Deleted | 96 | 68 |
| Replaced | 77 | 58 |
| Split | 3 | 3 |
| **Total** | **176** | **129** |

Collection reconciliation is exact: 147 common original node IDs remain unchanged and unaccounted; 6 accounted IDs retain their names with retargeted/split assertions; 43 nodes are new relative to T2. Thus `147 + 6 + 43 = 196`.

All **121 replacement references across 40 distinct collected nodes** resolve exactly. No reference is a bare parameterized function name; no per-entry duplicate reference was found; all replaced/split entries have references; deleted entries have none. References may be shared across entries where one independent test covers multiple retired assertions.

### Manifest test limitation

`tests/test_t3_architecture.py` derives disposition counts and compares them to `manifest["disposition_node_counts"]` and `manifest["disposition_group_counts"]` (lines 177–178), but does not assert the expected accepted values (96/77/3 nodes and 68/58/3 groups). If an entry's disposition and both recorded count objects are changed consistently, this check can still pass. The manifest's current counts are correct by my independent reconciliation, but the strengthened test does not meet the requested requirement to reject incorrect disposition counts. Add fixed expected counts (or an independent immutable expected inventory) so the test detects that mutation.

## G. Previously defective replacement references

The known mappings are corrected:

- **Snapshot-axis round-trip:** `test_snapshot_diagnostic_axes_roundtrip_preserve_class_geometry_and_instance_activity` now references the hand-built non-empty-axis reader test plus integrity cases. The manifest explicitly separates retained stored-data integrity from retired model-generated geometry, margin, activity-trajectory, and relation assertions.
- **Checksum reader:** the reference now uses the exact collected node `tests/test_classifier_backend_unavailable.py::test_checksum_validated_snapshot_values_remain_readable_without_model_import`.
- **Parameterized point projection:** references include exact collected IDs for dimensions `[2]`, `[5]`, `[8]`, and `[16]`.
- **Metadata inspection:** read-only claims cite `test_image_and_synthetic_metadata_inspection_are_read_only`, which actually invokes inspection, checks returned persisted metadata, and verifies the run tree is unchanged.
- **Rejected operations/no side effects:** claims cite direct unavailable-operation and CLI tests that check errors, output absence, and existing-run nonmutation, rather than unrelated point/vision tests.

I reviewed the remaining replacement groups by their collected IDs and asserted behavior. They plausibly cover the retained point, dataset/ontology, value-reduction, cache, metadata, checksum-reader, and rejection contracts. Partial replacements say which historical assertions were retired; no old classifier math has been recreated. The current dashboard implementation also displays stored run/snapshot JSON; a separate AppTest confirmed its values, while dashboard presentation redesign remains outside T3.

## H. Preserved infrastructure

The 196-node suite passes, including retained dataset/ontology and target/mask tests; canonical class-ID and known/unknown behavior; MobileNetV3/FPN, preprocessing, ROI and ROIAlign; point projection/composition; generic weighted reductions and effective-batch aggregation; feature-cache identity/integrity; historical run/config identity; and snapshot array/checksum reading. Both mixed-purpose ontology boundaries remain executable:

- `test_ontology_metadata_distinguishes_asserted_entailed_and_unknown_relations` still checks asserted, entailed, unknown, and snapshot-class identity metadata.
- `test_mushroom_vegetarian_ontology_entailment_is_independent_of_geometry` still checks the MushroomTopping → VegetarianTopping entailment without geometry.

Historical geometric relation calculations remain absent.

## I. Architectural regression assessment

All eight required historical implementation modules remain absent. Source/import scans and the passing architecture tests find no old classifier classes/factory, box/town geometry algorithms, membership/margins, score-to-loss conversion, checkpoint reconstruction, or synthetic training fallback. `ImagePointModel` remains classifier-free. Supported image/synthetic/geometry/relation operations still fail explicitly before publication; no fallback or scientific semantics was added.

## J. Verification

Environment: repository Python 3.12 virtual environment, CPU PyTorch.

- Full suite: `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q` — **196 passed, 0 failed, 0 skipped**, 7 existing Streamlit/Pandas warnings (52.48 s).
- Exact focused selection from the remediation addendum — **51 passed, 0 failed, 0 skipped**, 7 existing warnings (16.50 s).
- Current collection: **196**; accepted-T2 collection: **323**.
- `compileall -q src/town_model tests` — passed.
- `git diff --check 8386839e3c258e5f263dd7f07fa3aeaefe731f3d..5dc3f0937504abfdb61f8d35e669717324bb6d06` — passed.
- Dataset inspect and validate both exited 0; validation reported valid with 0 errors. Canonical validation report SHA remained `cd371bd190796552fdf831f084b365d48af42d363c448f9952fc5a2fafe07925`.
- Metadata-only image-run inspection exited 0 and returned stored metric `0.25`.
- Image experiment run, synthetic mapping, image relation, and geometry export each exited 2 with the unavailable-backend message and created no output artifacts.
- Dashboard server health returned HTTP 200/body `ok`; root returned HTTP 200. Dataset explorer AppTest passed in the full suite. Invalid-snapshot dashboard AppTest had no uncaught exception.
- The eight removed modules are absent; focused source/import scans found no historical executable algorithms or checkpoint path.

The accepted-T2 full-suite result was not rerun; the accepted independent T2 report supplies the documented 191/132 baseline. All 132 failure IDs were checked against the actual T2 collection and manifest.

## K. Defects and limitations

1. **Blocking, medium — unsigned decreasing component offsets pass integrity validation.** Reproduction and minimal correction are in section D.
2. **Required test-strength correction — disposition counts are not anchored to expected values.** Current manifest totals reconcile, but the architecture test would accept counts changed consistently with dispositions. Exact location and correction are in section F.

No defect remains in either original review finding. Discovery intentionally omits corrupt snapshots; invalid paths are not individually listed. Verification was CPU-only. No classifier mathematics or model reconstruction was attempted.

## L. Issue #6 acceptance-criteria mapping

| Criterion | Result |
|---|---|
| Old classifier/math definitions and runtime fallback remain absent | **Pass** — module, source/import, and full-suite checks. |
| Obsolete assertions are explicitly accounted without adapting historical math | **Pass with test-strength correction requested** — 176 IDs, 132 failures, and all replacement nodes reconcile; the architecture test must anchor disposition totals. |
| Dataset/ontology, vision/ROI, point, reducers, identity/storage, validation, and snapshot infrastructure remain functional | **Fail pending correction** — normal and specified malformed cases pass, but a valid-checksum unsigned decreasing-offset snapshot is accepted. |
| No predictions/scientific interpretation are fabricated; model operations reject honestly | **Pass** — source audit, CLI exit statuses, no-output checks. |
| No canonical datasets/checkpoints/specification/accepted history changed | **Pass** — exact diff limited to the eight expected report/remediation files. |
| Independent review before T4 | **Pending** — this review requests changes; #7 remains blocked. |

## M. Next-phase authorization

**T3 is not accepted.** Return #6 to implementation for the unsigned-offset integrity fix and regression, and anchor the manifest disposition counts in the architecture test. Keep #7 unclaimed, unstarted, and natively blocked by #6. Do not begin T4 or T5. Issues #2/#3 remain author-owned and unapproved; no replacement classifier mathematics is authorized.

**T3 NOT ACCEPTED — T4 REMAINS BLOCKED**
