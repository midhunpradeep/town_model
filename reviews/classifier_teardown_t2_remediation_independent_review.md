# Independent Re-Review — T2 Objective-Guard Remediation

## A. Verdict

**ACCEPT T2**

The two direct historical image-objective entrypoints now fail closed through the shared typed unavailable-backend mechanism. The guards run before the prohibited computations; the direct-call regressions pass; classifier-independent aggregation remains available. I found no new execution bypass or side-effect regression.

This accepts only the T2 disconnection boundary. It does not approve or reconstruct classifier mathematics. Historical formula source remains behind the guards for T3 removal.

## B. Provenance

- Accepted T0/T1 baseline: `cefbbc7a4334c28bec1d6071d700a26bb4977d21`.
- T2 implementation code/test commit previously submitted: `4072dde7ca73a3d4d705002067577e3e970bc4de`.
- Previously submitted T2 HEAD reviewed in the first review: `c4c6d9e70e0281ce12434f42c6a9b3ef390352cf`.
- First independent review commit (REQUEST CHANGES): `d44b8508ba0b588dd74ae33625227aba69a19900`.
- Objective-guard remediation implementation commit: `a95f2630bc67d4d0445b3fea2ac0038c3115cfac`.
- Exact remediation HEAD reviewed: `b8e1d6c0b292c5e9a5c43a7b833b6db106aa30d9`.
- Branch: `reconstruction/t2-objective-guard-remediation`.
- Review report commit: the documentation-only commit containing this report; its exact SHA is recorded in the issue #5 acceptance comment and review handoff. (A commit cannot embed its own object ID.)

At review start, `origin/reconstruction/t2-objective-guard-remediation` resolved to the exact submitted HEAD above. The current branch/worktree for the first review was clean at `d44b8508…`; the separate remediation worktree `/tmp/town-model-t2-objective-guard-remediation` was clean at `b8e1d6c…`. `d44b8508…` is an ancestor of remediation implementation `a95f263…`, which is an ancestor of submitted HEAD `b8e1d6c…`. No concurrent remote advance was present.

Issue #5 was open with its sole lifecycle label `workflow:in-review`; its recent journal contained the first review, remediation investigation, test checkpoint, and exact-SHA handoff. Issue #6 was open, unassigned, `workflow:defined`, and natively blocked by open issue #5. The remediation diff changes exactly these three files:

- `src/town_model/training/image_objectives.py`
- `tests/test_classifier_backend_unavailable.py`
- `docs/reconstruction/T2_DISCONNECTION_REPORT.md`

The diff does not modify the historical mathematical assertions, datasets, ontology assets, checkpoints, snapshots, accepted branches, or prior independent review reports.

## C. Direct objective guard findings

### `_penalties()`

In `src/town_model/training/image_objectives.py`, `_penalties()` now calls `raise_classifier_backend_unavailable("historical image classifier penalty computation")` as its first executable statement. This precedes family/objective selection, score/target access, tensor operations, and every historical formula.

I independently supplied finite, correctly shaped PyTorch score/target tensors (`scores=[[0.0, -0.25]]`, `targets=[[1.0, 0.0]]`) and called the function for all three historical configurations:

- baseline BCE-with-logits (`family="baseline"`);
- signed-margin BCE (`family="box", geometry_loss="signed_bce"`);
- signed-distance hinge (`family="box", geometry_loss="signed_distance"`).

Each call raised `ClassifierBackendUnavailableError` with the penalty-specific operation and actionable shared text. Inputs remained unchanged. The new parameterized tests exercise the same valid configurations and would fail if the guard were removed.

### `image_objective_terms()`

`image_objective_terms()` has its own independent call to `raise_classifier_backend_unavailable("historical image classifier objective computation")` immediately after its docstring. It is not relying on a later `_penalties()` guard. It precedes result/batch inspection, target packing, score extraction, and tensor calculations.

I constructed and validated a shape-consistent one-image `ImageBatch` and matching `ImageModelResult`: one valid component, one positive class target with a known mask, and supplied zero signed-margin tensors for component and pizza scores. The direct call with `family="box", geometry_loss="signed_distance"` raised the typed objective-specific error before calculation. The earlier review's approximately `0.1` hinge reproduction therefore rejects at the objective entrypoint rather than returning a loss. The tensors remained value-identical, and no artifact appeared.

The new tests check the exception type and operation-specific actionable text. Their objective test would fail if only the objective guard were removed: downstream execution would eventually produce the distinct penalty-operation error instead of the required objective-operation error. The direct penalty tests would fail if the penalty guard were removed and historical tensors were returned.

The old BCE/signed-margin/hinge source is unchanged behind the unconditional guards. No dummy values, approximations, classifier interface, scoring representation, new objective, or mathematical assumption was introduced. `ImageBatch`, `ImageModelResult`, and legacy score transport contracts are unchanged. No unrelated T2 entrypoint source was modified.

## D. Aggregation preservation

`_instance_terms()`, `aggregate_effective_batch()`, `ImageObjectiveTerms`, and `ImageObjectiveResult` are unchanged by the remediation diff. I independently supplied `ImageObjectiveTerms` with component values `[1, 99]` and eligibility `[true, false]`, and pizza values `[2, 6]` with both eligible. I instrumented both `_penalties()` and `image_objective_terms()` to raise if called, then ran `aggregate_effective_batch()` successfully:

| Result | Observed |
|---|---:|
| Component loss | 1.0 |
| Pizza loss | 4.0 |
| Total loss | 5.0 |
| Eligible component images | 1 |
| Eligible pizza images | 2 |
| Undefined tasks | none |

The aggregation regression test also replaces `_penalties()` with a failure sentinel, proving aggregation does not convert scores. Five additional existing utility tests passed independently: heterogeneous effective-batch gradient aggregation, known-mask/positive-weight reduction and gradients, split accumulation equivalence, unit-weight equivalence, and undefined-task handling. Thus known-mask weighting, aggregation, and undefined-task behavior remain usable and unchanged. This infrastructure result is not scientific approval of a future training objective.

## E. Test evidence

Environment: Python 3.12.0, PyTorch 2.5.1+cpu; CUDA unavailable.

- Focused T2 suite:
  `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q tests/test_classifier_backend_unavailable.py`
  — **29 passed, 0 failed**, 2 existing Streamlit/Pandas warnings.
- Exact surviving-infrastructure node selection from the first independent review (using the repository environment by absolute path because the isolated worktree has no `.venv`) — **168 passed, 2 deselected**, 5 existing warnings. The exact selection is reproduced here:

```bash
PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q \
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

  The count is the first review's 163 passed plus the five new successful regression nodes; the command and node selection are unchanged.
- Full suite, exact requested selection:
  `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q`
  — **191 passed, 132 failed, 0 skipped**, 9 warnings. This is a transitional suite, not a green suite.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests` — passed.
- `git diff --check d44b8508ba0b588dd74ae33625227aba69a19900..b8e1d6c0b292c5e9a5c43a7b833b6db106aa30d9` — passed.
- The five classifier-independent objective utility nodes listed in section D — **5 passed**.
- Direct valid-input calls to both guarded entrypoints and independent numerical aggregation instrumentation — passed.

## F. Historical-failure classification

The full suite produced the reported **191 passed / 132 failed**. The first review's 130 historical failures remain: each of the original 20 file groups has the same count as before, except `test_phase3_models.py`, which increased by exactly two. The complete observed per-file comparison is:

| Test file | First review | Current | Delta |
|---|---:|---:|---:|
| `test_classifier_teardown_boundary.py` | 1 | 1 | 0 |
| `test_dashboard_app.py` | 5 | 5 | 0 |
| `test_phase2_artifact_errors.py` | 5 | 5 | 0 |
| `test_phase2_cli.py` | 2 | 2 | 0 |
| `test_phase2_dashboard_geometry.py` | 8 | 8 | 0 |
| `test_phase2_experiments.py` | 3 | 3 | 0 |
| `test_phase2_geometry.py` | 14 | 14 | 0 |
| `test_phase2_gradients.py` | 3 | 3 | 0 |
| `test_phase2_min_width_preflight.py` | 5 | 5 | 0 |
| `test_phase2_models_losses.py` | 6 | 6 | 0 |
| `test_phase2_resume.py` | 6 | 6 | 0 |
| `test_phase3_cache.py` | 1 | 1 | 0 |
| `test_phase3_cli.py` | 1 | 1 | 0 |
| `test_phase3_d5a_ontology_entailments.py` | 1 | 1 | 0 |
| `test_phase3_d5b_relation_publication.py` | 1 | 1 | 0 |
| `test_phase3_models.py` | 29 | 31 | +2 |
| `test_phase3_relations.py` | 5 | 5 | 0 |
| `test_phase3_runs.py` | 15 | 15 | 0 |
| `test_phase3_training.py` | 18 | 18 | 0 |
| `test_phase3_weighting.py` | 1 | 1 | 0 |
| **Total** | **130** | **132** | **+2** |

The two additional failures are exactly the expected, previously passing historical assertions, both now raising the intended typed error:

1. `tests/test_phase3_models.py::test_default_family_objective_penalties_and_signed_distance_hinge_are_numerically_exact` — `_penalties()` rejects with operation `historical image classifier penalty computation`.
2. `tests/test_phase3_models.py::test_hand_computed_known_mask_positive_weight_and_per_image_component_reduction` — `image_objective_terms()` rejects with operation `historical image classifier objective computation`.

The other 130 failures retain the previously reviewed classifications: historical classifier construction/math, synthetic/image train/evaluate/resume/checkpoint execution, geometry/relation/snapshot and dashboard operations, CLI/worker/run success expectations, and the pre-existing stale-run inspection assertion that expects mutation despite `inspect_run()` being read-only. The full failure list contains no new failure file or unexplained failure. The historical mathematical test file was not edited or skipped. The complete suite accounted for 323 outcomes: 191 passed plus 132 failed, with no skipped tests.

## G. Broader T2 regression assessment

The first independent review found the other T2 execution boundaries satisfactory. The remediation diff is limited to the two image-objective guards, their regression tests, and the report; it does not alter the previously reviewed call paths. I reran the full focused T2 module and the surviving selection. They cover or retain the established rejection boundaries for:

- direct historical classifier construction/inference and geometry/relation execution;
- synthetic and image train/evaluate/resume, preflight, checkpoint loading, worker launch/execution, CLI rejection, and dashboard model-operation rejection;
- snapshot generation rejection and read-only historical snapshot/metadata inspection;
- no publication of new artifacts and preservation of existing run/checkpoint/lifecycle files;
- classifier-independent `ImagePointModel`, visual feature extraction/cache, dataset/ontology/target-mask infrastructure, and dataset dashboard surfaces.

No bypass or side-effect regression was found. The earlier lifecycle finding remains unchanged: `inspect_run()` is read-only; `transition_run()` and `reconcile_worker_status()` are explicit generic lifecycle mutators and were not made reachable by this remediation or by rejected classifier operations. No lifecycle rewrite is requested.

## H. Defects and limitations

- **Blocking defects:** none found.
- **Nonblocking limitations:** historical classifier/objective source remains behind unconditional guards pending T3. The full regression suite intentionally remains red for 132 historical expectations; those tests must be formally reconciled in T3 rather than silently deleted or skipped. Verification was CPU-only; no CUDA claim is made.
- **Review process:** this is a direct independent scoped review. The available harness did not provide separate parallel Standards/Spec subagents; no such subagent review is claimed.

## I. Issue #5 acceptance-criteria mapping

1. **No active training/evaluation/resume/preflight/geometry path automatically constructs a concrete historical classifier:** **Pass.** The first review audited those paths; this re-review found the direct objective bypass closed. The remediation does not modify other entrypoints, and the focused T2 tests pass.
2. **Unsupported operations fail with specific actionable errors and no dummy/fallback success:** **Pass.** Direct `_penalties()` and `image_objective_terms()` calls now raise `ClassifierBackendUnavailableError` with distinct operation names and the common actionable reconstruction/acceptance message.
3. **Failure before new artifacts and existing artifact mutation:** **Pass.** Both direct calls reject before computation; caller inputs remain unchanged and no artifact appears. Existing focused T2 no-publication/no-mutation checks pass, and no earlier guard boundary changed.
4. **Dataset CLI/dashboard, point/vision, and validated historical metadata remain available:** **Pass.** The exact surviving-infrastructure selection passed 168 tests with only its two specified deselections; focused retained-path tests also pass.
5. **Tests cover fail-closed operations and retained infrastructure:** **Pass.** Focused suite: 29 passed. New tests cover three valid historical penalty modes, direct valid objective inputs, operation-specific errors, input immutability/artifact absence, and aggregation without score conversion.
6. **No scientific interpretation/math added; independent review before T3:** **Pass.** Only unconditional guards were added to source; historical formulas and transport contracts are unchanged. No replacement mathematics was approved or introduced. This review accepts T2 only.

## J. Next-phase authorization

**ACCEPT T2 — Historical classifier-dependent execution, including direct image-objective scoring, is now disconnected. Unsupported operations fail explicitly before prohibited side effects. Classifier-independent infrastructure remains functional. T3 may become READY after coordinator verification of its native blockers. No replacement classifier mathematics has been approved.**

Issue #5 may move to `workflow:accepted` and close after this report is published. Issue #6 remains unclaimed and unstarted; the coordinator may verify its native blocker/readiness after #5 closes. This review does not start T3.
