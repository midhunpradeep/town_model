# Independent Review — Classifier Teardown T0/T1

## A. Verdict

**ACCEPT T0/T1**

This accepts the classifier-independent separation and compatibility-preserving refactor only. It does **not** validate the scientific desirability of the historical classifiers, approve their mathematics for reconstruction, or approve a replacement model.

## B. Reviewed revisions and provenance

- Baseline: `95438f2356f8415506dba93a45cb2908ff5cf875`
- Dependency map: `2e88d7ed8b9b6830a62a7861cefe1ed79067bc8d`
- Implementation reviewed: `fbed1bc008127f7024aeabd1641519e2a333a840`
- Accepted historical implementation: `2de9c5f29b436f727720e87132d0b4be25a40768`

The actual remote `reconstruction/classifier-teardown` head matched the implementation SHA at review start; the accepted historical remote ref matched `2de9c5f…`. The dependency-map commit is a child of the baseline and the implementation is a child of the map commit. The accepted historical revision is an ancestor of the baseline. Review tests ran from detached worktrees at the exact baseline and implementation SHAs; no implementation source or test was edited.

## C. Findings

### Dependency-map quality

`docs/reconstruction/CLASSIFIER_TEARDOWN_MAP.md` accurately documents the source split and active historical dependencies. It covers the old classifier definitions/factory; trainable box/town parameterizations, transforms, initialization, membership and margins; synthetic training; image objectives/training/evaluation; run preflight/lifecycle and identity; checkpoint load/resume; snapshot writing/reading; geometry plots; relation diagnostics; CLI/dashboard entrypoints; package exports; and related tests. It distinguishes data/vision/point/transport/storage from old classifier semantics, calls out indirect/lazy imports and private helpers, and identifies silent-fallback risks.

The `image_model_validation.py` helper and its function-local `geometry.validate_min_width` import are identified under the configuration/validation boundary. T2/T3 must retire or mark the historical model fields/validation when disconnecting execution/removing geometry. This is an explicit future boundary, not an import-isolation defect: importing the specified modules alone does not import geometry. The map does not prescribe replacement mathematics. It explicitly marks `ClassifierOutput` as legacy and its logits/margins as non-mandatory historical fields.

### Classifier-free image-to-point contract

`src/town_model/models/image_points.py::ImagePointModel` accepts `VisionFeatureBatch` plus caller-supplied sample IDs; construction takes dimensions, composition mode and seed, not a class vocabulary. It constructs no classifier and performs no class scoring. `ImagePointResult` in `src/town_model/contracts.py` contains sample/occurrence identities, validity mask, component/global/composed/final point tensors, composition mode and optional vision diagnostics only. It has no predictions, logits, margins, memberships, classifier outputs or losses.

Independent executions covered global, compositional and hybrid modes; a padded `[B,K]` batch; a wholly invalid padded component set; and `K=0`. Component IDs and validity masks were preserved. Component outputs had the expected `[B,K,d]` shape; global, composed and final pizza points had `[B,d]` shape, including the empty-component case. The selected final point matched the selected mode. The point-only model's state keys and tensors equal the compatibility model's shared point-module state; it has no classifier parameter names or placeholder classifier parameters. The mask continues to gate DeepSets aggregation as before.

### Legacy classifier compatibility and parameter/state parity

In deterministic CPU executions, baseline, box and town initial states matched exactly against the baseline revision: 28, 28 and 30 state-dictionary tensors respectively (**86 total**), with identical keys, parameter names/shapes, trainability and values. The established parameter counts were unchanged. Seeded construction left the global RNG state identical across revisions. Canonical dataset class IDs/names and image config identity hashes matched.

For all 3 classifier families × all 3 pizza modes, I compared point tensors and legacy classifier outputs on identical inputs. Hard predictions, baseline logits, box/town signed margins, fixed-positive outputs, and requested diagnostic tensors matched exactly. The point outputs also matched exactly with padded and empty component inputs. Six representative invalid model-config cases returned identical validation error types and messages.

I independently reproduced a real-image town/hybrid one-epoch run using `examples/pizza-48` (four train and two validation samples, frozen pretrained MobileNetV3/FPN, CPU, fixed seed and two optimizer steps). The full baseline and candidate checkpoint files had the same SHA-256 (`1579456b…778f04`). Their loaded payloads also compared exactly across 446 tensor/array leaves, including model/visual/optimizer/scheduler/RNG state, identity, metrics/history and class IDs. The baseline checkpoint loaded strictly into candidate visual/model modules with no missing or unexpected keys. The configuration identity hash was identical (`429e0c80…7deba7`).

I also wrote/loaded baseline and candidate snapshots from those equivalent states: all 42 numeric arrays and normalized headers matched, including 16 diagnostic-axis declarations and the source-checkpoint hash. Only worktree-specific absolute `dataset_root` and `source_checkpoint` path strings were excluded from header equality. This verifies saved diagnostic axes and snapshot transport; it does not endorse the stored historical interpretations.

### Import isolation and package architecture

In separate fresh Python processes, each of these imports left both `town_model.models.classifiers` and `town_model.models.geometry` absent from `sys.modules`:

- `town_model.models.image_points`
- `town_model.models.classifier_transport`
- `town_model.experiments.image_config`
- `town_model.cli`
- `town_model.dashboard.app`
- `town_model.inspection.image_snapshots`

`town_model.models` and its point/transport exports are lazy; explicitly importing/constructing a historical `BoxClassifier` still works in T1. A fresh process constructed and executed `ImagePointModel` without class vocabulary and without importing classifiers or geometry. Dataset CLI inspection did not import classifiers, geometry, synthetic experiments, image runs or the image trainer. Imports deferred inside snapshot/relation functions are limited to the classifier-specific operations that use them.

### Configuration, persistence, dataset/vision/dashboard preservation

The nine family/composition config identities and canonical class ordering matched. Strict checkpoint state loading and real one-epoch training/checkpoint parity passed. Snapshot arrays, saved axes and checksums matched as noted above.

`PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town --help` succeeded. `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town inspect examples/pizza-48 --json` succeeded and reported dataset `pizza-dcc148789aecf8af86a3`, 48 samples, `current_pass`. The dataset inspect operation did not import classifier/model-training modules. Streamlit AppTest ran the default dataset page without exceptions or classifier/geometry/synthetic/image-run/trainer imports. The actual dashboard server returned HTTP 200 / `ok` from `/_stcore/health` and was then stopped. The full suite also exercises preserved dataset, vision, point, dashboard, historical snapshot and model paths. No canonical dataset or repository research artifact was changed.

### T2–T5 blocker graph and decision authority

The native GitHub dependency graph is correct:

- #5 (T2) is blocked by #4 (T0/T1).
- #6 (T3) is blocked by #5.
- #7 (T4) is blocked by #6.
- #8 (T5) is blocked by each of #4, #5, #6 and #7.

Issues #5–#8 were open, unassigned and `workflow:defined`; none was started or made READY. Their scopes do not authorize replacement classifier implementations. Issues #2/#3 remain planning/author-decision work; #3 is pending and is not an implementation approval. Bootstrap issue #1 remains open `workflow:in-review`, assigned to its implementer; its independent acceptance is still pending and is not included in this verdict. No T2 work was started.

## D. Independent test evidence

All commands below ran in a detached candidate worktree at `fbed1bc…`, using the repository's installed Python 3.12 virtual environment by absolute path because that worktree has no local `.venv` directory. The baseline comparisons ran in a separate detached worktree at `95438f…`. PyTorch execution was CPU-only; deterministic algorithms were enabled for the independent parity runs.

- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q` — **294 passed, 0 failed, 0 skipped**, 11 existing Streamlit/Pandas `Pandas4Warning` deprecations, 162.40s.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m pytest -q tests/test_classifier_teardown_boundary.py tests/test_phase3_architecture.py tests/test_phase3_models.py tests/test_phase3_vision.py tests/test_phase3_dataset.py` — **55 passed**, 13.38s.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/python -m compileall -q src/town_model tests` — passed.
- `git diff --check 95438f2356f8415506dba93a45cb2908ff5cf875..fbed1bc008127f7024aeabd1641519e2a333a840` — passed.
- `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town --help` — exit 0. `PYTHONPATH=src /home/midhun/Projects/ovgu/sem8/thesis/town_model/.venv/bin/town inspect examples/pizza-48 --json` — exit 0; current-pass, 48-sample dataset as above.
- Six separate-process import-isolation checks — passed for all six modules listed above; lazy package exports and explicit historical classifier construction — passed.
- Dataset-only CLI import check — passed; no classifier, geometry, synthetic, image-run or image-trainer imports.
- Dashboard AppTest default dataset page — passed without model imports; live dashboard health check — HTTP 200 / `ok`.
- Independent seeded parity harness — exact across all three families/modes, including predictions/logits/margins/fixed outputs/diagnostics, masks/IDs, empty and padded components, config identities and 86 initial model-state tensors.
- Real-image town/hybrid one-epoch baseline/candidate comparison — exact checkpoint SHA and exact payload values; strict cross-revision state loading passed.
- Snapshot baseline/candidate comparison — all 42 arrays and normalized header matched, including diagnostic axes and checkpoint checksum.

No CUDA test was performed or is claimed. No new data/checkpoint was written inside the repository; comparative artifacts were written only under `/tmp`.

## E. Defects

**No material defects found.** No implementation-source or test correction is required for T0/T1. Historical classifiers and their training/objective/geometry paths remain active intentionally during T1; this review does not approve retaining them after their scheduled T2–T4 passes.

## F. Issue #4 acceptance criteria mapping

1. **Dependency map covers requested modules, callers/import edges, public interfaces, tests, dispositions and fallback risks:** demonstrated by review of `docs/reconstruction/CLASSIFIER_TEARDOWN_MAP.md` and source/import search; accepted.
2. **Classifier-independent point path and dataset-authoritative vocabulary:** independently constructed/executed `ImagePointModel` without class names or classifier/geometry imports; identities, masks and all point outputs verified; accepted.
3. **Existing points, model behavior, training behavior and run/checkpoint identities unchanged:** 86 model-state tensors, all nine family/mode outputs, config hashes, one-epoch real-image checkpoint and snapshot arrays/axes compared exactly; accepted.
4. **No new math, generated data or artifact modifications; Phase 4 remains on hold:** diff is structural/import/transport separation; no replacement semantics or repository artifacts introduced; accepted.
5. **T2–T5 bounded, falsifiable, separate from author math map, correct native edges:** issues #5–#8 and native graph inspected; all are defined/unassigned and contain no replacement-model authorization; accepted.
6. **Required tests, imports, compile/diff, CLI/dashboard checks run and actual results recorded:** evidence and counts are in section D; accepted.
7. **Fixed SHA submitted for independent review, no self-acceptance by implementer:** remote branch matched exact implementation SHA at start; this report is the independent review record. Issue #4 is transitioned only after this review; accepted.

## G. Next-phase authorization

**ACCEPT T0/T1 — The classifier-independent image-to-point boundary has been verified, historical behavior is preserved, and the teardown dependency map is sufficient. T2 may be made READY after the coordinator verifies its blocker state. No replacement classifier design has been approved.**
