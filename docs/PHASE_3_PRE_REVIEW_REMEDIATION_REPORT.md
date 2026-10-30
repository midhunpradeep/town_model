# Phase 3 pre-review remediation report

**Disposition: READY FOR INDEPENDENT REVIEW** of the documented pre-detector Phase 3 milestone. This is a remediation handoff, not independent acceptance; it does not authorize or claim Phase 4.

## Revisions and scope

- Repository: `midhunpradeep/town_model`
- Working branch: `remediation/phase-3-pre-review-findings`
- Exact initial implementation revision: `241cac3619022d9ae7707e1bc18667c872359527`
- Final remediation implementation revision (all source, test, configuration, and architecture changes): `3c2af2fc154a1823247c8bb5daa58620e018ba1d`
- The initial implementation revision was fetched from `origin`, verified to be the current implementation-branch HEAD and an ancestor, and preserved. No historical reports, accepted Phase 2 revisions, or `main` content were overwritten.
- Scientific authority remains `pizzaiolo_thesis_implementation_spec_v1_1_expanded.md`; it was not edited. All learning runs described here use the canonical prepared dataset's **train and validation partitions only**. No test-partition outcomes were inspected or used for selection.

The final implementation commit is followed by a documentation-only commit containing this report and the implementation-report addendum. The exact branch HEAD is supplied in the final handoff; the implementation revision above identifies the code and tests being reviewed.

## Finding status matrix

| Finding | Status | Summary |
|---|---|---|
| F1 — positive weighting and reduction | **Confirmed and corrected** | Both pizza and component instance reductions now divide by the sum of known class weights. Masking, image/component aggregation, effective-batch aggregation, and gradients are tested. |
| F2 — geometric objective defaults | **Confirmed and corrected** | Baseline uses BCE-with-logits; box/town default to signed-distance hinge with margin 0.1. Signed-margin BCE remains explicitly selectable. Objective identity and semantics now affect run/checkpoint identity. |
| F3 — dashboard lifecycle controls | **Confirmed and corrected** | Added validated JSON draft/preflight, detached local CLI launch, status/progress/log inspection, graceful stop, and identity-checked resume through the existing run service. |
| F4 — global ROI letterbox coordinates | **Confirmed and corrected** | Global ROI now maps the full original-image extent using the canonical component-box transform. Cache/checkpoint feature-contract identities were versioned to prevent stale reuse. |
| F5 — pizza-level learning evidence | **Evidence gap confirmed; investigated, partly resolved, scientifically mixed** | No label/mask/vocabulary or pizza-loss construction defect was found. Corrected experiments demonstrate both tasks learning for baseline and town. Box has active gradients and falling hinge loss but no sustained hard-membership F1 learning in the modest single-seed run; that limitation remains unresolved, not hidden behind a success claim. |

A directly related additional defect was also found and corrected (**medium severity**): box/town component diagnostic tensors were reshaped as if every diagnostic described per-point outputs. Box class-corner diagnostics are class parameters, while town activity diagnostics are point-wise. Reproduction: a box snapshot with batch 4, padded K=18, d=8, and two classes tried to reshape `box_low` (2×8 = 16 values) into `[4,18,8]`, aborting at the first configured snapshot and marking the run failed despite a safe checkpoint. Shape-aware handling preserves corners and reshapes only town activity tensors; regression tests cover both shapes.

## F1 — positive-class weighting and reduction

### Contract and correction

The Phase 2 shared reducer in `src/town_model/models/losses.py::reduce_supervision()` already uses the required sum-of-known-weights denominator; the Phase 3 image reducer had drifted from that contract. The old image reducer in `src/town_model/training/image_objectives.py::_instance_terms()` computed:

```text
sum(known_mask × class_weight × penalty) / count(known_mask)
```

The agreed contract is:

```text
sum(known_mask × class_weight × penalty)
/ sum(known_mask × class_weight)
```

Known negatives retain weight 1; only known positives use their configured class weight; unknown labels contribute neither numerator nor denominator. Both pizza and component objectives call the same `_instance_terms()` reducer. Component instances are still averaged within each eligible image, then eligible images are averaged over the complete effective batch. Pizza instances are averaged over eligible images. No microbatch-local denominator was introduced.

### Independent numerical reproduction

A direct penalty-vector test uses:

```text
penalties      = [2, 10, 4]
targets        = [1,  0, 0]
known_mask     = [1,  1, 0]
positive weight= [9,  1, 100]
```

The corrected result is `(9×2 + 1×10)/(9+1) = 2.8`; the former implementation returned `(9×2 + 10)/2 = 14`, a five-fold scale difference. The corrected gradients with respect to the three penalties are `[0.9, 0.1, 0]`. A second test changes the unknown target and changes its penalty from 4 to 1,000,000; neither scalar objective nor any gradient changes. With all weights equal to one, the corrected result exactly equals the ordinary known-label mean.

The updated suite also compares weighted full-batch and heterogeneous microbatch gradients and verifies the existing component-per-image and pizza aggregation behavior. The old hand-computed image-objective expectations were changed to the contract, not vice versa.

### Historical experiment impact and corrected reruns

Every previous experiment with a non-`none` positive-weight policy is affected, including the reported focused overfit (`inverse_frequency`), trainable-FPN (`inverse_frequency`), and broad all-class (`inverse_sqrt`) runs. Their historical metrics/checkpoints were not rewritten and must not be presented as measurements under the corrected reduction. All three configs were rerun on fresh run identities after the correction; they remain explicitly exploratory:

| Corrected rerun | Support / configuration | Corrected result |
|---|---|---|
| `image-267347881b9be9f8d292` | Baseline/global; 16 train / 32 validation; classes 0/2; validation has 2 pizza positives and 36 component positives; 20 epochs / 80 steps | Best monitor 0.0969 at epoch 2 (pizza F1 0, component F1 0.1937); final train and validation F1 were both 0/0. Train objective fell from 1.1102 (epoch 1) to 0.7104 (epoch 20); validation objective was 1.1110 to 0.7132. |
| `image-4e97c13f218c97de55e1` | Baseline/global with trainable FPN; 32/32; same selected classes; 4 epochs / 32 steps | Train and validation F1 remained 0/0. Validation objective fell from 0.9276 to 0.8414. First-step FPN gradient L2 was 4.178; final visual-state SHA differs from initialization (`fbb917…` → `0a45aef…`). Active visual gradients did not produce predictive learning in this pilot. |
| `image-39a48dbdb33d6cb1d370` | Baseline/hybrid, all 60 trainable classes; 128/48; 9 epochs / 72 steps | Best validation pizza/component F1 0.1803/0.0751 (monitor 0.1277, epoch 1); final 0.0581/0.1115 (monitor 0.0848). Final validation loss was 1.0390. Validation support: 328 known pizza positives / 2,552 negatives; 3,725 component positives / 44,035 negatives. |

For comparison, the historical report had quoted 0.958/0 pizza validation/component results in the focused run, 0.930/0 in the trainable-FPN run, and a broad-run best of about 0.197/0.098. Those values remain historical and are **not** corrected-objective results. The new runs do not support a claim of broad comparative learning.

## F2 — objective defaults and comparison semantics

### Correction and provenance

`ObjectiveConfig.geometry_loss` now defaults to `signed_distance`. The accepted Phase 2 geometric penalty path also only accepted `signed_distance`; the image default had diverged. The effective family objectives are:

| Family | Effective default |
|---|---|
| Baseline | `binary_cross_entropy_with_logits` for component and pizza supervision |
| Box | `signed_distance_hinge`, `max(0, margin − (2y−1) × signed_margin)`, margin `0.1` |
| Town | `signed_distance_hinge`, same formula and margin |

A zero hinge margin remains valid. Geometric `signed_bce` remains explicitly selectable and has its own name/formula/temperature. Baseline always uses ordinary BCE even though the resolved config retains a geometric-mode setting. The smoke/overfit/exploratory example configs now resolve to `signed_distance`.

The effective objective identity—including family, formula, parameters, weighting policy/cap, coefficients, and reducer semantics—is persisted in resolved config, run manifest, checkpoint config, metrics, and progress. Changing loss mode or margin changes scientific identity. `IMAGE_OBJECTIVE_REDUCTION` and `IMAGE_FEATURE_EXTRACTION_CONTRACT` are also identity fields; strict checkpoint loading rejects legacy identities rather than silently migrating old training state. Historical configs/checkpoints remain untouched and inspectable; there is no implicit rewrite or migration.

The numerical regression test checks signed margins `[0.02, 0.05]`, targets `[1,0]`, and margin `0.1`, yielding hinge penalties `[0.08,0.15]` and score gradients `[-1,+1]`. It separately checks margin zero and the baseline’s zero-logit BCE value `0.693147` per label. Real-image training/checkpoint tests verify the actual family objective recorded for all nine family×composition smoke runs. Scalar BCE and hinge losses are not compared as equivalent measures.

## F3 — dashboard run controls and subprocess lifecycle

The Image Embeddings page remains read-only for saved model/snapshot inspection, but it now also provides:

- JSON config-file loading and editable configuration draft;
- preflight against the shared config/experiment resolver before output creation;
- local launch through `launch_image_experiment()` and the existing `town experiment image run` CLI path;
- state, optimizer-step progress, objective identity, and captured worker logs;
- graceful-stop requests through the existing run lifecycle API;
- resume only for permitted states with a latest checkpoint and matching current identity.

`src/town_model/experiments/image_worker.py` is a detached wrapper around that same CLI entry path. The service uses `Popen(start_new_session=True)`, persistent status/log files, an atomic run-ID reservation, and the existing single-writer lock. It does not retain a process object in Streamlit session state. Each optimizer step updates `progress.json` and emits a prefixed progress line to the captured stderr/log stream; CLI stdout remains machine-readable JSON. Failure exit codes and captured CLI errors are surfaced in the dashboard.

Verification includes valid and invalid UI launch requests, duplicate launch rejection, stop-state rejection, compatible/incompatible resume, actual detached worker completion, graceful stop and resumed completion, nonzero subprocess status, saved-run inspection with a `torch.load` failure sentinel, and Streamlit rerun while the worker proceeds. The dashboard module is statically checked for direct trainer imports/loops or process ownership. AppTest is not described as a browser test.

A real headless **Google Chrome 134.0.6998.88** test also loaded the Streamlit app, selected Image Embeddings through the browser UI, verified the editable config textarea and launch/lifecycle controls, and ran a successful browser-triggered preflight. The real-browser check did not launch training; actual subprocess launch/stop/resume was exercised separately through the shared service/CLI tests. No multi-user deployment or distributed scheduler is claimed.

## F4 — global ROI coordinates with letterboxing

The former global ROI was the processed canvas `[0,0,processed_width,processed_height]`. For a 400×200 original image at 224 resolution, preprocessing resizes content to 224×112 and pads 56 pixels above and below. The correct transformed full-original extent is `[0,56,224,168]`; the old ROI was `[0,0,224,224]` and sampled the padded canvas too. The corrected code creates `[0,0,original_width,original_height]` and passes it through the same `transform_boxes_xyxy()` function used for components.

The test records the actual ROIAlign coordinates, computes and compares sampled spatial tensors, confirms the padded-canvas result differs, and checks an unpadded 224×224 input remains `[0,0,224,224]`. Empty-component global extraction and the required `[B,256,7,7]` output remain covered. The selected interpretation is the transformed full **original-content** extent—not the full padded canvas—because that is the same authoritative coordinate mapping used for component boxes; letterbox padding remains preprocessing context rather than part of the pizza ROI. `original_extent_roi_letterbox_padding_v2` was added to frozen-cache and checkpoint identity so old cached canvas features cannot be silently reused. Canonical Pizzaïolo images in the experiments are square, so their processed ROI coordinates are unchanged; the correction is directly demonstrated on the non-square case.

## F5 — learning evidence, task support, and remaining uncertainty

### Label, loss, and gradient investigation

The selected class IDs map to canonical target columns before both loss and metrics; the prepared dataset manifests and class names agree for noncontiguous selected IDs. Pizza targets/known masks are read from pizza records, component targets/masks from component records. Unknowns remain masked. Pizza penalties use pizza-level scores/targets and average eligible pizza instances; component penalties use component-level scores/targets and preserve within-image averaging. The shared classifier object is invoked for both point types. No vocabulary/mask selection or pizza-loss construction bug was found in this investigation.

The old focused report had only 2 known pizza positives in validation (component support was 36 positives). A zero pizza F1 at that support is not alone evidence of a bug. The broader historical run had many positive labels in aggregate but sparse class-wise support and modest performance. The corrected controlled experiment below provides higher support for both supervised levels and checks gradient flow from pizza-only objectives through all modes/families.

Pizza-only gradient tests force nonzero penalties and verify:

- global mode → global point projection + shared classifier;
- compositional mode → DeepSets + component projection + shared classifier;
- hybrid mode → global and component/DeepSets branches, fusion, and shared classifier.

These are tested for baseline, box, and town. The actual hybrid runs also record nonzero gradients for all expected trainable branches. Frozen FPN status is correctly reported as frozen.

### Controlled canonical-data learning demonstration

The reproducible base config is `configs/phase3_image_remediation_learning.json`; family is the only changed model factor. This is a **single-seed exploratory GT-ROI** demonstration, not a detector-pretrained stage-2 run or a comparison-quality study.

- Dataset ID `pizza-898a12a57e50a7a0d690`; manifest SHA-256 `580070e2deb301e28757bf394ecbeb31adfeae5c3eb7b41807750be75644d24d`; class-manifest SHA-256 `e803479520664a5de823848c41a29f45a70f6d2a755049a9be14b52312cddd54`.
- Train/validation subsets: 128/48; subset IDs `c19c4379f9071a069d4a9bcf4f5a2f26aeb23f78047175cce8ef9b9dc4bb7e7d` / `f36b1ad7304a4925400793c948d3d27c64e304cbf24586f041192ae4682ae80d`.
- Seed 42; selected ontology IDs 2 `AnchovyTopping` and 29 `NonVegetarianPizza`; hybrid mode; latent dimension 8; frozen ImageNet MobileNetV3-Large/FPN; `inverse_sqrt` positive weighting, cap 10; component/pizza coefficients 1; effective batch 16; maximum 12 epochs.
- The deterministic train subset has 75 known-positive / 53 known-negative pizza labels for `NonVegetarianPizza`, and 140 positive / 2,015 negative component labels for `AnchovyTopping`. Validation has 28/20 and 53/743 respectively. The cross-level classes are correctly known-negative only: zero pizza positives for Anchovy and zero component positives for NonVegetarianPizza. Unknown count is zero for selected targets. Macro-F1 excludes the zero-positive class at each level with the specified reason; it is not scored as a zero F1.
- Actual train-derived positive weights were component `[3.793791921397, 1.0]` and pizza `[1.0, 1.0]`; the run config and manifest now retain these vectors.

| Family | Initial validation loss (component / pizza / total) | Initial validation macro-F1 (pizza / component) | Final validation loss (component / pizza / total) | Final validation macro-F1 (pizza / component) | Steps / best monitor |
|---|---:|---:|---:|---:|---:|
| Baseline (BCE) | 0.7438 / 0.7770 / 1.5208 | 0.2353 / 0.1249 | 0.3074 / 0.3731 / 0.6805 | 1.0000 / 0.9709 | 96 / 0.9854 |
| Box (hinge) | 0.7466 / 0.5966 / 1.3432 | 0.7368 / 0.1249 | 0.0248 / 0.0573 / 0.0821 | 0.0000 / 0.0000 | 72 / 0.3684 |
| Town (hinge) | 0.0289 / 0.1934 / 0.2223 | 0.0000 / 0.0000 | 0.0036 / 0.0187 / 0.0223 | 1.0000 / 0.9808 | 96 / 0.9904 |

The corresponding **training** objective losses (component / pizza / total) were: baseline 0.7443/0.7770/1.5213 initially and 0.3065/0.3710/0.6775 finally; box 0.7503/0.5941/1.3444 initially and 0.0243/0.0574/0.0816 finally; town 0.0271/0.1930/0.2201 initially and 0.0035/0.0183/0.0218 finally. Training macro-F1 (pizza / component) changed baseline 0.2667/0.1220→1.0000/0.9630, box 0.7389/0.1220→0/0, and town 0/0→1.0000/0.9818. The validation losses and F1 are in the table above.

The initial baseline/town points are independently initialized according to the controlled protocol; town’s initial F1 is zero and its validation F1 rises to 1.0/0.9808. Baseline rises from 0.2353/0.1249 to 1.0/0.9709. Both tasks therefore demonstrate image-to-ontology learning in these two family runs. Box’s initial pizza F1 is a broad-region initialization effect; by the final checkpoint both F1 values are zero despite a decreasing hinge objective. Box learning remains explicitly unresolved; no threshold or success claim is imposed. In the validation macro averages, `AnchovyTopping` is excluded from pizza F1 (zero positive support) and `NonVegetarianPizza` is excluded from component F1 (zero positive support); their known negatives/predictions remain reported. These single-seed results are not claims of family superiority.

At the first optimizer update, all expected hybrid branches and the shared classifier had nonzero gradients. L2 norms (classifier / component projection / DeepSets / global projection / fusion) were baseline 0.609/0.896/0.193/0.387/0.325, box 0.0547/2.636/0.615/0.864/1.035, and town 0.0388/0.0862/0.262/0.516/0.367. The FPN was frozen by configuration. The separate pizza-only gradient test verifies branch paths in global, compositional, and hybrid modes for all three families.
Run IDs, best checkpoint hashes, class-wise supports/exclusions, initial/final training metrics/losses, and first-step gradient norms are persisted in ignored local artifacts under `artifacts/phase3/remediation-learning-final/`; summarized reproducibly in the corresponding run manifests/checkpoints. Best checkpoint SHA-256 values: baseline `33a0b2735842e099397f6009290ac8038f283b2f125c28f44c8c0450f48ea6c0`, box `a523d4dbbd4f2b01b6e78a1b09816b20e19e49dfd5bbd802ead261bc9b022490`, town `28587a4c19f7f1b51e0783e851408e1f6ebbdcf9122f051469df4e895381b609`. No generated dataset/checkpoint/cache artifacts were committed.

### Additional directly related defect

During the first box learning attempt, the epoch-4 snapshot failed with `RuntimeError: shape '[4, 18, 8]' is invalid for input of size 16`. Geometry returned box-corner diagnostics shaped `[classes,dimensions]`, but `_reshape_classifier_output` treated all diagnostics as per-component points. It now preserves box and town class-corner parameters and only reshapes town activity tensors shaped `[N,C,...]`. Tests cover box parameters and town per-point diagnostics. The corrected box and town runs completed snapshot save/load without this error.

## Checkpoint compatibility, reproducibility, and tests

- Legacy checkpoints use a different objective-reduction and feature-extraction identity. They are not implicitly migrated or overwritten. The dashboard disables legacy resume; direct resume fails strict identity checks. Tests cover refusal.
- Existing mid-epoch resume parity still passes (resumed optimizer/model/visual state matches uninterrupted training). The service lifecycle test verifies stop, latest checkpoint, resume, optimizer continuation, and snapshots. The detached CLI worker test verifies graceful stop, checkpoint availability, compatible resume in a new process, and completed state. Incompatible and completed-run resumes are rejected.
- FPN/ROI/global path and all nine model-composition smoke tests use real prepared RGB images from `examples/pizza-48`; each performs optimizer update, checkpoint save/reload, and validation equality. All 9 pass.
- Full regression command: `.venv/bin/python -m pytest -q` — **238 passed, 11 existing Streamlit/Pandas deprecation warnings**. This includes all nine real-image smokes.
- Isolated nine-run command: `.venv/bin/python -m pytest -vv tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate` — **9 passed** across all `{baseline, box, town} × {global, compositional, hybrid}` cells.
- Additional checks: `.venv/bin/python -m compileall -q src tests`, `.venv/bin/python -m pip check`, and `git diff --check` all pass.
- Runtime: Python 3.12.0, PyTorch `2.5.1+cpu`, Torchvision `0.20.1+cpu`. The host has an NVIDIA GeForce RTX 2060 (6 GiB), but the installed PyTorch build has `torch.cuda.is_available() == False` and no CUDA runtime; no CUDA experiment was run. CPU-only smoke and exploratory runs were used.

## Files changed

- `src/town_model/training/image_objectives.py`, `tests/test_phase3_models.py`: corrected weighted reducer and scalar/gradient/mask/unit-weight/effective-batch tests.
- `src/town_model/experiments/image_config.py`, `src/town_model/training/image_trainer.py`, `src/town_model/experiments/image_runs.py`, `configs/phase3_image_*.json`: hinge defaults, objective identities, semantic run/checkpoint versioning, resolved class weights, and training/validation objective-loss reporting.
- `src/town_model/vision/features.py`, `src/town_model/experiments/feature_cache.py`, `tests/test_phase3_vision.py`, `tests/test_phase3_cache.py`: authoritative original-extent ROI and strict feature-cache contract version.
- `src/town_model/experiments/image_runs.py`, new `src/town_model/experiments/image_worker.py`, `src/town_model/dashboard/image_embeddings.py`: detached worker lifecycle and dashboard controls.
- `src/town_model/models/image_embeddings.py`: family-correct diagnostic reshaping; regression coverage in `tests/test_phase3_models.py`.
- `tests/test_phase3_config.py`, `tests/test_phase3_runs.py`, `tests/test_phase3_training.py`, `tests/test_dashboard_app.py`, `tests/test_phase3_architecture.py`: provenance, subprocess, resume/stop, gradient-path, and architecture-boundary checks.
- `configs/phase3_image_remediation_learning.json`: exact controlled learning config.
- `README.md`, `docs/PHASE_3_ARCHITECTURE.md`, and the implementation-report addendum: updated contracts and handoff links.

## Remaining limitations and recommendation

- No Pizzaïolo detector-pretraining checkpoint, predicted-ROI training, detector-matched component supervision, joint detector/ontology training, or Phase 4 work exists in this milestone. All reported image runs are pre-detector GT-ROI experiments.
- The controlled learning demonstration is one seed on reduced train/validation subsets and only two selected ontology classes. Full 2,880/960 training/validation coverage, multi-seed comparisons, the broader selected-family matrix, and explicit final test evaluation remain unperformed. The test partition remains sealed.
- Box hinge training did not demonstrate sustained learning in the small hybrid experiment; this is a scientific/optimization limitation for independent review, not concealed as an implementation success. No arbitrary F1 threshold is asserted.
- CUDA is unavailable in the installed runtime. A real headless Chrome page/control/preflight test was run; multi-user deployment was not.

**Readiness recommendation: READY FOR INDEPENDENT REVIEW** of the stated pre-detector scope. The confirmed F1–F4 contract defects and the directly related snapshot defect are corrected and regression-tested; F5 now has actual image-learning evidence for baseline and town, with box’s poor result and the broader scientific limitations explicitly unresolved. This report does not claim independent acceptance, Phase 3 acceptance, or permission to begin Phase 4.
