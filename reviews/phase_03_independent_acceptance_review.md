# Independent Phase 3 Acceptance Review

**Project:** *Representing Ontologies Using Town Embeddings*

**Review scope:** Phase 3 image-to-ontology embeddings, pre-detector GT-ROI milestone

## A. Verdict

**REQUEST CHANGES**

The principal image-to-point path is present and passes substantial CPU integration evidence. The weighted instance reducer, default objectives, letterboxed global ROI, nine family-by-composition smoke runs, and most checkpoint/snapshot behavior independently verify. I cannot accept the milestone as submitted because (1) two accepted positive-weighting modes implement a different scientific weighting formula than specified, (2) configuration preflight accepts invalid model settings that create failed partial runs, and (3) abrupt-worker recovery leaves stale worker status and a malformed run lock can permanently block resume. A residual snapshot diagnostic-shape defect and unresolved specification coverage are also recorded below.

This is **not** a rejection for the intentionally deferred detector-pretraining, predicted-ROI training, or joint detector/ontology stage. All inspected real-image results are pre-detector exploratory GT-ROI results.

## B. Reviewed revision

- **Implementation SHA reviewed:** `3c2af2fc154a1823247c8bb5daa58620e018ba1d`
- **Documentation-only review-branch HEAD at review start:** `53c334b0e69903802371cd7f89cb83367b0fe8af`
- Branch: `remediation/phase-3-pre-review-findings`; `git fetch --all --prune` completed before review and local HEAD matched the remote branch.
- `3c2af2f` is a descendant of original Phase 3 implementation `241cac3619022d9ae7707e1bc18667c872359527` and accepted Phase 2 baseline/review `fbe1063f2451270a89c9e0c110b988d6aafa4a27`. The implementation SHA is an ancestor of the documentation-only branch HEAD. No newer implementation commit was present after fetch; the later branch commit only contains the remediation report/addendum.
- Reviewed the complete Phase 3 diff from `fbe1063..3c2af2f` (44 files, 7,457 insertions/11 deletions) and the remediation diff from `241cac3..3c2af2f` (24 files, 1,268 insertions/37 deletions), including source, tests, configuration, architecture/scope reports, and artifact metadata. `main` was not used as a substitute.
- This report-only commit is separate from both the implementation SHA and the documentation-only branch HEAD stated above.

## C. Acceptance gate matrix

| Gate | Outcome | Evidence and justification |
|---|---|---|
| **3A — Dataset/configuration/architecture** | **FAIL** | The ignored canonical 4,800-image dataset validates independently (0 errors; 177 low-support warnings), preserves the 2,880/960/960 original splits, 66 OWL classes/60 trainable classes, and the pinned generator provenance. Dataset preparation and model/UI import boundaries are sound. However, invalid config values can pass `validate()` and fail only after `run_image_experiment()` has created a run directory; see defect D2. |
| **3B — Visual features and ROI contracts** | **PASS** | Pinned Torch 2.5.1/Torchvision 0.20.1 adapter, ImageNet MobileNetV3-Large stages, 256-channel FPN, ROIAlign, RGB normalization, original-pixel XYXY transforms, 260-D component features, independent `[B,256,7,7]` global features, empty ROI and explicit-provider support all passed source inspection/tests. The 400×200 letterbox ROI was independently sampled and compared below. The FPN is newly initialized and frozen in the frozen-feature runs; it is not detector-pretrained. |
| **3C — Point models and composition** | **PASS** | Component/global projections, bounded outputs, shared Phase 2 classifier API, global/compositional/hybrid composition, DeepSets masking/empty behavior, multiplicity/permutation checks, and pizza-only gradient paths are covered by tests. The same component point is classified and composed. A residual snapshot diagnostic-shape issue is recorded under D4; persisted header geometry and dashboard visualization remain correct. |
| **3D — Training and experiment comparisons** | **FAIL** | The full nine-cell CPU real-RGB smoke matrix passed, as did trainable-FPN, checkpoint reload, effective-batch parity, and frozen paired-initialization checks. But `inverse_sqrt` and `effective_number` weight construction does not implement the accepted scientific formulas; the corrected learning runs select `inverse_sqrt`, so their weighting identity and F5 interpretation are not protocol-faithful. No detector-pretrained or comparison-quality multi-seed result is claimed. |
| **3E — Persistence, lifecycle and dashboard** | **FAIL** | Atomic checkpoint/snapshot writes, compatible stop/resume, best-checkpoint re-evaluation, read-only snapshot inspection, and a real Chrome view of a saved box snapshot worked. Invalid-config preflight and abrupt-worker/lock recovery are incomplete (D2/D3); snapshot diagnostic arrays retain a dimension ambiguity (D4). The run manifest can be recovered to `interrupted`, but the detached worker status can continue to say `running`. |

## D. Remediation closure matrix

| Remediation item | Status | Independent result |
|---|---|---|
| **F1 — Weighted-loss normalization** | **CLOSED** | `_instance_terms()` implements the specified sum-of-known-weights denominator. Independent scalar, masking, unit-weight and heterogeneous accumulation checks match. The separate *weight-vector construction* defect D1 remains open; it is not the original denominator defect. |
| **F2 — Geometric objectives** | **CLOSED** | Effective defaults are baseline BCE-with-logits and box/town signed-distance hinge, margin 0.1. Zero margin is valid. Signed-margin BCE is explicit; objective settings affect run identity and legacy objective identities are rejected. |
| **F3 — Dashboard run lifecycle** | **OPEN** | Normal detached launch/status/stop/resume paths pass tests and browser inspection, but invalid configuration can create a failed partial run, and abrupt termination leaves stale worker status/lock recovery gaps (D2/D3). |
| **F4 — Global ROI coordinates** | **CLOSED** | The full original image extent is transformed through the same coordinate function as components. Feature sampling, non-square/padded behavior, identity versioning, and stale cache rejection were independently verified. |
| **F5 — Demonstrated learning** | **INCONCLUSIVE** | The saved best checkpoints and metrics reproduce. Baseline and town show both-task validation learning; box’s poor hard F1 has no confirmed math/evaluation defect. However, all three F5 runs declare a nonconformant `inverse_sqrt` weighting scheme (D1), limiting the scientific interpretation of those numbers. |
| **Additional class-diagnostic reshape defect** | **OPEN** | The original fatal reshape is fixed and box/town snapshots can be loaded and viewed. A related shape heuristic in snapshot serialization still misclassifies class-level town corners when batch size equals class count (D4). |

## E. Independent numerical verification

### E1. Weighted supervision

For the required reference case, with

```text
penalties = [2, 10, 4]
targets   = [1,  0, 0]
known     = [1,  1, 0]
weights   = [9,  1, 100]
```

independent computation and `_instance_terms()` both give

```text
loss = (9×2 + 1×10)/(9+1) = 2.8
∂loss/∂penalties = [0.9, 0.1, 0]
```

Changing the unknown target and its penalty to `1,000,000` leaves both loss and every gradient unchanged. Unit weights reproduce the ordinary known-label mean exactly. Because positives and negatives share the normalized denominator, changing a supplied positive weight changes the positive/negative contribution without the old count-denominator scale error. Pizza and component paths both call this reducer; components are averaged within each eligible image, then over eligible images.

**Effective-batch reference:** I built a four-image heterogeneous case with component counts `[2,0,3,1]`, mixed known masks and positive/negative labels, weights `[5,2,8]`, three eligible component images and three eligible pizza images, and microbatch sizes `[1,2,1]`. Full-batch signed-hinge objective was `0.21815277636051178`; accumulated objective was exactly the same (absolute difference `0.0`). Maximum component-score and pizza-score gradient differences were both `0.0`. This exercised `image_objective_terms()` plus `aggregate_effective_batch()`, not only a hand-written mean. A zero-component image remained ineligible for component loss while retaining its pizza loss.

**Separate weighting defect:** `image_trainer._positive_weights()` currently computes `sqrt(n_known_negative/n_known_positive)` for `inverse_sqrt`, clamps the result to at least 1, and does not apply the specified common normalization. For `effective_number` it uses a negative-effective-count/positive-effective-count ratio. The specification instead defines positive weights proportional to `1/sqrt(n⁺)` or `(1−β)/(1−βⁿ⁺)`, with a documented common normalization/cap; negatives stay at 1. Example from a deterministic 128-image train subset (seed 42): PeperonataTopping has 24 known positives/2,151 negatives and VegetableTopping has 1,404/771. The implementation returns weights `(9.4670, 1.0)`, ratio `9.4670`; the accepted inverse-sqrt positive-count ratio is `sqrt(1404/24)=7.6485` before the common scale. F5’s seed-2025 subset also resolves Anchovy component support `140/2,015` to weight `3.7937919`, while the configured formula is described as inverse square root of positive count. The actual vectors are persisted, but their policy name/formula is wrong. `inverse_frequency` is an additional mode not in the authoritative weighting table.

### E2. Geometric hinge and predictions

With signed margins `[0.02,0.05]`, targets `[1,0]`, and `m=0.1`, the implementation returns `[0.08,0.15]` and score gradients `[-1,+1]`. With `m=0`, zero scores produce zero penalties; the validation path preserves zero. Baseline zero logits produce BCE losses `[0.693147,0.693147]`.

Hard prediction uses `contains`, not a margin threshold. For a closed box `[-0.5,0.5]^2`, points `(0,0)`, `(0.5,0)`, `(0.5001,0)` have contains `[true,true,false]` and margins `[0.5,0,-0.0001000166]`. For a frame `[-0.8,0.8]^2` with a closed window `[-0.2,0.2]^2`, a body-boundary point is included, a window-boundary point is excluded, and the window center is excluded; their margins are `[0,-0,-0.2]`. An overlapping second house re-includes the center (`house_contains=[false,true]`, town membership true, town margin `0.4`). Thus exact boundary semantics and margin semantics are kept distinct as specified.

### E3. Letterboxed global ROI

The actual preprocessing record for a 400×200 image at 224 resolution is resized `(112,224)`, padding `(left=0, top=56)`, scale `(0.56,0.56)`. Transforming original `[0,0,400,200]` yields `[0,56,224,168]`. The extractor passed exactly that box to global ROIAlign and returned `[1,256,7,7]`. Its result exactly equals an independent direct ROIAlign call on `[0,56,224,168]`; versus the obsolete full-canvas ROI, max absolute feature difference was `21.48898` and mean absolute difference `4.63067`. The unpadded square case remains `[0,0,224,224]`. Cache/checkpoint identities use `original_extent_roi_letterbox_padding_v2`.

### E4. Gradient and physical-geometry invariants

All nine family×mode real-image smoke cells passed with optimizer updates, checkpoint save/reload and exact validation metric equality. Pizza-only gradient-path tests cover every family and mode. The paired frozen F5 checkpoints share identical initial visual hash `842a05c3…` and common neural hash `ebdb6263…`; classifier families have distinct heads. The independent trainable-FPN smoke updates FPN weights with nonzero gradients while the ImageNet MobileNet blocks remain frozen.

The F5 box checkpoint’s classifier geometry state hash changes from initialization to best/latest. Its observed classifier gradient L2 is `0.0547168` at step 1 and `0.0561011` at step 72. Best/latest physical widths remain approximately `1.820–1.934` and corners remain in `[-1,1]`; no physical collapse or invalid box is observed. The latest town geometry has frame minimum width `1.818655`, window minimum width `1.529280`, and minimum contained clearance `0.126437`; corners and containment are valid.

## F. Box learning investigation

**Determination:** the zero final box F1 is a plausible optimization/representation limitation in this exploratory setup, not a confirmed implementation or evaluation defect. It remains a limitation, not evidence that boxes win or fail generally.

The canonical-data run is hybrid `d=8`, 128 original-train / 48 original-validation images, selected IDs `2 AnchovyTopping` and `29 NonVegetarianPizza`, frozen ImageNet MobileNet plus a randomly initialized, frozen FPN. Validation support is 53 Anchovy component positives/743 negatives and 28 NonVegetarian pizza positives/20 negatives; the cross-level classes have no positives and are excluded from their macro scores while predictions remain reported. Checkpoint evaluation uses the correct IDs, canonical known masks, exact box membership for geometric predictions, and the stated zero-positive-support convention.

Independent `town experiment image evaluate ... --split validation` reloads reproduce:

| Family/checkpoint | Epoch / step | Pizza macro-F1 | Component GT-ROI macro-F1 | Primary monitor | Validation objective component / pizza / total |
|---|---:|---:|---:|---:|---:|
| Baseline best | 7 / 56 | 1.0000 | 0.9709 | 0.9854 | 0.32116 / 0.39452 / 0.71569 |
| Baseline latest | 12 / 96 | 1.0000 | 0.9709 | 0.9854 | 0.30743 / 0.37310 / 0.68053 |
| Box **best** | 1 / 8 | 0.7368 | 0.0000 | 0.3684 | 0.02852 / 0.16775 / 0.19627 |
| Box latest | 9 / 72 | 0.0000 | 0.0000 | 0.0000 | 0.02484 / 0.05726 / 0.08210 |
| Town best | 5 / 40 | 1.0000 | 0.9808 | 0.9904 | 0.00414 / 0.02541 / 0.02954 |
| Town latest | 12 / 96 | 1.0000 | 0.9808 | 0.9904 | 0.00363 / 0.01868 / 0.02231 |

Best checkpoint SHA-256 values independently match the handoff: baseline `33a0b273…`, box `a523d4db…`, town `28587a4c…`. The box best checkpoint retains pizza F1 `0.7368` but has component F1 zero; final/latest F1 is zero on both tasks. It is therefore incorrect to report only latest or to claim the best box checkpoint has zero F1 on every task.

For the latest box snapshot’s deterministic, capped validation reference panel, all 9 retained known-positive and 94 known-negative Anchovy component margins were outside (`contains=false`) and clustered near `−0.100409`. This is near the negative-class hinge clearance boundary `−m`, not near zero membership. Negative examples are then hinge-satisfied while positives incur loss; the point encoder moved the examples outside a still-valid broad box. The observed pattern is consistent with a learned “all outside” tradeoff under the selected weights and class/task reduction.

A minimal valid hinge example shows why lower continuous loss need not improve F1. With `m=0.1`, one positive at margin `+0.05` and nine negatives at `−0.05` have mean hinge loss `0.05` and F1 `1.0`. Moving the positive to `−0.05` and negatives to `−0.15` gives mean loss `(0.15+9×0)/10=0.015` while F1 falls to `0`. The loss and hard metric optimize different aggregate behavior.

The box path has nonzero hinge gradients, optimizer updates, a valid best checkpoint, reload parity, and physically valid geometry. No incorrect matching/evaluation or broken gradient path was found. The F5 learned scores are nevertheless **not protocol-faithful inverse-sqrt-weighted results** because of D1; no conclusion about relative classifier performance or final thesis claims is warranted.

## G. Reproducibility evidence and limits

### Dataset and sample identity

- Local ignored dataset: `artifacts/datasets/pizza-4800-phase3`
- Dataset ID: `pizza-898a12a57e50a7a0d690`; manifest ID `c9a92dce69a3a6e650bf92c45f2575400d7870f988608d9bce4795c1587e30f1`
- Dataset manifest SHA-256: `580070e2deb301e28757bf394ecbeb31adfeae5c3eb7b41807750be75644d24d`; class manifest SHA-256 `e803479520664a5de823848c41a29f45a70f6d2a755049a9be14b52312cddd54`
- Manifest provenance names generator `SysReIC/pizzaiolo@6c0bcb221a957c184012beaf103bebcde4b1f793`; the local source checkout is at that exact SHA and clean. The dataset validator independently passes with 0 errors/177 low-support warnings and reports 4,800 images, 300 per generator type, original split sizes 2,880/960/960, 66 ontology classes and 60 trainable classes.
- I validated the existing local 4,800-image artifact; I did **not** regenerate all 4,800 images. The generator revision and preparation revision are provenance claims in the manifest, not independently reproduced generation evidence.
- F5 train/validation subset IDs: `c19c4379f9071a069d4a9bcf4f5a2f26aeb23f78047175cce8ef9b9dc4bb7e7d` / `f36b1ad7304a4925400793c948d3d27c64e304cbf24586f041192ae4682ae80d`; sizes 128/48 from original train/validation partitions. No test model outcomes were evaluated or used. Dataset integrity validation reports test support only as a pre-training audit.

### Commands and outcomes

```text
git fetch --all --prune
git rev-parse HEAD 3c2af2fc154a1823247c8bb5daa58620e018ba1d 241cac3619022d9ae7707e1bc18667c872359527 fbe1063f2451270a89c9e0c110b988d6aafa4a27
git diff --stat fbe1063f2451270a89c9e0c110b988d6aafa4a27..3c2af2fc154a1823247c8bb5daa58620e018ba1d
git diff --stat 241cac3619022d9ae7707e1bc18667c872359527..3c2af2fc154a1823247c8bb5daa58620e018ba1d
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -vv tests/test_phase3_training.py::test_nine_real_image_training_smokes_update_save_reload_and_evaluate
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m pip check
.venv/bin/town validate artifacts/datasets/pizza-4800-phase3 --no-write-report --json
.venv/bin/town inspect artifacts/datasets/pizza-4800-phase3 --json
```

Results: **238 passed**, 11 existing Streamlit/Pandas `Pandas4Warning` deprecation warnings; isolated matrix **9 passed in 16.05s**; compilation passed; `pip check`: no broken requirements; dataset inspect `current_pass`. No skips/failures. The dataset validation command is read-only (`--no-write-report`).

For the actual F5 best/latest evaluations, I ran `town experiment image evaluate <run-dir> --split validation --checkpoint best` and `--checkpoint latest` for each of the three saved runs. All reloaded validation metrics above reproduce; no test split was selected.

Resume parity was independently tested on a real 4/2 image subset with a two-epoch warmup-cosine schedule: stop after the first optimizer step, resume, and compare with uninterrupted training. Visual/model tensors, optimizer, scheduler, CPU RNG, sampler order/cursor and final step all matched exactly (`4` steps).

### Runtime and browser

- Environment: `.venv`, Python 3.12.0, PyTorch `2.5.1+cpu`, Torchvision `0.20.1+cpu`, NumPy 2.5.4, Streamlit 1.66.1; `torch.cuda.is_available() == False`, `torch.version.cuda is None`. The host NVIDIA GPU was not treated as CUDA evidence; **CUDA execution is unverified**.
- Real headless browser inspection used Chrome `134.0.6998.88` against `town dashboard --headless --address 127.0.0.1 --port 8765` via Chrome DevTools Protocol. I navigated to Image Embeddings, loaded the canonical dataset and persisted box run, and verified the epoch-9 snapshot, source checkpoint hash, GT-ROI reference sample/image overlay, exact geometry and raw corners in the rendered browser DOM. Four images were present; no Streamlit exception occurred. A screenshot was kept only at `/tmp/phase3-review-dashboard-geometry.png` and is not committed.
- This independent browser session **did not** click launch/stop/resume. Those paths were covered by the full `AppTest` and detached-service tests; those tests are not represented as browser verification. Actual Chrome displayed the controls and a saved run. No multi-user deployment was tested.
- The dashboard’s read-only saved-run test with a `torch.load` failure sentinel passed in the full suite. The dashboard consumes checksummed JSON/NPZ snapshots, not optimizer checkpoints.

### Artifacts and source-lineage limitations

The three F5 checkpoints share the same visual/common initialization hashes and persist objective, class IDs, split IDs, RNG/sampler/optimizer/scheduler state. However, checkpoint environment metadata records Python/platform/Torch/Torchvision/NumPy and device, but not the exact Git commit/dirty status or a complete dependency lock. The repository has no lockfile, and `purpose="comparison"` does not enforce the specification’s three independent seeds per configuration. The submitted F5 runs are labeled exploratory and are not treated as comparison-quality science; a full multi-seed research comparison remains unverified.

## H. Defect register

### D1 — Accepted positive-weighting formulas are not implemented

- **Severity:** Major
- **Affected module:** `src/town_model/training/image_trainer.py::_positive_weights`; resolved objective metadata/checkpoint identity.
- **Minimal reproduction:** canonical seed-2025, 128-image train subset, class IDs `[2,29]`: component supports `AnchovyTopping=140 positive/2015 negative`, `NonVegetarianPizza=0/2155`; declared `inverse_sqrt` resolves to `[3.7937919,1.0]`. Code computes `sqrt(n−/n+)`, not normalized `1/sqrt(n+)`. The 24/1,404 positive-count example in E1 also yields a different weight ratio than the spec.
- **Expected:** implement and persist the specified inverse-positive-count/effective-number positive weights, common normalization and cap; known negatives exactly weight 1; or expose any alternative as a separately named, author-approved objective.
- **Actual:** inverse-negative/positive ratios, clamped to at least one; effective-number ratio includes negative counts; an extra `inverse_frequency` mode is accepted. Checkpoints identify this as `inverse_sqrt`.
- **Consequence:** F5’s weighted metrics remain reproducible as measurements of the actual stored vector but are misidentified as the accepted weighting policy. Weight changes relative positive/negative gradients and can alter the box learning result. Existing run identity also lacks a separate weighting-formula semantic version.
- **Recommended correction:** use the approved formulas and explicit normalization, add exact vector/identity tests, version the weighting semantics, then rerun/relabel the affected weighted experiments. Do not rewrite historical runs.

### D2 — Invalid image-model configuration can publish a failed partial run

- **Severity:** Major
- **Affected module:** `src/town_model/experiments/image_config.py` validation and `image_runs.py::run_image_experiment` preflight ordering.
- **Minimal reproduction A:** `ImageModelConfig(family="town", windows=0)` passes `config.validate()` (`windows < 0` is the rejection test), but `TownGeometry` rejects `windows=0`. `run_image_experiment()` creates the run directory/manifest, then raises `ValueError: dimensions, classes, houses, and windows must be positive`; the manifest is left `failed`.
- **Minimal reproduction B:** `VisionConfig(visual_policy="trainable_fpn_last_blocks", trainable_backbone_blocks=(3.5,))` passes configuration validation, then the backbone constructor rejects it after run creation with `ValueError: trainable MobileNet blocks must be feature indices in [0,16]`.
- **Expected:** every unsupported value must fail preflight before output/run creation, or the corresponding behavior must actually be implemented. The dashboard preflight and CLI must share this resolution.
- **Actual:** invalid inputs can leave `resolved-config.json`, manifest, and a `failed` scientific run. Both cases were reproduced under `/tmp`; no repository artifacts were created.
- **Consequence:** violates the invalid-input/no-partial-run contract and pollutes run discovery with runs that could never train.
- **Recommended correction:** strict type/range validation (`windows >= 1` for the current TownGeometry, integer backbone block indices) plus preflight construction/validation of model-dependent settings before publishing a run directory. Add no-side-effect regressions.

### D3 — Abrupt worker death leaves stale status; malformed writer lock is not recoverable

- **Severity:** Moderate
- **Affected module:** `src/town_model/experiments/image_worker.py`, `experiments/image_runs.py::inspect_run`/`RunWriterLock`, dashboard worker status display.
- **Minimal reproduction:** launch a short detached worker, wait for `latest.pt` and progress, then `SIGKILL` its PID. The worker status JSON remains `state="running", exit_code=null`; the launch reservation file remains. `inspect_run()` does eventually change the run manifest to `interrupted`, and a subsequent launch can reclaim a stale reservation, but the dashboard can show worker `running` beside run `interrupted` until another launch overwrites worker status. Separately, writing malformed/partial JSON to `.writer.lock` then calling `RunWriterLock.acquire()` twice fails both times with “another process owns this image run lock”; the lock is never recovered because malformed ownership is treated as alive.
- **Expected:** reconcile dead-worker status to an explicit interrupted/unknown-exit state; make stale/partial lock recovery safe or surface a documented operator recovery path; preserve resumable latest checkpoints.
- **Actual:** run manifest recovery exists, but worker-status and malformed-lock recovery are incomplete. A lock crash between `O_EXCL` creation and complete JSON publication can permanently prevent resume without manual deletion.
- **Consequence:** misleading operational state and a recoverable checkpoint may be blocked by stale local metadata.
- **Recommended correction:** PID-aware worker status refresh and an atomic/robust lock record with recovery semantics; test forced worker death and partial-lock scenarios.

### D4 — Snapshot serialization still confuses class geometry and per-sample diagnostics

- **Severity:** Moderate
- **Affected module:** `src/town_model/inspection/image_snapshots.py` diagnostic collection around the `value.shape[0] == len(batch.sample_ids)` heuristic.
- **Minimal reproduction:** save a town snapshot with batch size `B=4` and selected class count `C=4`. The header correctly stores `frame_low` as `[4,2,2]`. But the snapshot writes `component_geometry_frame_low` as `(8,2)` and `pizza_geometry_frame_low` as `(4,2,2)` indexed by the coincident batch dimension. Town point-wise `house_margin` retains its correct `(47,4,2)` point/class/house shape. A box header similarly preserves corners, and the existing dashboard renders from that header.
- **Expected:** class/house/coordinate corners remain explicitly class-scoped `[C,H,d]` (and windows `[C,H,W,d]`); only activity/margin tensors are reshaped by instance dimensions.
- **Actual:** optional NPZ diagnostic arrays are omitted or serialized under batch-shaped dimensions when `B` and `C` collide. Header geometry and visual slices remain correct, so this is bounded to persisted diagnostic arrays, not the hard classifier outputs.
- **Consequence:** saved snapshots can contain misleading/malformed duplicated diagnostics even though their primary geometry view works.
- **Recommended correction:** use typed diagnostic keys/contracts rather than infer semantic axes from matching leading dimensions; add the `B=C` snapshot round-trip regression.

### D5 — Relation-diagnostic capability remains unresolved

- **Severity:** Moderate requirement gap / scope ambiguity
- **Affected module:** model/evaluation/inspection/dashboard APIs.
- **Evidence:** no common ontology relation-diagnostic implementation was found under `src/town_model`; current dashboards visualize individual exact slices only. The authoritative specification §8.3 requires exact or explicitly sampled containment/intersection/disjointness evidence and counterexamples. Phase 2 scope called advanced relation diagnostics deferred; the Phase 3 scope is only a sequencing amendment and does not explicitly resolve this requirement.
- **Expected:** at minimum an explicitly labeled exact/sampled relation interface with method/scope and counterexamples, or a recorded approved amendment that carries it to a later milestone. This is not a demand for the unsettled full RQ2 metric or proof for towns.
- **Actual/consequence:** point prediction and individual slices are available, but class-region relation capability is absent; the implementation cannot yet support the specification’s structural diagnostics or thesis RQ2 evidence.
- **Recommended correction:** resolve the scope explicitly and implement the minimum required exact/sampled diagnostics before claiming full-spec readiness. This omission is not conflated with detector deferral.

## I. Phase 4 readiness

**Can Phase 4 add detector training and predicted-ROI production through these interfaces without rewriting point heads, composition, or classifiers? Yes, architecturally, with a visual-adapter integration prerequisite.** `ROIProvider` consumes original-image boxes and identities, and the point/model result contracts do not depend on GT category labels or detector outputs. A predicted provider can replace the GT provider without changing component/global projections, DeepSets/fusion, or the three point classifiers.

It is not plug-and-play detector integration: `SharedVisualFeatureExtractor.forward()` currently computes FPN maps internally and has no public “precomputed feature maps plus detector losses/predictions” handoff. Phase 4 must integrate detector loss/prediction and ROI sampling around one shared backbone computation (or add a narrow feature-map injection seam), preserve detached selected coordinates and original-pixel transforms, and add matching/end-to-end metrics. Those are visual/detector-adapter changes, not rewrites of point heads or classifiers. No detector training or predicted-ROI result was tested or is claimed here.

## J. Final recommendation

**Smallest coherent remediation before acceptance:**

1. Correct or explicitly rename/version the positive-weighting formulas; verify objective identity and rerun/relabel affected weighted learning evidence.
2. Reject all currently unsupported model/vision configuration values before any run directory, lock, cache, or worker side effects; add the two reproduced no-side-effect tests.
3. Make crash recovery reconcile worker state and safely recover partial writer locks.
4. Make snapshot diagnostic serialization axis-aware and round-trip test the `B=C` case.
5. Resolve the outstanding relation-diagnostic scope against specification §8.3; implement the minimum exact/sampled capability or record an approved later-phase deferral.

The reviewed **pre-detector Phase 3 implementation milestone is not accepted**, and Phase 4 is **not authorized by this review**. This does not invalidate the independently reproduced baseline/town learning demonstrations or imply any conclusion about final thesis claims.
