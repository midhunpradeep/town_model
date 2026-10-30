# Phase 3 implementation report — real-image ontology points

**Disposition: implementation work completed on this branch for review; Phase 3 is not independently accepted.** The experiments below use ImageNet initialization plus GT component boxes **before detector pretraining**. They are pre-detector exploratory results, not the specification's detector-pretrained stage 2, not end-to-end detection, and not Phase 4.

## Branch and scope

- Branch: `implementation/phase-3-image-embeddings`
- Required Phase 2 source/base: `fbe1063f2451270a89c9e0c110b988d6aafa4a27` (accepted Phase 2 review commit)
- Phase 3 scope/sequencing note: [`PHASE_3_SCOPE.md`](PHASE_3_SCOPE.md)
- Architecture and module contracts: [`PHASE_3_ARCHITECTURE.md`](PHASE_3_ARCHITECTURE.md)
- Phase 1/2 APIs, historical reports/reviews, and science were left in place. No changes were made to `main`; no detector or Phase 4 work is claimed.

## Implemented gates

- **3A:** framework-independent prepared-image records and class-support counts; strict deterministic subsets from original partitions; typed image/ROI/model contracts; validated config and run identity; architecture dependency tests.
- **3B:** pinned Torchvision MobileNetV3-Large/FPN adapter, RGB ImageNet preprocessing and invertible letterbox metadata, GT ROIAlign plus independent whole-image ROI, provider substitution contract, strict versioned frozen-feature cache.
- **3C:** component/global bounded point heads, empty-safe permutation-invariant DeepSets, global/compositional/hybrid pizza modes, named classifier results, unchanged Phase 2 baseline/box/town geometry, masked image objectives.
- **3D:** effective-batch image loss aggregation, positive-class weighting, frozen/trainable-FPN optimizer groups, metrics and early stopping, atomic latest/best checkpoints, RNG/sampler/optimizer resume, and checkpoint-only evaluation. Nine family×composition real-image smoke configurations run on the committed 48-image fixture.
- **3E:** local single-writer run lifecycle and CLI run/matrix/stop/resume/inspect/evaluate controls; bounded checksummed JSON/NPZ research snapshots; Image Embeddings dashboard for stored points, geometry, GT overlays, metrics, and selected pooled ROI features.

This is substantial implementation evidence, not gate acceptance. The full detector-training sequence and independent review remain outstanding.

## Prepared canonical dataset

Generated locally at the ignored path `artifacts/datasets/pizza-4800-phase3`; it is not committed.

- Dataset ID: `pizza-898a12a57e50a7a0d690`
- Manifest ID: `c9a92dce69a3a6e650bf92c45f2575400d7870f988608d9bce4795c1587e30f1`
- Dataset-manifest SHA-256: `580070e2deb301e28757bf394ecbeb31adfeae5c3eb7b41807750be75644d24d`
- Class-manifest SHA-256: `e803479520664a5de823848c41a29f45a70f6d2a755049a9be14b52312cddd54`
- 4,800 images, 300 per each of the 16 generator types; original train/validation/test partition sizes are 2,880/960/960.
- Current integrity validation: **PASS, 0 errors, 177 `low_class_support` warnings**. The warnings identify trainable classes with no known positive example at a particular supervision level; they are retained rather than relabeled or hidden.
- No test rows were used in these runs, early stopping, or snapshot selection. The 960-image test partition has not been evaluated for this report.

Reproduce the generation/validation explicitly with the commands in `README.md`. Opening the dashboard, loading a run, or importing the model does not generate data.

## Real-image experiments and observed outcomes

Every run below is labeled **pre-detector exploratory GT-ROI**. The generated dataset and checkpoints live under ignored `artifacts/phase3/`; run IDs and subset IDs are listed so artifacts can be audited locally.

### 1. Controlled small-subset component overfit / held-out check

Config: `configs/phase3_image_overfit.json`.

- Run: `image-35e76e6e9b5f2b1d35da`
- Train/validation: 16/32 images, selected only from original partitions; subset IDs `af74abdcf721568b2fc8fc03e5340f0205607eb2f6e203fbed811907adb318f6` / `1b6feace27b3b8ff5317ecb1a2ba30a80b87b8a7c45a121d248b34b72f2b37f7`.
- Selected ontology IDs: 0 `American` (pizza-supervised) and 2 `AnchovyTopping` (component-supervised). The component validation set contains 36 known positives and 1,006 known negatives; pizza validation contains 2 positives and 62 negatives.
- Baseline global pizza point, frozen visual features and strict cache identity, 20 epochs / 80 optimizer updates, inverse-frequency positive weighting.
- Component GT-ROI train macro-F1 rose from **0.131** at epoch 1 to **1.000** at epoch 20. Validation component macro-F1 rose from **0.129** to **0.958** (best monitor epoch 14; monitor **0.4789**). This is a successful limited component-level fit/held-out engineering demonstration on one selected class.
- **Important failure:** pizza macro-F1 remained **0.000** on both train and validation throughout. The combined primary score is therefore only 0.4789. This is not broad multi-class evidence and does not establish ontology learning quality.

### 2. 128/48 all-class exploratory run

Config: `configs/phase3_image_exploratory.json`.

- Run: `image-ca167876f4fe0f1a3a1c`
- Train/validation sizes 128/48; subset IDs `c19c4379f9071a069d4a9bcf4f5a2f26aeb23f78047175cce8ef9b9dc4bb7e7d` / `f36b1ad7304a4925400793c948d3d27c64e304cbf24586f041192ae4682ae80d`.
- All 60 canonical trainable class IDs, frozen ImageNet MobileNet/FPN visual path, cached fixed features, baseline hybrid pizza mode.
- Stopped after 9 epochs / 72 updates because the primary validation monitor did not improve for the configured 8 checks. Best checkpoint is at epoch 1 / optimizer step 8: pizza validation macro-F1 **0.1974**, component GT-ROI macro-F1 **0.0980**, primary **0.1477**. The final epoch metrics were **0.0775** / **0.1127**. Effective-batch training objective fell from a first-epoch mean of 1.452 to a last-epoch mean of 1.134, but this did not translate into a sustained macro-F1 improvement.
- Validation support was 328 known pizza positives / 2,552 negatives and 3,725 known component positives / 44,035 negatives. The best checkpoint was reloaded and evaluated through `town experiment image evaluate ... --split validation --checkpoint best`; it reproduced the epoch-1 metrics. No test metrics were consulted.

### 3. Canonical-dataset trainable-FPN demonstration

Config: `configs/phase3_image_trainable_fpn.json`.

- Run: `image-d0b2384a62a826afeba0`
- Train/validation sizes 32/32; train subset `f5e9e9f2d7b587a5c583ecb1ab5aac77e79316adf15ca2fcbec766c428cf7636`; validation subset is the same 32-image original-validation subset used above.
- Selected IDs 0/2 as above, baseline global pizza point, trainable FPN and frozen MobileNet blocks, no frozen-feature cache, 4 epochs / 32 optimizer steps.
- FPN gradients were active (step-1 L2 norm **4.155**); MobileNet backbone gradients remained frozen. The visual state changed from initial SHA-256 `fbb917c35548588895b119eeb8cecf4c4b9c8d2d9c69928d1f5b0a5a0ca88267` to final state SHA-256 `904c42042feba9eebee1d128bad6edb6176ebefa227e28212fa616f593f3d72e`.
- Component GT-ROI train macro-F1 rose **0.800 → 0.976**; validation rose **0.835 → 0.930**. Pizza macro-F1 remained **0.000**. Best primary monitor was **0.4648**. This demonstrates the trainable-FPN path updates and learns on a real canonical-dataset subset, but it is one seed, one selected component class, and a pre-detector configuration.

### 4. Nine family×composition smokes

The test suite runs each of `{baseline, box, town} × {global, compositional, hybrid}` on real RGB images from the committed `examples/pizza-48` fixture. Each performs optimizer updates, checkpoint save/reload, and validation metric equality checks. Paired fixed-feature initialization checks verify byte-identical visual/common neural states across classifier families while classifier heads remain independent. These are CPU engineering smokes, not 4,800-image comparative experiments.

## Verification

Environment: Python 3.12.0, PyTorch 2.5.1+cpu, Torchvision 0.20.1+cpu, NumPy 2.5.4; CUDA unavailable. ImageNet weights were available from the local Torch hub cache. `pip check` reported no broken requirements.

Final repository regression run for this implementation: **211 passed**, with seven existing Streamlit/Pandas deprecation warnings. This includes dataset/config, vision geometry/provider, model/objective, all nine smokes, mid-epoch resume parity, exact checkpoint reload evaluation, cache identity/corruption/stale-lock checks, snapshot non-perturbation, CLI evaluation/lifecycle, architecture imports, and Streamlit `AppTest` against both an empty run directory and a saved image run. The saved-run dashboard test replaces `torch.load` with a failure sentinel to prove it does not deserialize model/optimizer checkpoints. `python -m compileall -q src tests` also passed; `pip check` reported no broken requirements.

The cache-vs-live training parity test uses tight numerical tolerances rather than bitwise equality: per-image cached extraction and batched convolution can differ by small CPU floating-point rounding. Cache producer identity/checksums are exact. Snapshot capture itself was tested not to change the continued training/optimizer trajectory.

## Explicit gaps and decision

1. **No detector-pretraining checkpoint.** All visual encoders start from ImageNet MobileNet weights plus the documented FPN initialization. GT-ROI results are not the specification's detector-frozen embedding stage. No predicted-ROI training or joint detector/ontology fine-tuning exists.
2. **No full canonical matrix.** The all-class run used a 128/48 subset; the component overfit used 16/32; the trainable-FPN run used 32/32. The nine-family matrix has not been run on all 2,880/960 train/validation images, and multi-seed uncertainty is absent.
3. **Mixed learning results.** Limited component results fit/transfer well, but pizza macro-F1 is zero in both selected-class demonstrations. The all-class run's best validation score was modest and degraded under continued training. No broad research conclusion is warranted.
4. **Test partition sealed/unreported.** No test labels were used for model selection and no post-selection test report was run for this implementation report.
5. **Platform/UI gaps.** CUDA is unavailable; only headless Streamlit `AppTest` was exercised, not a real browser or multi-user deployment.
6. **Independent review is outstanding.** Do not merge to `main`, mark Phase 3 accepted, start Phase 4, or describe these experiments as detector-pretrained/canonical stage-2 science until the specified detector stages and independent acceptance criteria are completed.
