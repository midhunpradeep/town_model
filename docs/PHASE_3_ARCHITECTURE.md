# Phase 3 architecture — image-to-ontology embeddings

**Status: implementation in progress; not independently accepted.** All image experiments in this implementation use ground-truth (GT) component boxes and must be described as **pre-detector exploratory work**. The module boundary is designed to accept predicted ROIs later; no detector or detector-pretrained checkpoint exists in this milestone.

## Dependency boundaries

```text
Phase 1 dataset schema / validator
              │
              ├── dataset.image_data (framework-independent manifests, support, subsets)
              │
              └── contracts.ImageBatch / ImageGeometry / ROI identities
                           │
              ┌────────────┴────────────┐
              │                         │
       vision adapter              models.image_embeddings
       Torchvision/FPN/ROIAlign    point projections/composition
              │                         │
              └────────────┬────────────┘
                           │
                 training.image_trainer
                 image_objectives
                           │
        experiments.image_config / image_runs / feature_cache
                           │
               inspection.image_snapshots
                    (JSON + NPZ)
                           │
          dashboard.Image Embeddings / CLI inspection
```

- `dataset.image_data` validates and views prepared records without importing PyTorch or Torchvision. It creates deterministic subsets *inside* the original train and validation partitions; it never repartitions data. The test partition has no training, early-stopping, or snapshot path.
- `contracts` is the typed handoff for image geometry, ragged GT boxes/component IDs, target-known masks, provider results, pooled features, and named model outputs. Detector category IDs are retained as provenance, not fed to the point model.
- `vision` owns RGB loading/preprocessing, original↔letterboxed pixel-edge box transforms, a replaceable `ROIProvider`, FPN feature extraction, ROIAlign, and whole-image feature extraction. It has no ontology model, objective, experiment, or dashboard imports.
- `models` owns image-to-point projections, DeepSets composition, hybrid fusion, and the existing neural/box/town point classifiers. It receives visual features; it does not import the image adapter or experiment runner.
- `training` owns masked objectives, accumulation, optimizer state, evaluation, deterministic sampler state, and atomic checkpoints. It does not own run directories or dashboard behavior; lifecycle/snapshot operations are injected by the experiment service.
- `experiments` resolves immutable run identity, validates support, owns the local writer lock and run lifecycle, and optionally produces/reuses frozen visual feature caches.
- `inspection` persists and reads bounded numerical traces. The dashboard reads those JSON/NPZ traces plus the independently validated dataset image; it does not load model checkpoints/optimizer state or launch training.

Import-cycle and dependency tests enforce the new graph. Existing Phase 1 and Phase 2 modules and workflows remain available.

## Data and model contracts

### Data identity and selection

The canonical prepared dataset remains the source of class IDs, class names, IRIs, pizza/component supervision scopes, target vectors, and known masks. Image records retain sample ID, original split, generator type for audit, image path/dimensions, source labels, corrected ontology targets, GT component occurrence IDs/categories/boxes, and provenance. The generator/source labels and detector categories do not become model inputs.

`deterministic_subset` uses a seeded SHA-256 ordering and round-robin selection across generator types. Its manifest records the full selected sample IDs, original split, seed/policy, class support, dataset/class-manifest digests, and a subset ID. Resolved run identity includes ordered train/validation IDs and IDs, the data/class manifest hashes, the selected trainable ontology IDs, the GT ROI source, and scientific/training configuration. Visualization preferences, diagnostic sampling settings, and the cache directory do not alter training identity.

### Vision adapter

The validated adapter is pinned to **Torch 2.5.1 / Torchvision 0.20.1** and Torchvision MobileNetV3-Large ImageNet weights. It selects stages 3/6/12/16 (24/40/112/960 channels at strides 4/8/16/32), with a 256-channel FPN and pooled extra level. Inputs are RGB, aspect-preserving letterboxed to the configured square resolution, ImageNet-normalized, and accompanied by exact scale/padding metadata. Original-image xyxy pixel-edge boxes are transformed before ROIAlign.

Component GT boxes and a separate whole-image box use 7×7 multi-scale ROIAlign over FPN maps 0–3. The global feature exists for empty-component images too. Each component feature is a 256-value pooled ROI mean plus four normalized original-box coordinates, yielding 260 values. ROI identities are occurrence IDs; category IDs/confidence never enter projections. `GroundTruthROIProvider` is the Phase 3 provider. `ExplicitROIProvider`/provider contracts demonstrate substitution, but predicted-ROI training remains deferred.

### Image-to-point model

- Component points: `260 → 256 → LayerNorm/ReLU → 128/ReLU → dropout → d → tanh`.
- Global pizza point: spatial `256×7×7 → 1×1 conv(32) → flatten(1568) → 256/ReLU → d/tanh`.
- Component composition is sum-based DeepSets (`phi` per occurrence, masked sum, `rho`), preserving permutation invariance and multiplicity sensitivity; the empty set maps through `rho(0)`.
- Pizza modes are global, compositional, and hybrid. Hybrid fusion combines global/composed points through a bounded learned projection. The run configuration declares which final pizza point receives pizza supervision; no auxiliary intermediate pizza losses are inferred.
- One independently seeded classifier instance is attached per run: the Phase 2 neural linear head (logit threshold 0), closed box geometry, or town union-of-frames-minus-windows. Exact Phase 2 geometry APIs/closed-boundary behavior are reused.

Common neural projection/composition module states and frozen visual initial states are hash-checked across paired classifiers. Classifier-specific heads remain independently initialized. A replaced/additional ROI provider need not change any point or classifier mathematics.

### Objectives, aggregation, and metrics

Pizza and component GT-ROI losses use their own canonical target/known-mask vectors. Unknown labels are never negative labels. Baseline outputs use unreduced BCE-with-logits; box/town outputs use the configured signed-margin BCE or hinge objective. Optional positive weighting supports none, inverse-square-root, inverse-frequency, and effective-number ratios; only known positive entries are reweighted (with a configured cap), known negatives remain weight 1, and the per-instance mean is normalized by the count of known labels. Empty component collections remain valid; images with no known labels for a task are ineligible for that task.

For gradient accumulation, per-image component and pizza terms are retained across microbatches, each task is reduced over eligible images in the *full effective batch*, then configured task coefficients are applied. This avoids averaging separately normalized microbatch means. If both enabled tasks are undefined, training aborts rather than performing a zero-loss step.

Metrics include known-positive/known-negative/unknown counts and per-class known-mask F1, macro exclusions/reasons, and micro/macro F1. The primary monitor is fixed to `0.5 × pizza macro-F1 + 0.5 × component GT-ROI macro-F1`. A comparison run must have known-positive training coverage for all in-scope selected classes at each task and validation positives for both tasks; undefined monitor is an error. Smoke runs may use reduced subsets and may report an undefined monitor. Training metrics are retained for diagnosis but do not select checkpoints.

## Configuration, cache, runs, and checkpoints

`ImageExperimentConfig` validates original split names, class IDs, pinned vision/ROI contracts, model geometry feasibility, resource limits, objectives, and diagnostic preferences before output creation. The JSON constructor is shared by the CLI and service. Family×composition matrices require `purpose="comparison"`, which in turn requires best-validation retention and known-positive training coverage for every selected in-scope class. `num_workers` is currently restricted to 0 and latest checkpoints to every optimizer step for faithful deterministic resume. CPU smoke has an explicit gate; `cuda`/`auto` are supported in code but CUDA was not tested here.

Optional frozen-feature caches are versioned and immutable. Producer identity includes dataset and manifest IDs/digests, exact selected train/validation sample IDs and splits, ROI source, visual state hash, architecture/stages/FPN shape, preprocessing/ROI size, Torch/Torchvision versions, and producer device. Each sample feature asset is checksummed; stale producer identity, missing/corrupt arrays, or mismatched GT occurrence IDs fail closed. Caches contain visual component/global features and occurrence IDs only—no targets, test rows, model/optimizer state, or detector predictions. Trainable-FPN conditions cannot use a frozen cache.

Each run is preflighted before its run directory is created, receives a content-derived run ID, and is protected by a single local writer lock. The atomic JSON manifest tracks draft/running/completed/stopped/interrupted/failed. Graceful stop is polled at safe optimizer boundaries; resume requires a latest safe checkpoint and identical scientific/training identity. Checkpoints include visual/model/optimizer/scheduler states, sampler order/cursor, RNG states, resolved config/data identity, environment, best-monitor state, and parent checkpoint hashes on resume. Latest is written atomically every optimizer step; best-validation is separately retained when enabled. PyTorch checkpoint deserialization is for trusted local artifacts only.

`town experiment image` supports run, paired family×composition matrix, stop, resume, inspect, and saved-checkpoint evaluation. Test evaluation is an explicit post-training command and is not part of training, early stopping, checkpoint selection, or dashboard snapshot creation.

## Inspection and dashboard

At configured epoch intervals and at the final/early-stop epoch, deterministic reference samples are selected from existing train/validation subsets under sample/component caps. Inference snapshots persist their source checkpoint hash/step, dataset/subset/class identities, source splits, original image paths/dimensions, GT component IDs/categories/boxes, targets/masks, component/global/composed/final points, per-component/per-pizza exact membership and logits/margins, optional pooled ROI means/coordinates/global spatial features, and saved classifier geometry/diagnostics. JSON and numeric NPZ data are checksummed and atomically published; no optimizer state is included. Capture preserves Python/NumPy/Torch RNG and module modes so it does not change continued training.

The Image Embeddings page requires the matching currently validated dataset manifest before showing an image. It displays GT overlays, raw points/scores and masks, selected pooled visual features, exact raw-coordinate box/town slices or baseline logit boundaries, metrics, history, and run provenance. Optional PCA is computed for display only. Snapshot loading is covered by Streamlit `AppTest`; a real browser, concurrent multi-user service, and CUDA path are not independently verified.

## Phase 3 gate status (implementation, not acceptance)

| Gate | Current implementation | Remaining acceptance work |
|---|---|---|
| 3A data/config/boundaries | Canonical prepared data loader, target/mask contracts, deterministic original-split subsets, config validation, architecture tests | Independent gate review and any final review findings |
| 3B visual path | Pinned MobileNetV3-Large/FPN, letterbox/box transforms, GT/global ROIAlign, replaceable provider, frozen cache | CUDA and detector/predicted ROI integration are unverified/deferred |
| 3C point models | Component/global heads, DeepSets, global/compositional/hybrid, baseline/box/town, image objectives | No detector-pretrained scientific stage comparison |
| 3D training | Resumable real-image trainer; nine 48-image family×mode smoke tests; 4,800-image subset/overfit/FPN explorations | Full canonical training/matrices, multi-seed evidence, final held-out test evaluation, independent review |
| 3E service/dashboard | Atomic local run lifecycle, snapshots, CLI lifecycle/evaluation, read-only Image Embeddings page and `AppTest` | Browser/CUDA verification, operational concurrency review, independent acceptance |

Phase 3 remains **unaccepted and incomplete**. The specification's detector-pretraining, detector-frozen GT-ROI training, and joint predicted-ROI detector/ontology tuning are not conflated with this pre-detector milestone. No Phase 4 work is included.
