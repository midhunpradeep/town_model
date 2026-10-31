# Representing Ontologies Using Town Embeddings

The repository contains independently accepted Phase 1 (Pizzaïolo dataset preparation), Phase 2 (synthetic geometry/classification), and Phase 3 (**pre-detector, GT-ROI image-to-ontology**) implementation milestones. Phase 3 acceptance is not evidence for scientific superiority or final RQ1/RQ2 conclusions: detector pretraining, predicted-ROI learning, joint training, and final scientific comparisons remain outstanding. **No image detector has been implemented or trained in the accepted Phase 3 system.** Phase 4 is authorized but intentionally on hold pending the author's model-consolidation work.

**Authoritative historical acceptance:** [Definitive Phase 3 acceptance review](reviews/phase_03_definitive_acceptance_review.md) reviews implementation `a098a0a3c66a30200bd85906212cb07ef8bd19eb` and records **292 passed, 0 failed, 0 skipped** in its environment. The acceptance record is commit `2de9c5f29b436f727720e87132d0b4be25a40768`; the accepted implementation SHA remains distinct from this report-only review commit. See the [acceptance index](docs/ACCEPTANCE_INDEX.md) and [current project status](docs/CURRENT_PROJECT_STATUS.md).

## Current classifier-reconstruction boundary

This branch intentionally disconnects historical classifier execution. Synthetic/image classifier training, evaluation, resume, execution preflight, historical classifier inference, geometry interpretation/export, relation diagnostics, background workers, and classifier-dependent dashboard actions fail closed with an explicit unavailable-backend error before model/run side effects. There is no replacement backend. Dataset preparation/validation/inspection, the classifier-free image-to-point and vision/ROI infrastructure, and read-only historical metadata/snapshot loading remain available. **T2 is awaiting independent review; it is not accepted.** The Phase 1–3 reports below are preserved historical evidence and do not describe currently executable model commands on this branch. See [T2 disconnection report](docs/reconstruction/T2_DISCONNECTION_REPORT.md) once submitted.

## Install and run

Python 3.11–3.12 and Git are required. For a CPU-only install, install the PyTorch CPU wheel first, then the project and tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu'
python -m pip install -e '.[test]'
```

The image adapter is pinned to PyTorch 2.5.1 and Torchvision 0.20.1. Use matching wheels for your platform; CUDA execution has not yet been verified in this implementation.

Generation is explicit. The sample config requests a modest 48-image dataset (three images per generator type):

```bash
town generate --config dataset.toml --output artifacts/datasets/pizza-48
town validate artifacts/datasets/pizza-48
town inspect artifacts/datasets/pizza-48
town dashboard
```

The first generation downloads the pinned upstream source into the ignored `artifacts/sources/` cache. It is never patched. Set `PIZZAIOLO_SOURCE_DIR` to an already available checkout at the pinned commit to avoid cloning. Output paths must not already exist; datasets are published atomically and never replaced in place. To rerun, choose a new output path.

See [Dataset schema](docs/DATASET_SCHEMA.md), [provenance and licensing](docs/PROVENANCE.md), and the [Phase 1 implementation report](docs/PHASE_1_REPORT.md). A saved small real-generator demonstration is in [`examples/pizza-48`](examples/pizza-48).

## Phase 2 synthetic classifier experiments — historical milestone

Phase 2's classifiers, experiments, checkpoints, evaluation, resume, and Geometry Laboratory are retained as historical source/evidence, but their execution is intentionally unavailable on this reconstruction branch. `town experiment fixed|mapping`, `town evaluate`, `town resume`, and `town export-geometry` remain visible for compatibility and return an explicit unavailable-operation error without creating or changing artifacts. The Geometry Laboratory page likewise reports that classifier execution is disconnected. No classifier replacement is approved. See [Phase 2 scope](docs/PHASE_2_SCOPE.md) and the [historical implementation report](docs/PHASE_2_IMPLEMENTATION_REPORT.md); those reports describe the accepted historical milestone, not the current operational boundary.

## Phase 3 image-to-ontology embeddings — historical accepted milestone

The pre-detector GT-ROI image implementation remains preserved as historical evidence and classifier-independent vision/point infrastructure. Image classifier preflight, run, matrix, resume, checkpoint evaluation, background-worker launch/stop, snapshot generation, and geometry/relation interpretation are unavailable during reconstruction; commands remain visible and fail closed. Read-only image run metadata inspection and checksummed historical snapshot loading remain available. Dataset, RGB preprocessing, MobileNetV3-Large/FPN, ROI/ROIAlign, visual feature extraction/cache, `ImagePointModel`, and point composition remain independent infrastructure. See [Phase 3 scope](docs/PHASE_3_SCOPE.md), [architecture](docs/PHASE_3_ARCHITECTURE.md), the [historical implementation report](docs/PHASE_3_IMPLEMENTATION_REPORT.md), and the [definitive acceptance review](reviews/phase_03_definitive_acceptance_review.md). The accepted milestone contains no implemented or trained image detector. Detector pretraining, predicted-ROI learning, joint detector/ontology training, full scientific comparisons, and final held-out evaluation remain outstanding. Phase 4 remains on hold pending author-directed model-consolidation work.

## Tests

```bash
pytest
```

## Scope

Phase 1, Phase 2, and the bounded Phase 3 pre-detector GT-ROI implementation milestone are independently accepted; acceptance does not turn exploratory results into scientific conclusions. The accepted Phase 3 system does not provide canonical detector-pretrained training, predicted-ROI learning, joint detector/ontology training, or end-to-end detection. Phase 4 is authorized but on hold. See the [current project status](docs/CURRENT_PROJECT_STATUS.md), [acceptance index](docs/ACCEPTANCE_INDEX.md), and [Phase 3 scope](docs/PHASE_3_SCOPE.md) for evidence and remaining requirements.
