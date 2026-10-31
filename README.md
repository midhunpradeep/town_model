# Representing Ontologies Using Town Embeddings

The repository contains independently accepted Phase 1 (Pizzaïolo dataset preparation), Phase 2 (synthetic geometry/classification), and Phase 3 (**pre-detector, GT-ROI image-to-ontology**) implementation milestones. Phase 3 acceptance is not evidence for scientific superiority or final RQ1/RQ2 conclusions: detector pretraining, predicted-ROI learning, joint training, and final scientific comparisons remain outstanding. **No image detector has been implemented or trained in the accepted Phase 3 system.** Phase 4 is authorized but intentionally on hold pending the author's model-consolidation work.

**Authoritative acceptance:** [Definitive Phase 3 acceptance review](reviews/phase_03_definitive_acceptance_review.md) reviews implementation `a098a0a3c66a30200bd85906212cb07ef8bd19eb` and records **292 passed, 0 failed, 0 skipped** in its environment. The acceptance record is commit `2de9c5f29b436f727720e87132d0b4be25a40768`; the accepted implementation SHA remains distinct from this report-only review commit. See the [acceptance index](docs/ACCEPTANCE_INDEX.md) and [current project status](docs/CURRENT_PROJECT_STATUS.md).

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

## Phase 2 synthetic classifier experiments

Run all three classifiers on the fixed-point simple and nonconvex fixtures, then run the trainable point-mapping fixture:

```bash
town experiment fixed --output artifacts/phase2/runs --seed 42 --points 320 --steps 60
town experiment mapping --output artifacts/phase2/runs --seed 42 --points 320 --steps 60
```

Each command creates completed run groups under the output directory. Checkpoints can be evaluated, inspected, exported, and resumed without Streamlit:

```bash
town evaluate artifacts/phase2/runs/<group>/town --split validation
town inspect-run artifacts/phase2/runs/<group>/town
town export-geometry artifacts/phase2/runs/<group>/town --output town-geometry.json
town resume artifacts/phase2/runs/<group>/town --additional-steps 10
```

PyTorch checkpoints use Python pickle deserialization. Load only run artifacts from trusted sources; checkpoint loading is not safe for arbitrary untrusted files.

Launch the existing dashboard with `town dashboard`. The **Geometry Laboratory** reads saved groups from `artifacts/phase2/runs` and does not retrain when widgets change. It displays each independently trained model in its own coordinate frame. For `d>2`, region views are exact 2D slices and point markers are explicitly identified as projections.

Read [Phase 2 scope](docs/PHASE_2_SCOPE.md) for the sequencing amendment and explicit deferrals, and [Phase 2 implementation report](docs/PHASE_2_IMPLEMENTATION_REPORT.md) for formulas, test results, sample outcomes, and limitations.

## Phase 3 image-to-ontology embeddings (independently accepted; pre-detector GT-ROI only)

Create the canonical dataset explicitly, then run a real-image exploratory config:

```bash
town generate --config dataset.toml --output artifacts/datasets/pizza-4800-phase3 \
  --total-instances 4800 --generation-seed 42 --split-seed 42 \
  --train-size 0.6 --valid-size 0.2 --test-size 0.2
town validate artifacts/datasets/pizza-4800-phase3
town experiment image run --config configs/phase3_image_exploratory.json --output artifacts/phase3/runs
```

The prepared dataset and run/checkpoint/cache artifacts are local, ignored data. Small smoke and controlled-overfit examples are in `configs/phase3_image_smoke.json` and `configs/phase3_image_overfit.json`. Run `town experiment image matrix --config <comparison-config.json> --output artifacts/phase3/runs` for a paired family × composition matrix; the config must set `purpose` to `comparison`, retain best-validation checkpoints, and have known-positive train support for each selected in-scope class. For local run lifecycle operations, use `town experiment image stop <run>`, `town experiment image resume <run>`, and `town experiment image inspect <run>`. All of these image runs are explicitly labelled **pre-detector exploratory GT-ROI work**.

Evaluate a saved checkpoint explicitly after training; a final test-partition evaluation is a separate post-selection command, never part of training/early stopping:

```bash
town experiment image evaluate artifacts/phase3/runs/<run-id> --split validation --checkpoint best
# Only after model selection: town experiment image evaluate artifacts/phase3/runs/<run-id> --split test --checkpoint best
```

The **Image Embeddings** dashboard includes validated config drafting/preflight, detached local run launch, status/progress/log inspection, graceful stop, and compatible resume. Saved snapshot inspection remains read-only and does not deserialize checkpoint optimizer state; exact geometry slices use raw saved latent coordinates and optional PCA is display-only. See [Phase 3 scope](docs/PHASE_3_SCOPE.md), [architecture](docs/PHASE_3_ARCHITECTURE.md), [implementation report](docs/PHASE_3_IMPLEMENTATION_REPORT.md), and the [definitive acceptance review](reviews/phase_03_definitive_acceptance_review.md). The accepted milestone uses ImageNet-initialized visual features and ground-truth ROIs before detector pretraining; **it contains no implemented or trained image detector**. Detector pretraining, predicted-ROI learning, joint detector/ontology training, full scientific comparisons, and final held-out evaluation remain outstanding. Phase 4 was authorized by the independent review but is intentionally on hold pending author-directed model-consolidation work. CUDA execution was not verified in the acceptance environment.

## Tests

```bash
pytest
```

## Scope

Phase 1, Phase 2, and the bounded Phase 3 pre-detector GT-ROI implementation milestone are independently accepted; acceptance does not turn exploratory results into scientific conclusions. The accepted Phase 3 system does not provide canonical detector-pretrained training, predicted-ROI learning, joint detector/ontology training, or end-to-end detection. Phase 4 is authorized but on hold. See the [current project status](docs/CURRENT_PROJECT_STATUS.md), [acceptance index](docs/ACCEPTANCE_INDEX.md), and [Phase 3 scope](docs/PHASE_3_SCOPE.md) for evidence and remaining requirements.
