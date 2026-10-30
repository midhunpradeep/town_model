# Representing Ontologies Using Town Embeddings

The repository contains the accepted Phase 1 Pizzaïolo dataset-preparation package and a bounded Phase 2 synthetic geometry/classification subsystem. Phase 2 adds trainable neural, box, and town classifiers, reproducible synthetic runs, and a Geometry Laboratory in the existing Streamlit application. It does not implement an image detector or the image-to-town thesis pipeline.

## Install and run

Python 3.11+ and Git are required. For a CPU-only install, install the PyTorch CPU wheel first, then the project and tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu'
python -m pip install -e '.[test]'
```

The dependency range permits newer PyTorch versions; use the appropriate supported wheel for your platform if you do not use this CPU-only Linux example.

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

## Tests

```bash
pytest
```

## Scope

Phase 2 delivers synthetic neural, box, and town classifiers, signed-distance geometry training, reproducible fixed-point and trainable-point comparisons, checkpoint evaluation/resumption, and a saved-run Geometry Laboratory. See [Phase 2 scope](docs/PHASE_2_SCOPE.md) and the [Phase 2 implementation report](docs/PHASE_2_IMPLEMENTATION_REPORT.md). The image-to-town pipeline—including image embeddings and object detection—remains deferred; Phase 2 does not claim detector or image-embedding functionality.
