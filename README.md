# Representing Ontologies Using Town Embeddings

Phase 1 provides a standalone Pizzaïolo dataset-preparation package, a shared application layer, a CLI, and the first usable version of the permanent local dashboard. Dataset artifacts are framework-neutral; this phase does not implement a detector or any later model functionality.

## Install and run

Python 3.11+ and Git are required. Install the project and its tests with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

Generation is explicit. The sample config requests a modest 48-image dataset (three images per generator type):

```bash
town generate --config dataset.toml --output artifacts/datasets/pizza-48
town validate artifacts/datasets/pizza-48
town inspect artifacts/datasets/pizza-48
town dashboard
```

The first generation downloads the pinned upstream source into the ignored `artifacts/sources/` cache. It is never patched. Set `PIZZAIOLO_SOURCE_DIR` to an already available checkout at the pinned commit to avoid cloning. Output paths must not already exist; datasets are published atomically and never replaced in place. To rerun, choose a new output path.

See [Dataset schema](docs/DATASET_SCHEMA.md), [provenance and licensing](docs/PROVENANCE.md), and the [Phase 1 implementation report](docs/PHASE_1_REPORT.md). A saved small real-generator demonstration is in [`examples/pizza-48`](examples/pizza-48).

## Tests

```bash
pytest
```

## Scope

Only dataset generation, ontology-aware target preparation, validation/reporting, inspection, and dashboard exploration/generation are implemented. CNN detection, embeddings, geometry, losses, training, and experiment comparison are explicitly deferred to later authorized phases.
